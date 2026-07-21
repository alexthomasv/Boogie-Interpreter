use crate::opcodes::{CompiledProgram, Expr, Stmt};
use crate::vm::{Value, VM};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

#[derive(Debug, Clone)]
struct ArrayMeta {
    mem_map: String,
    base_ptr: String,
    offset_delta: i64,
    elem_size: i64,
    num_elements: i64,
}

#[derive(Debug, Clone)]
struct FieldMeta {
    mem_map: String,
    base_ptr: String,
    offset_delta: i64,
    size: usize,
    kind: String,
}

#[derive(Debug, Clone)]
struct AllocRequest {
    mem_map: String,
    base_ptr: String,
    offset_delta: i64,
    size: i64,
}

#[derive(Debug, Clone, Copy)]
struct ObjectSpan {
    min_offset: i64,
    max_end: i64,
}

/// Materialized input objects live in a negative address band, disjoint from
/// SMACK's positive `$CurrAddr` heap. Static globals also use negative
/// addresses, but start near zero and would need over a terabyte of storage to
/// reach this reserved band.
const INPUT_OBJECT_BASE: i64 = -(1i64 << 40) + 8;

fn strip_shadow(name: &str) -> &str {
    name.strip_suffix(".shadow").unwrap_or(name)
}

pub fn initialize_vm_from_inputs(
    program: &CompiledProgram,
    vm: &mut VM,
    native_meta: &Bound<'_, PyDict>,
    program_inputs: &Bound<'_, PyAny>,
    extra_data: Option<Vec<u8>>,
) -> PyResult<()> {
    init_memory_maps(program, vm);
    load_static_scalars(program, vm, native_meta)?;
    // Seed scalars baked into the package (the self-contained `.swcp` path).
    // Empty for the in-memory `lower()` path, which seeds via native_meta above.
    for &(vid, value) in &program.static_scalars {
        vm.set_scalar(vid, value, true);
    }

    if let Some(data) = extra_data.or_else(|| extract_extra_data(program_inputs).ok().flatten()) {
        vm.external_buffer = data;
    }

    let variables = input_variables(program_inputs)?;
    let arr_inputs = parse_array_meta(native_meta)?;
    let field_inputs = parse_field_meta(native_meta)?;
    let ptr_aliases = parse_string_map(native_meta, "ptr_aliases")?;
    validate_shadow_inputs(program, &variables, &arr_inputs, &field_inputs)?;

    load_scalar_inputs(program, vm, &variables)?;
    load_havoc_sequences(program, vm, &variables)?;

    let ptr_assignments = allocate_addresses(&variables, &arr_inputs, &field_inputs, &ptr_aliases)?;
    validate_no_aliasing(
        &variables,
        &arr_inputs,
        &field_inputs,
        &ptr_aliases,
        &ptr_assignments,
    )?;
    for (ptr, value) in &ptr_assignments {
        if let Some(vid) = program.lookup_var(ptr) {
            vm.set_scalar(vid, *value, true);
        }
    }
    concretize_memory(program, vm, &variables, &arr_inputs, &field_inputs)?;
    Ok(())
}

pub fn prepare_symbolic_state(
    py: Python<'_>,
    program: &CompiledProgram,
    native_meta: &Bound<'_, PyDict>,
    program_inputs: &Bound<'_, PyAny>,
    extra_data: Option<Vec<u8>>,
    havoc_bound: usize,
) -> PyResult<PyObject> {
    let started = Instant::now();
    let mut vm = VM::new_no_trace(program);
    vm.log_read = false;
    initialize_vm_from_inputs(program, &mut vm, native_meta, program_inputs, extra_data)?;

    let variables = input_variables(program_inputs)?;
    let arr_inputs = parse_array_meta(native_meta)?;
    let field_inputs = parse_field_meta(native_meta)?;

    let out = PyDict::new_bound(py);
    out.set_item("var_store", export_var_store(py, program, &vm)?)?;
    out.set_item("memory_maps", export_memory_maps(py, &vm)?)?;
    out.set_item("mem_map_info", export_mem_map_info(py, &vm)?)?;
    out.set_item("extra_data", vm.external_buffer.clone())?;

    let symbols = PyList::empty_bound(py);
    let bindings = PyDict::new_bound(py);
    build_input_symbols(
        py,
        program,
        &vm,
        &variables,
        &arr_inputs,
        &field_inputs,
        havoc_bound,
        &symbols,
        &bindings,
    )?;
    out.set_item("symbols", &symbols)?;
    out.set_item("bindings", &bindings)?;
    out.set_item("symbol_count", symbols.len())?;
    out.set_item("state_ms", started.elapsed().as_secs_f64() * 1000.0)?;
    Ok(out.into())
}

fn init_memory_maps(program: &CompiledProgram, vm: &mut VM) {
    for info in &program.mem_maps {
        vm.init_memory_map(
            info.var_id,
            info.name.clone(),
            info.index_bit_width,
            info.element_bit_width,
        );
    }
}

fn export_var_store<'py>(
    py: Python<'py>,
    program: &CompiledProgram,
    vm: &VM,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for (idx, value) in vm.vars.iter().enumerate() {
        if let Value::Scalar(v) = value {
            if *v != 0 {
                out.set_item(&program.var_names[idx], *v)?;
            }
        }
    }
    Ok(out)
}

fn export_memory_maps<'py>(py: Python<'py>, vm: &VM) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for map in &vm.memory_maps {
        if map.is_init_empty() {
            continue;
        }
        let contents = PyDict::new_bound(py);
        for (addr, value) in map.iter_init() {
            contents.set_item(addr, value)?;
        }
        out.set_item(map.name.as_str(), contents)?;
    }
    Ok(out)
}

fn export_mem_map_info<'py>(py: Python<'py>, vm: &VM) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty_bound(py);
    for map in &vm.memory_maps {
        out.append((
            map.name.as_str(),
            map.index_bit_width,
            map.element_bit_width,
        ))?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn build_input_symbols(
    py: Python<'_>,
    program: &CompiledProgram,
    vm: &VM,
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
    havoc_bound: usize,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let mut items = Vec::new();
    for (key, inp) in variables.iter() {
        items.push((key.extract::<String>()?, inp));
    }
    items.sort_by(|a, b| a.0.cmp(&b.0));
    let havoc_outputs = havoc_output_aliases(program);

    for (var_name, inp) in items {
        if var_name.ends_with(".shadow") {
            continue;
        }
        if let Some(value) = input_value(&inp)? {
            if program.lookup_var(&var_name).is_some() {
                add_scalar_symbol(py, symbols, bindings, &var_name, value)?;
                if let Some(root_name) = havoc_outputs.get(&var_name) {
                    if input_havoc_attr(&inp)?.is_none() {
                        add_havoc_value_symbols(
                            py,
                            symbols,
                            bindings,
                            &var_name,
                            root_name,
                            value,
                            havoc_bound,
                        )?;
                    }
                }
            }
        }
        if let Some(seq_obj) = input_havoc_attr(&inp)? {
            if program.lookup_var(&var_name).is_some() {
                add_havoc_symbols(py, symbols, bindings, &var_name, &seq_obj, havoc_bound)?;
            }
        }
        if let Some(buffers) = input_list_attr(&inp, "buffers")? {
            add_buffer_symbols(
                py, program, vm, arr_inputs, &var_name, &buffers, symbols, bindings,
            )?;
        }
        if let Some(fields) = input_list_attr(&inp, "struct")? {
            add_struct_symbols(
                py,
                program,
                vm,
                arr_inputs,
                field_inputs,
                &var_name,
                &fields,
                symbols,
                bindings,
            )?;
        }
    }

    for (idx, byte) in vm.external_buffer.iter().enumerate() {
        add_extra_symbol(py, symbols, bindings, idx, *byte)?;
    }
    Ok(())
}

fn next_symbol_name(symbols: &Bound<'_, PyList>) -> String {
    format!("s{}", symbols.len())
}

fn add_base_symbol<'py>(
    py: Python<'py>,
    symbols: &Bound<'py, PyList>,
    bindings: &Bound<'py, PyDict>,
    kind: &str,
    bits: u32,
    value: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let name = next_symbol_name(symbols);
    let d = PyDict::new_bound(py);
    d.set_item("name", name.as_str())?;
    d.set_item("kind", kind)?;
    d.set_item("bits", bits)?;
    d.set_item("value", value)?;
    symbols.append(&d)?;
    bindings.set_item(name, &d)?;
    Ok(d)
}

fn add_scalar_symbol(
    py: Python<'_>,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
    var_name: &str,
    value: i64,
) -> PyResult<()> {
    let d = add_base_symbol(py, symbols, bindings, "scalar", 64, value as u64)?;
    d.set_item("var", var_name)?;
    Ok(())
}

fn add_havoc_symbols(
    py: Python<'_>,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
    var_name: &str,
    seq_obj: &Bound<'_, PyList>,
    havoc_bound: usize,
) -> PyResult<()> {
    let count = havoc_bound.max(seq_obj.len());
    for idx in 0..count {
        let value = if idx < seq_obj.len() {
            extract_i64(&seq_obj.get_item(idx)?)?
        } else {
            0
        };
        let d = add_base_symbol(py, symbols, bindings, "havoc", 64, value as u64)?;
        d.set_item("havoc_var", var_name)?;
        d.set_item("havoc_index", idx)?;
    }
    Ok(())
}

fn add_havoc_value_symbols(
    py: Python<'_>,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
    input_var_name: &str,
    havoc_var_name: &str,
    first_value: i64,
    havoc_bound: usize,
) -> PyResult<()> {
    let count = havoc_bound.max(1);
    for idx in 0..count {
        let value = if idx == 0 { first_value } else { 0 };
        let d = add_base_symbol(py, symbols, bindings, "havoc", 64, value as u64)?;
        d.set_item("input_var", input_var_name)?;
        d.set_item("havoc_var", havoc_var_name)?;
        d.set_item("havoc_index", idx)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_buffer_symbols(
    py: Python<'_>,
    program: &CompiledProgram,
    vm: &VM,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    var_name: &str,
    buffers: &Bound<'_, PyList>,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let non_shadow: Vec<&ArrayMeta> = arr_inputs
        .get(var_name)
        .map(|items| {
            items
                .iter()
                .filter(|m| !m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    for (buffer_idx, (buf, meta)) in buffers.iter().zip(non_shadow.iter()).enumerate() {
        let size = dict_i64(&buf, "size")? as usize;
        let contents = dict_get(&buf, "contents")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("contents"))?;
        let data = bytes_from_contents(&contents, size)?;
        let base = offset_value(vm, program, &meta.base_ptr, meta.offset_delta)?;
        for byte_idx in 0..size {
            let value = data.get(byte_idx).copied().unwrap_or(0);
            let d = add_base_symbol(py, symbols, bindings, "buffer_byte", 8, value as u64)?;
            d.set_item("map", meta.mem_map.as_str())?;
            d.set_item("addr", base + byte_idx as i64)?;
            d.set_item("input_var", var_name)?;
            d.set_item("buffer_index", buffer_idx)?;
            d.set_item("byte_index", byte_idx)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_struct_symbols(
    py: Python<'_>,
    program: &CompiledProgram,
    vm: &VM,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
    var_name: &str,
    fields: &Bound<'_, PyList>,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let field_infos: Vec<&FieldMeta> = field_inputs
        .get(var_name)
        .map(|items| {
            items
                .iter()
                .filter(|m| !m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    let arr_non_shadow: Vec<&ArrayMeta> = arr_inputs
        .get(var_name)
        .map(|items| {
            items
                .iter()
                .filter(|m| !m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    validate_struct_fields(var_name, fields, &field_infos)?;

    let mut field_idx = 0usize;
    let mut buffer_idx = 0usize;
    for struct_idx in 0..fields.len() {
        let field = fields.get_item(struct_idx)?;
        if dict_get(&field, "value")?.is_some() {
            if let Some(meta) = field_infos.get(field_idx) {
                add_struct_scalar_symbols(
                    py, program, vm, &field, meta, var_name, struct_idx, symbols, bindings,
                )?;
            }
            field_idx += 1;
        } else if let Some(buf) = dict_get(&field, "buffer")? {
            if let Some(meta) = arr_non_shadow.get(buffer_idx) {
                add_struct_buffer_symbols(
                    py, program, vm, &buf, meta, var_name, struct_idx, buffer_idx, symbols,
                    bindings,
                )?;
            }
            field_idx += 1;
            buffer_idx += 1;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_struct_scalar_symbols(
    py: Python<'_>,
    program: &CompiledProgram,
    vm: &VM,
    field: &Bound<'_, PyAny>,
    meta: &FieldMeta,
    var_name: &str,
    field_index: usize,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let value =
        dict_get(field, "value")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("value"))?;
    let mut contents = bytes_from_contents(&value, meta.size)?;
    contents.reverse();
    let base = offset_value(vm, program, &meta.base_ptr, meta.offset_delta)?;
    for (byte_idx, value) in contents.iter().enumerate() {
        let d = add_base_symbol(
            py,
            symbols,
            bindings,
            "struct_scalar_byte",
            8,
            *value as u64,
        )?;
        d.set_item("map", meta.mem_map.as_str())?;
        d.set_item("addr", base + byte_idx as i64)?;
        d.set_item("input_var", var_name)?;
        d.set_item("field_index", field_index)?;
        d.set_item("byte_index", byte_idx)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_struct_buffer_symbols(
    py: Python<'_>,
    program: &CompiledProgram,
    vm: &VM,
    buf: &Bound<'_, PyAny>,
    meta: &ArrayMeta,
    var_name: &str,
    field_index: usize,
    buffer_index: usize,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let size = dict_i64(buf, "size")? as usize;
    let contents = dict_get(buf, "contents")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("contents"))?;
    let data = bytes_from_contents(&contents, size)?;
    let base = offset_value(vm, program, &meta.base_ptr, meta.offset_delta)?;
    for byte_idx in 0..size {
        let value = data.get(byte_idx).copied().unwrap_or(0);
        let d = add_base_symbol(py, symbols, bindings, "struct_buffer_byte", 8, value as u64)?;
        d.set_item("map", meta.mem_map.as_str())?;
        d.set_item("addr", base + byte_idx as i64)?;
        d.set_item("input_var", var_name)?;
        d.set_item("field_index", field_index)?;
        d.set_item("buffer_index", buffer_index)?;
        d.set_item("byte_index", byte_idx)?;
    }
    Ok(())
}

fn add_extra_symbol(
    py: Python<'_>,
    symbols: &Bound<'_, PyList>,
    bindings: &Bound<'_, PyDict>,
    extra_index: usize,
    value: u8,
) -> PyResult<()> {
    let d = add_base_symbol(py, symbols, bindings, "extra_byte", 8, value as u64)?;
    d.set_item("extra_index", extra_index)?;
    Ok(())
}

fn load_static_scalars(
    program: &CompiledProgram,
    vm: &mut VM,
    native_meta: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let static_scalars = native_meta
        .get_item("static_scalars")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("static_scalars"))?;
    let scalars: &Bound<'_, PyDict> = static_scalars.downcast()?;
    for (key, val) in scalars.iter() {
        let name: String = key.extract()?;
        let value = extract_i64(&val)?;
        if let Some(vid) = program.lookup_var(&name) {
            vm.set_scalar(vid, value, true);
        }
    }
    Ok(())
}

fn input_variables<'py>(program_inputs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(program_inputs
        .getattr("variables")?
        .downcast_into::<PyDict>()?)
}

fn extract_extra_data(program_inputs: &Bound<'_, PyAny>) -> PyResult<Option<Vec<u8>>> {
    let extra = program_inputs.getattr("extra_data")?;
    if extra.is_none() {
        return Ok(None);
    }
    extra.extract::<Vec<u8>>().map(Some)
}

fn validate_shadow_inputs(
    program: &CompiledProgram,
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
) -> PyResult<()> {
    for (key, inp) in variables.iter() {
        let name: String = key.extract()?;
        if name.ends_with(".shadow") {
            continue;
        }

        let shadow_name = format!("{name}.shadow");
        let has_scalar = input_value(&inp)?.is_some();
        let has_havoc = input_havoc_attr(&inp)?.is_some();
        let buffers = input_list_attr(&inp, "buffers")?;
        let fields = input_list_attr(&inp, "struct")?;
        let scalar_shadow = program.lookup_var(&shadow_name).is_some();
        let shadow_array_count = shadow_array_count(arr_inputs, &shadow_name);
        let shadow_field_count = shadow_field_count(field_inputs, &shadow_name);

        let needs_scalar = scalar_shadow && has_scalar;
        let needs_havoc = scalar_shadow && has_havoc;
        let needs_buffers = buffers.is_some() && shadow_array_count > 0;
        let needs_struct = fields.is_some() && (shadow_array_count > 0 || shadow_field_count > 0);
        if !(needs_scalar || needs_havoc || needs_buffers || needs_struct) {
            continue;
        }

        let shadow = require_shadow_input(variables, &name, &shadow_name)?;
        if needs_scalar && input_value(&shadow)?.is_none() {
            return Err(missing_shadow_payload(&name, &shadow_name, "value"));
        }
        if needs_havoc && input_havoc_attr(&shadow)?.is_none() {
            return Err(missing_shadow_payload(&name, &shadow_name, "havoc_seq"));
        }
        if needs_buffers {
            let shadow_buffers = require_input_list_attr(&shadow, &name, &shadow_name, "buffers")?;
            validate_payload_count(
                &name,
                &shadow_name,
                "buffers",
                shadow_buffers.len(),
                shadow_array_count,
            )?;
        }
        if needs_struct {
            let shadow_struct = require_input_list_attr(&shadow, &name, &shadow_name, "struct")?;
            let base_struct = fields.as_ref().expect("needs_struct requires fields");
            validate_payload_count(
                &name,
                &shadow_name,
                "struct fields",
                shadow_struct.len(),
                base_struct.len(),
            )?;
        }
    }
    Ok(())
}

fn require_shadow_input<'py>(
    variables: &Bound<'py, PyDict>,
    base_name: &str,
    shadow_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    variables.get_item(shadow_name)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "input {base_name:?} requires explicit shadow input {shadow_name:?}"
        ))
    })
}

fn require_input_list_attr<'py>(
    inp: &Bound<'py, PyAny>,
    base_name: &str,
    shadow_name: &str,
    attr: &str,
) -> PyResult<Bound<'py, PyList>> {
    input_list_attr(inp, attr)?.ok_or_else(|| missing_shadow_payload(base_name, shadow_name, attr))
}

fn missing_shadow_payload(base_name: &str, shadow_name: &str, attr: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "input {base_name:?} requires {attr} on explicit shadow input {shadow_name:?}"
    ))
}

fn validate_payload_count(
    base_name: &str,
    shadow_name: &str,
    payload: &str,
    actual: usize,
    expected: usize,
) -> PyResult<()> {
    if actual == expected {
        return Ok(());
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "input {base_name:?} shadow {payload} count mismatch for {shadow_name:?}: expected {expected}, got {actual}"
    )))
}

fn shadow_array_count(arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>, name: &str) -> usize {
    arr_inputs
        .get(name)
        .map(|items| {
            items
                .iter()
                .filter(|meta| meta.mem_map.contains(".shadow"))
                .count()
        })
        .unwrap_or(0)
}

fn shadow_field_count(field_inputs: &FxHashMap<String, Vec<FieldMeta>>, name: &str) -> usize {
    field_inputs
        .get(name)
        .map(|items| {
            items
                .iter()
                .filter(|meta| meta.mem_map.contains(".shadow"))
                .count()
        })
        .unwrap_or(0)
}

fn load_scalar_inputs(
    program: &CompiledProgram,
    vm: &mut VM,
    variables: &Bound<'_, PyDict>,
) -> PyResult<()> {
    for (key, inp) in variables.iter() {
        let name: String = key.extract()?;
        if let Some(value) = input_value(&inp)? {
            if let Some(vid) = program.lookup_var(&name) {
                vm.set_scalar(vid, value, true);
            }
        }
    }
    Ok(())
}

fn load_havoc_sequences(
    program: &CompiledProgram,
    vm: &mut VM,
    variables: &Bound<'_, PyDict>,
) -> PyResult<()> {
    // Buffer/struct-only inputs cannot seed scalar havoc schedules. Avoid the
    // whole-program assignment-alias analysis in that common case: on a large
    // natively inlined program the temporary alias relation can otherwise be
    // much larger than the concrete state it is trying to initialize.
    let mut has_scalar_or_havoc_input = false;
    for (_, inp) in variables.iter() {
        if input_havoc_attr(&inp)?.is_some() || input_value(&inp)?.is_some() {
            has_scalar_or_havoc_input = true;
            break;
        }
    }
    if !has_scalar_or_havoc_input {
        return Ok(());
    }

    let havoc_outputs = havoc_output_aliases(program);
    for (key, inp) in variables.iter() {
        let name: String = key.extract()?;
        if let Some(vid) = program.lookup_var(&name) {
            let root_vid = havoc_outputs
                .get(&name)
                .and_then(|root| program.lookup_var(root))
                .unwrap_or(vid);
            let Some(seq_list) = input_havoc_attr(&inp)? else {
                if havoc_outputs.contains_key(&name) {
                    if let Some(value) = input_value(&inp)? {
                        vm.set_havoc_sequence(root_vid, vec![value]);
                    }
                }
                continue;
            };
            let mut seq = Vec::with_capacity(seq_list.len());
            for item in seq_list.iter() {
                seq.push(extract_i64(&item)?);
            }
            vm.set_havoc_sequence(root_vid, seq);
        }
    }
    Ok(())
}

fn havoc_output_aliases(program: &CompiledProgram) -> BTreeMap<String, String> {
    let mut var_to_root = BTreeMap::new();
    for block in &program.blocks {
        for stmt in &block.body {
            match stmt {
                Stmt::CallNondet { assignments } | Stmt::Havoc { vars: assignments } => {
                    for var_id in assignments {
                        var_to_root.insert(*var_id, *var_id);
                    }
                }
                Stmt::Assign1 { lhs, rhs } => {
                    if let Expr::Var(src) = rhs {
                        if let Some(root) = var_to_root.get(src).copied() {
                            var_to_root.insert(*lhs, root);
                        }
                    }
                }
                Stmt::AssignN { lhs, rhs } => {
                    for (lhs_id, rhs_expr) in lhs.iter().zip(rhs.iter()) {
                        if let Expr::Var(src) = rhs_expr {
                            if let Some(root) = var_to_root.get(src).copied() {
                                var_to_root.insert(*lhs_id, root);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = BTreeMap::new();
    for (var_id, root_id) in var_to_root {
        if let (Some(name), Some(root)) = (
            program.var_names.get(var_id as usize),
            program.var_names.get(root_id as usize),
        ) {
            out.insert(name.to_string(), root.to_string());
        }
    }
    out
}

fn parse_array_meta(
    native_meta: &Bound<'_, PyDict>,
) -> PyResult<FxHashMap<String, Vec<ArrayMeta>>> {
    let mut out = FxHashMap::default();
    let obj = native_meta
        .get_item("arr_inputs")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("arr_inputs"))?;
    let dict: &Bound<'_, PyDict> = obj.downcast()?;
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        let items: &Bound<'_, PyList> = value.downcast()?;
        let mut metas = Vec::with_capacity(items.len());
        for item in items.iter() {
            let d: &Bound<'_, PyDict> = item.downcast()?;
            let elem_size = required_i64(d, "elem_size")?;
            let num_elements = required_i64(d, "num_elements")?;
            if elem_size <= 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "array metadata elem_size must be positive, got {elem_size}"
                )));
            }
            if num_elements < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "array metadata num_elements must be non-negative, got {num_elements}"
                )));
            }
            metas.push(ArrayMeta {
                mem_map: required_string(d, "mem_map")?,
                base_ptr: required_string(d, "base_ptr")?,
                offset_delta: required_i64(d, "offset_delta")?,
                elem_size,
                num_elements,
            });
        }
        out.insert(key, metas);
    }
    Ok(out)
}

fn parse_field_meta(
    native_meta: &Bound<'_, PyDict>,
) -> PyResult<FxHashMap<String, Vec<FieldMeta>>> {
    let mut out = FxHashMap::default();
    let obj = native_meta
        .get_item("field_inputs")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("field_inputs"))?;
    let dict: &Bound<'_, PyDict> = obj.downcast()?;
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        let items: &Bound<'_, PyList> = value.downcast()?;
        let mut metas = Vec::with_capacity(items.len());
        for item in items.iter() {
            let d: &Bound<'_, PyDict> = item.downcast()?;
            let kind = required_string(d, "kind")?;
            if kind != "value" && kind != "buffer" {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "field metadata kind must be 'value' or 'buffer', got {kind:?}"
                )));
            }
            let size_i64 = required_i64(d, "size")?;
            let size = usize::try_from(size_i64).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "field metadata size must be non-negative and fit usize, got {size_i64}"
                ))
            })?;
            metas.push(FieldMeta {
                mem_map: required_string(d, "mem_map")?,
                base_ptr: required_string(d, "base_ptr")?,
                offset_delta: required_i64(d, "offset_delta")?,
                size,
                kind,
            });
        }
        out.insert(key, metas);
    }
    Ok(out)
}

fn parse_string_map(
    native_meta: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<FxHashMap<String, String>> {
    let mut out = FxHashMap::default();
    let obj = native_meta
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?;
    let dict: &Bound<'_, PyDict> = obj.downcast()?;
    for (k, v) in dict.iter() {
        out.insert(k.extract()?, v.extract()?);
    }
    Ok(out)
}

/// Static byte spans of every annotated input object, keyed by non-shadow
/// mem_map, then by alias-canonical non-shadow root pointer.
///
/// Each span retains both the smallest relative offset and largest exclusive
/// end. This keeps negative-offset windows inside their assigned allocation
/// instead of letting them overlap the preceding object. Shadow metas mirror
/// the non-shadow lane at the same numeric addresses in their own map, so only
/// the non-shadow lane is tabulated.
fn input_object_spans(
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
    aliases: &FxHashMap<String, String>,
) -> PyResult<BTreeMap<String, BTreeMap<String, ObjectSpan>>> {
    let mut spans: BTreeMap<String, BTreeMap<String, ObjectSpan>> = BTreeMap::new();
    let mut note = |mem_map: &str, base_ptr: &str, offset: i64, size: i64| -> PyResult<()> {
        if mem_map.ends_with(".shadow") {
            return Ok(());
        }
        if size < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input window for {base_ptr} has negative size {size}"
            )));
        }
        let end = offset.checked_add(size).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "input window for {base_ptr} overflows i64: offset={offset}, size={size}"
            ))
        })?;
        let root = canon(strip_shadow(base_ptr), aliases);
        let entry = spans
            .entry(mem_map.to_string())
            .or_default()
            .entry(root)
            .or_insert(ObjectSpan {
                min_offset: offset,
                max_end: end,
            });
        entry.min_offset = entry.min_offset.min(offset);
        entry.max_end = entry.max_end.max(end);
        Ok(())
    };
    for metas in arr_inputs.values() {
        for meta in metas {
            let size = meta
                .elem_size
                .checked_mul(meta.num_elements)
                .ok_or_else(|| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "array input size overflows i64 for {}",
                        meta.base_ptr
                    ))
                })?;
            note(&meta.mem_map, &meta.base_ptr, meta.offset_delta, size)?;
        }
    }
    for metas in field_inputs.values() {
        for meta in metas {
            note(
                &meta.mem_map,
                &meta.base_ptr,
                meta.offset_delta,
                meta.size as i64,
            )?;
        }
    }
    for req in collect_alloc_requests(variables, arr_inputs)? {
        note(&req.mem_map, &req.base_ptr, req.offset_delta, req.size)?;
    }
    Ok(spans)
}

fn allocate_addresses(
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
    aliases: &FxHashMap<String, String>,
) -> PyResult<FxHashMap<String, i64>> {
    let per_map_spans = input_object_spans(variables, arr_inputs, field_inputs, aliases)?;
    let mut object_spans: BTreeMap<String, ObjectSpan> = BTreeMap::new();
    for ptrs in per_map_spans.values() {
        for (ptr, span) in ptrs {
            let combined = object_spans.entry(ptr.clone()).or_insert(*span);
            combined.min_offset = combined.min_offset.min(span.min_offset);
            combined.max_end = combined.max_end.max(span.max_end);
        }
    }

    // Pointer values share one program-wide domain even when their pointees
    // occupy different typed memory maps. Allocate every alias-canonical root
    // once, reserving the union of its relative spans across every map.
    let mut out = FxHashMap::default();
    let mut current = INPUT_OBJECT_BASE;
    for (ptr, span) in object_spans {
        let width = span.max_end.checked_sub(span.min_offset).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "input span width overflows i64 for {ptr}"
            ))
        })?;
        let aligned = width.max(1).checked_add(7).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "input span alignment overflows i64 for {ptr}"
            ))
        })? & !7;
        let pointer = current.checked_sub(span.min_offset).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "input pointer assignment overflows i64 for {ptr}"
            ))
        })?;
        out.insert(ptr.clone(), pointer);
        out.insert(format!("{}.shadow", ptr), pointer);
        current = current.checked_add(aligned).ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err(
                "input object address allocation overflows i64",
            )
        })?;
    }

    for (alias, _target) in aliases {
        let root = canon(strip_shadow(alias), aliases);
        if let Some(value) = out.get(&root).copied() {
            out.insert(alias.clone(), value);
        }
    }
    Ok(out)
}

fn validate_no_aliasing(
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
    aliases: &FxHashMap<String, String>,
    ptr_assignments: &FxHashMap<String, i64>,
) -> PyResult<()> {
    let spans = input_object_spans(variables, arr_inputs, field_inputs, aliases)?;
    for (mem_map, ptrs) in spans {
        let mut ranges = Vec::new();
        for (ptr, span) in ptrs {
            if let Some(addr) = ptr_assignments.get(&ptr) {
                let start = addr.checked_add(span.min_offset).ok_or_else(|| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "input range start overflows i64 for {ptr}"
                    ))
                })?;
                let end = addr.checked_add(span.max_end).ok_or_else(|| {
                    pyo3::exceptions::PyOverflowError::new_err(format!(
                        "input range end overflows i64 for {ptr}"
                    ))
                })?;
                ranges.push((ptr, start, end));
            }
        }
        ranges.sort_by_key(|(_, start, _)| *start);
        for pair in ranges.windows(2) {
            let (left_ptr, _left_start, left_end) = &pair[0];
            let (right_ptr, right_start, _right_end) = &pair[1];
            if left_end > right_start {
                return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                    "BUFFER ALIASING in {}: {}[..{}) overlaps {}[{}..)",
                    mem_map, left_ptr, left_end, right_ptr, right_start
                )));
            }
        }
    }
    Ok(())
}

fn collect_alloc_requests(
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
) -> PyResult<Vec<AllocRequest>> {
    let mut requests = Vec::new();
    for (key, inp) in variables.iter() {
        let var_name: String = key.extract()?;
        if var_name.ends_with(".shadow") {
            continue;
        }
        let shadow_key = format!("{}.shadow", var_name);

        if let Some(buffers) = input_list_attr(&inp, "buffers")? {
            if let Some(arr_infos) = arr_inputs.get(&var_name) {
                let non_shadow: Vec<&ArrayMeta> = arr_infos
                    .iter()
                    .filter(|m| !m.mem_map.contains(".shadow"))
                    .collect();
                for (buf, meta) in buffers.iter().zip(non_shadow.iter()) {
                    requests.push(AllocRequest {
                        mem_map: meta.mem_map.clone(),
                        base_ptr: meta.base_ptr.clone(),
                        offset_delta: meta.offset_delta,
                        size: dict_i64(&buf, "size")?,
                    });
                }
            }
            if let Some(shadow_infos) = arr_inputs.get(&shadow_key) {
                let shadow: Vec<&ArrayMeta> = shadow_infos
                    .iter()
                    .filter(|m| m.mem_map.contains(".shadow"))
                    .collect();
                if !shadow.is_empty() {
                    let shadow_inp = require_shadow_input(variables, &var_name, &shadow_key)?;
                    let shadow_buffers =
                        require_input_list_attr(&shadow_inp, &var_name, &shadow_key, "buffers")?;
                    for (buf, meta) in shadow_buffers.iter().zip(shadow.iter()) {
                        requests.push(AllocRequest {
                            mem_map: meta.mem_map.clone(),
                            base_ptr: meta.base_ptr.clone(),
                            offset_delta: meta.offset_delta,
                            size: dict_i64(&buf, "size")?,
                        });
                    }
                }
            }
        }

        if let Some(fields) = input_list_attr(&inp, "struct")? {
            if let Some(arr_infos) = arr_inputs.get(&var_name) {
                let arr_non_shadow: Vec<&ArrayMeta> = arr_infos
                    .iter()
                    .filter(|m| !m.mem_map.contains(".shadow"))
                    .collect();
                let arr_shadow: Vec<&ArrayMeta> = arr_inputs
                    .get(&shadow_key)
                    .map(|items| {
                        items
                            .iter()
                            .filter(|m| m.mem_map.contains(".shadow"))
                            .collect()
                    })
                    .unwrap_or_default();
                let shadow_struct = if arr_shadow.is_empty() {
                    None
                } else {
                    let shadow_inp = require_shadow_input(variables, &var_name, &shadow_key)?;
                    Some(require_input_list_attr(
                        &shadow_inp,
                        &var_name,
                        &shadow_key,
                        "struct",
                    )?)
                };
                let mut buf_idx = 0usize;
                for (field_idx, field) in fields.iter().enumerate() {
                    if dict_get(&field, "buffer")?.is_some() {
                        if let Some(meta) = arr_non_shadow.get(buf_idx) {
                            let buf = dict_get(&field, "buffer")?.unwrap();
                            requests.push(AllocRequest {
                                mem_map: meta.mem_map.clone(),
                                base_ptr: meta.base_ptr.clone(),
                                offset_delta: meta.offset_delta,
                                size: dict_i64(&buf, "size")?,
                            });
                        }
                        if let Some(meta) = arr_shadow.get(buf_idx) {
                            let shadow_field = shadow_struct
                                .as_ref()
                                .expect("shadow struct validated above")
                                .get_item(field_idx)?;
                            let buf = dict_get(&shadow_field, "buffer")?.ok_or_else(|| {
                                missing_shadow_payload(&var_name, &shadow_key, "struct buffer")
                            })?;
                            requests.push(AllocRequest {
                                mem_map: meta.mem_map.clone(),
                                base_ptr: meta.base_ptr.clone(),
                                offset_delta: meta.offset_delta,
                                size: dict_i64(&buf, "size")?,
                            });
                        }
                        buf_idx += 1;
                    }
                }
            }
        }
    }
    Ok(requests)
}

fn concretize_memory(
    program: &CompiledProgram,
    vm: &mut VM,
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
) -> PyResult<()> {
    for (key, inp) in variables.iter() {
        let var_name: String = key.extract()?;
        if var_name.ends_with(".shadow") {
            continue;
        }
        if let Some(buffers) = input_list_attr(&inp, "buffers")? {
            write_input_buffers(program, vm, variables, arr_inputs, &var_name, &buffers)?;
        }
        if let Some(fields) = input_list_attr(&inp, "struct")? {
            write_input_struct(
                program,
                vm,
                variables,
                arr_inputs,
                field_inputs,
                &var_name,
                &fields,
            )?;
        }
    }
    Ok(())
}

fn write_input_buffers(
    program: &CompiledProgram,
    vm: &mut VM,
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    var_name: &str,
    buffers: &Bound<'_, PyList>,
) -> PyResult<()> {
    let shadow_key = format!("{}.shadow", var_name);
    let non_shadow: Vec<&ArrayMeta> = arr_inputs
        .get(var_name)
        .map(|items| {
            items
                .iter()
                .filter(|m| !m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    for (buf, meta) in buffers.iter().zip(non_shadow.iter()) {
        write_buffer(program, vm, &buf, meta)?;
    }

    let shadow: Vec<&ArrayMeta> = arr_inputs
        .get(&shadow_key)
        .map(|items| {
            items
                .iter()
                .filter(|m| m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    if shadow.is_empty() {
        return Ok(());
    }
    let shadow_inp = require_shadow_input(variables, var_name, &shadow_key)?;
    let shadow_buffers = require_input_list_attr(&shadow_inp, var_name, &shadow_key, "buffers")?;
    for (buf, meta) in shadow_buffers.iter().zip(shadow.iter()) {
        write_buffer(program, vm, &buf, meta)?;
    }
    Ok(())
}

fn write_input_struct(
    program: &CompiledProgram,
    vm: &mut VM,
    variables: &Bound<'_, PyDict>,
    arr_inputs: &FxHashMap<String, Vec<ArrayMeta>>,
    field_inputs: &FxHashMap<String, Vec<FieldMeta>>,
    var_name: &str,
    fields: &Bound<'_, PyList>,
) -> PyResult<()> {
    let shadow_key = format!("{}.shadow", var_name);
    let field_infos: Vec<&FieldMeta> = field_inputs
        .get(var_name)
        .map(|items| {
            items
                .iter()
                .filter(|m| !m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    validate_struct_fields(var_name, fields, &field_infos)?;
    let field_infos_shadow: Vec<&FieldMeta> = field_inputs
        .get(&shadow_key)
        .map(|items| {
            items
                .iter()
                .filter(|m| m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    let arr_non_shadow: Vec<&ArrayMeta> = arr_inputs
        .get(var_name)
        .map(|items| {
            items
                .iter()
                .filter(|m| !m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    let arr_shadow: Vec<&ArrayMeta> = arr_inputs
        .get(&shadow_key)
        .map(|items| {
            items
                .iter()
                .filter(|m| m.mem_map.contains(".shadow"))
                .collect()
        })
        .unwrap_or_default();
    let shadow_struct = if field_infos_shadow.is_empty() && arr_shadow.is_empty() {
        None
    } else {
        let shadow_inp = require_shadow_input(variables, var_name, &shadow_key)?;
        Some(require_input_list_attr(
            &shadow_inp,
            var_name,
            &shadow_key,
            "struct",
        )?)
    };
    if let Some(fields) = shadow_struct.as_ref() {
        validate_struct_fields(&shadow_key, fields, &field_infos_shadow)?;
    }

    let mut field_idx = 0usize;
    let mut buffer_idx = 0usize;
    for struct_idx in 0..fields.len() {
        let field = fields.get_item(struct_idx)?;
        let shadow_field = match shadow_struct.as_ref() {
            Some(items) => Some(items.get_item(struct_idx)?),
            None => None,
        };
        if dict_get(&field, "value")?.is_some() {
            if let Some(meta) = field_infos.get(field_idx) {
                write_field(program, vm, &field, meta)?;
            }
            if let Some(meta) = field_infos_shadow.get(field_idx) {
                write_field(
                    program,
                    vm,
                    shadow_field
                        .as_ref()
                        .expect("shadow field metadata requires shadow struct"),
                    meta,
                )?;
            }
            field_idx += 1;
        } else if let Some(buf) = dict_get(&field, "buffer")? {
            if let Some(meta) = arr_non_shadow.get(buffer_idx) {
                let addr = offset_value(vm, program, &meta.base_ptr, meta.offset_delta)?;
                let size = dict_i64(&field, "size")? as usize;
                let addr_field = field_with_value(field.py(), addr, size)?;
                if let Some(fmeta) = field_infos.get(field_idx) {
                    write_field(program, vm, &addr_field, fmeta)?;
                }
                if let Some(fmeta) = field_infos_shadow.get(field_idx) {
                    write_field(program, vm, &addr_field, fmeta)?;
                }
                write_buffer(program, vm, &buf, meta)?;
            }
            if let Some(meta) = arr_shadow.get(buffer_idx) {
                let shadow_buf = dict_get(
                    shadow_field
                        .as_ref()
                        .expect("shadow array metadata requires shadow struct"),
                    "buffer",
                )?
                .ok_or_else(|| missing_shadow_payload(var_name, &shadow_key, "struct buffer"))?;
                write_buffer(program, vm, &shadow_buf, meta)?;
            }
            field_idx += 1;
            buffer_idx += 1;
        }
    }
    Ok(())
}

fn validate_struct_fields(
    var_name: &str,
    fields: &Bound<'_, PyList>,
    metadata: &[&FieldMeta],
) -> PyResult<()> {
    if fields.len() != metadata.len() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "Struct field count mismatch for {var_name}: input has {} fields, metadata has {}",
            fields.len(),
            metadata.len()
        )));
    }
    for (index, meta) in metadata.iter().enumerate() {
        let field = fields.get_item(index)?;
        let has_value = dict_get(&field, "value")?.is_some();
        let has_buffer = dict_get(&field, "buffer")?.is_some();
        let actual_kind = match (has_value, has_buffer) {
            (true, false) => "value",
            (false, true) => "buffer",
            _ => "invalid",
        };
        if actual_kind != meta.kind {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Struct field kind mismatch for {var_name}[{index}]: expected {}, got {actual_kind}",
                meta.kind
            )));
        }
        let size = dict_i64(&field, "size")?;
        if size < 0 || size as usize != meta.size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Struct field size mismatch for {var_name}[{index}]: expected {}, got {size}",
                meta.size
            )));
        }
    }
    Ok(())
}

fn write_buffer(
    program: &CompiledProgram,
    vm: &mut VM,
    datum: &Bound<'_, PyAny>,
    meta: &ArrayMeta,
) -> PyResult<()> {
    let size = dict_i64(datum, "size")? as usize;
    let contents = dict_get(datum, "contents")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("contents"))?;
    let data = bytes_from_contents(&contents, size)?;
    let base = offset_value(vm, program, &meta.base_ptr, meta.offset_delta)?;
    let map_idx = map_index(program, vm, &meta.mem_map)?;
    for i in 0..size {
        let val = data.get(i).copied().unwrap_or(0) as i64;
        vm.memory_maps[map_idx].set(base + i as i64, val);
    }
    Ok(())
}

fn write_field(
    program: &CompiledProgram,
    vm: &mut VM,
    field_data: &Bound<'_, PyAny>,
    meta: &FieldMeta,
) -> PyResult<()> {
    let value = dict_get(field_data, "value")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("value"))?;
    let mut contents = bytes_from_contents(&value, meta.size)?;
    contents.reverse();
    if contents.len() != meta.size {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "Field size mismatch: data={}, annotation={}",
            contents.len(),
            meta.size
        )));
    }
    let map_idx = map_index(program, vm, &meta.mem_map)?;
    let elt_bytes = (vm.memory_maps[map_idx].element_bit_width as usize / 8).max(1);
    let base = offset_value(vm, program, &meta.base_ptr, meta.offset_delta)?;
    // MemoryMap addresses name element slots. load_wide/store_wide advance
    // one slot per element, irrespective of that element's bit width.
    for (i, chunk) in contents.chunks(elt_bytes).enumerate() {
        let mut value = 0i64;
        for (shift, byte) in chunk.iter().enumerate() {
            value |= (*byte as i64) << (8 * shift);
        }
        vm.memory_maps[map_idx].set(base + i as i64, value);
    }
    Ok(())
}

fn offset_value(
    vm: &VM,
    program: &CompiledProgram,
    base_ptr: &str,
    offset_delta: i64,
) -> PyResult<i64> {
    let Some(vid) = program.lookup_var(base_ptr) else {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "unknown base pointer {}",
            base_ptr
        )));
    };
    Ok(vm.get_scalar_silent(vid).wrapping_add(offset_delta))
}

fn map_index(program: &CompiledProgram, vm: &VM, map_name: &str) -> PyResult<usize> {
    let Some(vid) = program.lookup_var(map_name) else {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "unknown memory map {}",
            map_name
        )));
    };
    vm.var_to_map.get(&vid).copied().ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("uninitialized memory map {}", map_name))
    })
}

fn canon(name: &str, aliases: &FxHashMap<String, String>) -> String {
    let mut current = name.to_string();
    let mut seen = BTreeSet::new();
    while let Some(next) = aliases.get(&current) {
        if !seen.insert(current.clone()) {
            break;
        }
        current = next.clone();
    }
    current
}

fn input_value(inp: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    let value = inp.getattr("value")?;
    if value.is_none() {
        return Ok(None);
    }
    extract_i64(&value).map(Some)
}

fn input_list_attr<'py>(
    inp: &Bound<'py, PyAny>,
    attr: &str,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let value = inp.getattr(attr)?;
    if value.is_none() {
        return Ok(None);
    }
    let list = value.downcast_into::<PyList>()?;
    if list.is_empty() {
        return Ok(None);
    }
    Ok(Some(list))
}

fn input_havoc_attr<'py>(inp: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyList>>> {
    let value = inp.getattr("havoc_seq")?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.downcast_into::<PyList>()?))
}

fn dict_get<'py>(obj: &Bound<'py, PyAny>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let dict: &Bound<'_, PyDict> = obj.downcast()?;
    dict.get_item(key)
}

fn dict_i64(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<i64> {
    let value = dict_get(obj, key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?;
    extract_i64(&value)
}

fn required_string(d: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    d.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
        .extract()
}

fn required_i64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let value = d
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?;
    extract_i64(&value)
}

fn extract_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(v) = value.extract::<i64>() {
        return Ok(v);
    }
    let text = value.str()?.to_str()?.to_string();
    let parsed = text.parse::<i128>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("expected integer {text:?}: {e}"))
    })?;
    Ok(parsed as u64 as i64)
}

fn bytes_from_contents(value: &Bound<'_, PyAny>, size: usize) -> PyResult<Vec<u8>> {
    if let Ok(items) = value.downcast::<PyList>() {
        let mut out = Vec::with_capacity(size);
        for item in items.iter().take(size) {
            out.push((extract_i64(&item)? & 0xff) as u8);
        }
        out.resize(size, 0);
        return Ok(out);
    }
    let mut s: String = value.extract()?;
    s = s.trim().to_string();
    if let Some(n) = helper_size(&s, "zeros") {
        return Ok(vec![0; size.max(n)]);
    }
    if let Some(n) = helper_size(&s, "ones") {
        return Ok(vec![0xff; size.max(n)]);
    }
    if s.starts_with("0x") || s.starts_with("0X") {
        s = s[2..].to_string();
    }
    let mut out = Vec::with_capacity(size);
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i + 1 < bytes.len() && out.len() < size {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    out.resize(size, 0);
    Ok(out)
}

fn helper_size(s: &str, name: &str) -> Option<usize> {
    let prefix = format!("{name}(");
    if !s.starts_with(&prefix) || !s.ends_with(')') {
        return None;
    }
    s[prefix.len()..s.len() - 1].parse().ok()
}

fn hex_nibble(byte: u8) -> PyResult<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid hex byte {:?}",
            byte as char
        ))),
    }
}

fn field_with_value(py: Python<'_>, value: i64, size: usize) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new_bound(py);
    let mask = if size >= 8 {
        value as u64
    } else {
        (value as u64) & ((1u64 << (size * 8)) - 1)
    };
    dict.set_item("value", format!("0x{:0width$x}", mask, width = size * 2))?;
    dict.set_item("size", size)?;
    Ok(dict)
}
