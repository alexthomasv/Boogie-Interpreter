mod builtins;
mod lowering;
mod memory_map;
mod opcodes;
mod trace;
mod vm;

#[cfg(kani)]
mod kani_proofs;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PySet, PyTuple};

/// Lower a Python AST program into bytecode. Returns an opaque handle.
/// Called once; the result is passed to `execute`.
#[pyfunction]
fn lower(py: Python<'_>, program: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let compiled = lowering::lower_program(py, program)?;
    let wrapper = CompiledProgramWrapper { inner: compiled };
    Ok(Py::new(py, wrapper)?.into_py(py))
}

#[pyclass]
struct CompiledProgramWrapper {
    inner: opcodes::CompiledProgram,
}

/// Execute a pre-lowered program with pre-initialized state from Python.
///
/// Args:
///   compiled: result of `lower(program)`
///   var_store: dict[str, int] — scalar variable values after Python concretization
///   memory_maps: dict[str, dict[int, int]] — memory map contents after Python concretization
///   mem_map_info: list of (name, index_bw, element_bw) — metadata for each memory map
///   extra_data: optional bytes for read.cross_product
///   log_read: whether to log read ops in trace
///   no_trace: disable all tracing
///   pickled: if true, value sets contain pre-pickled bytes matching Python's format
///
/// Returns dict with 'explored_blocks' and 'compact_trace'.
#[pyfunction]
#[pyo3(signature = (compiled, var_store, memory_maps, mem_map_info, extra_data=None, log_read=true, no_trace=false, pickled=false))]
fn execute(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    var_store: &Bound<'_, PyDict>,
    memory_maps: &Bound<'_, PyDict>,
    mem_map_info: &Bound<'_, PyList>,
    extra_data: Option<Vec<u8>>,
    log_read: bool,
    no_trace: bool,
    pickled: bool,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let program = &wrapper.inner;

    let mut vm = vm::VM::new(program);
    vm.log_read = log_read;
    vm.no_trace = no_trace;
    if let Some(data) = extra_data {
        vm.external_buffer = data;
    }

    // Initialize memory maps from Python-provided metadata
    // pf:ensures:var_lookup.constant_time - O(1) HashMap lookup
    for item in mem_map_info.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let name: String = tuple.get_item(0)?.extract()?;
        let index_bw: u8 = tuple.get_item(1)?.extract()?;
        let element_bw: u8 = tuple.get_item(2)?.extract()?;

        if let Some(&vid) = program.name_to_var.get(&name) {
            vm.init_memory_map(vid, name.clone(), index_bw, element_bw);
        }
    }

    // Load scalar variable values from Python
    // pf:ensures:var_lookup.constant_time - O(1) HashMap lookup
    for (key, val) in var_store.iter() {
        let name: String = key.extract()?;
        let value: i64 = val.extract()?;
        if let Some(&vid) = program.name_to_var.get(&name) {
            vm.set_scalar(vid, value, true);
        }
    }

    // Load memory map contents from Python
    // pf:ensures:var_lookup.constant_time - O(1) HashMap lookup
    for (key, val) in memory_maps.iter() {
        let name: String = key.extract()?;
        let contents: &Bound<'_, PyDict> = val.downcast()?;
        if let Some(&vid) = program.name_to_var.get(&name) {
            if let Some(map_idx) = vm.var_to_map[vid as usize] {
                for (addr, value) in contents.iter() {
                    let a: i64 = addr.extract()?;
                    let v: i64 = value.extract()?;
                    vm.memory_maps[map_idx].set(a, v);
                }
            }
        }
    }

    // Execute
    let exec_start = std::time::Instant::now();
    vm.execute(program);
    let exec_elapsed = exec_start.elapsed();
    eprintln!(
        "[native] Execution: {:.1?}, {} blocks, {} trace entries",
        exec_elapsed,
        vm.explored_blocks.len(),
        vm.trace.total
    );

    // Build result
    let trace_start = std::time::Instant::now();
    let result = PyDict::new_bound(py);

    let blocks_set = PySet::empty_bound(py)?;
    for block in &vm.explored_blocks {
        blocks_set.add(block.as_str())?;
    }
    result.set_item("explored_blocks", blocks_set)?;

    let block_names: Vec<String> = program.blocks.iter().map(|b| b.name.clone()).collect();
    let compact = if pickled {
        build_compact_trace_pickled(py, &vm.trace, &program.var_names, &block_names)?
    } else {
        build_compact_trace_raw(py, &vm.trace, &program.var_names, &block_names)?
    };
    let trace_elapsed = trace_start.elapsed();
    eprintln!("[native] Trace build: {:.1?}", trace_elapsed);
    result.set_item("compact_trace", compact)?;

    Ok(result.into())
}

#[pyfunction]
fn get_var_names(py: Python<'_>, compiled: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let names: Vec<&str> = wrapper.inner.var_names.iter().map(|s| s.as_str()).collect();
    Ok(names.into_py(py))
}

/// Convert i64 to unsigned Python int (matching Python's unbounded int semantics).
#[inline]
fn to_py_int(py: Python<'_>, val: i64) -> PyObject {
    if val < 0 {
        (val as u64).into_py(py)
    } else {
        val.into_py(py)
    }
}

// ---- Pickle protocol 5 encoder for (int, "") tuples ----
//
// Format: PROTO(5) + FRAME(len) + <int_encoding> + SHORT_BINUNICODE("") + MEMOIZE + TUPLE2 + MEMOIZE + STOP
// The suffix after the int is always: 8c 00 94 86 94 2e

const PICKLE_SUFFIX: &[u8] = b"\x8c\x00\x94\x86\x94\x2e"; // SHORT_BINUNICODE(""), MEMOIZE, TUPLE2, MEMOIZE, STOP

/// Encode a Python unsigned integer in pickle protocol 5 format within a (int, "") tuple.
/// The value is i64 in Rust; negative values represent Python unsigned ints > 2^63.
fn pickle_int_tuple(val: i64) -> Vec<u8> {
    // Convert to Python's unsigned representation
    let pyval: u64 = val as u64;

    // Encode the integer part
    let int_bytes = encode_pickle_int(pyval);
    let payload_len = int_bytes.len() + PICKLE_SUFFIX.len();

    let mut buf = Vec::with_capacity(2 + 9 + payload_len);
    // PROTO 5
    buf.push(0x80);
    buf.push(0x05);
    // FRAME
    buf.push(0x95);
    buf.extend_from_slice(&(payload_len as u64).to_le_bytes());
    // Int encoding
    buf.extend_from_slice(&int_bytes);
    // Suffix
    buf.extend_from_slice(PICKLE_SUFFIX);
    buf
}

/// Encode a u64 value using pickle opcodes (BININT1, BININT2, BININT, or LONG1).
fn encode_pickle_int(val: u64) -> Vec<u8> {
    if val <= 255 {
        // BININT1: \x4b <byte>
        vec![0x4b, val as u8]
    } else if val <= 65535 {
        // BININT2: \x4d <2 bytes LE>
        let bytes = (val as u16).to_le_bytes();
        vec![0x4d, bytes[0], bytes[1]]
    } else if val <= 0x7FFF_FFFF {
        // BININT: \x4a <4 bytes signed LE> — fits in positive i32
        let bytes = (val as u32).to_le_bytes();
        vec![0x4a, bytes[0], bytes[1], bytes[2], bytes[3]]
    } else {
        // LONG1: \x8a <length byte> <signed LE bytes>
        // Need to encode as signed LE with minimal bytes
        encode_long1(val)
    }
}

/// Encode a u64 as LONG1 (variable-length signed little-endian).
fn encode_long1(val: u64) -> Vec<u8> {
    let bytes = val.to_le_bytes();
    // Find minimal length: trim trailing 0x00 bytes, but keep sign byte
    let mut len = 8;
    while len > 1 && bytes[len - 1] == 0 {
        len -= 1;
    }
    // If high bit is set, add a 0x00 sign byte to keep it positive
    let need_sign = bytes[len - 1] & 0x80 != 0;
    let total_len = len + if need_sign { 1 } else { 0 };

    let mut buf = Vec::with_capacity(2 + total_len);
    buf.push(0x8a); // LONG1
    buf.push(total_len as u8);
    buf.extend_from_slice(&bytes[..len]);
    if need_sign {
        buf.push(0x00);
    }
    buf
}

/// Encode a value for pickle registry (raw int or string, not a tuple).
fn pickle_raw_int(val: u32) -> Vec<u8> {
    let pyval = val as u64;
    let int_bytes = encode_pickle_int(pyval);
    let payload_len = int_bytes.len() + 1; // +1 for STOP

    let mut buf = Vec::with_capacity(2 + 9 + payload_len);
    buf.push(0x80);
    buf.push(0x05);
    buf.push(0x95);
    buf.extend_from_slice(&(payload_len as u64).to_le_bytes());
    buf.extend_from_slice(&int_bytes);
    buf.push(0x2e); // STOP
    buf
}

/// Encode a string for pickle registry.
fn pickle_raw_string(s: &str) -> Vec<u8> {
    let s_bytes = s.as_bytes();
    let s_len = s_bytes.len();

    // SHORT_BINUNICODE if len < 256, else BINUNICODE
    let str_enc_len = if s_len < 256 {
        2 + s_len // \x8c <len_byte> <data>
    } else {
        5 + s_len // \x8d <4-byte len> <data>
    };
    let payload_len = str_enc_len + 2; // +MEMOIZE +STOP

    let mut buf = Vec::with_capacity(2 + 9 + payload_len);
    buf.push(0x80);
    buf.push(0x05);
    buf.push(0x95);
    buf.extend_from_slice(&(payload_len as u64).to_le_bytes());
    if s_len < 256 {
        buf.push(0x8c);
        buf.push(s_len as u8);
    } else {
        buf.push(0x8d);
        buf.extend_from_slice(&(s_len as u32).to_le_bytes());
    }
    buf.extend_from_slice(s_bytes);
    buf.push(0x94); // MEMOIZE
    buf.push(0x2e); // STOP
    buf
}

/// Build compact trace with pre-pickled bytes (matching Python's flush_compact format).
/// Value sections contain pickle.dumps((int, "")) bytes.
/// Registry sections contain pickle.dumps(int) or pickle.dumps(str) bytes.
fn build_compact_trace_pickled(
    py: Python<'_>,
    trace: &trace::TraceAccumulator,
    var_names: &[String],
    block_names: &[String],
) -> PyResult<PyObject> {
    let compact = PyDict::new_bound(py);

    // Helper closure: build a dict of {key: set_of_pickled_bytes} for value sections
    macro_rules! build_value_section {
        ($section:expr, $iter:expr, $key_fmt:expr) => {{
            let section = PyDict::new_bound(py);
            for (composite_key, values) in $iter {
                let key = $key_fmt(composite_key);
                let pset = PySet::empty_bound(py)?;
                for tv in values {
                    let pickled = pickle_int_tuple(tv.value);
                    pset.add(PyBytes::new_bound(py, &pickled))?;
                }
                section.set_item(key, pset)?;
            }
            compact.set_item($section, section)?;
        }};
    }

    // pc_values
    build_value_section!("pc_values", &trace.pc_values, |k: &(u32, u32)| {
        format!("positive_examples_{}_{}", var_names[k.0 as usize], k.1)
    });

    // block_values
    build_value_section!("block_values", &trace.block_values, |k: &(u32, u32)| {
        format!(
            "positive_examples_{}_{}",
            var_names[k.0 as usize], block_names[k.1 as usize]
        )
    });

    // op_values
    build_value_section!("op_values", &trace.op_values, |k: &(u32, u32, u8)| {
        format!(
            "positive_examples_to_op_type_{}_{}_{}",
            var_names[k.0 as usize], k.1, k.2 as char
        )
    });

    // pc_registry: pickled raw ints
    let pc_registry = PyDict::new_bound(py);
    for (&var_id, pcs) in &trace.pc_registry {
        let key = format!("positive_examples_to_pc_{}", var_names[var_id as usize]);
        let pset = PySet::empty_bound(py)?;
        for &pc in pcs {
            let pickled = pickle_raw_int(pc);
            pset.add(PyBytes::new_bound(py, &pickled))?;
        }
        pc_registry.set_item(key, pset)?;
    }
    compact.set_item("pc_registry", pc_registry)?;

    // block_registry: pickled raw strings
    let block_registry = PyDict::new_bound(py);
    for (&var_id, blocks) in &trace.block_registry {
        let key = format!("positive_examples_to_block_{}", var_names[var_id as usize]);
        let pset = PySet::empty_bound(py)?;
        for &block_id in blocks {
            let pickled = pickle_raw_string(&block_names[block_id as usize]);
            pset.add(PyBytes::new_bound(py, &pickled))?;
        }
        block_registry.set_item(key, pset)?;
    }
    compact.set_item("block_registry", block_registry)?;

    compact.set_item("total", trace.total)?;

    Ok(compact.into())
}

/// Build compact trace with raw values (for comparison mode).
fn build_compact_trace_raw(
    py: Python<'_>,
    trace: &trace::TraceAccumulator,
    var_names: &[String],
    block_names: &[String],
) -> PyResult<PyObject> {
    let compact = PyDict::new_bound(py);

    // Encode TracedValue as 12-byte (8B value LE + 4B iteration_id LE) for v3 format
    fn traced_value_bytes(tv: &trace::TracedValue) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[..8].copy_from_slice(&(tv.value as u64).to_le_bytes());
        buf[8..].copy_from_slice(&tv.iteration_id.to_le_bytes());
        buf
    }

    let pc_values = PyDict::new_bound(py);
    for (&(var_id, pc), values) in &trace.pc_values {
        let key = format!("positive_examples_{}_{}", var_names[var_id as usize], pc);
        let pset = PySet::empty_bound(py)?;
        for tv in values {
            pset.add(PyBytes::new_bound(py, &traced_value_bytes(tv)))?;
        }
        pc_values.set_item(key, pset)?;
    }
    compact.set_item("pc_values", pc_values)?;

    let block_values = PyDict::new_bound(py);
    for (&(var_id, block_id), values) in &trace.block_values {
        let block_name = &block_names[block_id as usize];
        let key = format!(
            "positive_examples_{}_{}",
            var_names[var_id as usize], block_name
        );
        let pset = PySet::empty_bound(py)?;
        for tv in values {
            pset.add(PyBytes::new_bound(py, &traced_value_bytes(tv)))?;
        }
        block_values.set_item(key, pset)?;
    }
    compact.set_item("block_values", block_values)?;

    let op_values = PyDict::new_bound(py);
    for (&(var_id, pc, op), values) in &trace.op_values {
        let op_char = op as char;
        let key = format!(
            "positive_examples_to_op_type_{}_{}_{}",
            var_names[var_id as usize], pc, op_char
        );
        let pset = PySet::empty_bound(py)?;
        for tv in values {
            pset.add(PyBytes::new_bound(py, &traced_value_bytes(tv)))?;
        }
        op_values.set_item(key, pset)?;
    }
    compact.set_item("op_values", op_values)?;

    let pc_registry = PyDict::new_bound(py);
    for (&var_id, pcs) in &trace.pc_registry {
        let key = format!("positive_examples_to_pc_{}", var_names[var_id as usize]);
        let pset = PySet::empty_bound(py)?;
        for &pc in pcs {
            pset.add(pc)?;
        }
        pc_registry.set_item(key, pset)?;
    }
    compact.set_item("pc_registry", pc_registry)?;

    let block_registry = PyDict::new_bound(py);
    for (&var_id, blocks) in &trace.block_registry {
        let key = format!("positive_examples_to_block_{}", var_names[var_id as usize]);
        let pset = PySet::empty_bound(py)?;
        for &block_id in blocks {
            pset.add(block_names[block_id as usize].as_str())?;
        }
        block_registry.set_item(key, pset)?;
    }
    compact.set_item("block_registry", block_registry)?;

    compact.set_item("total", trace.total)?;

    Ok(compact.into())
}

#[pymodule]
fn swoosh_interp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lower, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(get_var_names, m)?)?;
    m.add_class::<CompiledProgramWrapper>()?;
    Ok(())
}
