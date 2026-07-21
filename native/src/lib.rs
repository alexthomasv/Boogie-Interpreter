mod builtins;
mod concolic;
mod debug_log;
mod input_state;
mod lowering;
mod memory_map;
mod opcodes;
mod raw_log;
mod raw_log_reader;
mod trace;
mod trace_index;
mod vm;

#[cfg(kani)]
mod kani_proofs;

use concolic::explore;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use std::time::Duration;
use vm::ExecutionStatus;

/// Lower a Python AST program into bytecode. Returns an opaque handle.
/// Called once; the result is passed to `execute`.
///
/// `loop_header_live` is an optional dict {block_name: [var_name, ...]}
/// mapping each loop header block to the variables live at its entry.
/// When provided, the interpreter snapshots those variables on each
/// loop header visit, enabling iteration-aware trace data.
#[pyfunction]
#[pyo3(signature = (program, loop_header_live=None, loop_metadata=None, *, mode))]
fn lower(
    py: Python<'_>,
    program: &Bound<'_, PyAny>,
    loop_header_live: Option<&Bound<'_, PyDict>>,
    loop_metadata: Option<&Bound<'_, PyDict>>,
    mode: &str,
) -> PyResult<PyObject> {
    let mode = parse_semantics_mode(mode)?;
    let compiled =
        lowering::lower_program_full(py, program, loop_header_live, loop_metadata, mode)?;
    let wrapper = CompiledProgramWrapper { inner: compiled };
    Ok(Py::new(py, wrapper)?.into_py(py))
}

/// Parse the required Python-side semantics-mode tag ("int"/"bv").
fn parse_semantics_mode(mode: &str) -> PyResult<opcodes::SemanticsMode> {
    opcodes::SemanticsMode::from_str(mode).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyclass]
pub(crate) struct CompiledProgramWrapper {
    pub(crate) inner: opcodes::CompiledProgram,
}

#[pymethods]
impl CompiledProgramWrapper {
    /// Semantics mode the program was lowered under: "int" or "bv".
    #[getter]
    fn mode(&self) -> &'static str {
        self.inner.mode.as_str()
    }

    #[getter]
    fn num_vars(&self) -> u32 {
        self.inner.num_vars
    }

    #[getter]
    fn num_blocks(&self) -> usize {
        self.inner.blocks.len()
    }

    #[getter]
    fn var_name_bytes(&self) -> usize {
        self.inner.var_names.iter().map(str::len).sum()
    }

    #[getter]
    fn block_name_bytes(&self) -> usize {
        self.inner.blocks.iter().map(|block| block.name.len()).sum()
    }

    #[getter]
    fn mem_map_count(&self) -> usize {
        self.inner.mem_maps.len()
    }
}

/// Inline `{:inline}` procedures and lower straight to bytecode, returning the
/// opaque CompiledProgram handle. Like `lower`, but the input is the *un-inlined*
/// shadowed AST — inlining happens natively in Rust (no Boogie, no reparse).
/// Used by the differential test harness and the in-memory interpret path.
#[pyfunction]
#[pyo3(signature = (program, mode))]
fn inline_lower(py: Python<'_>, program: &Bound<'_, PyAny>, mode: &str) -> PyResult<PyObject> {
    let mode = parse_semantics_mode(mode)?;
    let compiled = lowering::inline::inline_lower_program(
        py, program, None, /*retain_runtime_name_lookup*/ true, mode,
    )?;
    let wrapper = CompiledProgramWrapper { inner: compiled };
    Ok(Py::new(py, wrapper)?.into_py(py))
}

/// Inline + lower + serialize (bincode, zstd-compressed) to `path` as a `.swcp`
/// bytecode package — the compact, Python-AST-free artifact the interpreter
/// loads directly via `load_compiled`. `static_scalars` (name→value, computed in
/// Python from the un-inlined program) is baked in so a concrete run needs no
/// Python-AST-derived native_meta.
#[pyfunction]
#[pyo3(signature = (program, path, static_scalars=None, *, mode))]
fn inline_lower_to_file(
    py: Python<'_>,
    program: &Bound<'_, PyAny>,
    path: &str,
    static_scalars: Option<&Bound<'_, PyDict>>,
    mode: &str,
) -> PyResult<()> {
    let mode = parse_semantics_mode(mode)?;
    let compiled = lowering::inline::inline_lower_program(
        py,
        program,
        static_scalars,
        /*retain_runtime_name_lookup*/ false,
        mode,
    )?;
    // Stream beside the destination and publish only after the encoder has
    // finished, flushed, and synced. A budget kill may leave this hidden temp
    // file, but it cannot truncate a previously good package.
    let destination = std::path::Path::new(path);
    let parent = destination
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let file_name = destination
        .file_name()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("SWCP path has no file name"))?
        .to_string_lossy();
    let temporary = parent.join(format!(".{file_name}.tmp.{}", std::process::id()));
    let write_result = (|| -> PyResult<()> {
        let file = std::fs::File::create(&temporary).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("create {}: {}", temporary.display(), e))
        })?;
        let writer = std::io::BufWriter::new(file);
        let mut encoder = zstd::Encoder::new(writer, 3)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("zstd encoder: {}", e)))?;
        bincode::serialize_into(&mut encoder, &compiled).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("bincode serialize: {}", e))
        })?;
        let mut writer = encoder
            .finish()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("zstd finish: {}", e)))?;
        std::io::Write::flush(&mut writer).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("flush {}: {}", temporary.display(), e))
        })?;
        writer.get_ref().sync_all().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("sync {}: {}", temporary.display(), e))
        })?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&temporary, destination) {
        let _ = std::fs::remove_file(&temporary);
        return Err(pyo3::exceptions::PyIOError::new_err(format!(
            "publish {} as {}: {}",
            temporary.display(),
            destination.display(),
            error
        )));
    }
    Ok(())
}

/// Load a `.swcp` bytecode package (zstd + bincode), rebuild the derived lookup
/// maps, and return the opaque CompiledProgram handle for `execute`.
#[pyfunction]
fn load_compiled(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("open {}: {}", path, e)))?;
    let reader = std::io::BufReader::new(file);
    let decoder = zstd::Decoder::new(reader)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("zstd decoder: {}", e)))?;
    let mut compiled: opcodes::CompiledProgram =
        bincode::deserialize_from(decoder).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("bincode deserialize: {}", e))
        })?;
    compiled.rebuild_runtime_name_index();
    let wrapper = CompiledProgramWrapper { inner: compiled };
    Ok(Py::new(py, wrapper)?.into_py(py))
}

const FNV64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x100000001b3;

fn fnv64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

fn memory_summary<'py>(py: Python<'py>, vm: &vm::VM) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for map in &vm.memory_maps {
        let mut items: Vec<(i64, i64)> = map.iter_init().collect();
        items.sort_by_key(|(addr, _)| *addr);
        let mut hash = FNV64_OFFSET;
        for (addr, value) in &items {
            hash = fnv64_update(hash, &addr.to_le_bytes());
            hash = fnv64_update(hash, &value.to_le_bytes());
        }
        let summary = PyDict::new_bound(py);
        summary.set_item("entries", items.len())?;
        summary.set_item("min_addr", items.first().map(|(addr, _)| *addr))?;
        summary.set_item("max_addr", items.last().map(|(addr, _)| *addr))?;
        summary.set_item("hash", format!("{:016x}", hash))?;
        summary.set_item("index_bit_width", map.index_bit_width)?;
        summary.set_item("element_bit_width", map.element_bit_width)?;
        out.set_item(map.name.as_str(), summary)?;
    }
    Ok(out)
}

/// Full sparse contents of selected memory maps as `{map_name: {addr: value}}`.
/// Unlike `memory_summary` (which only hashes/bounds), this exposes raw bytes
/// so callers can read specific addresses (e.g. an output buffer) back out.
/// `map_names=None` preserves the legacy behavior of exporting every map;
/// `Some(names)` matches only those exact map names.
fn raw_memory<'py>(
    py: Python<'py>,
    vm: &vm::VM,
    map_names: Option<&[String]>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for map in &vm.memory_maps {
        if map_names.is_some_and(|names| !names.iter().any(|name| name == &map.name)) {
            continue;
        }
        let m = PyDict::new_bound(py);
        for (addr, value) in map.iter_init() {
            m.set_item(addr, value)?;
        }
        out.set_item(map.name.as_str(), m)?;
    }
    Ok(out)
}

fn validate_raw_memory_maps(vm: &vm::VM, map_names: &[String]) -> PyResult<()> {
    for (index, name) in map_names.iter().enumerate() {
        if name.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "raw_memory_maps contains an empty map name",
            ));
        }
        if map_names[..index].contains(name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "raw_memory_maps contains duplicate map name {name:?}"
            )));
        }
        if !vm.memory_maps.iter().any(|map| map.name == *name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "requested raw memory map {name:?} is not present in the compiled program"
            )));
        }
    }
    Ok(())
}

fn scalar_summary<'py>(
    py: Python<'py>,
    program: &opcodes::CompiledProgram,
    vm: &vm::VM,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for (idx, value) in vm.vars.iter().enumerate() {
        match value {
            vm::Value::Scalar(scalar) => {
                if let Some(name) = program.var_names.get(idx) {
                    out.set_item(name, *scalar)?;
                }
            }
            // Out-of-i64 exact integer (Int mode): surface the exact value
            // as an arbitrary-precision Python int.
            vm::Value::Big(big) => {
                if let Some(name) = program.var_names.get(idx) {
                    out.set_item(name, (**big).clone())?;
                }
            }
            vm::Value::Map(_) => {}
        }
    }
    Ok(out)
}

fn attach_raw_log(
    program: &opcodes::CompiledProgram,
    vm: &mut vm::VM,
    raw_log_path: &str,
) -> PyResult<()> {
    let path = std::path::Path::new(raw_log_path);
    let mut writer = raw_log::RawLogWriter::create(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "failed to create raw log at {}: {}",
            path.display(),
            e
        ))
    })?;
    let block_names: Vec<String> = program.blocks.iter().map(|b| b.name.clone()).collect();
    writer
        .write_header(&program.var_names, &block_names)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("raw log header: {}", e)))?;
    debug_log::event(
        "trace",
        "native_raw_log_open",
        &[
            ("path", raw_log_path.to_string()),
            ("vars", program.var_names.len().to_string()),
            ("blocks", block_names.len().to_string()),
        ],
    );
    vm.trace.raw_log = Some(writer);
    Ok(())
}

fn finish_vm_result(
    py: Python<'_>,
    program: &opcodes::CompiledProgram,
    vm: &mut vm::VM,
    status: &ExecutionStatus,
    exec_elapsed: Duration,
    return_memory_summary: bool,
    return_scalar_summary: bool,
    return_raw_memory: bool,
    raw_memory_maps: Option<&[String]>,
    return_block_sequence: bool,
    return_coverage: bool,
    quiet: bool,
) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);

    if return_coverage {
        let blocks_set = PySet::empty_bound(py)?;
        for (idx, explored) in vm.explored_blocks.iter().enumerate() {
            if *explored {
                if let Some(block) = program.blocks.get(idx) {
                    blocks_set.add(block.name.as_str())?;
                }
            }
        }
        result.set_item("explored_blocks", blocks_set)?;

        let edges_set = PySet::empty_bound(py)?;
        for (source_id, target_id) in &vm.explored_edges {
            let source = program.blocks[*source_id as usize].name.as_str();
            let target = program.blocks[*target_id as usize].name.as_str();
            edges_set.add((source, target))?;
        }
        result.set_item("explored_edges", edges_set)?;
    }

    // The per-entry sequence is one PyString per block *entry* — tens of
    // millions on long runs — so it's only materialized when the caller
    // asked for it (the high-level runner uses compact edges instead).
    if return_block_sequence {
        let block_sequence = PyList::empty_bound(py);
        for block_id in &vm.block_trace {
            let block = &program.blocks[*block_id as usize];
            block_sequence.append(block.name.as_str())?;
        }
        result.set_item("block_sequence", block_sequence)?;
    }

    match status {
        ExecutionStatus::Completed => {
            result.set_item("status", "ok")?;
        }
        ExecutionStatus::AssertViolation { pc, block } => {
            result.set_item("status", "assert_violation")?;
            result.set_item("violation_pc", *pc)?;
            result.set_item("violation_block", block)?;
        }
        ExecutionStatus::AssumeViolation {
            pc,
            block,
            reason,
            detail,
        } => {
            result.set_item("status", "assume_violation")?;
            result.set_item("violation_pc", *pc)?;
            result.set_item("violation_block", block)?;
            result.set_item("invalid_input", true)?;
            result.set_item("invalid_reason", *reason)?;
            result.set_item("invalid_detail", detail.as_str())?;
        }
        ExecutionStatus::StepLimit { pc, block } => {
            result.set_item("status", "step_limit")?;
            result.set_item("violation_pc", *pc)?;
            result.set_item("violation_block", block)?;
            result.set_item("message", "native execution step limit reached")?;
        }
    }

    let record_count = if let Some(writer) = vm.trace.raw_log.take() {
        let close_start = std::time::Instant::now();
        let count = writer
            .finish()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("raw log finish: {}", e)))?;
        if !quiet {
            eprintln!(
                "[native] Raw log close: {:.1?}, {} records",
                close_start.elapsed(),
                count
            );
        }
        debug_log::event(
            "trace",
            "native_raw_log_close",
            &[
                ("elapsed_ms", close_start.elapsed().as_millis().to_string()),
                ("records", count.to_string()),
            ],
        );
        count
    } else {
        0
    };
    result.set_item("trace_records", record_count)?;
    if return_memory_summary {
        result.set_item("memory_summary", memory_summary(py, vm)?)?;
    } else {
        result.set_item("memory_summary", PyDict::new_bound(py))?;
    }
    if return_scalar_summary {
        result.set_item("final_scalars", scalar_summary(py, program, vm)?)?;
    }
    if return_raw_memory {
        result.set_item("memory_raw", raw_memory(py, vm, raw_memory_maps)?)?;
    }
    result.set_item("external_consumed", vm.external_buffer_pos)?;
    result.set_item("exec_ns", exec_elapsed.as_nanos() as u64)?;
    result.set_item("exec_ms", exec_elapsed.as_secs_f64() * 1000.0)?;
    if return_coverage {
        result.set_item("blocks_explored", vm.explored_count)?;
    }
    result.set_item("block_sequence_len", vm.block_entries)?;
    result.set_item("no_trace", vm.no_trace)?;
    // Out-of-i64 exact values whose trace records were skipped (Int mode
    // escape — see vm::VM::big_trace_skips).
    result.set_item("trace_big_skips", vm.big_trace_skips)?;
    // Out-of-i64 exact values folded mod 2^64 at the memory interface
    // (Int mode escape — see vm::VM::mem_big_folds).
    result.set_item("mem_big_folds", vm.mem_big_folds)?;
    result.set_item("vars", program.var_names.len())?;
    result.set_item("memory_map_count", vm.memory_maps.len())?;
    debug_log::event(
        "exec",
        "native_result",
        &[
            (
                "status",
                match status {
                    ExecutionStatus::Completed => "ok".to_string(),
                    ExecutionStatus::AssertViolation { .. } => "assert_violation".to_string(),
                    ExecutionStatus::AssumeViolation { .. } => "assume_violation".to_string(),
                    ExecutionStatus::StepLimit { .. } => "step_limit".to_string(),
                },
            ),
            ("trace_records", record_count.to_string()),
            ("memory_maps", vm.memory_maps.len().to_string()),
            ("external_consumed", vm.external_buffer_pos.to_string()),
        ],
    );

    Ok(result.into())
}

/// Execute a pre-lowered program with pre-initialized state from Python.
///
/// All trace output is streamed to `raw_log_path` in the
/// `.trace.raw.zst` format (see `raw_log.rs`). There is no in-memory
/// compact trace — the driver loads the raw log straight into Redis.
///
/// Args:
///   compiled: result of `lower(program)`
///   var_store: dict[str, int] — scalar variable values after Python concretization
///   memory_maps: dict[str, dict[int, int]] — memory map contents after Python concretization
///   mem_map_info: list of (name, index_bw, element_bw) — metadata for each memory map
///   raw_log_path: path to write the `.trace.raw.zst` streaming log
///   extra_data: optional bytes for read.cross_product
///   log_read: whether to log read ops in trace
///   no_trace: disable all tracing (raw_log_path must still be valid but won't be written to)
///   return_memory_summary: whether to hash/summarize final memory maps
///   quiet: suppress progress output on stderr
///
/// Returns dict with 'explored_blocks' and 'trace_records' (count).
#[pyfunction]
#[pyo3(signature = (compiled, var_store, memory_maps, mem_map_info, raw_log_path, extra_data=None, log_read=true, no_trace=false, havoc_sequences=None, return_memory_summary=true, quiet=true, max_steps=0, return_scalar_summary=false, return_raw_memory=false, return_block_sequence=true))]
fn execute(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    var_store: &Bound<'_, PyDict>,
    memory_maps: &Bound<'_, PyDict>,
    mem_map_info: &Bound<'_, PyList>,
    raw_log_path: String,
    extra_data: Option<Vec<u8>>,
    log_read: bool,
    no_trace: bool,
    havoc_sequences: Option<&Bound<'_, PyDict>>,
    return_memory_summary: bool,
    quiet: bool,
    max_steps: usize,
    return_scalar_summary: bool,
    return_raw_memory: bool,
    return_block_sequence: bool,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let program = &wrapper.inner;

    let mut vm = if no_trace {
        vm::VM::new_no_trace(program)
    } else {
        vm::VM::new(program)
    };
    vm.log_read = log_read;
    vm.no_trace = no_trace;
    vm.record_block_trace = return_block_sequence;
    if let Some(data) = extra_data {
        vm.external_buffer = data;
    }

    if !no_trace {
        attach_raw_log(program, &mut vm, &raw_log_path)?;
    }

    // Initialize memory maps from Python-provided metadata
    for item in mem_map_info.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let name: String = tuple.get_item(0)?.extract()?;
        let index_bw: u8 = tuple.get_item(1)?.extract()?;
        let element_bw: u8 = tuple.get_item(2)?.extract()?;

        if let Some(vid) = program.lookup_var(&name) {
            vm.init_memory_map(vid, name.clone(), index_bw, element_bw);
        }
    }

    // Load scalar variable values from Python
    for (key, val) in var_store.iter() {
        let name: String = key.extract()?;
        let value: i64 = val.extract()?;
        if let Some(vid) = program.lookup_var(&name) {
            vm.set_scalar(vid, value, true);
        }
    }

    // Load memory map contents from Python
    for (key, val) in memory_maps.iter() {
        let name: String = key.extract()?;
        let contents: &Bound<'_, PyDict> = val.downcast()?;
        if let Some(vid) = program.lookup_var(&name) {
            if let Some(map_idx) = vm.var_to_map.get(&vid).copied() {
                for (addr, value) in contents.iter() {
                    let a: i64 = addr.extract()?;
                    let v: i64 = value.extract()?;
                    vm.memory_maps[map_idx].set(a, v);
                }
            }
        }
    }

    if let Some(seqs) = havoc_sequences {
        for (key, val) in seqs.iter() {
            let name: String = key.extract()?;
            let values: Vec<i64> = val.extract()?;
            if let Some(vid) = program.lookup_var(&name) {
                vm.set_havoc_sequence(vid, values);
            }
        }
    }

    // Execute
    let exec_start = std::time::Instant::now();
    let status = vm.execute_with_limit(program, max_steps);
    let exec_elapsed = exec_start.elapsed();
    if !quiet {
        eprintln!(
            "[native] Execution: {:.1?}, {} blocks, {} trace entries",
            exec_elapsed, vm.explored_count, vm.trace.total
        );
    }
    debug_log::event(
        "exec",
        "native_execution_end",
        &[
            ("elapsed_ms", exec_elapsed.as_millis().to_string()),
            ("blocks", vm.explored_count.to_string()),
            ("trace_entries", vm.trace.total.to_string()),
        ],
    );

    finish_vm_result(
        py,
        program,
        &mut vm,
        &status,
        exec_elapsed,
        return_memory_summary,
        return_scalar_summary,
        return_raw_memory,
        None,
        return_block_sequence,
        true,
        quiet,
    )
}

/// Execute a pre-lowered program by concretizing ProgramInputs directly in Rust.
///
/// This is the high-throughput native coverage entry point. It keeps one-time
/// PyO3 AST lowering and removes per-input Environment construction and dict
/// handoff.
#[pyfunction]
#[pyo3(signature = (compiled, native_meta, program_inputs, raw_log_path, extra_data=None, log_read=true, no_trace=false, return_memory_summary=true, quiet=true, max_steps=0, return_scalar_summary=false, return_raw_memory=false, return_block_sequence=true, raw_memory_maps=None, return_coverage=true))]
fn execute_inputs(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    native_meta: &Bound<'_, PyDict>,
    program_inputs: &Bound<'_, PyAny>,
    raw_log_path: String,
    extra_data: Option<Vec<u8>>,
    log_read: bool,
    no_trace: bool,
    return_memory_summary: bool,
    quiet: bool,
    max_steps: usize,
    return_scalar_summary: bool,
    return_raw_memory: bool,
    return_block_sequence: bool,
    raw_memory_maps: Option<Vec<String>>,
    return_coverage: bool,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let program = &wrapper.inner;

    let mut vm = if no_trace {
        vm::VM::new_no_trace(program)
    } else {
        vm::VM::new(program)
    };
    vm.log_read = log_read;
    vm.no_trace = no_trace;
    vm.record_block_trace = return_block_sequence;
    vm.record_coverage = return_coverage;

    if !no_trace {
        attach_raw_log(program, &mut vm, &raw_log_path)?;
    }

    let state_start = std::time::Instant::now();
    input_state::initialize_vm_from_inputs(
        program,
        &mut vm,
        native_meta,
        program_inputs,
        extra_data,
    )?;
    let state_elapsed = state_start.elapsed();

    if let Some(map_names) = raw_memory_maps.as_deref() {
        if !return_raw_memory {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "raw_memory_maps requires return_raw_memory=True",
            ));
        }
        validate_raw_memory_maps(&vm, map_names)?;
    }

    let exec_start = std::time::Instant::now();
    let status = vm.execute_with_limit(program, max_steps);
    let exec_elapsed = exec_start.elapsed();
    if !quiet {
        eprintln!(
            "[native] State: {:.1?}, execution: {:.1?}, {} blocks, {} trace entries",
            state_elapsed, exec_elapsed, vm.explored_count, vm.trace.total
        );
    }
    debug_log::event(
        "exec",
        "native_execution_end",
        &[
            ("state_ms", state_elapsed.as_millis().to_string()),
            ("elapsed_ms", exec_elapsed.as_millis().to_string()),
            ("blocks", vm.explored_count.to_string()),
            ("trace_entries", vm.trace.total.to_string()),
        ],
    );

    let result = finish_vm_result(
        py,
        program,
        &mut vm,
        &status,
        exec_elapsed,
        return_memory_summary,
        return_scalar_summary,
        return_raw_memory,
        raw_memory_maps.as_deref(),
        return_block_sequence,
        return_coverage,
        quiet,
    )?;
    let result_dict = result.bind(py).downcast::<PyDict>()?;
    result_dict.set_item("state_ns", state_elapsed.as_nanos() as u64)?;
    result_dict.set_item("state_ms", state_elapsed.as_secs_f64() * 1000.0)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (compiled, native_meta, program_inputs, extra_data=None, havoc_bound=8))]
fn prepare_symbolic_inputs(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    native_meta: &Bound<'_, PyDict>,
    program_inputs: &Bound<'_, PyAny>,
    extra_data: Option<Vec<u8>>,
    havoc_bound: usize,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    input_state::prepare_symbolic_state(
        py,
        &wrapper.inner,
        native_meta,
        program_inputs,
        extra_data,
        havoc_bound,
    )
}

/// Debug introspection: pretty-print one lowered block (or all blocks when
/// `block_name` is None) with per-statement PCs.
#[pyfunction]
#[pyo3(signature = (compiled, block_name=None))]
fn dump_block(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    block_name: Option<&str>,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let mut out = String::new();
    for block in &wrapper.inner.blocks {
        if let Some(name) = block_name {
            if block.name != name {
                continue;
            }
        }
        out.push_str(&format!(
            "block {} (id={}, start_pc={}):\n",
            block.name, block.id, block.start_pc
        ));
        let mut pc = block.start_pc;
        for stmt in &block.body {
            if stmt.is_internal_maintenance() {
                out.push_str(&format!("  internal {:?}\n", stmt));
            } else {
                out.push_str(&format!("  pc={} {:?}\n", pc, stmt));
                pc += 1;
            }
        }
        out.push_str(&format!("  term {:?}\n", block.terminator));
        if let Some(cond) = &block.assume_cond {
            out.push_str(&format!("  assume_cond {:?}\n", cond));
        }
    }
    Ok(out.into_py(py))
}

#[pyfunction]
fn get_var_names(py: Python<'_>, compiled: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let names: Vec<&str> = wrapper.inner.var_names.iter().collect();
    Ok(names.into_py(py))
}

/// Debug introspection: return one lowered variable name without materializing
/// the complete name table as a Python list.
#[pyfunction]
fn get_var_name(compiled: &Bound<'_, PyAny>, var_id: u32) -> PyResult<String> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    wrapper
        .inner
        .var_names
        .get(var_id as usize)
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "variable id {var_id} is out of range ({} names)",
                wrapper.inner.var_names.len()
            ))
        })
}

/// Debug introspection: return one lowered block name without materializing
/// the complete block-name table in Python.
#[pyfunction]
fn get_block_name(compiled: &Bound<'_, PyAny>, block_id: u32) -> PyResult<String> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    wrapper
        .inner
        .blocks
        .get(block_id as usize)
        .map(|block| block.name.clone())
        .ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "block id {block_id} is out of range ({} blocks)",
                wrapper.inner.blocks.len()
            ))
        })
}

#[pyfunction]
#[pyo3(signature = (compiled, var_store, memory_maps, mem_map_info, symbols, extra_data=None, covered_blocks=None, loop_bound=8, max_path_depth=512, max_solver_queries=10000, solver_timeout_ms=100, branch_distance_policy="auto"))]
fn concolic_suggest(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    var_store: &Bound<'_, PyDict>,
    memory_maps: &Bound<'_, PyDict>,
    mem_map_info: &Bound<'_, PyList>,
    symbols: &Bound<'_, PyList>,
    extra_data: Option<Vec<u8>>,
    covered_blocks: Option<&Bound<'_, PyAny>>,
    loop_bound: usize,
    max_path_depth: usize,
    max_solver_queries: usize,
    solver_timeout_ms: u64,
    branch_distance_policy: &str,
) -> PyResult<PyObject> {
    concolic::suggest(
        py,
        compiled,
        var_store,
        memory_maps,
        mem_map_info,
        symbols,
        extra_data,
        covered_blocks,
        loop_bound,
        max_path_depth,
        max_solver_queries,
        solver_timeout_ms,
        branch_distance_policy,
    )
}

/// Stream a `.trace.raw.zst` file directly into Redis.
///
/// Replaces the slow Python loop in
/// `AbductionState.init_positive_examples_raw_log`.  See
/// `raw_log_reader.rs` for the full format and schema docs.
///
/// Returns the total number of records consumed.
#[pyfunction]
#[pyo3(signature = (path, redis_url, iter_id_offset=0))]
fn load_raw_log_to_redis(
    py: Python<'_>,
    path: String,
    redis_url: String,
    iter_id_offset: u32,
) -> PyResult<u64> {
    // Release the GIL for the duration of the load — the Rust pipeline
    // spawns its own threads and does not touch any Python state, so
    // holding the GIL for the (multi-minute) duration would needlessly
    // starve every other Python thread in the driver process.
    py.allow_threads(|| {
        raw_log_reader::load_raw_log_to_redis(
            std::path::Path::new(&path),
            &redis_url,
            iter_id_offset,
        )
    })
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Build the production SQLite trace-evidence index from raw trace logs.
///
/// This is the native equivalent of
/// `src.state.trace_evidence.build_trace_index_from_raw_logs`.
#[pyfunction]
#[pyo3(signature = (raw_paths, sqlite_path, bench=None))]
fn build_trace_index_sqlite(
    py: Python<'_>,
    raw_paths: Vec<String>,
    sqlite_path: String,
    bench: Option<String>,
) -> PyResult<PyObject> {
    let paths: Vec<std::path::PathBuf> = raw_paths.into_iter().map(Into::into).collect();
    let sqlite_path_buf = std::path::PathBuf::from(sqlite_path);
    let result = py
        .allow_threads(|| {
            trace_index::build_trace_index_sqlite(&paths, &sqlite_path_buf, bench.as_deref())
        })
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let out = PyDict::new_bound(py);
    out.set_item("path", result.path)?;
    out.set_item("rows", result.rows)?;
    out.set_item("records", result.records)?;
    out.set_item("raw_files", result.raw_files)?;
    out.set_item("skipped_shadow", result.skipped_shadow)?;
    out.set_item("contexts", result.contexts)?;
    out.set_item("source", "raw_log")?;
    out.set_item("builder", "native_sqlite_sorted_runs")?;
    Ok(out.into_py(py))
}

#[pymodule]
fn swoosh_interp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lower, m)?)?;
    m.add_function(wrap_pyfunction!(inline_lower, m)?)?;
    m.add_function(wrap_pyfunction!(inline_lower_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(load_compiled, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(execute_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_symbolic_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(concolic_suggest, m)?)?;
    m.add_function(wrap_pyfunction!(explore, m)?)?;
    m.add_function(wrap_pyfunction!(get_var_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_var_name, m)?)?;
    m.add_function(wrap_pyfunction!(get_block_name, m)?)?;
    m.add_function(wrap_pyfunction!(dump_block, m)?)?;
    m.add_function(wrap_pyfunction!(load_raw_log_to_redis, m)?)?;
    m.add_function(wrap_pyfunction!(build_trace_index_sqlite, m)?)?;
    m.add_class::<CompiledProgramWrapper>()?;
    Ok(())
}
