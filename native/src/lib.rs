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
#[pyo3(signature = (program, loop_header_live=None, loop_metadata=None))]
fn lower(
    py: Python<'_>,
    program: &Bound<'_, PyAny>,
    loop_header_live: Option<&Bound<'_, PyDict>>,
    loop_metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let compiled = lowering::lower_program_full(py, program, loop_header_live, loop_metadata)?;
    let wrapper = CompiledProgramWrapper { inner: compiled };
    Ok(Py::new(py, wrapper)?.into_py(py))
}

#[pyclass]
pub(crate) struct CompiledProgramWrapper {
    pub(crate) inner: opcodes::CompiledProgram,
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
        let mut items: Vec<(i64, i64)> = map
            .memory
            .iter()
            .map(|(addr, value)| (*addr, *value))
            .collect();
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

fn scalar_summary<'py>(
    py: Python<'py>,
    program: &opcodes::CompiledProgram,
    vm: &vm::VM,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for (idx, value) in vm.vars.iter().enumerate() {
        if let vm::Value::Scalar(scalar) = value {
            if let Some(name) = program.var_names.get(idx) {
                out.set_item(name.as_str(), *scalar)?;
            }
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
    quiet: bool,
) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);

    let blocks_set = PySet::empty_bound(py)?;
    for block_id in &vm.explored_blocks {
        let block = &program.blocks[*block_id as usize];
        blocks_set.add(block.name.as_str())?;
    }
    result.set_item("explored_blocks", blocks_set)?;

    let block_sequence = PyList::empty_bound(py);
    for block_id in &vm.block_trace {
        let block = &program.blocks[*block_id as usize];
        block_sequence.append(block.name.as_str())?;
    }
    result.set_item("block_sequence", block_sequence)?;

    match status {
        ExecutionStatus::Completed => {
            result.set_item("status", "ok")?;
        }
        ExecutionStatus::AssertViolation { pc, block } => {
            result.set_item("status", "assert_violation")?;
            result.set_item("violation_pc", *pc)?;
            result.set_item("violation_block", block)?;
        }
        ExecutionStatus::AssumeViolation { pc, block, reason } => {
            result.set_item("status", "assume_violation")?;
            result.set_item("violation_pc", *pc)?;
            result.set_item("violation_block", block)?;
            result.set_item("invalid_input", true)?;
            result.set_item("invalid_reason", *reason)?;
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
    result.set_item("external_consumed", vm.external_buffer_pos)?;
    result.set_item("exec_ns", exec_elapsed.as_nanos() as u64)?;
    result.set_item("exec_ms", exec_elapsed.as_secs_f64() * 1000.0)?;
    result.set_item("blocks_explored", vm.explored_blocks.len())?;
    result.set_item("block_sequence_len", vm.block_trace.len())?;
    result.set_item("no_trace", vm.no_trace)?;
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
#[pyo3(signature = (compiled, var_store, memory_maps, mem_map_info, raw_log_path, extra_data=None, log_read=true, no_trace=false, havoc_sequences=None, return_memory_summary=true, quiet=true, max_steps=0, return_scalar_summary=false))]
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

        if let Some(&vid) = program.name_to_var.get(&name) {
            vm.init_memory_map(vid, name.clone(), index_bw, element_bw);
        }
    }

    // Load scalar variable values from Python
    for (key, val) in var_store.iter() {
        let name: String = key.extract()?;
        let value: i64 = val.extract()?;
        if let Some(&vid) = program.name_to_var.get(&name) {
            vm.set_scalar(vid, value, true);
        }
    }

    // Load memory map contents from Python
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

    if let Some(seqs) = havoc_sequences {
        for (key, val) in seqs.iter() {
            let name: String = key.extract()?;
            let values: Vec<i64> = val.extract()?;
            if let Some(&vid) = program.name_to_var.get(&name) {
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
            exec_elapsed,
            vm.explored_blocks.len(),
            vm.trace.total
        );
    }
    debug_log::event(
        "exec",
        "native_execution_end",
        &[
            ("elapsed_ms", exec_elapsed.as_millis().to_string()),
            ("blocks", vm.explored_blocks.len().to_string()),
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
        quiet,
    )
}

/// Execute a pre-lowered program by concretizing ProgramInputs directly in Rust.
///
/// This is the high-throughput native coverage entry point. It keeps one-time
/// PyO3 AST lowering and removes per-input Environment construction and dict
/// handoff.
#[pyfunction]
#[pyo3(signature = (compiled, native_meta, program_inputs, raw_log_path, extra_data=None, log_read=true, no_trace=false, return_memory_summary=true, quiet=true, max_steps=0, return_scalar_summary=false))]
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

    let exec_start = std::time::Instant::now();
    let status = vm.execute_with_limit(program, max_steps);
    let exec_elapsed = exec_start.elapsed();
    if !quiet {
        eprintln!(
            "[native] State: {:.1?}, execution: {:.1?}, {} blocks, {} trace entries",
            state_elapsed,
            exec_elapsed,
            vm.explored_blocks.len(),
            vm.trace.total
        );
    }
    debug_log::event(
        "exec",
        "native_execution_end",
        &[
            ("state_ms", state_elapsed.as_millis().to_string()),
            ("elapsed_ms", exec_elapsed.as_millis().to_string()),
            ("blocks", vm.explored_blocks.len().to_string()),
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

#[pyfunction]
fn get_var_names(py: Python<'_>, compiled: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let names: Vec<&str> = wrapper.inner.var_names.iter().map(|s| s.as_str()).collect();
    Ok(names.into_py(py))
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

#[pymodule]
fn swoosh_interp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lower, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(execute_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_symbolic_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(concolic_suggest, m)?)?;
    m.add_function(wrap_pyfunction!(explore, m)?)?;
    m.add_function(wrap_pyfunction!(get_var_names, m)?)?;
    m.add_function(wrap_pyfunction!(load_raw_log_to_redis, m)?)?;
    m.add_class::<CompiledProgramWrapper>()?;
    Ok(())
}
