use crate::builtins;
use crate::debug_log;
use crate::opcodes::*;
use crate::vm::{EvalResult, Value, VM};
use crate::CompiledProgramWrapper;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PySet, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

const MASK_64: i64 = -1i64;
const MAX_PROVENANCE_TERMS: usize = 32;
const MAX_BRANCH_DISTANCE_CANDIDATES: usize = 4096;

#[derive(Clone, Debug)]
struct InputSymbol {
    name: String,
    bits: u32,
    value: u64,
    var_id: Option<VarId>,
    map_key: Option<(String, i64)>,
    extra_index: Option<usize>,
    havoc_key: Option<(VarId, usize)>,
}

#[derive(Clone, Debug)]
enum Sym {
    Bv {
        text: String,
        bits: u32,
        provenance: Provenance,
    },
    Bool {
        text: String,
        provenance: Provenance,
    },
}

#[derive(Clone, Debug, Default)]
struct Provenance {
    terms: Vec<ProvTerm>,
    overflow: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProvTerm {
    name: String,
    low: u32,
    bits: u32,
}

#[derive(Clone, Debug)]
struct CEval {
    value: EvalResult,
    sym: Option<Sym>,
}

#[derive(Default)]
struct Stats {
    branches_seen: usize,
    objectives: usize,
    solver_queries: usize,
    solver_sat: usize,
    solver_unsat: usize,
    solver_unknown: usize,
    solver_errors: usize,
    solver_cache_hits: usize,
    solver_time_ms: u64,
    unsupported: usize,
    value_profile_hints: usize,
    value_profile_candidates: usize,
    value_profile_deferred: usize,
    branch_distance_objectives: usize,
    branch_distance_candidates: usize,
    branch_distance_overflowed_objectives: usize,
    branch_distance_best_delta: u64,
    frontier_ranked_objectives: usize,
    depth_bounded: bool,
    loop_bounded: bool,
    solver_query_bounded: bool,
}

impl Stats {
    fn merge(&mut self, other: &Stats) {
        self.branches_seen += other.branches_seen;
        self.objectives += other.objectives;
        self.solver_queries += other.solver_queries;
        self.solver_sat += other.solver_sat;
        self.solver_unsat += other.solver_unsat;
        self.solver_unknown += other.solver_unknown;
        self.solver_errors += other.solver_errors;
        self.solver_cache_hits += other.solver_cache_hits;
        self.solver_time_ms += other.solver_time_ms;
        self.unsupported += other.unsupported;
        self.value_profile_hints += other.value_profile_hints;
        self.value_profile_candidates += other.value_profile_candidates;
        self.value_profile_deferred += other.value_profile_deferred;
        self.branch_distance_objectives += other.branch_distance_objectives;
        self.branch_distance_candidates += other.branch_distance_candidates;
        self.branch_distance_overflowed_objectives += other.branch_distance_overflowed_objectives;
        self.branch_distance_best_delta = self
            .branch_distance_best_delta
            .max(other.branch_distance_best_delta);
        self.frontier_ranked_objectives += other.frontier_ranked_objectives;
        self.depth_bounded |= other.depth_bounded;
        self.loop_bounded |= other.loop_bounded;
        self.solver_query_bounded |= other.solver_query_bounded;
    }
}

struct Candidate {
    updates: Vec<(String, u64)>,
    source_block: String,
    target_block: String,
    branch_index: usize,
    kind: CandidateKind,
    priority: i64,
    distance_to_uncovered: i64,
    reachable_uncovered: usize,
}

enum BranchObjective {
    Sat(String),
    Unsat,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateKind {
    BranchTarget,
    RequiresRepair,
    AssertRepair,
    DistanceGuided,
    ValueProfile,
}

impl CandidateKind {
    fn as_str(self) -> &'static str {
        match self {
            CandidateKind::BranchTarget => "branch_target",
            CandidateKind::RequiresRepair => "requires_repair",
            CandidateKind::AssertRepair => "assert_repair",
            CandidateKind::DistanceGuided => "distance_guided",
            CandidateKind::ValueProfile => "value_profile",
        }
    }

    fn base_priority(self) -> i64 {
        match self {
            CandidateKind::BranchTarget => 4_000_000,
            CandidateKind::RequiresRepair => 3_000_000,
            CandidateKind::AssertRepair => 2_000_000,
            CandidateKind::DistanceGuided => 1_500_000,
            CandidateKind::ValueProfile => 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BranchDistancePolicy {
    Auto,
    On,
    Off,
}

impl BranchDistancePolicy {
    fn from_name(name: &str) -> Self {
        match name {
            "on" => BranchDistancePolicy::On,
            "off" => BranchDistancePolicy::Off,
            _ => BranchDistancePolicy::Auto,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            BranchDistancePolicy::Auto => "auto",
            BranchDistancePolicy::On => "on",
            BranchDistancePolicy::Off => "off",
        }
    }

    fn enabled(self) -> bool {
        !matches!(self, BranchDistancePolicy::Off)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PathPriority {
    Fifo,
    CoverageDirected,
}

impl PathPriority {
    fn from_name(name: &str) -> Self {
        match name {
            "fifo" => PathPriority::Fifo,
            _ => PathPriority::CoverageDirected,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            PathPriority::Fifo => "fifo",
            PathPriority::CoverageDirected => "coverage-directed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValueProfilePolicy {
    Enqueue,
    Late,
    Off,
}

impl ValueProfilePolicy {
    fn from_name(name: &str) -> Self {
        match name {
            "enqueue" => ValueProfilePolicy::Enqueue,
            "off" => ValueProfilePolicy::Off,
            _ => ValueProfilePolicy::Late,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ValueProfilePolicy::Enqueue => "enqueue",
            ValueProfilePolicy::Late => "late",
            ValueProfilePolicy::Off => "off",
        }
    }

    fn enqueue_value_profiles(self) -> bool {
        matches!(self, ValueProfilePolicy::Enqueue)
    }
}

#[derive(Clone)]
struct QueuedState {
    updates: Vec<(String, u64)>,
    priority: i64,
    seq: usize,
}

struct Engine<'a> {
    program: &'a CompiledProgram,
    vm: VM,
    sym_vars: Vec<Option<Sym>>,
    sym_maps: Vec<FxHashMap<i64, Sym>>,
    sym_external: Vec<Option<Sym>>,
    sym_havoc: FxHashMap<(VarId, usize), Sym>,
    path_constraints: Vec<String>,
    symbols: Vec<InputSymbol>,
    covered_blocks: FxHashSet<String>,
    candidates: Vec<Candidate>,
    candidate_sigs: FxHashMap<String, usize>,
    solver_cache: &'a mut FxHashMap<String, SolveResult>,
    solver: Option<SolverProcess>,
    stats: Stats,
    loop_counts: FxHashMap<BlockId, usize>,
    loop_bound: usize,
    max_path_depth: usize,
    max_solver_queries: usize,
    solver_timeout_ms: u64,
    target_all_branches: bool,
    path_priority: PathPriority,
    value_profile_policy: ValueProfilePolicy,
    branch_distance_policy: BranchDistancePolicy,
}

pub fn suggest(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    var_store: &Bound<'_, PyDict>,
    memory_maps: &Bound<'_, PyDict>,
    mem_map_info: &Bound<'_, PyList>,
    symbols_py: &Bound<'_, PyList>,
    extra_data: Option<Vec<u8>>,
    covered_blocks: Option<&Bound<'_, PyAny>>,
    loop_bound: usize,
    max_path_depth: usize,
    max_solver_queries: usize,
    solver_timeout_ms: u64,
    branch_distance_policy: &str,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let program = &wrapper.inner;

    let symbols = parse_symbols(symbols_py, program)?;
    let covered = parse_covered(covered_blocks)?;
    let vm = build_vm(program, var_store, memory_maps, mem_map_info, extra_data)?;
    let branch_distance_policy = BranchDistancePolicy::from_name(branch_distance_policy);
    let started = Instant::now();
    let mut solver_cache = FxHashMap::default();

    let mut engine = Engine::new(
        program,
        vm,
        symbols,
        covered,
        &mut solver_cache,
        loop_bound,
        max_path_depth,
        max_solver_queries,
        solver_timeout_ms,
    );
    engine.install_symbols();
    engine.branch_distance_policy = branch_distance_policy;
    engine.execute();
    let elapsed = started.elapsed();
    engine.candidates.sort_by(compare_candidates);
    let candidate_count = engine.candidates.len();

    let out = PyDict::new_bound(py);
    let cand_list = PyList::empty_bound(py);
    for cand in engine.candidates {
        let d = PyDict::new_bound(py);
        let updates = PyDict::new_bound(py);
        for (name, value) in cand.updates {
            updates.set_item(name, value)?;
        }
        d.set_item("updates", updates)?;
        d.set_item("source_block", cand.source_block)?;
        d.set_item("target_block", cand.target_block)?;
        d.set_item("branch_index", cand.branch_index)?;
        d.set_item("kind", cand.kind.as_str())?;
        d.set_item("priority", cand.priority)?;
        d.set_item("distance_to_uncovered", cand.distance_to_uncovered)?;
        d.set_item("reachable_uncovered", cand.reachable_uncovered)?;
        cand_list.append(d)?;
    }

    let stats = PyDict::new_bound(py);
    stats.set_item("branches_seen", engine.stats.branches_seen)?;
    stats.set_item("objectives", engine.stats.objectives)?;
    stats.set_item("solver_queries", engine.stats.solver_queries)?;
    stats.set_item("solver_sat", engine.stats.solver_sat)?;
    stats.set_item("solver_unsat", engine.stats.solver_unsat)?;
    stats.set_item("solver_unknown", engine.stats.solver_unknown)?;
    stats.set_item("solver_errors", engine.stats.solver_errors)?;
    stats.set_item("solver_cache_hits", engine.stats.solver_cache_hits)?;
    stats.set_item("solver_cache_entries", engine.solver_cache.len())?;
    stats.set_item("solver_time_ms", engine.stats.solver_time_ms)?;
    stats.set_item("elapsed_ms", elapsed.as_secs_f64() * 1000.0)?;
    stats.set_item(
        "solver_queries_per_sec",
        per_sec(engine.stats.solver_queries, elapsed),
    )?;
    stats.set_item("candidates_per_sec", per_sec(candidate_count, elapsed))?;
    stats.set_item(
        "solver_cache_hit_rate",
        ratio(
            engine.stats.solver_cache_hits,
            engine.stats.solver_cache_hits + engine.stats.solver_queries,
        ),
    )?;
    stats.set_item("unsupported", engine.stats.unsupported)?;
    stats.set_item("value_profile_hints", engine.stats.value_profile_hints)?;
    stats.set_item(
        "value_profile_candidates",
        engine.stats.value_profile_candidates,
    )?;
    stats.set_item(
        "value_profile_deferred",
        engine.stats.value_profile_deferred,
    )?;
    stats.set_item(
        "branch_distance_objectives",
        engine.stats.branch_distance_objectives,
    )?;
    stats.set_item(
        "branch_distance_candidates",
        engine.stats.branch_distance_candidates,
    )?;
    stats.set_item(
        "branch_distance_overflowed_objectives",
        engine.stats.branch_distance_overflowed_objectives,
    )?;
    stats.set_item(
        "branch_distance_best_delta",
        engine.stats.branch_distance_best_delta,
    )?;
    stats.set_item("branch_distance_policy", branch_distance_policy.as_str())?;
    stats.set_item(
        "frontier_ranked_objectives",
        engine.stats.frontier_ranked_objectives,
    )?;
    stats.set_item("depth_bounded", engine.stats.depth_bounded)?;
    stats.set_item("loop_bounded", engine.stats.loop_bounded)?;
    stats.set_item("solver_query_bounded", engine.stats.solver_query_bounded)?;
    stats.set_item("solver_backend", "z3-smtlib-persistent")?;
    stats.set_item("complete_within_bounds", false)?;

    out.set_item("candidates", cand_list)?;
    out.set_item("stats", stats)?;
    Ok(out.into())
}

#[pyfunction(name = "symbolic_explore")]
#[pyo3(signature = (compiled, var_store, memory_maps, mem_map_info, symbols_py, extra_data=None, covered_blocks=None, loop_bound=8, max_path_depth=512, max_solver_queries=10000, solver_timeout_ms=100, max_states=256, path_priority="coverage-directed", value_profile_policy="late", branch_distance_policy="auto"))]
pub fn explore(
    py: Python<'_>,
    compiled: &Bound<'_, PyAny>,
    var_store: &Bound<'_, PyDict>,
    memory_maps: &Bound<'_, PyDict>,
    mem_map_info: &Bound<'_, PyList>,
    symbols_py: &Bound<'_, PyList>,
    extra_data: Option<Vec<u8>>,
    covered_blocks: Option<&Bound<'_, PyAny>>,
    loop_bound: usize,
    max_path_depth: usize,
    max_solver_queries: usize,
    solver_timeout_ms: u64,
    max_states: usize,
    path_priority: &str,
    value_profile_policy: &str,
    branch_distance_policy: &str,
) -> PyResult<PyObject> {
    let wrapper: PyRef<'_, CompiledProgramWrapper> = compiled.extract()?;
    let program = &wrapper.inner;

    let symbols = parse_symbols(symbols_py, program)?;
    let covered = parse_covered(covered_blocks)?;
    let base_vm = build_vm(program, var_store, memory_maps, mem_map_info, extra_data)?;
    let path_priority = PathPriority::from_name(path_priority);
    let value_profile_policy = ValueProfilePolicy::from_name(value_profile_policy);
    let branch_distance_policy = BranchDistancePolicy::from_name(branch_distance_policy);

    let mut queue = Vec::new();
    let mut seen = FxHashSet::default();
    queue.push(QueuedState {
        updates: Vec::new(),
        priority: i64::MAX,
        seq: 0,
    });
    let mut next_seq = 1usize;
    seen.insert(String::new());

    let mut stats_total = Stats::default();
    let mut candidates_out = Vec::new();
    let mut states_explored = 0usize;
    let mut states_bounded = 0usize;
    let mut feasible_paths = 0usize;
    let mut unknown_paths = 0usize;
    let started = Instant::now();
    let mut solver_cache = FxHashMap::default();

    while let Some(state) = pop_queued_state(&mut queue, path_priority, states_explored) {
        if states_explored >= max_states {
            states_bounded = queue.len().saturating_add(1);
            unknown_paths += states_bounded;
            break;
        }
        if stats_total.solver_queries >= max_solver_queries {
            stats_total.solver_query_bounded = true;
            unknown_paths += queue.len().saturating_add(1);
            break;
        }

        let mut vm = base_vm.clone();
        apply_updates_to_vm(&mut vm, &symbols, &state.updates);
        let mut state_symbols = symbols.clone();
        apply_updates_to_symbols(&mut state_symbols, &state.updates);
        let remaining_queries = max_solver_queries.saturating_sub(stats_total.solver_queries);
        let mut engine = Engine::new(
            program,
            vm,
            state_symbols,
            covered.clone(),
            &mut solver_cache,
            loop_bound,
            max_path_depth,
            remaining_queries,
            solver_timeout_ms,
        );
        engine.target_all_branches = true;
        engine.path_priority = path_priority;
        engine.value_profile_policy = value_profile_policy;
        engine.branch_distance_policy = branch_distance_policy;
        engine.install_symbols();
        engine.execute();
        states_explored += 1;
        feasible_paths += 1;
        if engine.stats.depth_bounded
            || engine.stats.loop_bounded
            || engine.stats.solver_query_bounded
            || engine.stats.unsupported > 0
            || engine.stats.solver_unknown > 0
            || engine.stats.solver_errors > 0
        {
            unknown_paths += 1;
        }

        for cand in &engine.candidates {
            let sig = update_signature(&cand.updates);
            if seen.insert(sig) {
                let enqueue_candidate = match cand.kind {
                    CandidateKind::ValueProfile => value_profile_policy.enqueue_value_profiles(),
                    CandidateKind::DistanceGuided => false,
                    _ => true,
                };
                if enqueue_candidate {
                    queue.push(QueuedState {
                        updates: cand.updates.clone(),
                        priority: cand.priority,
                        seq: next_seq,
                    });
                    next_seq += 1;
                } else if cand.kind == CandidateKind::ValueProfile {
                    stats_total.value_profile_deferred += 1;
                }
                candidates_out.push(Candidate {
                    updates: cand.updates.clone(),
                    source_block: cand.source_block.clone(),
                    target_block: cand.target_block.clone(),
                    branch_index: cand.branch_index,
                    kind: cand.kind,
                    priority: cand.priority,
                    distance_to_uncovered: cand.distance_to_uncovered,
                    reachable_uncovered: cand.reachable_uncovered,
                });
            }
        }
        stats_total.merge(&engine.stats);
    }

    let out = PyDict::new_bound(py);
    let elapsed = started.elapsed();
    candidates_out.sort_by(compare_candidates);
    let candidate_count = candidates_out.len();
    let cand_list = PyList::empty_bound(py);
    for cand in candidates_out {
        let d = PyDict::new_bound(py);
        let updates = PyDict::new_bound(py);
        for (name, value) in cand.updates {
            updates.set_item(name, value)?;
        }
        d.set_item("updates", updates)?;
        d.set_item("source_block", cand.source_block)?;
        d.set_item("target_block", cand.target_block)?;
        d.set_item("branch_index", cand.branch_index)?;
        d.set_item("kind", cand.kind.as_str())?;
        d.set_item("priority", cand.priority)?;
        d.set_item("distance_to_uncovered", cand.distance_to_uncovered)?;
        d.set_item("reachable_uncovered", cand.reachable_uncovered)?;
        cand_list.append(d)?;
    }

    let stats = PyDict::new_bound(py);
    stats.set_item("branches_seen", stats_total.branches_seen)?;
    stats.set_item("objectives", stats_total.objectives)?;
    stats.set_item("solver_queries", stats_total.solver_queries)?;
    stats.set_item("solver_sat", stats_total.solver_sat)?;
    stats.set_item("solver_unsat", stats_total.solver_unsat)?;
    stats.set_item("solver_unknown", stats_total.solver_unknown)?;
    stats.set_item("solver_errors", stats_total.solver_errors)?;
    stats.set_item("solver_cache_hits", stats_total.solver_cache_hits)?;
    stats.set_item("solver_cache_entries", solver_cache.len())?;
    stats.set_item("solver_time_ms", stats_total.solver_time_ms)?;
    stats.set_item("elapsed_ms", elapsed.as_secs_f64() * 1000.0)?;
    stats.set_item("states_per_sec", per_sec(states_explored, elapsed))?;
    stats.set_item(
        "solver_queries_per_sec",
        per_sec(stats_total.solver_queries, elapsed),
    )?;
    stats.set_item("candidates_per_sec", per_sec(candidate_count, elapsed))?;
    stats.set_item(
        "solver_cache_hit_rate",
        ratio(
            stats_total.solver_cache_hits,
            stats_total.solver_cache_hits + stats_total.solver_queries,
        ),
    )?;
    stats.set_item("unsupported", stats_total.unsupported)?;
    stats.set_item("value_profile_hints", stats_total.value_profile_hints)?;
    stats.set_item(
        "value_profile_candidates",
        stats_total.value_profile_candidates,
    )?;
    stats.set_item("value_profile_deferred", stats_total.value_profile_deferred)?;
    stats.set_item(
        "branch_distance_objectives",
        stats_total.branch_distance_objectives,
    )?;
    stats.set_item(
        "branch_distance_candidates",
        stats_total.branch_distance_candidates,
    )?;
    stats.set_item(
        "branch_distance_overflowed_objectives",
        stats_total.branch_distance_overflowed_objectives,
    )?;
    stats.set_item(
        "branch_distance_best_delta",
        stats_total.branch_distance_best_delta,
    )?;
    stats.set_item(
        "frontier_ranked_objectives",
        stats_total.frontier_ranked_objectives,
    )?;
    stats.set_item("depth_bounded", stats_total.depth_bounded)?;
    stats.set_item("loop_bounded", stats_total.loop_bounded)?;
    stats.set_item("solver_query_bounded", stats_total.solver_query_bounded)?;
    stats.set_item("states_explored", states_explored)?;
    stats.set_item("states_bounded", states_bounded)?;
    stats.set_item("paths_feasible", feasible_paths)?;
    stats.set_item("paths_unknown", unknown_paths)?;
    stats.set_item("paths_infeasible", stats_total.solver_unsat)?;
    stats.set_item("worklist_remaining", queue.len())?;
    stats.set_item("path_priority", path_priority.as_str())?;
    stats.set_item("value_profile_policy", value_profile_policy.as_str())?;
    stats.set_item("branch_distance_policy", branch_distance_policy.as_str())?;
    stats.set_item("solver_backend", "z3-smtlib-persistent")?;
    stats.set_item(
        "complete_within_bounds",
        queue.is_empty()
            && states_bounded == 0
            && unknown_paths == 0
            && !stats_total.solver_query_bounded,
    )?;

    out.set_item("candidates", cand_list)?;
    out.set_item("stats", stats)?;
    Ok(out.into())
}

fn compare_candidates(a: &Candidate, b: &Candidate) -> Ordering {
    b.priority
        .cmp(&a.priority)
        .then_with(|| a.kind.as_str().cmp(b.kind.as_str()))
        .then_with(|| a.target_block.cmp(&b.target_block))
        .then_with(|| a.source_block.cmp(&b.source_block))
        .then_with(|| a.branch_index.cmp(&b.branch_index))
}

fn pop_queued_state(
    queue: &mut Vec<QueuedState>,
    path_priority: PathPriority,
    states_explored: usize,
) -> Option<QueuedState> {
    if queue.is_empty() {
        return None;
    }

    let idx = match path_priority {
        PathPriority::Fifo => 0,
        PathPriority::CoverageDirected => {
            if states_explored > 0 && states_explored % 8 == 7 {
                states_explored
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(12_345)
                    % queue.len()
            } else {
                let mut best = 0usize;
                for idx in 1..queue.len() {
                    if queued_state_cmp(&queue[idx], &queue[best]) == Ordering::Greater {
                        best = idx;
                    }
                }
                best
            }
        }
    };

    Some(queue.swap_remove(idx))
}

fn queued_state_cmp(a: &QueuedState, b: &QueuedState) -> Ordering {
    a.priority.cmp(&b.priority).then_with(|| b.seq.cmp(&a.seq))
}

impl<'a> Engine<'a> {
    fn new(
        program: &'a CompiledProgram,
        vm: VM,
        symbols: Vec<InputSymbol>,
        covered_blocks: FxHashSet<String>,
        solver_cache: &'a mut FxHashMap<String, SolveResult>,
        loop_bound: usize,
        max_path_depth: usize,
        max_solver_queries: usize,
        solver_timeout_ms: u64,
    ) -> Self {
        let sym_vars = vec![None; program.num_vars as usize];
        let sym_maps = vec![FxHashMap::default(); vm.memory_maps.len()];
        Self {
            program,
            vm,
            sym_vars,
            sym_maps,
            sym_external: Vec::new(),
            sym_havoc: FxHashMap::default(),
            path_constraints: Vec::new(),
            symbols,
            covered_blocks,
            candidates: Vec::new(),
            candidate_sigs: FxHashMap::default(),
            solver_cache,
            solver: None,
            stats: Stats::default(),
            loop_counts: FxHashMap::default(),
            loop_bound,
            max_path_depth,
            max_solver_queries,
            solver_timeout_ms,
            target_all_branches: false,
            path_priority: PathPriority::CoverageDirected,
            value_profile_policy: ValueProfilePolicy::Late,
            branch_distance_policy: BranchDistancePolicy::Auto,
        }
    }

    fn install_symbols(&mut self) {
        self.sym_external = vec![None; self.vm.external_buffer.len()];
        for sym_def in &self.symbols {
            let sym = Sym::Bv {
                text: sym_def.name.clone(),
                bits: sym_def.bits,
                provenance: Provenance::symbol(sym_def.name.clone(), sym_def.bits),
            };
            if let Some(var_id) = sym_def.var_id {
                let idx = var_id as usize;
                if idx < self.sym_vars.len() {
                    self.sym_vars[idx] = Some(sym.clone());
                }
            }
            if let Some((ref map_name, addr)) = sym_def.map_key {
                if let Some(map_idx) = self.find_map_idx(map_name) {
                    self.sym_maps[map_idx].insert(addr, sym.clone());
                }
            }
            if let Some(idx) = sym_def.extra_index {
                if idx < self.sym_external.len() {
                    self.sym_external[idx] = Some(sym.clone());
                }
            }
            if let Some(key) = sym_def.havoc_key {
                self.sym_havoc.insert(key, sym);
            }
        }
    }

    fn unsupported(&mut self, reason: &str, fields: &[(&str, String)]) {
        self.stats.unsupported += 1;
        if !debug_log::is_enabled("unsupported") {
            return;
        }
        let mut event_fields = vec![
            ("reason", reason.to_string()),
            ("pc", self.vm.pc.to_string()),
            ("block", self.vm.curr_block.clone()),
        ];
        for (key, value) in fields {
            event_fields.push((*key, value.clone()));
        }
        debug_log::event("unsupported", "symbolic_unsupported", &event_fields);
    }

    fn execute(&mut self) {
        if !self.check_entry_preconditions() {
            return;
        }
        let mut block_id = self.program.entry_block;
        let mut depth = 0usize;
        loop {
            if depth >= self.max_path_depth {
                self.stats.depth_bounded = true;
                if debug_log::is_enabled("solver") {
                    debug_log::event(
                        "solver",
                        "path_depth_bounded",
                        &[
                            ("depth", depth.to_string()),
                            ("max_path_depth", self.max_path_depth.to_string()),
                            ("block", self.vm.curr_block.clone()),
                        ],
                    );
                }
                break;
            }
            depth += 1;
            if !self.record_loop_visit(block_id) {
                if debug_log::is_enabled("solver") {
                    debug_log::event(
                        "solver",
                        "loop_bounded",
                        &[
                            ("block_id", block_id.to_string()),
                            ("loop_bound", self.loop_bound.to_string()),
                        ],
                    );
                }
                break;
            }

            let block = &self.program.blocks[block_id as usize];
            self.vm.explored_blocks.insert(block_id);
            self.vm.curr_block.clone_from(&block.name);
            self.vm.curr_block_id = block_id;
            self.vm.pc = block.start_pc;
            if !self.vm.no_trace {
                self.vm.trace.on_block_enter(block_id);
            }

            for stmt in &block.body {
                if !self.execute_stmt(stmt) {
                    return;
                }
                self.vm.pc += 1;
            }

            match &block.terminator {
                Stmt::Return => break,
                Stmt::Goto { targets } => {
                    if targets.len() == 1 {
                        block_id = targets[0];
                    } else if let Some(next) = self.resolve_branch(targets, block) {
                        block_id = next;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
    }

    fn check_entry_preconditions(&mut self) -> bool {
        if self.program.entry_preconditions.is_empty() {
            return true;
        }
        let entry = &self.program.blocks[self.program.entry_block as usize];
        self.vm.curr_block.clone_from(&entry.name);
        self.vm.curr_block_id = self.program.entry_block;
        self.vm.pc = entry.start_pc;
        for (idx, expr) in self.program.entry_preconditions.iter().enumerate() {
            let ev = self.eval_bool(expr);
            if !ev.value {
                if let Some(sym) = &ev.sym {
                    if let Some(bool_expr) = sym_bool_text(sym) {
                        self.stats.objectives += 1;
                        self.try_solve(
                            entry.name.clone(),
                            format!("requires@{}", idx),
                            idx,
                            bool_expr,
                            CandidateKind::RequiresRepair,
                        );
                    } else {
                        self.unsupported(
                            "requires_symbolic_bool",
                            &[("requires_index", idx.to_string())],
                        );
                    }
                }
                return false;
            }
            if let Some(sym) = ev.sym {
                if let Some(bool_expr) = sym_bool_text(&sym) {
                    self.path_constraints.push(bool_expr);
                }
            }
        }
        true
    }

    fn record_loop_visit(&mut self, block_id: BlockId) -> bool {
        if !self
            .program
            .is_loop_header
            .get(block_id as usize)
            .copied()
            .unwrap_or(false)
        {
            return true;
        }

        let count = self.loop_counts.entry(block_id).or_insert(0);
        *count += 1;
        if *count > self.loop_bound.saturating_add(1) {
            self.stats.loop_bounded = true;
            return false;
        }
        true
    }

    fn resolve_branch(&mut self, targets: &[BlockId], source: &Block) -> Option<BlockId> {
        self.stats.branches_seen += 1;
        let mut taken = None;
        let mut evals = Vec::new();
        for &target_id in targets {
            let block = &self.program.blocks[target_id as usize];
            if let Some(ref cond) = block.assume_cond {
                let ev = self.eval_bool(cond);
                if ev.value {
                    if taken.is_some() {
                        self.unsupported(
                            "multiple_true_branch_targets",
                            &[
                                ("source", source.name.clone()),
                                ("targets", targets.len().to_string()),
                            ],
                        );
                        if debug_log::is_enabled("branch") {
                            debug_log::event(
                                "branch",
                                "branch_unsupported",
                                &[
                                    ("source", source.name.clone()),
                                    ("reason", "multiple_true_targets".to_string()),
                                    ("targets", targets.len().to_string()),
                                ],
                            );
                        }
                        return None;
                    }
                    taken = Some(target_id);
                }
                evals.push((target_id, Some(ev)));
            } else {
                evals.push((target_id, None));
            }
        }

        let taken_id = match taken {
            Some(id) => id,
            None => {
                self.unsupported(
                    "no_true_branch_target",
                    &[
                        ("source", source.name.clone()),
                        ("targets", targets.len().to_string()),
                    ],
                );
                if debug_log::is_enabled("branch") {
                    debug_log::event(
                        "branch",
                        "branch_unsupported",
                        &[
                            ("source", source.name.clone()),
                            ("reason", "no_true_target".to_string()),
                            ("targets", targets.len().to_string()),
                        ],
                    );
                }
                return None;
            }
        };
        if debug_log::is_enabled("branch") {
            debug_log::event(
                "branch",
                "branch_resolved",
                &[
                    ("source", source.name.clone()),
                    ("taken", self.program.blocks[taken_id as usize].name.clone()),
                    ("targets", targets.len().to_string()),
                    ("pc", self.vm.pc.to_string()),
                    ("path_constraints", self.path_constraints.len().to_string()),
                ],
            );
        }

        let mut objective_indices: Vec<usize> = evals
            .iter()
            .enumerate()
            .filter_map(|(idx, (target_id, _ev))| {
                if *target_id == taken_id {
                    None
                } else {
                    Some(idx)
                }
            })
            .collect();
        objective_indices.sort_by(|a, b| {
            let a_id = evals[*a].0;
            let b_id = evals[*b].0;
            let a_uncovered = !self
                .covered_blocks
                .contains(&self.program.blocks[a_id as usize].name);
            let b_uncovered = !self
                .covered_blocks
                .contains(&self.program.blocks[b_id as usize].name);
            b_uncovered
                .cmp(&a_uncovered)
                .then_with(|| {
                    self.uncovered_reachable_count(b_id)
                        .cmp(&self.uncovered_reachable_count(a_id))
                })
                .then_with(|| a_id.cmp(&b_id))
        });

        for idx in objective_indices {
            let target_id = evals[idx].0;
            if target_id == taken_id {
                continue;
            }
            let target_name = &self.program.blocks[target_id as usize].name;
            if !self.target_all_branches && self.covered_blocks.contains(target_name) {
                continue;
            }
            self.stats.objectives += 1;
            self.stats.frontier_ranked_objectives += 1;
            match self.branch_objective_for_target(&evals, idx) {
                BranchObjective::Sat(expr) => {
                    self.try_solve(
                        source.name.clone(),
                        target_name.clone(),
                        idx,
                        expr,
                        CandidateKind::BranchTarget,
                    );
                }
                BranchObjective::Unsat => self.stats.solver_unsat += 1,
                BranchObjective::Unsupported => self.unsupported(
                    "branch_objective_unsupported",
                    &[
                        ("source", source.name.clone()),
                        ("target", target_name.clone()),
                    ],
                ),
            }
        }

        self.push_taken_branch_constraints(&evals, taken_id);

        Some(taken_id)
    }

    fn uncovered_reachable_count(&self, start: BlockId) -> usize {
        let mut stack = vec![start];
        let mut seen = FxHashSet::default();
        let mut count = 0usize;
        let mut budget = 64usize;
        while let Some(block_id) = stack.pop() {
            if budget == 0 || !seen.insert(block_id) {
                continue;
            }
            budget -= 1;
            let block = &self.program.blocks[block_id as usize];
            if !self.covered_blocks.contains(&block.name) {
                count += 1;
            }
            if let Stmt::Goto { targets } = &block.terminator {
                for target in targets {
                    stack.push(*target);
                }
            }
        }
        count
    }

    fn block_id_for_name(&self, name: &str) -> Option<BlockId> {
        self.program
            .blocks
            .iter()
            .position(|block| block.name == name)
            .map(|idx| idx as BlockId)
    }

    fn distance_to_uncovered(&self, start: BlockId) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut seen = FxHashSet::default();
        queue.push_back((start, 0usize));
        let mut budget = 256usize;
        while let Some((block_id, distance)) = queue.pop_front() {
            if budget == 0 || !seen.insert(block_id) {
                continue;
            }
            budget -= 1;
            let block = &self.program.blocks[block_id as usize];
            if !self.covered_blocks.contains(&block.name) {
                return Some(distance);
            }
            if let Stmt::Goto { targets } = &block.terminator {
                for target in targets {
                    queue.push_back((*target, distance + 1));
                }
            }
        }
        None
    }

    fn candidate_score(&self, kind: CandidateKind, target_block: &str) -> (i64, i64, usize) {
        let mut priority = kind.base_priority();
        let mut distance_to_uncovered = -1i64;
        let mut reachable_uncovered = 0usize;

        if let Some(block_id) = self.block_id_for_name(target_block) {
            let block = &self.program.blocks[block_id as usize];
            if !self.covered_blocks.contains(&block.name) {
                priority += 500_000;
            }
            reachable_uncovered = self.uncovered_reachable_count(block_id);
            priority += (reachable_uncovered as i64).saturating_mul(1_000);
            if let Some(distance) = self.distance_to_uncovered(block_id) {
                distance_to_uncovered = distance as i64;
                priority += 10_000i64.saturating_sub(distance as i64);
            }
        }

        (priority, distance_to_uncovered, reachable_uncovered)
    }

    fn branch_objective_for_target(
        &self,
        evals: &[(BlockId, Option<BoolEval>)],
        target_idx: usize,
    ) -> BranchObjective {
        let mut constraints = Vec::new();
        for (idx, (_target_id, ev)) in evals.iter().enumerate() {
            let Some(ev) = ev else {
                return BranchObjective::Unsupported;
            };
            let want_true = idx == target_idx;
            if let Some(sym) = &ev.sym {
                let Some(expr) = sym_bool_text(sym) else {
                    return BranchObjective::Unsupported;
                };
                if want_true {
                    constraints.push(expr);
                } else {
                    constraints.push(format!("(not {})", expr));
                }
            } else if ev.value != want_true {
                return BranchObjective::Unsat;
            }
        }
        if constraints.is_empty() {
            BranchObjective::Unsat
        } else if constraints.len() == 1 {
            BranchObjective::Sat(constraints.remove(0))
        } else {
            BranchObjective::Sat(format!("(and {})", constraints.join(" ")))
        }
    }

    fn push_taken_branch_constraints(
        &mut self,
        evals: &[(BlockId, Option<BoolEval>)],
        taken_id: BlockId,
    ) {
        for (target_id, ev) in evals {
            let Some(ev) = ev else {
                continue;
            };
            let Some(sym) = &ev.sym else {
                continue;
            };
            let Some(expr) = sym_bool_text(sym) else {
                continue;
            };
            if *target_id == taken_id {
                self.path_constraints.push(expr);
            } else {
                self.path_constraints.push(format!("(not {})", expr));
            }
        }
    }

    fn try_solve(
        &mut self,
        source_block: String,
        target_block: String,
        branch_index: usize,
        objective: String,
        kind: CandidateKind,
    ) {
        let mut constraints = self.path_constraints.clone();
        constraints.push(objective);
        let key = solve_cache_key(&constraints);

        let mut cache_hit = false;
        let result = if let Some(cached) = self.solver_cache.get(&key) {
            self.stats.solver_cache_hits += 1;
            cache_hit = true;
            cached.clone()
        } else {
            if self.stats.solver_queries >= self.max_solver_queries {
                self.stats.solver_query_bounded = true;
                if debug_log::is_enabled("solver") {
                    debug_log::event(
                        "solver",
                        "solver_query_bounded",
                        &[
                            ("source", source_block.clone()),
                            ("target", target_block.clone()),
                            ("max_solver_queries", self.max_solver_queries.to_string()),
                            ("constraints", constraints.len().to_string()),
                        ],
                    );
                }
                return;
            }
            self.stats.solver_queries += 1;
            let solve_start = Instant::now();
            let solved = solve_with_backend(
                &mut self.solver,
                &self.symbols,
                &constraints,
                self.solver_timeout_ms,
            );
            self.stats.solver_time_ms += solve_start.elapsed().as_millis() as u64;
            self.solver_cache.insert(key, solved.clone());
            solved
        };
        let result_name = match &result {
            SolveResult::Sat(_) => "sat",
            SolveResult::Unsat => "unsat",
            SolveResult::Unknown => "unknown",
            SolveResult::Error => "error",
        };
        if debug_log::is_enabled("solver") {
            debug_log::event(
                "solver",
                "solver_result",
                &[
                    ("source", source_block.clone()),
                    ("target", target_block.clone()),
                    ("branch_index", branch_index.to_string()),
                    ("result", result_name.to_string()),
                    ("cache_hit", cache_hit.to_string()),
                    ("constraints", constraints.len().to_string()),
                    ("path_constraints", self.path_constraints.len().to_string()),
                    ("solver_queries", self.stats.solver_queries.to_string()),
                ],
            );
        }

        match result {
            SolveResult::Sat(values) => {
                self.stats.solver_sat += 1;
                let updates = values
                    .into_iter()
                    .filter_map(|(name, value)| {
                        self.symbols
                            .iter()
                            .find(|s| s.name == name)
                            .map(|_| (name, value))
                    })
                    .collect();
                let (priority, distance_to_uncovered, reachable_uncovered) =
                    self.candidate_score(kind, &target_block);
                self.push_candidate(Candidate {
                    updates,
                    source_block,
                    target_block,
                    branch_index,
                    kind,
                    priority,
                    distance_to_uncovered,
                    reachable_uncovered,
                });
            }
            SolveResult::Unsat => self.stats.solver_unsat += 1,
            SolveResult::Unknown => self.stats.solver_unknown += 1,
            SolveResult::Error => self.stats.solver_errors += 1,
        }
    }

    fn push_candidate(&mut self, candidate: Candidate) -> bool {
        if candidate.updates.is_empty() {
            if debug_log::is_enabled("candidate") {
                debug_log::event(
                    "candidate",
                    "native_candidate_skip",
                    &[
                        ("reason", "empty_updates".to_string()),
                        ("source", candidate.source_block),
                        ("target", candidate.target_block),
                    ],
                );
            }
            return false;
        }
        let sig = update_signature(&candidate.updates);
        if let Some(existing_idx) = self.candidate_sigs.get(&sig).copied() {
            if candidate.priority > self.candidates[existing_idx].priority {
                if debug_log::is_enabled("candidate") {
                    debug_log::event(
                        "candidate",
                        "native_candidate_upgrade",
                        &[
                            ("source", candidate.source_block.clone()),
                            ("target", candidate.target_block.clone()),
                            ("kind", candidate.kind.as_str().to_string()),
                            ("priority", candidate.priority.to_string()),
                            (
                                "previous_kind",
                                self.candidates[existing_idx].kind.as_str().to_string(),
                            ),
                            (
                                "previous_priority",
                                self.candidates[existing_idx].priority.to_string(),
                            ),
                        ],
                    );
                }
                self.candidates[existing_idx] = candidate;
                true
            } else {
                if debug_log::is_enabled("candidate") {
                    debug_log::event(
                        "candidate",
                        "native_candidate_skip",
                        &[
                            ("reason", "duplicate".to_string()),
                            ("source", candidate.source_block),
                            ("target", candidate.target_block),
                            ("kind", candidate.kind.as_str().to_string()),
                        ],
                    );
                }
                false
            }
        } else {
            let idx = self.candidates.len();
            self.candidate_sigs.insert(sig, idx);
            if debug_log::is_enabled("candidate") {
                debug_log::event(
                    "candidate",
                    "native_candidate",
                    &[
                        ("source", candidate.source_block.clone()),
                        ("target", candidate.target_block.clone()),
                        ("branch_index", candidate.branch_index.to_string()),
                        ("kind", candidate.kind.as_str().to_string()),
                        ("priority", candidate.priority.to_string()),
                        ("updates", candidate.updates.len().to_string()),
                    ],
                );
            }
            self.candidates.push(candidate);
            true
        }
    }

    fn record_binop_value_profile(
        &mut self,
        op: BinOp,
        lhs_sym: &Option<Sym>,
        lhs_value: i64,
        rhs_sym: &Option<Sym>,
        rhs_value: i64,
    ) {
        match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {}
            _ => return,
        }
        if lhs_sym.is_some() && rhs_sym.is_none() {
            self.record_cmp_values(lhs_sym, op, rhs_value, 64, true, lhs_value);
        }
        if rhs_sym.is_some() && lhs_sym.is_none() {
            self.record_cmp_values(rhs_sym, invert_cmp_op(op), lhs_value, 64, true, rhs_value);
        }
    }

    fn record_builtin_value_profile(
        &mut self,
        fn_id: BuiltinFn,
        lhs_sym: &Option<Sym>,
        lhs_value: i64,
        rhs_sym: &Option<Sym>,
        rhs_value: i64,
    ) {
        let Some((op, bits, signed)) = builtin_cmp_profile(fn_id) else {
            return;
        };
        if lhs_sym.is_some() && rhs_sym.is_none() {
            self.record_cmp_values(lhs_sym, op, rhs_value, bits, signed, lhs_value);
        }
        if rhs_sym.is_some() && lhs_sym.is_none() {
            self.record_cmp_values(
                rhs_sym,
                invert_cmp_op(op),
                lhs_value,
                bits,
                signed,
                rhs_value,
            );
        }
    }

    fn record_cmp_values(
        &mut self,
        sym: &Option<Sym>,
        op: BinOp,
        constant: i64,
        bits: u32,
        signed: bool,
        expr_value: i64,
    ) {
        self.record_branch_distance_candidates(sym, op, constant, bits, signed, expr_value);
        let Some((name, sym_bits)) = sym.as_ref().and_then(direct_symbol) else {
            return;
        };
        self.stats.value_profile_hints += 1;
        if self.value_profile_policy == ValueProfilePolicy::Off {
            return;
        }
        let bits = bits.min(sym_bits).max(1);
        for value in cmp_profile_values(op, constant) {
            let updates = vec![(name.clone(), mask_to_bits(value, bits))];
            let target_block = format!("cmp@{}", self.vm.pc);
            let (priority, distance_to_uncovered, reachable_uncovered) =
                self.candidate_score(CandidateKind::ValueProfile, &target_block);
            let inserted = self.push_candidate(Candidate {
                updates,
                source_block: self.vm.curr_block.clone(),
                target_block,
                branch_index: 0,
                kind: CandidateKind::ValueProfile,
                priority,
                distance_to_uncovered,
                reachable_uncovered,
            });
            if inserted {
                self.stats.value_profile_candidates += 1;
            }
        }
    }

    fn record_branch_distance_candidates(
        &mut self,
        sym: &Option<Sym>,
        op: BinOp,
        constant: i64,
        bits: u32,
        signed: bool,
        expr_value: i64,
    ) {
        if !self.branch_distance_policy.enabled() {
            return;
        }
        let Some(sym) = sym.as_ref() else {
            return;
        };
        let bits = bits.min(sym.bits()).clamp(1, 64);
        let provenance = sym.provenance();
        if provenance.terms.is_empty() {
            return;
        }
        self.stats.branch_distance_objectives += 1;
        if provenance.overflow {
            self.stats.branch_distance_overflowed_objectives += 1;
        }
        let current_distance = branch_distance(op, expr_value, constant, bits, signed);
        self.stats.branch_distance_best_delta = self
            .stats
            .branch_distance_best_delta
            .max(current_distance.min(u64::MAX as u128) as u64);
        if current_distance == 0 {
            return;
        }
        if self.stats.branch_distance_candidates >= MAX_BRANCH_DISTANCE_CANDIDATES {
            return;
        }

        for desired in cmp_profile_values(op, constant) {
            if self.stats.branch_distance_candidates >= MAX_BRANCH_DISTANCE_CANDIDATES {
                break;
            }
            let Some(updates) = self.distance_updates(&provenance, expr_value, desired, bits)
            else {
                continue;
            };
            if updates.is_empty() {
                continue;
            }
            let target_block = format!("cmp@{}", self.vm.pc);
            let (mut priority, distance_to_uncovered, reachable_uncovered) =
                self.candidate_score(CandidateKind::DistanceGuided, &target_block);
            let distance_bonus = current_distance.min(1_000_000) as i64;
            priority += distance_bonus;
            let inserted = self.push_candidate(Candidate {
                updates,
                source_block: self.vm.curr_block.clone(),
                target_block,
                branch_index: 0,
                kind: CandidateKind::DistanceGuided,
                priority,
                distance_to_uncovered,
                reachable_uncovered,
            });
            if inserted {
                self.stats.branch_distance_candidates += 1;
            }
        }
    }

    fn distance_updates(
        &self,
        provenance: &Provenance,
        expr_value: i64,
        desired_expr_value: i64,
        bits: u32,
    ) -> Option<Vec<(String, u64)>> {
        if provenance.terms.len() > MAX_PROVENANCE_TERMS {
            return None;
        }
        let mut updates = FxHashMap::default();
        let expr_mask = mask_u128(bits);
        let current_expr = (expr_value as u64 as u128) & expr_mask;
        let desired_expr = (desired_expr_value as u64 as u128) & expr_mask;
        let delta = desired_expr.wrapping_sub(current_expr) & expr_mask;
        let current_pack = self.provenance_current_value(provenance)?;
        let desired_pack = current_pack.wrapping_add(delta) & expr_mask;
        for term in &provenance.terms {
            if term.bits == 0 || term.bits > 64 || term.low >= 64 {
                return None;
            }
            let value = ((desired_pack >> term.low) & mask_u128(term.bits)) as u64;
            match updates.get(term.name.as_str()).copied() {
                Some(existing) if existing != value => return None,
                Some(_) => {}
                None => {
                    updates.insert(term.name.as_str(), value);
                }
            }
        }
        let mut out: Vec<(String, u64)> = updates
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Some(out)
    }

    fn provenance_current_value(&self, provenance: &Provenance) -> Option<u128> {
        let mut out = 0u128;
        for term in &provenance.terms {
            let sym = self.symbols.iter().find(|s| s.name == term.name)?;
            if term.bits == 0 || term.bits > 64 || term.low >= 64 {
                return None;
            }
            let value = (sym.value as u128) & mask_u128(term.bits);
            out |= value << term.low;
        }
        Some(out)
    }

    fn execute_stmt(&mut self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Assign1 { lhs, rhs } => {
                let ev = self.eval(rhs);
                self.set_eval_result(*lhs, ev);
            }
            Stmt::AssignN { lhs, rhs } => {
                let vals: Vec<CEval> = rhs.iter().map(|r| self.eval(r)).collect();
                for (var_id, ev) in lhs.iter().zip(vals) {
                    self.set_eval_result(*var_id, ev);
                }
            }
            Stmt::Assert { expr } => {
                let ev = self.eval_bool(expr);
                if !ev.value {
                    if let Some(sym) = &ev.sym {
                        if let Some(bool_expr) = sym_bool_text(sym) {
                            self.stats.objectives += 1;
                            self.try_solve(
                                self.vm.curr_block.clone(),
                                format!("assert@{}", self.vm.pc),
                                0,
                                bool_expr,
                                CandidateKind::AssertRepair,
                            );
                        } else {
                            self.unsupported("assert_symbolic_bool", &[]);
                        }
                    }
                    return false;
                }
                if let Some(sym) = ev.sym {
                    if let Some(bool_expr) = sym_bool_text(&sym) {
                        self.path_constraints.push(bool_expr);
                    }
                }
            }
            Stmt::Assume { expr } => {
                let ev = self.eval_bool(expr);
                if !ev.value {
                    return false;
                }
                if let Some(sym) = ev.sym {
                    if let Some(bool_expr) = sym_bool_text(&sym) {
                        self.path_constraints.push(bool_expr);
                    }
                }
            }
            Stmt::AssumeTrue | Stmt::LoopHeaderSnap { .. } | Stmt::CallIgnored => {}
            Stmt::Havoc { vars } => {
                for &var_id in vars {
                    self.vm.clear_var(var_id);
                    self.sym_vars[var_id as usize] = None;
                }
            }
            Stmt::HavocCurrAddr {
                var_id,
                alloc_size_var,
            } => {
                if *alloc_size_var == u32::MAX {
                    self.unsupported(
                        "havoc_curr_addr_missing_size",
                        &[("var_id", var_id.to_string())],
                    );
                    return false;
                }
                let alloc_size = self.vm.get_scalar_silent(*alloc_size_var);
                let is_shadow = Some(*var_id) == self.vm.curr_addr_shadow_id;
                let old_addr = if is_shadow {
                    self.vm.alloc_addr_shadow
                } else {
                    self.vm.alloc_addr
                };
                let new_addr = (old_addr + alloc_size + 255) & !255;
                self.vm.set_scalar(*var_id, new_addr, false);
                self.sym_vars[*var_id as usize] = None;
            }
            Stmt::CallNondet { assignments } => {
                for &var_id in assignments {
                    let count = self.vm.havoc_count(var_id);
                    let value = self.vm.next_havoc_value(var_id);
                    self.vm.set_scalar(var_id, value, false);
                    self.sym_vars[var_id as usize] = self.sym_havoc.get(&(var_id, count)).cloned();
                }
            }
            Stmt::CallPrintf { args } => {
                for arg in args {
                    let _ = self.eval_i64(arg);
                }
            }
            Stmt::CallTime { assignments, args } => {
                for arg in args {
                    let _ = self.eval_i64(arg);
                }
                for &var_id in assignments {
                    self.vm.set_scalar(var_id, 0, false);
                    self.sym_vars[var_id as usize] = None;
                }
            }
            Stmt::CallWrite { assignments, args } => {
                for arg in args {
                    let _ = self.eval_i64(arg);
                }
                let val = self.vm.wlen_buf[self.vm.wlen_buf_idx % self.vm.wlen_buf.len()];
                self.vm.wlen_buf_idx += 1;
                for &var_id in assignments {
                    self.vm.set_scalar(var_id, val, false);
                    self.sym_vars[var_id as usize] = None;
                }
            }
            Stmt::CallRead { args } => {
                let vals: Vec<CEval> = args.iter().map(|a| self.eval(a)).collect();
                let buf_ptr = scalar_value(&vals[2].value).unwrap_or(0);
                let buf_ptr_shadow = scalar_value(&vals[3].value).unwrap_or(0);
                let read_len = scalar_value(&vals[4].value).unwrap_or(0);
                let read_len_shadow = scalar_value(&vals[5].value).unwrap_or(0);
                if read_len != read_len_shadow || read_len < 0 {
                    return false;
                }
                let bytes = self.read_external(read_len as usize);
                if let (Some(m0_id), Some(m0s_id)) = (self.vm.m0_id, self.vm.m0_shadow_id) {
                    if let (Some(m0_idx), Some(m0s_idx)) =
                        (self.get_map_idx(m0_id), self.get_map_idx(m0s_id))
                    {
                        for (i, (value, sym)) in bytes.into_iter().enumerate() {
                            let a = buf_ptr + i as i64;
                            let as_ = buf_ptr_shadow + i as i64;
                            self.vm.memory_maps[m0_idx].set(a, value);
                            self.vm.memory_maps[m0s_idx].set(as_, value);
                            if let Some(sym) = sym {
                                self.sym_maps[m0_idx].insert(a, sym.clone());
                                self.sym_maps[m0s_idx].insert(as_, sym);
                            }
                        }
                    }
                }
            }
            Stmt::CallMemmove { args } => {
                let vals: Vec<CEval> = args.iter().map(|a| self.eval_i64(a)).collect();
                if vals.len() >= 6 {
                    let dst = scalar_value(&vals[0].value).unwrap_or(0);
                    let dst_shadow = scalar_value(&vals[1].value).unwrap_or(0);
                    let src = scalar_value(&vals[2].value).unwrap_or(0);
                    let src_shadow = scalar_value(&vals[3].value).unwrap_or(0);
                    let len = scalar_value(&vals[4].value).unwrap_or(0);
                    let len_shadow = scalar_value(&vals[5].value).unwrap_or(0);
                    if len != len_shadow || len < 0 {
                        return false;
                    }
                    self.memmove_i8_maps(dst, dst_shadow, src, src_shadow, len);
                }
            }
            Stmt::QuantMemsetWrite {
                m_ret,
                dst,
                len,
                val,
            } => {
                self.quant_memset_write(*m_ret, *dst, *len, *val);
            }
            Stmt::QuantMemsetPreserveLt { m_ret, m_src, dst } => {
                self.copy_filtered(
                    *m_ret,
                    *m_src,
                    |addr, boundary| addr < boundary,
                    self.vm.get_scalar_silent(*dst),
                );
            }
            Stmt::QuantMemsetPreserveGe {
                m_ret,
                m_src,
                dst,
                len,
            } => {
                let boundary = self.vm.get_scalar_silent(*dst) + self.vm.get_scalar_silent(*len);
                self.copy_filtered(*m_ret, *m_src, |addr, boundary| addr >= boundary, boundary);
            }
            Stmt::QuantMemcpyWrite {
                m_ret,
                m_src,
                dst,
                src,
                len,
            } => {
                self.quant_memcpy_write(*m_ret, *m_src, *dst, *src, *len);
            }
            Stmt::QuantMemcpyPreserveLt { m_ret, m_src, dst } => {
                self.copy_filtered(
                    *m_ret,
                    *m_src,
                    |addr, boundary| addr < boundary,
                    self.vm.get_scalar_silent(*dst),
                );
            }
            Stmt::QuantMemcpyPreserveGe {
                m_ret,
                m_src,
                dst,
                len,
            } => {
                let boundary = self.vm.get_scalar_silent(*dst) + self.vm.get_scalar_silent(*len);
                self.copy_filtered(*m_ret, *m_src, |addr, boundary| addr >= boundary, boundary);
            }
            Stmt::Goto { .. } | Stmt::Return => return false,
        }
        true
    }

    fn read_external(&mut self, n: usize) -> Vec<(i64, Option<Sym>)> {
        let start = self.vm.external_buffer_pos;
        let end = (start + n).min(self.vm.external_buffer.len());
        self.vm.external_buffer_pos = end;
        (start..end)
            .map(|idx| {
                (
                    self.vm.external_buffer[idx] as i64,
                    self.sym_external.get(idx).and_then(|s| s.clone()),
                )
            })
            .collect()
    }

    fn quant_memset_write(&mut self, m_ret: VarId, dst: VarId, len: VarId, val: VarId) {
        let dst_val = self.vm.get_scalar_silent(dst);
        let len_val = self.vm.get_scalar_silent(len);
        let val_val = self.vm.get_scalar_silent(val);
        let val_sym = self.sym_vars[val as usize].clone();
        if let Some(map_idx) = self.get_map_idx(m_ret) {
            for addr in dst_val..dst_val + len_val {
                self.vm.memory_maps[map_idx].set(addr, val_val);
                if let Some(sym) = &val_sym {
                    self.sym_maps[map_idx].insert(addr, sym.clone());
                } else {
                    self.sym_maps[map_idx].remove(&addr);
                }
            }
        }
    }

    fn quant_memcpy_write(
        &mut self,
        m_ret: VarId,
        m_src: VarId,
        dst: VarId,
        src: VarId,
        len: VarId,
    ) {
        let dst_val = self.vm.get_scalar_silent(dst);
        let src_val = self.vm.get_scalar_silent(src);
        let len_val = self.vm.get_scalar_silent(len);
        if let (Some(src_idx), Some(dst_idx)) = (self.get_map_idx(m_src), self.get_map_idx(m_ret)) {
            for off in 0..len_val {
                let saddr = src_val + off;
                let daddr = dst_val + off;
                let value = self.vm.memory_maps[src_idx].get(saddr);
                self.vm.memory_maps[dst_idx].set(daddr, value);
                if let Some(sym) = self.sym_maps[src_idx].get(&saddr).cloned() {
                    self.sym_maps[dst_idx].insert(daddr, sym);
                } else {
                    self.sym_maps[dst_idx].remove(&daddr);
                }
            }
        }
    }

    fn memmove_i8_maps(&mut self, dst: i64, dst_shadow: i64, src: i64, src_shadow: i64, len: i64) {
        if len <= 0 {
            return;
        }
        for map_idx in 0..self.vm.memory_maps.len() {
            if self.vm.memory_maps[map_idx].element_bit_width != 8 {
                continue;
            }
            let is_shadow = self.vm.memory_maps[map_idx].name.ends_with(".shadow");
            let (dst_base, src_base) = if is_shadow {
                (dst_shadow, src_shadow)
            } else {
                (dst, src)
            };
            self.memmove_sparse_map(map_idx, dst_base, src_base, len);
        }
    }

    fn memmove_sparse_map(&mut self, map_idx: usize, dst: i64, src: i64, len: i64) {
        let Some(src_end) = src.checked_add(len) else {
            return;
        };
        let Some(dst_end) = dst.checked_add(len) else {
            return;
        };

        let copied: Vec<(i64, i64, Option<Sym>)> = self.vm.memory_maps[map_idx]
            .memory
            .iter()
            .filter_map(|(&addr, &value)| {
                if addr >= src && addr < src_end {
                    Some((
                        dst + (addr - src),
                        value,
                        self.sym_maps[map_idx].get(&addr).cloned(),
                    ))
                } else {
                    None
                }
            })
            .collect();
        let clear: Vec<i64> = self.vm.memory_maps[map_idx]
            .memory
            .keys()
            .filter(|&&addr| addr >= dst && addr < dst_end)
            .copied()
            .collect();

        for addr in clear {
            self.vm.memory_maps[map_idx].memory.remove(&addr);
            self.sym_maps[map_idx].remove(&addr);
        }
        for (addr, value, sym) in copied {
            self.vm.memory_maps[map_idx].set(addr, value);
            if let Some(sym) = sym {
                self.sym_maps[map_idx].insert(addr, sym);
            } else {
                self.sym_maps[map_idx].remove(&addr);
            }
        }
    }

    fn copy_filtered<F>(&mut self, m_ret: VarId, m_src: VarId, pred: F, boundary: i64)
    where
        F: Fn(i64, i64) -> bool,
    {
        if let (Some(src_idx), Some(dst_idx)) = (self.get_map_idx(m_src), self.get_map_idx(m_ret)) {
            let addrs: Vec<i64> = self.vm.memory_maps[src_idx]
                .memory
                .keys()
                .copied()
                .filter(|addr| pred(*addr, boundary))
                .collect();
            for addr in addrs {
                let value = self.vm.memory_maps[src_idx].get(addr);
                self.vm.memory_maps[dst_idx].set(addr, value);
                if let Some(sym) = self.sym_maps[src_idx].get(&addr).cloned() {
                    self.sym_maps[dst_idx].insert(addr, sym);
                } else {
                    self.sym_maps[dst_idx].remove(&addr);
                }
            }
        }
    }

    fn set_eval_result(&mut self, var_id: VarId, ev: CEval) {
        match ev.value {
            EvalResult::Scalar(v) => self.vm.set_scalar(var_id, v, false),
            EvalResult::Bool(b) => self.vm.set_scalar(var_id, b as i64, false),
            EvalResult::MapRef(map_idx) => {
                let vid = var_id as usize;
                if let Some(existing_idx) = self.vm.var_to_map[vid] {
                    if existing_idx != map_idx {
                        let new_name = self.vm.var_names[vid].clone();
                        let copied = self.vm.memory_maps[map_idx].copy_with_name(new_name);
                        self.vm.memory_maps[existing_idx] = copied;
                        self.sym_maps[existing_idx] = self.sym_maps[map_idx].clone();
                    }
                } else {
                    let new_name = self.vm.var_names[vid].clone();
                    let copied = self.vm.memory_maps[map_idx].copy_with_name(new_name);
                    let new_idx = self.vm.memory_maps.len();
                    self.vm.memory_maps.push(copied);
                    self.sym_maps.push(self.sym_maps[map_idx].clone());
                    self.vm.var_to_map[vid] = Some(new_idx);
                    self.vm.vars[vid] = Value::Map(new_idx);
                }
            }
        }
        self.sym_vars[var_id as usize] = ev.sym;
    }

    fn eval_bool(&mut self, expr: &Expr) -> BoolEval {
        let ev = self.eval(expr);
        let concrete = match ev.value {
            EvalResult::Scalar(v) => v != 0,
            EvalResult::Bool(b) => b,
            EvalResult::MapRef(_) => false,
        };
        BoolEval {
            value: concrete,
            sym: ev.sym.map(|s| s.as_bool()),
        }
    }

    fn eval_i64(&mut self, expr: &Expr) -> CEval {
        let ev = self.eval(expr);
        match ev.value {
            EvalResult::Scalar(_) | EvalResult::Bool(_) => ev,
            EvalResult::MapRef(_) => CEval {
                value: EvalResult::Scalar(0),
                sym: None,
            },
        }
    }

    fn eval(&mut self, expr: &Expr) -> CEval {
        match expr {
            Expr::Var(var_id) => {
                let vid = *var_id as usize;
                let value = self.vm.vars[vid].clone();
                CEval {
                    value: match value {
                        Value::Scalar(v) => EvalResult::Scalar(v),
                        Value::Map(idx) => EvalResult::MapRef(idx),
                    },
                    sym: self.sym_vars[vid].clone(),
                }
            }
            Expr::Const(v) => CEval {
                value: EvalResult::Scalar(*v),
                sym: None,
            },
            Expr::Bool(b) => CEval {
                value: EvalResult::Bool(*b),
                sym: None,
            },
            Expr::BinOp { op, lhs, rhs } => self.eval_binop(*op, lhs, rhs),
            Expr::Builtin { fn_id, args } => self.eval_builtin(*fn_id, args),
            Expr::Store {
                bit_width,
                map,
                index,
                value,
            } => self.eval_store(*bit_width, map, index, value),
            Expr::Load {
                bit_width,
                map,
                index,
            } => self.eval_load(*bit_width, map, index),
            Expr::IfThenElse { cond, then_, else_ } => {
                let c = self.eval_bool(cond);
                if c.value {
                    self.eval(then_)
                } else {
                    self.eval(else_)
                }
            }
            Expr::Not(inner) => {
                let ev = self.eval_bool(inner);
                CEval {
                    value: EvalResult::Bool(!ev.value),
                    sym: ev.sym.map(|s| s.not()),
                }
            }
            Expr::IsExternal => CEval {
                value: EvalResult::Scalar(0),
                sym: None,
            },
        }
    }

    fn eval_binop(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> CEval {
        let l = self.eval_i64(lhs);
        let r = self.eval_i64(rhs);
        let lv = eval_to_i64(&l.value);
        let rv = eval_to_i64(&r.value);
        let l_profile_sym = l.sym.clone();
        let r_profile_sym = r.sym.clone();
        self.record_binop_value_profile(op, &l_profile_sym, lv, &r_profile_sym, rv);
        let ls = l.sym.map(|s| s.as_bv(64));
        let rs = r.sym.map(|s| s.as_bv(64));
        let sym = match (ls, rs) {
            (Some(a), Some(b)) => Some(sym_binop(op, a, b)),
            (Some(a), None) => Some(sym_binop(op, a, Sym::bv_const(rv, 64))),
            (None, Some(b)) => Some(sym_binop(op, Sym::bv_const(lv, 64), b)),
            (None, None) => None,
        };
        let value = match op {
            BinOp::Eq => EvalResult::Bool(lv == rv),
            BinOp::Ne => EvalResult::Bool(lv != rv),
            BinOp::Lt => EvalResult::Bool(lv < rv),
            BinOp::Gt => EvalResult::Bool(lv > rv),
            BinOp::Le => EvalResult::Bool(lv <= rv),
            BinOp::Ge => EvalResult::Bool(lv >= rv),
            BinOp::And => EvalResult::Bool(lv != 0 && rv != 0),
            BinOp::Or => EvalResult::Bool(lv != 0 || rv != 0),
            BinOp::Implies => EvalResult::Bool(lv == 0 || rv != 0),
            BinOp::Iff => EvalResult::Bool((lv != 0) == (rv != 0)),
            BinOp::Sub => EvalResult::Scalar((lv.wrapping_sub(rv)) & MASK_64),
            BinOp::Mul => EvalResult::Scalar((lv.wrapping_mul(rv)) & MASK_64),
            BinOp::Add => EvalResult::Scalar((lv.wrapping_add(rv)) & MASK_64),
        };
        CEval { value, sym }
    }

    fn eval_builtin(&mut self, fn_id: BuiltinFn, args: &[Expr]) -> CEval {
        if builtins::num_args(fn_id) == 1 {
            let x = self.eval_i64(&args[0]);
            let xv = eval_to_i64(&x.value);
            let value = builtins::exec_unary(fn_id, xv);
            let sym = x.sym.map(|s| sym_unary_builtin(fn_id, s));
            CEval {
                value: EvalResult::Scalar(value),
                sym,
            }
        } else {
            let a = self.eval_i64(&args[0]);
            let b = self.eval_i64(&args[1]);
            let av = eval_to_i64(&a.value);
            let bv = eval_to_i64(&b.value);
            let (result, is_bool) = builtins::exec_binary(fn_id, av, bv);
            let a_profile_sym = a.sym.clone();
            let b_profile_sym = b.sym.clone();
            self.record_builtin_value_profile(fn_id, &a_profile_sym, av, &b_profile_sym, bv);
            let as_ = a.sym;
            let bs = b.sym;
            let sym = match (as_, bs) {
                (Some(a), Some(b)) => Some(sym_binary_builtin(fn_id, a, b)),
                (Some(a), None) => Some(sym_binary_builtin(fn_id, a, Sym::bv_const(bv, 64))),
                (None, Some(b)) => Some(sym_binary_builtin(fn_id, Sym::bv_const(av, 64), b)),
                (None, None) => None,
            };
            CEval {
                value: if is_bool {
                    EvalResult::Bool(result != 0)
                } else {
                    EvalResult::Scalar(result & MASK_64)
                },
                sym,
            }
        }
    }

    fn eval_store(&mut self, bit_width: u8, map: &Expr, index: &Expr, value: &Expr) -> CEval {
        let map_ev = self.eval(map);
        let map_idx = match map_ev.value {
            EvalResult::MapRef(idx) => idx,
            _ => {
                return CEval {
                    value: EvalResult::MapRef(0),
                    sym: None,
                }
            }
        };
        let idx_ev = self.eval_i64(index);
        if idx_ev.sym.is_some() {
            self.unsupported(
                "symbolic_store_index",
                &[("bit_width", bit_width.to_string())],
            );
        }
        let idx_val = eval_to_i64(&idx_ev.value);
        let val_ev = self.eval_i64(value);
        let val = eval_to_i64(&val_ev.value);
        let ew = self.vm.memory_maps[map_idx].element_bit_width;
        if bit_width == ew {
            self.vm.memory_maps[map_idx].set(idx_val, val);
            if let Some(sym) = val_ev.sym {
                self.sym_maps[map_idx].insert(idx_val, sym.as_bv(ew as u32));
            } else {
                self.sym_maps[map_idx].remove(&idx_val);
            }
        } else {
            let ew_mask = self.vm.memory_maps[map_idx].element_mask();
            let count = bit_width / ew;
            for i in 0..count as i64 {
                let addr = idx_val + i;
                self.vm.memory_maps[map_idx].set(addr, (val >> (i * ew as i64)) & ew_mask);
                if let Some(sym) = &val_ev.sym {
                    self.sym_maps[map_idx].insert(
                        addr,
                        sym.clone()
                            .as_bv(bit_width as u32)
                            .extract(((i + 1) as u32 * ew as u32) - 1, i as u32 * ew as u32),
                    );
                } else {
                    self.sym_maps[map_idx].remove(&addr);
                }
            }
        }
        CEval {
            value: EvalResult::MapRef(map_idx),
            sym: None,
        }
    }

    fn eval_load(&mut self, bit_width: u8, map: &Expr, index: &Expr) -> CEval {
        let map_ev = self.eval(map);
        let map_idx = match map_ev.value {
            EvalResult::MapRef(idx) => idx,
            _ => {
                return CEval {
                    value: EvalResult::Scalar(0),
                    sym: None,
                }
            }
        };
        let idx_ev = self.eval_i64(index);
        if idx_ev.sym.is_some() {
            self.unsupported(
                "symbolic_load_index",
                &[("bit_width", bit_width.to_string())],
            );
        }
        let idx_val = eval_to_i64(&idx_ev.value);
        let ew = self.vm.memory_maps[map_idx].element_bit_width;
        if bit_width == ew {
            CEval {
                value: EvalResult::Scalar(self.vm.memory_maps[map_idx].get(idx_val)),
                sym: self.sym_maps[map_idx].get(&idx_val).cloned(),
            }
        } else {
            let count = bit_width / ew;
            let mut result: i64 = 0;
            let mut sym: Option<Sym> = None;
            for i in 0..count as i64 {
                let addr = idx_val + i;
                result |= self.vm.memory_maps[map_idx].get(addr) << (i * ew as i64);
                let part = self.sym_maps[map_idx]
                    .get(&addr)
                    .cloned()
                    .map(|s| s.as_bv(bit_width as u32).bvshl_const(i as u32 * ew as u32));
                if let Some(part) = part {
                    sym = Some(match sym {
                        Some(acc) => acc.bvor(part),
                        None => part,
                    });
                }
            }
            CEval {
                value: EvalResult::Scalar(result),
                sym,
            }
        }
    }

    fn get_map_idx(&self, var_id: VarId) -> Option<usize> {
        match self.vm.vars.get(var_id as usize) {
            Some(Value::Map(idx)) => Some(*idx),
            _ => None,
        }
    }

    fn find_map_idx(&self, name: &str) -> Option<usize> {
        self.vm.memory_maps.iter().position(|m| m.name == name)
    }
}

struct BoolEval {
    value: bool,
    sym: Option<Sym>,
}

impl Sym {
    fn bv_const(value: i64, bits: u32) -> Self {
        let masked = if bits >= 64 {
            value as u64
        } else {
            (value as u64) & ((1u64 << bits) - 1)
        };
        Sym::Bv {
            text: format!("(_ bv{} {})", masked, bits),
            bits,
            provenance: Provenance::default(),
        }
    }

    fn as_bv(self, bits: u32) -> Self {
        match self {
            Sym::Bv {
                text,
                bits: old_bits,
                provenance,
            } if old_bits == bits => Sym::Bv {
                text,
                bits,
                provenance,
            },
            Sym::Bv {
                text,
                bits: old_bits,
                provenance,
            } if old_bits < bits => Sym::Bv {
                text: format!("((_ zero_extend {}) {})", bits - old_bits, text),
                bits,
                provenance,
            },
            Sym::Bv {
                text,
                bits: old_bits,
                provenance,
            } => Sym::Bv {
                text: format!("((_ extract {} 0) {})", bits - 1, text),
                bits: old_bits.min(bits),
                provenance: provenance.extract(bits - 1, 0),
            }
            .as_bv(bits),
            Sym::Bool { text, provenance } => Sym::Bv {
                text: format!("(ite {} (_ bv1 {}) (_ bv0 {}))", text, bits, bits),
                bits,
                provenance,
            },
        }
    }

    fn as_bool(self) -> Self {
        match self {
            Sym::Bool { text, provenance } => Sym::Bool { text, provenance },
            Sym::Bv {
                text,
                bits,
                provenance,
            } => Sym::Bool {
                text: format!("(not (= {} (_ bv0 {})))", text, bits),
                provenance,
            },
        }
    }

    fn not(self) -> Self {
        match self.as_bool() {
            Sym::Bool { text, provenance } => Sym::Bool {
                text: format!("(not {})", text),
                provenance,
            },
            _ => unreachable!(),
        }
    }

    fn bvshl_const(self, shift: u32) -> Self {
        let bits = self.bits();
        let provenance = self.provenance().shift_left(shift);
        Sym::Bv {
            text: format!(
                "(bvshl {} {})",
                self.bv_text(bits),
                Sym::bv_const(shift as i64, bits).bv_text(bits)
            ),
            bits,
            provenance,
        }
    }

    fn bvor(self, rhs: Sym) -> Self {
        let bits = self.bits().max(rhs.bits());
        let provenance = self.provenance().merge(&rhs.provenance());
        Sym::Bv {
            text: format!("(bvor {} {})", self.bv_text(bits), rhs.bv_text(bits)),
            bits,
            provenance,
        }
    }

    fn extract(self, high: u32, low: u32) -> Self {
        let provenance = self.provenance().extract(high, low);
        Sym::Bv {
            text: format!("((_ extract {} {}) {})", high, low, self.bv_text(high + 1)),
            bits: high - low + 1,
            provenance,
        }
    }

    fn bits(&self) -> u32 {
        match self {
            Sym::Bv { bits, .. } => *bits,
            Sym::Bool { .. } => 1,
        }
    }

    fn provenance(&self) -> Provenance {
        match self {
            Sym::Bv { provenance, .. } | Sym::Bool { provenance, .. } => provenance.clone(),
        }
    }

    fn bv_text(self, bits: u32) -> String {
        match self.as_bv(bits) {
            Sym::Bv { text, .. } => text,
            _ => unreachable!(),
        }
    }
}

impl Provenance {
    fn symbol(name: String, bits: u32) -> Self {
        Self {
            terms: vec![ProvTerm { name, low: 0, bits }],
            overflow: false,
        }
    }

    fn merge(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.overflow |= other.overflow;
        for term in &other.terms {
            if !out.terms.contains(term) {
                if out.terms.len() >= MAX_PROVENANCE_TERMS {
                    out.overflow = true;
                    break;
                }
                out.terms.push(term.clone());
            }
        }
        out
    }

    fn shift_left(&self, shift: u32) -> Self {
        let mut out = self.clone();
        for term in &mut out.terms {
            term.low = term.low.saturating_add(shift);
        }
        out
    }

    fn extract(&self, high: u32, low: u32) -> Self {
        let mut out = Self {
            terms: Vec::new(),
            overflow: self.overflow,
        };
        for term in &self.terms {
            let term_low = term.low;
            let term_high = term.low.saturating_add(term.bits.saturating_sub(1));
            if term_high < low || term_low > high {
                continue;
            }
            let clipped_low = term_low.max(low);
            let clipped_high = term_high.min(high);
            if out.terms.len() >= MAX_PROVENANCE_TERMS {
                out.overflow = true;
                break;
            }
            out.terms.push(ProvTerm {
                name: term.name.clone(),
                low: clipped_low - low,
                bits: clipped_high - clipped_low + 1,
            });
        }
        out
    }
}

fn sym_bool_text(sym: &Sym) -> Option<String> {
    match sym.clone().as_bool() {
        Sym::Bool { text, .. } => Some(text),
        _ => None,
    }
}

fn sym_binop(op: BinOp, lhs: Sym, rhs: Sym) -> Sym {
    let provenance = lhs.provenance().merge(&rhs.provenance());
    match op {
        BinOp::Eq => Sym::Bool {
            text: format!("(= {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            provenance,
        },
        BinOp::Ne => Sym::Bool {
            text: format!("(not (= {} {}))", lhs.bv_text(64), rhs.bv_text(64)),
            provenance,
        },
        BinOp::Lt => Sym::Bool {
            text: format!("(bvslt {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            provenance,
        },
        BinOp::Gt => Sym::Bool {
            text: format!("(bvsgt {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            provenance,
        },
        BinOp::Le => Sym::Bool {
            text: format!("(bvsle {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            provenance,
        },
        BinOp::Ge => Sym::Bool {
            text: format!("(bvsge {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            provenance,
        },
        BinOp::And => {
            let a = lhs.as_bool();
            let b = rhs.as_bool();
            let provenance = a.provenance().merge(&b.provenance());
            Sym::Bool {
                text: format!(
                    "(and {} {})",
                    sym_bool_text(&a).unwrap(),
                    sym_bool_text(&b).unwrap()
                ),
                provenance,
            }
        }
        BinOp::Or => {
            let a = lhs.as_bool();
            let b = rhs.as_bool();
            let provenance = a.provenance().merge(&b.provenance());
            Sym::Bool {
                text: format!(
                    "(or {} {})",
                    sym_bool_text(&a).unwrap(),
                    sym_bool_text(&b).unwrap()
                ),
                provenance,
            }
        }
        BinOp::Implies => {
            let a = lhs.as_bool();
            let b = rhs.as_bool();
            let provenance = a.provenance().merge(&b.provenance());
            Sym::Bool {
                text: format!(
                    "(=> {} {})",
                    sym_bool_text(&a).unwrap(),
                    sym_bool_text(&b).unwrap()
                ),
                provenance,
            }
        }
        BinOp::Iff => {
            let a = lhs.as_bool();
            let b = rhs.as_bool();
            let provenance = a.provenance().merge(&b.provenance());
            Sym::Bool {
                text: format!(
                    "(= {} {})",
                    sym_bool_text(&a).unwrap(),
                    sym_bool_text(&b).unwrap()
                ),
                provenance,
            }
        }
        BinOp::Sub => Sym::Bv {
            text: format!("(bvsub {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            bits: 64,
            provenance,
        },
        BinOp::Mul => Sym::Bv {
            text: format!("(bvmul {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            bits: 64,
            provenance,
        },
        BinOp::Add => Sym::Bv {
            text: format!("(bvadd {} {})", lhs.bv_text(64), rhs.bv_text(64)),
            bits: 64,
            provenance,
        },
    }
}

fn sym_unary_builtin(fn_id: BuiltinFn, x: Sym) -> Sym {
    let provenance = x.provenance();
    match fn_id {
        BuiltinFn::Not { bits } => Sym::Bv {
            text: format!("(bvnot {})", x.bv_text(bits as u32)),
            bits: bits as u32,
            provenance,
        },
        BuiltinFn::Sext { src, dst } => Sym::Bv {
            text: format!("((_ sign_extend {}) {})", dst - src, x.bv_text(src as u32)),
            bits: dst as u32,
            provenance,
        },
        BuiltinFn::Zext { src, dst } => Sym::Bv {
            text: format!("((_ zero_extend {}) {})", dst - src, x.bv_text(src as u32)),
            bits: dst as u32,
            provenance,
        },
        BuiltinFn::Trunc { dst } => x.as_bv(dst as u32),
        BuiltinFn::Bitcast | BuiltinFn::P2i | BuiltinFn::I2p => x.as_bv(64),
        _ => x,
    }
}

fn sym_binary_builtin(fn_id: BuiltinFn, a: Sym, b: Sym) -> Sym {
    match fn_id {
        BuiltinFn::Add { bits } => bv2("bvadd", a, b, bits),
        BuiltinFn::Sub { bits } => bv2("bvsub", a, b, bits),
        BuiltinFn::Mul { bits } => bv2("bvmul", a, b, bits),
        BuiltinFn::And { bits } => bv2("bvand", a, b, bits),
        BuiltinFn::Or { bits } => bv2("bvor", a, b, bits),
        BuiltinFn::Xor { bits } => bv2("bvxor", a, b, bits),
        BuiltinFn::Shl { bits } => bv2("bvshl", a, b, bits),
        BuiltinFn::Lshr { bits } => bv2("bvlshr", a, b, bits),
        BuiltinFn::Ashr { bits } => bv2("bvashr", a, b, bits),
        BuiltinFn::Udiv { bits } => bv2("bvudiv", a, b, bits),
        BuiltinFn::Sdiv { bits } => bv2("bvsdiv", a, b, bits),
        BuiltinFn::Urem { bits } => bv2("bvurem", a, b, bits),
        BuiltinFn::Srem { bits } => bv2("bvsrem", a, b, bits),
        BuiltinFn::Ult { bits } => bool_as_bv(cmp("bvult", a, b, bits), 64),
        BuiltinFn::Ule { bits } => bool_as_bv(cmp("bvule", a, b, bits), 64),
        BuiltinFn::Ugt { bits } => bool_as_bv(cmp("bvugt", a, b, bits), 64),
        BuiltinFn::Uge { bits } => bool_as_bv(cmp("bvuge", a, b, bits), 64),
        BuiltinFn::Slt { bits } => bool_as_bv(cmp("bvslt", a, b, bits), 64),
        BuiltinFn::Sle { bits } => bool_as_bv(cmp("bvsle", a, b, bits), 64),
        BuiltinFn::Sgt { bits } => bool_as_bv(cmp("bvsgt", a, b, bits), 64),
        BuiltinFn::Sge { bits } => bool_as_bv(cmp("bvsge", a, b, bits), 64),
        BuiltinFn::BvEq { bits } => bool_as_bv(eq(a, b, bits), 64),
        BuiltinFn::BvNe { bits } => bool_as_bv(eq(a, b, bits).not(), 64),
        BuiltinFn::SltBool { bits } => cmp("bvslt", a, b, bits),
        BuiltinFn::SleBool { bits } => cmp("bvsle", a, b, bits),
        BuiltinFn::SgtBool { bits } => cmp("bvsgt", a, b, bits),
        BuiltinFn::SgeBool { bits } => cmp("bvsge", a, b, bits),
        _ => a,
    }
}

fn bv2(op: &str, a: Sym, b: Sym, bits: u8) -> Sym {
    let bits = bits as u32;
    let provenance = a.provenance().merge(&b.provenance());
    Sym::Bv {
        text: format!("({} {} {})", op, a.bv_text(bits), b.bv_text(bits)),
        bits,
        provenance,
    }
}

fn cmp(op: &str, a: Sym, b: Sym, bits: u8) -> Sym {
    let bits = bits as u32;
    let provenance = a.provenance().merge(&b.provenance());
    Sym::Bool {
        text: format!("({} {} {})", op, a.bv_text(bits), b.bv_text(bits)),
        provenance,
    }
}

fn eq(a: Sym, b: Sym, bits: u8) -> Sym {
    let bits = bits as u32;
    let provenance = a.provenance().merge(&b.provenance());
    Sym::Bool {
        text: format!("(= {} {})", a.bv_text(bits), b.bv_text(bits)),
        provenance,
    }
}

fn bool_as_bv(sym: Sym, bits: u32) -> Sym {
    let provenance = sym.provenance();
    let text = bool_text(sym);
    Sym::Bv {
        text: format!("(ite {} (_ bv1 {}) (_ bv0 {}))", text, bits, bits),
        bits,
        provenance,
    }
}

fn bool_text(sym: Sym) -> String {
    match sym.as_bool() {
        Sym::Bool { text, .. } => text,
        _ => unreachable!(),
    }
}

fn direct_symbol(sym: &Sym) -> Option<(String, u32)> {
    match sym {
        Sym::Bv { text, bits, .. } if text.starts_with('s') => {
            if text[1..].chars().all(|c| c.is_ascii_digit()) {
                Some((text.clone(), *bits))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn invert_cmp_op(op: BinOp) -> BinOp {
    match op {
        BinOp::Lt => BinOp::Gt,
        BinOp::Gt => BinOp::Lt,
        BinOp::Le => BinOp::Ge,
        BinOp::Ge => BinOp::Le,
        other => other,
    }
}

fn cmp_profile_values(op: BinOp, constant: i64) -> Vec<i64> {
    match op {
        BinOp::Eq | BinOp::Iff => vec![constant, constant.wrapping_add(1)],
        BinOp::Ne => vec![constant, constant.wrapping_add(1)],
        BinOp::Lt => vec![constant.wrapping_sub(1), constant, constant.wrapping_add(1)],
        BinOp::Le => vec![constant, constant.wrapping_add(1)],
        BinOp::Gt => vec![constant.wrapping_add(1), constant],
        BinOp::Ge => vec![constant, constant.wrapping_sub(1)],
        _ => Vec::new(),
    }
}

fn mask_to_bits(value: i64, bits: u32) -> u64 {
    let raw = value as u64;
    if bits >= 64 {
        raw
    } else {
        raw & ((1u64 << bits) - 1)
    }
}

fn mask_u128(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn signed_value(value: i64, bits: u32) -> i128 {
    let bits = bits.clamp(1, 64);
    let raw = (value as u64 as u128) & mask_u128(bits);
    if bits == 64 {
        (raw as u64 as i64) as i128
    } else {
        let sign = 1u128 << (bits - 1);
        if raw & sign != 0 {
            raw as i128 - (1i128 << bits)
        } else {
            raw as i128
        }
    }
}

fn unsigned_value(value: i64, bits: u32) -> u128 {
    (value as u64 as u128) & mask_u128(bits.clamp(1, 64))
}

fn branch_distance(op: BinOp, lhs: i64, rhs: i64, bits: u32, signed: bool) -> u128 {
    match op {
        BinOp::Eq | BinOp::Iff => unsigned_value(lhs.wrapping_sub(rhs), bits)
            .min(unsigned_value(rhs.wrapping_sub(lhs), bits)),
        BinOp::Ne => {
            if unsigned_value(lhs, bits) != unsigned_value(rhs, bits) {
                0
            } else {
                1
            }
        }
        BinOp::Lt => {
            if signed {
                let l = signed_value(lhs, bits);
                let r = signed_value(rhs, bits);
                if l < r {
                    0
                } else {
                    (l - r + 1) as u128
                }
            } else {
                let l = unsigned_value(lhs, bits);
                let r = unsigned_value(rhs, bits);
                if l < r {
                    0
                } else {
                    l - r + 1
                }
            }
        }
        BinOp::Le => {
            if signed {
                let l = signed_value(lhs, bits);
                let r = signed_value(rhs, bits);
                if l <= r {
                    0
                } else {
                    (l - r) as u128
                }
            } else {
                let l = unsigned_value(lhs, bits);
                let r = unsigned_value(rhs, bits);
                if l <= r {
                    0
                } else {
                    l - r
                }
            }
        }
        BinOp::Gt => branch_distance(BinOp::Lt, rhs, lhs, bits, signed),
        BinOp::Ge => branch_distance(BinOp::Le, rhs, lhs, bits, signed),
        _ => 0,
    }
}

fn builtin_cmp_profile(fn_id: BuiltinFn) -> Option<(BinOp, u32, bool)> {
    match fn_id {
        BuiltinFn::BvEq { bits } => Some((BinOp::Eq, bits as u32, false)),
        BuiltinFn::BvNe { bits } => Some((BinOp::Ne, bits as u32, false)),
        BuiltinFn::Slt { bits } | BuiltinFn::SltBool { bits } => {
            Some((BinOp::Lt, bits as u32, true))
        }
        BuiltinFn::Sle { bits } | BuiltinFn::SleBool { bits } => {
            Some((BinOp::Le, bits as u32, true))
        }
        BuiltinFn::Sgt { bits } | BuiltinFn::SgtBool { bits } => {
            Some((BinOp::Gt, bits as u32, true))
        }
        BuiltinFn::Sge { bits } | BuiltinFn::SgeBool { bits } => {
            Some((BinOp::Ge, bits as u32, true))
        }
        BuiltinFn::Ult { bits } => Some((BinOp::Lt, bits as u32, false)),
        BuiltinFn::Ule { bits } => Some((BinOp::Le, bits as u32, false)),
        BuiltinFn::Ugt { bits } => Some((BinOp::Gt, bits as u32, false)),
        BuiltinFn::Uge { bits } => Some((BinOp::Ge, bits as u32, false)),
        _ => None,
    }
}

#[derive(Clone)]
enum SolveResult {
    Sat(Vec<(String, u64)>),
    Unsat,
    Unknown,
    Error,
}

struct SolverProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    query_index: usize,
    symbols: Vec<String>,
}

impl SolverProcess {
    fn new(symbols: &[InputSymbol], timeout_ms: u64) -> Option<Self> {
        if symbols.is_empty() {
            return None;
        }
        let mut child = Command::new("z3")
            .arg("-in")
            .arg("-smt2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;

        if writeln!(stdin, "(set-logic QF_BV)").is_err()
            || writeln!(stdin, "(set-option :produce-models true)").is_err()
            || writeln!(stdin, "(set-option :timeout {})", timeout_ms).is_err()
        {
            return None;
        }
        for s in symbols {
            if writeln!(stdin, "(declare-fun {} () (_ BitVec {}))", s.name, s.bits).is_err() {
                return None;
            }
        }
        if stdin.flush().is_err() {
            return None;
        }

        Some(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            query_index: 0,
            symbols: symbols.iter().map(|s| s.name.clone()).collect(),
        })
    }

    fn query(&mut self, constraints: &[String]) -> SolveResult {
        self.query_index = self.query_index.wrapping_add(1);
        let marker = format!("__SWOOSH_SOLVER_END_{}__", self.query_index);
        if writeln!(self.stdin, "(push 1)").is_err() {
            return SolveResult::Error;
        }
        for c in constraints {
            if writeln!(self.stdin, "(assert {})", c).is_err() {
                return SolveResult::Error;
            }
        }
        if writeln!(self.stdin, "(check-sat)").is_err() {
            return SolveResult::Error;
        }
        if write!(self.stdin, "(get-value (").is_err() {
            return SolveResult::Error;
        }
        for (idx, name) in self.symbols.iter().enumerate() {
            if idx > 0 && write!(self.stdin, " ").is_err() {
                return SolveResult::Error;
            }
            if write!(self.stdin, "{}", name).is_err() {
                return SolveResult::Error;
            }
        }
        if writeln!(self.stdin, "))").is_err()
            || writeln!(self.stdin, "(pop 1)").is_err()
            || writeln!(self.stdin, "(echo \"{}\")", marker).is_err()
            || self.stdin.flush().is_err()
        {
            return SolveResult::Error;
        }

        let mut text = String::new();
        loop {
            let mut line = String::new();
            match self.stdout.read_line(&mut line) {
                Ok(0) => return SolveResult::Error,
                Ok(_) => {
                    if line.trim() == marker {
                        break;
                    }
                    text.push_str(&line);
                }
                Err(_) => return SolveResult::Error,
            }
        }
        parse_solve_output(&text)
    }
}

impl Drop for SolverProcess {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "(exit)");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn solve(symbols: &[InputSymbol], constraints: &[String], timeout_ms: u64) -> SolveResult {
    if symbols.is_empty() {
        return SolveResult::Unsat;
    }
    let mut smt = String::new();
    smt.push_str("(set-logic QF_BV)\n");
    smt.push_str("(set-option :produce-models true)\n");
    smt.push_str(&format!("(set-option :timeout {})\n", timeout_ms));
    for s in symbols {
        smt.push_str(&format!(
            "(declare-fun {} () (_ BitVec {}))\n",
            s.name, s.bits
        ));
    }
    for c in constraints {
        smt.push_str(&format!("(assert {})\n", c));
    }
    smt.push_str("(check-sat)\n");
    smt.push_str("(get-value (");
    for (idx, s) in symbols.iter().enumerate() {
        if idx > 0 {
            smt.push(' ');
        }
        smt.push_str(&s.name);
    }
    smt.push_str("))\n");

    let mut child = match Command::new("z3")
        .arg("-in")
        .arg("-smt2")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return SolveResult::Error,
    };
    if let Some(mut stdin) = child.stdin.take() {
        if stdin.write_all(smt.as_bytes()).is_err() {
            return SolveResult::Error;
        }
    }
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(_) => return SolveResult::Error,
    };
    let text = String::from_utf8_lossy(&output.stdout);
    parse_solve_output(&text)
}

fn solve_with_backend(
    solver: &mut Option<SolverProcess>,
    symbols: &[InputSymbol],
    constraints: &[String],
    timeout_ms: u64,
) -> SolveResult {
    if solver.is_none() {
        *solver = SolverProcess::new(symbols, timeout_ms);
    }
    if let Some(process) = solver.as_mut() {
        let result = process.query(constraints);
        if !matches!(result, SolveResult::Error) {
            return result;
        }
    }
    *solver = None;
    solve(symbols, constraints, timeout_ms)
}

fn parse_solve_output(text: &str) -> SolveResult {
    let first = text.lines().next().unwrap_or("").trim();
    match first {
        "sat" => SolveResult::Sat(parse_model(&text)),
        "unsat" => SolveResult::Unsat,
        "unknown" => SolveResult::Unknown,
        _ => SolveResult::Error,
    }
}

fn solve_cache_key(constraints: &[String]) -> String {
    constraints.join("\n")
}

fn per_sec(count: usize, elapsed: std::time::Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return 0.0;
    }
    count as f64 / seconds
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

fn update_signature(updates: &[(String, u64)]) -> String {
    let mut pairs: Vec<(&str, u64)> = updates
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    for (name, value) in pairs {
        out.push_str(name);
        out.push('=');
        out.push_str(&value.to_string());
        out.push(';');
    }
    out
}

fn apply_updates_to_vm(vm: &mut VM, symbols: &[InputSymbol], updates: &[(String, u64)]) {
    let mut by_name = FxHashMap::default();
    for (name, value) in updates {
        by_name.insert(name.as_str(), *value);
    }

    for sym in symbols {
        let Some(value) = by_name.get(sym.name.as_str()).copied() else {
            continue;
        };
        let signed = value as i64;
        if let Some(var_id) = sym.var_id {
            vm.set_scalar(var_id, signed, true);
        }
        if let Some((ref map_name, addr)) = sym.map_key {
            if let Some(map_idx) = vm.memory_maps.iter().position(|m| &m.name == map_name) {
                vm.memory_maps[map_idx].set(addr, signed);
            }
        }
        if let Some(idx) = sym.extra_index {
            if idx < vm.external_buffer.len() {
                vm.external_buffer[idx] = (value & 0xff) as u8;
            }
        }
        if let Some((var_id, havoc_idx)) = sym.havoc_key {
            vm.set_havoc_value_at(var_id, havoc_idx, signed);
        }
    }
}

fn apply_updates_to_symbols(symbols: &mut [InputSymbol], updates: &[(String, u64)]) {
    let mut by_name = FxHashMap::default();
    for (name, value) in updates {
        by_name.insert(name.as_str(), *value);
    }
    for sym in symbols {
        if let Some(value) = by_name.get(sym.name.as_str()).copied() {
            sym.value = value;
        }
    }
}

fn parse_model(text: &str) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let cleaned = text.replace('(', " ").replace(')', " ");
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    let mut i = 0usize;
    while i + 1 < tokens.len() {
        let name = tokens[i];
        let val = tokens[i + 1];
        if name.starts_with('s') {
            if let Some(v) = parse_smt_bv(val, tokens.get(i + 2).copied()) {
                out.push((name.to_string(), v));
            }
        }
        i += 1;
    }
    out
}

fn parse_smt_bv(token: &str, next: Option<&str>) -> Option<u64> {
    if let Some(hex) = token.strip_prefix("#x") {
        return u64::from_str_radix(hex, 16).ok();
    }
    if let Some(bin) = token.strip_prefix("#b") {
        return u64::from_str_radix(bin, 2).ok();
    }
    if token == "_" {
        return next
            .and_then(|n| n.strip_prefix("bv"))
            .and_then(|n| n.parse().ok());
    }
    token.strip_prefix("bv").and_then(|n| n.parse().ok())
}

fn parse_symbols(
    symbols_py: &Bound<'_, PyList>,
    program: &CompiledProgram,
) -> PyResult<Vec<InputSymbol>> {
    let mut out = Vec::new();
    for item in symbols_py.iter() {
        let d: &Bound<'_, PyDict> = item.downcast()?;
        let name: String = required(d, "name")?.extract()?;
        let bits: u32 = required(d, "bits")?.extract()?;
        let value: u64 = required(d, "value")?.extract()?;
        let var_id = optional_string(d, "var")?.and_then(|v| program.name_to_var.get(&v).copied());
        let map_key = match (optional_string(d, "map")?, optional_i64(d, "addr")?) {
            (Some(m), Some(a)) => Some((m, a)),
            _ => None,
        };
        let extra_index = optional_usize(d, "extra_index")?;
        let havoc_key = match (
            optional_string(d, "havoc_var")?,
            optional_usize(d, "havoc_index")?,
        ) {
            (Some(v), Some(i)) => program.name_to_var.get(&v).copied().map(|vid| (vid, i)),
            _ => None,
        };
        out.push(InputSymbol {
            name,
            bits,
            value,
            var_id,
            map_key,
            extra_index,
            havoc_key,
        });
    }
    Ok(out)
}

fn parse_covered(obj: Option<&Bound<'_, PyAny>>) -> PyResult<FxHashSet<String>> {
    let mut out = FxHashSet::default();
    if let Some(obj) = obj {
        if let Ok(set) = obj.downcast::<PySet>() {
            for item in set.iter() {
                out.insert(item.extract()?);
            }
        } else if let Ok(list) = obj.downcast::<PyList>() {
            for item in list.iter() {
                out.insert(item.extract()?);
            }
        }
    }
    Ok(out)
}

fn build_vm(
    program: &CompiledProgram,
    var_store: &Bound<'_, PyDict>,
    memory_maps: &Bound<'_, PyDict>,
    mem_map_info: &Bound<'_, PyList>,
    extra_data: Option<Vec<u8>>,
) -> PyResult<VM> {
    let mut vm = VM::new_no_trace(program);
    vm.log_read = false;
    if let Some(data) = extra_data {
        vm.external_buffer = data;
    }
    for item in mem_map_info.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let name: String = tuple.get_item(0)?.extract()?;
        let index_bw: u8 = tuple.get_item(1)?.extract()?;
        let element_bw: u8 = tuple.get_item(2)?.extract()?;
        if let Some(&vid) = program.name_to_var.get(&name) {
            vm.init_memory_map(vid, name, index_bw, element_bw);
        }
    }
    for (key, val) in var_store.iter() {
        let name: String = key.extract()?;
        let value: i64 = val.extract()?;
        if let Some(&vid) = program.name_to_var.get(&name) {
            vm.set_scalar(vid, value, true);
        }
    }
    for (key, val) in memory_maps.iter() {
        let name: String = key.extract()?;
        let contents: &Bound<'_, PyDict> = val.downcast()?;
        if let Some(&vid) = program.name_to_var.get(&name) {
            if let Some(map_idx) = vm.var_to_map[vid as usize] {
                for (addr, value) in contents.iter() {
                    vm.memory_maps[map_idx].set(addr.extract()?, value.extract()?);
                }
            }
        }
    }
    Ok(vm)
}

fn required<'a>(d: &'a Bound<'_, PyDict>, key: &str) -> PyResult<Bound<'a, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
}

fn optional_string(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    Ok(d.get_item(key)?.map(|v| v.extract()).transpose()?)
}

fn optional_i64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    Ok(d.get_item(key)?.map(|v| v.extract()).transpose()?)
}

fn optional_usize(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    Ok(d.get_item(key)?.map(|v| v.extract()).transpose()?)
}

fn eval_to_i64(value: &EvalResult) -> i64 {
    match value {
        EvalResult::Scalar(v) => *v,
        EvalResult::Bool(b) => *b as i64,
        EvalResult::MapRef(_) => 0,
    }
}

fn scalar_value(value: &EvalResult) -> Option<i64> {
    match value {
        EvalResult::Scalar(v) => Some(*v),
        EvalResult::Bool(b) => Some(*b as i64),
        EvalResult::MapRef(_) => None,
    }
}
