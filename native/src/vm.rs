use crate::builtins;
use crate::builtins::int::{Z, ZResult};
use crate::memory_map::MemoryMap;
use crate::opcodes::*;
use crate::trace::{TraceAccumulator, OP_READ, OP_WRITE};
use num_bigint::BigInt;

const MASK_64: i64 = -1i64; // all bits set = u64::MAX as i64

/// Surface symbol for a Boogie binary operator (for failing-assume messages).
fn binop_symbol(op: &BinOp) -> &'static str {
    match op {
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::Implies => "==>",
        BinOp::Iff => "<==>",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Add => "+",
    }
}

/// Runtime value — a scalar, an out-of-i64 exact integer (Int mode only),
/// or a memory map index.
#[derive(Debug, Clone)]
pub enum Value {
    Scalar(i64),
    /// Exact-ℤ value strictly outside i64 range. Only `SemanticsMode::Int`
    /// evaluation produces these (BV mode is a closed i64 algebra).
    Big(Box<BigInt>),
    Map(usize), // index into VM.memory_maps
}

/// Structured VM stop reason for expected Boogie-level concrete execution
/// outcomes. Internal VM errors still use panic/assert so callers can tell
/// them apart from useful failing inputs.
#[derive(Debug, Clone)]
pub enum ExecutionStatus {
    Completed,
    AssertViolation {
        pc: u32,
        block: String,
    },
    AssumeViolation {
        pc: u32,
        block: String,
        reason: &'static str,
        /// Human-readable description of WHICH assume failed and why — the
        /// rendered condition plus the concrete values of the scalar variables
        /// it references (e.g. `($i3 >= 0)  [where $i3=-1]`). Empty when no
        /// expression is available (e.g. an infeasible goto with no candidate
        /// guard). Surfaced to Python as `invalid_detail` so an agent fixing a
        /// stale input knows exactly which precondition the input violates.
        detail: String,
    },
    StepLimit {
        pc: u32,
        block: String,
    },
}

/// The virtual machine that executes compiled Boogie programs.
#[derive(Clone)]
pub struct VM {
    /// Variable store: VarId → Value
    pub vars: Vec<Value>,
    /// Variable names for trace output
    pub var_names: Vec<String>,
    /// Memory maps, indexed by map_index in Value::Map
    pub memory_maps: Vec<MemoryMap>,
    /// Which VarId is a memory map? VarId → Some(map_index)
    pub var_to_map: Vec<Option<usize>>,
    /// Current PC
    pub pc: u32,
    /// Current block ID (names are materialized only on error paths and at
    /// the Python boundary — see `block_name`)
    pub curr_block_id: u32,
    /// Explored-block bitset, indexed by BlockId (parallel to program.blocks).
    pub explored_blocks: Vec<bool>,
    /// Number of distinct blocks explored (popcount of `explored_blocks`).
    pub explored_count: usize,
    /// Ordered block entries for lightweight path/edge coverage. Only
    /// recorded when `record_block_trace` is set; `block_entries` keeps the
    /// total count either way.
    pub block_trace: Vec<BlockId>,
    /// Whether to record the ordered `block_trace` (returned to Python as
    /// `block_sequence`). Long runs enter millions of blocks — callers that
    /// don't need the sequence turn this off.
    pub record_block_trace: bool,
    /// Total number of block entries (what `block_trace.len()` would be).
    pub block_entries: u64,
    /// Compact trace accumulator
    pub trace: TraceAccumulator,
    /// Whether to log reads
    pub log_read: bool,
    /// Whether to skip all tracing (for benchmarking)
    pub no_trace: bool,
    /// Allocation addresses
    pub alloc_addr: i64,
    pub alloc_addr_shadow: i64,
    /// VarId for $CurrAddr / $CurrAddr.shadow
    pub curr_addr_id: Option<VarId>,
    pub curr_addr_shadow_id: Option<VarId>,
    /// wlen_buf for write.cross_product
    pub wlen_buf: [i64; 6],
    pub wlen_buf_idx: usize,
    /// External read buffer
    pub external_buffer: Vec<u8>,
    pub external_buffer_pos: usize,
    /// VarId for $M.0 and $M.0.shadow (for read.cross_product)
    pub m0_id: Option<VarId>,
    pub m0_shadow_id: Option<VarId>,
    /// Per-variable nondet schedules loaded from .input int_seq entries.
    pub havoc_sequences: Vec<Option<Vec<i64>>>,
    pub havoc_counts: Vec<usize>,
    /// Reusable RHS-evaluation buffer for AssignN (avoids a per-statement Vec).
    scratch_evals: Vec<EvalResult>,
    /// Trace records SKIPPED because the value was an out-of-i64 exact
    /// integer (Int mode). The raw-log value field is i64; recording a
    /// wrapped/clamped stand-in could manufacture false trace-refutations
    /// downstream, so the escape is to drop the record and count it.
    pub big_trace_skips: u64,
    /// Out-of-i64 exact values folded mod 2^64 at the MEMORY interface
    /// (Int mode). SMACK emits negative pointer offsets as u64
    /// two's-complement literals (`p + 18446744073709551615` ≡ `p - 1`),
    /// so exact-ℤ address chains can leave i64; the memory map is a
    /// 64-bit-address store, and folding once at the boundary is
    /// congruent (mod 2^64) with the historical per-op wrap for +,-,*.
    pub mem_big_folds: u64,
}

/// Two's-complement fold of an exact integer into the 64-bit address/value
/// space of the memory map: `v mod 2^64`, read back as i64.
#[inline]
fn fold_big_to_i64(b: &BigInt) -> i64 {
    // Allocation-free: v mod 2^64 == low 64 bits of the magnitude, negated
    // (wrapping) for negative values.
    let low = b.iter_u64_digits().next().unwrap_or(0) as i64;
    match b.sign() {
        num_bigint::Sign::Minus => low.wrapping_neg(),
        _ => low,
    }
}

impl VM {
    pub fn new(program: &CompiledProgram) -> Self {
        Self::new_with_trace(program, true)
    }

    pub fn new_no_trace(program: &CompiledProgram) -> Self {
        Self::new_with_trace(program, false)
    }

    fn new_with_trace(program: &CompiledProgram, trace_enabled: bool) -> Self {
        let n = program.num_vars as usize;
        let vars = vec![Value::Scalar(0); n];
        let var_to_map = vec![None; n];
        let mut trace = TraceAccumulator::new();
        if trace_enabled {
            // Install loop metadata so packed iter_id emission works.  The
            // vectors are parallel to `program.blocks` and already include
            // the non-loop defaults (all false / None) when the compile
            // pipeline didn't pass metadata.
            trace.set_loop_metadata(
                program.is_loop_header.clone(),
                program.block_innermost_header.clone(),
                program.loop_parent_header.clone(),
            );
        }
        Self {
            vars,
            var_names: program.var_names.clone(),
            memory_maps: Vec::new(),
            var_to_map,
            pc: 0,
            curr_block_id: 0,
            explored_blocks: vec![false; program.blocks.len()],
            explored_count: 0,
            block_trace: Vec::new(),
            record_block_trace: true,
            block_entries: 0,
            trace,
            log_read: true,
            alloc_addr: 0,
            alloc_addr_shadow: 0,
            curr_addr_id: program.curr_addr_id,
            curr_addr_shadow_id: program.curr_addr_shadow_id,
            wlen_buf: [209, 42, 6, 37, 51, 23],
            wlen_buf_idx: 0,
            external_buffer: Vec::new(),
            external_buffer_pos: 0,
            m0_id: program.m0_id,
            m0_shadow_id: program.m0_shadow_id,
            no_trace: !trace_enabled,
            havoc_sequences: vec![None; n],
            havoc_counts: vec![0; n],
            scratch_evals: Vec::new(),
            big_trace_skips: 0,
            mem_big_folds: 0,
        }
    }

    /// Materialize the current block's name. Only used on error paths and at
    /// the Python boundary — the hot loop tracks `curr_block_id` only.
    #[inline]
    pub fn block_name(&self, program: &CompiledProgram) -> String {
        program
            .blocks
            .get(self.curr_block_id as usize)
            .map(|b| b.name.clone())
            .unwrap_or_default()
    }

    /// Mark a block as explored (bitset + distinct count).
    #[inline]
    pub fn mark_explored(&mut self, block_id: BlockId) {
        let idx = block_id as usize;
        if idx >= self.explored_blocks.len() {
            self.explored_blocks.resize(idx + 1, false);
        }
        if !self.explored_blocks[idx] {
            self.explored_blocks[idx] = true;
            self.explored_count += 1;
        }
    }

    pub fn set_havoc_sequence(&mut self, var_id: VarId, seq: Vec<i64>) {
        let vid = var_id as usize;
        if vid >= self.havoc_sequences.len() {
            return;
        }
        self.havoc_sequences[vid] = Some(seq);
        self.havoc_counts[vid] = 0;
    }

    #[inline]
    pub fn havoc_count(&self, var_id: VarId) -> usize {
        self.havoc_counts.get(var_id as usize).copied().unwrap_or(0)
    }

    pub fn set_havoc_value_at(&mut self, var_id: VarId, idx: usize, value: i64) {
        let vid = var_id as usize;
        if vid >= self.havoc_sequences.len() {
            return;
        }
        let seq = self.havoc_sequences[vid].get_or_insert_with(Vec::new);
        if seq.len() <= idx {
            seq.resize(idx + 1, 0);
        }
        seq[idx] = value;
        self.havoc_counts[vid] = 0;
    }

    #[inline]
    pub fn next_havoc_value(&mut self, var_id: VarId) -> i64 {
        let vid = var_id as usize;
        let Some(count) = self.havoc_counts.get_mut(vid) else {
            return 0;
        };
        let value = self
            .havoc_sequences
            .get(vid)
            .and_then(|seq| seq.as_ref())
            .and_then(|seq| seq.get(*count).copied())
            .unwrap_or(0);
        *count += 1;
        value
    }

    /// Initialize a memory map variable.
    pub fn init_memory_map(&mut self, var_id: VarId, name: String, index_bw: u8, element_bw: u8) {
        let map = MemoryMap::new(name, index_bw, element_bw);
        let idx = self.memory_maps.len();
        self.memory_maps.push(map);
        self.var_to_map[var_id as usize] = Some(idx);
        self.vars[var_id as usize] = Value::Map(idx);
    }

    /// Set a scalar variable.
    #[inline]
    pub fn set_scalar(&mut self, var_id: VarId, value: i64, silent: bool) {
        let vid = var_id as usize;
        // Track $CurrAddr
        if Some(var_id) == self.curr_addr_id {
            self.alloc_addr = value;
        } else if Some(var_id) == self.curr_addr_shadow_id {
            self.alloc_addr_shadow = value;
        }
        if !self.no_trace && !silent {
            self.trace
                .record(var_id, value, self.pc, self.curr_block_id, OP_WRITE);
        }
        self.vars[vid] = Value::Scalar(value);
    }

    /// Store an out-of-i64 exact integer (Int mode). No trace record is
    /// emitted — see `big_trace_skips` for why the escape is a counted skip.
    #[inline]
    pub fn set_big(&mut self, var_id: VarId, value: Box<BigInt>) {
        if Some(var_id) == self.curr_addr_id || Some(var_id) == self.curr_addr_shadow_id {
            panic!(
                "exact-int overflow escape: allocation cursor {} left i64 range ({})",
                self.var_names[var_id as usize], value
            );
        }
        if !self.no_trace {
            self.big_trace_skips += 1;
        }
        self.vars[var_id as usize] = Value::Big(value);
    }

    /// Memory-interface read of a variable: like `get_scalar`, but an
    /// out-of-i64 exact value folds mod 2^64 (counted) instead of panicking
    /// — used by the memcpy/memset/read handlers whose operands are
    /// 64-bit addresses/lengths by construction.
    #[inline]
    fn get_scalar_mem(&mut self, var_id: VarId) -> i64 {
        if let Value::Big(b) = &self.vars[var_id as usize] {
            let folded = fold_big_to_i64(b);
            self.mem_big_folds += 1;
            return folded;
        }
        self.get_scalar(var_id)
    }

    /// Get a scalar variable value, with optional read tracing.
    #[inline]
    pub fn get_scalar(&mut self, var_id: VarId) -> i64 {
        let vid = var_id as usize;
        match &self.vars[vid] {
            Value::Scalar(v) => {
                let v = *v;
                if !self.no_trace && self.log_read {
                    self.trace
                        .record(var_id, v, self.pc, self.curr_block_id, OP_READ);
                }
                v
            }
            Value::Big(b) => panic!(
                "exact-int overflow escape: {} = {} is outside i64 in an \
                 i64-only context (memory/handoff)",
                self.var_names[vid], b
            ),
            Value::Map(_) => panic!(
                "get_scalar called on memory map variable: {}",
                self.var_names[vid]
            ),
        }
    }

    /// Get a scalar variable value without tracing.
    #[inline]
    pub fn get_scalar_silent(&self, var_id: VarId) -> i64 {
        match &self.vars[var_id as usize] {
            Value::Scalar(v) => *v,
            Value::Big(b) => panic!(
                "exact-int overflow escape: {} = {} is outside i64 in an \
                 i64-only context (memory/handoff)",
                self.var_names[var_id as usize], b
            ),
            Value::Map(_) => panic!(
                "get_scalar_silent called on memory map variable: {}",
                self.var_names[var_id as usize]
            ),
        }
    }

    /// Get the map index for a variable.
    #[inline]
    fn get_map_idx(&self, var_id: VarId) -> usize {
        match &self.vars[var_id as usize] {
            Value::Map(idx) => *idx,
            _ => panic!(
                "get_map_idx called on non-map variable: {}",
                self.var_names[var_id as usize]
            ),
        }
    }

    /// Clear a variable (remove from store or clear its map).
    pub fn clear_var(&mut self, var_id: VarId) {
        let vid = var_id as usize;
        if let Some(map_idx) = self.var_to_map[vid] {
            self.memory_maps[map_idx].clear();
        } else {
            self.vars[vid] = Value::Scalar(0);
        }
    }

    /// Read n bytes from the external buffer.
    fn read_external(&mut self, n: usize) -> Vec<u8> {
        let end = (self.external_buffer_pos + n).min(self.external_buffer.len());
        let start = self.external_buffer_pos;
        self.external_buffer_pos = end;
        self.external_buffer[start..end].to_vec()
    }

    fn memmove_i8_maps(&mut self, dst: i64, dst_shadow: i64, src: i64, src_shadow: i64, len: i64) {
        if len <= 0 {
            return;
        }
        for map_idx in 0..self.memory_maps.len() {
            if self.memory_maps[map_idx].element_bit_width != 8 {
                continue;
            }
            let (dst_base, src_base) = if self.memory_maps[map_idx].is_shadow {
                (dst_shadow, src_shadow)
            } else {
                (dst, src)
            };
            self.memory_maps[map_idx].move_range(src_base, dst_base, len);
        }
    }

    /// Execute the program starting from the entry block.
    pub fn execute(&mut self, program: &CompiledProgram) -> ExecutionStatus {
        self.execute_with_limit(program, 0)
    }

    /// Execute with an optional instruction/block-entry budget.
    ///
    /// ``max_steps == 0`` keeps the historical unbounded behavior.  Corpus and
    /// benchmark harnesses pass a finite budget so intentionally nonterminating
    /// inputs become a structured result instead of hanging the process.
    pub fn execute_with_limit(
        &mut self,
        program: &CompiledProgram,
        max_steps: usize,
    ) -> ExecutionStatus {
        let mut block_id = program.entry_block;
        let mut steps = 0usize;
        if let Some(status) = self.check_entry_preconditions(program) {
            return status;
        }

        loop {
            let block = &program.blocks[block_id as usize];
            self.mark_explored(block_id);
            self.block_entries += 1;
            if self.record_block_trace {
                self.block_trace.push(block_id);
            }
            self.curr_block_id = block_id;
            self.pc = block.start_pc;
            // Let the trace accumulator update its loop stack for this
            // block entry — drives the packed iter_id semantics.
            if !self.no_trace {
                self.trace.on_block_enter(block_id);
            }
            if let Some(status) = self.consume_step(&mut steps, max_steps, program) {
                return status;
            }

            // Execute body statements
            for stmt in &block.body {
                if let Some(status) = self.consume_step(&mut steps, max_steps, program) {
                    return status;
                }
                if let Err(status) = self.execute_stmt(stmt, program) {
                    return status;
                }
                self.pc += 1;
            }

            // Handle terminator
            match &block.terminator {
                Stmt::Return => return ExecutionStatus::Completed,
                Stmt::Goto { targets } => {
                    if targets.len() == 1 {
                        block_id = targets[0];
                    } else {
                        match self.resolve_branch(targets, program) {
                            Ok(next) => block_id = next,
                            Err(status) => return status,
                        }
                    }
                }
                _ => panic!("Block terminator must be Goto or Return"),
            }
        }
    }

    #[inline(always)]
    fn consume_step(
        &self,
        steps: &mut usize,
        max_steps: usize,
        program: &CompiledProgram,
    ) -> Option<ExecutionStatus> {
        if max_steps == 0 {
            return None;
        }
        if *steps >= max_steps {
            return Some(ExecutionStatus::StepLimit {
                pc: self.pc,
                block: self.block_name(program),
            });
        }
        *steps += 1;
        None
    }

    fn check_entry_preconditions(&mut self, program: &CompiledProgram) -> Option<ExecutionStatus> {
        if program.entry_preconditions.is_empty() {
            return None;
        }
        let entry = &program.blocks[program.entry_block as usize];
        self.curr_block_id = program.entry_block;
        self.pc = entry.start_pc;
        for expr in &program.entry_preconditions {
            if !self.eval_bool(expr, program) {
                let detail = self.describe_assume_expr(expr, program);
                return Some(ExecutionStatus::AssumeViolation {
                    pc: self.pc,
                    block: self.block_name(program),
                    reason: "requires",
                    detail,
                });
            }
        }
        None
    }

    /// Resolve a multi-target goto by evaluating assume conditions.
    fn resolve_branch(
        &mut self,
        targets: &[BlockId],
        program: &CompiledProgram,
    ) -> Result<BlockId, ExecutionStatus> {
        let mut taken = None;
        for &target_id in targets {
            let block = &program.blocks[target_id as usize];
            if let Some(ref cond) = block.assume_cond {
                if self.eval_bool(cond, program) {
                    assert!(
                        taken.is_none(),
                        "Multiple goto conditions are true for targets: {:?}",
                        targets
                            .iter()
                            .map(|t| &program.blocks[*t as usize].name)
                            .collect::<Vec<_>>()
                    );
                    taken = Some(target_id);
                }
            }
        }
        taken.ok_or_else(|| {
            // No branch guard held for the current state: render each candidate
            // target's guard (and the live values) so the caller sees which
            // partition the input fell outside of.
            let mut detail = String::from("no goto target feasible; guards: ");
            for (i, &target_id) in targets.iter().enumerate() {
                if i > 0 {
                    detail.push_str(" | ");
                }
                let block = &program.blocks[target_id as usize];
                match &block.assume_cond {
                    Some(cond) => detail.push_str(&self.describe_assume_expr(cond, program)),
                    None => detail.push_str(&format!("{}=<unconditional>", block.name)),
                }
                if detail.len() > 400 {
                    detail.push('…');
                    break;
                }
            }
            ExecutionStatus::AssumeViolation {
                pc: self.pc,
                block: self.block_name(program),
                reason: "infeasible_goto",
                detail,
            }
        })
    }

    /// Execute a single statement.
    fn execute_stmt(
        &mut self,
        stmt: &Stmt,
        program: &CompiledProgram,
    ) -> Result<(), ExecutionStatus> {
        match stmt {
            Stmt::Assign1 { lhs, rhs } => {
                let val = self.eval(rhs, program);
                self.set_eval_result(*lhs, val);
            }
            Stmt::AssignN { lhs, rhs } => {
                let mut vals = std::mem::take(&mut self.scratch_evals);
                vals.clear();
                vals.extend(rhs.iter().map(|r| self.eval(r, program)));
                for (var_id, val) in lhs.iter().zip(vals.drain(..)) {
                    self.set_eval_result(*var_id, val);
                }
                self.scratch_evals = vals;
            }
            Stmt::Assert { expr } => {
                if !self.eval_bool(expr, program) {
                    return Err(ExecutionStatus::AssertViolation {
                        pc: self.pc,
                        block: self.block_name(program),
                    });
                }
            }
            Stmt::Assume { expr } => {
                // Concrete execution: assume is treated as assert — if the
                // expression is false, the inputs violate a precondition
                // the verifier is allowed to rely on, so fail loudly.
                // (`$isExternal` assumes are rewritten to AssumeTrue at
                // lowering — see lowering::normalize_is_external_assumes.)
                if !self.eval_bool(expr, program) {
                    let detail = self.describe_assume_expr(expr, program);
                    return Err(ExecutionStatus::AssumeViolation {
                        pc: self.pc,
                        block: self.block_name(program),
                        reason: "assume",
                        detail,
                    });
                }
            }
            Stmt::AssumeTrue => {}
            Stmt::LoopHeaderSnap { live_vars } => {
                if !self.no_trace {
                    for &vid in live_vars {
                        match &self.vars[vid as usize] {
                            Value::Scalar(val) => {
                                let val = *val;
                                self.trace
                                    .record(vid, val, self.pc, self.curr_block_id, OP_WRITE);
                            }
                            // Out-of-i64 exact value: counted trace skip.
                            Value::Big(_) => self.big_trace_skips += 1,
                            Value::Map(_) => {}
                        }
                    }
                }
            }
            Stmt::Havoc { vars } => {
                for &var_id in vars {
                    self.clear_var(var_id);
                }
            }
            Stmt::HavocCurrAddr {
                var_id,
                alloc_size_var,
            } => {
                // Replicate Python's handle_curr_addr + get_var fallback:
                // Python: handle_curr_addr sets alloc_addr, then clear_var removes from var_store.
                // But get_var has a $CurrAddr fallback: if not in var_store, returns alloc_addr.
                // So the net effect is: $CurrAddr always equals alloc_addr.
                // We just compute the new address and set it (no clear needed).
                assert!(
                    *alloc_size_var != u32::MAX,
                    "HavocCurrAddr alloc_size_var not resolved for {}",
                    self.var_names[*var_id as usize]
                );
                let alloc_size = self.get_scalar_silent(*alloc_size_var);
                let is_shadow = Some(*var_id) == self.curr_addr_shadow_id;
                let old_addr = if is_shadow {
                    self.alloc_addr_shadow
                } else {
                    self.alloc_addr
                };
                let new_addr = (old_addr + alloc_size + 255) & !255;

                if is_shadow {
                    self.alloc_addr_shadow = new_addr;
                } else {
                    self.alloc_addr = new_addr;
                }

                // Set the scalar — this traces the write and updates alloc_addr
                // Don't clear: Python's get_var falls back to alloc_addr when var is missing
                self.set_scalar(*var_id, new_addr, false);
                // Read for trace (matching Python's _get_var call in _handle_havoc)
                self.get_scalar(*var_id);
            }
            Stmt::CallIgnored => {}
            Stmt::CallNondet { assignments } => {
                for &var_id in assignments {
                    let value = self.next_havoc_value(var_id);
                    self.set_scalar(var_id, value, false);
                }
            }
            Stmt::CallPrintf { args } => {
                self.execute_printf(args, program);
            }
            Stmt::CallTime { assignments, args } => {
                // Evaluate args for side effects
                for arg in args {
                    self.eval(arg, program);
                }
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;
                for &var_id in assignments {
                    self.set_scalar(var_id, t, false);
                }
            }
            Stmt::CallWrite { assignments, args } => {
                for arg in args {
                    self.eval(arg, program);
                }
                let val = self.wlen_buf[self.wlen_buf_idx % self.wlen_buf.len()];
                self.wlen_buf_idx += 1;
                for &var_id in assignments {
                    self.set_scalar(var_id, val, false);
                }
            }
            Stmt::CallRead { args } => {
                let vals: Vec<EvalResult> = args.iter().map(|a| self.eval(a, program)).collect();
                // args: [fd, fd_shadow, buf_ptr, buf_ptr_shadow, read_len, read_len_shadow]
                let buf_ptr = match &vals[2] {
                    EvalResult::Scalar(v) => *v,
                    _ => panic!("read.cross_product: buf_ptr is not scalar"),
                };
                let buf_ptr_shadow = match &vals[3] {
                    EvalResult::Scalar(v) => *v,
                    _ => panic!("read.cross_product: buf_ptr_shadow is not scalar"),
                };
                let read_len = match &vals[4] {
                    EvalResult::Scalar(v) => *v,
                    _ => panic!("read.cross_product: read_len is not scalar"),
                };
                let read_len_shadow = match &vals[5] {
                    EvalResult::Scalar(v) => *v,
                    _ => panic!("read.cross_product: read_len_shadow is not scalar"),
                };
                assert_eq!(
                    read_len, read_len_shadow,
                    "read.cross_product: mismatched lengths"
                );
                let data = self.read_external(read_len as usize);
                if let (Some(m0_id), Some(m0s_id)) = (self.m0_id, self.m0_shadow_id) {
                    let m0_idx = self.get_map_idx(m0_id);
                    let m0s_idx = self.get_map_idx(m0s_id);
                    for i in 0..data.len() {
                        self.memory_maps[m0_idx].set(buf_ptr + i as i64, data[i] as i64);
                        self.memory_maps[m0s_idx].set(buf_ptr_shadow + i as i64, data[i] as i64);
                    }
                }
            }
            Stmt::CallMemmove { args } => {
                let vals: Vec<i64> = args.iter().map(|a| self.eval_mem_i64(a, program)).collect();
                if vals.len() >= 6 {
                    let len = vals[4];
                    let len_shadow = vals[5];
                    if len != len_shadow || len < 0 {
                        return Err(ExecutionStatus::AssumeViolation {
                            pc: self.pc,
                            block: self.block_name(program),
                            reason: "invalid_memmove",
                            detail: format!(
                                "memmove length check failed: len={} len_shadow={} \
                                 (requires len == len_shadow && len >= 0)",
                                len, len_shadow
                            ),
                        });
                    }
                    self.memmove_i8_maps(vals[0], vals[1], vals[2], vals[3], len);
                }
            }
            // Quantified assumes for memset/memcpy
            Stmt::QuantMemsetWrite {
                m_ret,
                dst,
                len,
                val,
            } => {
                let dst_val = self.get_scalar_mem(*dst);
                let len_val = self.get_scalar_mem(*len);
                let val_val = self.get_scalar_mem(*val);
                let map_idx = self.get_map_idx(*m_ret);
                self.memory_maps[map_idx].fill_range(dst_val, len_val, val_val);
            }
            Stmt::QuantMemsetPreserveLt { m_ret, m_src, dst } => {
                let dst_val = self.get_scalar_mem(*dst);
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                if src_idx != dst_idx {
                    let (dst_map, src_map) = two_maps(&mut self.memory_maps, dst_idx, src_idx);
                    dst_map.merge_below(src_map, dst_val);
                }
            }
            Stmt::QuantMemsetPreserveGe {
                m_ret,
                m_src,
                dst,
                len,
            } => {
                let dst_val = self.get_scalar_mem(*dst);
                let len_val = self.get_scalar_mem(*len);
                let boundary = dst_val + len_val;
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                if src_idx != dst_idx {
                    let (dst_map, src_map) = two_maps(&mut self.memory_maps, dst_idx, src_idx);
                    dst_map.merge_from(src_map, boundary);
                }
            }
            Stmt::QuantMemcpyWrite {
                m_ret,
                m_src,
                dst,
                src,
                len,
            } => {
                let dst_val = self.get_scalar_mem(*dst);
                let src_val = self.get_scalar_mem(*src);
                let len_val = self.get_scalar_mem(*len);
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                if src_idx == dst_idx {
                    self.memory_maps[dst_idx].move_range_all_init(src_val, dst_val, len_val);
                } else {
                    let (dst_map, src_map) = two_maps(&mut self.memory_maps, dst_idx, src_idx);
                    dst_map.copy_range_values(src_map, src_val, dst_val, len_val);
                }
            }
            Stmt::QuantMemcpyPreserveLt { m_ret, m_src, dst } => {
                let dst_val = self.get_scalar_mem(*dst);
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                if src_idx != dst_idx {
                    let (dst_map, src_map) = two_maps(&mut self.memory_maps, dst_idx, src_idx);
                    dst_map.merge_below(src_map, dst_val);
                }
            }
            Stmt::QuantMemcpyPreserveGe {
                m_ret,
                m_src,
                dst,
                len,
            } => {
                let dst_val = self.get_scalar_mem(*dst);
                let len_val = self.get_scalar_mem(*len);
                let boundary = dst_val + len_val;
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                if src_idx != dst_idx {
                    let (dst_map, src_map) = two_maps(&mut self.memory_maps, dst_idx, src_idx);
                    dst_map.merge_from(src_map, boundary);
                }
            }
            Stmt::If { cond, then_body, else_body } => {
                // Evaluate condition; pick branch; recursively execute its body.
                // Both bodies are guaranteed (by lowering) to contain no
                // terminator stmts, so recursion stays inside this block.
                let take_then = self.eval_bool(cond, program);
                let body: &Vec<Stmt> = if take_then { then_body } else { else_body };
                for inner in body {
                    self.execute_stmt(inner, program)?;
                    self.pc += 1;
                }
            }
            Stmt::While { cond, body } => {
                // Structured loop emitted by diffprod reify when a corerel
                // body contained a nested PWhile. Concrete execution loops
                // until the guard is false.
                while self.eval_bool(cond, program) {
                    for inner in body {
                        self.execute_stmt(inner, program)?;
                        self.pc += 1;
                    }
                }
            }
            Stmt::Goto { .. } | Stmt::Return => {
                panic!("Terminator should not be in body statements")
            }
        }
        Ok(())
    }

    /// Set a variable from an eval result.
    #[inline]
    fn set_eval_result(&mut self, var_id: VarId, result: EvalResult) {
        match result {
            EvalResult::Scalar(v) => self.set_scalar(var_id, v, false),
            EvalResult::Big(b) => self.set_big(var_id, b),
            EvalResult::Bool(b) => self.set_scalar(var_id, b as i64, false),
            EvalResult::MapRef(map_idx) => {
                // Assignment of a map — the store.iN returns the modified map
                let vid = var_id as usize;
                let existing_map = self.var_to_map[vid];
                if let Some(existing_idx) = existing_map {
                    if existing_idx != map_idx {
                        // Copy map contents
                        let new_name = self.var_names[vid].clone();
                        let src = &self.memory_maps[map_idx];
                        let copied = src.copy_with_name(new_name);
                        self.memory_maps[existing_idx] = copied;
                    }
                    // If same index, the map was modified in-place
                } else {
                    // New map variable — copy
                    let new_name = self.var_names[vid].clone();
                    let src = &self.memory_maps[map_idx];
                    let copied = src.copy_with_name(new_name);
                    let new_idx = self.memory_maps.len();
                    self.memory_maps.push(copied);
                    self.var_to_map[vid] = Some(new_idx);
                    self.vars[vid] = Value::Map(new_idx);
                }
            }
        }
    }

    /// Describe a failing assume for a human/agent: render the condition and
    /// append the concrete values of the scalar variables it references, so the
    /// caller knows EXACTLY which precondition the input violated and with what
    /// values. Read-only; safe to call on the error path after `eval_bool`.
    fn describe_assume_expr(&self, expr: &Expr, program: &CompiledProgram) -> String {
        let mut cond = String::new();
        let mut vars: Vec<VarId> = Vec::new();
        self.render_expr(expr, program, &mut cond, &mut vars);
        let mut vals = String::new();
        for vid in vars.iter().take(12) {
            if !vals.is_empty() {
                vals.push_str(", ");
            }
            let name = program
                .var_names
                .get(*vid as usize)
                .map(|s| s.as_str())
                .unwrap_or("?");
            match self.vars.get(*vid as usize) {
                Some(Value::Scalar(v)) => vals.push_str(&format!("{}={}", name, v)),
                Some(Value::Big(b)) => vals.push_str(&format!("{}={}", name, b)),
                Some(Value::Map(_)) => vals.push_str(&format!("{}=<map>", name)),
                None => vals.push_str(&format!("{}=?", name)),
            }
        }
        if vals.is_empty() {
            cond
        } else {
            format!("{}  [where {}]", cond, vals)
        }
    }

    /// Compact recursive render of an `Expr` into Boogie-ish surface syntax,
    /// collecting referenced variable ids (deduped) into `vars`. Bounded length
    /// so a pathological expression cannot produce an unbounded message.
    fn render_expr(
        &self,
        expr: &Expr,
        program: &CompiledProgram,
        out: &mut String,
        vars: &mut Vec<VarId>,
    ) {
        if out.len() > 300 {
            if !out.ends_with('…') {
                out.push('…');
            }
            return;
        }
        match expr {
            Expr::Var(id) => {
                let name = program
                    .var_names
                    .get(*id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                out.push_str(name);
                if !vars.contains(id) {
                    vars.push(*id);
                }
            }
            Expr::Const(v) => out.push_str(&v.to_string()),
            Expr::ConstBig(b) => out.push_str(&b.to_string()),
            Expr::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Expr::BinOp { op, lhs, rhs } => {
                out.push('(');
                self.render_expr(lhs, program, out, vars);
                out.push(' ');
                out.push_str(binop_symbol(op));
                out.push(' ');
                self.render_expr(rhs, program, out, vars);
                out.push(')');
            }
            Expr::Not(inner) => {
                out.push_str("!(");
                self.render_expr(inner, program, out, vars);
                out.push(')');
            }
            Expr::Builtin { fn_id, args } => {
                out.push_str(&format!("{:?}", fn_id));
                out.push('(');
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.render_expr(a, program, out, vars);
                }
                out.push(')');
            }
            Expr::Load {
                bit_width,
                map,
                index,
            } => {
                out.push_str(&format!("load.i{}(", bit_width));
                self.render_expr(map, program, out, vars);
                out.push_str(", ");
                self.render_expr(index, program, out, vars);
                out.push(')');
            }
            Expr::Store {
                bit_width,
                map,
                index,
                value,
            } => {
                out.push_str(&format!("store.i{}(", bit_width));
                self.render_expr(map, program, out, vars);
                out.push_str(", ");
                self.render_expr(index, program, out, vars);
                out.push_str(", ");
                self.render_expr(value, program, out, vars);
                out.push(')');
            }
            Expr::IfThenElse { cond, then_, else_ } => {
                out.push('(');
                self.render_expr(cond, program, out, vars);
                out.push_str(" ? ");
                self.render_expr(then_, program, out, vars);
                out.push_str(" : ");
                self.render_expr(else_, program, out, vars);
                out.push(')');
            }
            Expr::IsExternal => out.push_str("$isExternal"),
        }
    }

    /// Result type for expression evaluation.
    #[inline]
    fn eval_bool(&mut self, expr: &Expr, program: &CompiledProgram) -> bool {
        match self.eval(expr, program) {
            EvalResult::Scalar(v) => v != 0,
            // Big is strictly outside i64, hence never zero.
            EvalResult::Big(_) => true,
            EvalResult::Bool(b) => b,
            EvalResult::MapRef(_) => panic!("Expected bool, got map"),
        }
    }

    /// Evaluate an expression.
    #[inline]
    fn eval(&mut self, expr: &Expr, program: &CompiledProgram) -> EvalResult {
        match expr {
            Expr::Var(var_id) => {
                let vid = *var_id as usize;
                match &self.vars[vid] {
                    Value::Scalar(v) => {
                        let v = *v;
                        if !self.no_trace && self.log_read {
                            self.trace
                                .record(*var_id, v, self.pc, self.curr_block_id, OP_READ);
                        }
                        EvalResult::Scalar(v)
                    }
                    Value::Big(b) => {
                        // Out-of-i64 exact value: the read is NOT trace-
                        // recorded (i64 value field) — counted skip.
                        let b = b.clone();
                        if !self.no_trace && self.log_read {
                            self.big_trace_skips += 1;
                        }
                        EvalResult::Big(b)
                    }
                    Value::Map(idx) => EvalResult::MapRef(*idx),
                }
            }
            Expr::Const(v) => EvalResult::Scalar(*v),
            Expr::ConstBig(b) => EvalResult::Big(b.clone()),
            Expr::Bool(b) => EvalResult::Bool(*b),
            Expr::BinOp { op, lhs, rhs } => {
                if program.mode == SemanticsMode::Bv {
                    // BV mode: the historical wrapping i64 algebra, unchanged.
                    let l = self.eval_i64(lhs, program);
                    let r = self.eval_i64(rhs, program);
                    match op {
                        BinOp::Eq => EvalResult::Bool(l == r),
                        BinOp::Ne => EvalResult::Bool(l != r),
                        BinOp::Lt => EvalResult::Bool(l < r),
                        BinOp::Gt => EvalResult::Bool(l > r),
                        BinOp::Le => EvalResult::Bool(l <= r),
                        BinOp::Ge => EvalResult::Bool(l >= r),
                        BinOp::And => EvalResult::Bool(l != 0 && r != 0),
                        BinOp::Or => EvalResult::Bool(l != 0 || r != 0),
                        BinOp::Implies => EvalResult::Bool(l == 0 || r != 0),
                        BinOp::Iff => EvalResult::Bool((l != 0) == (r != 0)),
                        BinOp::Sub => EvalResult::Scalar((l.wrapping_sub(r)) & MASK_64),
                        BinOp::Mul => EvalResult::Scalar((l.wrapping_mul(r)) & MASK_64),
                        BinOp::Add => EvalResult::Scalar((l.wrapping_add(r)) & MASK_64),
                    }
                } else {
                    // Int mode: exact-ℤ core. Arithmetic uses checked ops with
                    // BigInt promotion; comparisons compare exact values.
                    let le = self.eval(lhs, program);
                    let re = self.eval(rhs, program);
                    // Hot path: both operands are in-i64 (the overwhelmingly
                    // common case) — direct i64 comparisons, checked arith.
                    if let (Some(l), Some(r)) = (as_small(&le), as_small(&re)) {
                        return match op {
                            BinOp::Eq => EvalResult::Bool(l == r),
                            BinOp::Ne => EvalResult::Bool(l != r),
                            BinOp::Lt => EvalResult::Bool(l < r),
                            BinOp::Gt => EvalResult::Bool(l > r),
                            BinOp::Le => EvalResult::Bool(l <= r),
                            BinOp::Ge => EvalResult::Bool(l >= r),
                            BinOp::And => EvalResult::Bool(l != 0 && r != 0),
                            BinOp::Or => EvalResult::Bool(l != 0 || r != 0),
                            BinOp::Implies => EvalResult::Bool(l == 0 || r != 0),
                            BinOp::Iff => EvalResult::Bool((l != 0) == (r != 0)),
                            BinOp::Sub => match l.checked_sub(r) {
                                Some(v) => EvalResult::Scalar(v),
                                None => z_to_eval(crate::builtins::int::sub(&Z::S(l), &Z::S(r))),
                            },
                            BinOp::Mul => match l.checked_mul(r) {
                                Some(v) => EvalResult::Scalar(v),
                                None => z_to_eval(crate::builtins::int::mul(&Z::S(l), &Z::S(r))),
                            },
                            BinOp::Add => match l.checked_add(r) {
                                Some(v) => EvalResult::Scalar(v),
                                None => z_to_eval(crate::builtins::int::add(&Z::S(l), &Z::S(r))),
                            },
                        };
                    }
                    let l = eval_result_to_z(le);
                    let r = eval_result_to_z(re);
                    use std::cmp::Ordering::*;
                    let ord = || crate::builtins::int::cmp(&l, &r);
                    match op {
                        BinOp::Eq => EvalResult::Bool(ord() == Equal),
                        BinOp::Ne => EvalResult::Bool(ord() != Equal),
                        BinOp::Lt => EvalResult::Bool(ord() == Less),
                        BinOp::Gt => EvalResult::Bool(ord() == Greater),
                        BinOp::Le => EvalResult::Bool(ord() != Greater),
                        BinOp::Ge => EvalResult::Bool(ord() != Less),
                        BinOp::And => EvalResult::Bool(!l.is_zero() && !r.is_zero()),
                        BinOp::Or => EvalResult::Bool(!l.is_zero() || !r.is_zero()),
                        BinOp::Implies => EvalResult::Bool(l.is_zero() || !r.is_zero()),
                        BinOp::Iff => EvalResult::Bool(l.is_zero() == r.is_zero()),
                        BinOp::Sub => z_to_eval(crate::builtins::int::sub(&l, &r)),
                        BinOp::Mul => z_to_eval(crate::builtins::int::mul(&l, &r)),
                        BinOp::Add => z_to_eval(crate::builtins::int::add(&l, &r)),
                    }
                }
            }
            Expr::Builtin { fn_id, args } => {
                if program.mode == SemanticsMode::Bv {
                    if builtins::num_args(*fn_id) == 1 {
                        let x = self.eval_i64(&args[0], program);
                        EvalResult::Scalar(builtins::exec_unary(*fn_id, x))
                    } else {
                        let a = self.eval_i64(&args[0], program);
                        let b = self.eval_i64(&args[1], program);
                        let (result, is_bool) = builtins::exec_binary(*fn_id, a, b);
                        if is_bool {
                            EvalResult::Bool(result != 0)
                        } else {
                            EvalResult::Scalar(result & MASK_64)
                        }
                    }
                } else if builtins::num_args(*fn_id) == 1 {
                    // Identity casts dominate: pass in-i64 values straight through.
                    let xe = self.eval(&args[0], program);
                    if !matches!(*fn_id, BuiltinFn::Not { .. }) {
                        if let EvalResult::Scalar(_) | EvalResult::Big(_) = xe {
                            return xe;
                        }
                    }
                    let x = eval_result_to_z(xe);
                    z_to_eval(builtins::int::exec_unary(*fn_id, &x))
                } else {
                    let ae = self.eval(&args[0], program);
                    let be = self.eval(&args[1], program);
                    // Hot path: both operands in-i64 — direct machine ops for
                    // the arithmetic/comparison family (semantically identical
                    // to the exec_binary Z path; see builtins::int).
                    if let (Some(a), Some(b)) = (as_small(&ae), as_small(&be)) {
                        match *fn_id {
                            BuiltinFn::Add { .. } => {
                                if let Some(v) = a.checked_add(b) {
                                    return EvalResult::Scalar(v);
                                }
                            }
                            BuiltinFn::Sub { .. } => {
                                if let Some(v) = a.checked_sub(b) {
                                    return EvalResult::Scalar(v);
                                }
                            }
                            BuiltinFn::Mul { .. } => {
                                if let Some(v) = a.checked_mul(b) {
                                    return EvalResult::Scalar(v);
                                }
                            }
                            BuiltinFn::Slt { .. } | BuiltinFn::Ult { .. } => {
                                return EvalResult::Scalar((a < b) as i64)
                            }
                            BuiltinFn::Sle { .. } | BuiltinFn::Ule { .. } => {
                                return EvalResult::Scalar((a <= b) as i64)
                            }
                            BuiltinFn::Sgt { .. } | BuiltinFn::Ugt { .. } => {
                                return EvalResult::Scalar((a > b) as i64)
                            }
                            BuiltinFn::Sge { .. } | BuiltinFn::Uge { .. } => {
                                return EvalResult::Scalar((a >= b) as i64)
                            }
                            BuiltinFn::BvEq { .. } => {
                                return EvalResult::Scalar((a == b) as i64)
                            }
                            BuiltinFn::BvNe { .. } => {
                                return EvalResult::Scalar((a != b) as i64)
                            }
                            BuiltinFn::SltBool { .. } => return EvalResult::Bool(a < b),
                            BuiltinFn::SleBool { .. } => return EvalResult::Bool(a <= b),
                            BuiltinFn::SgtBool { .. } => return EvalResult::Bool(a > b),
                            BuiltinFn::SgeBool { .. } => return EvalResult::Bool(a >= b),
                            _ => {}
                        }
                    }
                    let a = eval_result_to_z(ae);
                    let b = eval_result_to_z(be);
                    match builtins::int::exec_binary(*fn_id, &a, &b) {
                        ZResult::Num(z) => z_to_eval(z),
                        ZResult::Bool(b) => EvalResult::Bool(b),
                    }
                }
            }
            Expr::Store {
                bit_width,
                map,
                index,
                value,
            } => {
                let map_idx = match self.eval(map, program) {
                    EvalResult::MapRef(idx) => idx,
                    _ => panic!("store: expected map"),
                };
                let idx_val = self.eval_mem_i64(index, program);
                let val = self.eval_mem_i64(value, program);
                let bw = *bit_width as u8;
                let ew = self.memory_maps[map_idx].element_bit_width;
                if bw <= ew {
                    // Single cell. bw == ew is the common byte store; bw < ew is a
                    // sub-element store (e.g. $store.i1 of a bool into a byte-addressed
                    // map) — keep it one cell, masking the value to bw bits.
                    let v = if bw < ew { val & ((1i64 << bw) - 1) } else { val };
                    self.memory_maps[map_idx].set(idx_val, v);
                } else {
                    let count = (bw / ew) as u32;
                    self.memory_maps[map_idx].store_wide(idx_val, count, ew as u32, val);
                }
                EvalResult::MapRef(map_idx)
            }
            Expr::Load {
                bit_width,
                map,
                index,
            } => {
                let map_idx = match self.eval(map, program) {
                    EvalResult::MapRef(idx) => idx,
                    _ => panic!("load: expected map"),
                };
                let idx_val = self.eval_mem_i64(index, program);
                let bw = *bit_width as u8;
                let ew = self.memory_maps[map_idx].element_bit_width;
                if bw <= ew {
                    // Single cell; mask to bw bits for a sub-element load ($load.i1).
                    let raw = self.memory_maps[map_idx].get(idx_val);
                    let v = if bw < ew { raw & ((1i64 << bw) - 1) } else { raw };
                    EvalResult::Scalar(v)
                } else {
                    let count = (bw / ew) as u32;
                    EvalResult::Scalar(self.memory_maps[map_idx].load_wide(
                        idx_val,
                        count,
                        ew as u32,
                    ))
                }
            }
            Expr::IfThenElse { cond, then_, else_ } => {
                if self.eval_bool(cond, program) {
                    self.eval(then_, program)
                } else {
                    self.eval(else_, program)
                }
            }
            Expr::Not(inner) => {
                let v = self.eval_bool(inner, program);
                EvalResult::Bool(!v)
            }
            Expr::IsExternal => EvalResult::Scalar(0),
        }
    }

    /// Evaluate an expression and extract the i64 value.
    #[inline]
    fn eval_i64(&mut self, expr: &Expr, program: &CompiledProgram) -> i64 {
        match self.eval(expr, program) {
            EvalResult::Scalar(v) => v,
            EvalResult::Bool(b) => b as i64,
            EvalResult::Big(b) => panic!(
                "exact-int overflow escape: value {} is outside i64 in an \
                 i64-only context (memory index/value or call argument)",
                b
            ),
            EvalResult::MapRef(_) => panic!("Expected scalar, got map"),
        }
    }

    /// Memory-interface evaluation: like `eval_i64`, but an out-of-i64
    /// exact value folds mod 2^64 (counted). Used for load/store
    /// indices/values and external-call arguments — 64-bit address-space
    /// quantities by construction.
    #[inline]
    fn eval_mem_i64(&mut self, expr: &Expr, program: &CompiledProgram) -> i64 {
        match self.eval(expr, program) {
            EvalResult::Scalar(v) => v,
            EvalResult::Bool(b) => b as i64,
            EvalResult::Big(b) => {
                self.mem_big_folds += 1;
                fold_big_to_i64(&b)
            }
            EvalResult::MapRef(_) => panic!("Expected scalar, got map"),
        }
    }

    /// Evaluate an expression as an exact integer (Int mode).
    #[inline]
    fn eval_z(&mut self, expr: &Expr, program: &CompiledProgram) -> Z {
        eval_result_to_z(self.eval(expr, program))
    }

    /// Read a null-terminated C string from a memory map.
    fn read_cstring(&self, map_idx: usize, mut ptr: i64) -> String {
        let mut bytes = Vec::new();
        loop {
            let byte = self.memory_maps[map_idx].get(ptr) & 0xFF;
            if byte == 0 {
                break;
            }
            bytes.push(byte as u8);
            ptr += 1;
            if bytes.len() > 4096 {
                break;
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Execute a printf call: read format string from $M.0, format args, print.
    ///
    /// printf is in the shadowing pass's EXEMPTION_LIST (see
    /// passes/transform/shadowing.py:139), so its args are NOT doubled
    /// — what we receive here is the original [fmt, args...] tuple.
    /// Older revisions of this VM divided len()/2 assuming shadowing;
    /// that crashed on single-arg ``printf.ref(.str.6)`` calls
    /// (n==0 → vals[0] panic).
    fn execute_printf(&mut self, args: &[Expr], program: &CompiledProgram) {
        if args.is_empty() {
            return; // malformed printf without format string — skip silently
        }
        let vals: Vec<i64> = args.iter().map(|a| self.eval_mem_i64(a, program)).collect();

        let m0_id = match self.m0_id {
            Some(id) => id,
            None => return, // no $M.0 — can't read format string
        };
        let m0_idx = self.get_map_idx(m0_id);

        let fmt_ptr = vals[0];
        let fmt = self.read_cstring(m0_idx, fmt_ptr);
        let printf_args = &vals[1..];

        let output = format_printf(&fmt, printf_args, &self.memory_maps[m0_idx]);
        print!("{}", output);
    }
}

/// Read a null-terminated C string from a memory map (standalone helper).
fn read_cstring_from_map(map: &crate::memory_map::MemoryMap, mut ptr: i64) -> String {
    let mut bytes = Vec::new();
    loop {
        let byte = map.get(ptr) & 0xFF;
        if byte == 0 {
            break;
        }
        bytes.push(byte as u8);
        ptr += 1;
        if bytes.len() > 4096 {
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Format a C-style printf string with the given arguments.
fn format_printf(fmt: &str, args: &[i64], m0: &crate::memory_map::MemoryMap) -> String {
    use std::fmt::Write;
    let mut result = String::new();
    let bytes = fmt.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut arg_idx = 0;

    while i < len {
        if bytes[i] == b'%' {
            i += 1;
            if i >= len {
                break;
            }
            if bytes[i] == b'%' {
                result.push('%');
                i += 1;
                continue;
            }
            // Parse flags
            let mut flag_minus = false;
            let mut flag_zero = false;
            let mut flag_hash = false;
            while i < len {
                match bytes[i] {
                    b'-' => flag_minus = true,
                    b'0' => flag_zero = true,
                    b'+' | b' ' => {}
                    b'#' => flag_hash = true,
                    _ => break,
                }
                i += 1;
            }
            // Parse width
            let mut width: usize = 0;
            while i < len && bytes[i].is_ascii_digit() {
                width = width * 10 + (bytes[i] - b'0') as usize;
                i += 1;
            }
            // Parse precision
            let mut precision: Option<usize> = None;
            if i < len && bytes[i] == b'.' {
                i += 1;
                let mut p = 0usize;
                while i < len && bytes[i].is_ascii_digit() {
                    p = p * 10 + (bytes[i] - b'0') as usize;
                    i += 1;
                }
                precision = Some(p);
            }
            // Skip length modifiers
            while i < len && matches!(bytes[i], b'h' | b'l' | b'L' | b'z' | b'j' | b't') {
                i += 1;
            }
            if i >= len {
                break;
            }
            let spec = bytes[i] as char;
            i += 1;

            if arg_idx >= args.len() {
                result.push_str("<?>");
                continue;
            }
            let val = args[arg_idx];
            arg_idx += 1;

            match spec {
                'd' | 'i' => {
                    // signed
                    let signed_val = val;
                    let formatted = if width > 0 {
                        if flag_minus {
                            format!("{:<width$}", signed_val, width = width)
                        } else if flag_zero {
                            format!("{:0>width$}", signed_val, width = width)
                        } else {
                            format!("{:>width$}", signed_val, width = width)
                        }
                    } else {
                        format!("{}", signed_val)
                    };
                    result.push_str(&formatted);
                }
                'u' => {
                    let uval = val as u64;
                    let formatted = if width > 0 {
                        if flag_minus {
                            format!("{:<width$}", uval, width = width)
                        } else if flag_zero {
                            format!("{:0>width$}", uval, width = width)
                        } else {
                            format!("{:>width$}", uval, width = width)
                        }
                    } else {
                        format!("{}", uval)
                    };
                    result.push_str(&formatted);
                }
                'x' => {
                    let uval = val as u64;
                    let prefix = if flag_hash { "0x" } else { "" };
                    let hex = format!("{:x}", uval);
                    if width > 0 {
                        let pad_width = width.saturating_sub(prefix.len());
                        if flag_minus {
                            let _ = write!(result, "{}{:<width$}", prefix, hex, width = pad_width);
                        } else if flag_zero {
                            let _ = write!(result, "{}{:0>width$}", prefix, hex, width = pad_width);
                        } else {
                            let _ = write!(result, "{}{:>width$}", prefix, hex, width = pad_width);
                        }
                    } else {
                        let _ = write!(result, "{}{}", prefix, hex);
                    }
                }
                'X' => {
                    let uval = val as u64;
                    let prefix = if flag_hash { "0X" } else { "" };
                    let hex = format!("{:X}", uval);
                    if width > 0 {
                        let pad_width = width.saturating_sub(prefix.len());
                        if flag_zero {
                            let _ = write!(result, "{}{:0>width$}", prefix, hex, width = pad_width);
                        } else {
                            let _ = write!(result, "{}{:>width$}", prefix, hex, width = pad_width);
                        }
                    } else {
                        let _ = write!(result, "{}{}", prefix, hex);
                    }
                }
                'o' => {
                    let uval = val as u64;
                    if width > 0 {
                        if flag_zero {
                            let _ = write!(result, "{:0>width$o}", uval, width = width);
                        } else {
                            let _ = write!(result, "{:>width$o}", uval, width = width);
                        }
                    } else {
                        let _ = write!(result, "{:o}", uval);
                    }
                }
                'c' => {
                    result.push((val & 0xFF) as u8 as char);
                }
                's' => {
                    let mut s = read_cstring_from_map(m0, val);
                    if let Some(p) = precision {
                        s.truncate(p);
                    }
                    result.push_str(&s);
                }
                'p' => {
                    let _ = write!(result, "0x{:x}", val as u64);
                }
                _ => {
                    let _ = write!(result, "<{}?>", spec);
                }
            }
        } else if bytes[i] == b'\\' && i + 1 < len {
            match bytes[i + 1] {
                b'n' => {
                    result.push('\n');
                    i += 2;
                }
                b't' => {
                    result.push('\t');
                    i += 2;
                }
                b'0' => {
                    result.push('\0');
                    i += 2;
                }
                b'\\' => {
                    result.push('\\');
                    i += 2;
                }
                _ => {
                    result.push(bytes[i] as char);
                    i += 1;
                }
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

/// Disjoint `(&mut dst, &src)` borrows of two distinct memory maps.
#[inline]
fn two_maps(maps: &mut [MemoryMap], dst: usize, src: usize) -> (&mut MemoryMap, &MemoryMap) {
    debug_assert_ne!(dst, src);
    if dst < src {
        let (lo, hi) = maps.split_at_mut(src);
        (&mut lo[dst], &hi[0])
    } else {
        let (lo, hi) = maps.split_at_mut(dst);
        (&mut hi[0], &lo[src])
    }
}

/// Result of evaluating an expression.
#[derive(Debug, Clone)]
pub enum EvalResult {
    Scalar(i64),
    /// Exact-ℤ value strictly outside i64 range (Int mode only).
    Big(Box<BigInt>),
    Bool(bool),
    MapRef(usize),
}

/// Fold an exact integer back into an `EvalResult`, preserving the
/// "Big only when outside i64" invariant maintained by `builtins::int`.
#[inline]
fn z_to_eval(z: Z) -> EvalResult {
    match z {
        Z::S(v) => EvalResult::Scalar(v),
        Z::B(b) => EvalResult::Big(b),
    }
}

/// In-i64 numeric view of an eval result (Bool → 0/1); None for Big/Map.
#[inline]
fn as_small(e: &EvalResult) -> Option<i64> {
    match e {
        EvalResult::Scalar(v) => Some(*v),
        EvalResult::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// Exact-integer view of an eval result (Int mode).
#[inline]
fn eval_result_to_z(e: EvalResult) -> Z {
    match e {
        EvalResult::Scalar(v) => Z::S(v),
        EvalResult::Bool(b) => Z::S(b as i64),
        EvalResult::Big(b) => Z::B(b),
        EvalResult::MapRef(_) => panic!("Expected scalar, got map"),
    }
}

