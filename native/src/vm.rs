use crate::builtins;
use crate::memory_map::MemoryMap;
use crate::opcodes::*;
use crate::trace::{TraceAccumulator, OP_READ, OP_WRITE};
use rustc_hash::FxHashSet;

const MASK_64: i64 = -1i64; // all bits set = u64::MAX as i64

/// Runtime value — either a scalar or a memory map index.
#[derive(Debug, Clone)]
pub enum Value {
    Scalar(i64),
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
    /// Is this variable a shadow? (precomputed)
    pub is_shadow: Vec<bool>,
    /// Memory maps, indexed by map_index in Value::Map
    pub memory_maps: Vec<MemoryMap>,
    /// Which VarId is a memory map? VarId → Some(map_index)
    pub var_to_map: Vec<Option<usize>>,
    /// Current PC
    pub pc: u32,
    /// Current block name
    pub curr_block: String,
    /// Current block ID (for trace — avoids string allocation)
    pub curr_block_id: u32,
    /// Explored block IDs. Names are materialized only at the Python boundary.
    pub explored_blocks: FxHashSet<BlockId>,
    /// Ordered block entries for lightweight path/edge coverage.
    pub block_trace: Vec<BlockId>,
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
            is_shadow: program.is_shadow.clone(),
            memory_maps: Vec::new(),
            var_to_map,
            pc: 0,
            curr_block: String::new(),
            curr_block_id: 0,
            explored_blocks: FxHashSet::default(),
            block_trace: Vec::new(),
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
        if !self.no_trace && !silent && !self.is_shadow[vid] {
            self.trace
                .record(var_id, value, self.pc, self.curr_block_id, OP_WRITE);
        }
        self.vars[vid] = Value::Scalar(value);
    }

    /// Get a scalar variable value, with optional read tracing.
    #[inline]
    pub fn get_scalar(&mut self, var_id: VarId) -> i64 {
        let vid = var_id as usize;
        match &self.vars[vid] {
            Value::Scalar(v) => {
                let v = *v;
                if !self.no_trace && self.log_read && !self.is_shadow[vid] {
                    self.trace
                        .record(var_id, v, self.pc, self.curr_block_id, OP_READ);
                }
                v
            }
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
            let is_shadow = self.memory_maps[map_idx].name.ends_with(".shadow");
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

        let copied: Vec<(i64, i64)> = self.memory_maps[map_idx]
            .memory
            .iter()
            .filter_map(|(&addr, &value)| {
                if addr >= src && addr < src_end {
                    Some((dst + (addr - src), value))
                } else {
                    None
                }
            })
            .collect();
        let clear: Vec<i64> = self.memory_maps[map_idx]
            .memory
            .keys()
            .filter(|&&addr| addr >= dst && addr < dst_end)
            .copied()
            .collect();

        let map = &mut self.memory_maps[map_idx];
        for addr in clear {
            map.memory.remove(&addr);
        }
        for (addr, value) in copied {
            map.set(addr, value);
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
            self.explored_blocks.insert(block_id);
            self.block_trace.push(block_id);
            self.curr_block.clone_from(&block.name);
            self.curr_block_id = block_id;
            self.pc = block.start_pc;
            // Let the trace accumulator update its loop stack for this
            // block entry — drives the packed iter_id semantics.
            if !self.no_trace {
                self.trace.on_block_enter(block_id);
            }
            if let Some(status) = self.consume_step(&mut steps, max_steps) {
                return status;
            }

            // Execute body statements
            for stmt in &block.body {
                if let Some(status) = self.consume_step(&mut steps, max_steps) {
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
    fn consume_step(&self, steps: &mut usize, max_steps: usize) -> Option<ExecutionStatus> {
        if max_steps == 0 {
            return None;
        }
        if *steps >= max_steps {
            return Some(ExecutionStatus::StepLimit {
                pc: self.pc,
                block: self.curr_block.clone(),
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
        self.curr_block.clone_from(&entry.name);
        self.curr_block_id = program.entry_block;
        self.pc = entry.start_pc;
        for expr in &program.entry_preconditions {
            if !self.eval_bool(expr, program) {
                return Some(ExecutionStatus::AssumeViolation {
                    pc: self.pc,
                    block: self.curr_block.clone(),
                    reason: "requires",
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
        taken.ok_or_else(|| ExecutionStatus::AssumeViolation {
            pc: self.pc,
            block: self.curr_block.clone(),
            reason: "infeasible_goto",
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
                let vals: Vec<EvalResult> = rhs.iter().map(|r| self.eval(r, program)).collect();
                for (var_id, val) in lhs.iter().zip(vals) {
                    self.set_eval_result(*var_id, val);
                }
            }
            Stmt::Assert { expr } => {
                if !self.eval_bool(expr, program) {
                    return Err(ExecutionStatus::AssertViolation {
                        pc: self.pc,
                        block: self.curr_block.clone(),
                    });
                }
            }
            Stmt::Assume { expr } => {
                // `$isExternal` is a verifier-only hint that always reads 0 on
                // heap pointers during concrete execution; skip asserting it.
                if !expr_contains_is_external(expr) {
                    // Concrete execution: assume is treated as assert — if the
                    // expression is false, the inputs violate a precondition
                    // the verifier is allowed to rely on, so fail loudly.
                    if !self.eval_bool(expr, program) {
                        return Err(ExecutionStatus::AssumeViolation {
                            pc: self.pc,
                            block: self.curr_block.clone(),
                            reason: "assume",
                        });
                    }
                }
            }
            Stmt::AssumeTrue => {}
            Stmt::LoopHeaderSnap { live_vars } => {
                if !self.no_trace {
                    for &vid in live_vars {
                        if let Value::Scalar(val) = self.vars[vid as usize] {
                            if !self.is_shadow[vid as usize] {
                                self.trace
                                    .record(vid, val, self.pc, self.curr_block_id, OP_WRITE);
                            }
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
                let vals: Vec<i64> = args.iter().map(|a| self.eval_i64(a, program)).collect();
                if vals.len() >= 6 {
                    let len = vals[4];
                    let len_shadow = vals[5];
                    if len != len_shadow || len < 0 {
                        return Err(ExecutionStatus::AssumeViolation {
                            pc: self.pc,
                            block: self.curr_block.clone(),
                            reason: "invalid_memmove",
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
                let dst_val = self.get_scalar(*dst);
                let len_val = self.get_scalar(*len);
                let val_val = self.get_scalar(*val);
                let map_idx = self.get_map_idx(*m_ret);
                for addr in dst_val..dst_val + len_val {
                    self.memory_maps[map_idx].set(addr, val_val);
                }
            }
            Stmt::QuantMemsetPreserveLt { m_ret, m_src, dst } => {
                let dst_val = self.get_scalar(*dst);
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                // Collect addresses first to avoid borrow issues
                let addrs: Vec<(i64, i64)> = self.memory_maps[src_idx]
                    .memory
                    .iter()
                    .filter(|(&addr, _)| addr < dst_val)
                    .map(|(&addr, &val)| (addr, val))
                    .collect();
                for (addr, val) in addrs {
                    self.memory_maps[dst_idx].set(addr, val);
                }
            }
            Stmt::QuantMemsetPreserveGe {
                m_ret,
                m_src,
                dst,
                len,
            } => {
                let dst_val = self.get_scalar(*dst);
                let len_val = self.get_scalar(*len);
                let boundary = dst_val + len_val;
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                let addrs: Vec<(i64, i64)> = self.memory_maps[src_idx]
                    .memory
                    .iter()
                    .filter(|(&addr, _)| addr >= boundary)
                    .map(|(&addr, &val)| (addr, val))
                    .collect();
                for (addr, val) in addrs {
                    self.memory_maps[dst_idx].set(addr, val);
                }
            }
            Stmt::QuantMemcpyWrite {
                m_ret,
                m_src,
                dst,
                src,
                len,
            } => {
                let dst_val = self.get_scalar(*dst);
                let src_val = self.get_scalar(*src);
                let len_val = self.get_scalar(*len);
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                // Read all source values first
                let vals: Vec<i64> = (0..len_val)
                    .map(|offset| self.memory_maps[src_idx].get(src_val + offset))
                    .collect();
                for (offset, val) in vals.into_iter().enumerate() {
                    self.memory_maps[dst_idx].set(dst_val + offset as i64, val);
                }
            }
            Stmt::QuantMemcpyPreserveLt { m_ret, m_src, dst } => {
                let dst_val = self.get_scalar(*dst);
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                let addrs: Vec<(i64, i64)> = self.memory_maps[src_idx]
                    .memory
                    .iter()
                    .filter(|(&addr, _)| addr < dst_val)
                    .map(|(&addr, &val)| (addr, val))
                    .collect();
                for (addr, val) in addrs {
                    self.memory_maps[dst_idx].set(addr, val);
                }
            }
            Stmt::QuantMemcpyPreserveGe {
                m_ret,
                m_src,
                dst,
                len,
            } => {
                let dst_val = self.get_scalar(*dst);
                let len_val = self.get_scalar(*len);
                let boundary = dst_val + len_val;
                let src_idx = self.get_map_idx(*m_src);
                let dst_idx = self.get_map_idx(*m_ret);
                let addrs: Vec<(i64, i64)> = self.memory_maps[src_idx]
                    .memory
                    .iter()
                    .filter(|(&addr, _)| addr >= boundary)
                    .map(|(&addr, &val)| (addr, val))
                    .collect();
                for (addr, val) in addrs {
                    self.memory_maps[dst_idx].set(addr, val);
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

    /// Result type for expression evaluation.
    #[inline]
    fn eval_bool(&mut self, expr: &Expr, program: &CompiledProgram) -> bool {
        match self.eval(expr, program) {
            EvalResult::Scalar(v) => v != 0,
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
                        if !self.no_trace && self.log_read && !self.is_shadow[vid] {
                            self.trace
                                .record(*var_id, v, self.pc, self.curr_block_id, OP_READ);
                        }
                        EvalResult::Scalar(v)
                    }
                    Value::Map(idx) => EvalResult::MapRef(*idx),
                }
            }
            Expr::Const(v) => EvalResult::Scalar(*v),
            Expr::Bool(b) => EvalResult::Bool(*b),
            Expr::BinOp { op, lhs, rhs } => {
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
            }
            Expr::Builtin { fn_id, args } => {
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
                let idx_val = self.eval_i64(index, program);
                let val = self.eval_i64(value, program);
                let bw = *bit_width as u8;
                let ew = self.memory_maps[map_idx].element_bit_width;
                if bw == ew {
                    self.memory_maps[map_idx].set(idx_val, val);
                } else {
                    let ew_mask = self.memory_maps[map_idx].element_mask();
                    let count = bw / ew;
                    for i in 0..count as i64 {
                        self.memory_maps[map_idx]
                            .set(idx_val + i, (val >> (i * ew as i64)) & ew_mask);
                    }
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
                let idx_val = self.eval_i64(index, program);
                let bw = *bit_width as u8;
                let ew = self.memory_maps[map_idx].element_bit_width;
                if bw == ew {
                    EvalResult::Scalar(self.memory_maps[map_idx].get(idx_val))
                } else {
                    let mut result: i64 = 0;
                    let count = bw / ew;
                    for i in 0..count as i64 {
                        result |= self.memory_maps[map_idx].get(idx_val + i) << (i * ew as i64);
                    }
                    EvalResult::Scalar(result)
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
            EvalResult::MapRef(_) => panic!("Expected scalar, got map"),
        }
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
        let vals: Vec<i64> = args.iter().map(|a| self.eval_i64(a, program)).collect();

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

/// Result of evaluating an expression.
#[derive(Debug, Clone)]
pub enum EvalResult {
    Scalar(i64),
    Bool(bool),
    MapRef(usize),
}

/// True if `expr` mentions `$isExternal` anywhere in its tree. Used to skip
/// concrete assume-as-assert for verifier-only hints.
fn expr_contains_is_external(expr: &Expr) -> bool {
    match expr {
        Expr::IsExternal => true,
        Expr::Var(_) | Expr::Const(_) | Expr::Bool(_) => false,
        Expr::BinOp { lhs, rhs, .. } => {
            expr_contains_is_external(lhs) || expr_contains_is_external(rhs)
        }
        Expr::Builtin { args, .. } => args.iter().any(expr_contains_is_external),
        Expr::Store {
            map, index, value, ..
        } => {
            expr_contains_is_external(map)
                || expr_contains_is_external(index)
                || expr_contains_is_external(value)
        }
        Expr::Load { map, index, .. } => {
            expr_contains_is_external(map) || expr_contains_is_external(index)
        }
        Expr::IfThenElse { cond, then_, else_ } => {
            expr_contains_is_external(cond)
                || expr_contains_is_external(then_)
                || expr_contains_is_external(else_)
        }
        Expr::Not(inner) => expr_contains_is_external(inner),
    }
}
