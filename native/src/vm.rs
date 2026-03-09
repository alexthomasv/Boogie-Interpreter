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

/// The virtual machine that executes compiled Boogie programs.
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
    /// Previous block name
    pub last_block: String,
    /// Explored block names
    pub explored_blocks: FxHashSet<String>,
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
    wlen_buf: [i64; 6],
    wlen_buf_idx: usize,
    /// External read buffer
    pub external_buffer: Vec<u8>,
    pub external_buffer_pos: usize,
    /// VarId for $M.0 and $M.0.shadow (for read.cross_product)
    pub m0_id: Option<VarId>,
    pub m0_shadow_id: Option<VarId>,
}

impl VM {
    pub fn new(program: &CompiledProgram) -> Self {
        let n = program.num_vars as usize;
        let vars = vec![Value::Scalar(0); n];
        let var_to_map = vec![None; n];
        Self {
            vars,
            var_names: program.var_names.clone(),
            is_shadow: program.is_shadow.clone(),
            memory_maps: Vec::new(),
            var_to_map,
            pc: 0,
            curr_block: String::new(),
            curr_block_id: 0,
            last_block: String::new(),
            explored_blocks: FxHashSet::default(),
            trace: TraceAccumulator::new(),
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
            no_trace: false,
        }
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

    /// Execute the program starting from the entry block.
    pub fn execute(&mut self, program: &CompiledProgram) {
        let mut block_id = program.entry_block;

        loop {
            let block = &program.blocks[block_id as usize];
            self.explored_blocks.insert(block.name.clone());
            self.last_block = std::mem::replace(&mut self.curr_block, block.name.clone());
            self.curr_block_id = block_id;
            self.pc = block.start_pc;

            // Execute body statements
            for stmt in &block.body {
                self.execute_stmt(stmt, program);
                self.pc += 1;
            }

            // Handle terminator
            match &block.terminator {
                Stmt::Return => break,
                Stmt::Goto { targets } => {
                    if targets.len() == 1 {
                        block_id = targets[0];
                    } else {
                        block_id = self.resolve_branch(targets, program);
                    }
                }
                _ => panic!("Block terminator must be Goto or Return"),
            }
        }
    }

    /// Resolve a multi-target goto by evaluating assume conditions.
    fn resolve_branch(&mut self, targets: &[BlockId], program: &CompiledProgram) -> BlockId {
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
        taken.expect("No goto condition is true")
    }

    /// Execute a single statement.
    fn execute_stmt(&mut self, stmt: &Stmt, program: &CompiledProgram) {
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
                    panic!("Assertion failed at PC {}", self.pc);
                }
            }
            Stmt::Assume { expr } => {
                // Just evaluate; runtime assumes are consumed for side effects
                self.eval(expr, program);
            }
            Stmt::AssumeTrue => {}
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
}

/// Result of evaluating an expression.
#[derive(Debug, Clone)]
pub enum EvalResult {
    Scalar(i64),
    Bool(bool),
    MapRef(usize),
}
