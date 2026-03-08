use crate::opcodes::VarId;
use rustc_hash::{FxHashMap, FxHashSet};

/// Compact trace accumulator using VarId (u32) keys — zero string allocations in hot loop.
/// String formatting deferred to build_compact_trace at the end.
pub struct TraceAccumulator {
    /// (var_id, pc) → set of values
    pub pc_values: FxHashMap<(VarId, u32), FxHashSet<i64>>,
    /// (var_id, block_id) → set of values
    pub block_values: FxHashMap<(VarId, u32), FxHashSet<i64>>,
    /// (var_id, pc, op_type) → set of values
    pub op_values: FxHashMap<(VarId, u32, u8), FxHashSet<i64>>,
    /// var_id → set of PCs
    pub pc_registry: FxHashMap<VarId, FxHashSet<u32>>,
    /// var_id → set of block_ids
    pub block_registry: FxHashMap<VarId, FxHashSet<u32>>,
    /// Total number of trace entries
    pub total: u64,
}

/// op_type constants matching Python: 'W' = write, 'R' = read
pub const OP_WRITE: u8 = b'W';
pub const OP_READ: u8 = b'R';

impl TraceAccumulator {
    pub fn new() -> Self {
        Self {
            pc_values: FxHashMap::default(),
            block_values: FxHashMap::default(),
            op_values: FxHashMap::default(),
            pc_registry: FxHashMap::default(),
            block_registry: FxHashMap::default(),
            total: 0,
        }
    }

    /// Record a trace entry. All keys are u32 — no allocations.
    #[inline]
    pub fn record(
        &mut self,
        var_id: VarId,
        value: i64,
        pc: u32,
        block_id: u32,
        op_type: u8,
    ) {
        self.pc_values
            .entry((var_id, pc))
            .or_default()
            .insert(value);
        self.block_values
            .entry((var_id, block_id))
            .or_default()
            .insert(value);
        self.op_values
            .entry((var_id, pc, op_type))
            .or_default()
            .insert(value);
        self.pc_registry
            .entry(var_id)
            .or_default()
            .insert(pc);
        self.block_registry
            .entry(var_id)
            .or_default()
            .insert(block_id);
        self.total += 1;
    }
}
