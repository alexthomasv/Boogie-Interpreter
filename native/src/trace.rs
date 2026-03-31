use crate::opcodes::VarId;
use rustc_hash::{FxHashMap, FxHashSet};

/// A traced value with its iteration ID for loop co-occurrence tracking.
/// iteration_id is unique per (var, block) write sequence.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TracedValue {
    pub value: i64,
    pub iteration_id: u32,
}

/// Compact trace accumulator using VarId (u32) keys — zero string allocations in hot loop.
/// String formatting deferred to build_compact_trace at the end.
pub struct TraceAccumulator {
    /// (var_id, pc) → set of (value, iteration_id)
    pub pc_values: FxHashMap<(VarId, u32), FxHashSet<TracedValue>>,
    /// (var_id, block_id) → set of (value, iteration_id)
    pub block_values: FxHashMap<(VarId, u32), FxHashSet<TracedValue>>,
    /// (var_id, pc, op_type) → set of (value, iteration_id)
    pub op_values: FxHashMap<(VarId, u32, u8), FxHashSet<TracedValue>>,
    /// var_id → set of PCs
    pub pc_registry: FxHashMap<VarId, FxHashSet<u32>>,
    /// var_id → set of block_ids
    pub block_registry: FxHashMap<VarId, FxHashSet<u32>>,
    /// Total number of trace entries
    pub total: u64,
    /// Write iteration counter per (var_id, block_id) for loop co-occurrence
    write_counter: FxHashMap<(VarId, u32), u32>,
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
            write_counter: FxHashMap::default(),
        }
    }

    /// Record a trace entry. All keys are u32 — no allocations.
    #[inline]
    pub fn record(&mut self, var_id: VarId, value: i64, pc: u32, block_id: u32, op_type: u8) {
        // Track iteration ID for writes
        let iteration_id = if op_type == OP_WRITE {
            let counter = self.write_counter.entry((var_id, block_id)).or_insert(0);
            *counter += 1;
            *counter
        } else {
            // Reads use the current iteration of the most recent write
            *self.write_counter.get(&(var_id, block_id)).unwrap_or(&0)
        };

        let tv = TracedValue { value, iteration_id };
        self.pc_values
            .entry((var_id, pc))
            .or_default()
            .insert(tv);
        self.block_values
            .entry((var_id, block_id))
            .or_default()
            .insert(tv);
        self.op_values
            .entry((var_id, pc, op_type))
            .or_default()
            .insert(tv);
        self.pc_registry.entry(var_id).or_default().insert(pc);
        self.block_registry
            .entry(var_id)
            .or_default()
            .insert(block_id);
        self.total += 1;
    }
}
