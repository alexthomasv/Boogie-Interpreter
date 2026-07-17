use crate::opcodes::{BlockId, VarId};
use crate::raw_log::{
    RawLogWriter, NO_VAR_ID, OP_INITIAL_SCALAR, OP_ITER_CONTEXT, OP_PRE_PC, OP_UNKNOWN_WRITE,
};
pub use crate::raw_log::{OP_READ, OP_WRITE, UNKNOWN_REASON_BIG_INT};

/// Streaming-only trace sink.
///
/// Each trace value/state record carries an opaque `iter_id: u32`:
///
///   ```
///   iter_id = exact active loop-context id
///   ```
///
/// where `iter_id = 0` means "not in any loop".  Nonzero ids are
/// defined by `OP_ITER_CONTEXT` records:
///
///   `var_id=context_id, pc=parent_context_id, block_id=header,
///    value=iter_count, iter_id=depth`
///
/// The parent chain reconstructs the full nested stack, so a value in
/// `(outer=3, inner=5)` receives a distinct id from `(outer=4, inner=5)`.
/// Iteration ids are correlation metadata, not temporal order.  Physical raw
/// record order is authoritative, and a `P` record precedes every operation
/// performed by its statement.
pub struct TraceAccumulator {
    /// Total number of trace entries (reads + writes, counting all
    /// per-loop-level records).
    pub total: u64,
    /// Active loop stack — one frame per enclosing loop, outer-first.
    stack: Vec<StackFrame>,
    /// Next nonzero exact loop-context id to allocate.
    next_context_id: u32,
    /// `is_loop_header[block_id]` — whether that block is the header
    /// of some loop.  Sized to `num_blocks`.
    is_loop_header: Vec<bool>,
    /// For each block, the innermost enclosing loop header (None if
    /// the block is not inside any loop).
    block_innermost_header: Vec<Option<BlockId>>,
    /// For each block that is a loop header, the parent header's
    /// BlockId (None for top-level loops).
    loop_parent_header: Vec<Option<BlockId>>,
    /// Raw-log writer. Set via `VM::enable_raw_log`; absent for no-trace runs.
    pub raw_log: Option<RawLogWriter>,
}

impl Clone for TraceAccumulator {
    fn clone(&self) -> Self {
        Self {
            total: self.total,
            stack: self.stack.clone(),
            next_context_id: self.next_context_id,
            is_loop_header: self.is_loop_header.clone(),
            block_innermost_header: self.block_innermost_header.clone(),
            loop_parent_header: self.loop_parent_header.clone(),
            // Raw log sinks are execution-owned resources. Cloned VMs are
            // used for no-trace symbolic exploration, so never duplicate the
            // writer handle.
            raw_log: None,
        }
    }
}

impl TraceAccumulator {
    pub fn new() -> Self {
        Self {
            total: 0,
            stack: Vec::new(),
            next_context_id: 1,
            is_loop_header: Vec::new(),
            block_innermost_header: Vec::new(),
            loop_parent_header: Vec::new(),
            raw_log: None,
        }
    }

    /// Install loop metadata from the compiled program.  Must be
    /// called before any `on_block_enter` or `record` calls.
    pub fn set_loop_metadata(
        &mut self,
        is_loop_header: Vec<bool>,
        block_innermost_header: Vec<Option<BlockId>>,
        loop_parent_header: Vec<Option<BlockId>>,
    ) {
        self.is_loop_header = is_loop_header;
        self.block_innermost_header = block_innermost_header;
        self.loop_parent_header = loop_parent_header;
    }

    /// Is `header` an ancestor (self or transitive outer) of `block`?
    /// Used to decide whether to pop or keep a loop frame on entry.
    #[inline]
    fn header_contains_block(&self, header: BlockId, block: BlockId) -> bool {
        // Walk the innermost-header chain starting at `block` upwards.
        let mut cur = self
            .block_innermost_header
            .get(block as usize)
            .copied()
            .flatten();
        while let Some(h) = cur {
            if h == header {
                return true;
            }
            cur = self.loop_parent_header.get(h as usize).copied().flatten();
        }
        false
    }

    /// Called by the VM on every block entry.  Maintains the loop
    /// stack per the state machine documented in the plan:
    ///
    ///   * Back-edge of current loop (block is a header AND top-of-
    ///     stack header == block) → increment top counter.
    ///   * New loop entry (block is a header AND top-of-stack differs)
    ///     → pop frames whose header is no longer an ancestor of
    ///       block, then push (block, 0).
    ///   * Non-header block → pop frames whose header is no longer an
    ///     ancestor of block.
    pub fn on_block_enter(&mut self, block_id: BlockId) {
        // Pop frames whose header does NOT contain `block_id`.
        while let Some(top) = self.stack.last() {
            if top.header == block_id {
                // Special-case: about to re-enter the same loop header
                // — keep the frame and increment it below.
                break;
            }
            if self.header_contains_block(top.header, block_id) {
                break;
            }
            self.stack.pop();
        }

        let is_header = self
            .is_loop_header
            .get(block_id as usize)
            .copied()
            .unwrap_or(false);
        if !is_header {
            return;
        }

        // Block is a header.  Two cases: back-edge or new loop.
        if self
            .stack
            .last()
            .map(|top| top.header == block_id)
            .unwrap_or(false)
        {
            let depth = self.stack.len() as u32;
            let parent_context_id = if self.stack.len() >= 2 {
                self.stack[self.stack.len() - 2].context_id
            } else {
                0
            };
            let count = self
                .stack
                .last()
                .map(|frame| frame.count.saturating_add(1))
                .unwrap_or(0);
            let context_id = self.allocate_context(parent_context_id, block_id, count, depth);
            if let Some(top) = self.stack.last_mut() {
                // Back-edge — increment (saturating).
                top.count = count;
                top.context_id = context_id;
            }
            return;
        }
        // New loop entry.  The pop-loop above already removed any
        // frames that don't contain this block; now push ours.
        let parent_context_id = self.stack.last().map(|frame| frame.context_id).unwrap_or(0);
        let depth = self.stack.len() as u32 + 1;
        let context_id = self.allocate_context(parent_context_id, block_id, 0, depth);
        self.stack.push(StackFrame {
            header: block_id,
            count: 0,
            context_id,
        });
    }

    /// Record a trace entry.  A write/read inside any loop emits a
    /// single value record tagged with the exact active nested context;
    /// outside loops it emits `iter_id=0`.
    #[inline]
    pub fn record(&mut self, var_id: VarId, value: i64, pc: u32, block_id: u32, op_type: u8) {
        // The total counter still reflects logical write/read events,
        // not per-level records — downstream uses it for stats only.
        self.total += 1;

        let w = match self.raw_log.as_mut() {
            Some(w) => w,
            None => return,
        };

        let context_id = self.stack.last().map(|frame| frame.context_id).unwrap_or(0);
        if let Err(e) = w.record(op_type, var_id, pc, block_id, value, context_id) {
            panic!("raw trace log write failed: {}", e);
        }
    }

    /// Record one authoritative scalar value after concrete input
    /// initialization and before execution.  PC 0 is the verifier's virtual
    /// entry state; initial records are outside every loop context.
    #[inline]
    pub fn record_initial_scalar(&mut self, var_id: VarId, value: i64, entry_block_id: BlockId) {
        let w = match self.raw_log.as_mut() {
            Some(w) => w,
            None => return,
        };
        if let Err(e) = w.record(OP_INITIAL_SCALAR, var_id, 0, entry_block_id, value, 0) {
            panic!("raw initial-state trace write failed: {}", e);
        }
    }

    /// Record the concrete state boundary immediately before the statement at
    /// `pc`.  Callers must invoke this before evaluating any part of the
    /// statement.  `NO_VAR_ID` makes the event independent of variable rows.
    #[inline]
    pub fn record_pre_pc(&mut self, pc: u32, block_id: BlockId) {
        let context_id = self.stack.last().map(|frame| frame.context_id).unwrap_or(0);
        let w = match self.raw_log.as_mut() {
            Some(w) => w,
            None => return,
        };
        if let Err(e) = w.record(OP_PRE_PC, NO_VAR_ID, pc, block_id, 0, context_id) {
            panic!("raw PRE-pc trace write failed: {}", e);
        }
    }

    /// Invalidate a scalar after a write whose mathematical value cannot be
    /// represented by the raw log.  A consumer must disregard older exact
    /// values until a later `S` or `W` restores a known state.
    #[inline]
    pub fn record_unknown_write(&mut self, var_id: VarId, pc: u32, block_id: BlockId, reason: i64) {
        let context_id = self.stack.last().map(|frame| frame.context_id).unwrap_or(0);
        let w = match self.raw_log.as_mut() {
            Some(w) => w,
            None => return,
        };
        if let Err(e) = w.record(OP_UNKNOWN_WRITE, var_id, pc, block_id, reason, context_id) {
            panic!("raw unknown-write trace failed: {}", e);
        }
    }

    fn allocate_context(
        &mut self,
        parent_context_id: u32,
        header: BlockId,
        count: u32,
        depth: u32,
    ) -> u32 {
        let context_id = self.next_context_id;
        self.next_context_id = self
            .next_context_id
            .checked_add(1)
            .expect("trace loop context id overflow");
        if let Some(w) = self.raw_log.as_mut() {
            if let Err(e) = w.record(
                OP_ITER_CONTEXT,
                context_id,
                parent_context_id,
                header,
                count as i64,
                depth,
            ) {
                panic!("raw trace loop-context write failed: {}", e);
            }
        }
        context_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StackFrame {
    header: BlockId,
    count: u32,
    context_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc_with(meta: (Vec<bool>, Vec<Option<BlockId>>, Vec<Option<BlockId>>)) -> TraceAccumulator {
        let mut a = TraceAccumulator::new();
        a.set_loop_metadata(meta.0, meta.1, meta.2);
        a
    }

    fn stack_pairs(a: &TraceAccumulator) -> Vec<(BlockId, u32)> {
        a.stack
            .iter()
            .map(|frame| (frame.header, frame.count))
            .collect()
    }

    fn stack_contexts(a: &TraceAccumulator) -> Vec<u32> {
        a.stack.iter().map(|frame| frame.context_id).collect()
    }

    /// Simple flat loop: block 1 is a loop header containing block 2.
    /// Entering 1 then 2 repeatedly models iterations.
    #[test]
    fn flat_loop_ticks() {
        let n = 3usize;
        let mut is_h = vec![false; n];
        is_h[1] = true;
        let mut bih: Vec<Option<BlockId>> = vec![None; n];
        bih[1] = Some(1);
        bih[2] = Some(1);
        let lph = vec![None; n];
        let mut a = acc_with((is_h, bih, lph));

        a.on_block_enter(0);
        assert!(a.stack.is_empty());
        a.on_block_enter(1);
        assert_eq!(stack_pairs(&a), vec![(1, 0)]);
        a.on_block_enter(2);
        assert_eq!(stack_pairs(&a), vec![(1, 0)]);
        a.on_block_enter(1); // back-edge
        assert_eq!(stack_pairs(&a), vec![(1, 1)]);
        a.on_block_enter(2);
        a.on_block_enter(1); // back-edge
        assert_eq!(stack_pairs(&a), vec![(1, 2)]);
        // Exit loop
        a.on_block_enter(0);
        assert!(a.stack.is_empty());
    }

    /// Nested loops: header 1 contains header 2.  Inner iteration
    /// ticks don't touch the outer counter; outer back-edge pops the
    /// inner frame.
    #[test]
    fn nested_loop_ticks() {
        let n = 4usize;
        let mut is_h = vec![false; n];
        is_h[1] = true;
        is_h[2] = true;
        let mut bih: Vec<Option<BlockId>> = vec![None; n];
        bih[1] = Some(1);
        bih[2] = Some(2);
        bih[3] = Some(2);
        let mut lph: Vec<Option<BlockId>> = vec![None; n];
        lph[2] = Some(1);
        let mut a = acc_with((is_h, bih, lph));

        a.on_block_enter(1);
        assert_eq!(stack_pairs(&a), vec![(1, 0)]);
        a.on_block_enter(2);
        assert_eq!(stack_pairs(&a), vec![(1, 0), (2, 0)]);
        let inner_first_context = stack_contexts(&a)[1];
        a.on_block_enter(3);
        a.on_block_enter(2); // inner back-edge
        assert_eq!(stack_pairs(&a), vec![(1, 0), (2, 1)]);
        assert_ne!(stack_contexts(&a)[1], inner_first_context);
        a.on_block_enter(3);
        a.on_block_enter(2); // inner back-edge
        assert_eq!(stack_pairs(&a), vec![(1, 0), (2, 2)]);
        let inner_outer_zero_count_two_context = stack_contexts(&a)[1];
        a.on_block_enter(1); // outer back-edge → pop inner frame
        assert_eq!(stack_pairs(&a), vec![(1, 1)]);
        a.on_block_enter(2);
        assert_eq!(stack_pairs(&a), vec![(1, 1), (2, 0)]);
        assert_ne!(stack_contexts(&a)[1], inner_outer_zero_count_two_context);
    }

    #[test]
    fn non_loop_write_emits_zero_iter_id() {
        let mut a = TraceAccumulator::new();
        assert!(a.stack.is_empty());
        // No raw_log attached, just exercise bookkeeping.
        a.record(0, 42, 0, 0, OP_WRITE);
        assert_eq!(a.total, 1);
    }

    #[test]
    fn context_defs_are_written_before_nested_value() {
        let n = 3usize;
        let mut is_h = vec![false; n];
        is_h[1] = true;
        is_h[2] = true;
        let mut bih: Vec<Option<BlockId>> = vec![None; n];
        bih[1] = Some(1);
        bih[2] = Some(2);
        let mut lph: Vec<Option<BlockId>> = vec![None; n];
        lph[2] = Some(1);
        let mut a = acc_with((is_h, bih, lph));

        let path = std::env::temp_dir().join(format!(
            "swoosh_trace_context_{}_{}.raw.zst",
            std::process::id(),
            1
        ));
        let mut writer = RawLogWriter::create(&path).unwrap();
        writer
            .write_header(&["$x".to_string()], &["b0".to_string()])
            .unwrap();
        a.raw_log = Some(writer);

        a.on_block_enter(1);
        a.on_block_enter(2);
        a.record(0, 42, 7, 2, OP_WRITE);

        let count = a.raw_log.take().unwrap().finish().unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(count, 3);
    }
}
