use crate::opcodes::{BlockId, VarId};
use crate::raw_log::RawLogWriter;

/// op_type constants matching Python: 'W' = write, 'R' = read
pub const OP_WRITE: u8 = b'W';
pub const OP_READ: u8 = b'R';

/// Saturating cap for the 16-bit `iter_count` sub-field in packed iter_id.
const ITER_COUNT_MAX: u32 = 0xFFFF;

/// Streaming-only trace sink.
///
/// Each trace record carries a packed `iter_id: u32`:
///
///   ```
///   iter_id = (header_block_id << 16) | (iter_count & 0xFFFF)
///   ```
///
/// where `header_block_id` is the BlockId of the loop header that
/// iteration count refers to.  `iter_id = 0` means "not in any loop".
///
/// A single variable write emits ONE record per enclosing loop —
/// outer-loop writes tag their iter with the outer header; inner-loop
/// writes tag their iter with the inner header.  Downstream consumers
/// that want to compare variables at matching outer-loop iterations
/// simply intersect iter_id sets as before; the packed representation
/// preserves set-intersection semantics without changing the 12-byte
/// member format.
pub struct TraceAccumulator {
    /// Total number of trace entries (reads + writes, counting all
    /// per-loop-level records).
    pub total: u64,
    /// Active loop stack — one frame per enclosing loop, outer-first.
    /// Each frame is `(header_block_id, iter_count_at_this_level)`.
    stack: Vec<(BlockId, u32)>,
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
        let mut cur = self.block_innermost_header.get(block as usize).copied().flatten();
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
        while let Some(&(top_header, _)) = self.stack.last() {
            if top_header == block_id {
                // Special-case: about to re-enter the same loop header
                // — keep the frame and increment it below.
                break;
            }
            if self.header_contains_block(top_header, block_id) {
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
        if let Some(top) = self.stack.last_mut() {
            if top.0 == block_id {
                // Back-edge — increment (saturating).
                top.1 = top.1.saturating_add(1);
                return;
            }
        }
        // New loop entry.  The pop-loop above already removed any
        // frames that don't contain this block; now push ours.
        self.stack.push((block_id, 0));
    }

    /// Record a trace entry.  Writes emit ONE record per enclosing
    /// loop level (each tagged with that level's packed iter_id); a
    /// write outside all loops emits a single record with iter_id=0.
    /// Reads behave identically but share the current iter_count
    /// without ticking anything.
    #[inline]
    pub fn record(&mut self, var_id: VarId, value: i64, pc: u32, block_id: u32, op_type: u8) {
        // The total counter still reflects logical write/read events,
        // not per-level records — downstream uses it for stats only.
        self.total += 1;

        let w = match self.raw_log.as_mut() {
            Some(w) => w,
            None => return,
        };

        if self.stack.is_empty() {
            if let Err(e) = w.record(op_type, var_id, pc, block_id, value, 0) {
                panic!("raw trace log write failed: {}", e);
            }
            return;
        }

        for &(header, count) in self.stack.iter() {
            let packed = ((header as u32) << 16) | (count.min(ITER_COUNT_MAX));
            if let Err(e) = w.record(op_type, var_id, pc, block_id, value, packed) {
                panic!("raw trace log write failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc_with(meta: (Vec<bool>, Vec<Option<BlockId>>, Vec<Option<BlockId>>)) -> TraceAccumulator {
        let mut a = TraceAccumulator::new();
        a.set_loop_metadata(meta.0, meta.1, meta.2);
        a
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
        assert_eq!(a.stack, vec![(1, 0)]);
        a.on_block_enter(2);
        assert_eq!(a.stack, vec![(1, 0)]);
        a.on_block_enter(1); // back-edge
        assert_eq!(a.stack, vec![(1, 1)]);
        a.on_block_enter(2);
        a.on_block_enter(1); // back-edge
        assert_eq!(a.stack, vec![(1, 2)]);
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
        assert_eq!(a.stack, vec![(1, 0)]);
        a.on_block_enter(2);
        assert_eq!(a.stack, vec![(1, 0), (2, 0)]);
        a.on_block_enter(3);
        a.on_block_enter(2); // inner back-edge
        assert_eq!(a.stack, vec![(1, 0), (2, 1)]);
        a.on_block_enter(3);
        a.on_block_enter(2); // inner back-edge
        assert_eq!(a.stack, vec![(1, 0), (2, 2)]);
        a.on_block_enter(1); // outer back-edge → pop inner frame
        assert_eq!(a.stack, vec![(1, 1)]);
        a.on_block_enter(2);
        assert_eq!(a.stack, vec![(1, 1), (2, 0)]);
    }

    #[test]
    fn non_loop_write_emits_zero_iter_id() {
        let mut a = TraceAccumulator::new();
        assert!(a.stack.is_empty());
        // No raw_log attached, just exercise bookkeeping.
        a.record(0, 42, 0, 0, OP_WRITE);
        assert_eq!(a.total, 1);
    }
}
