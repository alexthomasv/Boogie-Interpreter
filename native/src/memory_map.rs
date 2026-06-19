use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Paged copy-on-write memory map used by the Rust VM.
///
/// Addresses live in dense bands (input buffers packed from 0 up, heap via
/// 256-aligned `$CurrAddr` growth, globals in a negative band), so storage is
/// a hashmap of 512-slot pages keyed by `masked_addr >> PAGE_SHIFT`
/// (arithmetic shift — negative addresses get negative page keys).
///
/// Pages are `Arc`-shared: `copy_with_name` (Boogie map-to-map assignment)
/// and concolic per-state VM clones are O(pages) refcount bumps, and writes
/// diverge per page via `Arc::make_mut`.
///
/// Each page carries an occupancy bitmap. The *exact* set of initialized
/// (addr, value) entries is observable downstream (memory_summary hashing,
/// export_memory_maps, the quantified memset/memcpy preserve ops), so
/// "initialized" is tracked per slot — never inferred from value != 0 or
/// page presence. Invariant: a slot whose init bit is clear holds data 0,
/// so reads (which return 0 for uninitialized addresses) never need the
/// bitmap.
pub const PAGE_SHIFT: u32 = 9;
pub const PAGE_SIZE: usize = 1 << PAGE_SHIFT; // 512 i64 slots / 4 KiB data
const INIT_WORDS: usize = PAGE_SIZE / 64;

#[derive(Debug, Clone)]
pub struct Page {
    pub data: [i64; PAGE_SIZE],
    pub init: [u64; INIT_WORDS],
}

impl Page {
    fn zeroed() -> Self {
        Page {
            data: [0; PAGE_SIZE],
            init: [0; INIT_WORDS],
        }
    }

    #[inline]
    fn set_init(&mut self, slot: usize) {
        self.init[slot >> 6] |= 1u64 << (slot & 63);
    }

    #[inline]
    fn clear_init(&mut self, slot: usize) {
        self.init[slot >> 6] &= !(1u64 << (slot & 63));
    }

    #[inline]
    fn is_init(&self, slot: usize) -> bool {
        self.init[slot >> 6] & (1u64 << (slot & 63)) != 0
    }

    #[inline]
    fn init_count(&self) -> usize {
        self.init.iter().map(|w| w.count_ones() as usize).sum()
    }
}

#[inline]
fn page_key(addr: i64) -> i64 {
    addr >> PAGE_SHIFT // arithmetic shift: negative addrs → negative keys
}

#[inline]
fn page_slot(addr: i64) -> usize {
    (addr & (PAGE_SIZE as i64 - 1)) as usize
}

#[derive(Debug, Clone)]
pub struct MemoryMap {
    pub name: String,
    /// Precomputed `name.ends_with(".shadow")` (hot in memmove dispatch).
    pub is_shadow: bool,
    pages: FxHashMap<i64, Arc<Page>>,
    pub index_bit_width: u8,
    pub element_bit_width: u8,
    index_mask: i64,
    element_mask: i64,
}

impl MemoryMap {
    pub fn new(name: String, index_bit_width: u8, element_bit_width: u8) -> Self {
        let index_mask = if index_bit_width >= 64 {
            -1i64 // all bits set
        } else {
            (1i64 << index_bit_width) - 1
        };
        let element_mask = if element_bit_width >= 64 {
            -1i64
        } else {
            (1i64 << element_bit_width) - 1
        };
        let is_shadow = name.ends_with(".shadow");
        Self {
            name,
            is_shadow,
            pages: FxHashMap::default(),
            index_bit_width,
            element_bit_width,
            index_mask,
            element_mask,
        }
    }

    #[inline]
    pub fn get(&self, addr: i64) -> i64 {
        let addr = addr & self.index_mask;
        match self.pages.get(&page_key(addr)) {
            Some(page) => page.data[page_slot(addr)],
            None => 0,
        }
    }

    #[inline]
    pub fn set(&mut self, addr: i64, value: i64) {
        let addr = addr & self.index_mask;
        let page = Arc::make_mut(
            self.pages
                .entry(page_key(addr))
                .or_insert_with(|| Arc::new(Page::zeroed())),
        );
        let slot = page_slot(addr);
        page.data[slot] = value & self.element_mask;
        page.set_init(slot);
    }

    /// Clear one slot (init bit + data) — the paged equivalent of removing a
    /// hashmap entry.
    pub fn remove(&mut self, addr: i64) {
        let addr = addr & self.index_mask;
        if let Some(arc) = self.pages.get_mut(&page_key(addr)) {
            let page = Arc::make_mut(arc);
            let slot = page_slot(addr);
            page.data[slot] = 0;
            page.clear_init(slot);
        }
    }

    pub fn clear(&mut self) {
        self.pages.clear();
    }

    /// O(pages) copy: the page table is cloned, pages themselves are shared
    /// until either side writes (Arc::make_mut).
    pub fn copy_with_name(&self, new_name: String) -> Self {
        let is_shadow = new_name.ends_with(".shadow");
        Self {
            name: new_name,
            is_shadow,
            pages: self.pages.clone(),
            index_bit_width: self.index_bit_width,
            element_bit_width: self.element_bit_width,
            index_mask: self.index_mask,
            element_mask: self.element_mask,
        }
    }

    #[inline]
    pub fn element_mask(&self) -> i64 {
        self.element_mask
    }

    /// True iff no slot is initialized.
    pub fn is_init_empty(&self) -> bool {
        self.pages
            .values()
            .all(|p| p.init.iter().all(|w| *w == 0))
    }

    /// Number of initialized slots.
    pub fn init_len(&self) -> usize {
        self.pages.values().map(|p| p.init_count()).sum()
    }

    /// `addr..addr+count` is contiguous under the index mask (no wraparound),
    /// so range ops can use raw address arithmetic.
    #[inline]
    fn range_unwrapped(&self, addr: i64, count: i64) -> bool {
        addr & self.index_mask == addr
            && match addr.checked_add(count - 1) {
                Some(last) => last & self.index_mask == last,
                None => false,
            }
    }

    /// Read `count` consecutive elements of `elem_bits` each, little-endian
    /// composed into one i64. Fast path: all slots in one page → one lookup.
    pub fn load_wide(&self, base: i64, count: u32, elem_bits: u32) -> i64 {
        let count_i = count as i64;
        if self.range_unwrapped(base, count_i) && page_key(base) == page_key(base + count_i - 1) {
            let slot = page_slot(base);
            match self.pages.get(&page_key(base)) {
                Some(page) => {
                    let mut result: i64 = 0;
                    for i in 0..count as usize {
                        result |= page.data[slot + i] << (i as u32 * elem_bits);
                    }
                    result
                }
                None => 0,
            }
        } else {
            let mut result: i64 = 0;
            for i in 0..count_i {
                result |= self.get(base + i) << (i * elem_bits as i64);
            }
            result
        }
    }

    /// Store an i64 as `count` consecutive elements of `elem_bits` each
    /// (little-endian split). Fast path: one page → one lookup/make_mut.
    pub fn store_wide(&mut self, base: i64, count: u32, elem_bits: u32, value: i64) {
        let count_i = count as i64;
        if self.range_unwrapped(base, count_i) && page_key(base) == page_key(base + count_i - 1) {
            let page = Arc::make_mut(
                self.pages
                    .entry(page_key(base))
                    .or_insert_with(|| Arc::new(Page::zeroed())),
            );
            let slot = page_slot(base);
            for i in 0..count as usize {
                page.data[slot + i] = (value >> (i as u32 * elem_bits)) & self.element_mask;
                page.set_init(slot + i);
            }
        } else {
            for i in 0..count_i {
                self.set(base + i, value >> (i * elem_bits as i64));
            }
        }
    }

    /// Set every slot in `[start, start+len)` to `value`, marking all of them
    /// initialized (QuantMemsetWrite semantics). Page-chunked.
    pub fn fill_range(&mut self, start: i64, len: i64, value: i64) {
        if len <= 0 {
            return;
        }
        if !self.range_unwrapped(start, len) {
            // Wrapping range — match the per-slot masking of the old loop.
            for i in 0..len {
                self.set(start + i, value);
            }
            return;
        }
        let value = value & self.element_mask;
        let mut addr = start;
        let end = start + len;
        while addr < end {
            let page_end = ((page_key(addr) + 1) << PAGE_SHIFT).min(end);
            let page = Arc::make_mut(
                self.pages
                    .entry(page_key(addr))
                    .or_insert_with(|| Arc::new(Page::zeroed())),
            );
            let lo = page_slot(addr);
            let hi = lo + (page_end - addr) as usize;
            page.data[lo..hi].fill(value);
            for slot in lo..hi {
                page.set_init(slot);
            }
            addr = page_end;
        }
    }

    /// Clear every slot in `[start, start+len)` (init bits + data).
    pub fn clear_range(&mut self, start: i64, len: i64) {
        if len <= 0 {
            return;
        }
        if !self.range_unwrapped(start, len) {
            for i in 0..len {
                self.remove(start + i);
            }
            return;
        }
        let mut addr = start;
        let end = start + len;
        while addr < end {
            let page_end = ((page_key(addr) + 1) << PAGE_SHIFT).min(end);
            if let Some(arc) = self.pages.get_mut(&page_key(addr)) {
                let page = Arc::make_mut(arc);
                let lo = page_slot(addr);
                let hi = lo + (page_end - addr) as usize;
                page.data[lo..hi].fill(0);
                for slot in lo..hi {
                    page.clear_init(slot);
                }
            }
            addr = page_end;
        }
    }

    /// QuantMemcpyWrite: `self[dst+i] = src[src_start+i]` for i in 0..len,
    /// marking ALL `len` destination slots initialized (even where the source
    /// was uninitialized and read as 0). This differs from `move_range`,
    /// which preserves the source's init set — the asymmetry is observable
    /// downstream and load-bearing.
    pub fn copy_range_values(&mut self, src: &MemoryMap, src_start: i64, dst_start: i64, len: i64) {
        if len <= 0 {
            return;
        }
        if !self.range_unwrapped(dst_start, len) {
            for i in 0..len {
                self.set(dst_start + i, src.get(src_start + i));
            }
            return;
        }
        let mut off = 0i64;
        while off < len {
            let dst_addr = dst_start + off;
            let page_end_off = (((page_key(dst_addr) + 1) << PAGE_SHIFT) - dst_start).min(len);
            let page = Arc::make_mut(
                self.pages
                    .entry(page_key(dst_addr))
                    .or_insert_with(|| Arc::new(Page::zeroed())),
            );
            let lo = page_slot(dst_addr);
            for i in off..page_end_off {
                let slot = lo + (i - off) as usize;
                // src.get masks the address; values in src are pre-masked but
                // element widths can differ between maps, so re-mask.
                page.data[slot] = src.get(src_start + i) & self.element_mask;
                page.set_init(slot);
            }
            off = page_end_off;
        }
    }

    /// Same-map QuantMemcpyWrite: gather `len` values from `[src, src+len)`
    /// (uninitialized slots read 0), then store ALL of them into
    /// `[dst, dst+len)` marking every destination slot initialized —
    /// matching the old gather-then-set loop exactly.
    pub fn move_range_all_init(&mut self, src: i64, dst: i64, len: i64) {
        if len <= 0 {
            return;
        }
        let vals: Vec<i64> = (0..len).map(|i| self.get(src + i)).collect();
        for (i, v) in vals.into_iter().enumerate() {
            self.set(dst + i as i64, v);
        }
    }

    /// memmove within one map: copy `[src, src+len)` over `[dst, dst+len)`,
    /// preserving the source's init set exactly (dst slots whose source was
    /// uninitialized end up uninitialized). Overlap-safe: gathers first.
    pub fn move_range(&mut self, src: i64, dst: i64, len: i64) {
        if len <= 0 {
            return;
        }
        // Match the old sparse implementation's guard.
        if src.checked_add(len).is_none() || dst.checked_add(len).is_none() {
            return;
        }
        let gathered: Vec<(i64, i64, bool)> = (0..len)
            .map(|i| {
                let a = (src + i) & self.index_mask;
                match self.pages.get(&page_key(a)) {
                    Some(p) => {
                        let slot = page_slot(a);
                        (i, p.data[slot], p.is_init(slot))
                    }
                    None => (i, 0, false),
                }
            })
            .collect();
        self.clear_range(dst, len);
        for (i, value, init) in gathered {
            if init {
                self.set(dst + i, value);
            }
        }
    }

    /// Preserve-Lt: copy every initialized `(addr, value)` of `src` with
    /// `addr < bound` into `self` (overwriting), leaving other `self` entries
    /// alone. Pages entirely below `bound` that `self` doesn't have are
    /// shared (Arc clone) instead of copied.
    pub fn merge_below(&mut self, src: &MemoryMap, bound: i64) {
        // Arc-sharing skips the per-value re-mask that the old per-entry
        // `set()` loop applied; only valid when element widths match (they
        // always do for the $M.k map-version families these ops act on).
        let can_share = self.element_mask == src.element_mask;
        for (&key, src_page) in &src.pages {
            let page_base = key << PAGE_SHIFT;
            let page_last = page_base + (PAGE_SIZE as i64 - 1);
            if page_base >= bound {
                continue;
            }
            let full = page_last < bound;
            if full && can_share && !self.pages.contains_key(&key) {
                self.pages.insert(key, Arc::clone(src_page));
                continue;
            }
            let limit = if full {
                PAGE_SIZE
            } else {
                (bound - page_base) as usize
            };
            self.merge_page_slots(key, src_page, 0, limit);
        }
    }

    /// Preserve-Ge: same as `merge_below` but for `addr >= bound`.
    pub fn merge_from(&mut self, src: &MemoryMap, bound: i64) {
        let can_share = self.element_mask == src.element_mask;
        for (&key, src_page) in &src.pages {
            let page_base = key << PAGE_SHIFT;
            let page_last = page_base + (PAGE_SIZE as i64 - 1);
            if page_last < bound {
                continue;
            }
            let full = page_base >= bound;
            if full && can_share && !self.pages.contains_key(&key) {
                self.pages.insert(key, Arc::clone(src_page));
                continue;
            }
            let start = if full {
                0
            } else {
                (bound - page_base) as usize
            };
            self.merge_page_slots(key, src_page, start, PAGE_SIZE);
        }
    }

    /// Copy initialized slots `[lo, hi)` of `src_page` into our page `key`.
    fn merge_page_slots(&mut self, key: i64, src_page: &Arc<Page>, lo: usize, hi: usize) {
        // Skip entirely-uninitialized source ranges without allocating a page.
        let any_init = (lo..hi).any(|slot| src_page.is_init(slot));
        if !any_init {
            return;
        }
        let page = Arc::make_mut(
            self.pages
                .entry(key)
                .or_insert_with(|| Arc::new(Page::zeroed())),
        );
        for slot in lo..hi {
            if src_page.is_init(slot) {
                page.data[slot] = src_page.data[slot] & self.element_mask;
                page.set_init(slot);
            }
        }
    }

    /// All initialized `(addr, value)` entries, in arbitrary order (callers
    /// sort where order matters, same as the old hashmap iteration).
    pub fn iter_init(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.pages.iter().flat_map(|(&key, page)| {
            let base = key << PAGE_SHIFT;
            (0..PAGE_SIZE)
                .filter(move |&slot| page.is_init(slot))
                .map(move |slot| (base + slot as i64, page.data[slot]))
        })
    }

    /// Initialized entries with `addr < bound`.
    pub fn iter_init_below(&self, bound: i64) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.iter_init().filter(move |(addr, _)| *addr < bound)
    }

    /// Initialized entries with `addr >= bound`.
    pub fn iter_init_from(&self, bound: i64) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.iter_init().filter(move |(addr, _)| *addr >= bound)
    }

    /// Initialized entries with `start <= addr < end`.
    pub fn iter_init_range(&self, start: i64, end: i64) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.iter_init()
            .filter(move |(addr, _)| *addr >= start && *addr < end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// Reference model: exact init-set semantics over a sparse map.
    #[derive(Clone, Default)]
    struct Model {
        entries: BTreeMap<i64, i64>,
        index_mask: i64,
        element_mask: i64,
    }

    impl Model {
        fn new(index_bw: u8, element_bw: u8) -> Self {
            Model {
                entries: BTreeMap::new(),
                index_mask: if index_bw >= 64 { -1 } else { (1i64 << index_bw) - 1 },
                element_mask: if element_bw >= 64 { -1 } else { (1i64 << element_bw) - 1 },
            }
        }
        fn get(&self, addr: i64) -> i64 {
            *self.entries.get(&(addr & self.index_mask)).unwrap_or(&0)
        }
        fn set(&mut self, addr: i64, value: i64) {
            self.entries
                .insert(addr & self.index_mask, value & self.element_mask);
        }
        fn remove(&mut self, addr: i64) {
            self.entries.remove(&(addr & self.index_mask));
        }
        fn init_set(&self) -> BTreeSet<(i64, i64)> {
            self.entries.iter().map(|(&a, &v)| (a, v)).collect()
        }
    }

    fn map_init_set(m: &MemoryMap) -> BTreeSet<(i64, i64)> {
        m.iter_init().collect()
    }

    fn assert_same(m: &MemoryMap, model: &Model) {
        assert_eq!(map_init_set(m), model.init_set());
    }

    /// Deterministic xorshift so tests don't need external rand.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo) as u64) as i64
        }
    }

    #[test]
    fn get_set_remove_match_model() {
        let mut rng = Rng(0x5eed);
        let mut m = MemoryMap::new("$M.0".into(), 64, 8);
        let mut model = Model::new(64, 8);
        for _ in 0..20_000 {
            // Mixed bands: negative globals, dense low, sparse high.
            let addr = match rng.next() % 4 {
                0 => rng.range(-21_000, -20_000),
                1 => rng.range(0, 2_000),
                2 => rng.range(1 << 20, (1 << 20) + 600),
                _ => rng.range(i64::MIN / 4, i64::MAX / 4),
            };
            match rng.next() % 4 {
                0 | 1 => {
                    let v = rng.next() as i64;
                    m.set(addr, v);
                    model.set(addr, v);
                }
                2 => {
                    m.remove(addr);
                    model.remove(addr);
                }
                _ => assert_eq!(m.get(addr), model.get(addr), "addr {}", addr),
            }
        }
        assert_same(&m, &model);
        assert_eq!(m.init_len(), model.entries.len());
    }

    #[test]
    fn wide_ops_match_per_slot_loops() {
        let mut rng = Rng(0x1de_ABCD);
        for elem_bits in [8u32, 16, 32] {
            let count = 64 / elem_bits;
            let mut a = MemoryMap::new("a".into(), 64, elem_bits as u8);
            let mut b = MemoryMap::new("b".into(), 64, elem_bits as u8);
            for _ in 0..4_000 {
                // Bias bases toward page boundaries to exercise the slow path.
                let base = match rng.next() % 3 {
                    0 => (rng.range(-4, 4) << PAGE_SHIFT) - rng.range(0, count as i64 + 2),
                    _ => rng.range(-1_000, 1_000),
                };
                let value = rng.next() as i64;
                a.store_wide(base, count, elem_bits, value);
                for i in 0..count as i64 {
                    b.set(base + i, value >> (i * elem_bits as i64));
                }
                assert_eq!(a.load_wide(base, count, elem_bits), {
                    let mut r = 0i64;
                    for i in 0..count as i64 {
                        r |= b.get(base + i) << (i * elem_bits as i64);
                    }
                    r
                });
            }
            assert_eq!(map_init_set(&a), map_init_set(&b));
        }
    }

    #[test]
    fn fill_and_clear_range_match_model() {
        let mut rng = Rng(0xF111);
        let mut m = MemoryMap::new("m".into(), 64, 8);
        let mut model = Model::new(64, 8);
        for _ in 0..400 {
            let start = rng.range(-1_200, 1_200);
            let len = rng.range(0, 1_400); // spans multiple pages
            let v = rng.next() as i64;
            if rng.next() % 2 == 0 {
                m.fill_range(start, len, v);
                for i in 0..len {
                    model.set(start + i, v);
                }
            } else {
                m.clear_range(start, len);
                for i in 0..len {
                    model.remove(start + i);
                }
            }
        }
        assert_same(&m, &model);
    }

    #[test]
    fn copy_range_values_inits_full_destination() {
        let mut src = MemoryMap::new("src".into(), 64, 8);
        src.set(100, 7);
        src.set(102, 9); // 101 uninitialized
        let mut dst = MemoryMap::new("dst".into(), 64, 8);
        dst.set(1_000, 42);
        dst.copy_range_values(&src, 100, 200, 3);
        // ALL three destination slots initialized; uninit src read as 0.
        let got = map_init_set(&dst);
        let want: BTreeSet<(i64, i64)> =
            [(200, 7), (201, 0), (202, 9), (1_000, 42)].into_iter().collect();
        assert_eq!(got, want);
    }

    #[test]
    fn move_range_preserves_init_set_and_handles_overlap() {
        let mut rng = Rng(0x3013_1234);
        for _ in 0..300 {
            let mut m = MemoryMap::new("m".into(), 64, 8);
            let mut model = Model::new(64, 8);
            for _ in 0..200 {
                let addr = rng.range(-100, 1_500);
                if rng.next() % 3 == 0 {
                    m.remove(addr);
                    model.remove(addr);
                } else {
                    let v = rng.next() as i64;
                    m.set(addr, v);
                    model.set(addr, v);
                }
            }
            let src = rng.range(-50, 1_000);
            let dst = rng.range(-50, 1_000); // often overlapping
            let len = rng.range(0, 700);
            m.move_range(src, dst, len);
            // Reference: gather, clear dst range, scatter initialized only.
            let gathered: Vec<(i64, Option<i64>)> = (0..len)
                .map(|i| (i, model.entries.get(&(src + i)).copied()))
                .collect();
            for i in 0..len {
                model.remove(dst + i);
            }
            for (i, v) in gathered {
                if let Some(v) = v {
                    model.set(dst + i, v);
                }
            }
            assert_same(&m, &model);
        }
    }

    #[test]
    fn merge_below_and_from_match_model() {
        let mut rng = Rng(0x4e16_e077);
        for _ in 0..300 {
            let mut src = MemoryMap::new("s".into(), 64, 8);
            let mut dst = MemoryMap::new("d".into(), 64, 8);
            let mut model_src = Model::new(64, 8);
            let mut model_dst = Model::new(64, 8);
            for _ in 0..300 {
                let addr = rng.range(-600, 1_800);
                let v = rng.next() as i64;
                if rng.next() % 2 == 0 {
                    src.set(addr, v);
                    model_src.set(addr, v);
                } else {
                    dst.set(addr, v);
                    model_dst.set(addr, v);
                }
            }
            let bound = rng.range(-700, 1_900);
            if rng.next() % 2 == 0 {
                dst.merge_below(&src, bound);
                for (&a, &v) in &model_src.entries {
                    if a < bound {
                        model_dst.set(a, v);
                    }
                }
            } else {
                dst.merge_from(&src, bound);
                for (&a, &v) in &model_src.entries {
                    if a >= bound {
                        model_dst.set(a, v);
                    }
                }
            }
            assert_same(&dst, &model_dst);
            assert_same(&src, &model_src); // src untouched
        }
    }

    #[test]
    fn cow_aliasing_after_copy_with_name() {
        let mut a = MemoryMap::new("$M.1".into(), 64, 64);
        for addr in 0..600 {
            a.set(addr, addr);
        }
        let mut b = a.copy_with_name("$M.1.shadow".into());
        assert!(b.is_shadow);
        // Diverge both sides; neither must see the other's writes.
        b.set(10, 99);
        a.set(20, 77);
        b.clear_range(512, 50);
        assert_eq!(a.get(10), 10);
        assert_eq!(a.get(20), 77);
        assert_eq!(b.get(10), 99);
        assert_eq!(b.get(20), 20);
        assert_eq!(a.get(520), 520);
        assert_eq!(b.get(520), 0);
    }

    #[test]
    fn narrow_index_mask_wraps_like_old_per_slot_loop() {
        // 8-bit index space: addresses wrap mod 256.
        let mut m = MemoryMap::new("m".into(), 8, 8);
        let mut model = Model::new(8, 8);
        m.store_wide(254, 4, 8, 0x0403_0201);
        for i in 0..4i64 {
            model.set(254 + i, 0x0403_0201i64 >> (i * 8));
        }
        assert_same(&m, &model);
        assert_eq!(
            m.load_wide(254, 4, 8),
            (0..4i64).fold(0i64, |acc, i| acc | (model.get(254 + i) << (i * 8)))
        );
    }
}
