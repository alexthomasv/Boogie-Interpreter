use rustc_hash::FxHashMap;

/// Sparse memory map — same semantics as Python MemoryMap.
/// Uninitialized addresses return 0.
#[derive(Debug, Clone)]
pub struct MemoryMap {
    pub name: String,
    pub memory: FxHashMap<i64, i64>,
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
        Self {
            name,
            memory: FxHashMap::default(),
            index_bit_width,
            element_bit_width,
            index_mask,
            element_mask,
        }
    }

    #[inline]
    pub fn get(&self, addr: i64) -> i64 {
        *self.memory.get(&(addr & self.index_mask)).unwrap_or(&0)
    }

    #[inline]
    pub fn set(&mut self, addr: i64, value: i64) {
        self.memory
            .insert(addr & self.index_mask, value & self.element_mask);
    }

    pub fn clear(&mut self) {
        self.memory.clear();
    }

    pub fn copy_with_name(&self, new_name: String) -> Self {
        Self {
            name: new_name,
            memory: self.memory.clone(),
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
}
