INIT_UNINITIALIZED_MEMORY = True

class MemoryMap:
    __slots__ = ('test_name', 'memory', 'name', 'index_bit_width', 'element_bit_width',
                 'env', '_index_mask', '_element_mask')

    def __init__(self, test_name, name, index_bit_width, element_bit_width, env):
        self.test_name = test_name
        self.memory = {}
        self.name = name
        self.index_bit_width = index_bit_width
        self.element_bit_width = element_bit_width
        self.env = env
        self._index_mask = (1 << index_bit_width) - 1
        self._element_mask = (1 << element_bit_width) - 1

    def clear(self):
        self.memory.clear()

    def copy(self, new_map_name):
        new_map = MemoryMap(self.test_name, new_map_name, self.index_bit_width, self.element_bit_width, self.env)
        new_map.memory = self.memory.copy()
        return new_map

    def get(self, addr):
        return self.memory.get(addr & self._index_mask, 0)

    def set(self, addr, value):
        self.memory[addr & self._index_mask] = value & self._element_mask

    def __eq__(self, other):
        if not isinstance(other, MemoryMap):
            return False
        if len(self.memory) != len(other.memory):
            return False
        if f"{self.name}.shadow" == other.name:
            return True
        return self.memory == other.memory

    def __repr__(self):
        lines = [f"{self.name}:"]
        for addr in sorted(self.memory):
            lines.append(f"     {hex(addr)} -> {hex(self.memory[addr])}")
        return "\n".join(lines) + "\n"
