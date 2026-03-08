"""Program input parsing and array/field metadata extraction."""

import json
from collections import defaultdict
from dataclasses import dataclass, replace

from parser.declaration import ImplementationDeclaration
from parser.statement import CallStatement

from utils.program import extract_boogie_variables, RE_SMACK


@dataclass
class Input:
    """Program input variable.

    Three kinds:
      - Scalar:  value is set (int)
      - Buffer:  buffers is a list of {"contents": "0x...", "size": N}
      - Struct:  struct is an ordered list of fields, each either:
                   {"name": ..., "size": N, "value": "0x..."}           — scalar field
                   {"name": ..., "size": N, "buffer": {"contents":..}}  — pointer field
    """
    name: str
    private: bool
    value: int | None = None
    buffers: list | None = None
    struct: list | None = None

    def __str__(self):
        if self.value is not None:
            return f"{self.name} <- {self.value}"
        if self.buffers:
            return f"{self.name} <- {len(self.buffers)} buffer(s)"
        if self.struct:
            return f"{self.name} <- struct({len(self.struct)} fields)"
        return f"{self.name} <- (empty)"

    @property
    def struct_buffers(self):
        """Return the list of buffer dicts from pointer fields, in order."""
        if not self.struct:
            return []
        return [f['buffer'] for f in self.struct if 'buffer' in f]

    @property
    def struct_scalars(self):
        """Return the list of scalar field dicts, in order."""
        if not self.struct:
            return []
        return [f for f in self.struct if 'value' in f]


class ProgramInputs:
    """Parsed program inputs from a JSON file."""

    def __init__(self, variables: dict[str, Input], extra_data: bytes | None = None):
        self.variables = variables
        self.extra_data = extra_data

    def with_shadows(self) -> dict[str, Input]:
        """Return variables dict with shadow copies for public variables."""
        result = dict(self.variables)
        for name, inp in self.variables.items():
            if not inp.private:
                result[f"{name}.shadow"] = replace(inp, name=f"{name}.shadow")
        return result


def parse_inputs(input_json) -> ProgramInputs:
    with open(input_json, 'r') as f:
        raw = json.load(f)

    variables = {}
    extra_data = None

    for entry in raw:
        if 'extra_data' in entry:
            extra_data = bytes.fromhex(entry['extra_data'])
            continue

        name = entry['var']
        inp = Input(
            name=name,
            private=entry['private'],
            value=entry.get('value'),
            buffers=entry.get('buffers'),
            struct=entry.get('struct'),
        )
        variables[name] = inp

    return ProgramInputs(variables=variables, extra_data=extra_data)


# ── Array / field metadata ───────────────────────────────────────────────

class ArrayInfo:
    def __init__(self, mem_map, base_ptr, offset, elem_size, num_elements):
        self.mem_map = mem_map
        self.base_ptr = base_ptr
        self.offset = offset
        self.elem_size = elem_size
        self.num_elements = num_elements

    def __str__(self):
        return f"ArrayInfo(mem_map={self.mem_map}, base_ptr={self.base_ptr}, elem_size={self.elem_size}, offset={self.offset}, num_elements={self.num_elements})"

    def __repr__(self):
        return self.__str__()


class FieldInfo:
    def __init__(self, var_name, mem_map, base_ptr, size):
        self.var_name = var_name
        self.mem_map = mem_map
        self.base_ptr = base_ptr
        self.size = size

    def __str__(self):
        return f"FieldInfo(var_name={self.var_name}, mem_map={self.mem_map}, base_ptr={self.base_ptr}, size={self.size})"

    def __repr__(self):
        return self.__str__()


def process_field_stmt(stmt, is_shadow):
    var_name = stmt.get_attribute("name")[0].name
    field_info = stmt.get_attribute("field")
    base_ptr = field_info[2]
    if is_shadow:
        mem_map = f"{field_info[1].name}.shadow"
        var_name = f"{var_name}.shadow"
    else:
        mem_map = field_info[1].name
    size = int(field_info[3].value)
    return FieldInfo(var_name, mem_map, base_ptr, size)


def process_array_stmt(stmt, is_shadow):
    array_info = stmt.get_attribute("array")
    offset = array_info[2]
    base_ptr_vars = extract_boogie_variables(offset)
    assert len(base_ptr_vars) == 1, f"Expected 1 base pointer variable, got {base_ptr_vars}"
    base_ptr = base_ptr_vars.pop()

    if is_shadow:
        mem_map = f"{array_info[1].name}.shadow"
        base_ptr = f"{base_ptr.name}.shadow"
    else:
        mem_map = array_info[1].name
        base_ptr = base_ptr.name
    elem_size = int(array_info[3].value)
    num_elements = int(array_info[4].value)
    return ArrayInfo(mem_map, base_ptr, offset, elem_size, num_elements)


def gather_field_info_stmts(proc):
    assert isinstance(proc, ImplementationDeclaration), f"{type(proc)}"
    field_info_stmts = []
    seen_offsets = set()
    for block in proc.body.blocks:
        for stmt in block.statements:
            if isinstance(stmt, CallStatement):
                if RE_SMACK.match(stmt.procedure.name):
                    if stmt.has_attribute("field"):
                        name_attr = stmt.get_attribute("name")
                        field_attr = stmt.get_attribute("field")
                        key = (name_attr[0].name, str(field_attr[2]))
                        if key not in seen_offsets:
                            seen_offsets.add(key)
                            field_info_stmts.append(stmt)
    return field_info_stmts


def gather_array_info_stmts(proc):
    assert isinstance(proc, ImplementationDeclaration), f"{type(proc)}"
    array_info_stmts = []
    for block in proc.body.blocks:
        for stmt in block.statements:
            if isinstance(stmt, CallStatement):
                if RE_SMACK.match(stmt.procedure.name):
                    if not stmt.has_attribute("array"):
                        continue
                    array_info_stmts.append(stmt)
    return array_info_stmts


def preprocess_external_inputs(proc):
    """Extract array and field metadata from BPL annotations.

    Returns:
        arr_map:   dict[name, list[ArrayInfo]]  — keyed by {:name} attribute
        field_map: dict[name, list[FieldInfo]]  — keyed by {:name} attribute
    """
    arr_map = defaultdict(list)
    field_map = defaultdict(list)
    array_stmts = gather_array_info_stmts(proc)
    field_stmts = gather_field_info_stmts(proc)
    for stmt in array_stmts:
        name_attr = stmt.get_attribute("name")
        name = name_attr[0].name if name_attr else None

        arr_info = process_array_stmt(stmt, False)
        key = name if name else arr_info.base_ptr
        arr_map[key].append(arr_info)

        arr_info_shadow = process_array_stmt(stmt, True)
        shadow_key = f"{key}.shadow" if name else arr_info_shadow.base_ptr
        arr_map[shadow_key].append(arr_info_shadow)

    for stmt in field_stmts:
        field_info = process_field_stmt(stmt, False)
        field_map[field_info.var_name].append(field_info)
        field_info_shadow = process_field_stmt(stmt, True)
        field_map[field_info_shadow.var_name].append(field_info_shadow)
    return arr_map, field_map
