from parser.expression import FunctionApplication, MapSelect, StorageIdentifier, ProcedureIdentifier, BinaryExpression, UnaryExpression, BooleanLiteral, IntegerLiteral, OldExpression, LogicalNegation, QuantifiedExpression, IfExpression, Identifier
from parser.declaration import ImplementationDeclaration
from parser.statement import CallStatement, AssertStatement, AssumeStatement, Block
from parser.type import BooleanType, IntegerType, CustomType, MapType
import json
import re
from collections import defaultdict
import pickle
import hashlib
import zlib

FINISH_PC = -1337
ERROR_PC = -1338

ASSERT_IGNORE_LIST = set(
    ["__VERIFIER_assert",
    "__SMACK_check_overflow",
    "__SMACK_loop_exit"])


RE_VERIFIER  = re.compile(r'^__VERIFIER_nondet_[A-Za-z0-9_]+$')
RE_RECORD    = re.compile(r'^boogie_si_record_(?:i[0-9]+|ref)$')
RE_INIT_XP   = re.compile(r'^\$initialize\.cross_product$')
RE_ALLOC     = re.compile(r'^\$alloc$')
RE_ATOMIC    = re.compile(r'^corral_atomic_(?:begin|end)$')
RE_PRINTF = re.compile(
    r'^printf\.ref(?:\.(?:i[0-9]+|bool|ref))*'   # ← added “ref”
    r'(?:\.cross_product)?$'
)
RE_SMACK     = re.compile(r'^__SMACK_values?[^ ]*$')
RE_SMACK_VALUE  = re.compile(r"__SMACK_value\.(?:ref|i\d+)\b")
RE_LLVM_LIFETIME = re.compile(r'@?llvm\.lifetime\.(?:start|end)\.[^\s(]+')

CALL_IGNORE_FN_PATTERNS = (
    RE_VERIFIER,
    RE_RECORD,
    RE_INIT_XP,
    RE_ALLOC,
    RE_ATOMIC,
    RE_PRINTF,
    RE_SMACK,
    RE_LLVM_LIFETIME,
)

CALL_IGNORE_LIST = set(
    ["__VERIFIER_nondet_int",
    "$initialize.cross_product",
    "$alloc",
    "corral_atomic_begin",
    "corral_atomic_end",
    "__SMACK_value.ref.cross_product",
    ])

CALL_RECORD_LIST = set(
    ["boogie_si_record_i8",
    "boogie_si_record_i32",
    "boogie_si_record_i64",
    "boogie_si_record_ref",
    ])

boogie_type_bitwidth = {
    "i1": 1,
    "bool": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "ref": 64
}

def convert_type_to_bitwidth(type_):
    if isinstance(type_, BooleanType):
        return 1
    elif isinstance(type_, IntegerType):
        return 32
    elif isinstance(type_, CustomType):
        if type_.name == "i1":
            return 1
        elif type_.name == "i8":
            return 8
        elif type_.name == "i16":
            return 16
        elif type_.name == "i32":
            return 32
        elif type_.name == "i64":
            return 64
        elif type_.name == "bool":
            return 1
        elif type_.name == "ref":
            return 64
        elif type_.name == "$mop":
            return 64
    elif isinstance(type_, MapType):
        domain = [convert_type_to_bitwidth(t) for t in type_.domain]
        elementSort = convert_type_to_bitwidth(type_.range)
        
        if len(domain) == 1:
            return [domain[0], elementSort]
        else:
            assert False
    assert False, f"unknown type {type_} {type(type_)}"

def extract_boogie_variables(stmt) -> set[Identifier]:
    boogie_vars = set()
    for x in stmt.each():
        if isinstance(x, StorageIdentifier):
            boogie_vars.add(x)
    return boogie_vars

def mask_bits(value: int, bit_width: int) -> int:
    """Extracts the first 'bit_width' bits of an integer."""
    mask = (1 << bit_width) - 1  # Create a mask with 'bit_width' 1s
    return value & mask  # Apply the mask to extract the bits

from dataclasses import dataclass, replace

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
                        # Deduplicate by (name, base_ptr string) to handle
                        # cases where both __SMACK_value and __SMACK_values
                        # generate {:field} for the same struct field.
                        name_attr = stmt.get_attribute("name")
                        field_attr = stmt.get_attribute("field")
                        key = (name_attr[0].name, str(field_attr[2]))
                        if key not in seen_offsets:
                            seen_offsets.add(key)
                            field_info_stmts.append(stmt)
    return field_info_stmts

# Gather all array metadata statements
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

def is_assert(stmt):
    if isinstance(stmt, AssertStatement):
        return True
    elif isinstance(stmt, CallStatement):
        if stmt.procedure.name == "__VERIFIER_assert":
            return True
    elif isinstance(stmt, FunctionApplication):
        if stmt.function.name == "__VERIFIER_assert":
            return True
    else:
        return False

def extract_first_assume_stmt(block):
    for stmt in block.statements:
        if isinstance(stmt, AssertStatement) and stmt.has_attribute("hhoudini"):
            continue
        elif isinstance(stmt, AssumeStatement):
            if stmt.has_attribute("sourceloc") or stmt.has_attribute("verifier.code"):
                continue
            return stmt
        else:
            assert False, f"Unknown statement type: {stmt} {type(stmt)}"
    assert False, f"No assume statement found in {block.name}"

def generate_label_to_block(program):
    label_to_block = {}
    for decl in program.declarations:
        if isinstance(decl, ImplementationDeclaration):
            for block in decl.body.blocks:
                label_to_block[block.name] = block
    return label_to_block


def lshr_32(value: int, shift: int) -> int:
    """Simulates logical shift right (lshr) for a 32-bit unsigned integer."""
    value &= 0xFFFFFFFF  # Ensure it's treated as an unsigned 32-bit integer
    return (value >> shift) & 0xFFFFFFFF  # Logical shift with zero-fill

def lshr_64(value: int, shift: int) -> int:
    """Simulates logical shift right (lshr) for a 64-bit unsigned integer."""
    value &= 0xFFFFFFFFFFFFFFFF  # Ensure it's treated as an unsigned 64-bit integer
    return (value >> shift) & 0xFFFFFFFFFFFFFFFF  # Logical shift with zero-fill

def ashr_32(value: int, shift: int) -> int:
    """Simulates arithmetic shift right (ashr) for a 32-bit signed integer."""
    value &= 0xFFFFFFFF  # Ensure it's within 32-bit range
    if value & 0x80000000:  # If negative (MSB is set)
        return ((value >> shift) | (0xFFFFFFFF << (32 - shift))) & 0xFFFFFFFF
    else:
        return (value >> shift) & 0xFFFFFFFF  # Standard shift

def ashr_64(value: int, shift: int) -> int:
    """Simulates arithmetic shift right (ashr) for a 64-bit signed integer."""
    value &= 0xFFFFFFFFFFFFFFFF  # Ensure it's within 64-bit range
    if value & 0x8000000000000000:  # If negative (MSB is set)
        return ((value >> shift) | (0xFFFFFFFFFFFFFFFF << (64 - shift))) & 0xFFFFFFFFFFFFFFFF
    else:
        return (value >> shift) & 0xFFFFFFFFFFFFFFFF  # Standard shift

def _mask(w):        return (1 << w) - 1
def _sign(x, w):     return x if x < (1 << (w-1)) else x - (1 << w)

# ── helper: interpret value as two-complement signed --------------------------
def _to_signed(val: int, bits: int) -> int:
    val &= (1 << bits) - 1                     # truncate
    return val - (1 << bits) if val & (1 << (bits - 1)) else val

def sdiv_fn(w):
    """signed division, result truncated to w bits"""
    m = _mask(w)
    def _sdiv(x, y):
        sx, sy = _sign(x, w), _sign(y, w)
        if sy == 0:            # mimic LLVM / Boogie UB-free style
            raise ZeroDivisionError("division by zero")
        return (_sign(sx // sy, w)) & m
    return _sdiv

def trunc_fn(dst_bits: int):
    if dst_bits <= 0:
        raise ValueError("dst_bits must be a positive integer")
    mask = (1 << dst_bits) - 1          # compute once, close over it
    return lambda value, _mask=mask: value & _mask

def sext_fn(src_bits: int, dst_bits: int):
    if dst_bits < src_bits:
        raise ValueError("dst_bits must be >= src_bits")

    src_mask = (1 << src_bits) - 1           # 0b00..0011..11  (src_bits ones)
    dst_mask = (1 << dst_bits) - 1           # 0b00..0011..11  (dst_bits ones)
    sign_bit = 1 << (src_bits - 1)           # MSB of the source width

    def _sext(x: int) -> int:
        x &= src_mask                        # keep only src_bits
        if x & sign_bit:                     # negative in two’s complement?
            x |= ~src_mask                   #   -> extend with ones
        return x & dst_mask                  # final dst_bits-wide vector

    return _sext

def slt_fn(src_bits: int):
    """
    Build a signed-less-than comparator for `src_bits`-wide operands.
    Result is **always 0 or 1** (1-bit integer).
    """
    def _slt(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) < _to_signed(y, src_bits))
    return _slt

def sle_fn(src_bits: int):
    """
    Build a signed-less-or-equal (≤) comparator for `src_bits`-wide operands.
    Result is always the 1-bit integer 0 or 1.
    """
    def _sle(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) <= _to_signed(y, src_bits))
    return _sle

def sgt_fn(src_bits: int):
    """
    Build a signed-greater-than (>) comparator for `src_bits`-wide operands.
    Result is always 0 or 1 (single-bit integer).
    """
    def _sgt(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) > _to_signed(y, src_bits))
    return _sgt


def sge_fn(src_bits: int):
    """
    Build a signed-greater-or-equal (≥) comparator for `src_bits`-wide operands.
    Result is always 0 or 1 (single-bit integer).
    """
    def _sge(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) >= _to_signed(y, src_bits))
    return _sge

def srem_fn(bits: int):
    """
    Build a signed-remainder function for `bits`-wide operands.
    Result is truncated back to `bits` so it fits the bit-vector width.
    """
    mask = (1 << bits) - 1

    def _srem(x: int, y: int) -> int:
        sx = _to_signed(x, bits)
        sy = _to_signed(y, bits)
        if sy == 0:
            raise ZeroDivisionError("srem by zero")

        # truncating division toward zero
        q = int(sx / sy)
        r = sx - q * sy                # remainder with sign of dividend
        return r & mask                # back to unsigned bit-vector form

    return _srem

def xor_fn(bits: int):
    m = _mask(bits)
    def _xor(x: int, y: int) -> int:
        return ((x & m) ^ (y & m)) & m
    return _xor


def or_fn(bits: int):
    m = _mask(bits)
    def _or(x: int, y: int) -> int:
        return ((x & m) | (y & m)) & m
    return _or

def and_fn(bits: int):
    m = _mask(bits)
    def _and(x: int, y: int) -> int:
        return ((x & m) & (y & m)) & m
    return _and

def add_fn(bits: int):
    m = _mask(bits)
    def _add(x: int, y: int) -> int:
        return ((x & m) + (y & m)) & m
    return _add

def sub_fn(bits: int):
    m = _mask(bits)
    def _sub(x: int, y: int) -> int:
        return ((x & m) - (y & m)) & m
    return _sub

def not_fn(bits: int):
    m = _mask(bits)
    def _not(x: int) -> int:
        return (~x) & m
    return _not

def mul_fn(bits: int):
    m = _mask(bits)
    def _mul(x: int, y: int) -> int:
        return ((x & m) * (y & m)) & m
    return _mul

def eq_fn(bits: int):
    m = _mask(bits)
    def _eq(x: int, y: int) -> int:
        return int((x & m) == (y & m))   # 0 or 1
    return _eq

def ne_fn(bits: int):
    m = _mask(bits)
    def _ne(x: int, y: int) -> int:
        return int((x & m) != (y & m))   # 0 or 1
    return _ne

def udiv_fn(bits: int):
    m = _mask(bits)

    def _udiv(x: int, y: int) -> int:
        return ((x & m) // (y & m)) & m    # Python // is already floor for unsigned
    return _udiv

def ult_fn(bits: int):
    m = _mask(bits)
    def _ult(x: int, y: int) -> int:
        return int((x & m) < (y & m))
    return _ult

def ugt_fn(bits: int):
    m = _mask(bits)
    def _ugt(x: int, y: int) -> int:
        return int((x & m) > (y & m))
    return _ugt

def uge_fn(bits: int):
    m = _mask(bits)
    def _uge(x: int, y: int) -> int:
        return int((x & m) >= (y & m))
    return _uge

def ule_fn(bits: int):
    m = _mask(bits)
    def _ule(x: int, y: int) -> int:
        return int((x & m) <= (y & m))          # 0 or 1
    return _ule

def urem_fn(bits: int):
    m = _mask(bits)
    def _urem(x: int, y: int) -> int:
        return ((x & m) % (y & m)) & m          # ZeroDivisionError if y == 0
    return _urem

def shl_fn(bits: int):
    m = _mask(bits)
    sa_mask = bits - 1                      # mask for shift amount

    def _shl(x: int, y: int) -> int:
        return ((x & m) << (y & sa_mask)) & m

    return _shl

fn_map_to_op = {
    # ——— bit-vector arithmetic ————————————————————————————————————————————
    "$mul.ref": (mul_fn(64), 2, 64, 64),
    "$mul.i64": (mul_fn(64), 2, 64, 64),
    "$mul.i32": (mul_fn(32), 2, 32, 32),
    "$mul.i8":  (mul_fn(8), 2,  8,  8),

    "$add.ref": (add_fn(64), 2, 64, 64),
    "$add.i64": (add_fn(64), 2, 64, 64),
    "$add.i32": (add_fn(32), 2, 32, 32),
    "$add.i8":  (add_fn(8), 2,  8,  8),

    "$sub.ref": (sub_fn(64), 2, 64, 64),
    "$sub.i64": (sub_fn(64), 2, 64, 64),
    "$sub.i32": (sub_fn(32), 2, 32, 32),
    "$sub.i16": (sub_fn(16), 2, 16, 16),
    "$sub.i8":  (sub_fn(8), 2,  8,  8),

    # Unary
    "$not.i1": (not_fn(1), 1, 1, 1),
    "$not.i8": (not_fn(8), 1, 8, 8),
    "$not.i32": (not_fn(32), 1, 32, 32),
    "$not.i64": (not_fn(64), 1, 64, 64),
    "$not.ref": (not_fn(64), 1, 64, 64),
    "$not.i16": (not_fn(16), 1, 16, 16),

    # ——— bit-vector bit-wise ————————————————————————————————————————————
    "$and.ref": (and_fn(64), 2, 64, 64),
    "$and.i64": (and_fn(64), 2, 64, 64),
    "$and.i32": (and_fn(32), 2, 32, 32),
    "$and.i8":  (and_fn(8), 2,  8,  8),
    "$and.i1":  (and_fn(1), 2,  1,  1),

    "$or.ref":  (or_fn(64), 2, 64, 64),
    "$or.i64":  (or_fn(64), 2, 64, 64),
    "$or.i32":  (or_fn(32), 2, 32, 32),
    "$or.i8":   (or_fn(8), 2,  8,  8),
    "$or.i1":   (or_fn(1), 2,  1,  1),

    "$xor.ref": (xor_fn(64), 2, 64, 64),
    "$xor.i64": (xor_fn(64), 2, 64, 64),
    "$xor.i32": (xor_fn(32), 2, 32, 32),
    "$xor.i8":  (xor_fn(8), 2,  8,  8),
    "$xor.i1":  (xor_fn(1), 2,  1,  1),

    # ——— (in)equalities ————————————————————————————————————————————————
    "$ne.ref": (ne_fn(64), 2, 64, 64),
    "$ne.i64": (ne_fn(64), 2, 64, 64),
    "$ne.i32": (ne_fn(32), 2, 32, 32),
    "$ne.i8":  (ne_fn(8), 2,  8,  8),

    "$eq.ref": (eq_fn(64), 2, 64, 64),
    "$eq.i64": (eq_fn(64), 2, 64, 64),
    "$eq.i32": (eq_fn(32), 2, 32, 32),
    "$eq.i8":  (eq_fn(8), 2,  8,  8),
    "$eq.i1":  (eq_fn(1), 2,  1,  1),

    # ——— unsigned division / comparisons ——————————————————————————————
    "$udiv.ref": (udiv_fn(64), 2, 64, 64),
    "$udiv.i64": (udiv_fn(64), 2, 64, 64),
    "$udiv.i32": (udiv_fn(32), 2, 32, 32),
    "$udiv.i8":  (udiv_fn(8), 2,  8,  8),
    "$sdiv.i64": (sdiv_fn(64), 2, 64, 64),
    "$sdiv.i32": (sdiv_fn(32), 2, 32, 32),

    "$ult.ref": (ult_fn(64), 2, 64, 64),
    "$ult.i64": (ult_fn(64), 2, 64, 64),
    "$ult.i32": (ult_fn(32), 2, 32, 32),

    "$ugt.i64": (ugt_fn(64), 2, 64, 64),
    "$ugt.i32": (ugt_fn(32), 2, 32, 32),

    "$uge.i64": (uge_fn(64), 2, 64, 64),
    "$uge.i32": (uge_fn(32), 2, 32, 32),

    "$sgt.ref.bool": (sgt_fn(64), 2, 64, bool),
    "$sgt.i32":      (sgt_fn(32), 2, 32, 32),
    "$sgt.i64":      (sgt_fn(64), 2, 64, 64),
    "$sge.ref.bool": (sge_fn(64), 2, 64, bool),
    "$sge.i32":      (sge_fn(32), 2, 32, 32),
    "$sge.i64":      (sge_fn(64), 2, 64, 64),

    "$sle.i32": (sle_fn(32), 2, 32, 32),
    "$sle.i64": (sle_fn(64), 2, 64, 64),
    "$sle.ref.bool": (sle_fn(64), 2, 64, bool),

    "$slt.ref.bool": (slt_fn(64), 2, 64, bool),
    "$slt.i64":      (slt_fn(64), 2, 64, 64),
    "$slt.i32":      (slt_fn(32), 2, 32, 32),
    "$slt.i8":       (slt_fn(8), 2,  8,  8),

    "$ule.i64": (ule_fn(64), 2, 64, 64),
    "$ule.i32": (ule_fn(32), 2, 32, 32),
    "$ule.i8":  (ule_fn(8), 2,  8,  8),

    # ——— remainders ————————————————————————————————————————————————
    "$urem.i64": (urem_fn(64), 2, 64, 64),
    "$urem.i32": (urem_fn(32), 2, 32, 32),
    "$urem.i8":  (urem_fn(8), 2,  8,  8),

    "$srem.i64": (srem_fn(64), 2, 64, 64),
    "$srem.i32": (srem_fn(32), 2, 32, 32),
    "$srem.i8":  (srem_fn(8), 2,  8,  8),

    # ——— shifts ————————————————————————————————————————————————
    "$shl.i64": (shl_fn(64), 2, 64, 64),
    "$shl.i32": (shl_fn(32), 2, 32, 32),

    "$lshr.i64": (lshr_64, 2, 64, 64),
    "$lshr.i32": (lshr_32, 2, 32, 32),

    "$ashr.i64": (ashr_64, 2, 64, 64),
    "$ashr.i32": (ashr_32, 2, 32, 32),

    # ——— casts / identity ————————————————————————————————————————————
    "$bitcast.ref.ref": (lambda x: x, 1, 64, 64),
    "$p2i.ref.i64": (lambda x: x, 1, 64, 64),
    "$i2p.i64.ref": (lambda x: x, 1, 64, 64),
    
    # ——— generic Boolean / arithmetic operators (for parser convenience) —
    "==":  (lambda x, y: x == y, 2, None, None),
    "!=":  (lambda x, y: x != y, 2, None, None),
    "==>": (None,                    2, None, None),  # implication, handled separately
    "||":  (lambda x, y: x |  y, 2, bool, bool),
    "&&":  (lambda x, y: x &  y, 2, bool, bool),
    "<":   (lambda x, y: x <  y, 2, None, None),
    ">":   (lambda x, y: x >  y, 2, None, None),
    "<=":  (lambda x, y: x <= y, 2, None, None),
    ">=":  (lambda x, y: x >= y, 2, None, None),
    "+":   (lambda x, y: x +  y, 2, None, None),
    "-":   (lambda x, y: x -  y, 2, None, None),
    "*":   (lambda x, y: x *  y, 2, None, None),
    "/":   (lambda x, y: x // y, 2, None, None),
}

def generate_function_map():
    """
    Returns a lazily initialized more_func_map using the provided stateful solver.
    """
    return {
        "$sext.i32.i64": (sext_fn(32, 64), 1, 32, 64),
        "$sext.i8.i32": (sext_fn(8, 32), 1, 8, 32),
        "$sext.i16.i32": (sext_fn(16, 32), 1, 16, 32),
        "$sext.i32.i64": (sext_fn(32, 64), 1, 32, 64),
        "$zext.i32.i64": (lambda x: x, 1, 32, 64),
        "$zext.i8.i32": (lambda x: x, 1, 8, 32),
        "$zext.i8.i64": (lambda x: x, 1, 8, 64),
        "$zext.i1.i32": (lambda x: x, 1, 1, 32),
        "$zext.i1.i64": (lambda x: x, 1, 1, 64),
        "$zext.i16.i32": (lambda x: x, 1, 16, 32),
        "$zext.i16.i64": (lambda x: x, 1, 16, 64),
        "$trunc.i32.i8": (trunc_fn(8), 1, 32, 8),
        "$trunc.i32.i16": (trunc_fn(16), 1, 32, 16),
        "$trunc.i64.i8": (trunc_fn(8), 1, 64, 8),
        "$trunc.i64.i16": (trunc_fn(16), 1, 64, 16),
        "$trunc.i64.i32": (trunc_fn(32), 1, 64, 32),
        "$trunc.i32.i1": (trunc_fn(1), 1, 32, 1),
        "$trunc.i64.i1": (trunc_fn(1), 1, 64, 1),
    } | fn_map_to_op

def collect_all_label_conditions(proc):
    label_to_conditions = defaultdict(set)
    ret_label_to_conditions = defaultdict(set)

    if not proc.body:
        return label_to_conditions
    for block in proc.body.blocks:
        if len(block.statements) > 0 and isinstance(block.statements[0], AssumeStatement):
            for stmt in block.statements:
                if not isinstance(stmt, AssumeStatement):
                    break
                if isinstance(stmt.expression, BooleanLiteral):
                    continue
                label_to_conditions[block.name].add(stmt)
            # stmt = block.statements[0]
            # while(isinstance(stmt, AssumeStatement)):
            #     if isinstance(stmt.expression, BooleanLiteral):
            #         stmt = stmt.next_sibling()
            #         continue
            #     label_to_conditions[block.name].add(stmt)
            #     stmt = stmt.next_sibling()
    
    for label in label_to_conditions:
        # if len(label_to_conditions[label]) > 1:
        #     print(f"HAS MORE THAN ONE: {label_to_conditions[label]}")
        ret_label_to_conditions[label] = list(label_to_conditions[label])[0]
    return ret_label_to_conditions


def initialize_code_metadata(proc):
        # Initialize label_to_pc and pc_to_stmt
        pc_to_stmt = {}
        label_to_pc = {}
        pc_to_block = {}
        pc = 0
        for block in proc.body.blocks:
            label_to_pc[block.name] = pc
            for stmt in block.statements:
                # Initialize pc_next_map 
                pc_to_stmt[pc] = stmt
                pc_to_block[pc] = block.name
                pc += 1

        pc_to_label = {}
        for label in label_to_pc:
            pc_to_label[label_to_pc[label]] = label
        return pc_to_stmt, label_to_pc, pc_to_block, pc_to_label

def stmt_to_block(stmt):
    return next((b for b in stmt.each_ancestor() if isinstance(b, Block)), None)

from functools import lru_cache

@lru_cache(maxsize=10000)
def _cached_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def get_state(serialized_state_key: bytes, state_cache):
    sha256_hex = _cached_sha256(serialized_state_key)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.get(f"state_key_{sha256_hex}")
    pipe.get(f"coi_key_{sha256_hex}")
    serialized_state, serialized_coi = pipe.execute()
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        if serialized_coi:
            state.coi = pickle.loads(zlib.decompress(serialized_coi))
        if state.coi:
            state.coi.deserialize(state_cache)
        return state
    else:
        return None

def get_state_only(serialized_state_key: bytes, state_cache):
    sha256_hex = _cached_sha256(serialized_state_key)
    serialized_state = state_cache.redis_runtime.get(f"state_key_{sha256_hex}")
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        return state
    else:
        return None

def get_state_raw(redis_key: str, state_cache):
    serialized_state = state_cache.redis_runtime.get(redis_key)
    if serialized_state:
        state = pickle.loads(zlib.decompress(serialized_state))
        if state.coi:
            state.coi.deserialize(state_cache)
        return state
    else:
        return None

def find_unpicklable(obj, path="root"):
    from utils.utils_cvc5 import HollowCvc5Term
    """
    Recursively searches for objects of type HollowCvc5Term.
    """
    # 1. Check specifically for the HollowCvc5Term instance
    # Ensure HollowCvc5Term is imported/available in this scope
    if isinstance(obj, HollowCvc5Term):
        print(f"FOUND CULPRIT at {path}: {type(obj)} -> {obj}")
        return

    # 2. Recursively search containers
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_unpicklable(v, f"{path}['{k}']")
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i, v in enumerate(obj):
            find_unpicklable(v, f"{path}[{i}]")
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            find_unpicklable(v, f"{path}.{k}")

def put_state(serialized_state_key : bytes, state, state_cache):
    # try:
    serialized_state = pickle.dumps(state)
    try:
        serialized_coi = pickle.dumps(state.coi)
    except Exception as e:
        print(f"Error serializing state: {e}")
        # find_unpicklable(state)
        find_unpicklable(state.coi)
        assert False, f"Error serializing state: {e}"
    sha256_hex = _cached_sha256(serialized_state_key)
    compressed_state = zlib.compress(serialized_state)
    compressed_coi = zlib.compress(serialized_coi)
    pipe = state_cache.redis_runtime.pipeline()
    pipe.set(f"state_key_{sha256_hex}", compressed_state)
    pipe.set(f"coi_key_{sha256_hex}", compressed_coi)
    pipe.execute()

def create_df_key(target_serialized, serialized_key):
    sha256_hex_target = _cached_sha256(target_serialized)
    sha256_hex_key = _cached_sha256(serialized_key)
    return f"df_key_{sha256_hex_target}_{sha256_hex_key}"

def put_df(df, df_serialized_key, state_cache):
    serialized_df = pickle.dumps(df)
    compressed_df = zlib.compress(serialized_df)
    state_cache.redis_runtime.set(df_serialized_key, compressed_df)

def get_df(df_serialized_key, state_cache):
    compressed_df = state_cache.redis_runtime.get(df_serialized_key)
    if compressed_df:
        df = pickle.loads(zlib.decompress(compressed_df))
        df.deserialize(state_cache)
        return df
    else:
        return None

import threading
import logging
import builtins
import time

class IndentLogger:
    _local = threading.local()
    _logger = logging.getLogger("ray")

    # Global toggle for direct printing
    _direct_print_mode = True
    # Minimum elapsed time (seconds) to show timing on block exit
    _timing_threshold = 0.1

    def __init__(self, label=None, *args):
        """Create a context manager instance.

        Usage:
            with indent_log:                              # silent indent (backward compat)
            with indent_log("tag", "msg %r", val):        # logs on enter/exit with timing
        """
        self._label = label
        self._args = args
        self._start_time = None

    def __call__(self, label, *args):
        """Create a labeled context manager: with indent_log("tag", "msg", ...):"""
        return IndentLogger(label, *args)

    @classmethod
    def set_direct_print(cls, enabled: bool):
        cls._direct_print_mode = enabled

    @classmethod
    def set_timing_threshold(cls, seconds: float):
        """Set minimum elapsed seconds to display timing. 0 = always show."""
        cls._timing_threshold = seconds

    @classmethod
    def _get_level(cls):
        if not hasattr(cls._local, 'level'):
            cls._local.level = 0
        return cls._local.level

    @classmethod
    def _get_indent(cls):
        return "  " * cls._get_level()

    @classmethod
    def _format_msg(cls, *args):
        if not args:
            return ""
        if len(args) > 1 and isinstance(args[0], str) and '%' in args[0]:
            try:
                return args[0] % args[1:]
            except TypeError:
                return " ".join(map(str, args))
        return " ".join(map(str, args))

    @classmethod
    def _output(cls, level, msg, direct_print=None):
        use_print = direct_print if direct_print is not None else cls._direct_print_mode
        if not use_print and not cls._logger.isEnabledFor(level):
            return
        indented_msg = f"{cls._get_indent()}{msg}"
        if use_print:
            import builtins
            builtins.print(indented_msg)
        else:
            cls._logger.log(level, indented_msg)

    @classmethod
    def log(cls, level, *args, direct_print=None):
        use_print = direct_print if direct_print is not None else cls._direct_print_mode
        if not use_print and not cls._logger.isEnabledFor(level):
            return
        msg = cls._format_msg(*args)
        cls._output(level, msg, direct_print=direct_print)

    @classmethod
    def info(cls, *args, **kwargs): cls.log(logging.INFO, *args, **kwargs)
    @classmethod
    def debug(cls, *args, **kwargs): cls.log(logging.DEBUG, *args, **kwargs)
    @classmethod
    def error(cls, *args, **kwargs): cls.log(logging.ERROR, *args, **kwargs)

    @classmethod
    def debug_items(cls, header, items, **kwargs):
        """Print a header then each item on its own indented line.

        Usage:
            IndentLogger.debug_items("[tag] candidates:", candidate_set)
            IndentLogger.debug_items("[tag] keys:", key_list, max_items=10)
        """
        max_items = kwargs.get('max_items', 50)
        cls.debug(header)
        items_list = list(items) if not isinstance(items, list) else items
        for i, item in enumerate(items_list):
            if i >= max_items:
                cls.debug("  ... and %d more", len(items_list) - max_items)
                break
            cls.debug("  - %r", item)

    # Context manager
    def __enter__(self):
        if self._label:
            msg = f"[{self._label}]"
            if self._args:
                msg += " " + self._format_msg(*self._args)
            self._output(logging.DEBUG, f"▶ {msg}")
        if not hasattr(self._local, 'level'):
            self._local.level = 0
        self._local.level += 1
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._local.level -= 1
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self._label:
            status = "✗" if exc_type else "✓"
            timing = ""
            if elapsed >= self._timing_threshold:
                if elapsed >= 60:
                    timing = f" ({elapsed/60:.1f}m)"
                elif elapsed >= 1:
                    timing = f" ({elapsed:.1f}s)"
                else:
                    timing = f" ({elapsed*1000:.0f}ms)"
            self._output(logging.DEBUG, f"{status} [{self._label}]{timing}")

# Global instance — works both as bare context manager and as callable
indent_log = IndentLogger()