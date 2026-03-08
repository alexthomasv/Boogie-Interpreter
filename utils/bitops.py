"""Bitvector arithmetic and logic operations for Boogie built-in functions."""

__all__ = [
    'mask_bits', '_mask', '_sign', '_to_signed',
    'lshr_32', 'lshr_64', 'ashr_32', 'ashr_64', 'shl_fn',
    'sdiv_fn', 'srem_fn', 'udiv_fn', 'urem_fn',
    'add_fn', 'sub_fn', 'mul_fn',
    'and_fn', 'or_fn', 'xor_fn', 'not_fn',
    'eq_fn', 'ne_fn',
    'slt_fn', 'sle_fn', 'sgt_fn', 'sge_fn',
    'ult_fn', 'ule_fn', 'ugt_fn', 'uge_fn',
    'trunc_fn', 'sext_fn',
    'fn_map_to_op', 'generate_function_map',
]


def mask_bits(value: int, bit_width: int) -> int:
    """Extracts the first 'bit_width' bits of an integer."""
    mask = (1 << bit_width) - 1
    return value & mask


def _mask(w):        return (1 << w) - 1
def _sign(x, w):     return x if x < (1 << (w-1)) else x - (1 << w)


def _to_signed(val: int, bits: int) -> int:
    """Interpret value as two's-complement signed."""
    val &= (1 << bits) - 1
    return val - (1 << bits) if val & (1 << (bits - 1)) else val


# ── Shift operations ──────────────────────────────────────────────────────

def lshr_32(value: int, shift: int) -> int:
    value &= 0xFFFFFFFF
    return (value >> shift) & 0xFFFFFFFF

def lshr_64(value: int, shift: int) -> int:
    value &= 0xFFFFFFFFFFFFFFFF
    return (value >> shift) & 0xFFFFFFFFFFFFFFFF

def ashr_32(value: int, shift: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return ((value >> shift) | (0xFFFFFFFF << (32 - shift))) & 0xFFFFFFFF
    else:
        return (value >> shift) & 0xFFFFFFFF

def ashr_64(value: int, shift: int) -> int:
    value &= 0xFFFFFFFFFFFFFFFF
    if value & 0x8000000000000000:
        return ((value >> shift) | (0xFFFFFFFFFFFFFFFF << (64 - shift))) & 0xFFFFFFFFFFFFFFFF
    else:
        return (value >> shift) & 0xFFFFFFFFFFFFFFFF

def shl_fn(bits: int):
    m = _mask(bits)
    sa_mask = bits - 1
    def _shl(x: int, y: int) -> int:
        return ((x & m) << (y & sa_mask)) & m
    return _shl


# ── Signed arithmetic ────────────────────────────────────────────────────

def sdiv_fn(w):
    m = _mask(w)
    def _sdiv(x, y):
        sx, sy = _sign(x, w), _sign(y, w)
        if sy == 0:
            raise ZeroDivisionError("division by zero")
        return (_sign(sx // sy, w)) & m
    return _sdiv

def srem_fn(bits: int):
    mask = (1 << bits) - 1
    def _srem(x: int, y: int) -> int:
        sx = _to_signed(x, bits)
        sy = _to_signed(y, bits)
        if sy == 0:
            raise ZeroDivisionError("srem by zero")
        q = int(sx / sy)
        r = sx - q * sy
        return r & mask
    return _srem


# ── Unsigned arithmetic ──────────────────────────────────────────────────

def udiv_fn(bits: int):
    m = _mask(bits)
    def _udiv(x: int, y: int) -> int:
        return ((x & m) // (y & m)) & m
    return _udiv

def urem_fn(bits: int):
    m = _mask(bits)
    def _urem(x: int, y: int) -> int:
        return ((x & m) % (y & m)) & m
    return _urem


# ── Basic arithmetic ─────────────────────────────────────────────────────

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

def mul_fn(bits: int):
    m = _mask(bits)
    def _mul(x: int, y: int) -> int:
        return ((x & m) * (y & m)) & m
    return _mul


# ── Bitwise ──────────────────────────────────────────────────────────────

def and_fn(bits: int):
    m = _mask(bits)
    def _and(x: int, y: int) -> int:
        return ((x & m) & (y & m)) & m
    return _and

def or_fn(bits: int):
    m = _mask(bits)
    def _or(x: int, y: int) -> int:
        return ((x & m) | (y & m)) & m
    return _or

def xor_fn(bits: int):
    m = _mask(bits)
    def _xor(x: int, y: int) -> int:
        return ((x & m) ^ (y & m)) & m
    return _xor

def not_fn(bits: int):
    m = _mask(bits)
    def _not(x: int) -> int:
        return (~x) & m
    return _not


# ── Comparisons ──────────────────────────────────────────────────────────

def eq_fn(bits: int):
    m = _mask(bits)
    def _eq(x: int, y: int) -> int:
        return int((x & m) == (y & m))
    return _eq

def ne_fn(bits: int):
    m = _mask(bits)
    def _ne(x: int, y: int) -> int:
        return int((x & m) != (y & m))
    return _ne

def slt_fn(src_bits: int):
    def _slt(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) < _to_signed(y, src_bits))
    return _slt

def sle_fn(src_bits: int):
    def _sle(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) <= _to_signed(y, src_bits))
    return _sle

def sgt_fn(src_bits: int):
    def _sgt(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) > _to_signed(y, src_bits))
    return _sgt

def sge_fn(src_bits: int):
    def _sge(x: int, y: int) -> int:
        return int(_to_signed(x, src_bits) >= _to_signed(y, src_bits))
    return _sge

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
        return int((x & m) <= (y & m))
    return _ule


# ── Cast / extension / truncation ────────────────────────────────────────

def trunc_fn(dst_bits: int):
    if dst_bits <= 0:
        raise ValueError("dst_bits must be a positive integer")
    mask = (1 << dst_bits) - 1
    return lambda value, _mask=mask: value & _mask

def sext_fn(src_bits: int, dst_bits: int):
    if dst_bits < src_bits:
        raise ValueError("dst_bits must be >= src_bits")
    src_mask = (1 << src_bits) - 1
    dst_mask = (1 << dst_bits) - 1
    sign_bit = 1 << (src_bits - 1)
    def _sext(x: int) -> int:
        x &= src_mask
        if x & sign_bit:
            x |= ~src_mask
        return x & dst_mask
    return _sext


# ── Function map (Boogie name → (fn, arity, in_bits, out_bits)) ──────────

fn_map_to_op = {
    # arithmetic
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

    # unary not
    "$not.i1": (not_fn(1), 1, 1, 1),
    "$not.i8": (not_fn(8), 1, 8, 8),
    "$not.i32": (not_fn(32), 1, 32, 32),
    "$not.i64": (not_fn(64), 1, 64, 64),
    "$not.ref": (not_fn(64), 1, 64, 64),
    "$not.i16": (not_fn(16), 1, 16, 16),

    # bitwise
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

    # equality
    "$ne.ref": (ne_fn(64), 2, 64, 64),
    "$ne.i64": (ne_fn(64), 2, 64, 64),
    "$ne.i32": (ne_fn(32), 2, 32, 32),
    "$ne.i8":  (ne_fn(8), 2,  8,  8),

    "$eq.ref": (eq_fn(64), 2, 64, 64),
    "$eq.i64": (eq_fn(64), 2, 64, 64),
    "$eq.i32": (eq_fn(32), 2, 32, 32),
    "$eq.i8":  (eq_fn(8), 2,  8,  8),
    "$eq.i1":  (eq_fn(1), 2,  1,  1),

    # unsigned div/cmp
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

    # remainders
    "$urem.i64": (urem_fn(64), 2, 64, 64),
    "$urem.i32": (urem_fn(32), 2, 32, 32),
    "$urem.i8":  (urem_fn(8), 2,  8,  8),

    "$srem.i64": (srem_fn(64), 2, 64, 64),
    "$srem.i32": (srem_fn(32), 2, 32, 32),
    "$srem.i8":  (srem_fn(8), 2,  8,  8),

    # shifts
    "$shl.i64": (shl_fn(64), 2, 64, 64),
    "$shl.i32": (shl_fn(32), 2, 32, 32),

    "$lshr.i64": (lshr_64, 2, 64, 64),
    "$lshr.i32": (lshr_32, 2, 32, 32),

    "$ashr.i64": (ashr_64, 2, 64, 64),
    "$ashr.i32": (ashr_32, 2, 32, 32),

    # casts / identity
    "$bitcast.ref.ref": (lambda x: x, 1, 64, 64),
    "$p2i.ref.i64": (lambda x: x, 1, 64, 64),
    "$i2p.i64.ref": (lambda x: x, 1, 64, 64),

    # generic Boolean / arithmetic operators
    "==":  (lambda x, y: x == y, 2, None, None),
    "!=":  (lambda x, y: x != y, 2, None, None),
    "==>": (None,                    2, None, None),
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
