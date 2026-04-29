"""Structure-aware mutations for ProgramInputs.

Operates on the ProgramInputs data model without touching the Boogie IR.
All mutations preserve structural validity (buffer sizes, struct layouts).
"""

import copy
import random
from dataclasses import replace

from interpreter.utils.inputs import Input, ProgramInputs

# Boundary values that commonly trigger edge cases in programs
_BYTE_BOUNDS = [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF]
_INT_BOUNDS = [0, 1, -1, 127, 128, 255, 256, 65535, 65536,
               2**31 - 1, 2**31, -(2**31)]


def mutate(prog_inputs: ProgramInputs) -> ProgramInputs:
    """Apply one random structure-aware mutation. Returns a new ProgramInputs."""
    mutable = [name for name, inp in prog_inputs.variables.items()
               if (inp.value is not None or inp.buffers
                   or inp.struct or inp.havoc_seq is not None)]
    if not mutable:
        return prog_inputs

    name = random.choice(mutable)
    inp = prog_inputs.variables[name]
    new_inp = _mutate_input(inp)

    new_vars = dict(prog_inputs.variables)
    new_vars[name] = new_inp
    return ProgramInputs(variables=new_vars, extra_data=prog_inputs.extra_data)


def splice(a: ProgramInputs, b: ProgramInputs) -> ProgramInputs:
    """Crossover: take a prefix of variables from a, suffix from b."""
    keys = list(a.variables.keys())
    if len(keys) < 2:
        return mutate(a)
    split = random.randint(1, len(keys) - 1)
    new_vars = {}
    for k in keys[:split]:
        new_vars[k] = copy.deepcopy(a.variables[k])
    for k in keys[split:]:
        src = b.variables.get(k, a.variables[k])
        new_vars[k] = copy.deepcopy(src)
    return ProgramInputs(variables=new_vars, extra_data=a.extra_data)


# ── Per-input-type mutators ──────────────────────────────────────────────────

def _mutate_input(inp: Input) -> Input:
    if inp.value is not None:
        return replace(inp, value=_mutate_scalar(inp.value))
    if inp.buffers:
        return replace(inp, buffers=[_mutate_buffer(b) for b in inp.buffers])
    if inp.struct:
        return replace(inp, struct=_mutate_struct(inp.struct))
    if inp.havoc_seq is not None:
        return replace(inp, havoc_seq=_mutate_havoc(inp.havoc_seq))
    return inp


def _mutate_scalar(value: int) -> int:
    op = random.randint(0, 3)
    if op == 0:
        return value + random.randint(-16, 16)
    if op == 1:
        return random.choice(_INT_BOUNDS)
    if op == 2:
        bit = random.randint(0, 31)
        return value ^ (1 << bit)
    return random.randint(0, 2**32 - 1)


def _mutate_buffer(buf: dict) -> dict:
    contents = buf.get("contents", "0x")
    size = buf.get("size", 0)
    if not size:
        return buf
    hex_str = contents[2:] if contents.startswith("0x") else contents
    # Pad or trim to size
    hex_str = hex_str.ljust(size * 2, "0")[:size * 2]
    try:
        data = bytearray.fromhex(hex_str)
    except ValueError:
        data = bytearray(size)

    op = random.randint(0, 4)
    if op == 0:
        # Bit flip at random byte
        idx = random.randint(0, len(data) - 1)
        data[idx] ^= (1 << random.randint(0, 7))
    elif op == 1:
        # Random byte replacement
        idx = random.randint(0, len(data) - 1)
        data[idx] = random.randint(0, 255)
    elif op == 2:
        # Boundary byte
        idx = random.randint(0, len(data) - 1)
        data[idx] = random.choice(_BYTE_BOUNDS)
    elif op == 3:
        # Chunk zeroing (1–8 bytes)
        start = random.randint(0, len(data) - 1)
        end = min(start + random.randint(1, 8), len(data))
        for i in range(start, end):
            data[i] = 0
    else:
        # Fill chunk with a random byte
        start = random.randint(0, len(data) - 1)
        end = min(start + random.randint(1, 8), len(data))
        fill = random.randint(0, 255)
        for i in range(start, end):
            data[i] = fill

    return {"contents": "0x" + data.hex(), "size": size}


def _mutate_struct(struct: list) -> list:
    """Mutate one field in a struct, avoiding length/size scalar fields."""
    struct = copy.deepcopy(struct)
    # Prefer buffer (pointer) fields over scalar fields
    buf_idxs = [i for i, f in enumerate(struct) if "buffer" in f]
    val_idxs = [i for i, f in enumerate(struct) if "value" in f]
    candidates = buf_idxs if buf_idxs else val_idxs
    if not candidates:
        return struct

    idx = random.choice(candidates)
    field = struct[idx]
    if "buffer" in field:
        struct[idx] = dict(field, buffer=_mutate_buffer(field["buffer"]))
    elif "value" in field:
        try:
            v = int(field["value"], 16) if isinstance(field["value"], str) else int(field["value"])
            new_v = _mutate_scalar(v) & 0xFFFF_FFFF_FFFF_FFFF
            struct[idx] = dict(field, value=f"0x{new_v:x}")
        except (ValueError, TypeError):
            pass
    return struct


def _mutate_havoc(seq: list) -> list:
    """Mutate a havoc sequence (nondet call values / loop iteration counts)."""
    seq = list(seq)
    if not seq:
        return [random.randint(0, 100)]
    op = random.randint(0, 3)
    if op == 0 and len(seq) > 1:
        seq.pop(random.randint(0, len(seq) - 1))
    elif op == 1:
        seq.insert(random.randint(0, len(seq)), random.randint(0, 100))
    elif op == 2:
        seq[random.randint(0, len(seq) - 1)] = random.randint(0, 100)
    else:
        seq.append(random.randint(0, 100))
    return seq
