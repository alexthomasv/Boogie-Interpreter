"""Python-side candidate materialization for Rust symbolic execution."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from interpreter.utils.inputs import ProgramInputs


_HELPER_RE = re.compile(r"^(zeros|ones)\((\d+)\)$")


@dataclass
class LazySymbolicCandidate:
    """Symbolic candidate that delays ProgramInputs deepcopy until replay."""

    base_inputs: ProgramInputs
    bindings: Mapping[str, Mapping[str, Any]]
    havoc_bound: int
    _swoosh_candidate_meta: dict
    _materialized: ProgramInputs | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._swoosh_symbolic_bindings = self.bindings
        self._swoosh_havoc_bound = self.havoc_bound

    def materialize(self) -> ProgramInputs:
        if self._materialized is None:
            self._materialized = apply_symbolic_updates(
                self.base_inputs,
                self.bindings,
                self._swoosh_candidate_meta.get("updates") or {},
                havoc_bound=self.havoc_bound,
            )
            self._materialized._swoosh_candidate_meta = self._swoosh_candidate_meta
            self._materialized._swoosh_symbolic_bindings = self.bindings
            self._materialized._swoosh_havoc_bound = self.havoc_bound
        return self._materialized


def program_inputs_cache_key(program_inputs: ProgramInputs, havoc_bound: int) -> tuple:
    variables = getattr(program_inputs, "variables", program_inputs)
    return (
        int(havoc_bound),
        bytes(getattr(program_inputs, "extra_data", None) or b""),
        tuple(
            sorted(
                (name, _input_key(inp))
                for name, inp in variables.items()
            )
        ),
    )


def apply_symbolic_updates(
    program_inputs: ProgramInputs,
    bindings: Mapping[str, Mapping[str, Any]],
    updates: Mapping[str, int],
    *,
    havoc_bound: int,
) -> ProgramInputs:
    candidate = copy.deepcopy(program_inputs)
    variables = candidate.variables

    for symbol_name, raw_value in updates.items():
        spec = bindings.get(symbol_name)
        if spec is None:
            continue
        value = int(raw_value)
        kind = spec.get("kind")

        if kind == "scalar":
            var_name = spec.get("var")
            if var_name in variables:
                variables[var_name].value = _signed_64(value)
        elif kind == "havoc":
            var_name = spec.get("input_var") or spec.get("havoc_var")
            if var_name in variables:
                seq = list(variables[var_name].havoc_seq or [])
                idx = int(spec.get("havoc_index", 0))
                target_len = max(int(havoc_bound), idx + 1)
                if len(seq) < target_len:
                    seq.extend([0] * (target_len - len(seq)))
                seq[idx] = _signed_64(value)
                variables[var_name].havoc_seq = seq
        elif kind == "buffer_byte":
            var_name = spec.get("input_var")
            inp = variables.get(var_name)
            if inp is not None and inp.buffers:
                _update_buffer_byte(
                    inp.buffers[int(spec.get("buffer_index", 0))],
                    int(spec.get("byte_index", 0)),
                    value,
                )
        elif kind == "struct_buffer_byte":
            var_name = spec.get("input_var")
            inp = variables.get(var_name)
            if inp is not None and inp.struct:
                field = inp.struct[int(spec.get("field_index", 0))]
                if "buffer" in field:
                    _update_buffer_byte(
                        field["buffer"],
                        int(spec.get("byte_index", 0)),
                        value,
                    )
        elif kind == "struct_scalar_byte":
            var_name = spec.get("input_var")
            inp = variables.get(var_name)
            if inp is not None and inp.struct:
                field = inp.struct[int(spec.get("field_index", 0))]
                if "value" in field:
                    _update_struct_scalar_byte(
                        field,
                        int(spec.get("byte_index", 0)),
                        value,
                    )
        elif kind == "extra_byte":
            idx = int(spec.get("extra_index", 0))
            data = bytearray(candidate.extra_data or b"")
            if len(data) <= idx:
                data.extend([0] * (idx + 1 - len(data)))
            data[idx] = value & 0xff
            candidate.extra_data = bytes(data)

    return candidate


def symbolic_candidate_meta(raw_index: int,
                            candidate: Mapping[str, Any]) -> dict:
    updates = dict(candidate.get("updates") or {})
    return {
        "raw_index": raw_index,
        "source_block": candidate.get("source_block"),
        "target_block": candidate.get("target_block"),
        "branch_index": candidate.get("branch_index"),
        "kind": candidate.get("kind", "unknown"),
        "priority": candidate.get("priority", 0),
        "distance_to_uncovered": candidate.get("distance_to_uncovered", -1),
        "reachable_uncovered": candidate.get("reachable_uncovered", 0),
        "updates": updates,
    }


def _input_key(inp) -> tuple:
    return (
        bool(getattr(inp, "private", False)),
        getattr(inp, "value", None),
        _stable(getattr(inp, "buffers", None)),
        _stable(getattr(inp, "struct", None)),
        tuple(getattr(inp, "havoc_seq", None) or ()),
    )


def _stable(value):
    if isinstance(value, Mapping):
        return tuple(sorted((k, _stable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable(v) for v in value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return value


def _signed_64(value: int) -> int:
    value &= (1 << 64) - 1
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def _update_buffer_byte(buf: dict, byte_index: int, value: int) -> None:
    size = int(buf.get("size", 0) or 0)
    data = _contents_to_bytes(buf.get("contents", ""), size)
    if byte_index >= len(data):
        data.extend([0] * (byte_index + 1 - len(data)))
    data[byte_index] = value & 0xff
    buf["contents"] = _bytes_to_hex(data[:size or len(data)])


def _update_struct_scalar_byte(field: dict, byte_index: int, value: int) -> None:
    size = int(field.get("size", 0) or 0)
    data = _contents_to_bytes(field.get("value", ""), size)
    memory_order = bytearray(reversed(data))
    if byte_index >= len(memory_order):
        memory_order.extend([0] * (byte_index + 1 - len(memory_order)))
    memory_order[byte_index] = value & 0xff
    field["value"] = _bytes_to_hex(bytearray(reversed(memory_order[:size or len(memory_order)])))


def _contents_to_bytes(value, size: int) -> bytearray:
    if isinstance(value, list):
        data = bytearray(int(item) & 0xff for item in value[:size])
        data.extend([0] * max(0, size - len(data)))
        return data
    text = str(value or "").strip()
    match = _HELPER_RE.match(text)
    if match:
        fill = 0xff if match.group(1) == "ones" else 0
        return bytearray([fill] * max(size, int(match.group(2))))
    if text.startswith(("0x", "0X")):
        text = text[2:]
    if len(text) % 2:
        text = "0" + text
    data = bytearray.fromhex(text) if text else bytearray()
    data.extend([0] * max(0, size - len(data)))
    return data[:size]


def _bytes_to_hex(data: bytearray | bytes) -> str:
    return "0x" + bytes(data).hex()
