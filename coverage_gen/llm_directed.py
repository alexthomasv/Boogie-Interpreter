"""Typed, bounded admission for LLM-directed concrete inputs.

The model never owns an input file.  It returns one strict JSON object whose
candidate operations update values inside a host-owned :class:`ProgramInputs`:

.. code-block:: json

   {"candidates": [{"operations": [
     {"op": "set_scalar", "variable": "$i2", "value": 20},
     {"op": "set_buffer_bytes", "variable": "$p1",
      "buffer_index": 0, "offset": 0, "hex": "a94a"},
     {"op": "set_struct_scalar", "variable": "$p3",
      "field": "n_bitlen", "value": 2048},
     {"op": "set_struct_buffer_bytes", "variable": "$p3",
      "field": "p", "offset": 2, "hex": "0102"}
   ]}]}

Names, payload kinds, privacy, field order, field widths, and buffer lengths
come exclusively from the host input.  A candidate is exposed only after a
canonical writer/parser round trip and a bounded native evaluation with novel
reachable block or edge coverage.

This module intentionally contains no production LLM provider or filesystem
publication policy.  Callers inject a text callback and decide how admitted
canonical text is named and published.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Mapping, Protocol, Sequence

from interpreter.coverage_gen.corpus import path_features_from_sequence
from interpreter.coverage_gen.evaluator import EvaluationResult
from interpreter.coverage_gen.symbolic_state import program_inputs_cache_key
from interpreter.coverage_gen.writer import write_input_file
from interpreter.utils.input_parser import parse_input_file
from interpreter.utils.inputs import Input, ProgramInputs


RECEIPT_SCHEMA = "swoosh.llm-directed-input/v1"

HARD_MAX_RESPONSE_BYTES = 64 * 1024
HARD_MAX_OPERATIONS = 64
HARD_MAX_CHANGED_BYTES = 2 * 1024
HARD_MAX_CANDIDATES = 8
MAX_RECEIPT_EVIDENCE_BYTES = 1024

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1

Edge = tuple[str, str]
TextCaller = Callable[[str], str]


class BoundedEvaluator(Protocol):
    """The small part of :class:`Evaluator` required by this boundary."""

    timeout: int
    max_steps_per_input: int

    def run_result(
        self, program_inputs: ProgramInputs, input_name: str
    ) -> EvaluationResult: ...


class DirectedInputError(ValueError):
    """A host contract or complete model response is inadmissible."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        # Digest of the model response this error rejected, when one was
        # received; lets callers keep an audit trail for schema failures.
        self.response_sha256: str | None = None


class _PatchError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DirectedInputLimits:
    """Hard-capped generation and evaluation limits.

    Values may be lowered by tests or callers.  They cannot relax the module's
    hard response, operation, changed-byte, or candidate caps.
    """

    max_response_bytes: int = HARD_MAX_RESPONSE_BYTES
    max_operations: int = HARD_MAX_OPERATIONS
    max_changed_bytes: int = HARD_MAX_CHANGED_BYTES
    max_candidates: int = HARD_MAX_CANDIDATES
    evaluation_timeout_s: int = 15
    evaluation_max_steps: int = 25_000_000

    def __post_init__(self) -> None:
        _bounded_positive(
            "max_response_bytes", self.max_response_bytes, HARD_MAX_RESPONSE_BYTES
        )
        _bounded_positive("max_operations", self.max_operations, HARD_MAX_OPERATIONS)
        _bounded_positive(
            "max_changed_bytes", self.max_changed_bytes, HARD_MAX_CHANGED_BYTES
        )
        _bounded_positive("max_candidates", self.max_candidates, HARD_MAX_CANDIDATES)
        _positive_int("evaluation_timeout_s", self.evaluation_timeout_s)
        _positive_int("evaluation_max_steps", self.evaluation_max_steps)

    def to_mapping(self) -> dict[str, int]:
        return {
            "max_response_bytes": self.max_response_bytes,
            "max_operations": self.max_operations,
            "max_changed_bytes": self.max_changed_bytes,
            "max_candidates": self.max_candidates,
            "evaluation_timeout_s": self.evaluation_timeout_s,
            "evaluation_max_steps": self.evaluation_max_steps,
        }


@dataclass(frozen=True)
class CandidateReceipt:
    index: int
    patch_sha256: str
    operation_count: int
    changed_bytes: int
    decision: str
    reason: str
    operations: tuple[str, ...] = ()
    canonical_input_sha256: str | None = None
    evaluation_status: str | None = None
    violation_pc: int | None = None
    violation_block: str | None = None
    evaluation_message: str | None = None
    invalid_reason: str | None = None
    invalid_detail: str | None = None
    reachable_blocks: tuple[str, ...] = ()
    reachable_edges: tuple[Edge, ...] = ()
    new_blocks: tuple[str, ...] = ()
    new_edges: tuple[Edge, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "patch_sha256": self.patch_sha256,
            "operation_count": self.operation_count,
            "operations": [json.loads(operation) for operation in self.operations],
            "changed_bytes": self.changed_bytes,
            "decision": self.decision,
            "reason": self.reason,
            "canonical_input_sha256": self.canonical_input_sha256,
            "evaluation_status": self.evaluation_status,
            "violation_pc": self.violation_pc,
            "violation_block": self.violation_block,
            "evaluation_message": self.evaluation_message,
            "invalid_reason": self.invalid_reason,
            "invalid_detail": self.invalid_detail,
            "reachable_blocks": list(self.reachable_blocks),
            "reachable_edges": [list(edge) for edge in self.reachable_edges],
            "new_blocks": list(self.new_blocks),
            "new_edges": [list(edge) for edge in self.new_edges],
        }


@dataclass(frozen=True)
class DirectedInputReceipt:
    prompt_sha256: str
    response_sha256: str
    base_input_sha256: str
    limits: DirectedInputLimits
    evaluator_timeout_s: int
    evaluator_max_steps: int
    candidates: tuple[CandidateReceipt, ...]

    @property
    def admitted_count(self) -> int:
        return sum(item.decision == "admitted" for item in self.candidates)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "base_input_sha256": self.base_input_sha256,
            "limits": self.limits.to_mapping(),
            "evaluator": {
                "timeout_s": self.evaluator_timeout_s,
                "max_steps": self.evaluator_max_steps,
            },
            "candidate_count": len(self.candidates),
            "admitted_count": self.admitted_count,
            "candidates": [item.to_mapping() for item in self.candidates],
        }


@dataclass(frozen=True)
class AdmittedDirectedInput:
    """One host-owned candidate ready for a caller-chosen publication name."""

    candidate_index: int
    program_inputs: ProgramInputs
    canonical_text: str
    covered_blocks: frozenset[str]
    covered_edges: frozenset[Edge]
    new_blocks: frozenset[str]
    new_edges: frozenset[Edge]


@dataclass(frozen=True)
class DirectedInputBatch:
    admitted: tuple[AdmittedDirectedInput, ...]
    receipt: DirectedInputReceipt


@dataclass(frozen=True)
class AppliedDirectedInputPatch:
    """One bounded typed patch over a host-owned canonical input.

    Unlike :class:`AdmittedDirectedInput`, this result carries no coverage or
    proof claim.  It is the reusable normalization boundary for diagnostic
    callers (for example, concrete falsification): the caller supplies typed
    operations, while the host preserves the input shape and owns the final
    writer/parser round trip.
    """

    program_inputs: ProgramInputs
    canonical_text: str
    changed_bytes: int


def apply_directed_input_operations(
    *,
    base_inputs: ProgramInputs,
    operations: Sequence[Mapping[str, object]],
    params_line: str = "",
    field_sizes: Mapping[str, Sequence[int]] | None = None,
    paired_variables: Collection[tuple[str, str]] = (),
    limits: DirectedInputLimits | None = None,
) -> AppliedDirectedInputPatch:
    """Apply a bounded operation list and return canonical input text.

    This exposes the typed patch/round-trip portion of directed-input
    generation without importing its LLM, novelty, or publication policy.
    Successful return means only that the concrete input is well-formed and
    shape-preserving; semantic evidence belongs to the caller's execution.
    """

    if limits is None:
        limits = DirectedInputLimits()
    elif type(limits) is not DirectedInputLimits:
        raise DirectedInputError(
            "invalid_limits", "limits must be DirectedInputLimits"
        )
    if isinstance(operations, (str, bytes)):
        raise DirectedInputError(
            "invalid_operations", "operations must be a sequence of objects"
        )
    try:
        normalized_operations = tuple(dict(operation) for operation in operations)
    except (TypeError, ValueError) as exc:
        raise DirectedInputError(
            "invalid_operations", "operations must be a sequence of objects"
        ) from exc
    if not normalized_operations:
        raise DirectedInputError(
            "invalid_operations", "operations must be non-empty"
        )
    if len(normalized_operations) > limits.max_operations:
        raise DirectedInputError(
            "too_many_operations",
            f"operations exceed {limits.max_operations} entries",
        )
    for operation in normalized_operations:
        _validate_operation_schema(operation)

    _validate_params_line(params_line)
    normalized_field_sizes = _validate_field_sizes(field_sizes or {})
    shape = _program_shape(base_inputs)
    canonical_base, _base_text = _canonicalize_host_input(
        base_inputs,
        params_line=params_line,
        field_sizes=normalized_field_sizes,
        expected_shape=shape,
    )
    shape = _program_shape(canonical_base)
    mirror_map = _validate_paired_variables(
        paired_variables, canonical_base
    )
    try:
        patched, changed_bytes = _apply_candidate(
            canonical_base,
            {"operations": list(normalized_operations)},
            mirror_map=mirror_map,
        )
    except _PatchError as exc:
        raise DirectedInputError(
            exc.code, f"typed input patch was rejected: {exc.code}"
        ) from exc
    if changed_bytes > limits.max_changed_bytes:
        raise DirectedInputError(
            "changed_bytes_exceeded",
            f"patch changes {changed_bytes} bytes; limit is "
            f"{limits.max_changed_bytes}",
        )
    canonical_text = _canonical_round_trip(
        patched,
        params_line=params_line,
        field_sizes=normalized_field_sizes,
        expected_shape=shape,
        host_input=False,
    )
    return AppliedDirectedInputPatch(
        program_inputs=patched,
        canonical_text=canonical_text,
        changed_bytes=changed_bytes,
    )


def generate_directed_inputs(
    *,
    prompt: str,
    call_llm: TextCaller,
    base_inputs: ProgramInputs,
    evaluator: BoundedEvaluator,
    reachable_blocks: Collection[str],
    known_blocks: Collection[str] = (),
    known_edges: Collection[Edge] = (),
    params_line: str = "",
    field_sizes: Mapping[str, Sequence[int]] | None = None,
    paired_variables: Collection[tuple[str, str]] = (),
    limits: DirectedInputLimits | None = None,
) -> DirectedInputBatch:
    """Request, transform, and admit a bounded batch of directed inputs.

    ``reachable_blocks`` is the caller's package-bound static universe.
    Coverage novelty is computed against ``known_blocks``/``known_edges`` and
    then updated in candidate order, so later candidates must add something not
    already supplied by an earlier admitted candidate.
    """

    if limits is None:
        limits = DirectedInputLimits()
    elif type(limits) is not DirectedInputLimits:
        raise DirectedInputError(
            "invalid_limits", "limits must be DirectedInputLimits"
        )
    if type(prompt) is not str:
        raise DirectedInputError("invalid_prompt", "prompt must be a string")
    if not callable(call_llm):
        raise DirectedInputError("invalid_caller", "call_llm must be callable")
    _validate_params_line(params_line)
    normalized_field_sizes = _validate_field_sizes(field_sizes or {})
    shape = _program_shape(base_inputs)
    mirror_map = _validate_paired_variables(paired_variables, base_inputs)
    base_inputs, base_text = _canonicalize_host_input(
        base_inputs,
        params_line=params_line,
        field_sizes=normalized_field_sizes,
        expected_shape=shape,
    )
    shape = _program_shape(base_inputs)
    evaluator_timeout, evaluator_steps = _validate_bounded_evaluator(
        evaluator, limits
    )
    reachable = _string_set("reachable_blocks", reachable_blocks)
    seen_blocks = _string_set("known_blocks", known_blocks)
    seen_edges = _edge_set("known_edges", known_edges)

    try:
        response = call_llm(prompt)
    except Exception as exc:
        raise DirectedInputError(
            "llm_call_failed",
            f"LLM text callback failed: {type(exc).__name__}: {exc}",
        ) from exc
    if type(response) is not str:
        raise DirectedInputError(
            "non_text_response", "LLM text callback must return a string"
        )
    response_bytes = response.encode("utf-8")
    response_sha256 = hashlib.sha256(response_bytes).hexdigest()
    try:
        if len(response_bytes) > limits.max_response_bytes:
            raise DirectedInputError(
                "response_too_large",
                f"LLM response exceeds {limits.max_response_bytes} bytes",
            )

        candidates = _parse_response(response, limits)
        prepared: list[
            tuple[dict[str, object], ProgramInputs | None, int, str | None]
        ] = []
        total_changed = 0
        for candidate in candidates:
            try:
                patched, changed_bytes = _apply_candidate(
                    base_inputs, candidate, mirror_map=mirror_map
                )
                if _program_shape(patched) != shape:
                    raise _PatchError("shape_changed")
            except _PatchError as exc:
                prepared.append((candidate, None, 0, exc.code))
                continue
            total_changed += changed_bytes
            prepared.append((candidate, patched, changed_bytes, None))
        if total_changed > limits.max_changed_bytes:
            raise DirectedInputError(
                "changed_bytes_exceeded",
                f"candidate patches exceed {limits.max_changed_bytes} changed bytes",
            )
    except DirectedInputError as exc:
        exc.response_sha256 = response_sha256
        raise

    admitted: list[AdmittedDirectedInput] = []
    receipts: list[CandidateReceipt] = []
    semantic_keys = {program_inputs_cache_key(base_inputs, 0)}
    for index, (candidate, patched, changed_bytes, patch_error) in enumerate(prepared):
        patch_sha = _sha256_text(_canonical_json(candidate))
        operations = _canonical_operations(candidate)
        receipt_patch = dict(
            index=index,
            patch_sha256=patch_sha,
            operation_count=len(operations),
            operations=operations,
        )
        if patch_error is not None:
            receipts.append(CandidateReceipt(
                **receipt_patch,
                changed_bytes=0,
                decision="rejected",
                reason=patch_error,
            ))
            continue
        assert patched is not None

        semantic_key = program_inputs_cache_key(patched, 0)
        if semantic_key in semantic_keys:
            receipts.append(CandidateReceipt(
                **receipt_patch,
                changed_bytes=changed_bytes,
                decision="rejected",
                reason="duplicate_candidate",
            ))
            continue
        semantic_keys.add(semantic_key)

        try:
            canonical_text = _canonical_round_trip(
                patched,
                params_line=params_line,
                field_sizes=normalized_field_sizes,
                expected_shape=shape,
                host_input=False,
            )
        except _PatchError as exc:
            receipts.append(CandidateReceipt(
                **receipt_patch,
                changed_bytes=changed_bytes,
                decision="rejected",
                reason=exc.code,
            ))
            continue

        canonical_sha = _sha256_text(canonical_text)
        try:
            evaluation = evaluator.run_result(
                copy.deepcopy(patched), f"llm_directed_{index:03d}"
            )
        except Exception:
            receipts.append(CandidateReceipt(
                **receipt_patch,
                changed_bytes=changed_bytes,
                decision="rejected",
                reason="evaluation_exception",
                canonical_input_sha256=canonical_sha,
            ))
            continue
        if not isinstance(evaluation, EvaluationResult):
            receipts.append(CandidateReceipt(
                **receipt_patch,
                changed_bytes=changed_bytes,
                decision="rejected",
                reason="invalid_evaluation_result",
                canonical_input_sha256=canonical_sha,
            ))
            continue

        status = evaluation.status
        try:
            if type(status) is not str or not status:
                raise DirectedInputError(
                    "invalid_evaluation_result",
                    "evaluation status must be a non-empty string",
                )
            covered, edges = _reachable_coverage(evaluation, reachable)
        except DirectedInputError:
            receipts.append(CandidateReceipt(
                **receipt_patch,
                changed_bytes=changed_bytes,
                decision="rejected",
                reason="invalid_evaluation_result",
                canonical_input_sha256=canonical_sha,
            ))
            continue
        new_blocks = covered - seen_blocks
        new_edges = edges - seen_edges
        receipt_base = dict(
            **receipt_patch,
            changed_bytes=changed_bytes,
            canonical_input_sha256=canonical_sha,
            evaluation_status=status,
            violation_pc=(
                evaluation.violation_pc
                if type(evaluation.violation_pc) is int
                and evaluation.violation_pc >= 0
                else None
            ),
            violation_block=_receipt_evidence_text(evaluation.violation_block),
            evaluation_message=_receipt_evidence_text(evaluation.message),
            invalid_reason=_receipt_evidence_text(evaluation.invalid_reason),
            invalid_detail=_receipt_evidence_text(evaluation.invalid_detail),
            reachable_blocks=tuple(sorted(covered)),
            reachable_edges=tuple(sorted(edges)),
            new_blocks=tuple(sorted(new_blocks)),
            new_edges=tuple(sorted(new_edges)),
        )
        if status != "ok":
            receipts.append(CandidateReceipt(
                **receipt_base,
                decision="rejected",
                reason="status_not_ok",
            ))
            continue
        if not new_blocks and not new_edges:
            receipts.append(CandidateReceipt(
                **receipt_base,
                decision="rejected",
                reason="no_novel_reachable_coverage",
            ))
            continue

        seen_blocks |= covered
        seen_edges |= edges
        admitted.append(AdmittedDirectedInput(
            candidate_index=index,
            program_inputs=patched,
            canonical_text=canonical_text,
            covered_blocks=frozenset(covered),
            covered_edges=frozenset(edges),
            new_blocks=frozenset(new_blocks),
            new_edges=frozenset(new_edges),
        ))
        receipts.append(CandidateReceipt(
            **receipt_base,
            decision="admitted",
            reason="novel_reachable_coverage",
        ))

    receipt = DirectedInputReceipt(
        prompt_sha256=_sha256_text(prompt),
        response_sha256=response_sha256,
        base_input_sha256=_sha256_text(base_text),
        limits=limits,
        evaluator_timeout_s=evaluator_timeout,
        evaluator_max_steps=evaluator_steps,
        candidates=tuple(receipts),
    )
    return DirectedInputBatch(tuple(admitted), receipt)


def _receipt_evidence_text(value: object) -> str | None:
    if type(value) is not str or not value:
        return None
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_RECEIPT_EVIDENCE_BYTES:
        return value
    marker = " [truncated]"
    budget = MAX_RECEIPT_EVIDENCE_BYTES - len(marker.encode("utf-8"))
    return encoded[:budget].decode("utf-8", errors="ignore") + marker


def _parse_response(
    response: str, limits: DirectedInputLimits
) -> tuple[dict[str, object], ...]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise DirectedInputError(
                    "duplicate_json_key", f"duplicate JSON key: {key!r}"
                )
            value[key] = item
        return value

    try:
        payload = json.loads(
            response,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DirectedInputError(
                    "invalid_json_constant", f"invalid JSON constant: {token}"
                )
            ),
        )
    except DirectedInputError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise DirectedInputError("invalid_json", "LLM response is not valid JSON") from exc
    if type(payload) is not dict or set(payload) != {"candidates"}:
        raise DirectedInputError(
            "invalid_response_shape",
            "response must contain exactly one 'candidates' field",
        )
    raw_candidates = payload["candidates"]
    if type(raw_candidates) is not list or not raw_candidates:
        raise DirectedInputError(
            "invalid_candidates", "candidates must be a non-empty list"
        )
    if len(raw_candidates) > limits.max_candidates:
        raise DirectedInputError(
            "too_many_candidates",
            f"response exceeds {limits.max_candidates} candidates",
        )

    operation_count = 0
    candidates: list[dict[str, object]] = []
    for raw_candidate in raw_candidates:
        if type(raw_candidate) is not dict or set(raw_candidate) != {"operations"}:
            raise DirectedInputError(
                "invalid_candidate_shape",
                "each candidate must contain exactly one 'operations' field",
            )
        operations = raw_candidate["operations"]
        if type(operations) is not list or not operations:
            raise DirectedInputError(
                "invalid_operations", "candidate operations must be a non-empty list"
            )
        for operation in operations:
            _validate_operation_schema(operation)
        operation_count += len(operations)
        candidates.append(raw_candidate)
    if operation_count > limits.max_operations:
        raise DirectedInputError(
            "too_many_operations",
            f"response exceeds {limits.max_operations} operations",
        )
    return tuple(candidates)


_OPERATION_KEYS = {
    "set_scalar": {"op", "variable", "value"},
    "set_havoc_value": {"op", "variable", "index", "value"},
    "set_buffer_bytes": {
        "op", "variable", "buffer_index", "offset", "hex"
    },
    "set_struct_scalar": {"op", "variable", "field", "value"},
    "set_struct_buffer_bytes": {
        "op", "variable", "field", "offset", "hex"
    },
}


def _validate_operation_schema(operation: object) -> None:
    if type(operation) is not dict:
        raise DirectedInputError("invalid_operation", "operation must be an object")
    op = operation.get("op")
    if type(op) is not str or op not in _OPERATION_KEYS:
        raise DirectedInputError("unknown_operation", "operation kind is unsupported")
    if set(operation) != _OPERATION_KEYS[op]:
        raise DirectedInputError(
            "invalid_operation_fields", f"fields for {op!r} are not exact"
        )
    variable = operation.get("variable")
    if type(variable) is not str or not variable:
        raise DirectedInputError(
            "invalid_variable", "operation variable must be a non-empty string"
        )
    if "field" in operation:
        field = operation.get("field")
        if type(field) is not str or not field:
            raise DirectedInputError(
                "invalid_field", "operation field must be a non-empty string"
            )
    for key in ("buffer_index", "offset", "index"):
        if key in operation and not _is_nonnegative_int(operation[key]):
            raise DirectedInputError(
                f"invalid_{key}", f"operation {key} must be a non-negative integer"
            )
    if "value" in operation and not _is_i64(operation["value"]):
        raise DirectedInputError(
            "invalid_value", "operation value must be a signed 64-bit integer"
        )
    if "hex" in operation:
        raw_hex = operation["hex"]
        if (
            type(raw_hex) is not str
            or not raw_hex
            or len(raw_hex) % 2
            or _HEX_RE.fullmatch(raw_hex) is None
        ):
            raise DirectedInputError(
                "invalid_hex", "operation hex must contain whole hexadecimal bytes"
            )


def _apply_candidate(
    base_inputs: ProgramInputs,
    candidate: Mapping[str, object],
    *,
    mirror_map: Mapping[str, str],
) -> tuple[ProgramInputs, int]:
    patched = copy.deepcopy(base_inputs)
    changed_bytes = 0
    operations = candidate["operations"]
    assert type(operations) is list
    for operation in operations:
        assert type(operation) is dict
        variable = operation["variable"]
        assert type(variable) is str
        changed_bytes += _apply_operation(patched, operation)
        shadow_name = mirror_map.get(variable)
        if shadow_name is not None:
            shadow_operation = dict(operation)
            shadow_operation["variable"] = shadow_name
            changed_bytes += _apply_operation(patched, shadow_operation)
    return patched, changed_bytes


def _validate_paired_variables(
    pairs: Collection[tuple[str, str]],
    base_inputs: ProgramInputs,
) -> dict[str, str]:
    if isinstance(pairs, (str, bytes)):
        raise DirectedInputError(
            "invalid_paired_variables", "paired_variables must contain pairs"
        )
    mirror_map: dict[str, str] = {}
    try:
        items = tuple(pairs)
    except TypeError as exc:
        raise DirectedInputError(
            "invalid_paired_variables", "paired_variables must be iterable"
        ) from exc
    for pair in items:
        if type(pair) is not tuple or len(pair) != 2:
            raise DirectedInputError(
                "invalid_paired_variables",
                "each paired_variables item must be a two-name tuple",
            )
        variable, shadow_name = pair
        if (
            type(variable) is not str
            or type(shadow_name) is not str
            or not variable
            or shadow_name != f"{variable}.shadow"
            or variable.endswith(".shadow")
        ):
            raise DirectedInputError(
                "invalid_paired_variables", "paired variable names are invalid"
            )
        if (
            variable not in base_inputs.variables
            or shadow_name not in base_inputs.variables
        ):
            raise DirectedInputError(
                "invalid_paired_variables",
                "paired variables must both exist in the host input",
            )
        if variable in mirror_map:
            raise DirectedInputError(
                "invalid_paired_variables", "paired variable is duplicated"
            )
        mirror_map[variable] = shadow_name
    return mirror_map


def _apply_operation(
    program_inputs: ProgramInputs, operation: Mapping[str, object]
) -> int:
    variable = operation["variable"]
    assert type(variable) is str
    inp = program_inputs.variables.get(variable)
    if inp is None:
        raise _PatchError("unknown_variable")
    op = operation["op"]
    if op == "set_scalar":
        if _input_kind(inp) != "scalar":
            raise _PatchError("kind_mismatch")
        inp.value = operation["value"]
        return 8
    if op == "set_havoc_value":
        if _input_kind(inp) != "havoc":
            raise _PatchError("kind_mismatch")
        assert inp.havoc_seq is not None
        index = operation["index"]
        assert type(index) is int
        if index >= len(inp.havoc_seq):
            raise _PatchError("havoc_index_out_of_range")
        inp.havoc_seq[index] = operation["value"]
        return 8
    if op == "set_buffer_bytes":
        if _input_kind(inp) != "buffer":
            raise _PatchError("kind_mismatch")
        assert inp.buffers is not None
        buffer_index = operation["buffer_index"]
        assert type(buffer_index) is int
        if buffer_index >= len(inp.buffers):
            raise _PatchError("buffer_index_out_of_range")
        return _patch_buffer(
            inp.buffers[buffer_index], operation["offset"], operation["hex"]
        )
    if _input_kind(inp) != "struct":
        raise _PatchError("kind_mismatch")
    assert inp.struct is not None
    field_name = operation["field"]
    assert type(field_name) is str
    field = next(
        (item for item in inp.struct if item["name"] == field_name), None
    )
    if field is None:
        raise _PatchError("unknown_field")
    if op == "set_struct_scalar":
        if "value" not in field:
            raise _PatchError("field_kind_mismatch")
        width = field["size"]
        value = operation["value"]
        assert type(width) is int and type(value) is int
        if value < 0 or value >= (1 << (width * 8)):
            raise _PatchError("struct_value_out_of_range")
        field["value"] = f"0x{value:0{width * 2}x}"
        return width
    if "buffer" not in field:
        raise _PatchError("field_kind_mismatch")
    return _patch_buffer(
        field["buffer"], operation["offset"], operation["hex"]
    )


def _patch_buffer(buffer: dict, offset: object, raw_hex: object) -> int:
    assert type(offset) is int and type(raw_hex) is str
    data = bytearray(_decode_buffer(buffer))
    replacement = bytes.fromhex(raw_hex)
    end = offset + len(replacement)
    if end > len(data):
        raise _PatchError("patch_out_of_bounds")
    data[offset:end] = replacement
    buffer["contents"] = "0x" + data.hex()
    return len(replacement)


def _canonical_round_trip(
    program_inputs: ProgramInputs,
    *,
    params_line: str,
    field_sizes: Mapping[str, Sequence[int]],
    expected_shape: tuple,
    host_input: bool,
) -> str:
    text = write_input_file(program_inputs, params_line)
    try:
        with tempfile.TemporaryDirectory(prefix="swoosh-llm-input-") as raw_dir:
            path = Path(raw_dir) / "candidate.input"
            path.write_text(text, encoding="utf-8")
            reparsed = parse_input_file(path, field_sizes=field_sizes)
        if _program_shape(reparsed) != expected_shape:
            raise _PatchError("roundtrip_shape_mismatch")
        if program_inputs_cache_key(reparsed, 0) != program_inputs_cache_key(
            program_inputs, 0
        ):
            raise _PatchError("roundtrip_value_mismatch")
    except _PatchError:
        if host_input:
            raise DirectedInputError(
                "noncanonical_base_input",
                "host ProgramInputs does not round-trip canonically",
            )
        raise
    except Exception as exc:
        if host_input:
            raise DirectedInputError(
                "noncanonical_base_input",
                "host ProgramInputs cannot be written and reparsed",
            ) from exc
        raise _PatchError("roundtrip_parse_error") from exc
    return text


def _canonicalize_host_input(
    program_inputs: ProgramInputs,
    *,
    params_line: str,
    field_sizes: Mapping[str, Sequence[int]],
    expected_shape: tuple,
) -> tuple[ProgramInputs, str]:
    raw_text = write_input_file(program_inputs, params_line)
    try:
        reparsed = _parse_input_text(raw_text, field_sizes)
        if _program_shape(reparsed) != expected_shape:
            raise _PatchError("roundtrip_shape_mismatch")
        canonical_text = write_input_file(reparsed, params_line)
        canonical = _parse_input_text(canonical_text, field_sizes)
        if _program_shape(canonical) != expected_shape:
            raise _PatchError("roundtrip_shape_mismatch")
        if program_inputs_cache_key(canonical, 0) != program_inputs_cache_key(
            reparsed, 0
        ):
            raise _PatchError("roundtrip_value_mismatch")
    except _PatchError as exc:
        raise DirectedInputError(
            "noncanonical_base_input",
            f"host ProgramInputs cannot be canonicalized: {exc.code}",
        ) from exc
    except Exception as exc:
        raise DirectedInputError(
            "noncanonical_base_input",
            "host ProgramInputs cannot be written and reparsed",
        ) from exc
    return canonical, canonical_text


def _parse_input_text(
    text: str,
    field_sizes: Mapping[str, Sequence[int]],
) -> ProgramInputs:
    with tempfile.TemporaryDirectory(prefix="swoosh-llm-input-") as raw_dir:
        path = Path(raw_dir) / "candidate.input"
        path.write_text(text, encoding="utf-8")
        return parse_input_file(path, field_sizes=field_sizes)


def _program_shape(program_inputs: ProgramInputs) -> tuple:
    if type(program_inputs) is not ProgramInputs:
        raise DirectedInputError(
            "invalid_base_input", "base_inputs must be ProgramInputs"
        )
    items = []
    for name, inp in sorted(program_inputs.variables.items()):
        if type(inp) is not Input or inp.name != name or type(inp.private) is not bool:
            raise DirectedInputError(
                "invalid_base_input", "host input variable metadata is invalid"
            )
        kind = _input_kind(inp)
        if kind == "invalid":
            raise DirectedInputError(
                "invalid_base_shape", f"host input {name!r} has ambiguous payload kind"
            )
        if kind == "scalar":
            if not _is_i64(inp.value):
                raise DirectedInputError(
                    "invalid_base_scalar", f"host scalar {name!r} is not signed 64-bit"
                )
            detail = ()
        elif kind == "buffer":
            assert inp.buffers is not None
            if not inp.buffers:
                raise DirectedInputError(
                    "invalid_base_buffer", f"host buffer {name!r} is empty"
                )
            detail = tuple(_buffer_shape(buf) for buf in inp.buffers)
        elif kind == "struct":
            assert inp.struct is not None
            if not inp.struct:
                raise DirectedInputError(
                    "invalid_base_struct", f"host struct {name!r} is empty"
                )
            field_names = [field.get("name") for field in inp.struct]
            if (
                any(type(field_name) is not str or not field_name for field_name in field_names)
                or len(field_names) != len(set(field_names))
            ):
                raise DirectedInputError(
                    "invalid_base_struct", f"host struct {name!r} has invalid fields"
                )
            detail = tuple(_field_shape(name, field) for field in inp.struct)
        else:
            assert inp.havoc_seq is not None
            if any(not _is_i64(value) for value in inp.havoc_seq):
                raise DirectedInputError(
                    "invalid_base_havoc", f"host havoc sequence {name!r} is invalid"
                )
            detail = (len(inp.havoc_seq),)
        items.append((name, inp.private, kind, detail))
    extra = bytes(program_inputs.extra_data or b"")
    return (extra, tuple(items))


def _input_kind(inp: Input) -> str:
    present = [
        ("scalar", inp.value is not None),
        ("buffer", inp.buffers is not None),
        ("struct", inp.struct is not None),
        ("havoc", inp.havoc_seq is not None),
    ]
    kinds = [name for name, available in present if available]
    return kinds[0] if len(kinds) == 1 else "invalid"


def _buffer_shape(buffer: object) -> tuple[str, int]:
    if type(buffer) is not dict or set(buffer) != {"contents", "size"}:
        raise DirectedInputError(
            "invalid_base_buffer", "host buffer must contain contents and size"
        )
    data = _decode_buffer(buffer)
    return ("buffer", len(data))


def _decode_buffer(buffer: Mapping[str, object]) -> bytes:
    size = buffer.get("size")
    contents = buffer.get("contents")
    if not _is_nonnegative_int(size) or size == 0:
        raise DirectedInputError(
            "invalid_base_buffer", "host buffer size must be positive"
        )
    if (
        type(contents) is not str
        or not contents.startswith("0x")
        or len(contents) != 2 + (size * 2)
        or _HEX_RE.fullmatch(contents[2:]) is None
    ):
        raise DirectedInputError(
            "invalid_base_buffer", "host buffer contents do not match its size"
        )
    return bytes.fromhex(contents[2:])


def _field_shape(variable: str, field: object) -> tuple:
    if type(field) is not dict:
        raise DirectedInputError(
            "invalid_base_struct", f"host struct {variable!r} has a non-object field"
        )
    name = field.get("name")
    size = field.get("size")
    if type(name) is not str or not name or not _is_nonnegative_int(size) or size == 0:
        raise DirectedInputError(
            "invalid_base_struct", f"host struct {variable!r} field metadata is invalid"
        )
    if set(field) == {"name", "size", "value"}:
        value = field["value"]
        if (
            type(value) is not str
            or not value.startswith("0x")
            or len(value) != 2 + (size * 2)
            or _HEX_RE.fullmatch(value[2:]) is None
        ):
            raise DirectedInputError(
                "invalid_base_struct",
                f"host struct scalar {variable!r}.{name} has the wrong width",
            )
        return (name, "scalar", size)
    if set(field) == {"name", "size", "buffer"}:
        buffer_shape = _buffer_shape(field["buffer"])
        return (name, "buffer", size, buffer_shape[1])
    raise DirectedInputError(
        "invalid_base_struct", f"host struct {variable!r}.{name} has ambiguous shape"
    )


def _reachable_coverage(
    evaluation: EvaluationResult, reachable: set[str]
) -> tuple[set[str], set[Edge]]:
    try:
        covered = set(evaluation.covered or ())
        sequence = tuple(evaluation.block_sequence or ())
        edges = _edge_set("evaluation_edges", evaluation.covered_edges or ())
    except TypeError as exc:
        raise DirectedInputError(
            "invalid_evaluation_result", "evaluation coverage is not iterable"
        ) from exc
    if any(type(block) is not str or not block for block in covered | set(sequence)):
        raise DirectedInputError(
            "invalid_evaluation_result", "evaluation blocks must be non-empty strings"
        )
    covered &= reachable
    edges |= set(path_features_from_sequence(sequence)["edges"])
    edges = {
        (source, target)
        for source, target in edges
        if source in reachable and target in reachable
    }
    return covered, edges


def _validate_bounded_evaluator(
    evaluator: BoundedEvaluator, limits: DirectedInputLimits
) -> tuple[int, int]:
    timeout = getattr(evaluator, "timeout", None)
    max_steps = getattr(evaluator, "max_steps_per_input", None)
    if (
        type(timeout) is not int
        or timeout <= 0
        or timeout > limits.evaluation_timeout_s
    ):
        raise DirectedInputError(
            "unbounded_evaluator",
            "evaluator timeout must be positive and within the configured bound",
        )
    if (
        type(max_steps) is not int
        or max_steps <= 0
        or max_steps > limits.evaluation_max_steps
    ):
        raise DirectedInputError(
            "unbounded_evaluator",
            "evaluator max_steps_per_input must be positive and bounded",
        )
    if not callable(getattr(evaluator, "run_result", None)):
        raise DirectedInputError(
            "invalid_evaluator", "evaluator must provide run_result"
        )
    return timeout, max_steps


def _validate_params_line(params_line: str) -> None:
    if type(params_line) is not str or "\n" in params_line or "\r" in params_line:
        raise DirectedInputError(
            "invalid_params_line", "params_line must be one string line"
        )
    if params_line and not params_line.strip().startswith("// @params"):
        raise DirectedInputError(
            "invalid_params_line", "params_line must be a @params comment"
        )


def _validate_field_sizes(
    field_sizes: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    if not isinstance(field_sizes, Mapping):
        raise DirectedInputError("invalid_field_sizes", "field_sizes must be a mapping")
    result: dict[str, tuple[int, ...]] = {}
    for name, sizes in field_sizes.items():
        if (
            type(name) is not str
            or not name
            or not isinstance(sizes, (list, tuple))
        ):
            raise DirectedInputError(
                "invalid_field_sizes", "field size entries are malformed"
            )
        normalized = tuple(sizes)
        if any(type(size) is not int or size <= 0 for size in normalized):
            raise DirectedInputError(
                "invalid_field_sizes", "field sizes must be positive integers"
            )
        result[name] = normalized
    return result


def _string_set(name: str, values: Collection[str]) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise DirectedInputError(
            f"invalid_{name}", f"{name} must be a collection of strings"
        )
    try:
        result = set(values)
    except TypeError as exc:
        raise DirectedInputError(f"invalid_{name}", f"{name} must be iterable") from exc
    if any(type(value) is not str or not value for value in result):
        raise DirectedInputError(
            f"invalid_{name}", f"{name} must contain non-empty strings"
        )
    return result


def _edge_set(name: str, values: Collection[Edge]) -> set[Edge]:
    result: set[Edge] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise DirectedInputError(f"invalid_{name}", f"{name} must be iterable") from exc
    for value in iterator:
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or any(type(item) is not str or not item for item in value)
        ):
            raise DirectedInputError(
                f"invalid_{name}", f"{name} must contain string pairs"
            )
        result.add((value[0], value[1]))
    return result


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _canonical_operations(candidate: Mapping[str, object]) -> tuple[str, ...]:
    """Freeze validated operations as exact, deterministic JSON values.

    The response parser has already bounded the aggregate operation count and
    response bytes.  Keeping one canonical string per operation makes receipt
    memory immutable without adding another, potentially divergent schema.
    ``CandidateReceipt.to_mapping`` materializes fresh objects for consumers.
    """
    operations = candidate["operations"]
    assert type(operations) is list
    return tuple(_canonical_json(operation) for operation in operations)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_i64(value: object) -> bool:
    return type(value) is int and _I64_MIN <= value <= _I64_MAX


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _bounded_positive(name: str, value: object, hard_max: int) -> None:
    _positive_int(name, value)
    assert type(value) is int
    if value > hard_max:
        raise ValueError(f"{name} cannot exceed {hard_max}")
