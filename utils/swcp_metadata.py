"""Input-concretization metadata for self-contained ``.swcp`` packages.

The bytecode package contains executable code and baked static scalars, but
buffer/struct inputs also need the SMACK ``{:array}``/``{:field}`` annotations
that are present only in the Boogie AST.  Keep that small metadata beside the
package and bind it to the exact package bytes so a stale sidecar cannot drive
execution with the wrong memory layout.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from interpreter.utils.static_eval import (
    compute_static_scalars,
    compute_static_values,
    expr_base_ptr,
    offset_delta,
)


SWCP_INPUT_METADATA_SCHEMA = "swoosh.swcp-input-metadata.v1"


def find_swcp_entry_point(program):
    """Find the body-bearing entry accepted by native SWCP lowering.

    SMACK commonly emits a body directly on ``procedure`` while hand-written
    Boogie often uses a separate ``implementation``.  Both are valid inputs to
    the native inliner.
    """
    from interpreter.parser.declaration import ProcedureDeclaration

    procedures = [
        decl
        for decl in program.declarations
        if isinstance(decl, ProcedureDeclaration)
    ]
    direct = [
        decl
        for decl in procedures
        if decl.body is not None and decl.has_attribute("entrypoint")
    ]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise ValueError(
            f"post-shadow program has multiple body-bearing entrypoints: "
            f"{[str(decl.name) for decl in direct]}"
        )

    declared_names = {
        str(decl.name)
        for decl in procedures
        if decl.has_attribute("entrypoint")
    }
    matching_bodies = [
        decl
        for decl in procedures
        if decl.body is not None and str(decl.name) in declared_names
    ]
    if len(matching_bodies) == 1:
        return matching_bodies[0]
    raise ValueError(
        "post-shadow program does not have exactly one body for its entrypoint; "
        f"entry declarations={sorted(declared_names)}, "
        f"matching bodies={[str(decl.name) for decl in matching_bodies]}"
    )


def _declared_names(program, impl_decl):
    from interpreter.parser.declaration import StorageDeclaration

    names = set()
    for decl in program.declarations:
        if isinstance(decl, StorageDeclaration):
            names.update(str(name) for name in decl.names)
    for decl in list(impl_decl.parameters) + list(impl_decl.returns):
        names.update(str(name) for name in decl.names)
    if impl_decl.body is not None:
        for decl in impl_decl.body.locals:
            if isinstance(decl, StorageDeclaration):
                names.update(str(name) for name in decl.names)
    return names


def _entry_input_names(impl_decl):
    """Return the exact formal-input names accepted by the entry body.

    SMACK emits the same ``{:name ...}`` memory annotations for caller-owned
    inputs and for local/return objects that merely carry results.  Only entry
    parameters are supplied by ``ProgramInputs``.  After CT shadowing this set
    naturally includes the explicit ``.shadow`` formals as separate lanes.
    """
    return {
        str(name)
        for declaration in impl_decl.parameters
        for name in declaration.names
    }


def build_native_input_meta(program, impl_decl):
    """Build the native VM's input-concretization metadata from an entry AST."""
    from interpreter.utils.inputs import gather_ptr_aliases, preprocess_external_inputs

    arr_info, field_info = preprocess_external_inputs(impl_decl)
    static_values = compute_static_values(program)
    declared_names = _declared_names(program, impl_decl)
    entry_input_names = _entry_input_names(impl_decl)

    arr_inputs = {}
    for key, infos in arr_info.items():
        if key not in entry_input_names:
            continue
        infos = [
            info
            for info in infos
            if info.base_ptr in declared_names and info.mem_map in declared_names
        ]
        if not infos:
            continue
        arr_inputs[key] = [
            {
                "mem_map": info.mem_map,
                "base_ptr": info.base_ptr,
                "offset_delta": offset_delta(
                    info.offset, info.base_ptr, static_values
                ),
                "elem_size": info.elem_size,
                "num_elements": info.num_elements,
            }
            for info in infos
        ]

    field_inputs = {}
    for key, infos in field_info.items():
        if key not in entry_input_names:
            continue
        items = []
        for info in infos:
            base_ptr = expr_base_ptr(info.base_ptr)
            if base_ptr not in declared_names or info.mem_map not in declared_names:
                continue
            items.append(
                {
                    "var_name": info.var_name,
                    "mem_map": info.mem_map,
                    "base_ptr": base_ptr,
                    "offset_delta": offset_delta(
                        info.base_ptr, base_ptr, static_values
                    ),
                    "size": info.size,
                    "kind": info.kind,
                }
            )
        if items:
            field_inputs[key] = items

    ptr_aliases = {
        alias: target
        for alias, target in gather_ptr_aliases(impl_decl).items()
        if alias in declared_names and target in declared_names
    }

    return {
        "static_scalars": compute_static_scalars(program),
        "arr_inputs": arr_inputs,
        "field_inputs": field_inputs,
        "ptr_aliases": ptr_aliases,
    }


def swcp_input_metadata_path(swcp_path) -> Path:
    """Return the canonical sidecar path for *swcp_path*."""
    path = Path(swcp_path)
    return path.with_name(f"{path.name}.meta.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as package:
        for chunk in iter(lambda: package.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_native_meta_shape(native_meta):
    if type(native_meta) is not dict:
        raise ValueError("SWCP input metadata payload must be an object")
    for key in ("static_scalars", "arr_inputs", "field_inputs", "ptr_aliases"):
        if type(native_meta.get(key)) is not dict:
            raise ValueError(
                f"SWCP input metadata field {key!r} must be an object"
            )


def write_swcp_input_metadata(swcp_path, native_meta) -> Path:
    """Atomically write metadata bound to the exact package SHA-256."""
    package = Path(swcp_path)
    _validate_native_meta_shape(native_meta)
    sidecar = swcp_input_metadata_path(package)
    payload = {
        "schema": SWCP_INPUT_METADATA_SCHEMA,
        "swcp_sha256": _sha256_file(package),
        "native_meta": native_meta,
    }
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        os.replace(temporary, sidecar)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sidecar


def load_swcp_input_metadata(swcp_path):
    """Load package-matched adjacent input metadata.

    A missing sidecar returns ``None`` so callers that provide no memory-backed
    inputs remain compatible with older packages.  A present but malformed or
    stale sidecar is always an error.  The hash detects package/sidecar drift;
    it is not an authentication tag over the metadata payload.
    """
    package = Path(swcp_path)
    sidecar = swcp_input_metadata_path(package)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SWCP input metadata {sidecar}: {exc}") from exc
    if type(payload) is not dict:
        raise ValueError(f"SWCP input metadata {sidecar} must be an object")
    if payload.get("schema") != SWCP_INPUT_METADATA_SCHEMA:
        raise ValueError(
            f"unsupported SWCP input metadata schema in {sidecar}: "
            f"{payload.get('schema')!r}"
        )
    expected = payload.get("swcp_sha256")
    actual = _sha256_file(package)
    if expected != actual:
        raise ValueError(
            f"SWCP input metadata {sidecar} does not match package {package}; "
            "rebuild the package and sidecar together"
        )
    native_meta = payload.get("native_meta")
    _validate_native_meta_shape(native_meta)
    return native_meta


def require_memory_input_metadata(program_inputs, native_meta, *, swcp_path=None):
    """Reject memory inputs that the supplied metadata would silently ignore."""
    arr_inputs = native_meta.get("arr_inputs", {}) if native_meta else {}
    field_inputs = native_meta.get("field_inputs", {}) if native_meta else {}
    problems = []

    def lane_infos(table, name):
        return [
            info
            for info in table.get(name, [])
            if (".shadow" in info.get("mem_map", ""))
            == name.endswith(".shadow")
        ]

    expected_memory_inputs = {
        name
        for name in set(arr_inputs) | set(field_inputs)
        if lane_infos(arr_inputs, name) or lane_infos(field_inputs, name)
    }
    for name in sorted(expected_memory_inputs - program_inputs.variables.keys()):
        problems.append(f"{name}: expected memory input is missing")

    for name, inp in program_inputs.variables.items():
        array_infos = lane_infos(arr_inputs, name)
        struct_infos = lane_infos(field_inputs, name)

        if name in expected_memory_inputs and inp.buffers is None and inp.struct is None:
            problems.append(f"{name}: expected memory input has no buffer/struct payload")

        if inp.buffers is not None:
            if struct_infos:
                problems.append(
                    f"{name}: buffer payload supplied for struct metadata"
                )
            if len(array_infos) != len(inp.buffers):
                problems.append(
                    f"{name}: {len(inp.buffers)} buffer(s), "
                    f"{len(array_infos)} array binding(s)"
                )
            else:
                for index, (buffer, info) in enumerate(
                    zip(inp.buffers, array_infos, strict=True)
                ):
                    expected_size = info["elem_size"] * info["num_elements"]
                    actual_size = buffer.get("size")
                    if actual_size != expected_size:
                        problems.append(
                            f"{name} buffer {index}: expected {expected_size} "
                            f"bytes, got {actual_size}"
                        )

        if inp.struct is not None:
            buffer_fields = sum("buffer" in field for field in inp.struct)
            if len(struct_infos) != len(inp.struct):
                problems.append(
                    f"{name}: {len(inp.struct)} struct field(s), "
                    f"{len(struct_infos)} field binding(s)"
                )
            if len(array_infos) != buffer_fields:
                problems.append(
                    f"{name}: {buffer_fields} struct buffer field(s), "
                    f"{len(array_infos)} array binding(s)"
                )
            else:
                buffers = [
                    field["buffer"] for field in inp.struct if "buffer" in field
                ]
                for index, (buffer, info) in enumerate(
                    zip(buffers, array_infos, strict=True)
                ):
                    expected_size = info["elem_size"] * info["num_elements"]
                    actual_size = buffer.get("size")
                    if actual_size != expected_size:
                        problems.append(
                            f"{name} struct buffer {index}: expected "
                            f"{expected_size} bytes, got {actual_size}"
                        )
            for index, (field, info) in enumerate(
                zip(inp.struct, struct_infos, strict=False)
            ):
                actual_kinds = [kind for kind in ("value", "buffer") if kind in field]
                expected_kind = info.get("kind")
                if expected_kind not in {"value", "buffer"}:
                    problems.append(
                        f"{name} struct field {index}: metadata has no exact payload kind"
                    )
                elif actual_kinds != [expected_kind]:
                    actual = "/".join(actual_kinds) if actual_kinds else "none"
                    problems.append(
                        f"{name} struct field {index}: expected {expected_kind} "
                        f"payload, got {actual}"
                    )
                actual_size = field.get("size")
                expected_size = info.get("size")
                if actual_size != expected_size:
                    problems.append(
                        f"{name} struct field {index}: expected field size "
                        f"{expected_size}, got {actual_size}"
                    )

    if problems:
        package = f" for {swcp_path}" if swcp_path is not None else ""
        raise ValueError(
            "memory-backed .input values have no exact SWCP metadata binding"
            f"{package}: {'; '.join(problems)}. Rebuild with "
            "`python3 -m tools.build_swcp ...` to create the adjacent "
            ".meta.json sidecar."
        )
