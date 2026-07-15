"""
Rust interpreter runner.

Usage:
    python -m interpreter.runner <test_pkg_path> [--engine=native] [--force] [--full-trace]

Architecture:
    Python still loads pickled Boogie ASTs and performs one-time PyO3 lowering.
    Per-input concretization and execution run in Rust.
"""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import multiprocessing
import pickle
import struct
import time
from pathlib import Path
from multiprocessing.connection import wait as wait_for_process_message
import functools
import os

from swoosh_cli.layout import current_layout
from swoosh_cli.trace_contract import (
    TraceBundle,
    TraceContractError,
    TraceOutputPlan,
    TraceState,
    native_runtime_fingerprint,
)

from interpreter.errors import AssertViolation, AssumeViolation
from interpreter.utils.debug_log import DebugLogger
from interpreter.utils.program import find_entry_point
from interpreter.utils.support_matrix import support_matrix_summary
from interpreter.utils.static_eval import (
    compute_static_scalars,
    compute_static_values,
    expr_base_ptr,
    offset_delta,
)


def _normalize_engine_result(result: dict, *, engine: str, input_name: str) -> dict:
    out = dict(result)
    out["engine"] = engine
    out["input_name"] = input_name
    out["status"] = out.get("status", "ok")
    out["explored_blocks"] = set(out.get("explored_blocks") or [])
    out.setdefault("violation_pc", None)
    out.setdefault("violation_block", None)
    out.setdefault("message", None)
    out.setdefault("trace_records", None)
    out.setdefault("memory_summary", {})
    if out["status"] == "assume_violation":
        out["invalid_input"] = True
        out.setdefault("invalid_reason", "assume")
    else:
        out.setdefault("invalid_input", False)
        out.setdefault("invalid_reason", None)
    return out


_LEGACY_PYTHON_MESSAGE = (
    "The Python interpreter runtime has been archived at "
    "archive/legacy_python/runtime/python and is no longer an active engine. "
    "Use the Rust native engine."
)


def _legacy_python_runtime_disabled(feature: str):
    raise RuntimeError(f"{feature} is deprecated. {_LEGACY_PYTHON_MESSAGE}")


def _reject_legacy_engine(engine: str):
    if engine != "native":
        _legacy_python_runtime_disabled(f"engine={engine!r}")


def _admit_runtime_generation(
    expected_fingerprint: str | None,
    *,
    read_trace: bool,
    full_trace: bool,
) -> str:
    """Require the exec'd runner to match its parent's producer generation."""
    current = native_runtime_fingerprint(
        read_trace=read_trace,
        full_trace=full_trace,
    )
    if expected_fingerprint is not None and expected_fingerprint != current:
        raise RuntimeError(
            "interpreter runtime changed between Swoosh launch and child "
            "admission: expected "
            f"{expected_fingerprint}, got {current}"
        )
    return current


def _build_loop_header_live(test_path, *, package_outputs=None):
    """Load loop header → live variable names from compiled package metadata.

    Returns dict {block_name: [var_name, ...]} suitable for swoosh_interp.lower().
    """
    pkg_dir = test_path.parent
    name = test_path.stem
    try:
        if package_outputs is None:
            live_in_path = pkg_dir / f"{name}_live_in.pkl"
            loops_path = pkg_dir / f"{name}_loops.pkl"
            if not live_in_path.exists() or not loops_path.exists():
                return None
            live_in = pickle.loads(live_in_path.read_bytes())
            loops = pickle.loads(loops_path.read_bytes())
        else:
            live_in_bytes = package_outputs.get(f"{name}_live_in.pkl")
            loops_bytes = package_outputs.get(f"{name}_loops.pkl")
            if live_in_bytes is None or loops_bytes is None:
                return None
            live_in = pickle.loads(live_in_bytes)
            loops = pickle.loads(loops_bytes)
        result = {}
        for proc, proc_loops in loops.items():
            proc_live = live_in.get(proc, {})
            for header_block in proc_loops:
                live_vars = proc_live.get(header_block, set())
                if live_vars:
                    result[header_block] = sorted(live_vars)
        return result if result else None
    except Exception:
        return None


def _build_loop_metadata(test_path, *, package_outputs=None):
    """Load loop nesting metadata from the compiled package.

    Returns a dict with three keys ready for ``swoosh_interp.lower``:

      * ``is_loop_header``: list of block NAMES that are loop headers.
      * ``block_innermost_header``: dict block_name -> innermost header name.
      * ``loop_parent_header``: dict inner_header -> parent header name.

    The Rust side resolves names to block ids via its own label map, so
    no block-id numbering has to agree across the FFI boundary.
    Returns None if any required .pkl is missing.
    """
    pkg_dir = test_path.parent
    name = test_path.stem
    try:
        filenames = (
            f"{name}_loops.pkl",
            f"{name}_loop_parents.pkl",
            f"{name}_block_to_loop.pkl",
        )
        if package_outputs is None:
            paths = tuple(pkg_dir / filename for filename in filenames)
            if any(not path.exists() for path in paths):
                return None
            values = tuple(pickle.loads(path.read_bytes()) for path in paths)
        else:
            encoded = tuple(package_outputs.get(filename) for filename in filenames)
            if any(value is None for value in encoded):
                return None
            values = tuple(pickle.loads(value) for value in encoded)
        loops, parents, btl = values

        # Flatten across procedures.  The package only ever has one
        # entry procedure after inlining, but the dicts are still
        # keyed by proc name, so we union them.
        headers = set()
        for proc_headers in loops.values():
            headers.update(proc_headers)

        block_to_header = {}
        for proc_map in btl.values():
            block_to_header.update(proc_map)

        parent_map = {}
        for proc_parents in parents.values():
            for inner, outer in proc_parents.items():
                if outer is not None:
                    parent_map[inner] = outer

        return {
            "is_loop_header": sorted(headers),
            "block_innermost_header": block_to_header,
            "loop_parent_header": parent_map,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared state for fork-based COW sharing across worker processes.
# Loaded once in the parent; forked children inherit via COW pages.
# ---------------------------------------------------------------------------
_SHARED_PROGRAM = None
_SHARED_COMPILED = None   # CompiledProgramWrapper from swoosh_interp.lower()
_SHARED_PREPARED = None   # PreparedNativeProgram
_SHARED_FIELD_SIZES = None


def _build_trace_name_tables(program, entry):
    """Return complete var/block name tables for raw init logs."""
    from interpreter.parser.declaration import StorageDeclaration, ConstantDeclaration

    var_names = []
    seen = set()

    def add_var(name):
        if name and name not in seen:
            seen.add(name)
            var_names.append(name)

    for d in program.declarations:
        if isinstance(d, StorageDeclaration):
            for n in d.names:
                add_var(n)
        elif isinstance(d, ConstantDeclaration):
            for n in d.names:
                add_var(n)
    if entry.body:
        for local_decl in entry.body.locals:
            if isinstance(local_decl, StorageDeclaration):
                for n in local_decl.names:
                    add_var(n)
    for p in entry.parameters:
        for n in p.names:
            add_var(n)

    block_names = ['GLOBAL']
    seen_b = {'GLOBAL'}
    if entry.body:
        for b in entry.body.blocks:
            if b.name not in seen_b:
                seen_b.add(b.name)
                block_names.append(b.name)
    return var_names, block_names


def _package_manifest_mode(test_path, *, package_manifest=None):
    """Semantics mode recorded in the package build manifest, or None.

    Reads ``<pkg_dir>/<name>.manifest.json`` written by ``tools/compile.py``.
    Returns "int"/"bv" from the ``integer_encoding`` key, or None when the
    manifest (or the key — legacy pre-mode packages) is absent.
    """
    import json

    from interpreter.utils.integer_encoding import mode_from_integer_encoding

    if test_path is None:
        return None
    if package_manifest is None:
        manifest_path = (
            Path(test_path).parent / f"{Path(test_path).stem}.manifest.json"
        )
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path) as fp:
                manifest = json.load(fp)
        except (json.JSONDecodeError, OSError):
            return None
    else:
        manifest = package_manifest
    flag = manifest.get("integer_encoding")
    if flag is None:
        return None
    return mode_from_integer_encoding(flag)


def _check_semantics_mode(ast_mode, manifest_mode, compiled_mode, *, context):
    """FAIL LOUDLY if any two PRESENT semantics-mode tags disagree.

    The three tags are: content-derived from the loaded AST
    (``detect_integer_encoding``), the package manifest's
    ``integer_encoding`` key, and the mode baked into an already-compiled
    program (``swoosh_interp`` wrapper / ``.swcp``). Absent tags (legacy
    packages / manifests) are skipped — absence cannot disagree.
    """
    tags = {
        "ast": ast_mode,
        "manifest": manifest_mode,
        "compiled": compiled_mode,
    }
    present = {k: v for k, v in tags.items() if v is not None}
    if len(set(present.values())) > 1:
        raise RuntimeError(
            f"semantics-mode mismatch for {context}: {present} — the package "
            "on disk, the loaded AST and/or the pre-lowered bytecode were "
            "produced under different integer encodings. Rebuild the package "
            "('./swoosh build <name>') and any .swcp so all tags agree; "
            "running would evaluate the program under the wrong arithmetic."
        )


class PreparedNativeProgram:
    """Static native-run state reused across many ProgramInputs.

    The Rust VM is already fast for warm no-trace executions. This object
    removes repeated Python-side scans from the hot path: finding entry
    declarations, block/PC metadata, external-input annotations, pointer
    aliases, declaration initialization shape, trace name tables, and native
    lowering.
    """

    def __init__(
        self,
        program,
        *,
        test_path=None,
        compiled=None,
        package_publication=None,
    ):
        from interpreter.parser.declaration import (
            AxiomDeclaration,
            ImplementationDeclaration,
            ProcedureDeclaration,
        )
        from interpreter.utils.inputs import preprocess_external_inputs, gather_ptr_aliases
        from interpreter.utils.program import generate_label_to_block, initialize_code_metadata
        from interpreter.parser.desugar import desugar_while_statements

        self.program = program
        # Rewrite any structured `while` loops into goto-form blocks before
        # we read PC/label metadata or invoke the Rust lowering — the native
        # interpreter has no opcode for `WhileStatement`, and downstream
        # passes (initialize_code_metadata, swoosh_interp.lower) must agree
        # on the block list. The pass is a no-op for SMACK-generated input.
        desugar_while_statements(self.program)
        self.test_path = Path(test_path) if test_path is not None else None
        self.package_manifest = (
            package_publication.manifest
            if package_publication is not None
            else None
        )
        self.package_outputs = (
            {output.name: output.content for output in package_publication.outputs}
            if package_publication is not None
            else None
        )

        impl_decls = [
            d for d in program.declarations
            if isinstance(d, ImplementationDeclaration) and d.body
        ]
        impl_names = {d.name for d in impl_decls}
        proc_decls = [
            d for d in program.declarations
            if isinstance(d, ProcedureDeclaration)
            and not isinstance(d, ImplementationDeclaration)
            and d.name in impl_names
        ]
        assert len(proc_decls) == 1 and len(impl_decls) == 1, (
            f"We only support inlined procedures. "
            f"{[p.name for p in proc_decls]} {len(proc_decls)} {len(impl_decls)} "
            f"{[d.name for d in program.declarations if isinstance(d, ProcedureDeclaration)]}"
        )

        self.impl_decl = impl_decls[0]
        self.impl_decl_name = self.impl_decl.name
        self.proc_decl = proc_decls[0]
        self.entry = self.impl_decl
        self.label_to_block = generate_label_to_block(program)
        (
            self.pc_to_stmt,
            self.label_to_pc,
            self.pc_to_block_name,
            self.pc_to_label,
        ) = initialize_code_metadata(self.impl_decl)
        self.global_axioms = tuple(
            decl for decl in program.declarations
            if isinstance(decl, AxiomDeclaration)
        )
        self.arr_inputs, self.field_inputs = preprocess_external_inputs(self.impl_decl)
        self.ptr_aliases = gather_ptr_aliases(self.impl_decl)
        self.trace_var_names, self.trace_block_names = _build_trace_name_tables(
            program, self.entry)

        # Semantics mode is content-derived from the loaded AST (never from
        # flags), cross-checked against the package manifest and against any
        # pre-lowered bytecode handed in (which carries the mode it was
        # lowered/serialized with). Any disagreement is a hard error.
        from interpreter.utils.integer_encoding import detect_semantics_mode
        self.semantics_mode = detect_semantics_mode(self.program)
        _check_semantics_mode(
            self.semantics_mode,
            _package_manifest_mode(
                self.test_path,
                package_manifest=self.package_manifest,
            ),
            getattr(compiled, "mode", None) if compiled is not None else None,
            context=(f"package {self.test_path}" if self.test_path is not None
                     else f"program entry {self.impl_decl_name!r}"),
        )

        self.compiled = compiled if compiled is not None else self._lower_program()
        self.native_meta = self._build_native_meta()

    def _lower_program(self):
        import swoosh_interp

        try:
            if self.test_path is not None:
                lh_live = _build_loop_header_live(
                    self.test_path, package_outputs=self.package_outputs
                )
                loop_meta = _build_loop_metadata(
                    self.test_path, package_outputs=self.package_outputs
                )
                return swoosh_interp.lower(self.program, lh_live, loop_meta,
                                           mode=self.semantics_mode)
            return swoosh_interp.lower(self.program, mode=self.semantics_mode)
        except TypeError as exc:
            if "mode" not in str(exc):
                raise
            raise RuntimeError(
                "Installed Rust native module is too old: lower() has no "
                "semantics-mode parameter. Rebuild with: "
                "cd interpreter/native && maturin develop --release"
            ) from exc

    def _build_native_meta(self):
        """Build static metadata for Rust-side per-input concretization."""
        static_scalars = compute_static_scalars(self.program)
        static_values = compute_static_values(self.program)

        arr_inputs = {}
        for key, infos in self.arr_inputs.items():
            arr_inputs[key] = [
                {
                    "mem_map": ai.mem_map,
                    "base_ptr": ai.base_ptr,
                    "offset_delta": offset_delta(ai.offset, ai.base_ptr, static_values),
                    "elem_size": ai.elem_size,
                    "num_elements": ai.num_elements,
                }
                for ai in infos
            ]

        field_inputs = {}
        for key, infos in self.field_inputs.items():
            items = []
            for fi in infos:
                base_ptr = expr_base_ptr(fi.base_ptr)
                items.append({
                    "var_name": fi.var_name,
                    "mem_map": fi.mem_map,
                    "base_ptr": base_ptr,
                    "offset_delta": offset_delta(fi.base_ptr, base_ptr, static_values),
                    "size": fi.size,
                })
            field_inputs[key] = items

        return {
            "static_scalars": static_scalars,
            "arr_inputs": arr_inputs,
            "field_inputs": field_inputs,
            "ptr_aliases": dict(self.ptr_aliases),
        }


def prepare_native(
    program, *, test_path=None, compiled=None, package_publication=None
):
    """Prepare reusable static state for high-throughput native execution."""
    return PreparedNativeProgram(
        program,
        test_path=test_path,
        compiled=compiled,
        package_publication=package_publication,
    )


# ---------------------------------------------------------------------------
# Injected asserts (falsification probes)
# ---------------------------------------------------------------------------
#
# `--inject-assert PC:EXPR` inserts `assert EXPR;` at PC and evaluates it
# in-state; `--inject-at {before,after}` (default `before`) picks WHICH state:
#   before — insert immediately BEFORE stmt(PC): evaluates in the state on
#            ENTRY to PC (pre-state). This is the CANDIDATE / precondition role:
#            an abduced sufficient condition C must hold BEFORE stmt(PC) runs.
#   after  — insert immediately AFTER stmt(PC), before the block terminator:
#            evaluates in the state on EXIT from PC (post-state). This is the
#            OBLIGATION / postcondition role: an obligation (pc, P) means P
#            holds AFTER stmt(pc), so a value DEFINED at PC reads its freshly-
#            assigned value, not a stale entry value. Refused when stmt(PC) is
#            itself a goto/return (no reachable slot after a block terminator).
# The predicate is injected RAW either way — placement is the only axis; there
# is no weakest-precondition transform. A concrete input that fires the assert
# is a witness the obligation/candidate is NOT invariant; a surviving run
# proves nothing (one input), it is only recorded evidence.
#
# `--probe-block LABEL` is observation-only. Labels are validated before
# execution, then each successfully completed run reports whether LABEL is in
# its explored-block set. A probe never mutates control flow or terminates the
# run, and an early-terminated run emits no block-probe verdict.

_INJECT_ASSERT_SPECS = []     # [(pc:int, expr_text:str)] set by main()
_INJECT_ASSERT_AST_SPECS = []  # [(pc, expr_ast, kind)] set by main() (no parse)
_INJECT_WHERE = "before"      # "before" | "after"; injected asserts only
_PROBE_BLOCK_SPECS = []       # [block_label:str] set by main()
_INJECTED_ASSERTS = {}        # final_pc -> {kind, expr, block, requested_pc}
_INJECT_ASSERT_KINDS = frozenset({"predicate", "carrier_guard"})


def _normalize_inject_assert_ast_specs(rows):
    """Normalize pickle rows to ``(pc, expr_ast, kind)`` triples.

    Two-element rows are the legacy predicate form. Three-element rows may
    explicitly identify a normal predicate or the reachability carrier guard.
    """
    normalized = []
    for index, row in enumerate(rows or ()):
        if not isinstance(row, (tuple, list)) or len(row) not in (2, 3):
            raise ValueError(
                "inject-assert-ast: row "
                f"{index} must be (pc, expr_ast) or (pc, expr_ast, kind)")
        pc, expr_ast = row[:2]
        kind = "predicate" if len(row) == 2 else str(row[2]).strip()
        if kind not in _INJECT_ASSERT_KINDS:
            allowed = "|".join(sorted(_INJECT_ASSERT_KINDS))
            raise ValueError(
                f"inject-assert-ast: row {index} kind must be {allowed}, "
                f"got {kind!r}")
        normalized.append((int(pc), expr_ast, kind))
    return normalized


def inject_asserts(program, assert_specs, block_probes, ast_specs=(),
                   where="before"):
    """Mutate ``program`` in place, inserting requested asserts; return the
    final-pc map (the program is renumbered by the insertions).

    ``assert_specs`` are ``(pc, expr_text)`` pairs whose text is parsed.
    ``ast_specs`` are ``(pc, expr_ast)`` legacy pairs or
    ``(pc, expr_ast, kind)`` triples carrying a PRE-BUILT Boogie expression
    AST (an ``interpreter.parser.expression`` node) — injected verbatim, with
    NO text parse. ``kind`` is ``predicate`` or ``carrier_guard`` and is
    retained in the result metadata and execution verdict. This is the
    obligation-faithful path: the frozen obligation's live cvc5 term is
    lowered to AST upstream (``cvc5_to_boogie_ast``) so the predicate never
    round-trips through a lossy infix display string (``A => B /\\ C`` with
    tabs, ambiguous grouping).

    ``where`` (``"before"`` | ``"after"``) picks the state a PREDICATE assert
    evaluates in — pre-state (entry to pc, candidate/precondition role) vs
    post-state (exit from pc, obligation/postcondition role). Block probes are
    observation-only and are never inserted. ``"after"`` is refused when
    ``stmt(pc)`` is a block terminator (goto/return)."""
    from interpreter.parser.boogie_parser import parse_expr
    from interpreter.parser.statement import (
        AssertStatement, GotoStatement, ReturnStatement)
    from interpreter.parser.declaration import ImplementationDeclaration
    from interpreter.parser.desugar import desugar_while_statements
    from interpreter.utils.program import initialize_code_metadata

    # Pin the numbering the specs refer to (no-op for SMACK input).
    desugar_while_statements(program)
    impl = next(d for d in program.declarations
                if isinstance(d, ImplementationDeclaration) and d.body)
    pc_to_stmt, _, pc_to_block, _ = initialize_code_metadata(impl)
    blocks_by_name = {b.name: b for b in impl.body.blocks}

    # Validate every requested label before mutating the program. The actual
    # verdict is read from run_native's explored-block set after successful
    # execution, so probes never inject statements.
    for label in block_probes:
        if label not in blocks_by_name:
            raise ValueError(f"probe-block: no block labeled {label!r}")

    # Each work item carries an expression payload tagged ("text", str) for the
    # parse path or ("ast", node) for the pre-built-AST path.
    work = []  # (pc_for_ordering, ("text"|"ast", payload), kind, block_label)
    for pc, text in assert_specs:
        pc = int(pc)
        if pc not in pc_to_stmt:
            raise ValueError(f"inject-assert: pc {pc} is not a statement pc")
        work.append((pc, ("text", str(text)), "predicate", pc_to_block[pc]))
    for pc, expr_ast, kind in _normalize_inject_assert_ast_specs(ast_specs):
        if pc not in pc_to_stmt:
            raise ValueError(
                f"inject-assert-ast: pc {pc} is not a statement pc")
        work.append((pc, ("ast", expr_ast), kind, pc_to_block[pc]))

    # `after` (post-state) has no reachable slot when stmt(pc) is the block
    # terminator — reject loudly rather than silently landing pre-terminator
    # in a way that misreads the obligation.
    if where == "after":
        for pc, _payload, _kind, _label in work:
            if isinstance(pc_to_stmt[int(pc)],
                          (GotoStatement, ReturnStatement)):
                raise ValueError(
                    f"inject-assert: cannot insert 'after' pc {pc}: it is a "
                    f"{type(pc_to_stmt[int(pc)]).__name__} (block terminator); "
                    f"no reachable slot exists after it")

    inserted = []  # (stmt_obj, kind, expr_text, block_label, requested_pc)
    # Insert bottom-up so earlier insertions don't shift later targets.
    for pc, (ptype, payload), kind, label in sorted(
            work, key=lambda w: -int(w[0])):
        stmt = AssertStatement()
        stmt.expression = payload if ptype == "ast" else parse_expr(payload)
        # `expr` is for the [INJECTED_ASSERT] log line only; render the AST.
        expr_text = repr(payload) if ptype == "ast" else payload
        blk = blocks_by_name[label]
        # before → at stmt(pc) (pre-state); after → one past it (post-state,
        # before the terminator, guaranteed a valid slot by the guard above).
        idx = blk.statements.index(pc_to_stmt[int(pc)])
        blk.statements.insert(idx + (1 if where == "after" else 0), stmt)
        inserted.append((stmt, kind, expr_text, label, int(pc)))

    new_pc_to_stmt, _, _, _ = initialize_code_metadata(impl)
    by_id = {id(s): (k, t, l, rp) for s, k, t, l, rp in inserted}
    final = {}
    for fpc, stmt in new_pc_to_stmt.items():
        meta = by_id.get(id(stmt))
        if meta is not None:
            final[int(fpc)] = {"kind": meta[0], "expr": meta[1],
                               "block": meta[2], "requested_pc": meta[3]}
    return final


def _emit_block_probe_statuses(input_name, explored):
    """Report passive block observations for one successfully completed run."""
    explored = set(explored or ())
    for label in _PROBE_BLOCK_SPECS:
        status = "BLOCK_REACHED" if label in explored else "BLOCK_NOT_REACHED"
        print(f"[{status}] input={input_name} block={label!r}")


def _load_shared(test_path, engine, *, package_publication):
    """Load program + compile bytecode once in the parent process."""
    global _SHARED_PROGRAM, _SHARED_COMPILED, _SHARED_PREPARED, _SHARED_FIELD_SIZES
    global _INJECTED_ASSERTS
    _reject_legacy_engine(engine)

    program_output = next(
        output for output in package_publication.outputs
        if output.name == Path(test_path).name
    )
    _SHARED_PROGRAM = pickle.loads(program_output.content)

    if _INJECT_ASSERT_SPECS or _INJECT_ASSERT_AST_SPECS or _PROBE_BLOCK_SPECS:
        import json as _json
        try:
            _INJECTED_ASSERTS = inject_asserts(
                _SHARED_PROGRAM, _INJECT_ASSERT_SPECS, _PROBE_BLOCK_SPECS,
                ast_specs=_INJECT_ASSERT_AST_SPECS, where=_INJECT_WHERE)
        except Exception as ex:
            print(f"[INJECT_ASSERT_ERROR] error={_json.dumps(str(ex))}")
            raise
        for fpc, meta in sorted(_INJECTED_ASSERTS.items()):
            print(f"[INJECTED_ASSERT] pc={fpc} "
                  f"requested_pc={meta['requested_pc']} "
                  f"kind={meta['kind']} block={meta['block']!r} "
                  f"expr={_json.dumps(meta['expr'])}")
        # Nondet-site visibility for falsification callers: which pc assigns
        # which $-variable from a nondet call, so an attack input authored with
        # an inverted `@params` mapping (e.g. n/k swapped) is visible in the
        # tool result instead of silently running a vacuous attack.
        try:
            from interpreter.parser.declaration import ImplementationDeclaration
            from interpreter.parser.statement import CallStatement
            from interpreter.utils.program import initialize_code_metadata
            _impl = next(d for d in _SHARED_PROGRAM.declarations
                         if isinstance(d, ImplementationDeclaration) and d.body)
            _pc_to_stmt, _, _, _ = initialize_code_metadata(_impl)
            _sites = [
                {"pc": int(fpc),
                 "var": ", ".join(str(a) for a in (stmt.assignments or [])),
                 "stmt": repr(stmt).strip()[:80]}
                for fpc, stmt in sorted(_pc_to_stmt.items())
                if isinstance(stmt, CallStatement)
                and "nondet" in str(stmt.procedure).lower()]
            if _sites:
                print(f"[NONDET_SITES] {_json.dumps(_sites)}")
        except Exception:
            pass

    from interpreter.utils.input_parser import get_bpl_field_sizes
    _SHARED_FIELD_SIZES = get_bpl_field_sizes(test_path.parent, program=_SHARED_PROGRAM)

    try:
        __import__("swoosh_interp")
    except ImportError as exc:
        raise ImportError(
            "Rust native interpreter not built. Run: "
            "cd interpreter/native && maturin develop --release"
        ) from exc

    # Lower inside PreparedNativeProgram so the content-derived semantics
    # mode is threaded to swoosh_interp.lower and cross-checked against the
    # package manifest in ONE place (loop-header/loop-metadata loading is
    # identical — see PreparedNativeProgram._lower_program).
    _SHARED_PREPARED = prepare_native(
        _SHARED_PROGRAM,
        test_path=test_path,
        package_publication=package_publication,
    )
    _SHARED_COMPILED = _SHARED_PREPARED.compiled


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_reachable_blocks(program):
    """Compute the set of blocks reachable from the entry block via the CFG.

    After inlining, some blocks become unreachable dead code. This function
    walks the CFG from the entry block to find only reachable blocks.
    Returns (reachable_set, entry_block_name) or (None, None) if no entry.
    """
    entry = find_entry_point(program)
    if entry is None:
        return None, None

    blocks = entry.body.blocks
    if not blocks:
        return set(), None

    from interpreter.parser.statement import GotoStatement

    # Build adjacency from goto statements
    successors = {}
    for block in blocks:
        targets = set()
        if block.statements:
            last = block.statements[-1]
            if isinstance(last, GotoStatement):
                targets = {ident.name for ident in last.identifiers}
        successors[block.name] = targets

    # BFS from entry block
    entry_block = blocks[0].name
    visited = set()
    queue = [entry_block]
    while queue:
        b = queue.pop()
        if b in visited:
            continue
        visited.add(b)
        for s in successors.get(b, ()):
            if s not in visited:
                queue.append(s)

    return visited, entry_block


def compute_coverage(program, explored_blocks):
    """Compute block and statement coverage from explored blocks.

    Reports both raw totals (all blocks in the procedure) and reachable
    totals (blocks reachable from entry via CFG). After inlining, some
    blocks are dead code — reachable coverage is the meaningful metric.
    """
    entry = find_entry_point(program)
    if entry is None:
        return None
    blocks = entry.body.blocks
    block_stmts = {b.name: len(b.statements) for b in blocks}
    total_blocks = len(blocks)
    total_stmts = sum(block_stmts.values())
    covered_blocks = sum(1 for b in explored_blocks if b in block_stmts)
    covered_stmts = sum(block_stmts[b] for b in explored_blocks if b in block_stmts)

    # Compute reachable subset
    reachable, _ = compute_reachable_blocks(program)
    if reachable is not None:
        reachable_blocks_total = len(reachable & block_stmts.keys())
        reachable_stmts_total = sum(block_stmts[b] for b in reachable if b in block_stmts)
        unreachable_blocks = total_blocks - reachable_blocks_total
        # Coverage of reachable blocks only
        reachable_covered = sum(1 for b in explored_blocks if b in reachable and b in block_stmts)
        reachable_stmts_covered = sum(block_stmts[b] for b in explored_blocks if b in reachable and b in block_stmts)
    else:
        reachable_blocks_total = total_blocks
        reachable_stmts_total = total_stmts
        unreachable_blocks = 0
        reachable_covered = covered_blocks
        reachable_stmts_covered = covered_stmts

    return {
        "blocks_covered": covered_blocks,
        "blocks_total": total_blocks,
        "stmts_covered": covered_stmts,
        "stmts_total": total_stmts,
        "block_pct": round(100.0 * covered_blocks / total_blocks, 1) if total_blocks else 0,
        "stmt_pct": round(100.0 * covered_stmts / total_stmts, 1) if total_stmts else 0,
        "reachable_blocks_total": reachable_blocks_total,
        "reachable_blocks_covered": reachable_covered,
        "reachable_stmts_total": reachable_stmts_total,
        "reachable_stmts_covered": reachable_stmts_covered,
        "unreachable_blocks": unreachable_blocks,
        "reachable_block_pct": round(100.0 * reachable_covered / reachable_blocks_total, 1) if reachable_blocks_total else 0,
        "reachable_stmt_pct": round(100.0 * reachable_stmts_covered / reachable_stmts_total, 1) if reachable_stmts_total else 0,
        "explored_blocks": sorted(explored_blocks & block_stmts.keys()),
    }


def write_trace_binary(path, compact_trace):
    """Write a compact trace dict in the legacy streaming binary format.

    This compatibility writer is still used by golden trace tests and older
    trace readers.  New execution paths may use the raw v2 format, but v1 is a
    stable on-disk contract for pickled compact traces.
    """
    import zstandard as zstd

    categories = [
        (0, "pc_values"),
        (1, "block_values"),
        (2, "op_values"),
        (3, "pc_registry"),
        (4, "block_registry"),
    ]

    total = compact_trace.get("total", 0)
    cctx = zstd.ZstdCompressor(level=3, threads=-1)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as writer:
            writer.write(b"SWTR")
            writer.write(struct.pack("<BQ", 1, total))
            for cat_id, section_name in categories:
                section = compact_trace.get(section_name, {})
                writer.write(struct.pack("<BI", cat_id, len(section)))
                for key_str, members in section.items():
                    key_bytes = key_str.encode() if isinstance(key_str, str) else key_str
                    writer.write(struct.pack("<H", len(key_bytes)))
                    writer.write(key_bytes)
                    writer.write(struct.pack("<I", len(members)))
                    for member in members:
                        writer.write(struct.pack("<H", len(member)))
                        writer.write(member)
            writer.write(b"DONE")


# ---------------------------------------------------------------------------
# Native engine
# ---------------------------------------------------------------------------

#: The native execution status vocabulary. MINTED IN RUST — the PyO3
#: boundary (native/src/lib.rs) stamps these strings on the result dict; the
#: interpreter test suite pins this tuple against the Rust source so the two
#: cannot drift.
NATIVE_STATUSES = ("ok", "assert_violation", "assume_violation", "step_limit")


@dataclass(frozen=True)
class NativeResult:
    """Typed view of the dict the Rust VM returns across PyO3.

    The raw dict stays the wire shape (callers with ``return_status=True``
    receive it untouched); this wrapper is the ONE Python-side reading of
    its key vocabulary, so consumers stop re-guessing field names.
    """

    status: str
    violation_pc: "int | None"
    violation_block: "str | None"
    invalid_detail: str
    invalid_reason: str
    raw: dict

    @classmethod
    def from_dict(cls, result) -> "NativeResult":
        return cls(
            status=result.get("status", "ok"),
            violation_pc=result.get("violation_pc"),
            violation_block=result.get("violation_block"),
            invalid_detail=(result.get("invalid_detail") or "").strip(),
            invalid_reason=result.get("invalid_reason") or "assume",
            raw=result,
        )


def _finish_native_result(result, *, return_status):
    if return_status:
        return result

    native = NativeResult.from_dict(result)
    if native.status == 'assert_violation':
        raise AssertViolation(
            None,
            pc=native.violation_pc,
            block=native.violation_block,
            expr_str='<native assertion>',
        )
    if native.status == 'assume_violation':
        # `invalid_detail` (from the native VM) names exactly which assume failed
        # and the concrete values that violated it, e.g.
        #   ($i3 >= 0)  [where $i3=-1]
        # so an agent fixing a stale input knows precisely which precondition to
        # satisfy rather than just "an assume failed somewhere".
        if native.invalid_detail:
            _expr_str = (f"the {native.invalid_reason} condition is false: "
                         f"{native.invalid_detail}")
        else:
            _expr_str = (f"concrete {native.invalid_reason} failed "
                         f"(no condition detail available)")
        raise AssumeViolation(
            native.violation_pc,
            native.violation_block,
            _expr_str,
            reason=native.invalid_reason,
        )
    if native.status == 'step_limit':
        raise TimeoutError(
            "native execution step limit reached at "
            f"pc={native.violation_pc} "
            f"block={native.violation_block!r}"
        )

    return result['explored_blocks']


def run_native(program, program_inputs, test_name, input_name, raw_log_path,
               extra_data=None, log_read=True, compiled=None, no_trace=False,
               init_raw_log_path=None, return_status=False,
               debug_logger=None, prepared=None,
               return_memory_summary=True, validate_handoff=True,
               quiet=True, max_steps=0, return_scalar_summary=False,
               return_raw_memory=False):
    """Run the Rust native interpreter.

    ``raw_log_path`` is the ``.trace.raw.zst`` the Rust VM writes its
    execution records to.  ``init_raw_log_path`` is accepted for old callers
    but no separate Python-init trace is produced in the Rust-only runtime.

    Returns ``explored_blocks`` by default. With ``return_status=True``, returns
    the native result dict including ``status`` and any violation metadata.
    """
    try:
        import swoosh_interp
    except ImportError:
        raise ImportError(
            "Native interpreter not built. Run: cd interpreter/native && maturin develop --release"
        )

    debug = (debug_logger or DebugLogger.disabled()).bind(
        engine="native", input_name=input_name)
    debug.event("exec", "engine_start", engine="native",
                raw_log_path=raw_log_path, no_trace=no_trace,
                log_read=log_read, prepared=prepared is not None,
                return_memory_summary=return_memory_summary,
                validate_handoff=validate_handoff,
                init_raw_log_path=init_raw_log_path,
                quiet=quiet,
                return_scalar_summary=return_scalar_summary)

    if not hasattr(swoosh_interp, "execute_inputs"):
        raise RuntimeError(
            "Installed Rust native module is too old: execute_inputs is missing. "
            "Rebuild with: cd interpreter/native && maturin develop --release"
        )

    if prepared is None:
        prepared = prepare_native(program, compiled=compiled)
        compiled = prepared.compiled
    else:
        program = prepared.program
        if compiled is None:
            compiled = prepared.compiled

    if compiled is None:
        from interpreter.utils.integer_encoding import detect_semantics_mode
        compiled = swoosh_interp.lower(program,
                                       mode=detect_semantics_mode(program))

    ext_data = extra_data
    if ext_data is None and hasattr(program_inputs, "extra_data"):
        ext_data = program_inputs.extra_data

    env_updates = debug.native_env()
    old_env = {key: os.environ.get(key) for key in env_updates}
    try:
        os.environ.update(env_updates)
        try:
            result = swoosh_interp.execute_inputs(
                compiled,
                prepared.native_meta,
                program_inputs,
                str(raw_log_path),
                extra_data=ext_data,
                log_read=log_read,
                no_trace=no_trace,
                return_memory_summary=return_memory_summary,
                quiet=quiet,
                max_steps=max_steps,
                return_scalar_summary=return_scalar_summary,
                return_raw_memory=return_raw_memory,
                # The runner never reads the per-entry block sequence and on
                # long runs it is tens of millions of PyStrings — skip it.
                return_block_sequence=False,
            )
        except TypeError as exc:
            msg = str(exc)
            if ("return_scalar_summary" not in msg
                    and "return_raw_memory" not in msg
                    and "return_block_sequence" not in msg):
                raise
            if return_scalar_summary or return_raw_memory:
                raise RuntimeError(
                    "Installed Rust native module is too old for "
                    "return_scalar_summary/return_raw_memory. Rebuild with: "
                    "cd interpreter/native && maturin develop --release"
                ) from exc
            result = swoosh_interp.execute_inputs(
                compiled,
                prepared.native_meta,
                program_inputs,
                str(raw_log_path),
                extra_data=ext_data,
                log_read=log_read,
                no_trace=no_trace,
                return_memory_summary=return_memory_summary,
                quiet=quiet,
                max_steps=max_steps,
            )
    finally:
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    result = _normalize_engine_result(result, engine="native", input_name=input_name)
    result["init_ms"] = 0.0
    result["handoff_ms"] = result.get("state_ms", 0.0)
    result["prepared"] = True
    result["rust_input_state"] = True

    debug.event(
        "exec",
        "engine_end",
        engine="native",
        status=result.get("status", "ok"),
        explored_blocks=len(result.get("explored_blocks") or []),
        trace_records=result.get("trace_records"),
        exec_ms=result.get("exec_ms"),
        state_ms=result.get("state_ms"),
        init_ms=0.0,
        handoff_ms=result.get("handoff_ms"),
        prepared=True,
        rust_input_state=True,
        blocks_explored=result.get("blocks_explored"),
        max_steps=max_steps,
        violation_pc=result.get("violation_pc"),
        violation_block=result.get("violation_block"),
        memory_maps=len(result.get("memory_summary") or {}),
    )

    return _finish_native_result(result, return_status=return_status)


# ---------------------------------------------------------------------------
# Single-input processing
# ---------------------------------------------------------------------------

def process_single_input(input_file, test_name, test_path, engine='native',
                         force=False, full_trace=False, no_read_trace=False,
                         program=None, field_sizes=None, compiled=None,
                         prepared=None, debug_logger=None,
                         trace_output_plan: TraceOutputPlan | None = None):
    """Process one input selected by the parent's immutable output plan.

    Cache selection happens once, in :func:`main`.  This worker therefore
    always replaces the selected input's output pair instead of applying an
    independent existence-based skip rule.  ``force`` remains in the public
    signature for compatibility with direct callers.
    """
    try:
        _reject_legacy_engine(engine)
        if program is None:
            with open(test_path, 'rb') as file:
                program = pickle.load(file)

        input_name = Path(str(input_file)).stem
        debug = (debug_logger or DebugLogger.disabled()).bind(
            input_name=input_name, engine=engine)
        debug.event("exec", "input_start", input_file=str(input_file))

        print(f"Processing input file: {input_file}")
        from interpreter.utils.input_parser import parse_input_file, get_bpl_field_sizes
        if field_sizes is None:
            field_sizes = get_bpl_field_sizes(test_path.parent, program=program)
        program_inputs = parse_input_file(input_file, field_sizes=field_sizes)

        trace_dir = (
            trace_output_plan.trace_dir
            if trace_output_plan is not None
            else current_layout().trace_dir(test_name)
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        if trace_output_plan is not None:
            raw_log_path, explored_path = trace_output_plan.output_paths(
                Path(input_file).name
            )
            init_raw_log_path = trace_output_plan.init_output_path(
                Path(input_file).name
            )
        else:
            explored_path = trace_dir / f"{input_name}.explored_blocks.txt"
            raw_log_path = trace_dir / f"{input_name}.trace.raw.zst"
            init_raw_log_path = trace_dir / f"{input_name}.init.raw.zst"

        # A raw trace and explored-block marker are one logical output.  Remove
        # both before replacement so a failed run cannot pair new partial raw
        # bytes with an old completion marker.
        for p in (raw_log_path, init_raw_log_path, explored_path):
            if p.exists() or p.is_symlink():
                p.unlink()

        try:
            explored = run_native(
                program, program_inputs, test_name, input_name,
                raw_log_path=raw_log_path,
                init_raw_log_path=init_raw_log_path,
                extra_data=program_inputs.extra_data,
                log_read=not no_read_trace,
                compiled=compiled,
                debug_logger=debug,
                prepared=prepared,
            )
        except AssertViolation as v:
            # Structured line for agent_loop's subprocess parser.  The
            # parent process reads child stdout to collect per-input
            # violations; expression is JSON-quoted so embedded quotes
            # / newlines in the Boogie expr don't break the line.
            # Injected predicates own their pc (the insertion renumbered the
            # program), so a violation there is the injected assertion's
            # answer, not a program-assert failure. Passive block probes emit
            # nothing on this early-termination path.
            import json as _json
            inj = _INJECTED_ASSERTS.get(int(v.pc)) if _INJECTED_ASSERTS \
                else None
            if inj is not None:
                print(f"[INJECTED_ASSERT_VIOLATION] "
                      f"input={input_name} pc={inj['requested_pc']} "
                      f"kind={inj['kind']} block={v.block!r} "
                      f"expr={_json.dumps(inj['expr'])}")
                debug.event("exec", "input_injected_assert_violation",
                            pc=v.pc, requested_pc=inj["requested_pc"],
                            kind=inj["kind"], block=v.block,
                            expression=v.expr_str)
                return (input_name, None, set())
            print(f"[ASSERT_VIOLATION] "
                  f"input={input_name} pc={v.pc} block={v.block!r} "
                  f"expr={_json.dumps(v.expr_str)}")
            debug.event("exec", "input_assert_violation",
                        pc=v.pc, block=v.block, expression=v.expr_str)
            return (input_name, None, set())

        # Observation-only probes are reported only after run_native returned
        # normally. Assert/assume/error early exits above therefore emit no
        # potentially partial BLOCK_* verdicts.
        _emit_block_probe_statuses(input_name, explored)

        if _INJECTED_ASSERTS:
            # Run completed: every injected assert that executed HELD.
            # Block-visitation from the explored set tells executed-and-held
            # apart from never-reached.
            for fpc, meta in sorted(_INJECTED_ASSERTS.items()):
                visited = meta["block"] in explored
                print(f"[INJECTED_ASSERT_SURVIVED] input={input_name} "
                      f"pc={meta['requested_pc']} kind={meta['kind']} "
                      f"block={meta['block']!r} "
                      f"block_visited={'true' if visited else 'false'}")

        with open(explored_path, "w") as f:
            for block in sorted(explored):
                f.write(f"{block}\n")

        cov = compute_coverage(program, explored)
        if cov:
            print(f"[coverage] {input_name}: {cov['blocks_covered']}/{cov['blocks_total']} blocks ({cov['block_pct']}%), "
                  f"{cov['stmts_covered']}/{cov['stmts_total']} stmts ({cov['stmt_pct']}%)")
        debug.event("exec", "input_end", explored_blocks=len(explored),
                    coverage=cov)
        return (input_name, cov, explored)

    except Exception:
        import traceback
        traceback.print_exc()
        raise


def _process_input_shared(input_file, test_name, test_path, engine='native',
                          force=False, full_trace=False, no_read_trace=False,
                          debug_logger=None, trace_output_plan=None):
    """Worker function that uses fork-inherited _SHARED_* globals."""
    return process_single_input(
        input_file, test_name=test_name, test_path=test_path, engine=engine,
        force=force, full_trace=full_trace, no_read_trace=no_read_trace,
        program=_SHARED_PROGRAM, field_sizes=_SHARED_FIELD_SIZES,
        compiled=_SHARED_COMPILED,
        prepared=_SHARED_PREPARED,
        debug_logger=debug_logger,
        trace_output_plan=trace_output_plan,
    )


def _run_worker_to_conn(worker_func, input_file, conn):
    try:
        conn.send(("ok", worker_func(input_file)))
    except BaseException:
        import traceback
        conn.send(("error", traceback.format_exc()))
        raise
    finally:
        conn.close()



# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def _print_coverage_summary(results, test_name, test_path, *, trace_bundle=None):
    """Print aggregate coverage and write coverage.json."""
    per_input = {}
    all_explored = set()

    for input_name, cov, explored in results:
        if cov:
            per_input[input_name] = {k: v for k, v in cov.items() if k != "explored_blocks"}
        if explored:
            all_explored |= explored

    if not per_input:
        return

    # Compute aggregate coverage using the union of all explored blocks
    program = _SHARED_PROGRAM
    if program is None:
        with open(test_path, 'rb') as f:
            program = pickle.load(f)
    agg = compute_coverage(program, all_explored)
    if not agg:
        return

    n_inputs = len(per_input)
    unreachable = agg.get('unreachable_blocks', 0)
    rblk = f", {unreachable} unreachable" if unreachable else ""
    rblk_pct = f" (reachable: {agg['reachable_block_pct']}%)" if unreachable else ""
    print(f"\n[coverage] aggregate ({n_inputs} input{'s' if n_inputs != 1 else ''}): "
          f"{agg['blocks_covered']}/{agg['blocks_total']} blocks ({agg['block_pct']}%){rblk_pct}{rblk}, "
          f"{agg['stmts_covered']}/{agg['stmts_total']} stmts ({agg['stmt_pct']}%)")

    # Write coverage.json
    trace_dir = (
        trace_bundle.trace_dir
        if trace_bundle is not None
        else current_layout().trace_dir(test_name)
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    coverage_data = {
        "per_input": per_input,
        "aggregate": {k: v for k, v in agg.items() if k != "explored_blocks"},
    }
    with open(trace_dir / "coverage.json", "w") as f:
        json.dump(coverage_data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(publication_stack: ExitStack):
    parser = argparse.ArgumentParser(description='Rust Boogie interpreter runner')
    parser.add_argument('test_pkg_path', type=str, help='Path to the test package')
    parser.add_argument('--engine', choices=['native'], default='native',
                        help='Interpreter engine (native is the only supported runtime)')
    parser.add_argument('--force', action='store_true', help='Force re-interpretation')
    parser.add_argument('--full-trace', action='store_true', help='Write full text trace')
    parser.add_argument('--no-read-trace', action='store_true', help='Skip read tracing')
    parser.add_argument(
        '--expected-runtime-fingerprint',
        default=None,
        help=(
            'Parent-computed interpreter producer identity; Swoosh passes '
            'this to reject an exec that imported a different code generation'
        ),
    )
    parser.add_argument('--debug-log',
                        help='Directory for structured debug JSONL sidecar logs')
    parser.add_argument('--debug-categories', default='all',
                        help='Comma-separated debug categories '
                             '(default: all; e.g. exec,branch,solver)')
    parser.add_argument('--inject-assert', action='append', default=[],
                        metavar='PC:EXPR',
                        help='Inject `assert EXPR;` at flat pc PC, evaluating '
                             'in the state chosen by --inject-at (falsification '
                             'probe; repeatable)')
    parser.add_argument('--inject-at', choices=['before', 'after'],
                        default='before',
                        help='State for injected predicate/carrier asserts '
                             '(--inject-assert/--inject-assert-ast): before '
                             '(default) = pre-state on entry to pc '
                             '(candidate/precondition role); after = post-state '
                             'on exit from pc, before the block terminator '
                             '(obligation/postcondition role). Refused when pc '
                             'is itself a goto/return. Does NOT affect '
                             '--probe-block (observation-only).')
    parser.add_argument('--inject-assert-ast', default=None,
                        metavar='PICKLE_PATH',
                        help='Path to a pickle of [(pc:int, expr_ast)] or '
                             '[(pc:int, expr_ast, kind)] Boogie expression '
                             'ASTs to inject verbatim; kind is predicate or '
                             'carrier_guard — '
                             'the obligation-faithful path that lowers the '
                             'live cvc5 term to AST upstream instead of '
                             'parsing a lossy display string')
    parser.add_argument('--probe-block', action='append', default=[],
                        metavar='LABEL',
                        help='Observe block LABEL without changing execution; '
                             'after a successful run reports BLOCK_REACHED or '
                             'BLOCK_NOT_REACHED from the explored-block set '
                             '(repeatable)')
    parser.add_argument('--diagnostic-input', default=None,
                        metavar='INPUT_PATH',
                        help='Execute exactly one scratch .input file for an '
                             'injected assertion or block probe. The file is '
                             'admitted through the same typed trace recipe as '
                             'canonical inputs, but is not published as part '
                             'of the target input corpus.')
    args = parser.parse_args()
    try:
        _reject_legacy_engine(args.engine)
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        _admit_runtime_generation(
            args.expected_runtime_fingerprint,
            read_trace=not args.no_read_trace,
            full_trace=args.full_trace,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    global _INJECT_ASSERT_SPECS, _INJECT_ASSERT_AST_SPECS, _PROBE_BLOCK_SPECS
    global _INJECT_WHERE
    _INJECT_WHERE = args.inject_at
    for spec in args.inject_assert:
        pc_s, sep, expr = str(spec).partition(':')
        if not sep or not pc_s.strip().isdigit() or not expr.strip():
            parser.error(f"--inject-assert expects PC:EXPR, got {spec!r}")
        _INJECT_ASSERT_SPECS.append((int(pc_s), expr.strip()))
    if args.inject_assert_ast:
        try:
            with open(args.inject_assert_ast, 'rb') as _f:
                loaded = pickle.load(_f)
            _INJECT_ASSERT_AST_SPECS.extend(
                _normalize_inject_assert_ast_specs(loaded))
        except Exception as ex:
            parser.error(
                f"--inject-assert-ast could not load {args.inject_assert_ast!r}: "
                f"{type(ex).__name__}: {ex}")
    _PROBE_BLOCK_SPECS.extend(str(b).strip() for b in args.probe_block
                              if str(b).strip())
    instrumented_run = bool(
        _INJECT_ASSERT_SPECS or _INJECT_ASSERT_AST_SPECS or _PROBE_BLOCK_SPECS
    )
    if args.diagnostic_input and not instrumented_run:
        parser.error(
            '--diagnostic-input requires an injected assertion or block probe'
        )
    if instrumented_run:
        # Injection/probe runs must actually execute and never reuse stale
        # results. Assertions mutate the program; passive block probes need
        # one complete execution for an input-scoped observation.
        args.force = True

    test_pkg_dir = Path(args.test_pkg_path)
    test_name = test_pkg_dir.name.removesuffix("_pkg")
    test_path = test_pkg_dir / f"{test_name}.pkl"
    from tools.build_manifest import ValidatedPackagePublication
    from tools.package_contract import package_publication_lock
    try:
        publication_stack.enter_context(
            package_publication_lock(test_pkg_dir, exclusive=False)
        )
        package_publication = ValidatedPackagePublication.open(
            test_pkg_dir,
            test_name,
            capture_outputs=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"package publication is invalid: {exc}")
    debug_logger = DebugLogger.from_options(
        args.debug_log,
        args.debug_categories,
        run_id=f"{test_name}-{int(time.time())}",
    ).bind(test_name=test_name, command="runner")
    debug_logger.event("exec", "runner_start",
                       test_pkg_path=str(test_pkg_dir), engine=args.engine)

    input_directory = current_layout().input_dir(test_name)
    trace_dir = current_layout().trace_dir(test_name)
    diagnostic_inputs = None
    if args.diagnostic_input:
        diagnostic_inputs = (Path(args.diagnostic_input),)
    selected_inputs = diagnostic_inputs
    if selected_inputs is None:
        from swoosh_cli.input_contract import read_input_publication

        input_publication = publication_stack.enter_context(
            read_input_publication(
                input_directory,
                expected_target=test_name,
            )
        )
        selected_inputs = input_publication.paths
    try:
        bundle = TraceBundle.create(
            test_name,
            package_dir=test_pkg_dir,
            input_dir=input_directory,
            trace_dir=trace_dir,
            input_paths=selected_inputs,
            read_trace=not args.no_read_trace,
            full_trace=args.full_trace,
        )
    except (OSError, TraceContractError) as exc:
        parser.error(str(exc))
    if bundle.recipe.package_fingerprint != package_publication.fingerprint:
        parser.error(
            "package publication changed while constructing the trace recipe"
        )
    all_input_files = [
        input_directory / item.filename for item in bundle.recipe.inputs
    ]

    # The current recipe owns the canonical raw/marker/init filenames.  Drop
    # contract-owned outputs for inputs that disappeared before inspecting a
    # prior manifest, so a corpus shrink cannot leak evidence downstream.
    removed_outputs = bundle.prune_obsolete_outputs()
    if removed_outputs:
        debug_logger.event(
            "exec",
            "trace_obsolete_outputs_removed",
            files=[path.name for path in removed_outputs],
        )

    inspection = bundle.inspect()
    debug_logger.event(
        "exec",
        "runner_context",
        package_fingerprint=bundle.recipe.package_fingerprint,
        native_runtime_fingerprint=bundle.recipe.native_runtime_fingerprint,
        trace_state=inspection.state.value,
        trace_reason=inspection.reason,
    )
    debug_logger.event("unsupported", "support_matrix",
                       summary=support_matrix_summary())

    if inspection.state is TraceState.NOT_APPLICABLE:
        # An empty corpus publishes the exact empty output set.  A manifest
        # from an older non-empty corpus must not remain as a plausible commit
        # marker even though consumers already ignore it.
        bundle.invalidate_manifest()
        print(f"No input files found in {input_directory} — tracing not applicable")
        return
    if not args.force and inspection.is_ready:
        print("Skipping interpretation — trace recipe and all outputs match")
        return

    if args.force or inspection.state is TraceState.STALE:
        pending_names = {item.filename for item in bundle.recipe.inputs}
    else:
        pending_names = set(inspection.pending_inputs)
    input_files = [path for path in all_input_files if path.name in pending_names]
    if not input_files:
        parser.error(
            f"trace bundle is {inspection.state.value} but has no pending inputs: "
            f"{inspection.reason}"
        )

    # Publication is a parent-owned transaction.  Revoke the old commit and
    # remove every selected output pair *before* workers parse inputs or start
    # native execution.  Otherwise an early parse/startup failure can leave an
    # old pair in place and a later write_manifest() can falsely certify those
    # bytes under the new package/runtime/input recipe.
    publication = publication_stack.enter_context(
        bundle.rebuild(input_file.name for input_file in input_files)
    )
    worker_output_plan = publication.output_plan()

    max_workers = min(max(1, os.cpu_count() - 1), len(input_files))
    print(
        f"Using {max_workers} workers for {len(input_files)}/{len(all_input_files)} "
        f"inputs (engine={args.engine}; {inspection.reason})"
    )

    # Load program + compile bytecode once, then fork workers to share via COW
    assert hasattr(os, 'fork'), "Fork-based multiprocessing required (Linux only)"
    t0 = time.time()
    _load_shared(
        test_path,
        args.engine,
        package_publication=package_publication,
    )
    print(f"Loaded program + compiled bytecode in {time.time() - t0:.1f}s (shared via fork)")
    mp_context = multiprocessing.get_context('fork')
    worker_func = functools.partial(
        _process_input_shared,
        test_name=test_name,
        test_path=test_path,
        engine=args.engine,
        # Selection was validated centrally.  A selected worker must replace
        # its pair even when this is a non-forced partial resume.
        force=True,
        full_trace=args.full_trace,
        no_read_trace=args.no_read_trace,
        debug_logger=debug_logger,
        trace_output_plan=worker_output_plan,
    )

    per_input_timeout = int(os.environ.get("SWOOSH_INTERP_TIMEOUT", "60"))
    # A value of 0 (or negative) means "run until completion, no timeout".
    if per_input_timeout <= 0:
        per_input_timeout = None
    debug_logger.event("exec", "runner_workers",
                       inputs=len(input_files),
                       max_workers=max_workers,
                       per_input_timeout=per_input_timeout)
    failed = False
    skipped = 0
    results = []

    pending = iter(input_files)
    active = []

    def _start_worker(input_file):
        result_recv, result_send = mp_context.Pipe(duplex=False)
        p = mp_context.Process(
            target=_run_worker_to_conn,
            args=(worker_func, input_file, result_send),
        )
        p.start()
        result_send.close()
        return {
            "input_file": input_file,
            "input_name": Path(str(input_file)).name,
            "process": p,
            "conn": result_recv,
            "started": time.monotonic(),
        }

    def _finish_worker(task, status, payload):
        nonlocal failed, skipped
        input_file = task["input_file"]
        input_name = task["input_name"]
        p = task["process"]
        conn = task["conn"]
        if status == "timeout" and p.is_alive():
            p.kill()
        p.join(timeout=5)
        try:
            conn.close()
        except Exception:
            pass

        if status == "timeout":
            print(f"TIMEOUT ({per_input_timeout}s) on {input_name} — killing (likely infinite loop)")
            skipped += 1
            debug_logger.event("exec", "input_timeout",
                               input_file=str(input_file),
                               timeout=per_input_timeout)
        elif p.exitcode != 0 and p.exitcode is not None:
            print(f"Worker failed on {input_name} (exit={p.exitcode})")
            debug_logger.event("exec", "input_worker_failed",
                               input_file=str(input_file),
                               exitcode=p.exitcode,
                               error=payload if status == "error" else None)
            failed = True
        elif status is None:
            print(f"Worker failed on {input_name}: worker exited without result")
            debug_logger.event("exec", "input_worker_failed",
                               input_file=str(input_file),
                               exitcode=p.exitcode,
                               error="worker exited without result")
            failed = True
        elif status == "ok":
            if payload is not None:
                results.append(payload)
        else:
            print(f"Worker failed on {input_name}: {payload}")
            debug_logger.event("exec", "input_worker_failed",
                               input_file=str(input_file),
                               error=payload)
            failed = True

    # Keep a bounded set of forked workers active. This preserves the
    # existing hard per-input timeout behavior while actually using the
    # max_workers concurrency computed above.
    while True:
        while len(active) < max_workers:
            try:
                active.append(_start_worker(next(pending)))
            except StopIteration:
                break
        if not active:
            break

        wait_objs = []
        for task in active:
            wait_objs.append(task["conn"])
            wait_objs.append(task["process"].sentinel)
        ready = wait_for_process_message(wait_objs, timeout=0.2)
        now = time.monotonic()
        done = []

        for task in active:
            conn = task["conn"]
            p = task["process"]
            status = payload = None
            if conn in ready:
                try:
                    status, payload = conn.recv()
                except EOFError:
                    status, payload = None, None
                done.append((task, status, payload))
                continue
            if p.sentinel in ready:
                if conn.poll():
                    try:
                        status, payload = conn.recv()
                    except EOFError:
                        status, payload = None, None
                done.append((task, status, payload))
                continue
            if per_input_timeout is not None and now - task["started"] >= per_input_timeout:
                done.append((task, "timeout", None))

        for task, status, payload in done:
            if task in active:
                active.remove(task)
                _finish_worker(task, status, payload)

    for task in active:
        _finish_worker(task, "timeout", None)

    post_inspection = None
    publication_error = None
    if instrumented_run or failed or skipped:
        # Probe/assert injection changes program semantics but writes to the
        # same diagnostic output paths. Failed/timed-out workers likewise do
        # not own a complete generation. None of those outputs may certify the
        # unmodified package recipe.
        publication.abort()
    else:
        try:
            publication.commit()
        except TraceContractError as exc:
            publication_error = str(exc)
        else:
            post_inspection = publication.inspect()

    if skipped:
        print(f"Skipped {skipped} non-terminating input(s)")

    # A partial resume dispatches only pending inputs.  Include contract-
    # validated outputs in the aggregate so coverage.json still describes the
    # whole corpus rather than only this invocation's repaired subset.
    result_names = {result[0] for result in results}
    reusable_results = (
        inspection.valid_inputs
        if not args.force and inspection.state is not TraceState.STALE
        else ()
    )
    for filename in reusable_results:
        input_name = Path(filename).stem
        if input_name in result_names:
            continue
        explored_path = bundle.output_paths(filename)[1]
        with explored_path.open() as stream:
            explored = {line.strip() for line in stream if line.strip()}
        results.append(
            (input_name, compute_coverage(_SHARED_PROGRAM, explored), explored)
        )

    # Coverage summary
    if results:
        _print_coverage_summary(
            results, test_name, test_path, trace_bundle=bundle
        )

    # Coverage reuses explored-block paths from the selected generation, so
    # it is part of the protected read lifetime too. ExitStack also closes the
    # publication transaction on every exception above.
    publication_stack.close()

    if failed:
        print("Interpretation did not complete — run again to resume")
        raise SystemExit(1)
    if publication_error is not None:
        print(f"Interpretation did not produce a complete trace bundle — {publication_error}")
        raise SystemExit(1)
    if post_inspection is not None and not post_inspection.is_ready:
        print(
            "Interpretation did not produce a complete trace bundle — "
            f"{post_inspection.reason}"
        )
        raise SystemExit(1)

    print(f"Done. Engine={args.engine}")
    debug_logger.event("exec", "runner_end",
                       engine=args.engine,
                       inputs=len(input_files),
                       results=len(results),
                       failed=failed,
                       skipped=skipped)


def main():
    """Run one trace publication with failure-safe transaction cleanup."""
    with ExitStack() as publication_stack:
        return _main(publication_stack)


if __name__ == '__main__':
    main()
