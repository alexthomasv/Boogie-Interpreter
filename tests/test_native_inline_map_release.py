"""Native-inliner map frames release storage without changing program semantics."""

from pathlib import Path

import pytest

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.runner import prepare_native, run_native
from interpreter.tests.helpers.boogie_cases import isolated_cwd
from interpreter.utils.inputs import ProgramInputs
from interpreter.utils.integer_encoding import detect_semantics_mode
from passes.transform.inlining import AstInliner, mark_inline_procedures


pytestmark = pytest.mark.native


MAP_RETURN = """
procedure main();
implementation {:entrypoint} main() {
  var M: [ref]i8;
  var R: [ref]i8;
entry:
  M := $store.i8(M, 7, 11);
  call R := touch(M);
  assert $load.i8(R, 7) == 11;
  assert $load.i8(R, 8) == 22;
  assert $load.i8(M, 8) == 0;
  return;
}

procedure touch(I: [ref]i8) returns (O: [ref]i8);
implementation touch(I: [ref]i8) returns (O: [ref]i8) {
  var T: [ref]i8;
entry:
  T := I;
  T := $store.i8(T, 8, 22);
  O := T;
  return;
}
"""


LOOPED_MAP_RETURN = """
procedure main();
implementation {:entrypoint} main() {
  var M: [ref]i8;
  var R: [ref]i8;
  var i: int;
entry:
  M := $store.i8(M, 7, 11);
  i := 0;
  goto loop;
loop:
  goto body, done;
body:
  assume i < 4;
  call R := touch(M);
  i := i + 1;
  goto loop;
done:
  assume i >= 4;
  assert $load.i8(R, 7) == 11;
  assert $load.i8(R, 8) == 22;
  assert $load.i8(M, 8) == 0;
  return;
}

procedure touch(I: [ref]i8) returns (O: [ref]i8);
implementation touch(I: [ref]i8) returns (O: [ref]i8) {
  var T: [ref]i8;
entry:
  T := I;
  T := $store.i8(T, 8, 22);
  O := T;
  return;
}
"""


LOOPED_FRESH_MAP_FRAME = """
procedure main();
implementation {:entrypoint} main() {
  var M: [ref]i8;
  var R: [ref]i8;
  var i: int;
entry:
  M := $store.i8(M, 7, 11);
  i := 0;
  goto loop;
loop:
  goto body, done;
body:
  assume i < 3;
  call R := map_step(M);
  i := i + 1;
  goto loop;
done:
  assume i >= 3;
  assert $load.i8(R, 7) == 11;
  return;
}

procedure map_step(I: [ref]i8) returns (M.ret: [ref]i8);
implementation map_step(I: [ref]i8) returns (M.ret: [ref]i8) {
  var M.tmp: [ref]i8;
entry:
  assume $load.i8(M.ret, 7) == 0;
  assume $load.i8(M.tmp, 7) == 0;
  M.tmp := I;
  M.ret := M.tmp;
  return;
}
"""


SYNTHETIC_NAME_COLLISION = """
procedure main();
implementation {:entrypoint} main() {
  var M: [ref]i8;
  var R: [ref]i8;
  var inline$touch$0$T: [ref]i8;
entry:
  M := $store.i8(M, 7, 11);
  inline$touch$0$T := $store.i8(inline$touch$0$T, 9, 99);
  call R := touch(M);
  assert $load.i8(inline$touch$0$T, 9) == 99;
  return;
$inline_cont$1:
  return;
}

procedure touch(I: [ref]i8) returns (O: [ref]i8);
implementation touch(I: [ref]i8) returns (O: [ref]i8) {
  var T: [ref]i8;
entry:
  O := I;
  return;
}
"""


GENERATED_PREFIX_COLLISION = """
procedure main();
implementation {:entrypoint} main() {
entry:
  call a$1();
  call a();
  return;
}

procedure a$1();
implementation a$1() {
entry:
  return;
}

procedure a();
implementation a() {
entry:
  return;
}
"""


def _compile(source: str, package: Path | None = None):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    program = parse_boogie(source)
    mark_inline_procedures(program)
    mode = detect_semantics_mode(program)
    if package is None:
        return swoosh_interp.inline_lower(program, mode)
    swoosh_interp.inline_lower_to_file(program, str(package), {}, mode=mode)
    return swoosh_interp.load_compiled(str(package))


def _run(source: str, compiled, tmp_path: Path) -> dict:
    proof_program = AstInliner().run(parse_boogie(source))
    prepared = prepare_native(proof_program, compiled=compiled)
    with isolated_cwd(tmp_path):
        return run_native(
            proof_program,
            ProgramInputs({}),
            "input_0",
            "native.trace.raw.zst",
            no_trace=True,
            log_read=False,
            return_status=True,
            prepared=prepared,
        )


def _assert_only_frame_maps_are_empty(memory_summary: dict) -> None:
    assert memory_summary["M"]["entries"] == 1
    assert memory_summary["R"]["entries"] == 2
    frame_maps = {
        name: summary
        for name, summary in memory_summary.items()
        if name.startswith("inline$")
    }
    assert frame_maps
    assert all(summary["entries"] == 0 for summary in frame_maps.values())


def test_map_return_survives_release_and_release_owns_no_verifier_pc(tmp_path):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    compiled = _compile(MAP_RETURN, tmp_path / "map_return.swcp")

    dump = swoosh_interp.dump_block(compiled, "$inline_cont$0")
    # The call-entry frame havoc owns one proof/native PC, so the continuation
    # return copy starts at 9. ReleaseMaps remains zero-PC maintenance.
    assert "pc=9 Assign1" in dump
    assert "internal ReleaseMaps" in dump
    assert "pc=10 Assert" in dump

    result = _run(MAP_RETURN, compiled, tmp_path / "run")
    assert result["status"] == "ok", result
    _assert_only_frame_maps_are_empty(result["memory_summary"])


def test_loop_reuses_static_map_slots_after_each_release(tmp_path):
    compiled = _compile(LOOPED_MAP_RETURN)
    result = _run(LOOPED_MAP_RETURN, compiled, tmp_path)

    assert result["status"] == "ok", result
    assert result["memory_map_count"] == compiled.mem_map_count
    _assert_only_frame_maps_are_empty(result["memory_summary"])


def test_loop_reentry_havocs_map_return_and_local_in_both_inliners(tmp_path):
    proof_program = AstInliner().run(parse_boogie(LOOPED_FRESH_MAP_FRAME))
    native_compiled = _compile(LOOPED_FRESH_MAP_FRAME)

    prepared_ast = prepare_native(proof_program)
    prepared_native = prepare_native(proof_program, compiled=native_compiled)
    results = []
    for name, prepared in [("ast", prepared_ast), ("native", prepared_native)]:
        with isolated_cwd(tmp_path / name):
            results.append(
                run_native(
                    proof_program,
                    ProgramInputs({}),
                    "input_0",
                    "native.trace.raw.zst",
                    no_trace=True,
                    log_read=False,
                    return_status=True,
                    prepared=prepared,
                )
            )

    ast_result, native_result = results
    assert ast_result["status"] == "ok", ast_result
    assert native_result["status"] == "ok", native_result
    for name in ["M", "R"]:
        assert ast_result["memory_summary"][name] == native_result["memory_summary"][name]

    # The native pre-call block owns one real PC for the parallel reset. This
    # mirrors the proof AST's HavocStatement and occurs before parameter binds.
    dump = pytest.importorskip("swoosh_interp").dump_block(native_compiled, "body")
    assert "Havoc { vars:" in dump
    assert dump.index("Havoc { vars:") < dump.index("Assign1")


def test_release_uses_collision_free_frame_and_continuation_names(tmp_path):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    compiled = _compile(SYNTHETIC_NAME_COLLISION)

    # Frame candidate 0 collides with the caller's local, and continuation 1
    # collides with a source label. Match the proof inliner by advancing the
    # frame number and suffixing the continuation label.
    dump = swoosh_interp.dump_block(compiled, "$inline_cont$1$")
    assert "internal ReleaseMaps" in dump

    result = _run(SYNTHETIC_NAME_COLLISION, compiled, tmp_path)
    assert result["status"] == "ok", result
    summary = result["memory_summary"]
    assert summary["inline$touch$0$T"]["entries"] == 1
    assert summary["inline$touch$1$T"]["entries"] == 0


def test_generated_name_can_reserve_a_later_frame_prefix(tmp_path):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    compiled = _compile(GENERATED_PREFIX_COLLISION)

    # Expanding `a$1` as instance 0 creates names beginning `inline$a$1$`.
    # Therefore instance 1 for callee `a` collides and must advance to 2,
    # matching AstInliner's starts-with rule even for `$N$` in a callee name.
    dump = swoosh_interp.dump_block(compiled, "$inline_cont$2")
    assert "block $inline_cont$2" in dump

    result = _run(GENERATED_PREFIX_COLLISION, compiled, tmp_path)
    assert result["status"] == "ok", result


def test_file_build_lookup_eviction_preserves_exact_lowered_program(tmp_path):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    retained = _compile(GENERATED_PREFIX_COLLISION)
    evicted = _compile(
        GENERATED_PREFIX_COLLISION,
        tmp_path / "generated-prefix-collision.swcp",
    )

    assert swoosh_interp.get_var_names(evicted) == swoosh_interp.get_var_names(
        retained
    )
    assert swoosh_interp.dump_block(evicted, None) == swoosh_interp.dump_block(
        retained, None
    )

    named = _compile(MAP_RETURN)
    names = swoosh_interp.get_var_names(named)
    assert names
    assert swoosh_interp.get_var_name(named, 0) == names[0]
    assert swoosh_interp.get_block_name(evicted, 0) in swoosh_interp.dump_block(
        evicted, None
    )
    with pytest.raises(IndexError, match="variable id"):
        swoosh_interp.get_var_name(named, named.num_vars)
    with pytest.raises(IndexError, match="block id"):
        swoosh_interp.get_block_name(evicted, evicted.num_blocks)
