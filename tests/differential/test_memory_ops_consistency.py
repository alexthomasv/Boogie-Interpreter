import textwrap

import pytest

from interpreter.tests.helpers.boogie_cases import run_native_case


pytestmark = [pytest.mark.differential, pytest.mark.native]


def _compare_memory(source: str, tmp_path, test_name: str):
    native = run_native_case(source, tmp_path=tmp_path, test_name=f"{test_name}_native")

    assert native["status"] == "ok"
    return native["memory_summary"]


def _memset_program(assume_expr: str) -> str:
    return textwrap.dedent(
        f"""
        procedure main();
        implementation {{:entrypoint}} main() {{
          var inline$$memset.i8.cross_product$0$$M: [ref]i8;
          var inline$$memset.i8.cross_product$0$$M.ret: [ref]i8;
          var inline$$memset.i8.cross_product$0$$dst: ref;
          var inline$$memset.i8.cross_product$0$$len: ref;
          var inline$$memset.i8.cross_product$0$$val: i8;
        entry:
          inline$$memset.i8.cross_product$0$$M := $store.i8(inline$$memset.i8.cross_product$0$$M, 99, 11);
          inline$$memset.i8.cross_product$0$$M := $store.i8(inline$$memset.i8.cross_product$0$$M, 103, 22);
          inline$$memset.i8.cross_product$0$$dst := 100;
          inline$$memset.i8.cross_product$0$$len := 3;
          inline$$memset.i8.cross_product$0$$val := 171;
          {assume_expr}
          return;
        }}
        """
    )


def _memcpy_program(assume_exprs: list[str]) -> str:
    assumes = "\n  ".join(assume_exprs)
    return textwrap.dedent(
        f"""
        procedure main();
        implementation {{:entrypoint}} main() {{
          var inline$$memcpy.i8.cross_product$0$$M.dst: [ref]i8;
          var inline$$memcpy.i8.cross_product$0$$M.src: [ref]i8;
          var inline$$memcpy.i8.cross_product$0$$M.ret: [ref]i8;
          var inline$$memcpy.i8.cross_product$0$$dst: ref;
          var inline$$memcpy.i8.cross_product$0$$src: ref;
          var inline$$memcpy.i8.cross_product$0$$len: ref;
        entry:
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 99, 11);
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 103, 22);
          inline$$memcpy.i8.cross_product$0$$M.src := $store.i8(inline$$memcpy.i8.cross_product$0$$M.src, 200, 41);
          inline$$memcpy.i8.cross_product$0$$M.src := $store.i8(inline$$memcpy.i8.cross_product$0$$M.src, 201, 42);
          inline$$memcpy.i8.cross_product$0$$M.src := $store.i8(inline$$memcpy.i8.cross_product$0$$M.src, 202, 43);
          inline$$memcpy.i8.cross_product$0$$dst := 100;
          inline$$memcpy.i8.cross_product$0$$src := 200;
          inline$$memcpy.i8.cross_product$0$$len := 3;
          {assumes}
          return;
        }}
        """
    )


MEMSET_FILL_STANDARD = (
    r"assume (forall x: ref :: ($sle.ref.bool(inline$$memset.i8.cross_product$0$$dst, x) "
    r"&& ($slt.ref.bool(x, $add.ref(inline$$memset.i8.cross_product$0$$dst, "
    r"inline$$memset.i8.cross_product$0$$len)) "
    r"==> (inline$$memset.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memset.i8.cross_product$0$$val))));"
)

MEMSET_FILL_OUTER_IMPLIES = (
    r"assume (forall x: ref :: (($sle.ref.bool(inline$$memset.i8.cross_product$0$$dst, x) "
    r"&& $slt.ref.bool(x, $add.ref(inline$$memset.i8.cross_product$0$$dst, "
    r"inline$$memset.i8.cross_product$0$$len))) "
    r"==> (inline$$memset.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memset.i8.cross_product$0$$val)));"
)

MEMSET_PRESERVE_BELOW = (
    r"assume (forall x: ref :: ($slt.ref.bool(x, inline$$memset.i8.cross_product$0$$dst) "
    r"==> (inline$$memset.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memset.i8.cross_product$0$$M[x])));"
)

MEMSET_PRESERVE_ABOVE = (
    r"assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memset.i8.cross_product$0$$dst, "
    r"inline$$memset.i8.cross_product$0$$len), x) "
    r"==> (inline$$memset.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memset.i8.cross_product$0$$M[x])));"
)

MEMCPY_FILL_STANDARD = (
    r"assume (forall x: ref :: ($sle.ref.bool(inline$$memcpy.i8.cross_product$0$$dst, x) "
    r"&& ($slt.ref.bool(x, $add.ref(inline$$memcpy.i8.cross_product$0$$dst, "
    r"inline$$memcpy.i8.cross_product$0$$len)) "
    r"==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memcpy.i8.cross_product$0$$M.src[$add.ref($sub.ref("
    r"inline$$memcpy.i8.cross_product$0$$src, inline$$memcpy.i8.cross_product$0$$dst), x)]))));"
)

MEMCPY_FILL_OUTER_IMPLIES = (
    r"assume (forall x: ref :: (($sle.ref.bool(inline$$memcpy.i8.cross_product$0$$dst, x) "
    r"&& $slt.ref.bool(x, $add.ref(inline$$memcpy.i8.cross_product$0$$dst, "
    r"inline$$memcpy.i8.cross_product$0$$len))) "
    r"==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memcpy.i8.cross_product$0$$M.src[$add.ref($sub.ref("
    r"inline$$memcpy.i8.cross_product$0$$src, inline$$memcpy.i8.cross_product$0$$dst), x)])));"
)

MEMCPY_PRESERVE_BELOW = (
    r"assume (forall x: ref :: ($slt.ref.bool(x, inline$$memcpy.i8.cross_product$0$$dst) "
    r"==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memcpy.i8.cross_product$0$$M.dst[x])));"
)

MEMCPY_PRESERVE_ABOVE = (
    r"assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memcpy.i8.cross_product$0$$dst, "
    r"inline$$memcpy.i8.cross_product$0$$len), x) "
    r"==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == "
    r"inline$$memcpy.i8.cross_product$0$$M.dst[x])));"
)


@pytest.mark.parametrize("assume_expr", [
    MEMSET_FILL_STANDARD,
    MEMSET_FILL_OUTER_IMPLIES,
    MEMSET_PRESERVE_BELOW,
    MEMSET_PRESERVE_ABOVE,
])
def test_native_memset_quantified_assumes_update_memory(tmp_path, assume_expr):
    summary = _compare_memory(
        _memset_program(assume_expr),
        tmp_path,
        "memset_quantified",
    )

    assert "inline$$memset.i8.cross_product$0$$M.ret" in summary


@pytest.mark.parametrize("assume_expr", [
    MEMCPY_FILL_STANDARD,
    MEMCPY_FILL_OUTER_IMPLIES,
    MEMCPY_PRESERVE_BELOW,
    MEMCPY_PRESERVE_ABOVE,
])
def test_native_memcpy_quantified_assumes_update_memory(tmp_path, assume_expr):
    summary = _compare_memory(
        _memcpy_program([assume_expr]),
        tmp_path,
        "memcpy_quantified",
    )

    assert "inline$$memcpy.i8.cross_product$0$$M.ret" in summary


def test_native_memcpy_same_map_snapshot_behavior(tmp_path):
    source = textwrap.dedent(
        f"""
        procedure main();
        implementation {{:entrypoint}} main() {{
          var inline$$memcpy.i8.cross_product$0$$M.dst: [ref]i8;
          var inline$$memcpy.i8.cross_product$0$$M.src: [ref]i8;
          var inline$$memcpy.i8.cross_product$0$$M.ret: [ref]i8;
          var inline$$memcpy.i8.cross_product$0$$dst: ref;
          var inline$$memcpy.i8.cross_product$0$$src: ref;
          var inline$$memcpy.i8.cross_product$0$$len: ref;
        entry:
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 99, 9);
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 100, 1);
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 101, 2);
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 102, 3);
          inline$$memcpy.i8.cross_product$0$$M.dst := $store.i8(inline$$memcpy.i8.cross_product$0$$M.dst, 105, 5);
          inline$$memcpy.i8.cross_product$0$$M.src := inline$$memcpy.i8.cross_product$0$$M.dst;
          inline$$memcpy.i8.cross_product$0$$dst := 102;
          inline$$memcpy.i8.cross_product$0$$src := 100;
          inline$$memcpy.i8.cross_product$0$$len := 3;
          {MEMCPY_FILL_STANDARD}
          {MEMCPY_PRESERVE_BELOW}
          {MEMCPY_PRESERVE_ABOVE}
          return;
        }}
        """
    )

    summary = _compare_memory(source, tmp_path, "memcpy_same_map")
    ret = summary["inline$$memcpy.i8.cross_product$0$$M.ret"]
    assert ret["entries"] == 7
    assert ret["min_addr"] == 99
    assert ret["max_addr"] == 105


def test_native_llvm_memmove_call_updates_byte_maps_with_overlap(tmp_path):
    source = textwrap.dedent(
        """
        procedure llvm.memmove.p0i8.p0i8.i64.cross_product(
          dst: ref, dst_shadow: ref, src: ref, src_shadow: ref,
          len: i64, len_shadow: i64, volatile: i1, volatile_shadow: i1);

        procedure main();
        implementation {:entrypoint} main() {
          var $M.0: [ref]i8;
          var $M.0.shadow: [ref]i8;
        entry:
          $M.0 := $store.i8($M.0, 100, 1);
          $M.0 := $store.i8($M.0, 101, 2);
          $M.0 := $store.i8($M.0, 102, 3);
          $M.0 := $store.i8($M.0, 103, 9);
          $M.0.shadow := $store.i8($M.0.shadow, 200, 11);
          $M.0.shadow := $store.i8($M.0.shadow, 201, 12);
          $M.0.shadow := $store.i8($M.0.shadow, 202, 13);
          $M.0.shadow := $store.i8($M.0.shadow, 203, 19);
          call llvm.memmove.p0i8.p0i8.i64.cross_product(102, 202, 100, 200, 3, 3, 0, 0);
          return;
        }
        """
    )

    summary = _compare_memory(source, tmp_path, "llvm_memmove")
    assert summary["$M.0"]["entries"] == 5
    assert summary["$M.0"]["min_addr"] == 100
    assert summary["$M.0"]["max_addr"] == 104
    assert summary["$M.0.shadow"]["entries"] == 5
    assert summary["$M.0.shadow"]["min_addr"] == 200
    assert summary["$M.0.shadow"]["max_addr"] == 204
