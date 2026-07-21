"""SMACK native-BV spelling lowers to the existing BV opcode semantics."""

import pytest

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.tests.helpers.boogie_cases import run_native_case


pytestmark = pytest.mark.native


PROGRAM = """
procedure main() returns ($result: bv64);

implementation {:entrypoint} main() returns ($result: bv64) {
  var $M: [ref]bv8;
  var $wrapped: bv32;
  var $signed: bv32;
entry:
  $wrapped := $add.bv32(4294967295bv32, 2bv32);
  assert $wrapped == 1bv32;
  $signed := $sext.bv8.bv32(255bv8);
  assert $signed == 4294967295bv32;

  $M := $store.bytes.bv8($M, 16, 511bv8);
  assert $load.bytes.bv8($M, 16) == 255bv8;
  $M := $store.bytes.bv32($M, 24, 287454020bv32);
  assert $load.bytes.bv32($M, 24) == 287454020bv32;
  assert $load.bytes.bv8($M, 24) == 68bv8;
  assert $load.bytes.bv8($M, 27) == 17bv8;

  $M := $store.bytes.ref($M, 32, 72623859790382856);
  assert $load.bytes.ref($M, 32) == 72623859790382856;
  $result := $bv2int.64($zext.bv32.bv64($wrapped));
  return;
}
"""


@pytest.mark.parametrize("lowering", ["lower", "inline_lower"])
def test_bv_intrinsic_aliases_lower_in_both_native_paths(lowering):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    program = parse_boogie(PROGRAM)

    if lowering == "lower":
        compiled = swoosh_interp.lower(program, mode="bv")
    else:
        compiled = swoosh_interp.inline_lower(program, "bv")

    assert compiled.mode == "bv"


def test_bv_intrinsic_aliases_execute_with_bv_and_byte_memory_semantics(tmp_path):
    result = run_native_case(
        PROGRAM,
        tmp_path=tmp_path,
        test_name="native_bv_intrinsics",
        return_scalar_summary=True,
    )

    assert result["status"] == "ok", result
    assert result["final_scalars"]["$result"] == 1


def test_streamed_swcp_round_trip(tmp_path):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    program = parse_boogie(PROGRAM)
    package = tmp_path / "native_bv_intrinsics.swcp"

    swoosh_interp.inline_lower_to_file(
        program, str(package), {}, mode="bv"
    )
    compiled = swoosh_interp.load_compiled(str(package))

    assert compiled.mode == "bv"
    assert "block " in swoosh_interp.dump_block(compiled)


def test_native_lowering_rejects_bitvector_literals_wider_than_vm(tmp_path):
    source = PROGRAM.replace("4294967295bv32", "1bv65", 1)

    with pytest.raises(ValueError, match=r"width 1\.\.=64, got bv65"):
        run_native_case(source, tmp_path=tmp_path, test_name="native_bv65")
