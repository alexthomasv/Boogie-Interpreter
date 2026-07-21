"""Native lowering accepts Boogie's explicit bit-vector map types."""

import pytest

from interpreter.parser.boogie_parser import parse_boogie


PROGRAM = """
var $M: [bv64]bv8;

procedure main();

implementation main()
{
entry:
  return;
}
"""


@pytest.mark.native
@pytest.mark.parametrize("lowering", ["lower", "inline_lower"])
def test_native_lowering_accepts_bv64_to_bv8_map(lowering):
    swoosh_interp = pytest.importorskip("swoosh_interp")
    program = parse_boogie(PROGRAM)

    if lowering == "lower":
        compiled = swoosh_interp.lower(program, mode="bv")
    else:
        compiled = swoosh_interp.inline_lower(program, "bv")

    assert compiled.mode == "bv"
