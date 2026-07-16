"""Mirror pin: the native execution status vocabulary is MINTED in Rust
(native/src/lib.rs stamps the strings on the PyO3 result dict); the Python
side reads it through runner.NATIVE_STATUSES / NativeResult. This test pins
the hand-mirrored tuple against the Rust source text — drift here means the
Python reader silently stops recognizing a status and misclassifies a
violation as a clean run.
"""
import re
from pathlib import Path

import pytest

from interpreter.runner import NATIVE_STATUSES, NativeResult

RUST_SRC_DIR = Path(__file__).resolve().parents[1] / "native" / "src"


def test_every_python_status_is_minted_in_rust():
    rust = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in RUST_SRC_DIR.glob("*.rs"))
    for status in NATIVE_STATUSES:
        assert re.search(rf'"{status}"', rust), (
            f"status {status!r} not found in native/src/*.rs")


def test_from_dict_reads_the_wire_keys():
    raw = {
        "status": "assume_violation",
        "violation_pc": 2938,
        "violation_block": "$bb7",
        "invalid_detail": "($i3 >= 0)  [where $i3=-1]",
        "invalid_reason": "assume",
        "explored_blocks": [],
    }
    native = NativeResult.from_dict(raw)
    assert native.status == "assume_violation"
    assert native.violation_pc == 2938
    assert native.violation_block == "$bb7"
    assert native.invalid_detail.startswith("($i3 >= 0)")
    assert native.raw is raw
    with pytest.raises(KeyError, match="status"):
        NativeResult.from_dict({"explored_blocks": []})
