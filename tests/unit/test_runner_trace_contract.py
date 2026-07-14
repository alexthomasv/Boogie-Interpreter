"""Focused tests for the runner side of the shared trace contract."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from interpreter import runner


@pytest.mark.unit
def test_selected_worker_replaces_output_pair_without_local_skip(
    monkeypatch, tmp_path
):
    trace_root = tmp_path / "traces"

    class _Layout:
        def trace_dir(self, name):
            return trace_root / name

    monkeypatch.setattr(runner, "current_layout", lambda: _Layout())
    monkeypatch.setattr(
        "interpreter.utils.input_parser.parse_input_file",
        lambda *_args, **_kwargs: SimpleNamespace(extra_data=None),
    )
    monkeypatch.setattr(runner, "compute_coverage", lambda *_args: {})

    calls = []

    def _run_native(*_args, raw_log_path, **_kwargs):
        calls.append(raw_log_path)
        raw_log_path.write_bytes(b"new-raw")
        return {"new-block"}

    monkeypatch.setattr(runner, "run_native", _run_native)

    input_file = tmp_path / "sample.input"
    input_file.write_text("input")
    trace_dir = trace_root / "demo"
    trace_dir.mkdir(parents=True)
    raw = trace_dir / "sample.trace.raw.zst"
    explored = trace_dir / "sample.explored_blocks.txt"
    raw.write_bytes(b"old-raw")
    explored.write_text("old-block\n")

    result = runner.process_single_input(
        input_file,
        test_name="demo",
        test_path=tmp_path / "demo.pkl",
        force=False,
        program=object(),
        field_sizes={},
    )

    assert calls == [raw]
    assert raw.read_bytes() == b"new-raw"
    assert explored.read_text() == "new-block\n"
    assert result == ("sample", {}, {"new-block"})


@pytest.mark.unit
def test_runner_cli_rejects_archived_python_engine():
    with pytest.raises(RuntimeError, match="archived"):
        runner._reject_legacy_engine("python")


@pytest.mark.unit
def test_native_preparation_uses_pinned_package_metadata_bytes(tmp_path):
    test_path = tmp_path / "demo_pkg" / "demo.pkl"
    outputs = {
        "demo_live_in.pkl": pickle.dumps({"main": {"head": {"x", "y"}}}),
        "demo_loops.pkl": pickle.dumps({"main": ["head"]}),
        "demo_loop_parents.pkl": pickle.dumps({"main": {"head": None}}),
        "demo_block_to_loop.pkl": pickle.dumps(
            {"main": {"body": "head"}}
        ),
    }

    assert runner._build_loop_header_live(
        test_path, package_outputs=outputs
    ) == {"head": ["x", "y"]}
    assert runner._build_loop_metadata(
        test_path, package_outputs=outputs
    ) == {
        "is_loop_header": ["head"],
        "block_innermost_header": {"body": "head"},
        "loop_parent_header": {},
    }
    assert runner._package_manifest_mode(
        test_path,
        package_manifest={"integer_encoding": True},
    ) == "int"
