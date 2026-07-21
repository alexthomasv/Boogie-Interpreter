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

    monkeypatch.setenv("SWOOSH_OUT_DIR", str(tmp_path))
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
        program=object(),
        field_sizes={},
    )

    assert calls == [raw]
    assert raw.read_bytes() == b"new-raw"
    assert explored.read_text() == "new-block\n"
    assert result == ("sample", {}, {"new-block"})


@pytest.mark.unit
def test_diagnostic_resource_controls_reach_native_execution(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SWOOSH_OUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        "interpreter.utils.input_parser.parse_input_file",
        lambda *_args, **_kwargs: SimpleNamespace(extra_data=None),
    )
    monkeypatch.setattr(runner, "compute_coverage", lambda *_args: {})

    observed = {}

    def _run_native(*_args, **kwargs):
        observed.update(kwargs)
        return {"entry"}

    monkeypatch.setattr(runner, "run_native", _run_native)
    input_file = tmp_path / "diagnostic.input"
    input_file.write_text("input")

    result = runner.process_single_input(
        input_file,
        test_name="demo",
        test_path=tmp_path / "demo.pkl",
        no_trace=True,
        max_steps=123,
        program=object(),
        field_sizes={},
    )

    assert observed["no_trace"] is True
    assert observed["max_steps"] == 123
    assert result == ("diagnostic", {}, {"entry"})


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


@pytest.mark.unit
@pytest.mark.parametrize("manifest", [{}, {"integer_encoding": None}])
def test_package_manifest_requires_current_semantics_mode(tmp_path, manifest):
    with pytest.raises((TypeError, ValueError), match="integer_encoding"):
        runner._package_manifest_mode(
            tmp_path / "demo.pkl", package_manifest=manifest)
