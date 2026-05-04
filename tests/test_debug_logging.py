import json

import pytest

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.runner import run_native
from interpreter.utils.debug_log import DebugLogger
from interpreter.utils.support_matrix import support_matrix_summary


def _read_events(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_debug_logger_filters_and_sanitizes(tmp_path):
    logger = DebugLogger.from_options(
        tmp_path,
        "exec",
        run_id="test-run",
    ).bind(test_name="debug_test")

    logger.event("solver", "ignored", value=1)
    logger.event("exec", "kept", long_text="x" * 5000)

    events = _read_events(tmp_path / "debug.jsonl")
    assert len(events) == 1
    event = events[0]
    assert event["run_id"] == "test-run"
    assert event["schema_version"] == 1
    assert event["test_name"] == "debug_test"
    assert event["category"] == "exec"
    assert event["event"] == "kept"
    assert "truncated" in event["long_text"]


def test_debug_logger_write_failures_are_non_disruptive(tmp_path):
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("plain file")
    logger = DebugLogger.from_options(not_a_dir, "exec", run_id="bad-path")

    logger.event("exec", "cannot_write")


def test_support_matrix_summary_exposes_known_areas():
    summary = support_matrix_summary()

    assert "statements" in summary
    assert "expressions" in summary
    assert "execution_engines" in summary
    assert summary["statements"]["unsupported"] > 0


@pytest.mark.native
def test_run_native_result_reports_structured_assert(tmp_path, monkeypatch):
    pytest.importorskip("swoosh_interp")
    monkeypatch.chdir(tmp_path)
    program = parse_boogie(
        """
        procedure main();
        implementation {:entrypoint} main() {
        entry:
          assert false;
          return;
        }
        """
    )
    logger = DebugLogger.from_options(tmp_path / "logs", "exec", run_id="native-result")

    result = run_native(
        program,
        {},
        "debug_prog",
        "input_0",
        raw_log_path=tmp_path / "assert.raw.zst",
        no_trace=True,
        return_status=True,
        debug_logger=logger,
    )

    assert result["status"] == "assert_violation"
    assert result["violation_block"] == "entry"
    assert result["engine"] == "native"
    assert result["explored_blocks"] == {"entry"}

    events = _read_events(tmp_path / "logs" / "debug.jsonl")
    assert any(
        event["event"] == "engine_end"
        and event["status"] == "assert_violation"
        for event in events
    )
