import json

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.python.Environment import Environment
from interpreter.runner import run_python_result
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


def test_environment_logs_uninitialized_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    logger = DebugLogger.from_options(log_dir, "exec", run_id="env-test")
    env = Environment("debug_test", "input_0", debug_logger=logger)

    assert env.get_var("$missing") == 0

    events = _read_events(log_dir / "debug.jsonl")
    assert any(
        event["event"] == "uninitialized_var_defaulted"
        and event["var"] == "$missing"
        for event in events
    )


def test_support_matrix_summary_exposes_known_areas():
    summary = support_matrix_summary()

    assert "statements" in summary
    assert "expressions" in summary
    assert "execution_engines" in summary
    assert summary["statements"]["unsupported"] > 0


def test_run_python_result_reports_structured_assert(tmp_path, monkeypatch):
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
    logger = DebugLogger.from_options(tmp_path / "logs", "exec", run_id="py-result")

    result = run_python_result(
        program,
        {},
        "debug_prog",
        "input_0",
        debug_logger=logger,
    )

    assert result["status"] == "assert_violation"
    assert result["violation_block"] == "entry"
    assert result["engine"] == "python"
    assert result["explored_blocks"] == {"entry"}

    events = _read_events(tmp_path / "logs" / "debug.jsonl")
    assert any(
        event["event"] == "engine_end"
        and event["status"] == "assert_violation"
        for event in events
    )
