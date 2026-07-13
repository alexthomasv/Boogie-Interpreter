from types import SimpleNamespace

import pytest

from interpreter import runner
from interpreter.errors import AssertViolation
from interpreter.parser.boogie_parser import parse_boogie, parse_expr
from interpreter.parser.declaration import ImplementationDeclaration


def _program():
    return parse_boogie(
        """
        var x: int;
        procedure main();
        implementation {:entrypoint} main() {
        entry:
          x := 0;
          goto goal;
        goal:
          return;
        }
        """
    )


def _implementation(program):
    return next(
        declaration for declaration in program.declarations
        if isinstance(declaration, ImplementationDeclaration)
        and declaration.body
    )


def test_block_probe_validates_label_without_injecting_statement():
    program = _program()
    impl = _implementation(program)
    before = {
        block.name: tuple(block.statements) for block in impl.body.blocks
    }

    metadata = runner.inject_asserts(program, [], ["goal"])

    after = {
        block.name: tuple(block.statements) for block in impl.body.blocks
    }
    assert metadata == {}
    assert after == before

    with pytest.raises(ValueError, match="no block labeled 'missing'"):
        runner.inject_asserts(_program(), [], ["missing"])


def test_text_injection_remains_an_ordinary_predicate():
    metadata = runner.inject_asserts(_program(), [(1, "x == 0")], [])

    assert len(metadata) == 1
    assert next(iter(metadata.values())) == {
        "kind": "predicate",
        "expr": "x == 0",
        "block": "entry",
        "requested_pc": 1,
    }


@pytest.mark.parametrize(
    ("row", "expected_kind"),
    [
        (lambda ast: (1, ast), "predicate"),
        (lambda ast: (1, ast, "predicate"), "predicate"),
        (lambda ast: (1, ast, "carrier_guard"), "carrier_guard"),
    ],
)
def test_ast_injection_rows_preserve_normalized_kind(row, expected_kind):
    program = _program()
    expression = parse_expr("x == 0")

    metadata = runner.inject_asserts(
        program, [], [], ast_specs=[row(expression)])

    assert len(metadata) == 1
    injected = next(iter(metadata.values()))
    assert injected == {
        "kind": expected_kind,
        "expr": repr(expression),
        "block": "entry",
        "requested_pc": 1,
    }


@pytest.mark.parametrize(
    "row",
    [
        (1,),
        (1, object(), "block_probe"),
        (1, object(), ""),
    ],
)
def test_ast_injection_rows_reject_invalid_shape_or_kind(row):
    with pytest.raises(ValueError, match="inject-assert-ast: row 0"):
        runner._normalize_inject_assert_ast_specs([row])


def _prepare_process_test(monkeypatch, tmp_path, *, explored=None,
                          violation=None):
    input_file = tmp_path / "sample.input"
    input_file.write_text("")
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

    def _run_native(*_args, **_kwargs):
        if violation is not None:
            raise violation
        return set(explored or ())

    monkeypatch.setattr(runner, "run_native", _run_native)
    return input_file


def test_successful_run_reports_all_passive_probes_and_typed_survival(
        monkeypatch, tmp_path, capsys):
    input_file = _prepare_process_test(
        monkeypatch, tmp_path, explored={"entry", "goal"})
    monkeypatch.setattr(runner, "_PROBE_BLOCK_SPECS", ["goal", "dead"])
    monkeypatch.setattr(
        runner,
        "_INJECTED_ASSERTS",
        {
            99: {
                "kind": "carrier_guard",
                "expr": "x == 0",
                "block": "entry",
                "requested_pc": 7,
            }
        },
    )

    result = runner.process_single_input(
        input_file,
        test_name="sample_program",
        test_path=tmp_path / "sample_program.pkl",
        force=True,
        program=object(),
        field_sizes={},
    )

    assert result == ("sample", {}, {"entry", "goal"})
    lines = capsys.readouterr().out.splitlines()
    assert "[BLOCK_REACHED] input=sample block='goal'" in lines
    assert "[BLOCK_NOT_REACHED] input=sample block='dead'" in lines
    assert (
        "[INJECTED_ASSERT_SURVIVED] input=sample pc=7 "
        "kind=carrier_guard block='entry' block_visited=true"
    ) in lines


def test_early_termination_emits_no_block_status_and_logs_injection_kind(
        monkeypatch, tmp_path, capsys):
    violation = AssertViolation(None, 23, "entry", "x > 0")
    input_file = _prepare_process_test(
        monkeypatch, tmp_path, violation=violation)
    monkeypatch.setattr(runner, "_PROBE_BLOCK_SPECS", ["entry"])
    monkeypatch.setattr(
        runner,
        "_INJECTED_ASSERTS",
        {
            23: {
                "kind": "predicate",
                "expr": "x > 0",
                "block": "entry",
                "requested_pc": 7,
            }
        },
    )

    result = runner.process_single_input(
        input_file,
        test_name="sample_program",
        test_path=tmp_path / "sample_program.pkl",
        force=True,
        program=object(),
        field_sizes={},
    )

    assert result == ("sample", None, set())
    output = capsys.readouterr().out
    assert "[BLOCK_REACHED]" not in output
    assert "[BLOCK_NOT_REACHED]" not in output
    assert (
        "[INJECTED_ASSERT_VIOLATION] input=sample pc=7 "
        "kind=predicate block='entry' expr=\"x > 0\""
    ) in output
