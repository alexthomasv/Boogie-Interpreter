from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Mapping

import pytest

from interpreter.coverage_gen.evaluator import Evaluator
from interpreter.parser.boogie_parser import parse_boogie
from interpreter.runner import run_both, run_native, run_python_result
from interpreter.utils.inputs import Input, ProgramInputs


@contextmanager
def isolated_cwd(path: Path | None):
    if path is None:
        yield
        return
    path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def make_program(source: str):
    return parse_boogie(source)


def scalar_inputs(values: Mapping[str, int] | None = None, *,
                  private: set[str] | None = None,
                  extra_data: bytes | None = None) -> ProgramInputs:
    private = private or set()
    variables = {
        name: Input(name=name, private=name in private, value=value)
        for name, value in (values or {}).items()
    }
    return ProgramInputs(variables, extra_data=extra_data)


def havoc_inputs(values: Mapping[str, list[int]], *,
                 extra_data: bytes | None = None) -> ProgramInputs:
    return ProgramInputs(
        {
            name: Input(name=name, private=False, havoc_seq=list(seq))
            for name, seq in values.items()
        },
        extra_data=extra_data,
    )


def run_python_case(source: str, inputs: ProgramInputs | dict | None = None,
                    *, tmp_path: Path | None = None,
                    test_name: str = "case",
                    input_name: str = "input_0") -> dict:
    with isolated_cwd(tmp_path):
        return run_python_result(
            make_program(source),
            inputs or {},
            test_name,
            input_name,
            no_read_trace=True,
        )


def run_native_case(source: str, inputs: ProgramInputs | dict | None = None,
                    *, tmp_path: Path | None = None,
                    test_name: str = "case",
                    input_name: str = "input_0") -> dict:
    pytest.importorskip("swoosh_interp")
    with isolated_cwd(tmp_path):
        raw_log = Path("native.trace.raw.zst")
        return run_native(
            make_program(source),
            inputs or {},
            test_name,
            input_name,
            raw_log_path=raw_log,
            no_trace=True,
            log_read=False,
            return_status=True,
        )


def run_both_case(source: str, inputs: ProgramInputs | dict | None = None,
                  *, tmp_path: Path | None = None,
                  test_name: str = "case",
                  input_name: str = "input_0"):
    pytest.importorskip("swoosh_interp")
    with isolated_cwd(tmp_path):
        return run_both(
            make_program(source),
            inputs or {},
            test_name,
            input_name,
            no_read_trace=True,
        )


def concolic_candidates(source: str, inputs: ProgramInputs,
                        covered_blocks: set[str],
                        *, tmp_path: Path | None = None,
                        test_name: str = "symbolic_case",
                        input_name: str = "seed",
                        max_solver_queries: int = 16):
    pytest.importorskip("swoosh_interp")
    with isolated_cwd(tmp_path):
        program = make_program(source)
        evaluator = Evaluator(program, test_name, timeout=5, engine="native")
        return evaluator.concolic_suggest(
            inputs,
            input_name,
            covered_blocks,
            max_solver_queries=max_solver_queries,
        )


def symbolic_candidates(source: str, inputs: ProgramInputs,
                        covered_blocks: set[str],
                        *, tmp_path: Path | None = None,
                        test_name: str = "symbolic_case",
                        input_name: str = "seed",
                        max_solver_queries: int = 16,
                        max_states: int = 16):
    pytest.importorskip("swoosh_interp")
    with isolated_cwd(tmp_path):
        program = make_program(source)
        evaluator = Evaluator(program, test_name, timeout=5, engine="native")
        return evaluator.symbolic_explore(
            inputs,
            input_name,
            covered_blocks,
            max_solver_queries=max_solver_queries,
            max_states=max_states,
        )


def assert_ok(result: dict, *, blocks: set[str] | None = None):
    assert result["status"] == "ok"
    if blocks is not None:
        assert set(result["explored_blocks"]) == blocks


def assert_violation(result: dict, status: str, block: str):
    assert result["status"] == status
    assert result["violation_block"] == block

