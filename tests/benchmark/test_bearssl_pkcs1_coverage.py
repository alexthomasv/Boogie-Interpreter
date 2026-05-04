from __future__ import annotations

import json
import os
import pickle
import shutil
import statistics
import time
from pathlib import Path

import pytest

from interpreter.coverage_gen.driver import generate_coverage_inputs
from interpreter.runner import compute_coverage, run_native
from interpreter.utils.input_parser import get_bpl_field_sizes, parse_input_file


pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.requires_compiled_package,
    pytest.mark.native,
]


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
BASELINES = json.loads((Path(__file__).with_name("pkcs1_baselines.json")).read_text())


def _asset_paths(test_name: str) -> tuple[Path, Path, Path]:
    pkg_dir = PROJECT_ROOT / "test_packages" / f"{test_name}_pkg"
    pkl_path = pkg_dir / f"{test_name}.pkl"
    input_dir = PROJECT_ROOT / "test_input" / test_name
    if not pkl_path.exists() or not input_dir.exists():
        pytest.skip(f"missing benchmark assets for {test_name}")
    return pkg_dir, pkl_path, input_dir


def _load_program(test_name: str):
    _pkg_dir, pkl_path, _input_dir = _asset_paths(test_name)
    with pkl_path.open("rb") as fh:
        return pickle.load(fh)


def _copy_seed_subset(test_name: str, tmp_path: Path, seed_names: list[str]) -> Path:
    _pkg_dir, _pkl_path, input_dir = _asset_paths(test_name)
    seed_dir = tmp_path / f"{test_name}_seeds"
    seed_dir.mkdir()
    for seed_name in seed_names:
        src = input_dir / seed_name
        if not src.exists():
            pytest.skip(f"missing seed {src}")
        shutil.copy2(src, seed_dir / seed_name)
    return seed_dir


def _package_hash(test_name: str) -> str | None:
    pkg_dir, _pkl_path, _input_dir = _asset_paths(test_name)
    hash_path = pkg_dir / f"{test_name}.hash"
    if hash_path.exists():
        return hash_path.read_text().strip()
    return None


def _baseline(test_name: str) -> dict:
    data = BASELINES.get(test_name, {})
    if data.get("hash") != _package_hash(test_name):
        pytest.skip(f"baseline hash mismatch for {test_name}")
    return data


def _assert_coverage_metrics(report: dict) -> None:
    assert report["engine_used"] == "native"
    assert not report["native_fallback"], report.get("native_fallback_reason")
    assert report["seed_inputs"] >= 1
    assert report["candidates_evaluated"] >= report["seed_inputs"]
    assert report["step_limits"] >= 0

    coverage = report["coverage"]
    initial = report["coverage_initial"]
    rates = report["coverage_per_time"]
    assert coverage["blocks_covered"] >= initial["blocks_covered"]
    assert rates["final_reachable_blocks"] >= rates["initial_reachable_blocks"]
    assert rates["candidates_per_sec"] >= 0
    assert rates["useful_inputs_per_sec"] >= 0

    for phase in report["phase_timings"]:
        if "blocks_added" in phase:
            assert "blocks_per_sec" in phase
        if "new_inputs" in phase:
            assert "new_inputs_per_sec" in phase


def _run_generation(test_name: str, tmp_path: Path, seed_names: list[str], **overrides) -> dict:
    pytest.importorskip("swoosh_interp")
    program = _load_program(test_name)
    seed_dir = _copy_seed_subset(test_name, tmp_path, seed_names)
    out_dir = tmp_path / f"{test_name}_coverage"
    return generate_coverage_inputs(
        program,
        test_name,
        seed_dir,
        out_dir,
        engine="native",
        profile=overrides.pop("profile", "fast-traces"),
        path_engine=overrides.pop("path_engine", "fuzz"),
        rng_seed=0,
        jobs=1,
        timeout=overrides.pop("timeout", 10),
        progress_interval=1000,
        **overrides,
    )


def test_pkcs1_i15_fast_coverage_smoke(tmp_path):
    report = _run_generation(
        "bearssl_test_pkcs1_i15",
        tmp_path,
        ["fuzz_0001.input"],
        deterministic_budget=4,
        rounds=1,
        iters=8,
        loop_bound=4,
        havoc_bound=4,
        max_path_depth=128,
        max_solver_queries=8,
        max_symbolic_states=4,
        path_seed_limit=1,
    )
    _assert_coverage_metrics(report)


@pytest.mark.slow
def test_pkcs1_i31_fast_coverage_smoke(tmp_path):
    report = _run_generation(
        "bearssl_test_pkcs1_i31",
        tmp_path,
        ["test_01b_standard_2048.input"],
        deterministic_budget=1,
        rounds=1,
        iters=1,
        loop_bound=2,
        havoc_bound=2,
        max_path_depth=64,
        max_solver_queries=2,
        max_symbolic_states=1,
        path_seed_limit=1,
        timeout=30,
    )
    _assert_coverage_metrics(report)


@pytest.mark.slow
@pytest.mark.exhaustive
def test_pkcs1_i15_intensive_coverage_floor(tmp_path):
    baseline = _baseline("bearssl_test_pkcs1_i15")
    report = _run_generation(
        "bearssl_test_pkcs1_i15",
        tmp_path,
        ["fuzz_0001.input"],
        profile="max-coverage",
        deterministic_budget=int(os.environ.get("SWOOSH_PKCS1_DETERMINISTIC", "128")),
        rounds=int(os.environ.get("SWOOSH_PKCS1_ROUNDS", "2")),
        iters=int(os.environ.get("SWOOSH_PKCS1_ITERS", "512")),
        loop_bound=int(os.environ.get("SWOOSH_PKCS1_LOOP_BOUND", "8")),
        havoc_bound=int(os.environ.get("SWOOSH_PKCS1_HAVOC_BOUND", "8")),
        max_path_depth=int(os.environ.get("SWOOSH_PKCS1_MAX_DEPTH", "512")),
        max_solver_queries=int(os.environ.get("SWOOSH_PKCS1_SOLVER_QUERIES", "2048")),
        max_symbolic_states=int(os.environ.get("SWOOSH_PKCS1_SYMBOLIC_STATES", "128")),
        path_seed_limit=int(os.environ.get("SWOOSH_PKCS1_PATH_SEEDS", "2")),
        timeout=int(os.environ.get("SWOOSH_PKCS1_TIMEOUT", "30")),
    )
    _assert_coverage_metrics(report)
    rates = report["coverage_per_time"]
    assert rates["final_reachable_blocks"] >= baseline["min_reachable_blocks_covered"]
    assert report["candidates_evaluated"] >= baseline["min_candidates_evaluated"]
    assert (
        rates["reachable_blocks_added"] > 0
        or rates["final_reachable_blocks"] >= baseline["min_reachable_blocks_covered"]
    )


def test_pkcs1_i15_native_no_trace_perf_metrics(tmp_path):
    pytest.importorskip("swoosh_interp")
    test_name = "bearssl_test_pkcs1_i15"
    pkg_dir, _pkl_path, input_dir = _asset_paths(test_name)
    program = _load_program(test_name)
    field_sizes = get_bpl_field_sizes(pkg_dir, program=program)
    program_inputs = parse_input_file(input_dir / "fuzz_0001.input", field_sizes=field_sizes)

    import swoosh_interp

    compiled = swoosh_interp.lower(program)
    samples = []
    last_result = None
    for idx in range(3):
        started = time.perf_counter()
        last_result = run_native(
            program,
            program_inputs,
            test_name,
            f"perf_{idx}",
            tmp_path / f"perf_{idx}.trace.raw.zst",
            compiled=compiled,
            no_trace=True,
            log_read=False,
            return_status=True,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        assert last_result["status"] == "ok"
        assert last_result["no_trace"] is True
        assert last_result["exec_ms"] >= 0
        assert last_result["blocks_explored"] == len(last_result["explored_blocks"])
        samples.append(wall_ms)

    median_ms = statistics.median(samples)
    coverage = compute_coverage(program, set(last_result["explored_blocks"]))
    assert coverage["reachable_blocks_covered"] >= _baseline(test_name)["min_reachable_blocks_covered"]

    baseline_path = os.environ.get("SWOOSH_PERF_BASELINE")
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text())
        min_blocks_per_sec = baseline[test_name]["native_blocks_per_sec"] * float(
            os.environ.get("SWOOSH_PERF_MIN_RATIO", "0.85")
        )
        actual = last_result["blocks_explored"] / max(median_ms / 1000.0, 1e-9)
        assert actual >= min_blocks_per_sec
