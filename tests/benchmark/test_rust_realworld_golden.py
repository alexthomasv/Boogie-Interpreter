import hashlib
import json
import pickle
from pathlib import Path

import pytest

from interpreter.runner import prepare_native, run_native
from interpreter.utils.input_parser import get_bpl_field_sizes, parse_input_file


pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.exhaustive,
    pytest.mark.native,
    pytest.mark.requires_compiled_package,
]

INTERPRETER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = INTERPRETER_ROOT.parent
FIXTURE = INTERPRETER_ROOT / "tests" / "fixtures" / "rust_realworld_golden.json"


def test_rust_realworld_golden_matches_boogie_packages(tmp_path):
    pytest.importorskip("swoosh_interp")
    if not FIXTURE.exists():
        pytest.skip(
            "missing rust real-world golden fixture; run "
            "python tests/tools/bless_rust_golden.py"
        )

    payload = json.loads(FIXTURE.read_text())
    entries = payload.get("entries") or []
    if not entries:
        pytest.skip("rust real-world golden fixture has no entries")
    expected_keys = _discover_entry_keys()
    fixture_keys = {(entry["package"], entry["input"]) for entry in entries}
    assert fixture_keys == expected_keys, (
        "rust real-world golden fixture is not exhaustive for current "
        "test_packages/test_input assets; re-run "
        "python tests/tools/bless_rust_golden.py"
    )

    cache = {}
    failures = []
    for expected in entries:
        try:
            actual = _run_entry(expected, cache, tmp_path)
            _assert_entry_matches(expected, actual)
        except AssertionError as exc:
            failures.append(f"{expected['package']}/{expected['input']}: {exc}")

    assert not failures, "\n".join(failures)


def _run_entry(expected: dict, cache: dict, tmp_path: Path) -> dict:
    package = expected["package"]
    input_name = expected["input"]
    ctx = cache.get(package)
    if ctx is None:
        pkg_dir = PROJECT_ROOT / "test_packages" / f"{package}_pkg"
        pkl_path = pkg_dir / f"{package}.pkl"
        assert pkl_path.exists(), f"missing package pickle {pkl_path}"
        assert _sha256_file(pkl_path) == expected["pkl_sha256"], (
            "package pickle hash changed; re-bless rust_realworld_golden.json"
        )
        if expected.get("package_hash") is not None:
            hash_path = pkg_dir / f"{package}.hash"
            assert hash_path.exists(), f"missing package hash {hash_path}"
            assert hash_path.read_text().strip() == expected["package_hash"], (
                "package .hash changed; re-bless rust_realworld_golden.json"
            )
        with pkl_path.open("rb") as fh:
            program = pickle.load(fh)
        ctx = {
            "program": program,
            "prepared": prepare_native(program, test_path=pkl_path),
            "field_sizes": get_bpl_field_sizes(pkg_dir, program=program),
        }
        cache[package] = ctx

    input_file = PROJECT_ROOT / "test_input" / package / input_name
    assert input_file.exists(), f"missing input {input_file}"
    assert _sha256_file(input_file) == expected["input_sha256"], (
        "input hash changed; re-bless rust_realworld_golden.json"
    )
    program_inputs = parse_input_file(input_file, field_sizes=ctx["field_sizes"])
    result = run_native(
        ctx["program"],
        program_inputs,
        package,
        input_file.stem,
        raw_log_path=tmp_path / f"{package}-{input_file.stem}.raw.zst",
        extra_data=program_inputs.extra_data,
        log_read=False,
        prepared=ctx["prepared"],
        no_trace=True,
        return_status=True,
        return_memory_summary=True,
        validate_handoff=False,
        quiet=True,
        max_steps=100_000_000,
    )
    explored = sorted(result.get("explored_blocks") or [])
    return {
        "status": result.get("status", "ok"),
        "invalid_input": bool(result.get("invalid_input", False)),
        "invalid_reason": result.get("invalid_reason"),
        "violation_block": result.get("violation_block"),
        "violation_pc": result.get("violation_pc"),
        "external_consumed": int(result.get("external_consumed", 0) or 0),
        "explored_count": len(explored),
        "explored_hash": _hash_names(explored),
        "memory_summary": _normalize_json(result.get("memory_summary") or {}),
    }


def _assert_entry_matches(expected: dict, actual: dict) -> None:
    comparable = {
        key: expected.get(key)
        for key in (
            "status",
            "invalid_input",
            "invalid_reason",
            "violation_block",
            "violation_pc",
            "external_consumed",
            "explored_count",
            "explored_hash",
            "memory_summary",
        )
    }
    assert actual == comparable


def _normalize_json(value):
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in sorted(value.items())}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json(v) for v in value]
    return value


def _hash_names(names: list[str]) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(b"\n")
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_entry_keys() -> set[tuple[str, str]]:
    keys = set()
    packages_root = PROJECT_ROOT / "test_packages"
    inputs_root = PROJECT_ROOT / "test_input"
    for pkg_dir in sorted(packages_root.glob("*_pkg")):
        package = pkg_dir.name.removesuffix("_pkg")
        pkl_path = pkg_dir / f"{package}.pkl"
        input_dir = inputs_root / package
        if not pkl_path.exists() or not input_dir.exists():
            continue
        for input_file in sorted(input_dir.glob("*.input")):
            keys.add((package, input_file.name))
    return keys
