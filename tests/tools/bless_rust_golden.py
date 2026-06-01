#!/usr/bin/env python3
"""Bless Rust interpreter golden results for compiled Boogie packages."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path


INTERPRETER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = INTERPRETER_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(INTERPRETER_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERPRETER_ROOT))

from interpreter.runner import prepare_native, run_native  # noqa: E402
from interpreter.utils.input_parser import get_bpl_field_sizes, parse_input_file  # noqa: E402


def main() -> int:
    args = _parse_args()
    packages_root = args.packages_root.resolve()
    inputs_root = args.inputs_root.resolve()
    output = args.output.resolve()

    entries = []
    package_dirs = _package_dirs(packages_root, args.package)
    if args.limit is not None:
        package_dirs = package_dirs[:args.limit]

    for pkg_dir in package_dirs:
        name = pkg_dir.name.removesuffix("_pkg")
        input_dir = inputs_root / name
        pkl_path = pkg_dir / f"{name}.pkl"
        if not pkl_path.exists() or not input_dir.exists():
            continue
        input_files = sorted(input_dir.glob("*.input"))
        if args.input_limit is not None:
            input_files = input_files[:args.input_limit]
        if not input_files:
            continue

        with pkl_path.open("rb") as fh:
            program = pickle.load(fh)
        field_sizes = get_bpl_field_sizes(pkg_dir, program=program)
        prepared = prepare_native(program, test_path=pkl_path)
        package_hash = _read_text(pkg_dir / f"{name}.hash")
        pkl_sha = _sha256_file(pkl_path)

        for input_file in input_files:
            program_inputs = parse_input_file(input_file, field_sizes=field_sizes)
            raw_log_path = Path(tempfile.gettempdir()) / "swoosh-golden" / f"{name}-{input_file.stem}.raw.zst"
            raw_log_path.parent.mkdir(parents=True, exist_ok=True)
            with _suppress_stdout_fd():
                result = run_native(
                    program,
                    program_inputs,
                    name,
                    input_file.stem,
                    raw_log_path=raw_log_path,
                    extra_data=program_inputs.extra_data,
                    log_read=False,
                    prepared=prepared,
                    no_trace=True,
                    return_status=True,
                    return_memory_summary=True,
                    validate_handoff=False,
                    quiet=True,
                    max_steps=args.max_steps,
                )
            entries.append(_entry(name, package_hash, pkl_sha, input_file, result))
            print(f"{name}/{input_file.name}: {result.get('status', 'ok')}", flush=True)

    payload = {
        "schema": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "packages_root": str(packages_root),
        "inputs_root": str(inputs_root),
        "entry_count": len(entries),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(entries)} entries to {output}")
    return 0


@contextlib.contextmanager
def _suppress_stdout_fd():
    """Suppress stdout produced by interpreted programs during golden runs."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packages-root",
        type=Path,
        default=PROJECT_ROOT / "test_packages",
    )
    parser.add_argument(
        "--inputs-root",
        type=Path,
        default=PROJECT_ROOT / "test_input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERPRETER_ROOT / "tests" / "fixtures" / "rust_realworld_golden.json",
    )
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=100_000_000)
    return parser.parse_args()


def _package_dirs(packages_root: Path, names: list[str]) -> list[Path]:
    if names:
        return [packages_root / f"{name.removesuffix('_pkg')}_pkg" for name in names]
    return sorted(
        path for path in packages_root.iterdir()
        if path.is_dir() and path.name.endswith("_pkg")
    )


def _entry(package: str, package_hash: str | None, pkl_sha: str,
           input_file: Path, result: dict) -> dict:
    explored = sorted(result.get("explored_blocks") or [])
    return {
        "package": package,
        "input": input_file.name,
        "input_sha256": _sha256_file(input_file),
        "package_hash": package_hash,
        "pkl_sha256": pkl_sha,
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


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text().strip()


if __name__ == "__main__":
    raise SystemExit(main())
