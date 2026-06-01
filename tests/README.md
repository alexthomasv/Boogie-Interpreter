# Test Organization

The suite is split by behavior instead of implementation detail:

- `helpers/`: shared Boogie program builders and execution helpers.
- `unit/`: isolated tests for small helpers.
- `interpreter/`: concrete Rust interpreter semantics.
- `symbolic/`: Rust symbolic/concolic candidate generation and replay checks.
- `differential/`: Rust runtime consistency checks.
- `property/`: property-style semantic invariants.
- `integration/`: cross-subsystem workflows and benchmark-oriented tests.
- `benchmark/`: asset-backed coverage and native performance checks.

Pytest markers are registered in `pytest.ini`. Slow, benchmark, compiled-package, and exhaustive tests are skipped by default so `python -m pytest` stays suitable for local development.

## Common Commands

Run the default development suite:

```bash
python -m pytest
```

Run focused correctness suites:

```bash
python -m pytest tests/interpreter tests/symbolic tests/differential tests/property
```

Run slow or benchmark tests explicitly:

```bash
python -m pytest --run-slow -m slow
python -m pytest --run-benchmark -m benchmark
```

Run BearSSL PKCS1 coverage/performance checks:

```bash
python -m pytest tests/benchmark/test_bearssl_pkcs1_coverage.py --run-benchmark
python -m pytest tests/benchmark/test_bearssl_pkcs1_coverage.py --run-benchmark --run-slow --run-exhaustive
```

Run exhaustive tests for deeper validation:

```bash
python -m pytest --run-exhaustive -m exhaustive
```

Check Rust symbolic/concolic candidate replay:

```bash
python -m pytest tests/symbolic
```

Refresh gated real-world and symbolic performance baselines:

```bash
python tests/tools/bless_rust_golden.py
python tests/tools/bless_symbolic_perf.py
```

Native tests use `pytest.importorskip("swoosh_interp")`, so they skip cleanly when the compiled PyO3 module is unavailable.
