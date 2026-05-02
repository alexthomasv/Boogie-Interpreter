# Test Organization

The suite is split by behavior instead of implementation detail:

- `helpers/`: shared Boogie program builders and execution helpers.
- `unit/`: isolated tests for small Python helpers.
- `interpreter/`: concrete Python interpreter semantics.
- `symbolic/`: concolic and symbolic executor behavior.
- `differential/`: Python/native cross-checks.
- `property/`: property-style semantic invariants.
- `integration/`: cross-subsystem workflows and benchmark-oriented tests.

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

Run exhaustive tests for deeper validation:

```bash
python -m pytest --run-exhaustive -m exhaustive
```

Native tests use `pytest.importorskip("swoosh_interp")`, so they skip cleanly when the compiled PyO3 module is unavailable.

