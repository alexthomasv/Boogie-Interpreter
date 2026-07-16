# Legacy Python Interpreter Archive

This directory contains the deprecated Python interpreter runtime and tests.
It is kept only as historical reference while the active interpreter runtime is
Rust-only.

The public import-path shim has been removed. Native execution should use
`interpreter.runner.run_native`, `interpreter.runner.prepare_native`, and the
`swoosh_interp` Rust extension. The few explicit differential checks that need
the archived implementation import it by its full archival path.

Archived contents:

- `runtime/python/`: old concrete Python interpreter, environment, memory map,
  and buffer model.
- `tests/`: old Python-runtime and Python/native differential tests.
- `inspectors/`: ad hoc debugging scripts that instantiated the Python
  interpreter directly.
