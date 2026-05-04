# Legacy Python Interpreter Archive

This directory contains the deprecated Python interpreter runtime and tests.
It is kept only as historical reference while the active interpreter runtime is
Rust-only.

Active code should not import from `archive/legacy_python/runtime/python` or
from `interpreter.python`. Use `interpreter.runner.run_native`,
`interpreter.runner.prepare_native`, and the `swoosh_interp` Rust extension.

Archived contents:

- `runtime/python/`: old concrete Python interpreter, environment, memory map,
  and buffer model.
- `tests/`: old Python-runtime and Python/native differential tests.
- `inspectors/`: ad hoc debugging scripts that instantiated the Python
  interpreter directly.
