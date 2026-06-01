# Boogie Interpreter

Rust concrete execution engine for Boogie IVL programs. This package is part of
the Swoosh verification pipeline.

## Structure

```
├── boogie.lark              # Boogie grammar (Lark LALR)
├── parser/                  # Boogie parser and AST nodes
├── utils/                   # Shared input, bitvector, trace, and metadata helpers
├── native/                  # Rust interpreter (PyO3 module: swoosh_interp)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       ├── lib.rs           # PyO3 entry points
│       ├── lowering.rs      # Boogie AST to bytecode
│       ├── input_state.rs   # Rust ProgramInputs concretization
│       ├── vm.rs            # Bytecode VM execution
│       ├── opcodes.rs       # Bytecode format definitions
│       ├── trace.rs         # Trace accumulator
│       ├── builtins.rs      # Built-in function implementations
│       └── memory_map.rs
├── runner.py                # Rust-only CLI/runtime facade
├── coverage_gen/            # Rust-backed coverage and symbolic input generation
├── tests/                   # Active Rust-runtime tests
└── archive/legacy_python/   # Deprecated Python runtime reference archive
```

The Python interpreter runtime is deprecated and archived under
`archive/legacy_python/runtime/python`. Active code should use the Rust native
engine through `interpreter.runner`.

## Usage

```bash
python -m interpreter.runner test_packages/<name>_pkg/ --engine=native
```

`--engine=python` and `--engine=both` intentionally fail with a deprecation
message.

## Building

```bash
cd interpreter/native
maturin develop --release
```

## Tests

```bash
python -m pytest tests/differential tests/interpreter -q
python -m pytest tests/symbolic -q
python -m pytest tests/benchmark -q --run-benchmark
python -m pytest tests/benchmark/test_rust_realworld_golden.py --run-benchmark --run-exhaustive
```
