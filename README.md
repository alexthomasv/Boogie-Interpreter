# Boogie Interpreter

Concrete execution engine for Boogie IVL programs. Part of the [Swoosh](https://github.com/AminMoraworksAtUBC/boogie-parser) verification pipeline.

## Structure

```
interpreter/
├── python/              # Python interpreter
│   ├── interpreter.py   # BoogieInterpreter class
│   ├── Environment.py   # Variable bindings, trace recording
│   ├── MemoryMap.py     # Heap model (sparse map)
│   ├── Context.py       # Call stack frames
│   └── Buffer.py        # I/O buffer for cross_product reads
├── native/              # Rust native interpreter (PyO3)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       ├── lib.rs       # PyO3 entry point + pickle encoder
│       ├── lowering.rs  # Python AST → bytecode
│       ├── vm.rs        # Bytecode VM execution
│       ├── opcodes.rs   # Bytecode format definitions
│       ├── trace.rs     # Trace accumulator
│       ├── builtins.rs  # 100+ built-in function implementations
│       └── memory_map.rs
├── runner.py            # Unified entry: --engine=python|native|both
└── tests/               # All interpreter tests
```

## Usage

```bash
# Python interpreter (PyPy recommended)
pypy -m interpreter.python.interpreter test_packages/<name>_pkg/

# Rust native interpreter
python -m interpreter.runner test_packages/<name>_pkg/ --engine=native

# Cross-validation (run both, assert identical output)
python -m interpreter.runner test_packages/<name>_pkg/ --engine=both
```

## Building the native interpreter

```bash
cd native
maturin develop --release
```

## External dependencies

This package expects `parser/` and `utils/` from the parent Swoosh repo to be on `sys.path`.
