# Boogie Interpreter

Concrete execution engine for Boogie IVL programs. Part of the [Swoosh](https://github.com/alexthomasv/boogie-parser) verification pipeline.

## Structure

```
├── boogie.lark              # Boogie grammar (Lark LALR)
├── parser/                  # Boogie parser (AST nodes + transformer)
│   ├── boogie_parser.py     # Lark parser + transformer
│   ├── expression.py        # Expression AST nodes
│   ├── statement.py         # Statement AST nodes
│   ├── declaration.py       # Declaration AST nodes
│   └── ...
├── utils/                   # Shared utilities
│   └── utils.py             # Built-in functions, input parsing, code metadata
├── interpreter/             # Interpreter implementations
│   ├── python/              # Python interpreter
│   │   ├── interpreter.py   # BoogieInterpreter class
│   │   ├── Environment.py   # Variable bindings, trace recording
│   │   ├── MemoryMap.py     # Heap model (sparse map)
│   │   ├── Context.py       # Call stack frames
│   │   └── Buffer.py        # I/O buffer for cross_product reads
│   ├── native/              # Rust native interpreter (PyO3)
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml
│   │   └── src/
│   │       ├── lib.rs       # PyO3 entry point + pickle encoder
│   │       ├── lowering.rs  # Python AST → bytecode
│   │       ├── vm.rs        # Bytecode VM execution
│   │       ├── opcodes.rs   # Bytecode format definitions
│   │       ├── trace.rs     # Trace accumulator
│   │       ├── builtins.rs  # 100+ built-in function implementations
│   │       └── memory_map.rs
│   └── runner.py            # Unified entry: --engine=python|native|both
└── tests/                   # All tests (under interpreter/tests/)
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
cd interpreter/native
maturin develop --release
```

## Running tests

```bash
# Unit tests
python -m pytest interpreter/tests/test_interpreter.py -q

# Expression op tests (Python vs Rust cross-validation)
python -m pytest interpreter/tests/test_expression_ops.py -q
```
