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
│       ├── opcodes.rs       # Bytecode format definitions (SemanticsMode)
│       ├── trace.rs         # Trace accumulator
│       ├── builtins.rs      # Built-in dispatch (num_args / bool output)
│       ├── builtins/bv.rs   # BV-mode (wrapping/masked) intrinsic semantics
│       ├── builtins/int.rs  # Int-mode exact-ℤ intrinsic semantics
│       └── memory_map.rs
├── runner.py                # Rust-only CLI/runtime facade
├── coverage_gen/            # Rust-backed coverage and symbolic input generation
├── tests/                   # Active Rust-runtime tests
└── archive/legacy_python/   # Deprecated Python runtime reference archive
```

The Python interpreter runtime is deprecated and archived under
`archive/legacy_python/runtime/python`. Active code should use the Rust native
engine through `interpreter.runner`.

## Semantics modes

Programs carry a content-derived `SemanticsMode` (`interpreter/utils/
integer_encoding.py`; never a flag):

* **`bv`** — SMACK bit-vector encoding: the wrapping/masked i64 algebra
  (`builtins/bv.rs`) is the correct model. Historical default.
* **`int`** — SMACK unbounded-integer encoding (`type i32 = int`): the
  exact-ℤ core evaluates (`builtins/int.rs` + checked/BigInt arithmetic in
  `vm.rs`), matching the verifier's cvc5 model (`utils_cvc5` under
  `set_integer_encoding(True)`). Values outside i64 are carried exactly
  (`Value::Big`); their trace records are skipped and counted
  (`trace_big_skips` in the result dict) because the raw-log value field is
  i64 and a wrapped stand-in could manufacture false trace-refutations.
  At the MEMORY interface (load/store addresses+values, memcpy/memset/read
  handlers) out-of-i64 values fold mod 2^64 (counted as `mem_big_folds`) —
  SMACK spells negative pointer offsets as u64 two's-complement literals, and
  the fold is ring-congruent with the historical per-op wrap for +,-,*
  address chains. Division/modulus by zero panics loudly (SMT-LIB leaves it
  uninterpreted — no concrete value would be faithful).

**Concolic/symbolic input generation is BV-only**: `concolic_suggest` /
`symbolic_explore` return no candidates for Int-mode programs (the engine's
path algebra is the wrapping i64 mirror). This only affects input finding —
never verdicts. The stats dict carries `disabled_reason`.

The anti-drift tripwire between the interpreter and the cvc5 model is
`tests/differential/test_smt_kernel_diff.py` (both lanes must be 100% green;
future divergences get pinned in its `INT_LANE_DRIFT` as strict xfails).

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
