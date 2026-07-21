"""Differential kernel harness: interpreter opcode semantics vs the cvc5 model.

Permanent anti-drift tripwire between the TWO evaluators of the same Boogie
expression language:

* the native Rust interpreter (``interpreter/native/src/{vm,builtins}.rs``),
  driven here through the public parse -> ``swoosh_interp.lower`` -> execute
  path with each kernel expression compiled as ``$rK := <expr>;``;
* the verifier's cvc5 conversion (``interpreter.utils.utils_cvc5
  .convert_expr_cvc5``) under ``set_integer_encoding`` for the lane, with
  concrete operand VALUES bound to the operand names, evaluated via
  ``Solver.simplify``.

Every ``BinOp`` and ``BuiltinFn`` variant of ``opcodes.rs`` is enumerated
(see ``test_covers_every_opcode``) over boundary + seeded-random operands,
plus ``IfThenElse``, logical ``Not`` and literals (±2^31, ±2^63, > 2^63).

Two lanes:

* ``bv`` — SMACK bit-vector encoding. The interpreter's wrapping algebra is
  the CORRECT model here; every case must PASS bit-identically. Operands fed
  to the interpreter are pre-masked to the op's operand width (matching how
  values circulate in a real bv-mode program). The bv-lane program carries no
  ``type i32 = int`` alias, so content-derived detection lowers it under
  ``SemanticsMode::Bv``.
* ``int`` — SMACK unbounded-integer encoding (mathematical Z, what every
  current c2i package uses). The int-lane program declares ``type i32 =
  int;`` so detection lowers it under ``SemanticsMode::Int`` and the
  exact-Z core evaluates. Since the exact-Z core landed, EVERY int-lane
  case must pass; ``INT_LANE_DRIFT`` (the strict-xfail drift inventory that
  held 73 entries pre-Step-3) stays as the permanent mechanism for pinning
  any FUTURE divergence and must normally be empty.

Division/modulus by zero (int lane): SMT-LIB leaves ``div/mod _ 0``
uninterpreted, so the oracle cannot evaluate it to a value AND the exact-Z
interpreter refuses with a loud panic. The two evaluators AGREE the value is
undefined — such cases run isolated and pass exactly when BOTH sides refuse.

The c2i_094 incident class ($add.i32 accumulation past 2^31 + $sgt.i32
reinterpreting the sum as negative) is caught by ``add.i32/*ovf*`` and
``sgt.i32/hi_vs_small`` below — this harness failed on that class before the
exact-Z core landed, and would fail again on any regression.
"""

import random
from dataclasses import dataclass

import pytest

from interpreter.runner import prepare_native, run_native
from interpreter.tests.helpers.boogie_cases import make_program, scalar_inputs

pytestmark = [pytest.mark.differential, pytest.mark.native]

LANES = ("int", "bv")
SEED = 20260711


def _mask(width):
    return (1 << width) - 1


def _to_i64(value):
    """Two's-complement fold of an unsigned 64-bit pattern into Python i64."""
    value &= _mask(64)
    return value - (1 << 64) if value >= (1 << 63) else value


@dataclass(frozen=True)
class Case:
    cid: str                 # unique id, e.g. "add.i32/ovf_carry"
    template: str            # expr with {a},{b},{c} operand slots
    args: tuple              # raw signed operand values (math ints, i64-range)
    width: int               # operand bit width for bv lane sorts/masking
    covers: str              # opcodes.rs construct this case exercises
    lanes: tuple = LANES     # which lanes run this case
    isolated: bool = False   # lower/run alone (may fail to even compile)


def _rng():
    return random.Random(SEED)


def _rand_pair(rng, w):
    m = 1 << (w - 1)
    return (rng.randrange(-m, m), rng.randrange(-m, m))


def _arith_pairs(w, rng):
    m = 1 << (w - 1)
    return [
        ("small", (7, 3)),
        ("neg", (-7, 3)),
        ("ovf_carry", (m - 1, 1)),          # crosses 2^(w-1): the c2i_094 class
        ("minneg", (-m, -1)),
        ("rand0", _rand_pair(rng, w)),
        ("rand1", _rand_pair(rng, w)),
    ]


def _cmp_pairs(w, rng):
    m = 1 << (w - 1)
    return [
        ("small", (3, 7)),
        ("neg_vs_zero", (-1, 0)),
        ("hi_vs_small", (m + 130882 if w < 64 else m - 1, 131070)),
        ("eq", (5, 5)),
        ("rand0", _rand_pair(rng, w)),
    ]


def _bit_pairs(w, rng):
    return [
        ("small", (0b1100, 0b1010)),
        ("neg", (-1, 0x0F)),
        ("rand0", _rand_pair(rng, w)),
    ]


def _shift_pairs(w, rng):
    # Shift amounts stay < w: SMT-LIB and the interpreter agree only there
    # (interpreter masks/mods the amount; SMT-LIB saturates at >= w — an
    # LLVM-UB region no SMACK-generated program reaches).
    return [
        ("small", (0b1100, 2)),
        ("neg_val", (-8, 2)),
        ("max_shift", (1, w - 1)),
        ("rand0", (_rand_pair(rng, w)[0], rng.randrange(0, w))),
    ]


def _div_pairs(w, rng):
    m = 1 << (w - 1)
    return [
        ("pos", (7, 3)),
        ("neg_num", (-7, 3)),
        ("neg_den", (7, -3)),
        ("neg_both", (-7, -3)),
        ("by_zero", (5, 0)),
        ("minneg", (-m, -1)),
    ]


def _build_cases():
    rng = _rng()
    cases = []

    def add(cid, template, args, width, covers, lanes=LANES, isolated=False):
        cases.append(Case(cid, template, tuple(args), width, covers,
                          tuple(lanes), isolated))

    # ---- BuiltinFn: arithmetic -------------------------------------------
    for name, w, covers in [("add.i32", 32, "Add"), ("add.i64", 64, "Add"),
                            ("add.i8", 8, "Add"), ("sub.i32", 32, "Sub"),
                            ("mul.i32", 32, "Mul")]:
        for tag, pair in _arith_pairs(w, rng):
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, w, covers)

    # ---- BuiltinFn: bitwise ----------------------------------------------
    for name, covers in [("and.i32", "And"), ("or.i32", "Or"),
                         ("xor.i32", "Xor")]:
        for tag, pair in _bit_pairs(32, rng):
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, 32, covers)
    for tag, val in [("small", (0b1100,)), ("neg", (-1,)),
                     ("rand0", (_rand_pair(rng, 32)[0],))]:
        add(f"not.i32/{tag}", "$not.i32({a})", val, 32, "Not{bits}")

    # ---- BuiltinFn: shifts ------------------------------------------------
    for name, covers in [("shl.i32", "Shl"), ("lshr.i32", "Lshr"),
                         ("ashr.i32", "Ashr")]:
        for tag, pair in _shift_pairs(32, rng):
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, 32, covers)

    # ---- BuiltinFn: comparisons ------------------------------------------
    for name, covers in [("slt.i32", "Slt"), ("sle.i32", "Sle"),
                         ("sgt.i32", "Sgt"), ("sge.i32", "Sge"),
                         ("ult.i32", "Ult"), ("ule.i32", "Ule"),
                         ("ugt.i32", "Ugt"), ("uge.i32", "Uge"),
                         ("eq.i32", "BvEq"), ("ne.i32", "BvNe")]:
        for tag, pair in _cmp_pairs(32, rng):
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, 32, covers)
    for name, covers in [("slt.ref.bool", "SltBool"), ("sle.ref.bool", "SleBool"),
                         ("sgt.ref.bool", "SgtBool"), ("sge.ref.bool", "SgeBool")]:
        for tag, pair in _cmp_pairs(64, rng)[:3]:
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, 64, covers)

    # ---- BuiltinFn: division / remainder ---------------------------------
    # by_zero runs isolated: in the int lane the interpreter refuses loudly
    # (SMT-LIB div/mod by 0 is uninterpreted) and would poison the batch.
    for name, covers in [("udiv.i32", "Udiv"), ("sdiv.i32", "Sdiv"),
                         ("urem.i32", "Urem"), ("srem.i32", "Srem")]:
        for tag, pair in _div_pairs(32, rng):
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, 32, covers,
                isolated=(tag == "by_zero"))

    # ---- BuiltinFn: residual Euclidean div/mod ($idiv/$smod) --------------
    # Int-encoding residual intrinsics ({:builtin "div"/"mod"}); no bv-mode
    # spelling exists, so they run int-lane only.
    for name, covers in [("idiv.i32", "Idiv"), ("smod.i32", "Smod")]:
        for tag, pair in _div_pairs(32, rng):
            add(f"{name}/{tag}", "$%s({a}, {b})" % name, pair, 32, covers,
                lanes=("int",), isolated=(tag == "by_zero"))

    # ---- BuiltinFn: casts --------------------------------------------------
    for tag, val in [("pos", (5,)), ("neg", (-5,)), ("hi", (1 << 31,))]:
        add(f"sext.i32.i64/{tag}", "$sext.i32.i64({a})", val, 32, "Sext")
    add("sext.i8.i32/neg", "$sext.i8.i32({a})", (-100,), 8, "Sext")
    for tag, val in [("pos", (5,)), ("neg", (-1,))]:
        add(f"zext.i8.i32/{tag}", "$zext.i8.i32({a})", val, 8, "Zext")
    add("zext.i1.i32/one", "$zext.i1.i32({a})", (1,), 1, "Zext")
    for tag, val in [("pos", (5,)), ("neg", (-1,)), ("hi", ((1 << 40) + 17,))]:
        add(f"trunc.i64.i32/{tag}", "$trunc.i64.i32({a})", val, 64, "Trunc")
    add("trunc.i32.i8/hi", "$trunc.i32.i8({a})", (0x1FF,), 32, "Trunc")
    add("bitcast.ref.ref/id", "$bitcast.ref.ref({a})", (-77,), 64, "Bitcast")
    add("p2i.ref.i64/id", "$p2i.ref.i64({a})", (-77,), 64, "P2i")
    add("i2p.i64.ref/id", "$i2p.i64.ref({a})", (-77,), 64, "I2p")

    # ---- Core BinOp: arithmetic (Boogie-level +,-,*) ----------------------
    # Int-sorted in the int lane; BV64 in the bv lane. i64-overflow cases run
    # int-lane only (in the bv lane the 64-bit wrap IS the model).
    core_arith = [
        ("small", (7, 3), LANES),
        ("neg", (-7, 3), LANES),
        ("i64_ovf", (1 << 62, 1 << 62), ("int",)),
    ]
    for op, covers in [("+", "BinOp::Add"), ("-", "BinOp::Sub"),
                       ("*", "BinOp::Mul")]:
        for tag, pair, lanes in core_arith:
            if op == "*" and tag == "i64_ovf":
                pair = (1 << 32, 1 << 32)
            add(f"core{op}/{tag}", "({a} %s {b})" % op, pair, 64, covers,
                lanes=lanes)

    # ---- Core BinOp: division / modulus ---------------------------------
    # Generic Boogie `/` and `%` are Euclidean Int operations in the integer
    # lane and unsigned bvudiv/bvurem in the BV lane. Zero divisors are total
    # only in BV; Int follows SMT-LIB's undefined interpretation and the
    # native engine refuses to fabricate a concrete value.
    for op, covers in [("/", "BinOp::Div"), ("%", "BinOp::Mod")]:
        for tag, pair in _div_pairs(64, rng):
            add(f"core{op}/{tag}", "({a} %s {b})" % op, pair, 64, covers,
                isolated=(tag == "by_zero"))

    # ---- Core BinOp: comparisons / equality ------------------------------
    # bv lane uses BV64 operand sorts, where the model maps < to UNSIGNED
    # bvult — so signed/unsigned-ambiguous operands (negatives) run int-lane
    # only; non-negative operands agree in both readings.
    for op, covers in [("<", "BinOp::Lt"), (">", "BinOp::Gt"),
                       ("<=", "BinOp::Le"), (">=", "BinOp::Ge"),
                       ("==", "BinOp::Eq"), ("!=", "BinOp::Ne")]:
        add(f"core{op}/small", "({a} %s {b})" % op, (3, 7), 64, covers)
        add(f"core{op}/eq", "({a} %s {b})" % op, (5, 5), 64, covers)
        add(f"core{op}/neg", "({a} %s {b})" % op, (-1, 0), 64, covers,
            lanes=("int",))

    # ---- Core BinOp: connectives ------------------------------------------
    for op, covers in [("&&", "BinOp::And"), ("||", "BinOp::Or"),
                       ("==>", "BinOp::Implies")]:
        for tag, pair in [("tt", (1, 1)), ("tf", (1, 0)), ("ft", (0, 1)),
                          ("ff", (0, 0))]:
            add(f"core{op}/{tag}", "(({a} != 0) %s ({b} != 0))" % op, pair,
                64, covers)
    # <==> has no utils_cvc5 mapping; the oracle term is built from the
    # equivalent bool equality (documented oracle-side rewrite).
    for tag, pair in [("tt", (1, 1)), ("tf", (1, 0))]:
        add(f"core<==>/{tag}", "(({a} != 0) <==> ({b} != 0))", pair, 64,
            "BinOp::Iff")

    # ---- IfThenElse / logical Not / literals ------------------------------
    add("ite/lt", "(if {a} < {b} then {a} else {b})", (3, 7), 64, "IfThenElse")
    add("ite/ge", "(if {a} < {b} then {a} else {b})", (9, 7), 64, "IfThenElse")
    add("lognot/true", "!({a} < {b})", (3, 7), 64, "Expr::Not")
    add("lognot/false", "!({a} < {b})", (9, 7), 64, "Expr::Not")
    add("lit/true", "true", (), 64, "Expr::Bool")
    add("lit/false", "false", (), 64, "Expr::Bool")
    add("lit/2^31-1", "2147483647", (), 64, "Expr::Const")
    add("lit/2^31", "2147483648", (), 64, "Expr::Const")
    add("lit/-2^31", "(0 - 2147483648)", (), 64, "Expr::Const",
        lanes=("int",))
    add("lit/2^63-1", "9223372036854775807", (), 64, "Expr::Const")
    add("lit/-2^63", "(0 - 9223372036854775808)", (), 64, "Expr::Const",
        lanes=("int",), isolated=True)
    add("lit/2^63", "9223372036854775808", (), 64, "Expr::Const",
        lanes=("int",), isolated=True)
    add("lit/2^63+4711", "9223372036854780519", (), 64, "Expr::Const",
        lanes=("int",), isolated=True)

    return cases


CASES = _build_cases()
_CASE_BY_ID = {c.cid: c for c in CASES}
assert len(_CASE_BY_ID) == len(CASES), "duplicate case ids"


# ---------------------------------------------------------------------------
# Drift inventory (int lane) — OBSERVED, not predicted.
#
# Pre-Step-3 this pinned 73 int-lane divergences (wrapping interpreter and
# oracle-map gaps vs the math-Z model). The exact-Z interpreter core plus the
# utils_cvc5 int-encoding fixes (the built-out bitwise/shift int→bv→nat
# handler, the $not arity fix, and the $srem prelude formula) flipped every
# entry; each strict xfail XPASSed and was removed, exactly as designed.
#
# The MECHANISM is permanent: any future divergence gets pinned here as a
# strict xfail with a "wrap:"/"oracle:" reason until its side is fixed.
# ---------------------------------------------------------------------------
INT_LANE_DRIFT = {}


# ---------------------------------------------------------------------------
# Interpreter side: batch-compile `$rK := expr;` per lane and execute once.
# ---------------------------------------------------------------------------

class _ExecError(Exception):
    pass


def _param_names(idx, case):
    return {slot: f"$p{idx}_{slot}" for slot, _ in
            zip("abc", range(len(case.args)))}


def _interp_operand(value, width, lane):
    """Operand value as fed to the interpreter for this lane.

    bv lane: masked to the op's operand width and re-folded into i64 —
    the representation values have inside a real bv-mode program.
    int lane: the raw math value (i64-range by construction).
    """
    if lane == "bv":
        return _to_i64(value & _mask(width))
    return value


def _run_batch(lane, cases, tmp_path):
    params, assigns, values = [], [], {}
    for idx, case in enumerate(cases):
        names = _param_names(idx, case)
        for slot, pname in names.items():
            params.append(f"{pname}: int")
            values[pname] = _interp_operand(
                case.args["abc".index(slot)], case.width, lane)
        assigns.append(f"  $r{idx} := {case.template.format(**names)};")
    plist = ", ".join(params)
    body = "\n".join(assigns)
    locals_ = "\n".join(f"  var $r{i}: int;" for i in range(len(cases)))
    # The int lane carries SMACK's `type i32 = int` alias so the content-
    # derived detector (interpreter.utils.integer_encoding) selects
    # SemanticsMode::Int — the mode is never passed as a flag.
    alias = "type i32 = int;\n" if lane == "int" else ""
    source = f"""
    {alias}procedure main({plist});
    implementation {{:entrypoint}} main({plist}) {{
{locals_}
    entry:
{body}
      return;
    }}
    """
    program = make_program(source)
    prepared = prepare_native(program)
    result = run_native(
        program, scalar_inputs(values), "batch",
        tmp_path / f"kernel_{lane}.raw.zst",
        no_trace=True, log_read=False, return_status=True,
        prepared=prepared, return_scalar_summary=True,
        return_memory_summary=False,
    )
    assert result["status"] == "ok", (
        f"kernel batch must run clean, got {result['status']} at "
        f"{result.get('violation_block')}")
    scalars = result["final_scalars"]
    return {case.cid: scalars[f"$r{idx}"] for idx, case in enumerate(cases)}


def _run_isolated(lane, case, tmp_path):
    try:
        return _run_batch(lane, [case], tmp_path)[case.cid]
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # lowering/exec REFUSAL is the observation
        # BaseException, not Exception: the exact-Z core refuses div/mod by
        # zero with a Rust panic, which PyO3 surfaces as PanicException
        # (a direct BaseException subclass).
        return _ExecError(f"{type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def interp_results(tmp_path_factory):
    """lane -> cid -> interpreter value (or _ExecError)."""
    out = {}
    for lane in LANES:
        tmp = tmp_path_factory.mktemp(f"kernel_{lane}")
        lane_cases = [c for c in CASES if lane in c.lanes]
        batched = [c for c in lane_cases if not c.isolated]
        out[lane] = _run_batch(lane, batched, tmp)
        for case in (c for c in lane_cases if c.isolated):
            out[lane][case.cid] = _run_isolated(lane, case, tmp)
    return out


# ---------------------------------------------------------------------------
# Oracle side: convert_expr_cvc5 with concrete VALUE terms bound to the
# operand names, evaluated by cvc5.
# ---------------------------------------------------------------------------

class _OracleVars:
    def __init__(self, terms):
        self._terms = terms
        self.cached_id_to_cvc5 = {}

    def cvc5_var(self, name):
        return self._terms.get(name)


class _OracleError(Exception):
    pass


@pytest.fixture(scope="module")
def oracle():
    import cvc5

    from interpreter.utils import utils_cvc5 as u

    solver = cvc5.Solver()
    solver.setOption("produce-models", "true")
    fn_maps = {}
    for lane in LANES:
        u.set_integer_encoding(lane == "int")
        fn_maps[lane] = u.generate_cvc5_function_map(solver)
    u.set_integer_encoding(False)

    def evaluate(case, lane):
        """Return ("bool"|"bv"|"int", python value) for the oracle result."""
        from interpreter.parser.boogie_parser import parse_expr

        names = _param_names(0, case)
        template = case.template
        if "<==>" in template:
            # utils_cvc5 has no <==> mapping; model it as bool equality.
            template = template.replace("<==>", "==")
        text = template.format(**names)
        terms = {}
        for slot, pname in names.items():
            raw = case.args["abc".index(slot)]
            if lane == "int":
                terms[pname] = solver.mkInteger(str(raw))
            else:
                terms[pname] = solver.mkBitVector(case.width,
                                                  raw & _mask(case.width))
        u.set_integer_encoding(lane == "int")
        try:
            ast = parse_expr(text)
            term = u.convert_expr_cvc5(fn_maps[lane], _OracleVars(terms),
                                       solver, ast, True)
            value = solver.simplify(term)
        except Exception as exc:
            raise _OracleError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            u.set_integer_encoding(False)
        if value.getSort().isBoolean():
            if not value.isBooleanValue():
                raise _OracleError(f"unevaluated bool term: {value}")
            return "bool", int(value.getBooleanValue())
        if value.getSort().isBitVector():
            if not value.isBitVectorValue():
                raise _OracleError(f"unevaluated BV term: {value}")
            return "bv", (int(value.getBitVectorValue(2), 2),
                          value.getSort().getBitVectorSize())
        if value.getSort().isInteger():
            if not value.isIntegerValue():
                raise _OracleError(f"unevaluated Int term: {value}")
            return "int", int(value.getIntegerValue())
        raise _OracleError(f"unexpected oracle sort: {value.getSort()}")

    return evaluate


# ---------------------------------------------------------------------------
# The differential assertion
# ---------------------------------------------------------------------------

def _params():
    out = []
    for lane in LANES:
        for case in CASES:
            if lane not in case.lanes:
                continue
            marks = []
            if lane == "int" and case.cid in INT_LANE_DRIFT:
                marks.append(pytest.mark.xfail(
                    strict=True, reason=INT_LANE_DRIFT[case.cid]))
            out.append(pytest.param(lane, case.cid,
                                    id=f"{lane}:{case.cid}", marks=marks))
    return out


@pytest.mark.parametrize("lane,cid", _params())
def test_kernel_matches_cvc5(lane, cid, interp_results, oracle):
    case = _CASE_BY_ID[cid]
    interp_value = interp_results[lane][cid]

    try:
        kind, expected = oracle(case, lane)
    except _OracleError as exc:
        if "unevaluated" in str(exc):
            # The model leaves the term uninterpreted (SMT-LIB div/mod by
            # zero): agreement means the interpreter refuses a value too.
            assert isinstance(interp_value, _ExecError), (
                f"{cid} [{lane}]: model has NO value (uninterpreted: {exc}) "
                f"but the interpreter produced one: {interp_value!r}")
            return
        raise

    if isinstance(interp_value, _ExecError):
        pytest.fail(f"interpreter could not evaluate {cid}: {interp_value}")
    if kind == "bool":
        actual = int(interp_value)
        assert actual in (0, 1), f"non-boolean interp value {interp_value}"
        assert actual == expected, (
            f"{cid} [{lane}] bool: interp={actual} cvc5={expected}")
    elif kind == "bv":
        exp_val, exp_width = expected
        actual = interp_value & _mask(exp_width)
        assert actual == exp_val, (
            f"{cid} [{lane}] bv{exp_width}: interp={interp_value} "
            f"(masked {actual}) cvc5={exp_val}")
    else:
        assert interp_value == expected, (
            f"{cid} [{lane}] int: interp={interp_value} cvc5={expected}")


# ---------------------------------------------------------------------------
# Meta checks: the enumeration is complete and the inventory has no zombies.
# ---------------------------------------------------------------------------

# Mirrors interpreter/native/src/opcodes.rs — update BOTH when adding ops.
_OPCODES_RS_BINOPS = [
    "BinOp::Eq", "BinOp::Ne", "BinOp::Lt", "BinOp::Gt", "BinOp::Le",
    "BinOp::Ge", "BinOp::And", "BinOp::Or", "BinOp::Implies", "BinOp::Iff",
    "BinOp::Sub", "BinOp::Mul", "BinOp::Add",
    "BinOp::Div", "BinOp::Mod",
]
_OPCODES_RS_BUILTINS = [
    "Add", "Sub", "Mul", "And", "Or", "Xor", "Not{bits}", "Shl", "Lshr",
    "Ashr", "Slt", "Sle", "Sgt", "Sge", "Ult", "Ule", "Ugt", "Uge", "BvEq",
    "BvNe", "Udiv", "Sdiv", "Urem", "Srem", "Sext", "Zext", "Trunc",
    "Bitcast", "P2i", "I2p", "SltBool", "SleBool", "SgtBool", "SgeBool",
    "Idiv", "Smod",
]
_OTHER_CONSTRUCTS = ["IfThenElse", "Expr::Not", "Expr::Bool", "Expr::Const"]


def test_covers_every_opcode():
    covered = {c.covers for c in CASES}
    missing = [op for op in (_OPCODES_RS_BINOPS + _OPCODES_RS_BUILTINS
                             + _OTHER_CONSTRUCTS) if op not in covered]
    assert not missing, f"kernel harness lost coverage for: {missing}"


def test_drift_inventory_has_no_stale_entries():
    stale = [cid for cid in INT_LANE_DRIFT if cid not in _CASE_BY_ID]
    assert not stale, f"INT_LANE_DRIFT names unknown cases: {stale}"
    not_in_int_lane = [cid for cid in INT_LANE_DRIFT
                       if "int" not in _CASE_BY_ID[cid].lanes]
    assert not not_in_int_lane, (
        f"INT_LANE_DRIFT entries not run in the int lane: {not_in_int_lane}")
