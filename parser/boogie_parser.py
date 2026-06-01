import re
from pathlib import Path

from lark import Lark
from lark.exceptions import LarkError, UnexpectedInput

from .transformer import BoogieToObjectTransformer


_GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "boogie.lark"
boogie_grammar = _GRAMMAR_PATH.read_text()

_START_SYMBOLS = ("start", "decl", "param_decl", "spec", "blocks", "stmt", "expr", "type")
_KIND_TO_START = {
    "program": "start",
    "start": "start",
    "decl": "decl",
    "declaration": "decl",
    "param": "param_decl",
    "param_decl": "param_decl",
    "spec": "spec",
    "blocks": "blocks",
    "stmt": "stmt",
    "statement": "stmt",
    "expr": "expr",
    "expression": "expr",
    "type": "type",
}


class BoogieParseError(SyntaxError):
    def __init__(self, kind, source, cause):
        self.kind = kind
        self.source = source
        self.cause = cause
        context = ""
        if isinstance(cause, UnexpectedInput):
            context = "\n" + cause.get_context(source)
        super().__init__(f"failed to parse Boogie {kind}: {cause}{context}")


parser = Lark(
    boogie_grammar,
    parser="lalr",
    transformer=BoogieToObjectTransformer(visit_tokens=True),
    maybe_placeholders=False,
    propagate_positions=False,
    start=list(_START_SYMBOLS),
    debug=False,
)


def _parse_with_start(source, kind):
    try:
        return parser.parse(source, start=kind)
    except LarkError as exc:
        raise BoogieParseError(kind, source, exc) from exc


def parse_boogie(source):
    return _parse_with_start(source, "start")


def parse_decl(source):
    return _parse_with_start(source, "decl")


def parse_param(source):
    return _parse_with_start(source, "param_decl")


def parse_spec(source):
    return _parse_with_start(source, "spec")


def parse_blocks(source):
    return _parse_with_start(source, "blocks")


def parse_block(source):
    blocks = parse_blocks(source)
    if len(blocks) != 1:
        cause = ValueError(f"expected exactly one block, got {len(blocks)}")
        raise BoogieParseError("block", source, cause)
    return blocks[0]


def parse_stmt(source):
    return _parse_with_start(source, "stmt")


def parse_expr(source):
    return _parse_with_start(source, "expr")


def parse_type(source):
    return _parse_with_start(source, "type")


def parse_kind(source, kind):
    normalized = str(kind).lower().strip()
    if normalized == "auto":
        return parse_auto(source)
    if normalized == "block":
        return parse_block(source)
    try:
        start = _KIND_TO_START[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted([*_KIND_TO_START, "auto", "block"]))
        raise ValueError(f"unknown Boogie parse kind {kind!r}; expected one of: {choices}") from exc
    return _parse_with_start(source, start)


def parse_auto(source):
    stripped = source.strip()
    candidates = []
    if re.search(r"\b(type|const|function|axiom|var|procedure|implementation|uses)\b", stripped):
        candidates.append(parse_decl)
    if re.match(r"\A\s*[\w$\\.]+:.*;.*\s+[\w$\\.]+:", stripped, re.MULTILINE):
        candidates.append(parse_blocks)
    elif re.match(r"\A\s*[\w$\\.]+:.*;", stripped, re.MULTILINE):
        candidates.append(parse_block)
    if re.search(r"\b(requires|ensures|modifies|invariant)\b", stripped):
        candidates.append(parse_spec)
    if re.search(r"\b(assert|assume|havoc|call|if|while|break|goto|return|yield|pop|par)\b|:=", stripped):
        if not re.match(r"\A\s*if\b.*\bthen\b.*\belse\b", stripped, re.DOTALL):
            candidates.append(parse_stmt)
    if re.search(r"[^<:]:[^:]", stripped):
        candidates.append(parse_param)
    candidates.append(parse_expr)

    seen = set()
    last_error = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return candidate(stripped)
        except BoogieParseError as exc:
            last_error = exc
    raise last_error


def parse_str(source):
    return parse_auto(source)


def bpl(source, kind=None):
    if kind is None:
        return parse_auto(source)
    return parse_kind(source, kind)
