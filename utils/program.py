"""Boogie program structure helpers: variables, blocks, PC mapping, constants."""

import re

from parser.expression import (
    FunctionApplication, StorageIdentifier, Identifier,
    BooleanLiteral,
)
from parser.declaration import ImplementationDeclaration
from parser.statement import (
    AssertStatement, AssumeStatement, CallStatement, Block,
)
from parser.type import BooleanType, IntegerType, CustomType, MapType
from collections import defaultdict


# ── Constants ────────────────────────────────────────────────────────────

FINISH_PC = -1337
ERROR_PC = -1338

ASSERT_IGNORE_LIST = set(
    ["__VERIFIER_assert",
    "__SMACK_check_overflow",
    "__SMACK_loop_exit"])

# ── Regex patterns for call classification ───────────────────────────────

RE_VERIFIER  = re.compile(r'^__VERIFIER_nondet_[A-Za-z0-9_]+$')
RE_RECORD    = re.compile(r'^boogie_si_record_(?:i[0-9]+|ref)$')
RE_INIT_XP   = re.compile(r'^\$initialize\.cross_product$')
RE_ALLOC     = re.compile(r'^\$alloc$')
RE_ATOMIC    = re.compile(r'^corral_atomic_(?:begin|end)$')
RE_PRINTF = re.compile(
    r'^printf\.ref(?:\.(?:i[0-9]+|bool|ref))*'
    r'(?:\.cross_product)?$'
)
RE_SMACK     = re.compile(r'^__SMACK_values?[^ ]*$')
RE_SMACK_VALUE  = re.compile(r"__SMACK_value\.(?:ref|i\d+)\b")
RE_LLVM_LIFETIME = re.compile(r'@?llvm\.lifetime\.(?:start|end)\.[^\s(]+')

CALL_IGNORE_FN_PATTERNS = (
    RE_VERIFIER,
    RE_RECORD,
    RE_INIT_XP,
    RE_ALLOC,
    RE_ATOMIC,
    RE_PRINTF,
    RE_SMACK,
    RE_LLVM_LIFETIME,
)

CALL_IGNORE_LIST = set(
    ["__VERIFIER_nondet_int",
    "$initialize.cross_product",
    "$alloc",
    "corral_atomic_begin",
    "corral_atomic_end",
    "__SMACK_value.ref.cross_product",
    ])

CALL_RECORD_LIST = set(
    ["boogie_si_record_i8",
    "boogie_si_record_i32",
    "boogie_si_record_i64",
    "boogie_si_record_ref",
    ])

# ── Type helpers ─────────────────────────────────────────────────────────

boogie_type_bitwidth = {
    "i1": 1,
    "bool": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "ref": 64
}

def convert_type_to_bitwidth(type_):
    if isinstance(type_, BooleanType):
        return 1
    elif isinstance(type_, IntegerType):
        return 32
    elif isinstance(type_, CustomType):
        if type_.name == "i1":
            return 1
        elif type_.name == "i8":
            return 8
        elif type_.name == "i16":
            return 16
        elif type_.name == "i32":
            return 32
        elif type_.name == "i64":
            return 64
        elif type_.name == "bool":
            return 1
        elif type_.name == "ref":
            return 64
        elif type_.name == "$mop":
            return 64
    elif isinstance(type_, MapType):
        domain = [convert_type_to_bitwidth(t) for t in type_.domain]
        elementSort = convert_type_to_bitwidth(type_.range)
        if len(domain) == 1:
            return [domain[0], elementSort]
        else:
            assert False
    assert False, f"unknown type {type_} {type(type_)}"


# ── Variable extraction ──────────────────────────────────────────────────

def extract_boogie_variables(stmt) -> set[Identifier]:
    boogie_vars = set()
    for x in stmt.each():
        if isinstance(x, StorageIdentifier):
            boogie_vars.add(x)
    return boogie_vars


# ── Statement / block helpers ────────────────────────────────────────────

def is_assert(stmt):
    if isinstance(stmt, AssertStatement):
        return True
    elif isinstance(stmt, CallStatement):
        if stmt.procedure.name == "__VERIFIER_assert":
            return True
    elif isinstance(stmt, FunctionApplication):
        if stmt.function.name == "__VERIFIER_assert":
            return True
    else:
        return False

def extract_first_assume_stmt(block):
    for stmt in block.statements:
        if isinstance(stmt, AssertStatement) and stmt.has_attribute("hhoudini"):
            continue
        elif isinstance(stmt, AssumeStatement):
            if stmt.has_attribute("sourceloc") or stmt.has_attribute("verifier.code"):
                continue
            return stmt
        else:
            assert False, f"Unknown statement type: {stmt} {type(stmt)}"
    assert False, f"No assume statement found in {block.name}"

def generate_label_to_block(program):
    label_to_block = {}
    for decl in program.declarations:
        if isinstance(decl, ImplementationDeclaration):
            for block in decl.body.blocks:
                label_to_block[block.name] = block
    return label_to_block

def collect_all_label_conditions(proc):
    label_to_conditions = defaultdict(set)
    ret_label_to_conditions = defaultdict(set)

    if not proc.body:
        return label_to_conditions
    for block in proc.body.blocks:
        if len(block.statements) > 0 and isinstance(block.statements[0], AssumeStatement):
            for stmt in block.statements:
                if not isinstance(stmt, AssumeStatement):
                    break
                if isinstance(stmt.expression, BooleanLiteral):
                    continue
                label_to_conditions[block.name].add(stmt)

    for label in label_to_conditions:
        ret_label_to_conditions[label] = list(label_to_conditions[label])[0]
    return ret_label_to_conditions

def initialize_code_metadata(proc):
    pc_to_stmt = {}
    label_to_pc = {}
    pc_to_block = {}
    pc = 0
    for block in proc.body.blocks:
        label_to_pc[block.name] = pc
        for stmt in block.statements:
            pc_to_stmt[pc] = stmt
            pc_to_block[pc] = block.name
            pc += 1

    pc_to_label = {}
    for label in label_to_pc:
        pc_to_label[label_to_pc[label]] = label
    return pc_to_stmt, label_to_pc, pc_to_block, pc_to_label

def stmt_to_block(stmt):
    return next((b for b in stmt.each_ancestor() if isinstance(b, Block)), None)
