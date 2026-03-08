"""Store alias checking utility.

Extracted from tools/progress.py so that runtime code (StateIterator)
doesn't have to import from an analysis/visualization script.
"""

import logging
from cvc5 import Kind
from interpreter.parser.expression import FunctionApplication
from interpreter.parser.statement import AssignStatement
from interpreter.utils.utils_cvc5 import extract_variable_terms

log = logging.getLogger(__name__)


def store_aliases(ae, state_cache, pc, predicate):
    assert predicate.is_eq_predicate_with_memory_select() or predicate.is_eq_const_with_memory_select_predicate(), f"Predicate is not a memory select: {predicate}"
    stmt = state_cache.pc_to_stmt(pc)
    if not isinstance(stmt, AssignStatement):
        return False
    if len(stmt.rhs) != 1:
        return False
    if not isinstance(stmt.rhs[0], FunctionApplication):
        return False

    store_term = ae.to_cvc5(stmt.rhs[0])
    addr_term = store_term[1]

    if addr_term.getSymbol().endswith(".shadow"):
        addr_name = addr_term.getSymbol().removesuffix(".shadow")
        pc_pos_vals = pc - 1
    else:
        addr_name = addr_term.getSymbol()
        pc_pos_vals = pc

    concrete_addr = predicate.predicate[0][1]
    concrete_addr = int(concrete_addr.getBitVectorValue(), 2)
    pos_vals = ae.state_cache_new.positive_values_from_pcs(addr_name, [pc_pos_vals])
    return concrete_addr in pos_vals


def detect_store_load_correlation(ae, state_cache, store_pc, target_addr):
    """Detect when a store M[x] <- y at store_pc aliases with target_addr
    and y originates from a load M[z] where z varies in a loop.

    This is the pattern that causes proof explosions: in a loop body like
    i15_sub, M[p1981] <- M[p1971] copies array elements. When proving
    EQ(M[1170]), the store address p1981 can be 1170 in some iteration,
    but the system doesn't know which p1971 value corresponds to p1981==1170.

    Returns a dict with detection info, or None if pattern not detected:
        {
            'store_pc': int,
            'store_addr_var': str,        # e.g. '$p1981'
            'load_pc': int,
            'load_addr_var': str,         # e.g. '$p1971'
            'correlated_load_addr': int,  # z value when x == target_addr
            'offset': int,                # constant offset z - x
            'target_addr': int,
        }
    """
    stmt = state_cache.pc_to_stmt(store_pc)
    if not isinstance(stmt, AssignStatement):
        return None
    if len(stmt.rhs) != 1:
        return None
    if not isinstance(stmt.rhs[0], FunctionApplication):
        return None

    store_term = ae.to_cvc5(stmt.rhs[0])
    if store_term.getKind() != Kind.STORE:
        return None

    store_addr = store_term[1]  # x
    store_val = store_term[2]   # y (the value being stored)

    # Get the store address variable (use base name without .shadow)
    store_addr_vars = extract_variable_terms(store_addr)
    if len(store_addr_vars) != 1:
        return None
    store_addr_var = next(iter(store_addr_vars))
    store_addr_name = store_addr_var.getSymbol()
    base_store_addr_name = store_addr_name.removesuffix(".shadow")

    # Check that store address can alias with target_addr
    store_addr_pc = store_pc if not store_addr_name.endswith(".shadow") else store_pc - 1
    store_pos_vals = state_cache.positive_values_from_pcs(base_store_addr_name, [store_addr_pc])
    if target_addr not in store_pos_vals:
        return None
    if len(store_pos_vals) < 2:
        return None  # Not a loop-varying address

    # Get the stored value variable
    val_vars = extract_variable_terms(store_val)
    if len(val_vars) != 1:
        return None
    val_var = next(iter(val_vars))
    val_var_name = val_var.getSymbol()

    # Find the definition of the stored value — look for a load
    defines = state_cache.define(val_var_name)
    if not defines:
        return None

    for def_pc, def_block in defines:
        def_stmt = state_cache.pc_to_stmt(def_pc)
        if not isinstance(def_stmt, AssignStatement):
            continue
        if len(def_stmt.rhs) != 1:
            continue
        if not isinstance(def_stmt.rhs[0], FunctionApplication):
            continue

        def_rhs = ae.to_cvc5(def_stmt.rhs[0])
        # Check if it's a load (SELECT from array)
        if def_rhs.getKind() != Kind.SELECT:
            continue

        load_addr = def_rhs[1]
        load_addr_vars = extract_variable_terms(load_addr)
        if len(load_addr_vars) != 1:
            continue
        load_addr_var = next(iter(load_addr_vars))
        load_addr_name = load_addr_var.getSymbol()
        base_load_addr_name = load_addr_name.removesuffix(".shadow")

        # Get positive values for the load address
        load_addr_pc = def_pc if not load_addr_name.endswith(".shadow") else def_pc - 1
        load_pos_vals = state_cache.positive_values_from_pcs(base_load_addr_name, [load_addr_pc])
        if not load_pos_vals or len(load_pos_vals) < 2:
            continue

        # Check for consistent offset between store and load addresses
        sorted_store = sorted(store_pos_vals)
        sorted_load = sorted(load_pos_vals)

        if len(sorted_store) != len(sorted_load):
            log.debug("[detect_store_load_correlation] Mismatched lengths: "
                      "store %s=%d load %s=%d",
                      base_store_addr_name, len(sorted_store),
                      base_load_addr_name, len(sorted_load))
            continue

        offsets = [z - x for x, z in zip(sorted_store, sorted_load)]
        if len(set(offsets)) != 1:
            log.debug("[detect_store_load_correlation] Non-constant offset: "
                      "store %s load %s offsets=%s",
                      base_store_addr_name, base_load_addr_name, offsets[:5])
            continue

        offset = offsets[0]
        correlated_addr = target_addr + offset

        log.debug("[detect_store_load_correlation] DETECTED: "
                  "M[%s] <- M[%s] at pc=%d, offset=%d, "
                  "target=%d -> correlated=%d",
                  base_store_addr_name, base_load_addr_name,
                  store_pc, offset, target_addr, correlated_addr)

        return {
            'store_pc': store_pc,
            'store_addr_var': base_store_addr_name,
            'load_pc': def_pc,
            'load_addr_var': base_load_addr_name,
            'correlated_load_addr': correlated_addr,
            'offset': offset,
            'target_addr': target_addr,
        }

    return None
