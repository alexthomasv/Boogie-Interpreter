"""Program input parsing and array/field metadata extraction."""

import copy
import os
import re
from collections import defaultdict
from dataclasses import dataclass

from interpreter.parser.declaration import (
    ImplementationDeclaration,
    ProcedureDeclaration,
)
from interpreter.parser.expression import BinaryExpression, StorageIdentifier
from interpreter.parser.statement import CallStatement
from interpreter.parser.specifications import RequiresClause

from interpreter.utils.program import (
    RE_SMACK,
    extract_boogie_variables,
    find_entry_point,
)

# ---------------------------------------------------------------------------
# Value helpers — expand shorthand in buffer contents / struct field values
# ---------------------------------------------------------------------------

_HELPER_RE = re.compile(r'^(zeros|ones|random)\((\d+)\)$')


def _expand_contents(value):
    """Expand value helper syntax in buffer/field contents.

    Supported:
      "zeros(N)"      -> "0x" + "00" * N
      "ones(N)"       -> "0x" + "ff" * N
      "random(N)"     -> "0x" + <N random bytes as hex>
      [0x00, 0x01, …] -> "0x" + hex-encoded bytes (C-style array)
      "0x..."         -> pass through unchanged
    """
    if isinstance(value, list):
        return "0x" + "".join(f"{b & 0xff:02x}" for b in value)
    if isinstance(value, str):
        m = _HELPER_RE.match(value.strip())
        if m:
            func, n = m.group(1), int(m.group(2))
            if func == "zeros":
                return "0x" + "00" * n
            elif func == "ones":
                return "0x" + "ff" * n
            elif func == "random":
                return "0x" + os.urandom(n).hex()
    return value


@dataclass
class Input:
    """Program input variable.

    Four kinds:
      - Scalar:    value is set (int)
      - Buffer:    buffers is a list of {"contents": "0x...", "size": N}
      - Struct:    struct is an ordered list of fields, each either:
                     {"name": ..., "size": N, "value": "0x..."}           — scalar field
                     {"name": ..., "size": N, "buffer": {"contents":..}}  — pointer field
      - Havoc seq: havoc_seq is a list[int] consumed in order by
                   successive havocs of the named variable.  Used to
                   pin SMACK-inlined ``__VERIFIER_nondet_int()``
                   calls (e.g. ``inline$__VERIFIER_nondet_int$0$$i0``)
                   so the interpreter can drive a specific concrete
                   counterexample path through nondet-driven loops.
                   When the sequence is exhausted, the interpreter
                   falls back to 0 (which naturally exits
                   ``while(__VERIFIER_nondet_int())`` loops).
    """
    name: str
    private: bool
    value: int | None = None
    buffers: list | None = None
    struct: list | None = None
    havoc_seq: list | None = None

    def __str__(self):
        if self.value is not None:
            return f"{self.name} <- {self.value}"
        if self.buffers:
            return f"{self.name} <- {len(self.buffers)} buffer(s)"
        if self.struct:
            return f"{self.name} <- struct({len(self.struct)} fields)"
        if self.havoc_seq is not None:
            return f"{self.name} <- havoc_seq({len(self.havoc_seq)})"
        return f"{self.name} <- (empty)"

    @property
    def struct_buffers(self):
        """Return the list of buffer dicts from pointer fields, in order."""
        if not self.struct:
            return []
        return [f['buffer'] for f in self.struct if 'buffer' in f]

    @property
    def struct_scalars(self):
        """Return the list of scalar field dicts, in order."""
        if not self.struct:
            return []
        return [f for f in self.struct if 'value' in f]


class ProgramInputs:
    """Canonical input payload passed to the native interpreter.

    Shadow lanes are explicit data, not derived aliases. If the compiled
    program exposes ``name.shadow`` for a supplied input, ``variables`` must
    contain a separate ``Input`` under that exact key with the same payload
    kind. The native boundary rejects omitted or structurally mismatched
    shadow rows.
    """

    def __init__(self, variables: dict[str, Input], extra_data: bytes | None = None):
        if type(variables) is not dict:
            raise TypeError("ProgramInputs.variables must be a dict")
        for name, value in variables.items():
            if type(name) is not str or not name:
                raise TypeError(
                    "ProgramInputs variable names must be non-empty strings")
            if type(value) is not Input:
                raise TypeError(
                    f"ProgramInputs[{name!r}] must be Input, got "
                    f"{type(value).__name__}")
            if value.name != name:
                raise ValueError(
                    f"ProgramInputs key {name!r} does not match Input.name "
                    f"{value.name!r}")
        if extra_data is not None and type(extra_data) is not bytes:
            raise TypeError("ProgramInputs.extra_data must be bytes or None")
        self.variables = dict(variables)
        self.extra_data = extra_data


def complete_declared_shadow_inputs(
    program,
    program_inputs: ProgramInputs,
) -> ProgramInputs:
    """Materialize omitted entrypoint shadow lanes at a pipeline boundary.

    Input files may compactly specify only a base lane.  Coverage and trace
    preparation call this helper before the strict native boundary so the
    executed payload still contains every shadow formal declared by the
    cross-product entrypoint.  An explicitly supplied shadow remains
    authoritative.
    """
    shadow_names = _entry_shadow_parameter_names(program)
    if not shadow_names:
        return program_inputs
    if type(program_inputs) is not ProgramInputs:
        raise TypeError("shadow completion requires ProgramInputs")

    missing = []
    for shadow_name in sorted(shadow_names):
        if shadow_name in program_inputs.variables:
            continue
        base_name = shadow_name.removesuffix(".shadow")
        if base_name in program_inputs.variables:
            missing.append((shadow_name, base_name))

    if not missing:
        return program_inputs

    variables = copy.deepcopy(program_inputs.variables)
    for shadow_name, base_name in missing:
        clone = copy.deepcopy(variables[base_name])
        clone.name = shadow_name
        variables[shadow_name] = clone
    return ProgramInputs(variables, extra_data=program_inputs.extra_data)


def _entry_shadow_parameter_names(program) -> frozenset[str]:
    """Return the shadow formals from the entrypoint's procedure contract."""
    try:
        entry = find_entry_point(program)
    except Exception:
        return frozenset()
    if entry is None:
        return frozenset()

    source = entry
    for declaration in getattr(program, "declarations", ()) or ():
        if (
            isinstance(declaration, ProcedureDeclaration)
            and not isinstance(declaration, ImplementationDeclaration)
            and getattr(declaration, "name", None) == getattr(entry, "name", None)
            and getattr(declaration, "parameters", None)
        ):
            source = declaration
            break

    names = []
    for parameter in getattr(source, "parameters", ()) or ():
        raw_names = getattr(parameter, "names", ()) or ()
        if len(raw_names) != 1:
            continue
        name = str(raw_names[0])
        if name.endswith(".shadow"):
            names.append(name)
    return frozenset(names)


def input_contract_from_requires(proc_decl, input_names=None) -> tuple[set[str], set[str]]:
    """Extract annotation privacy metadata for input-template rendering.

    This is not execution/precondition semantics.  Runtime input validity is
    driven only by ordinary ``requires`` expressions; the compiler materializes
    CT annotation specs such as ``public_in`` into those expressions in the
    final cross-product BPL.
    """
    public: set[str] = set()
    private: set[str] = set()
    if proc_decl is None:
        return public, private

    known = None
    if input_names is not None:
        known = {
            str(name) for name in input_names
            if str(name) and not str(name).endswith(".shadow")
        }

    for spec in getattr(proc_decl, "specifications", []) or []:
        if not isinstance(spec, RequiresClause):
            continue
        for attr in getattr(spec, "attributes", []) or []:
            key = getattr(attr, "key", None)
            if key not in {"public_in", "private_in"}:
                continue
            target = public if key == "public_in" else private
            target.update(_input_names_from_attr_values(
                getattr(attr, "values", []) or [], known))
    return public, private


def input_equalities_from_requires(
    proc_decl, input_names=None
) -> list[tuple[str, str]]:
    """Return direct, typed input equalities from ordinary preconditions.

    Only equality between two declared entry parameters is input metadata.
    Equalities involving globals, locals, or results remain semantic
    preconditions but do not constrain seed-field construction here.
    """
    if proc_decl is None:
        return []

    parameter_types: dict[str, str] = {}
    parameter_order: dict[str, int] = {}
    for declaration in getattr(proc_decl, "parameters", ()) or ():
        for raw_name in getattr(declaration, "names", ()) or ():
            name = str(raw_name)
            parameter_order[name] = len(parameter_order)
            parameter_types[name] = str(getattr(declaration, "type", ""))

    known = set(parameter_types)
    if input_names is not None:
        known &= {str(name) for name in input_names}

    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for spec in getattr(proc_decl, "specifications", ()) or ():
        if not isinstance(spec, RequiresClause):
            continue
        expression = getattr(spec, "expression", None)
        if not (
            isinstance(expression, BinaryExpression)
            and expression.op == "=="
            and isinstance(expression.lhs, StorageIdentifier)
            and isinstance(expression.rhs, StorageIdentifier)
        ):
            continue
        left = str(expression.lhs.name)
        right = str(expression.rhs.name)
        if left == right or left not in known or right not in known:
            continue
        if parameter_types[left] != parameter_types[right]:
            raise ValueError(
                "input equality requires matching parameter types: "
                f"{left}: {parameter_types[left]} != "
                f"{right}: {parameter_types[right]}"
            )
        key = frozenset((left, right))
        if key in seen:
            continue
        seen.add(key)
        if parameter_order[left] <= parameter_order[right]:
            pairs.append((left, right))
        else:
            pairs.append((right, left))
    return pairs


def _input_names_from_attr_values(values, known: set[str] | None) -> set[str]:
    if known is not None:
        text = " ".join(str(v) for v in values)
        return {
            name for name in known
            if re.search(rf"(?<![\w.$]){re.escape(name)}(?![\w.$])", text)
        }

    names = set()
    for value in values:
        name = getattr(value, "name", None)
        if not name or str(name).endswith(".shadow"):
            continue
        if str(name).startswith("$M."):
            continue
        names.add(str(name))
    return names


# ── Array / field metadata ───────────────────────────────────────────────

class ArrayInfo:
    def __init__(self, mem_map, base_ptr, offset, elem_size, num_elements):
        self.mem_map = mem_map
        self.base_ptr = base_ptr
        self.offset = offset
        self.elem_size = elem_size
        self.num_elements = num_elements

    def __str__(self):
        return f"ArrayInfo(mem_map={self.mem_map}, base_ptr={self.base_ptr}, elem_size={self.elem_size}, offset={self.offset}, num_elements={self.num_elements})"

    def __repr__(self):
        return self.__str__()


class FieldInfo:
    def __init__(self, var_name, mem_map, base_ptr, size, kind):
        self.var_name = var_name
        self.mem_map = mem_map
        self.base_ptr = base_ptr
        self.size = size
        self.kind = kind

    def __str__(self):
        return f"FieldInfo(var_name={self.var_name}, mem_map={self.mem_map}, base_ptr={self.base_ptr}, size={self.size})"

    def __repr__(self):
        return self.__str__()


def process_field_stmt(stmt, is_shadow):
    var_name = stmt.get_attribute("name")[0].name
    field_info = stmt.get_attribute("field")
    base_ptr = field_info[2]
    if is_shadow:
        mem_map = f"{field_info[1].name}.shadow"
        var_name = f"{var_name}.shadow"
    else:
        mem_map = field_info[1].name
    size = int(field_info[3].value)
    kind = "buffer" if stmt.has_attribute("array") else "value"
    return FieldInfo(var_name, mem_map, base_ptr, size, kind)


def process_array_stmt(stmt, is_shadow):
    array_info = stmt.get_attribute("array")
    offset = array_info[2]
    base_ptr_vars = extract_boogie_variables(offset)
    assert len(base_ptr_vars) == 1, f"Expected 1 base pointer variable, got {base_ptr_vars}"
    base_ptr = base_ptr_vars.pop()

    if is_shadow:
        mem_map = f"{array_info[1].name}.shadow"
        base_ptr = f"{base_ptr.name}.shadow"
    else:
        mem_map = array_info[1].name
        base_ptr = base_ptr.name
    elem_size = int(array_info[3].value)
    num_elements = int(array_info[4].value)
    return ArrayInfo(mem_map, base_ptr, offset, elem_size, num_elements)


def gather_field_info_stmts(proc):
    assert isinstance(proc, ProcedureDeclaration) and proc.body is not None, (
        f"expected a procedure with a body, got {type(proc)}"
    )
    field_info_stmts = []
    seen_offsets = set()
    for block in proc.body.blocks:
        for stmt in block.statements:
            if isinstance(stmt, CallStatement):
                if RE_SMACK.match(stmt.procedure.name):
                    if stmt.has_attribute("field"):
                        name_attr = stmt.get_attribute("name")
                        field_attr = stmt.get_attribute("field")
                        key = (name_attr[0].name, str(field_attr[2]))
                        if key not in seen_offsets:
                            seen_offsets.add(key)
                            field_info_stmts.append(stmt)
    return field_info_stmts


def gather_array_info_stmts(proc):
    assert isinstance(proc, ProcedureDeclaration) and proc.body is not None, (
        f"expected a procedure with a body, got {type(proc)}"
    )
    array_info_stmts = []
    for block in proc.body.blocks:
        for stmt in block.statements:
            if isinstance(stmt, CallStatement):
                if RE_SMACK.match(stmt.procedure.name):
                    if not stmt.has_attribute("array"):
                        continue
                    array_info_stmts.append(stmt)
    return array_info_stmts


def preprocess_external_inputs(proc):
    """Extract array and field metadata from BPL annotations.

    Returns:
        arr_map:   dict[name, list[ArrayInfo]]  — keyed by {:name} attribute
        field_map: dict[name, list[FieldInfo]]  — keyed by {:name} attribute
    """
    arr_map = defaultdict(list)
    field_map = defaultdict(list)
    array_stmts = gather_array_info_stmts(proc)
    field_stmts = gather_field_info_stmts(proc)
    for stmt in array_stmts:
        name_attr = stmt.get_attribute("name")
        name = name_attr[0].name if name_attr else None

        arr_info = process_array_stmt(stmt, False)
        key = name if name else arr_info.base_ptr
        arr_map[key].append(arr_info)

        arr_info_shadow = process_array_stmt(stmt, True)
        shadow_key = f"{key}.shadow" if name else arr_info_shadow.base_ptr
        arr_map[shadow_key].append(arr_info_shadow)

    for stmt in field_stmts:
        field_info = process_field_stmt(stmt, False)
        field_map[field_info.var_name].append(field_info)
        field_info_shadow = process_field_stmt(stmt, True)
        field_map[field_info_shadow.var_name].append(field_info_shadow)
    return arr_map, field_map


def gather_ptr_aliases(proc):
    """Collect {:ptr_alias first, this} equivalences from BPL annotations.

    Returns a dict {this_ptr: first_ptr} — both for base and .shadow variants.
    """
    aliases = {}
    for block in proc.body.blocks:
        for stmt in block.statements:
            if not isinstance(stmt, CallStatement):
                continue
            if not RE_SMACK.match(stmt.procedure.name):
                continue
            alias_attr_list = stmt.attributes if hasattr(stmt, "attributes") else []
            alias_attr = next((a for a in alias_attr_list if a.key == "ptr_alias"), None)
            if alias_attr is None or len(alias_attr.values) < 2:
                continue
            first = alias_attr.values[0].name
            this = alias_attr.values[1].name
            if first != this:
                aliases[this] = first
                aliases[f"{this}.shadow"] = f"{first}.shadow"
    return aliases
