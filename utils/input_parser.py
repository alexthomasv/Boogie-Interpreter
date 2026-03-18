"""Parser for C-style .input files.

Converts C-like initializer declarations into ProgramInputs objects
that the interpreter can consume.

Syntax overview:
    // @params hash_oid:$p0 hash:$p1 hash_len:$i2 sk:$p3 x:$p4
    unsigned char hash[20] = zeros(20);          // buffer
    size_t hash_len = 20;                        // scalar
    struct br_rsa_private_key sk = {             // struct
        .n_bitlen = 2048,
        .p = zeros(256),
        .plen = 256,
    };
    // @private
    unsigned char k[32] = {0x42, 0x90, ...};     // private buffer
"""

import re
from pathlib import Path

from interpreter.utils.inputs import Input, ProgramInputs, _expand_contents


# ---------------------------------------------------------------------------
# Value expression parsing
# ---------------------------------------------------------------------------

_HELPER_RE = re.compile(r'^(zeros|ones|random)\((\d+)\)$')
_BYTES_RE = re.compile(r'^bytes\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)$')


def _parse_value_expr(expr):
    """Parse a value expression from .input syntax.

    Returns:
        ("scalar", int_value)
        ("buffer", hex_contents, size)
    """
    expr = expr.strip().rstrip(',').rstrip(';').strip()

    # Helper: zeros(N), ones(N), random(N)
    m = _HELPER_RE.match(expr)
    if m:
        func, n = m.group(1), int(m.group(2))
        contents = _expand_contents(f"{func}({n})")
        return ("buffer", contents, n)

    # bytes("0x...", N)
    m = _BYTES_RE.match(expr)
    if m:
        hex_str = m.group(1).strip()
        size = int(m.group(2))
        if not hex_str.startswith("0x"):
            hex_str = "0x" + hex_str
        return ("buffer", hex_str, size)

    # Brace-enclosed byte array: {0xAA, 0xBB, ...}
    if expr.startswith('{') and expr.endswith('}'):
        inner = expr[1:-1].strip()
        if inner:
            byte_strs = [s.strip() for s in inner.split(',') if s.strip()]
            byte_vals = [int(b, 0) for b in byte_strs]
            contents = "0x" + "".join(f"{b & 0xff:02x}" for b in byte_vals)
            return ("buffer", contents, len(byte_vals))
        return ("buffer", "0x", 0)

    # Integer literal
    try:
        val = int(expr, 0)
        return ("scalar", val)
    except ValueError:
        pass

    # Hex string literal: "0x..."
    if expr.startswith('"') and expr.endswith('"'):
        inner = expr[1:-1]
        return ("buffer", inner, len(inner.replace("0x", "")) // 2)

    raise ValueError(f"Cannot parse value expression: {expr!r}")


# ---------------------------------------------------------------------------
# Declaration parsing
# ---------------------------------------------------------------------------

# Matches: type name = expr;  or  type name[size] = expr;
_DECL_RE = re.compile(
    r'^(?:[\w\s\*]+?)\s+'       # type (e.g., "unsigned char", "size_t", "const unsigned char *")
    r'(\w+)'                     # name
    r'(?:\[(\d+)\])?'           # optional [size]
    r'\s*=\s*'                   # =
    r'(.+);'                     # value expression + semicolon
    r'\s*$'
)

# Matches: struct type_name name = {
_STRUCT_START_RE = re.compile(
    r'^struct\s+(\w+)\s+'        # struct type_name
    r'(\w+)\s*=\s*\{'           # name = {
)

# Matches a struct field: .field_name = expr,  (supports -> in field names like sk->p)
_FIELD_RE = re.compile(
    r'^\s*\.(?:[\w]+->)*(\w+)\s*=\s*(.+?)\s*,?\s*$'
)

# @params directive
_PARAMS_RE = re.compile(r'^//\s*@params\s+(.+)$')


def parse_input_file(path, field_sizes=None):
    """Parse a .input file into a ProgramInputs object.

    Args:
        path: Path to the .input file
        field_sizes: Optional dict {bpl_name: [field_size, ...]} from BPL annotations
                     Used to set correct field sizes for struct scalar/pointer fields.
    Returns:
        ProgramInputs object
    """
    path = Path(path)
    text = path.read_text()
    lines = text.split('\n')

    # Extract @params mapping
    name_map = {}  # c_name -> bpl_name
    for line in lines:
        m = _PARAMS_RE.match(line.strip())
        if m:
            for pair in m.group(1).split():
                if ':' in pair:
                    c_name, bpl_name = pair.split(':', 1)
                    name_map[c_name.strip()] = bpl_name.strip()

    # Parse declarations
    variables = {}
    private_next = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Comments
        if line.startswith('//'):
            if '@private' in line:
                private_next = True
            i += 1
            continue

        # Struct declaration (multi-line)
        m = _STRUCT_START_RE.match(line)
        if m:
            struct_type = m.group(1)
            c_name = m.group(2)
            bpl_name = name_map.get(c_name, c_name)
            bpl_sizes = (field_sizes or {}).get(bpl_name, [])

            # Collect lines until };
            fields = []
            i += 1
            while i < len(lines):
                fline = lines[i].strip()
                if fline.startswith('};'):
                    break
                if fline.startswith('//') or not fline:
                    i += 1
                    continue
                fm = _FIELD_RE.match(fline)
                if fm:
                    fname = fm.group(1)
                    fval_str = fm.group(2)
                    fval = _parse_value_expr(fval_str)
                    field_idx = len(fields)
                    field_size = bpl_sizes[field_idx] if field_idx < len(bpl_sizes) else None

                    if fval[0] == "buffer":
                        _, contents, buf_size = fval
                        fields.append({
                            "name": fname,
                            "size": field_size or 8,  # pointer size
                            "buffer": {"contents": contents, "size": buf_size},
                        })
                    else:
                        _, int_val = fval
                        sz = field_size or 4
                        hex_val = f"0x{int_val:0{sz * 2}x}"
                        fields.append({
                            "name": fname,
                            "size": sz,
                            "value": hex_val,
                        })
                i += 1

            inp = Input(
                name=bpl_name,
                private=private_next,
                struct=fields,
            )
            variables[bpl_name] = inp
            private_next = False
            i += 1
            continue

        # Scalar or buffer declaration
        m = _DECL_RE.match(line)
        if m:
            c_name = m.group(1)
            array_size = int(m.group(2)) if m.group(2) else None
            val_str = m.group(3)
            bpl_name = name_map.get(c_name, c_name)

            val = _parse_value_expr(val_str)

            if val[0] == "buffer":
                _, contents, buf_size = val
                # Use array_size from declaration if available
                if array_size and array_size > buf_size:
                    # Pad contents to declared size
                    hex_data = contents[2:] if contents.startswith("0x") else contents
                    hex_data += "00" * (array_size - buf_size)
                    contents = "0x" + hex_data
                    buf_size = array_size
                elif array_size:
                    buf_size = array_size

                inp = Input(
                    name=bpl_name,
                    private=private_next,
                    buffers=[{"contents": contents, "size": buf_size}],
                )
            else:
                _, int_val = val
                inp = Input(
                    name=bpl_name,
                    private=private_next,
                    value=int_val,
                )

            variables[bpl_name] = inp
            private_next = False
            i += 1
            continue

        # Unknown line — skip
        i += 1

    return ProgramInputs(variables=variables)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(program_inputs, pkg_path):
    """Validate inputs against BPL annotations.

    Returns list of (level, message) tuples where level is 'error' or 'warning'.
    """
    import pickle
    from interpreter.parser.declaration import ImplementationDeclaration, ProcedureDeclaration
    from interpreter.utils.inputs import preprocess_external_inputs

    pkg_path = Path(pkg_path)
    name = pkg_path.name.removesuffix("_pkg")
    program_pkl = pkg_path / f"{name}.pkl"
    if not program_pkl.exists():
        return [("warning", f"Package not found: {pkg_path}")]

    with open(program_pkl, 'rb') as f:
        program = pickle.load(f)

    # Find entry point
    impl_decl = None
    for d in program.declarations:
        if isinstance(d, ImplementationDeclaration) and d.has_attribute("entrypoint"):
            impl_decl = d
            break
    if impl_decl is None:
        return [("warning", "No entry point found in program")]

    # Get BPL parameter names
    proc_decl = None
    for d in program.declarations:
        if (isinstance(d, ProcedureDeclaration)
                and not isinstance(d, ImplementationDeclaration)
                and d.name == impl_decl.name):
            proc_decl = d
            break

    source = proc_decl if proc_decl and proc_decl.parameters else impl_decl
    bpl_params = []
    for param in source.parameters:
        assert len(param.names) == 1
        p = param.names[0]
        if '.shadow' not in p:
            bpl_params.append(p)

    arr_map, field_map = preprocess_external_inputs(impl_decl)
    issues = []

    # Check that all input variables are valid BPL parameters
    for var_name in program_inputs.variables:
        if var_name not in bpl_params:
            issues.append(("warning", f"Variable '{var_name}' not in BPL parameters: {bpl_params}"))

    # Check missing parameters
    for p in bpl_params:
        if p not in program_inputs.variables:
            issues.append(("warning", f"BPL parameter '{p}' not provided in input"))

    # Check struct field counts
    for var_name, inp in program_inputs.variables.items():
        if inp.struct is not None:
            field_infos = [fi for fi in field_map.get(var_name, [])
                           if '.shadow' not in fi.mem_map]
            if field_infos and len(inp.struct) != len(field_infos):
                issues.append(("error",
                    f"Struct '{var_name}': {len(inp.struct)} fields provided, "
                    f"BPL expects {len(field_infos)}"))

            # Check field sizes
            for idx, field in enumerate(inp.struct):
                if idx < len(field_infos):
                    expected_size = field_infos[idx].size
                    if field["size"] != expected_size:
                        issues.append(("warning",
                            f"Struct '{var_name}' field '{field.get('name', idx)}': "
                            f"size={field['size']}, BPL expects {expected_size}"))

    return issues


def get_bpl_field_sizes(pkg_path, program=None):
    """Extract BPL field sizes for validation and parsing.

    Returns dict: {bpl_param_name: [field_size, ...]}
    If *program* is provided, uses it directly instead of loading from disk.
    """
    import pickle
    from interpreter.parser.declaration import ImplementationDeclaration
    from interpreter.parser.statement import CallStatement
    from interpreter.utils.inputs import process_field_stmt
    from interpreter.utils.program import RE_SMACK

    if program is None:
        pkg_path = Path(pkg_path)
        name = pkg_path.name.removesuffix("_pkg")
        program_pkl = pkg_path / f"{name}.pkl"
        if not program_pkl.exists():
            return {}

        with open(program_pkl, 'rb') as f:
            program = pickle.load(f)

    impl_decl = None
    for d in program.declarations:
        if isinstance(d, ImplementationDeclaration) and d.has_attribute("entrypoint"):
            impl_decl = d
            break
    if impl_decl is None:
        return {}

    # For each parameter that has field annotations, get ordered sizes
    result = {}
    # Gather all param names that have fields
    param_names = set()
    for block in impl_decl.body.blocks:
        for stmt in block.statements:
            if isinstance(stmt, CallStatement) and RE_SMACK.match(stmt.procedure.name):
                if stmt.has_attribute("field"):
                    name_attr = stmt.get_attribute("name")
                    if name_attr:
                        param_names.add(name_attr[0].name)

    for param_name in param_names:
        sizes = []
        seen_offsets = set()
        for block in impl_decl.body.blocks:
            for stmt in block.statements:
                if not isinstance(stmt, CallStatement):
                    continue
                if not RE_SMACK.match(stmt.procedure.name):
                    continue
                if not stmt.has_attribute("field"):
                    continue
                name_attr = stmt.get_attribute("name")
                if not name_attr or name_attr[0].name != param_name:
                    continue
                field_attr = stmt.get_attribute("field")
                key = (name_attr[0].name, str(field_attr[2]))
                if key in seen_offsets:
                    continue
                seen_offsets.add(key)
                fi = process_field_stmt(stmt, is_shadow=False)
                sizes.append(fi.size)
        if sizes:
            result[param_name] = sizes

    return result
