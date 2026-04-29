"""Generate input templates and deterministic seeds from a compiled package.

Primary source: the C harness file (has param names, types, struct fields, buffer sizes).
Fallback: BPL annotations ({:array}, {:field}) when no C source is found.
"""

import copy
import hashlib
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

from interpreter.parser.declaration import (
    ImplementationDeclaration, ProcedureDeclaration,
)
from interpreter.parser.statement import CallStatement
from interpreter.utils.inputs import (
    preprocess_external_inputs, process_field_stmt, process_array_stmt,
)
from interpreter.utils.program import RE_SMACK, RE_SMACK_NONDET, RE_VERIFIER


# ---------------------------------------------------------------------------
# C harness parser
# ---------------------------------------------------------------------------

# Matches: public_in(__SMACK_value(expr)) or public_in(__SMACK_values(expr, size))
_ANN_RE = re.compile(
    r'(public_in|private_in)\s*\(\s*__SMACK_values?\s*\(\s*([^)]+)\s*\)\s*\)'
)
# Matches a C function signature (return_type name(params))
_SIG_RE = re.compile(
    r'(?:[\w\s\*]+?)\s+(\w+)\s*\(([^)]+)\)\s*\{'
)


def _find_c_harness(entry_point_name, examples_dir):
    """Search examples/ for the C file containing the entry point wrapper function."""
    examples = Path(examples_dir)
    if not examples.is_dir():
        return None
    for c_file in examples.rglob("*.c"):
        try:
            text = c_file.read_text()
        except Exception:
            continue
        # Look for the function definition
        if re.search(rf'\b{re.escape(entry_point_name)}\s*\(', text):
            return c_file
    return None


def _parse_c_harness(c_file, entry_point_name):
    """Parse a C harness file to extract parameter info and SMACK annotations.

    Returns:
        params: list of (c_type, c_name) tuples in order
        annotations: list of (privacy, base_expr, size_or_None) tuples
    """
    text = Path(c_file).read_text()

    # Find the wrapper function signature
    params = []
    for m in _SIG_RE.finditer(text):
        if m.group(1) == entry_point_name:
            params_str = m.group(2)
            for p in params_str.split(','):
                p = p.strip()
                # Split "const unsigned char *hash_oid" -> type + name
                # Find last word as name, everything before as type
                parts = p.rsplit(None, 1)
                if len(parts) == 2:
                    c_type, c_name = parts
                    c_name = c_name.lstrip('*')
                    c_type = c_type.rstrip() + ' *' if '*' in p else c_type.rstrip()
                    params.append((c_type, c_name))
            break

    # Extract all SMACK annotations within the wrapper function body
    # Find the function body
    func_start = text.find(f'{entry_point_name}')
    if func_start == -1:
        return params, []

    # Find opening brace
    brace_pos = text.find('{', func_start)
    if brace_pos == -1:
        return params, []

    # Find the function call (last statement before closing brace) to bound our search
    # Just search from brace to end of file — annotations come before the actual call
    body = text[brace_pos:]

    annotations = []
    for m in _ANN_RE.finditer(body):
        privacy = m.group(1)  # "public_in" or "private_in"
        args = m.group(2).strip()

        # Parse args: either "expr" or "expr, size"
        # Handle expressions like "sk->p + 2, 254"
        # Split on the LAST comma to separate size from expression
        parts = args.rsplit(',', 1)
        if len(parts) == 2:
            expr = parts[0].strip()
            size_str = parts[1].strip()
            # Check if size_str is numeric or a known constant
            try:
                size = int(size_str, 0)
            except ValueError:
                size = size_str  # keep as string (e.g., "sizeof(*clen_p)")
        else:
            expr = args
            size = None  # __SMACK_value (single value, not buffer)

        annotations.append((privacy, expr, size))

    return params, annotations


def _build_template_from_c(params, annotations, bpl_param_names,
                           impl_decl=None):
    """Build template JSON from parsed C harness info.

    Maps C parameter names to BPL parameter names ($p0, $i2, etc.) by position.
    Uses BPL {:field} annotations for accurate field sizes when available.
    """
    # Map C name -> BPL name by position
    c_to_bpl = {}
    for i, (c_type, c_name) in enumerate(params):
        if i < len(bpl_param_names):
            c_to_bpl[c_name] = bpl_param_names[i]

    # Extract BPL metadata for accurate field sizes and buffer fallback sizes
    bpl_field_sizes_map = {}  # bpl_name -> [size, size, ...]
    bpl_arr_map = {}
    if impl_decl is not None:
        bpl_arr_map, _ = preprocess_external_inputs(impl_decl)
        for bpl_name in bpl_param_names:
            layout = _gather_struct_layout(impl_decl, bpl_name)
            if layout:
                sizes = []
                for kind, stmt in layout:
                    fi = process_field_stmt(stmt, is_shadow=False)
                    sizes.append(fi.size)
                bpl_field_sizes_map[bpl_name] = sizes

    # For each parameter, collect its annotations
    param_info = {}  # c_name -> {type, fields, buffers, privacy, ...}
    for c_type, c_name in params:
        is_pointer = '*' in c_type
        param_info[c_name] = {
            'c_type': c_type,
            'c_name': c_name,
            'bpl_name': c_to_bpl.get(c_name, c_name),
            'is_pointer': is_pointer,
            'fields': [],       # struct fields: [field_name, size, privacy]
            'buffer_size': 0,   # total buffer size (for simple buffers)
            'has_private': False,
            'is_struct': False,
        }

    # Process annotations
    for privacy, expr, size in annotations:
        base_param = None
        field_name = None

        # Check for struct field access: param->field or param->field + offset
        field_match = re.match(r'(\w+)->(\w+)(?:\s*\+\s*\d+)?$', expr)
        if field_match:
            base_param = field_match.group(1)
            field_name = field_match.group(2)
        else:
            # Simple parameter reference: "param" or "param + offset"
            param_match = re.match(r'(\w+)(?:\s*\+\s*\d+)?$', expr)
            if param_match:
                base_param = param_match.group(1)

        if base_param not in param_info:
            continue

        info = param_info[base_param]
        if privacy == 'private_in':
            info['has_private'] = True

        if field_name:
            info['is_struct'] = True
            int_size = size if isinstance(size, int) else 0
            existing = [f for f in info['fields'] if f[0] == field_name]
            if existing:
                existing[0][1] += int_size  # accumulate size
            else:
                if size is not None:
                    info['fields'].append([field_name, int_size, privacy])
                else:
                    info['fields'].append([field_name, None, privacy])  # scalar field
        elif size is not None:
            int_size = size if isinstance(size, int) else 0
            info['buffer_size'] += int_size

    # Build template entries
    template = []
    for c_type, c_name in params:
        info = param_info[c_name]
        bpl_name = info['bpl_name']
        # A struct param is public (only its contents may be private)
        is_private = info['has_private'] and not info['is_struct']

        if info['is_struct']:
            bpl_sizes = bpl_field_sizes_map.get(bpl_name)
            entry = _build_struct_from_c(bpl_name, False, c_name, info,
                                         bpl_field_sizes=bpl_sizes)
        elif info['buffer_size'] > 0:
            entry = {
                'var': bpl_name,
                'private': is_private,
                '_comment': f'{c_name} — buffer ({info["buffer_size"]} bytes)',
                'buffers': [{'contents': f'zeros({info["buffer_size"]})', 'size': info['buffer_size']}],
            }
        elif info['is_pointer']:
            # Try to get size from BPL array annotation as fallback
            arr_infos = [ai for ai in bpl_arr_map.get(bpl_name, [])
                         if '.shadow' not in ai.mem_map]
            if arr_infos:
                bpl_size = arr_infos[0].elem_size * arr_infos[0].num_elements
                entry = {
                    'var': bpl_name,
                    'private': is_private,
                    '_comment': f'{c_name} — buffer ({bpl_size} bytes)',
                    'buffers': [{'contents': f'zeros({bpl_size})', 'size': bpl_size}],
                }
            else:
                entry = {
                    'var': bpl_name,
                    'private': is_private,
                    '_comment': f'{c_name} — buffer (set size)',
                    'buffers': [{'contents': 'zeros(0)', 'size': 0}],
                }
        else:
            entry = {
                'var': bpl_name,
                'private': is_private,
                '_comment': f'{c_name} — scalar ({info["c_type"]})',
                'value': 0,
            }

        template.append(entry)

    return template


def _build_struct_from_c(bpl_name, is_private, c_name, info, bpl_field_sizes=None):
    """Build a struct template entry from C harness field annotations.

    bpl_field_sizes: list of int sizes from BPL {:field} annotations, in order.
    """
    fields = []
    seen_fields = {}  # field_name -> entry index (dedup)

    for field_name, size, privacy in info['fields']:
        if field_name in seen_fields:
            # Accumulate buffer size for split annotations (e.g., public 2 + private 254)
            idx = seen_fields[field_name]
            if 'buffer' in fields[idx]:
                fields[idx]['buffer']['size'] += (size or 0)
                fields[idx]['buffer']['contents'] = f'zeros({fields[idx]["buffer"]["size"]})'
            continue

        # Use BPL field size if available, otherwise default
        field_idx = len(fields)
        if bpl_field_sizes and field_idx < len(bpl_field_sizes):
            field_size = bpl_field_sizes[field_idx]
        else:
            field_size = 8 if (size is not None and size > 0) else 4

        seen_fields[field_name] = field_idx
        if size is not None and size > 0:
            # Pointer field (buffer)
            fields.append({
                'name': field_name,
                'size': field_size,
                'buffer': {'contents': f'zeros({size})', 'size': size},
            })
        else:
            # Scalar field
            fields.append({
                'name': field_name,
                'size': field_size,
                'value': '0x' + '00' * field_size,
            })

    comment = f'{c_name} — struct ({len(fields)} fields)'
    return {
        'var': bpl_name,
        'private': is_private,
        '_comment': comment,
        'struct': fields,
    }


# ---------------------------------------------------------------------------
# BPL-only fallback (when no C source is found)
# ---------------------------------------------------------------------------

def _find_declarations(program):
    """Find the entry-point implementation and its matching procedure declaration."""
    impl_decl = None
    for d in program.declarations:
        if isinstance(d, ImplementationDeclaration) and d.has_attribute("entrypoint"):
            impl_decl = d
            break
    if impl_decl is None:
        return None, None

    proc_decl = None
    for d in program.declarations:
        if (isinstance(d, ProcedureDeclaration)
                and not isinstance(d, ImplementationDeclaration)
                and d.name == impl_decl.name):
            proc_decl = d
            break
    return impl_decl, proc_decl


def _get_bpl_param_names(program):
    """Get BPL parameter names (non-shadow) from the entry point."""
    impl_decl, proc_decl = _find_declarations(program)
    if impl_decl is None:
        return [], impl_decl, proc_decl

    source = proc_decl if proc_decl and proc_decl.parameters else impl_decl
    names = []
    for param in source.parameters:
        assert len(param.names) == 1
        name = param.names[0]
        if '.shadow' not in name:
            names.append(name)
    return names, impl_decl, proc_decl


def _safe_c_name(name):
    """Return a declaration-safe C-ish name for generated .input files."""
    out = re.sub(r'[^A-Za-z0-9_$]', '_', str(name).lstrip('$'))
    if not out:
        out = "input"
    if out[0].isdigit():
        out = f"v_{out}"
    return out


def _find_havoc_vars(program):
    """Find source-level nondet vars that can be driven by int_seq inputs.

    SMACK-heavy crypto packages contain thousands of compiler-internal
    ``__SMACK_nondet_*`` temporaries.  Exposing all of them makes templates
    unusable and does not help target user-controlled paths.  Keep only the
    inline source-level nondet names, such as Code2Inv's
    ``inline$__VERIFIER_nondet_int$...`` variables.
    """
    names = []
    seen = set()
    for decl in getattr(program, "declarations", []):
        if isinstance(decl, ImplementationDeclaration):
            for block in getattr(decl.body, "blocks", []):
                for stmt in getattr(block, "statements", []):
                    if not isinstance(stmt, CallStatement):
                        continue
                    proc_name = getattr(getattr(stmt, "procedure", None),
                                        "name", "")
                    if not (RE_VERIFIER.match(proc_name)
                            or RE_SMACK_NONDET.match(proc_name)):
                        continue
                    for ident in getattr(stmt, "assignments", []) or []:
                        name = str(getattr(ident, "name", ident))
                        if ".shadow" in name or name in seen:
                            continue
                        if "__VERIFIER_nondet" not in name and "__SMACK_nondet" not in name:
                            continue
                        seen.add(name)
                        names.append(name)
    return names


def _gather_struct_layout(proc, param_name):
    """BPL fallback: determine struct field layout from {:field}/{:array} annotations."""
    layout = []
    seen_offsets = set()
    for block in proc.body.blocks:
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
            if stmt.has_attribute("array"):
                layout.append(("pointer", stmt))
            else:
                layout.append(("scalar", stmt))
    return layout


def _generate_bpl_fallback(program, bpl_param_names, impl_decl, proc_decl):
    """Generate template using only BPL annotations (no C source available)."""
    arr_map, field_map = preprocess_external_inputs(impl_decl)

    template = []
    for param_name in bpl_param_names:
        field_infos = [fi for fi in field_map.get(param_name, [])
                       if '.shadow' not in fi.mem_map]
        arr_infos = [ai for ai in arr_map.get(param_name, [])
                     if '.shadow' not in ai.mem_map]

        if field_infos:
            layout = _gather_struct_layout(impl_decl, param_name)
            fields = []
            for kind, stmt in layout:
                fi = process_field_stmt(stmt, is_shadow=False)
                if kind == "scalar":
                    fields.append({
                        'name': f'field_{len(fields)}',
                        'size': fi.size,
                        'value': '0x' + '00' * fi.size,
                    })
                else:
                    ai = process_array_stmt(stmt, is_shadow=False)
                    buf_size = ai.elem_size * ai.num_elements
                    fields.append({
                        'name': f'field_{len(fields)}',
                        'size': fi.size,
                        'buffer': {'contents': f'zeros({buf_size})', 'size': buf_size},
                    })
            entry = {
                'var': param_name, 'private': False,
                '_comment': f'struct ({len(fields)} fields) — sizes may need adjustment',
                'struct': fields,
            }
        elif arr_infos:
            ai = arr_infos[0]
            size = ai.elem_size * ai.num_elements
            entry = {
                'var': param_name, 'private': False,
                '_comment': f'buffer ({size} bytes) — size may need adjustment',
                'buffers': [{'contents': f'zeros({size})', 'size': size}],
            }
        else:
            entry = {
                'var': param_name, 'private': False,
                '_comment': 'scalar', 'value': 0,
            }
        template.append(entry)

    return template


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_template(pkg_path):
    """Generate a JSON-serializable template list from a compiled package.

    Tries to find and parse the C harness file for accurate names/sizes.
    Falls back to BPL annotations if no C source is found.

    Args:
        pkg_path: Path to test_packages/<name>_pkg/
    Returns:
        list of dicts suitable for json.dump
    """
    pkg_path = Path(pkg_path)
    name = pkg_path.name.removesuffix("_pkg")
    program_pkl = pkg_path / f"{name}.pkl"

    with open(program_pkl, 'rb') as f:
        program = pickle.load(f)

    bpl_param_names, impl_decl, proc_decl = _get_bpl_param_names(program)
    assert impl_decl is not None, "No {:entrypoint} implementation found in program"

    # Get entry point name (strip .cross_product suffix)
    entry_name = impl_decl.name.rsplit('.cross_product', 1)[0]

    # Try to find the C harness
    examples_dir = pkg_path.parent.parent / "examples"
    c_file = _find_c_harness(entry_name, examples_dir)

    if c_file:
        params, annotations = _parse_c_harness(c_file, entry_name)
        if params:
            print(f"[gen-input] Using C harness: {c_file}")
            return _build_template_from_c(params, annotations, bpl_param_names,
                                          impl_decl=impl_decl)

    # Fallback to BPL annotations
    print(f"[gen-input] No C harness found for '{entry_name}', using BPL annotations (sizes may be inaccurate)")
    return _generate_bpl_fallback(program, bpl_param_names, impl_decl, proc_decl)


def _load_template_context(pkg_path):
    """Load template plus enough metadata to render C-style .input files."""
    pkg_path = Path(pkg_path)
    name = pkg_path.name.removesuffix("_pkg")
    json_template = generate_template(pkg_path)

    program_pkl = pkg_path / f"{name}.pkl"
    with open(program_pkl, 'rb') as f:
        program = pickle.load(f)
    _bpl_param_names, impl_decl, _proc_decl = _get_bpl_param_names(program)
    entry_name = impl_decl.name.rsplit('.cross_product', 1)[0]

    examples_dir = pkg_path.parent.parent / "examples"
    c_file = _find_c_harness(entry_name, examples_dir)
    c_params = {}
    c_name_order = []
    if c_file:
        params, _ = _parse_c_harness(c_file, entry_name)
        for c_type, c_name in params:
            c_params[c_name] = c_type
            c_name_order.append(c_name)

    params_map = []
    seen_bpl = set()
    for i, entry in enumerate(json_template):
        bpl_name = entry['var']
        comment = entry.get('_comment', '')
        if '—' in comment:
            c_name = comment.split('—', 1)[0].strip().split(' ')[0]
        elif ' - ' in comment:
            c_name = comment.split(' - ', 1)[0].strip().split(' ')[0]
        else:
            c_name = c_name_order[i] if i < len(c_name_order) else bpl_name
        params_map.append((c_name, bpl_name))
        seen_bpl.add(bpl_name)

    havoc_vars = [v for v in _find_havoc_vars(program) if v not in seen_bpl]
    for var in havoc_vars:
        params_map.append((_safe_c_name(var), var))

    return {
        "name": name,
        "json_template": json_template,
        "params_map": params_map,
        "c_params": c_params,
        "havoc_vars": havoc_vars,
    }


def _with_havoc_entries(json_template, havoc_vars):
    """Return template entries plus explicit int_seq entries for havoc vars."""
    entries = copy.deepcopy(json_template)
    for var in havoc_vars:
        entries.append({
            'var': var,
            'private': False,
            '_comment': f'{var} - havoc sequence',
            'havoc_seq': [0],
        })
    return entries


def _lookup_c_name(params_map, bpl_name, index):
    if index < len(params_map) and params_map[index][1] == bpl_name:
        return params_map[index][0]
    for c_name, mapped in params_map:
        if mapped == bpl_name:
            return c_name
    return _safe_c_name(bpl_name)


def _buffer_expr(contents, size):
    contents = str(contents or f'zeros({size})')
    if (contents.startswith("zeros(") or contents.startswith("ones(")
            or contents.startswith("random(") or contents.startswith("bytes(")
            or contents.startswith("{") or contents.startswith('"')):
        return contents
    if contents.startswith("0x"):
        return f'bytes("{contents}", {size})'
    return contents


def _scalar_text(value, size=None):
    if isinstance(value, str):
        try:
            return str(int(value, 16)) if value.startswith("0x") else str(int(value, 0))
        except (TypeError, ValueError):
            return value
    try:
        val = int(value)
    except (TypeError, ValueError):
        val = 0
    if size is not None and size > 0:
        val &= (1 << (size * 8)) - 1
    return str(val)


def _render_c_input(pkg_name, json_template, params_map, c_params):
    lines = []
    lines.append(f"// Auto-generated by: swoosh gen-input {pkg_name}")
    lines.append(f"// @params {' '.join(f'{c}:{b}' for c, b in params_map)}")
    lines.append("")

    for i, entry in enumerate(json_template):
        bpl_name = entry['var']
        c_name = _lookup_c_name(params_map, bpl_name, i)
        c_type = c_params.get(c_name, '')
        is_private = entry.get('private', False)
        comment = entry.get('_comment', '')

        if is_private:
            lines.append("// @private")

        if entry.get('struct'):
            struct_type = c_type.replace('const ', '').replace(' *', '').replace('*', '').strip()
            if not struct_type:
                struct_type = f"struct_{c_name}"
            lines.append(f"// {comment}")
            lines.append(f"struct {struct_type} {c_name} = {{")
            for field in entry['struct']:
                fname = field['name']
                if 'buffer' in field:
                    buf = field['buffer']
                    size = int(buf.get('size', 0) or 0)
                    expr = _buffer_expr(buf.get('contents'), size)
                    lines.append(f"    .{fname} = {expr},")
                else:
                    lines.append(
                        f"    .{fname} = "
                        f"{_scalar_text(field.get('value', 0), field.get('size'))},")
            lines.append("};")

        elif entry.get('buffers'):
            buf = entry['buffers'][0]
            size = int(buf.get('size', 0) or 0)
            c_type_str = c_type if c_type else 'unsigned char'
            c_type_str = c_type_str.replace('const ', '').replace('*', '').strip()
            if not c_type_str or c_type_str in ('ref',):
                c_type_str = 'unsigned char'
            lines.append(f"// {comment}")
            lines.append(
                f"{c_type_str} {c_name}[{size}] = "
                f"{_buffer_expr(buf.get('contents'), size)};")

        elif 'havoc_seq' in entry:
            seq = ", ".join(str(int(v)) for v in entry.get('havoc_seq', []))
            lines.append(f"// {comment or bpl_name + ' - havoc sequence'}")
            lines.append(f"int_seq {c_name} = {{{seq}}};")

        elif 'value' in entry:
            c_type_str = c_type if c_type else 'size_t'
            c_type_str = c_type_str.replace('const ', '').strip()
            lines.append(f"// {comment}")
            lines.append(f"{c_type_str} {c_name} = {_scalar_text(entry['value'])};")

        lines.append("")

    return '\n'.join(lines)


def _repeat_hex(size, pattern):
    if size <= 0:
        return "0x"
    data = bytes(pattern[i % len(pattern)] for i in range(size))
    return "0x" + data.hex()


def _random_hex(size, salt):
    rng = random.Random(int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16))
    return "0x" + bytes(rng.randrange(0, 256) for _ in range(size)).hex()


def _set_buffer(buf, variant, salt):
    size = int(buf.get('size', 0) or 0)
    if variant == "zeros":
        buf['contents'] = f'zeros({size})'
    elif variant == "ones":
        buf['contents'] = f'ones({size})'
    elif variant == "boundaries":
        buf['contents'] = _repeat_hex(size, [0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff])
    elif variant == "alternating":
        buf['contents'] = _repeat_hex(size, [0xaa, 0x55])
    elif variant == "random":
        buf['contents'] = _random_hex(size, salt)


def _set_scalar(container, key, variant, index, size=None):
    values = {
        "zeros": 0,
        "ones": 1,
        "boundaries": [0, 1, 127, 128, 255, 256, 65535][index % 7],
        "alternating": 0xaa if index % 2 == 0 else 0x55,
        "random": random.Random(index + 0x51A7).randrange(0, 1 << 32),
    }
    value = values.get(variant, 0)
    if size is not None and size > 0:
        value &= (1 << (size * 8)) - 1
        container[key] = '0x' + f"{value:0{size * 2}x}"
    else:
        container[key] = value


def _apply_seed_variant(entries, variant):
    scalar_idx = 0
    for entry in entries:
        if entry.get('buffers'):
            for buf_idx, buf in enumerate(entry['buffers']):
                _set_buffer(buf, variant, f"{variant}:{entry['var']}:{buf_idx}")
        elif entry.get('struct'):
            for field_idx, field in enumerate(entry['struct']):
                if 'buffer' in field:
                    _set_buffer(field['buffer'], variant,
                                f"{variant}:{entry['var']}:{field_idx}")
                elif 'value' in field:
                    _set_scalar(field, 'value', variant, scalar_idx,
                                size=field.get('size'))
                    scalar_idx += 1
        elif 'havoc_seq' in entry:
            entry['havoc_seq'] = {
                "zeros": [0],
                "ones": [1, 0],
                "boundaries": [1, 1, 0],
                "alternating": [1, 0, 1, 0],
                "random": [1, 1, 1, 0],
            }.get(variant, [0])
        elif 'value' in entry:
            _set_scalar(entry, 'value', variant, scalar_idx)
            scalar_idx += 1


def generate_seed_inputs(pkg_path):
    """Return deterministic seed .input contents keyed by filename."""
    ctx = _load_template_context(pkg_path)
    base_entries = _with_havoc_entries(
        ctx["json_template"], ctx["havoc_vars"])

    variants = [
        ("seed_00_zeros.input", "zeros"),
        ("seed_01_ones.input", "ones"),
        ("seed_02_boundaries.input", "boundaries"),
        ("seed_03_alternating.input", "alternating"),
        ("seed_04_random.input", "random"),
    ]
    out = {}
    for filename, variant in variants:
        entries = copy.deepcopy(base_entries)
        _apply_seed_variant(entries, variant)
        out[filename] = _render_c_input(
            ctx["name"], entries, ctx["params_map"], ctx["c_params"])
    return out


def generate_c_template(pkg_path):
    """Generate a C-style .input template from a compiled package.

    Returns the template as a string.
    """
    ctx = _load_template_context(pkg_path)
    entries = _with_havoc_entries(ctx["json_template"], ctx["havoc_vars"])
    return _render_c_input(
        ctx["name"], entries, ctx["params_map"], ctx["c_params"])
