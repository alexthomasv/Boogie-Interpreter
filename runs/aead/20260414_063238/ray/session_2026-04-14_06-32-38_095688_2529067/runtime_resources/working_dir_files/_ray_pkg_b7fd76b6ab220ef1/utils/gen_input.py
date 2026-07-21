"""Generate input template JSON from a compiled Boogie program package.

Primary source: the C harness file (has param names, types, struct fields, buffer sizes).
Fallback: BPL annotations ({:array}, {:field}) when no C source is found.
"""

import pickle
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
from interpreter.utils.program import RE_SMACK


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


def generate_c_template(pkg_path):
    """Generate a C-style .input template from a compiled package.

    Returns the template as a string.
    """
    pkg_path = Path(pkg_path)
    name = pkg_path.name.removesuffix("_pkg")

    # Get JSON template first (reuses all the C harness parsing logic)
    json_template = generate_template(pkg_path)

    program_pkl = pkg_path / f"{name}.pkl"
    with open(program_pkl, 'rb') as f:
        program = pickle.load(f)
    bpl_param_names, impl_decl, proc_decl = _get_bpl_param_names(program)
    entry_name = impl_decl.name.rsplit('.cross_product', 1)[0]

    # Try to get C param info for type names
    examples_dir = pkg_path.parent.parent / "examples"
    c_file = _find_c_harness(entry_name, examples_dir)
    c_params = {}  # c_name -> c_type
    c_name_order = []
    if c_file:
        params, _ = _parse_c_harness(c_file, entry_name)
        for c_type, c_name in params:
            c_params[c_name] = c_type
            c_name_order.append(c_name)

    # Build @params mapping
    params_map = []
    for i, entry in enumerate(json_template):
        bpl_name = entry['var']
        comment = entry.get('_comment', '')
        # Extract C name from comment (format: "c_name — ...")
        c_name = comment.split(' —')[0].split(' ')[0] if '—' in comment else \
                 (c_name_order[i] if i < len(c_name_order) else bpl_name)
        params_map.append((c_name, bpl_name))

    lines = []
    lines.append(f"// Auto-generated by: swoosh gen-input {name}")
    lines.append(f"// @params {' '.join(f'{c}:{b}' for c, b in params_map)}")
    lines.append("")

    for i, entry in enumerate(json_template):
        c_name = params_map[i][0]
        c_type = c_params.get(c_name, '')
        is_private = entry.get('private', False)
        comment = entry.get('_comment', '')

        if is_private:
            lines.append("// @private")

        if entry.get('struct'):
            # Struct declaration
            struct_type = c_type.replace('const ', '').replace(' *', '').replace('*', '').strip()
            if not struct_type:
                struct_type = f"struct_{c_name}"
            lines.append(f"// {comment}")
            lines.append(f"struct {struct_type} {c_name} = {{")
            for field in entry['struct']:
                fname = field['name']
                if 'buffer' in field:
                    buf = field['buffer']
                    lines.append(f"    .{fname} = zeros({buf['size']}),")
                else:
                    # Scalar — convert hex value to int for readability
                    hex_val = field.get('value', '0x0')
                    try:
                        int_val = int(hex_val, 16)
                        lines.append(f"    .{fname} = {int_val},")
                    except (ValueError, TypeError):
                        lines.append(f"    .{fname} = {hex_val},")
            lines.append("};")

        elif entry.get('buffers'):
            buf = entry['buffers'][0]
            size = buf['size']
            c_type_str = c_type if c_type else 'unsigned char'
            # Clean up type for declaration
            c_type_str = c_type_str.replace('const ', '').replace('*', '').strip()
            if not c_type_str or c_type_str in ('ref',):
                c_type_str = 'unsigned char'
            lines.append(f"// {comment}")
            lines.append(f"{c_type_str} {c_name}[{size}] = zeros({size});")

        elif 'value' in entry:
            c_type_str = c_type if c_type else 'size_t'
            c_type_str = c_type_str.replace('const ', '').strip()
            lines.append(f"// {comment}")
            lines.append(f"{c_type_str} {c_name} = {entry['value']};")

        lines.append("")

    return '\n'.join(lines)
