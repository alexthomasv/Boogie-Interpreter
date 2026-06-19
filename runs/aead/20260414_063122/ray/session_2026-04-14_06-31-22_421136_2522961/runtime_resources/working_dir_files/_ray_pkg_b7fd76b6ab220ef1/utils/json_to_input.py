"""Convert JSON input files to C-style .input format.

Usage:
    python3 -m interpreter.utils.json_to_input test_input/aead/input_0.json
    python3 -m interpreter.utils.json_to_input test_input/aead/  # convert all .json in dir
"""

import json
import sys
from pathlib import Path


def _hex_abbrev(hex_str, size):
    """Convert a hex string to the most concise .input representation."""
    raw = hex_str.replace("0x", "").replace("0X", "")
    if not raw:
        return f"zeros({size})"
    # Pad to size if needed
    if len(raw) < size * 2:
        raw = raw + "00" * (size - len(raw) // 2)
    # Check patterns
    byte_set = set(raw[i:i+2] for i in range(0, len(raw), 2))
    if byte_set == {"00"}:
        return f"zeros({size})"
    if byte_set == {"ff"}:
        return f"ones({size})"
    # Use brace notation for small arrays, hex string for large
    if size <= 32:
        bytes_list = [f"0x{raw[i:i+2]}" for i in range(0, len(raw), 2)]
        return "{" + ", ".join(bytes_list) + "}"
    return f'"0x{raw}"'


def json_to_input(json_path):
    """Convert a JSON input file to C-style .input format string."""
    with open(json_path) as f:
        entries = json.load(f)

    lines = []
    params_parts = []
    private_vars = set()

    # First pass: collect param names and privacy
    for entry in entries:
        if 'extra_data' in entry:
            continue
        var = entry['var']
        # Use var name without $ as C name
        c_name = var.lstrip('$')
        params_parts.append(f"{c_name}:{var}")
        if entry.get('private', False):
            private_vars.add(var)

    lines.append(f"// @params {' '.join(params_parts)}")
    lines.append("")

    for entry in entries:
        if 'extra_data' in entry:
            continue
        var = entry['var']
        c_name = var.lstrip('$')
        is_private = entry.get('private', False)

        if is_private:
            lines.append("// @private")

        if entry.get('struct'):
            # Struct
            lines.append(f"struct auto_{c_name} {c_name} = {{")
            for field in entry['struct']:
                fname = field['name']
                if 'buffer' in field:
                    buf = field['buffer']
                    contents = buf.get('contents', f"zeros({buf['size']})")
                    abbr = _hex_abbrev(contents, buf['size'])
                    lines.append(f"    .{fname} = {abbr},")
                elif 'value' in field:
                    val = field['value']
                    if isinstance(val, str) and val.startswith("0x"):
                        int_val = int(val, 16)
                    elif isinstance(val, int):
                        int_val = val
                    else:
                        int_val = 0
                    lines.append(f"    .{fname} = {int_val},")
            lines.append("};")

        elif entry.get('buffers'):
            buf = entry['buffers'][0]
            size = buf['size']
            contents = buf.get('contents', f"zeros({size})")
            abbr = _hex_abbrev(contents, size)
            lines.append(f"unsigned char {c_name}[{size}] = {abbr};")

        elif 'value' in entry and entry['value'] is not None:
            lines.append(f"size_t {c_name} = {entry['value']};")

        else:
            # No value, no buffers, no struct — bare parameter (pointer, value=0)
            lines.append(f"size_t {c_name} = 0;")

        lines.append("")

    return '\n'.join(lines)


def convert_file(json_path):
    """Convert a .json file to .input, writing next to the original."""
    json_path = Path(json_path)
    input_path = json_path.with_suffix('.input')
    content = json_to_input(json_path)
    input_path.write_text(content)
    print(f"Converted {json_path} -> {input_path}")
    return input_path


def convert_directory(dir_path):
    """Convert all .json files in a directory."""
    dir_path = Path(dir_path)
    converted = []
    for json_file in sorted(dir_path.glob("*.json")):
        converted.append(convert_file(json_file))
    return converted


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 -m interpreter.utils.json_to_input <path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        convert_directory(path)
    elif path.is_file():
        convert_file(path)
    else:
        print(f"Not found: {path}")
        sys.exit(1)
