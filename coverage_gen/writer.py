"""Serialize ProgramInputs back to .input file text.

Generated files use Boogie variable names as both C names and Boogie names
in the @params line, which makes them directly parseable by input_parser
without losing information.

Format produced:
    // @params $p0:$p0 $p1:$p1 ...   (auto-generated if params_line not given)
    unsigned char $p0[N] = {0xAA, 0xBB, ...};
    size_t $i2 = 42;
    // @private
    unsigned char $p6[32] = {...};
"""

from interpreter.utils.inputs import Input, ProgramInputs


def write_input_file(program_inputs: ProgramInputs, params_line: str = "") -> str:
    """Serialize ProgramInputs to .input file text.

    Args:
        program_inputs: the inputs to serialize
        params_line: the original '// @params ...' line from the seed;
                     if empty, one is auto-generated from the variable names.
    """
    lines: list[str] = []

    if params_line:
        lines.append(params_line)
    else:
        pairs = " ".join(
            f"{_c_name(name)}:{name}"
            for name in program_inputs.variables
        )
        lines.append(f"// @params {pairs}")

    lines.append("")

    public = [(n, v) for n, v in program_inputs.variables.items() if not v.private]
    private = [(n, v) for n, v in program_inputs.variables.items() if v.private]

    for name, inp in public:
        lines.extend(_format_input(_c_name(name), inp))

    if private:
        lines.append("// @private")
        for name, inp in private:
            lines.extend(_format_input(_c_name(name), inp))

    return "\n".join(lines) + "\n"


def _c_name(bpl_name: str) -> str:
    """Strip leading $ from a Boogie name to get a C-safe identifier."""
    return bpl_name.lstrip("$")


def _format_input(c_name: str, inp: Input) -> list[str]:
    if inp.value is not None:
        return [f"size_t {c_name} = {inp.value};", ""]

    if inp.buffers:
        lines = []
        for buf in inp.buffers:
            contents = buf.get("contents", "0x")
            size = buf.get("size", 0)
            hex_data = contents[2:] if contents.startswith("0x") else contents
            hex_data = hex_data.ljust(size * 2, "0")[:size * 2]
            bytes_list = [f"0x{hex_data[j:j+2]}" for j in range(0, len(hex_data), 2)]
            if not bytes_list:
                bytes_list = ["0x00"]
            inner = ", ".join(bytes_list)
            lines.append(f"unsigned char {c_name}[{size}] = {{{inner}}};")
        lines.append("")
        return lines

    if inp.struct:
        lines = [f"struct __s {c_name} = {{"]
        for field in inp.struct:
            fname = field.get("name", "?")
            if "buffer" in field:
                buf = field["buffer"]
                contents = buf.get("contents", "0x")
                size = buf.get("size", 0)
                hex_data = contents[2:] if contents.startswith("0x") else contents
                hex_data = hex_data.ljust(size * 2, "0")[:size * 2]
                bytes_list = [f"0x{hex_data[j:j+2]}" for j in range(0, len(hex_data), 2)]
                inner = ", ".join(bytes_list) if bytes_list else "0x00"
                lines.append(f"    .{fname} = {{{inner}}},")
            elif "value" in field:
                lines.append(f"    .{fname} = {field['value']},")
        lines.extend(["};", ""])
        return lines

    if inp.havoc_seq is not None:
        seq_str = ", ".join(str(v) for v in inp.havoc_seq)
        return [f"int_seq {c_name} = {{{seq_str}}};", ""]

    return [f"// {c_name}: (empty)", ""]
