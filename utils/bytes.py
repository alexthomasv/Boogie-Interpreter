"""Byte parsing helpers shared by Rust-only execution paths."""


def hex_to_bytes(s: str, *, as_bytearray: bool = True):
    s = "".join(s.split()).lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) % 2:
        raise ValueError("hex string must contain an even number of digits")
    blob = bytes.fromhex(s)
    return bytearray(blob) if as_bytearray else blob
