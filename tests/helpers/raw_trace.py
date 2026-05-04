import struct
from pathlib import Path

import zstandard as zstd


RAW_TRACE_RECORD_SIZE = 25


def raw_trace_summary(path: Path) -> dict:
    with Path(path).open("rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(
            fh, read_across_frames=True)
        data = reader.read()

    assert data[:4] == b"SWRL"
    assert data[4] == 1
    offset = 5
    var_names, offset = _read_name_table(data, offset)
    block_names, offset = _read_name_table(data, offset)
    records = data[offset:]
    assert len(records) % RAW_TRACE_RECORD_SIZE == 0
    return {
        "var_names": var_names,
        "block_names": block_names,
        "record_count": len(records) // RAW_TRACE_RECORD_SIZE,
    }


def assert_raw_trace_file(path: Path, *, expected_records: int | None = None) -> dict:
    path = Path(path)
    assert path.exists()
    assert path.stat().st_size > 0
    summary = raw_trace_summary(path)
    assert summary["var_names"]
    assert summary["block_names"]
    assert summary["record_count"] > 0
    if expected_records is not None:
        assert summary["record_count"] == expected_records
    return summary


def _read_name_table(data: bytes, offset: int) -> tuple[list[str], int]:
    count, = struct.unpack_from("<I", data, offset)
    offset += 4
    names = []
    for _ in range(count):
        size, = struct.unpack_from("<H", data, offset)
        offset += 2
        names.append(data[offset:offset + size].decode())
        offset += size
    return names, offset
