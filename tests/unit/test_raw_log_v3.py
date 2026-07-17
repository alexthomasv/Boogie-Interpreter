import struct

import pytest
import zstandard as zstd

from interpreter.utils.raw_log import (
    MAGIC,
    NO_VAR_ID,
    OP_INITIAL_SCALAR,
    OP_PRE_PC,
    OP_UNKNOWN_WRITE,
    OP_WRITE,
    RECORD_SIZE,
    UNKNOWN_REASON_BIG_INT,
    VERSION,
    RawLogWriter,
)


def _read_table(data: bytes, offset: int) -> tuple[list[str], int]:
    count, = struct.unpack_from("<I", data, offset)
    offset += 4
    names = []
    for _ in range(count):
        size, = struct.unpack_from("<H", data, offset)
        offset += 2
        names.append(data[offset:offset + size].decode())
        offset += size
    return names, offset


def test_v3_state_events_keep_fixed_width_and_physical_order(tmp_path):
    path = tmp_path / "ordered.trace.raw.zst"
    writer = RawLogWriter(path)
    writer.write_header(["$x"], ["entry"])
    writer.record(OP_INITIAL_SCALAR, 0, 0, 0, 7, 0)
    writer.record(OP_PRE_PC, NO_VAR_ID, 11, 0, 0, 0)
    writer.record(OP_WRITE, 0, 11, 0, 9, 0)
    writer.record(
        OP_UNKNOWN_WRITE,
        0,
        12,
        0,
        UNKNOWN_REASON_BIG_INT,
        0,
    )
    assert writer.finish() == 4

    with path.open("rb") as handle:
        data = zstd.ZstdDecompressor().stream_reader(
            handle, read_across_frames=True).read()

    assert data[:4] == MAGIC
    assert data[4] == VERSION == 3
    variables, offset = _read_table(data, 5)
    blocks, offset = _read_table(data, offset)
    assert variables == ["$x"]
    assert blocks == ["entry"]

    payload = data[offset:]
    assert len(payload) == 4 * RECORD_SIZE
    records = [
        struct.unpack_from("<BIIIqI", payload, index * RECORD_SIZE)
        for index in range(4)
    ]
    assert [record[0] for record in records] == [
        OP_INITIAL_SCALAR,
        OP_PRE_PC,
        OP_WRITE,
        OP_UNKNOWN_WRITE,
    ]
    assert records[1][1:] == (NO_VAR_ID, 11, 0, 0, 0)
    assert records[3][1:] == (0, 12, 0, UNKNOWN_REASON_BIG_INT, 0)


def test_v3_exact_state_events_reject_out_of_i64_values(tmp_path):
    path = tmp_path / "overflow.trace.raw.zst"
    writer = RawLogWriter(path)
    writer.write_header(["$x"], ["entry"])

    with pytest.raises(OverflowError, match="OP_UNKNOWN_WRITE"):
        writer.record(OP_INITIAL_SCALAR, 0, 0, 0, 1 << 63, 0)
    with pytest.raises(OverflowError, match="OP_UNKNOWN_WRITE"):
        writer.record(OP_WRITE, 0, 10, 0, -(1 << 63) - 1, 0)

    # The explicit invalidation is representable and remains the only record.
    writer.record(
        OP_UNKNOWN_WRITE, 0, 10, 0, UNKNOWN_REASON_BIG_INT, 0)
    assert writer.finish() == 1
