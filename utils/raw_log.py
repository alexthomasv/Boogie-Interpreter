"""Streaming `.trace.raw.zst` writer — the canonical trace format.

On-disk layout (little-endian throughout):

  Header (in frame 0):
    magic         = b"SWRL"           4 bytes
    version       = 1                 1 byte
    var_table_len (u32)               4 bytes
    var_table = [len:u16, bytes]*     variable
    block_table_len (u32)             4 bytes
    block_table = [len:u16, bytes]*   variable

  Record (repeated across all frames):
    kind (u8)    1 byte    — b'W' write / b'R' read
    var_id (u32) 4 bytes   — index into var_table
    pc (u32)     4 bytes
    block_id(u32)4 bytes   — index into block_table
    value(i64)   8 bytes
    iter_id(u32) 4 bytes
  = 25 bytes/record

The byte stream is wrapped in one or more zstd frames, all concatenated
into a single .raw.zst file. Each frame is 25-byte-record-aligned (the
writer only closes a frame on record boundaries), so the reader can
decompress frames independently and concatenate the outputs without
worrying about records straddling frame boundaries.

Multi-frame output unlocks N-way parallel decompression on the reader
side — the writer flushes a new frame every ``FRAME_FLUSH_BYTES``
uncompressed bytes (default 64 MiB), which yields ~50 frames for a
pkcs1-sized trace. Legacy single-frame files are backwards-compatible
(the reader just sees 1 frame).

This is the only trace format; everything else (compact pickle, compact
binary v1/v2/v3) was removed.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

import zstandard as zstd


MAGIC = b"SWRL"
VERSION = 1
RECORD_SIZE = 1 + 4 + 4 + 4 + 8 + 4

# Uncompressed-bytes threshold at which we close the current zstd frame
# and start a new one. ~64 MiB gives ~50 frames for a pkcs1 trace, which
# is enough to saturate an 8-core parallel decoder without making the
# file index table bloated. Bigger frames reduce parallelism; smaller
# frames add per-frame zstd init overhead.
FRAME_FLUSH_BYTES = 64 * 1024 * 1024


class RawLogWriter:
    """Streaming raw-log writer.

    Usage:

        w = RawLogWriter(path)
        w.write_header(var_names, block_names)  # var_id / block_id are indices here
        for ... :
            w.record(kind, var_id, pc, block_id, value, iter_id)
        w.finish()

    Variable / block ids are caller-assigned u32 indices into the tables
    passed to ``write_header``.  For Python runs the caller builds those
    tables on the fly as new names appear (see
    ``Environment.enable_raw_log``).
    """

    def __init__(self, path: Path | str, level: int = 3):
        self._fh = open(path, "wb")
        self._cctx = zstd.ZstdCompressor(level=level, threads=-1)
        self._writer = self._cctx.stream_writer(self._fh)
        self._buf = bytearray()
        self._flush_threshold = 1 << 20  # 1 MiB
        self.count = 0
        self._header_written = False
        # Uncompressed bytes written to the current zstd frame. When this
        # crosses ``FRAME_FLUSH_BYTES`` we close the frame and start a
        # new one so the reader can parallel-decompress.
        self._frame_bytes = 0

    def write_header(self, var_names: list[str], block_names: list[str]) -> None:
        """Serialise the var + block tables.  Must be called exactly once
        before the first ``record``.

        The header goes into frame 0. We then immediately close that
        frame so frame 0 contains ONLY the header — that way the
        reader can parse the header by decompressing just frame 0 and
        parallel-decode record frames without them needing to know
        anything about the header layout. Uncompressed-size impact:
        ~1 MB of table data vs the gigabytes of record data, so
        dedicating a frame to it is nearly free.
        """
        assert not self._header_written
        self._header_written = True
        self._writer.write(MAGIC)
        self._writer.write(bytes([VERSION]))
        _write_name_table(self._writer, var_names)
        _write_name_table(self._writer, block_names)
        # Close the header-only frame so records start at frame 1.
        self._writer.flush(zstd.FLUSH_FRAME)
        self._frame_bytes = 0

    def record(self, kind: int, var_id: int, pc: int, block_id: int,
               value: int, iter_id: int) -> None:
        """Append one record.  Values outside signed-i64 range are wrapped
        modulo 2**64 to preserve C-like semantics."""
        INT64_MIN = -(1 << 63)
        INT64_MAX = (1 << 63) - 1
        if value < INT64_MIN or value > INT64_MAX:
            value = ((value + (1 << 63)) % (1 << 64)) - (1 << 63)
        self._buf.extend(struct.pack('<BIIIqI', kind & 0xFF, var_id, pc,
                                     block_id, value, iter_id))
        self.count += 1
        if len(self._buf) >= self._flush_threshold:
            self._flush()

    def _flush(self) -> None:
        """Flush pending records to the current zstd frame. Also closes
        the frame and starts a new one when the per-frame uncompressed
        size crosses ``FRAME_FLUSH_BYTES``. Frame boundaries are always
        record-aligned because this is the only path from ``self._buf``
        to the compressor and ``self._buf`` only grows by full records.
        """
        if self._buf:
            nbytes = len(self._buf)
            self._writer.write(bytes(self._buf))
            self._buf.clear()
            self._frame_bytes += nbytes
            if self._frame_bytes >= FRAME_FLUSH_BYTES:
                self._writer.flush(zstd.FLUSH_FRAME)
                self._frame_bytes = 0

    def finish(self) -> int:
        """Flush the zstd frame, close the file, return the record count.

        ``zstd.stream_writer.close()`` also closes the underlying file,
        so we only fsync it if still open.
        """
        self._flush()
        self._writer.flush(zstd.FLUSH_FRAME)
        self._writer.close()
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        return self.count


def _write_name_table(writer, names: list[str]) -> None:
    writer.write(struct.pack('<I', len(names)))
    for name in names:
        b = name.encode('utf-8')
        # Cap defensively — Boogie names are tiny, so this is just a format guard.
        if len(b) > 0xFFFF:
            b = b[:0xFFFF]
        writer.write(struct.pack('<H', len(b)))
        writer.write(b)
