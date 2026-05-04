class ReadBuffer:
    """
    A minimal in-memory byte buffer that supports
        • write(b) append bytes
        • read(n) consume & return up to n bytes
    It behaves like a very small subset of a real file/socket.
    """
    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes | bytearray | None = None):
        self._buf = bytearray(data or b"")
        self._pos = 0                 # next unread index

    # ---------- producer side ----------
    def write(self, data: bytes | bytearray) -> None:
        """Append *data* to the buffer."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("write() expects a bytes-like object")
        self._buf.extend(data)

    # ---------- consumer side ----------
    def read(self, n: int = -1) -> bytes:
        """
        Return up to *n* bytes and advance the read cursor.
        • n < 0  →  read the rest of the buffer.
        • n == 0 →  always returns b"".
        """
        if n == 0:
            return b""
        if n < 0:                     # “read all”
            start, self._pos = self._pos, len(self._buf)
            return bytes(self._buf[start:])

        # normal case
        end = min(self._pos + n, len(self._buf))
        start, self._pos = self._pos, end
        return bytes(self._buf[start:end])

    # ---------- helpers ----------
    def unread_bytes(self) -> int:
        """How many bytes are still waiting to be read?"""
        return len(self._buf) - self._pos

    def empty(self) -> bool:         # convenience alias
        return self.unread_bytes() == 0