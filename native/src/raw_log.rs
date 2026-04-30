//! Append-only raw trace log: one record per trace event.
//!
//! For expensive benchmarks (e.g. 2048-bit RSA) the in-memory dedup
//! `TraceAccumulator` blows up RAM because `iter_id` is unique per write —
//! dedup cannot collapse anything. This module streams every record to a
//! zstd-wrapped file instead; a separate Python pass merges the raw log
//! into the standard compact-v3 format.
//!
//! On-disk format:
//!   header (in frame 0):
//!     magic = b"SWRL"                4 bytes
//!     version = 1                    1 byte
//!     var_table_len (u32 LE)         4 bytes
//!     var_table = [len:u16, bytes]*  variable
//!     block_table_len (u32 LE)       4 bytes
//!     block_table = [len:u16, bytes]*variable
//!   records (repeated across frames 1..N):
//!     kind (u8)    1 byte  — `b'W'` write / `b'R'` read
//!     var_id (u32) 4 bytes
//!     pc (u32)     4 bytes
//!     block_id(u32)4 bytes
//!     value(i64)   8 bytes
//!     iter_id(u32) 4 bytes
//!   = 25 bytes/record
//!
//! All fields are little-endian. The byte stream is wrapped in one or
//! more zstd frames concatenated into a single file. Each frame is
//! 25-byte-record-aligned (frames are closed only on record boundaries)
//! so `raw_log_reader` can decompress them in parallel without records
//! straddling frame boundaries. Frame 0 is header-only; frames 1..N
//! carry records. The writer closes a frame every ~64 MiB
//! uncompressed, giving ~50 frames for a pkcs1-sized trace.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"SWRL";
pub const VERSION: u8 = 1;
pub const RECORD_SIZE: usize = 1 + 4 + 4 + 4 + 8 + 4;

/// Number of zstd encoder worker threads. Multi-worker encoding within
/// a SINGLE frame uses worker threads internally but still produces
/// one frame. Our parallelism comes from emitting multiple frames via
/// explicit frame boundaries (see `FRAME_FLUSH_BYTES`), so we set this
/// low — 4 is enough for the per-frame compression itself.
const ENCODER_WORKERS: u32 = 4;

/// Uncompressed bytes-per-frame threshold. After this many bytes are
/// written to the current frame, the writer closes it and starts a new
/// one. 64 MiB matches the Python writer's `FRAME_FLUSH_BYTES`.
const FRAME_FLUSH_BYTES: u64 = 64 * 1024 * 1024;

/// Streaming raw-log writer. Buffers records and flushes to a zstd-wrapped
/// file. Call `write_header` exactly once, then `record` as many times as
/// needed, then `finish`.
///
/// The writer emits a *multi-frame* zstd file: a new frame is started
/// every ``FRAME_FLUSH_BYTES`` uncompressed bytes, and frame 0 is
/// dedicated to the SWRL header. This lets the reader parallel-decode
/// frames 1..N. When the trace is smaller than `FRAME_FLUSH_BYTES`,
/// you get exactly 2 frames — the header-only frame 0 and a single
/// record frame — which still works with the parallel-or-sequential
/// reader.
pub struct RawLogWriter {
    /// Underlying file wrapped in a buffered writer. The zstd encoder
    /// is re-created on every frame boundary; we keep the raw BufWriter
    /// across frames so all frames go to the same file.
    inner: Option<BufWriter<File>>,
    /// Active encoder (``Some`` between frame-start and frame-close).
    enc: Option<zstd::stream::write::Encoder<'static, BufWriter<File>>>,
    /// Uncompressed bytes written to the current frame. Reset on
    /// ``close_frame``.
    frame_bytes: u64,
    /// Number of records appended so far.
    pub count: u64,
}

impl RawLogWriter {
    /// Open a new raw log at `path`. zstd compression level 3 matches what
    /// the Python-side `write_trace_binary_v2` uses.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let buf = BufWriter::with_capacity(4 * 1024 * 1024, file);
        let mut w = Self {
            inner: Some(buf),
            enc: None,
            frame_bytes: 0,
            count: 0,
        };
        w.start_new_frame()?;
        Ok(w)
    }

    /// Wrap the inner BufWriter in a fresh zstd encoder. Called at
    /// construction and after every ``close_frame``.
    fn start_new_frame(&mut self) -> std::io::Result<()> {
        debug_assert!(self.enc.is_none());
        let inner = self
            .inner
            .take()
            .expect("start_new_frame called without inner");
        let mut enc = zstd::stream::write::Encoder::new(inner, 3)?;
        let _ = enc.multithread(ENCODER_WORKERS);
        self.enc = Some(enc);
        self.frame_bytes = 0;
        Ok(())
    }

    /// Finalise the current frame (closes the zstd frame, writing the
    /// frame footer + checksum) and immediately start a new one so the
    /// caller can continue writing records.
    fn roll_frame(&mut self) -> std::io::Result<()> {
        let enc = self
            .enc
            .take()
            .expect("roll_frame called with no active encoder");
        let inner = enc.finish()?;
        self.inner = Some(inner);
        self.start_new_frame()
    }

    /// Write the header (magic + version + var/block name tables). Must be
    /// called once before any records. The header occupies frame 0
    /// exclusively — we close the frame immediately after the header so
    /// the reader can read the SWRL table by decompressing a tiny
    /// prefix of the file.
    pub fn write_header(
        &mut self,
        var_names: &[String],
        block_names: &[String],
    ) -> std::io::Result<()> {
        let enc = self.enc.as_mut().expect("writer uninitialised");
        enc.write_all(MAGIC)?;
        enc.write_all(&[VERSION])?;
        write_name_table(enc, var_names)?;
        write_name_table(enc, block_names)?;
        self.roll_frame()?;
        Ok(())
    }

    /// Append a single trace record.
    #[inline]
    pub fn record(
        &mut self,
        kind: u8,
        var_id: u32,
        pc: u32,
        block_id: u32,
        value: i64,
        iter_id: u32,
    ) -> std::io::Result<()> {
        let mut buf = [0u8; RECORD_SIZE];
        buf[0] = kind;
        buf[1..5].copy_from_slice(&var_id.to_le_bytes());
        buf[5..9].copy_from_slice(&pc.to_le_bytes());
        buf[9..13].copy_from_slice(&block_id.to_le_bytes());
        buf[13..21].copy_from_slice(&(value as u64).to_le_bytes());
        buf[21..25].copy_from_slice(&iter_id.to_le_bytes());
        {
            let enc = self.enc.as_mut().expect("writer uninitialised");
            enc.write_all(&buf)?;
        }
        self.count += 1;
        self.frame_bytes += RECORD_SIZE as u64;
        if self.frame_bytes >= FRAME_FLUSH_BYTES {
            self.roll_frame()?;
        }
        Ok(())
    }

    /// Flush the zstd encoder and close the underlying file. The
    /// reader's ``scan_frame_ranges`` filters out zero-content trailing
    /// frames that can appear if the last record happens to land
    /// exactly on a ``FRAME_FLUSH_BYTES`` boundary.
    pub fn finish(mut self) -> std::io::Result<u64> {
        let count = self.count;
        let enc = self.enc.take().expect("writer uninitialised at finish");
        let buf = enc.finish()?;
        let mut file = buf.into_inner().map_err(|e| e.into_error())?;
        file.flush()?;
        Ok(count)
    }
}

fn write_name_table<W: Write>(w: &mut W, names: &[String]) -> std::io::Result<()> {
    w.write_all(&(names.len() as u32).to_le_bytes())?;
    for name in names {
        let bytes = name.as_bytes();
        // Cap at u16::MAX — Boogie identifiers are always tiny so this is
        // just a format guard, never expected to trip.
        let len = bytes.len().min(u16::MAX as usize) as u16;
        w.write_all(&len.to_le_bytes())?;
        w.write_all(&bytes[..len as usize])?;
    }
    Ok(())
}
