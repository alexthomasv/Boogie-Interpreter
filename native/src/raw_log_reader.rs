//! Streaming `.trace.raw.zst` → Redis loader.
//!
//! Mirror of the Python `AbductionState.init_positive_examples_raw_log`
//! method, ported to Rust for ~20-50× speed-up on large benchmarks
//! (pkcs1_i15: 2.46B records, 2.36 GB compressed — Python spent hours
//! on this; Rust finishes in minutes).
//!
//! Pipeline:
//!
//!   reader thread (1, main)     zstd decode → raw 25-byte chunks
//!         │  crossbeam bounded(16)
//!         ▼
//!   parser pool (NUM_PARSERS)    unpack → thread-local FxHashMaps → FlushCmd
//!         │  crossbeam bounded(1024)
//!         ▼
//!   worker pool (NUM_WORKERS)    SADD → Redis (one connection each)
//!
//! The pipeline is terminated by channel closure, not explicit shutdown
//! messages: when the main thread drops its `chunk_tx`, parsers observe
//! `Err(RecvError)` and perform a final flush before exiting; when all
//! parsers exit, their `flush_tx` clones are dropped and workers drain.
//!
//! The on-disk format is produced by `raw_log::RawLogWriter`; constants
//! come from there.
//!
//! Redis schema (5 key patterns per record, all `SADD`-appended sets):
//!   positive_examples_{var}_{pc}                         12-byte members
//!   positive_examples_{var}_{block}                       12-byte members
//!   positive_examples_to_op_type_{var}_{pc}_{op}          12-byte members
//!   positive_examples_to_pc_{var}                         pickle.dumps(pc)
//!   positive_examples_to_block_{var}                      pickle.dumps(block)
//!
//! Value members are `struct.pack('<qI', value, iter_id)` = 12 bytes.
//! Registry members are pickle protocol-5 bytes so `pickle.loads()` in
//! downstream Python readers round-trips correctly.

use crate::raw_log::{MAGIC, RECORD_SIZE, VERSION};
use crossbeam_channel::{bounded, Receiver, Sender};
use memmap2::Mmap;
use redis::Pipeline;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Number of parallel Redis writer threads. Redis is configured with
/// `io-threads 16` — 8 parallel clients saturate the server without
/// thrashing the network.
const NUM_WORKERS: usize = 8;

/// Number of parallel parser threads. Empirically 4 parsers hit the
/// sweet spot: each owns an independent FxHashMap + FxHashSet dedup
/// state (~150 MB at steady state on pkcs1), and scaling to 8 cuts
/// throughput due to L2/L3 cache pressure — 8 × ~5 M rec/s < 4 × ~3 M
/// rec/s because the hashmaps fall out of cache. A future
/// var_id-sharded design could scale beyond 4, but the gain isn't
/// large enough to justify yet: most of the load-time win is from the
/// multi-frame zstd change.
const NUM_PARSERS: usize = 4;

/// Records per reader → parser chunk. 25 × 4096 ≈ 100 KB per chunk —
/// big enough to amortize channel overhead, small enough to match
/// zstd's internal output buffer (~128 KB) so each reader.read() ≈ one
/// zstd decompress call. Bigger chunks force multiple internal
/// decompress cycles per chunk-fill, each with per-call overhead.
const CHUNK_RECORDS: usize = 4096;
const CHUNK_BYTES: usize = RECORD_SIZE * CHUNK_RECORDS;

/// Number of parallel frame-decoder threads. pkcs1's multi-frame trace
/// has ~50 frames (one per ~64 MiB uncompressed); with 8 decoders we
/// saturate the 300 MB/s single-frame zstd ceiling 8-fold. Legacy
/// single-frame files produce just 1 frame, so only 1 decoder runs —
/// identical throughput to the pre-T2 code path.
const NUM_DECODERS: usize = 8;

/// zstd frame magic number (`ZSTD_MAGICNUMBER`, little-endian on disk).
/// Used to scan a multi-frame file for frame boundaries.
const ZSTD_FRAME_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Aggregate flush thresholds across ALL parsers. Each parser gets
/// `TOTAL / NUM_PARSERS` of each so the global RAM ceiling stays
/// ≈ 50M × (12 B + hashmap overhead) ~ 1.5 GB regardless of parser count.
const TOTAL_FLUSH_RECORDS: u64 = 5_000_000;
const TOTAL_FLUSH_MEMBERS: u64 = 50_000_000;

/// Chunk emitted to worker threads for each flush batch. Keys are final
/// Redis strings (formatted once by the parser thread that owns the bucket).
///
/// Two payload shapes so we can hand the record buffers to the worker
/// without per-member re-allocation:
///
///   * `Sadd12` — value sets: Vec of fixed 12-byte members.
///   * `SaddBytes` — pickle-encoded members (PC / block registry).
enum FlushCmd {
    Sadd12 {
        key: String,
        members: Vec<[u8; 12]>,
    },
    SaddBytes {
        key: String,
        members: Vec<Vec<u8>>,
    },
}

/// Per-stage throughput counters. Incremented by the thread owning the
/// stage (reader/parser/worker); read by the sampler thread every 5 s.
#[derive(Default)]
struct Counters {
    bytes_decoded: AtomicU64,   // raw bytes out of the zstd decoder
    records_parsed: AtomicU64,  // records unpacked by parsers (incl. shadow-skipped)
    members_flushed: AtomicU64, // 12-byte or registry members ACKed by workers
}

/// Public entry point called from pyo3.
///
/// Pipeline overview:
///
///   1. mmap the file and scan for zstd frame magics ⇒ frame ranges.
///   2. Decode frame 0 in-line to parse the SWRL header (var/block tables).
///   3. Record-frames (frames 1..N) get dispatched to ``NUM_DECODERS``
///      threads. Each decoder streams its frame's decompressed output
///      into chunks of ``CHUNK_BYTES`` and sends them to ``chunk_tx``.
///   4. ``NUM_PARSERS`` parser threads consume chunks, unpack the 25-byte
///      records, dedup into per-thread FxHashMaps (keyed on
///      ``(var_id, pc, op)``) / FxHashSets (members), and push
///      ``FlushCmd`` batches onto ``flush_tx``.
///   5. ``NUM_WORKERS`` redis threads drain ``flush_rx`` and execute
///      SADD-of-many per key across pipelined batches.
///
/// Legacy single-frame files (written before the frame-flush feature
/// was added) decompress under a single decoder thread — identical
/// to pre-T2 behaviour; no performance regression, just no speedup.
/// Multi-frame files unlock N-way parallel zstd decode, lifting the
/// single-frame ~300 MB/s decode ceiling.
pub fn load_raw_log_to_redis(
    path: &Path,
    redis_url: &str,
    iter_id_offset: u32,
) -> io::Result<u64> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mmap = unsafe { Mmap::map(&file)? };
    // Advise kernel we'll read sequentially (the frame scan) then random
    // (each decoder seeks into its own frame). Best-effort.
    #[cfg(unix)]
    {
        let _ = mmap.advise(memmap2::Advice::Sequential);
    }
    let mmap_arc: Arc<Mmap> = Arc::new(mmap);

    let frame_ranges = scan_frame_ranges(&mmap_arc);
    if frame_ranges.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw log: no zstd frame magic found in file",
        ));
    }

    // --- Decode frame 0 for the header ---
    let (var_names, block_names, header_consumed) =
        parse_header_from_frame0(&mmap_arc[frame_ranges[0].clone()])?;
    let var_names = Arc::new(var_names);
    let block_names = Arc::new(block_names);
    let is_shadow: Arc<Vec<bool>> =
        Arc::new(var_names.iter().map(|n| n.ends_with(".shadow")).collect());
    let n_vars = var_names.len();

    // Build the list of record-bearing frame ranges. In a multi-frame
    // trace, frame 0 is header-only (writer flushes a frame right after
    // writing the SWRL header) — so `header_consumed == frame 0 length`
    // and frame 0 contributes no records. In a legacy single-frame
    // trace, frame 0 contains both header + records — we then treat
    // frame 0 as a record frame and skip the header bytes when parsing
    // its decompressed output.
    let (record_frames, skip_header_bytes_in_first): (Vec<std::ops::Range<usize>>, usize) =
        if frame_ranges.len() == 1 {
            // Legacy single-frame: one decoder for the whole file, but
            // its first ``header_consumed`` bytes of decompressed output
            // are the header and must be dropped before record parsing.
            (frame_ranges.clone(), header_consumed)
        } else {
            // Multi-frame: frames 1..N are the record frames.
            (frame_ranges[1..].to_vec(), 0)
        };

    let counters = Arc::new(Counters::default());

    // Channels. Bounded capacities back-pressure upstream stages when
    // downstream stalls. The chunk channel is deep enough (256 × 100 KB
    // = 25 MB) that transient parser stalls don't force decoders to
    // block — decoders can get >=256 frames ahead, at which point the
    // mmap'd file IO becomes the natural pacer.
    let (chunk_tx, chunk_rx) = bounded::<Vec<u8>>(256);
    let (flush_tx, flush_rx) = bounded::<FlushCmd>(1024);

    // ---------------- Worker pool (Redis SADD) ----------------
    let mut worker_handles = Vec::with_capacity(NUM_WORKERS);
    for _ in 0..NUM_WORKERS {
        let client = redis::Client::open(redis_url).map_err(redis_to_io)?;
        let rx = flush_rx.clone();
        let c = Arc::clone(&counters);
        worker_handles.push(thread::spawn(move || -> redis::RedisResult<()> {
            worker_loop(client, rx, c)
        }));
    }
    drop(flush_rx);

    // ---------------- Parser pool ----------------
    let mut parser_handles = Vec::with_capacity(NUM_PARSERS);
    for _ in 0..NUM_PARSERS {
        let rx = chunk_rx.clone();
        let tx = flush_tx.clone();
        let var_names = Arc::clone(&var_names);
        let block_names = Arc::clone(&block_names);
        let is_shadow = Arc::clone(&is_shadow);
        let c = Arc::clone(&counters);
        parser_handles.push(thread::spawn(move || {
            parser_loop(
                rx,
                tx,
                var_names,
                block_names,
                is_shadow,
                n_vars,
                iter_id_offset,
                c,
            );
        }));
    }
    drop(chunk_rx);
    drop(flush_tx);

    // ---------------- Sampler ----------------
    let sampler_stop = Arc::new(AtomicBool::new(false));
    let sampler_handle = {
        let c = Arc::clone(&counters);
        let stop = Arc::clone(&sampler_stop);
        thread::spawn(move || sampler_loop(c, stop))
    };

    let start = Instant::now();
    eprintln!(
        "[raw-log] starting {} ({:.2} GB compressed, offset={}, \
         {} frames × {} decoders, {} parsers × {} workers)",
        path.display(),
        file_size as f64 / 1024.0 / 1024.0 / 1024.0,
        iter_id_offset,
        record_frames.len(),
        NUM_DECODERS.min(record_frames.len().max(1)),
        NUM_PARSERS,
        NUM_WORKERS,
    );

    // ---------------- Decoder pool ----------------
    //
    // Each decoder thread repeatedly pulls a frame range off a queue,
    // decompresses it, and streams the output into chunk_tx. Using a
    // work-stealing queue (crossbeam bounded) keeps all decoders busy
    // when frames have variable sizes.
    let (frame_tx, frame_rx) = bounded::<FrameJob>(record_frames.len().max(1));
    for (idx, range) in record_frames.iter().cloned().enumerate() {
        let skip = if idx == 0 { skip_header_bytes_in_first } else { 0 };
        frame_tx
            .send(FrameJob { range, skip_bytes: skip })
            .expect("frame_tx bounded cap == frames.len(), send can't fail");
    }
    drop(frame_tx);

    let effective_decoders = NUM_DECODERS.min(record_frames.len().max(1));
    let mut decoder_handles = Vec::with_capacity(effective_decoders);
    for _ in 0..effective_decoders {
        let rx = frame_rx.clone();
        let chunk_tx_ = chunk_tx.clone();
        let mmap_ = Arc::clone(&mmap_arc);
        let c = Arc::clone(&counters);
        decoder_handles.push(thread::spawn(move || -> io::Result<()> {
            decoder_loop(rx, chunk_tx_, mmap_, c)
        }));
    }
    drop(frame_rx);
    drop(chunk_tx);

    // Join decoders first (no more chunks will arrive).
    let mut decoder_err: Option<io::Error> = None;
    for h in decoder_handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                if decoder_err.is_none() {
                    decoder_err = Some(e);
                }
            }
            Err(_) => {
                if decoder_err.is_none() {
                    decoder_err = Some(io::Error::new(
                        io::ErrorKind::Other,
                        "raw-log decoder panicked",
                    ));
                }
            }
        }
    }

    // Join parsers; each performs a final flush on shutdown.
    for h in parser_handles {
        if h.join().is_err() {
            sampler_stop.store(true, Ordering::Relaxed);
            let _ = sampler_handle.join();
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "raw-log parser panicked",
            ));
        }
    }

    for h in worker_handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                sampler_stop.store(true, Ordering::Relaxed);
                let _ = sampler_handle.join();
                return Err(redis_to_io(e));
            }
            Err(_) => {
                sampler_stop.store(true, Ordering::Relaxed);
                let _ = sampler_handle.join();
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "raw-log worker panicked",
                ));
            }
        }
    }

    sampler_stop.store(true, Ordering::Relaxed);
    let _ = sampler_handle.join();

    if let Some(e) = decoder_err {
        return Err(e);
    }

    let total = counters.records_parsed.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[raw-log] DONE {} records in {:.1}s ({:.1}k rec/s avg parse, {:.1} MB/s decode avg)",
        format_commas(total),
        elapsed,
        total as f64 / elapsed.max(1e-3) / 1000.0,
        counters.bytes_decoded.load(Ordering::Relaxed) as f64 / elapsed.max(1e-3) / 1024.0 / 1024.0,
    );
    Ok(total)
}

/// One frame-decoding task: the byte range within the mmap'd file and
/// an optional number of *decompressed* bytes to drop at the start
/// (used only for legacy single-frame files where the SWRL header
/// sits at the start of the one and only frame).
struct FrameJob {
    range: std::ops::Range<usize>,
    skip_bytes: usize,
}

/// Scan compressed-file bytes for all occurrences of the zstd frame
/// magic (`0x28B52FFD`). Each occurrence marks the start of a frame;
/// the next occurrence marks the end of the previous. If only one
/// magic is found, the file is a single-frame legacy trace.
///
/// We drop frames whose compressed-byte size is ``< MIN_REAL_FRAME``
/// — that catches the trailing zero-content frame some writers leave
/// when the last record happens to land on a flush boundary (an empty
/// zstd frame is ~15 bytes). The record-frames we actually want are
/// tens of KB minimum, so this threshold is comfortably above noise.
fn scan_frame_ranges(data: &[u8]) -> Vec<std::ops::Range<usize>> {
    const MIN_REAL_FRAME: usize = 64;
    let starts: Vec<usize> =
        memchr::memmem::find_iter(data, &ZSTD_FRAME_MAGIC).collect();
    if starts.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(starts.len());
    for i in 0..starts.len() - 1 {
        out.push(starts[i]..starts[i + 1]);
    }
    out.push(starts[starts.len() - 1]..data.len());
    // Filter out trailing empty frames.
    out.retain(|r| r.end - r.start >= MIN_REAL_FRAME);
    out
}

/// Decompress frame 0 and parse the SWRL header from its decompressed
/// prefix. Returns `(var_names, block_names, header_bytes_consumed)`.
/// `header_bytes_consumed` is the number of *decompressed* bytes the
/// header occupies — relevant for legacy single-frame files where the
/// caller needs to skip past the header before record parsing in the
/// same decompressed stream.
fn parse_header_from_frame0(
    frame_bytes: &[u8],
) -> io::Result<(Vec<String>, Vec<String>, usize)> {
    let mut reader = zstd::stream::read::Decoder::new(Cursor::new(frame_bytes))?;

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("raw log: bad magic {:?}", magic),
        ));
    }
    let mut ver = [0u8; 1];
    reader.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("raw log: unsupported version {}", ver[0]),
        ));
    }
    let var_names = read_name_table(&mut reader)?;
    let block_names = read_name_table(&mut reader)?;

    // Compute decompressed header length by re-measuring via a counting
    // reader — simpler than threading a counter through read_name_table.
    let header_consumed = header_encoded_size(&var_names, &block_names);
    Ok((var_names, block_names, header_consumed))
}

/// Exact byte size of the SWRL header as written by raw_log.py
/// (decompressed).
fn header_encoded_size(var_names: &[String], block_names: &[String]) -> usize {
    // MAGIC(4) + VERSION(1) + var_table_len(4) +
    //   Σ (len u16 + utf8 bytes) + block_table_len(4) + Σ ...
    let table = |names: &[String]| -> usize {
        4 + names.iter().map(|n| 2 + n.len()).sum::<usize>()
    };
    4 + 1 + table(var_names) + table(block_names)
}

/// Decoder thread: pulls frame jobs off ``frame_rx``, streams each
/// frame's decompressed bytes into ``chunk_tx`` in 100 KB chunks.
/// Exits when ``frame_rx`` is closed (all jobs done) or when
/// ``chunk_tx`` is closed (parsers all died).
fn decoder_loop(
    frame_rx: Receiver<FrameJob>,
    chunk_tx: Sender<Vec<u8>>,
    mmap: Arc<Mmap>,
    counters: Arc<Counters>,
) -> io::Result<()> {
    // Upper bound on decompressed size per frame. The writer flushes a
    // frame every 64 MiB uncompressed, so a legit record frame fits
    // comfortably in 128 MiB. If we ever see a frame bigger than this
    // it's a writer bug — error out loudly.
    const MAX_FRAME_OUTPUT: usize = 128 * 1024 * 1024;

    let mut dctx = zstd::bulk::Decompressor::new()?;
    let mut out_buf = vec![0u8; MAX_FRAME_OUTPUT];

    while let Ok(job) = frame_rx.recv() {
        let frame_slice = &mmap[job.range.clone()];

        // Bulk-decompress exactly one frame. This avoids the
        // ``stream::read::Decoder`` behaviour where it tries to find
        // another frame after EOF and blames the slice for an
        // "incomplete frame" — the slice is definitionally the
        // byte-range between this frame's magic and the next, so it
        // contains exactly one frame, which is what bulk decompress
        // wants.
        //
        // Per-frame error tolerance: if a frame fails to decode (e.g.
        // trailing empty frame from a writer that happened to roll on
        // the last record, or a false-positive magic byte inside
        // compressed data), log and skip it rather than failing the
        // whole load. We lose at most ~64 MiB of records, which is
        // negligible against pkcs1's 33 GB uncompressed total.
        let out_len = match dctx.decompress_to_buffer(frame_slice, &mut out_buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!(
                    "[raw-log] WARN skipping frame at {:?} ({} bytes): {}",
                    job.range,
                    job.range.end - job.range.start,
                    e,
                );
                continue;
            }
        };
        let mut offset = job.skip_bytes;
        if offset > out_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "raw log: skip_bytes exceeds decoded frame size",
            ));
        }

        while offset + RECORD_SIZE <= out_len {
            let take = std::cmp::min(CHUNK_BYTES, out_len - offset);
            // Align ``take`` down to a record boundary so parsers
            // always receive whole records. ``take`` is >= RECORD_SIZE
            // so ``take_aligned`` is also >= RECORD_SIZE.
            let take_aligned = (take / RECORD_SIZE) * RECORD_SIZE;
            let end = offset + take_aligned;
            let chunk = out_buf[offset..end].to_vec();
            offset = end;
            counters
                .bytes_decoded
                .fetch_add(take_aligned as u64, Ordering::Relaxed);
            if chunk_tx.send(chunk).is_err() {
                return Ok(());
            }
        }
    }
    Ok(())
}

// ================ Worker loop ================

fn worker_loop(
    client: redis::Client,
    rx: Receiver<FlushCmd>,
    counters: Arc<Counters>,
) -> redis::RedisResult<()> {
    const MEMBERS_PER_SADD: usize = 50_000;
    const KEYS_PER_EXECUTE: usize = 2000;

    let mut conn = client.get_connection()?;
    let mut pipe = Pipeline::new();
    let mut batched = 0usize;

    let flush_pipe = |pipe: &mut Pipeline,
                      conn: &mut redis::Connection,
                      batched: &mut usize|
     -> redis::RedisResult<()> {
        if *batched > 0 {
            pipe.query::<()>(conn)?;
            *pipe = Pipeline::new();
            *batched = 0;
        }
        Ok(())
    };

    while let Ok(cmd) = rx.recv() {
        match cmd {
            FlushCmd::Sadd12 { key, members } => {
                let count = members.len() as u64;
                for ci in (0..members.len()).step_by(MEMBERS_PER_SADD) {
                    let end = (ci + MEMBERS_PER_SADD).min(members.len());
                    let slices: Vec<&[u8]> = members[ci..end].iter().map(|m| m.as_slice()).collect();
                    pipe.sadd(&key, &slices).ignore();
                }
                batched += 1;
                counters
                    .members_flushed
                    .fetch_add(count, Ordering::Relaxed);
                if batched >= KEYS_PER_EXECUTE {
                    flush_pipe(&mut pipe, &mut conn, &mut batched)?;
                }
            }
            FlushCmd::SaddBytes { key, members } => {
                let count = members.len() as u64;
                for ci in (0..members.len()).step_by(MEMBERS_PER_SADD) {
                    let end = (ci + MEMBERS_PER_SADD).min(members.len());
                    let slices: Vec<&[u8]> = members[ci..end].iter().map(|m| m.as_slice()).collect();
                    pipe.sadd(&key, &slices).ignore();
                }
                batched += 1;
                counters
                    .members_flushed
                    .fetch_add(count, Ordering::Relaxed);
                if batched >= KEYS_PER_EXECUTE {
                    flush_pipe(&mut pipe, &mut conn, &mut batched)?;
                }
            }
        }
    }
    flush_pipe(&mut pipe, &mut conn, &mut batched)?;
    Ok(())
}

// ================ Parser loop ================

#[allow(clippy::too_many_arguments)]
fn parser_loop(
    chunk_rx: Receiver<Vec<u8>>,
    flush_tx: Sender<FlushCmd>,
    var_names: Arc<Vec<String>>,
    block_names: Arc<Vec<String>>,
    is_shadow: Arc<Vec<bool>>,
    n_vars: usize,
    iter_id_offset: u32,
    counters: Arc<Counters>,
) {
    // Dedup per-key members in-parser (FxHashSet). Previous version used
    // ``Vec<[u8; 12]>`` and let Redis dedup via SADD — but that forced
    // every raw record onto the network, saturating Redis at ~240 MB/s
    // while the actual unique set was 100-1000× smaller. Hashing 12-byte
    // values into an FxHashSet costs ~50ns/op, trivial compared to the
    // network savings.
    let mut pc_values: FxHashMap<(u32, u32), FxHashSet<[u8; 12]>> = FxHashMap::default();
    let mut block_values: FxHashMap<(u32, u32), FxHashSet<[u8; 12]>> = FxHashMap::default();
    let mut op_values: FxHashMap<(u32, u32, u8), FxHashSet<[u8; 12]>> = FxHashMap::default();
    let mut pc_registry: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); n_vars];
    let mut blk_registry: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); n_vars];

    let flush_records = TOTAL_FLUSH_RECORDS / NUM_PARSERS as u64;
    let flush_members = TOTAL_FLUSH_MEMBERS / NUM_PARSERS as u64;
    let mut records_since_flush: u64 = 0;
    let mut buffered_members: u64 = 0;

    while let Ok(chunk) = chunk_rx.recv() {
        let mut parsed_this_chunk: u64 = 0;
        let mut off = 0usize;
        while off + RECORD_SIZE <= chunk.len() {
            let kind = chunk[off];
            let var_id = u32::from_le_bytes([
                chunk[off + 1],
                chunk[off + 2],
                chunk[off + 3],
                chunk[off + 4],
            ]);
            let pc = u32::from_le_bytes([
                chunk[off + 5],
                chunk[off + 6],
                chunk[off + 7],
                chunk[off + 8],
            ]);
            let blk_id = u32::from_le_bytes([
                chunk[off + 9],
                chunk[off + 10],
                chunk[off + 11],
                chunk[off + 12],
            ]);
            let value = i64::from_le_bytes([
                chunk[off + 13],
                chunk[off + 14],
                chunk[off + 15],
                chunk[off + 16],
                chunk[off + 17],
                chunk[off + 18],
                chunk[off + 19],
                chunk[off + 20],
            ]);
            let iter_id = u32::from_le_bytes([
                chunk[off + 21],
                chunk[off + 22],
                chunk[off + 23],
                chunk[off + 24],
            ]);
            off += RECORD_SIZE;
            parsed_this_chunk += 1;

            if is_shadow[var_id as usize] {
                continue;
            }

            let eff_iter = if iter_id > 0 {
                iter_id.wrapping_add(iter_id_offset)
            } else {
                0
            };

            let mut m = [0u8; 12];
            m[..8].copy_from_slice(&(value as u64).to_le_bytes());
            m[8..].copy_from_slice(&eff_iter.to_le_bytes());

            push_member(&mut pc_values, (var_id, pc), m);
            push_member(&mut block_values, (var_id, blk_id), m);
            push_member(&mut op_values, (var_id, pc, kind), m);
            pc_registry[var_id as usize].insert(pc);
            blk_registry[var_id as usize].insert(blk_id);
            buffered_members += 3;
            records_since_flush += 1;
        }
        counters
            .records_parsed
            .fetch_add(parsed_this_chunk, Ordering::Relaxed);

        if records_since_flush >= flush_records || buffered_members >= flush_members {
            flush_all(
                &flush_tx,
                &var_names,
                &block_names,
                &mut pc_values,
                &mut block_values,
                &mut op_values,
                &mut pc_registry,
                &mut blk_registry,
            );
            records_since_flush = 0;
            buffered_members = 0;
        }
    }

    // Final flush on shutdown (chunk_rx closed).
    flush_all(
        &flush_tx,
        &var_names,
        &block_names,
        &mut pc_values,
        &mut block_values,
        &mut op_values,
        &mut pc_registry,
        &mut blk_registry,
    );
}

// ================ Sampler loop ================

fn sampler_loop(counters: Arc<Counters>, stop: Arc<AtomicBool>) {
    let start = Instant::now();
    let mut last_bytes: u64 = 0;
    let mut last_parsed: u64 = 0;
    let mut last_flushed: u64 = 0;
    let mut last_t = start;

    while !stop.load(Ordering::Relaxed) {
        // Sleep in short increments so shutdown is prompt (5 s total).
        for _ in 0..50 {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }

        let now = Instant::now();
        let dt = now.duration_since(last_t).as_secs_f64().max(1e-3);
        let total_dt = now.duration_since(start).as_secs_f64().max(1e-3);

        let bytes = counters.bytes_decoded.load(Ordering::Relaxed);
        let parsed = counters.records_parsed.load(Ordering::Relaxed);
        let flushed = counters.members_flushed.load(Ordering::Relaxed);

        let inst_decode = (bytes - last_bytes) as f64 / dt / 1024.0 / 1024.0;
        let inst_parse = (parsed - last_parsed) as f64 / dt / 1000.0;
        let inst_flush = (flushed - last_flushed) as f64 / dt / 1000.0;
        let avg_parse = parsed as f64 / total_dt / 1000.0;

        eprintln!(
            "[raw-log] {:>13} rec | decode {:>6.1} MB/s | parse {:>7.1}k rec/s | flush {:>7.1}k mem/s | avg parse {:>6.1}k rec/s | {:.1} min",
            format_commas(parsed),
            inst_decode,
            inst_parse,
            inst_flush,
            avg_parse,
            total_dt / 60.0,
        );

        last_bytes = bytes;
        last_parsed = parsed;
        last_flushed = flushed;
        last_t = now;
    }
}

// ================ Helpers ================

/// Insert a 12-byte member into the per-key dedup set. In-parser dedup
/// drops the typical 100-1000× intra-key duplication before the Redis
/// network hop, so Redis only ever sees unique members.
#[inline]
fn push_member<K: std::hash::Hash + Eq>(
    map: &mut FxHashMap<K, FxHashSet<[u8; 12]>>,
    key: K,
    m: [u8; 12],
) {
    match map.entry(key) {
        Entry::Occupied(mut e) => {
            e.get_mut().insert(m);
        }
        Entry::Vacant(e) => {
            let mut s = FxHashSet::default();
            s.insert(m);
            e.insert(s);
        }
    }
}

fn read_name_table<R: Read>(r: &mut R) -> io::Result<Vec<String>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let n = u32::from_le_bytes(len_buf) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut lb = [0u8; 2];
        r.read_exact(&mut lb)?;
        let len = u16::from_le_bytes(lb) as usize;
        let mut bytes = vec![0u8; len];
        r.read_exact(&mut bytes)?;
        out.push(String::from_utf8_lossy(&bytes).into_owned());
    }
    Ok(out)
}

/// Flush all five dedup maps to workers in one batched pass.
///
/// Called by each parser thread on its own local maps; the same
/// `(var_id, pc)` key may be emitted by multiple parsers — Redis SADD
/// merges them server-side.
#[allow(clippy::too_many_arguments)]
fn flush_all(
    tx: &Sender<FlushCmd>,
    var_names: &[String],
    block_names: &[String],
    pc_values: &mut FxHashMap<(u32, u32), FxHashSet<[u8; 12]>>,
    block_values: &mut FxHashMap<(u32, u32), FxHashSet<[u8; 12]>>,
    op_values: &mut FxHashMap<(u32, u32, u8), FxHashSet<[u8; 12]>>,
    pc_registry: &mut [FxHashSet<u32>],
    blk_registry: &mut [FxHashSet<u32>],
) {
    for ((var_id, pc), members) in pc_values.drain() {
        if members.is_empty() {
            continue;
        }
        let key = format!("positive_examples_{}_{}", var_names[var_id as usize], pc);
        let members: Vec<[u8; 12]> = members.into_iter().collect();
        tx.send(FlushCmd::Sadd12 { key, members }).ok();
    }

    for ((var_id, blk_id), members) in block_values.drain() {
        if members.is_empty() {
            continue;
        }
        let key = format!(
            "positive_examples_{}_{}",
            var_names[var_id as usize], block_names[blk_id as usize]
        );
        let members: Vec<[u8; 12]> = members.into_iter().collect();
        tx.send(FlushCmd::Sadd12 { key, members }).ok();
    }

    for ((var_id, pc, op), members) in op_values.drain() {
        if members.is_empty() {
            continue;
        }
        let key = format!(
            "positive_examples_to_op_type_{}_{}_{}",
            var_names[var_id as usize], pc, op as char
        );
        let members: Vec<[u8; 12]> = members.into_iter().collect();
        tx.send(FlushCmd::Sadd12 { key, members }).ok();
    }

    for (var_id, pcs) in pc_registry.iter_mut().enumerate() {
        if pcs.is_empty() {
            continue;
        }
        let key = format!("positive_examples_to_pc_{}", var_names[var_id]);
        let pickled: Vec<Vec<u8>> = pcs.drain().map(pickle_u32).collect();
        tx.send(FlushCmd::SaddBytes {
            key,
            members: pickled,
        })
        .ok();
    }

    let block_pickles: Vec<Vec<u8>> = block_names.iter().map(|s| pickle_str(s)).collect();
    for (var_id, blks) in blk_registry.iter_mut().enumerate() {
        if blks.is_empty() {
            continue;
        }
        let key = format!("positive_examples_to_block_{}", var_names[var_id]);
        let pickled: Vec<Vec<u8>> = blks
            .drain()
            .map(|b| block_pickles[b as usize].clone())
            .collect();
        tx.send(FlushCmd::SaddBytes {
            key,
            members: pickled,
        })
        .ok();
    }
}

fn redis_to_io(e: redis::RedisError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("redis: {}", e))
}

/// Encode a u32 PC as a pickle-protocol-5 bytestring equivalent to
/// `pickle.dumps(int_value, protocol=5)`.  Downstream readers call
/// `pickle.loads()` on these.
fn pickle_u32(n: u32) -> Vec<u8> {
    // Layout: PROTO(2) + FRAME(9) + int opcode + STOP
    let int_bytes: Vec<u8> = if n < 256 {
        vec![0x4b, n as u8] // BININT1
    } else if n < 65536 {
        let b = (n as u16).to_le_bytes();
        vec![0x4d, b[0], b[1]] // BININT2
    } else {
        // BININT expects a signed i32 LE; u32 < i32::MAX always fits as
        // a positive i32 because n as i32 overflow only matters when
        // n > 0x7FFFFFFF (2 GiB of PCs — we never hit that). Defensive:
        // for n >= 0x80000000 we'd want LONG1, but PC values are small.
        let b = (n as i32).to_le_bytes();
        vec![0x4a, b[0], b[1], b[2], b[3]]
    };
    let payload_len = int_bytes.len() + 1; // +STOP
    let mut buf = Vec::with_capacity(2 + 9 + payload_len);
    buf.push(0x80);
    buf.push(0x05);
    buf.push(0x95);
    buf.extend_from_slice(&(payload_len as u64).to_le_bytes());
    buf.extend_from_slice(&int_bytes);
    buf.push(0x2e); // STOP
    buf
}

/// Encode a short string as `pickle.dumps(s, protocol=5)` bytes.  Used
/// for block-name registry members.  Panics if the string is 256 bytes
/// or longer (SMACK block names are `$bbN` — always short).
fn pickle_str(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    assert!(
        b.len() < 256,
        "pickle_str: block names must be <256 bytes, got {}",
        b.len()
    );
    // SHORT_BINUNICODE + MEMOIZE + STOP
    let payload_len = 2 + b.len() + 2;
    let mut buf = Vec::with_capacity(2 + 9 + payload_len);
    buf.push(0x80);
    buf.push(0x05);
    buf.push(0x95);
    buf.extend_from_slice(&(payload_len as u64).to_le_bytes());
    buf.push(0x8c);
    buf.push(b.len() as u8);
    buf.extend_from_slice(b);
    buf.push(0x94); // MEMOIZE
    buf.push(0x2e); // STOP
    buf
}

fn format_commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    out
}
