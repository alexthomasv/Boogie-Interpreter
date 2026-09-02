//! Native `.trace.raw.zst` -> SQLite trace-evidence index builder.
//!
//! The hot path uses sorted intermediate runs:
//! raw trace -> parser-local dedup maps -> sorted run files -> k-way merge ->
//! one SQLite insert per final evidence key. This avoids the expensive
//! SQLite read/modify/write loop that dominated large traces.

use crate::raw_log::{
    MAGIC, OP_INITIAL_SCALAR, OP_ITER_CONTEXT, OP_PRE_PC, OP_READ, OP_UNKNOWN_WRITE, OP_WRITE,
    RECORD_SIZE, VERSION,
};
use crossbeam_channel::{bounded, Receiver, Sender};
use memmap2::Mmap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ffi::CStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// v4 retains the v3 aggregate evidence and adds exact, physically ordered PRE
// visits and scalar S/W/U timelines.  Sequence numbers are reconstructed from
// frame/record position; the raw record stays compact and unchanged.
// SWRL v2 inputs are intentionally rejected: publishing a v4 SQLite index from
// an aggregate-only raw stream could mislabel missing PRE visits as authority.
// "5": both lanes of the composed program are indexed ("4" dropped every
// `.shadow` record).  Must match `TRACE_INDEX_VERSION` in
// `src/state/trace_evidence.py`.
const TRACE_INDEX_VERSION: &str = "5";
const ORDERED_TRACE_VERSION: &str = "1";

const NUM_DECODERS: usize = 8;
const NUM_PARSERS: usize = 4;
const CHUNK_RECORDS: usize = 4096;
const CHUNK_BYTES: usize = RECORD_SIZE * CHUNK_RECORDS;
const MAX_FRAME_OUTPUT: usize = 128 * 1024 * 1024;

const DEFAULT_PARSER_FLUSH_MEMBERS: u64 = 12_500_000;
const DEFAULT_MAX_MERGE_FAN_IN: usize = 64;
const MERGE_FD_HEADROOM: u64 = 32;
const RUN_MAGIC: &[u8; 4] = b"SWIR";
const RUN_VERSION: u8 = 1;
const RUN_KEY_SIZE: usize = 10;

const KIND_PC: u8 = 0;
const KIND_BLOCK: u8 = 1;
const KIND_OP: u8 = 2;
const ORDERED_VISIT_MEMBER_SIZE: usize = 8;
const ORDERED_SCALAR_MEMBER_SIZE: usize = 8 + 1 + 4 + 8;

#[derive(Debug)]
pub struct BuildResult {
    pub path: String,
    pub rows: u64,
    pub records: u64,
    pub raw_files: u64,
    pub skipped_shadow: u64,
    pub contexts: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct IndexKey {
    kind: u8,
    var_id: u32,
    loc: u32,
    op: u8,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct GlobalKey {
    kind: u8,
    var_id: u32,
    loc: u32,
    op: u8,
}

#[derive(Clone, Copy)]
struct IndexKinds {
    pc: bool,
    block: bool,
    op: bool,
}

impl IndexKinds {
    fn from_env() -> io::Result<Self> {
        let raw = match std::env::var("SWOOSH_TRACE_INDEX_KINDS") {
            Ok(value) if !value.trim().is_empty() => value,
            _ => {
                return Ok(Self {
                    pc: true,
                    block: true,
                    op: true,
                })
            }
        };
        let mut kinds = Self {
            pc: false,
            block: false,
            op: false,
        };
        for part in raw.split(',') {
            match part.trim().to_ascii_lowercase().as_str() {
                "" => {}
                "pc" | "pcs" => kinds.pc = true,
                "block" | "blocks" => kinds.block = true,
                "op" | "ops" => kinds.op = true,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unknown SWOOSH_TRACE_INDEX_KINDS entry {other:?}"),
                    ))
                }
            }
        }
        if !kinds.pc && !kinds.block && !kinds.op {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SWOOSH_TRACE_INDEX_KINDS enabled no index kinds",
            ));
        }
        Ok(kinds)
    }

    fn per_record_members(self) -> u64 {
        self.pc as u64 + self.block as u64 + self.op as u64
    }
}

struct ParsedBatch {
    var_names: Arc<Vec<String>>,
    block_names: Arc<Vec<String>>,
    values: FxHashMap<IndexKey, FxHashSet<[u8; 12]>>,
    contexts: Vec<LocalContextDef>,
}

enum ParserOutput {
    Aggregate(ParsedBatch),
    Ordered(OrderedBatch),
}

struct DecodedChunk {
    frame_ordinal: u32,
    first_record_ordinal: u32,
    bytes: Vec<u8>,
}

struct OrderedBatch {
    execution_id: u32,
    var_names: Arc<Vec<String>>,
    visits: Vec<OrderedVisitChunk>,
    scalar_timelines: Vec<OrderedScalarChunk>,
}

struct OrderedVisitChunk {
    pc: u32,
    first_event_seq: u64,
    last_event_seq: u64,
    event_count: u32,
    events: Vec<u8>,
}

struct OrderedScalarChunk {
    var_id: u32,
    first_event_seq: u64,
    last_event_seq: u64,
    event_count: u32,
    events: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct LocalContextDef {
    iter_id: u32,
    parent_iter_id: u32,
    depth: u32,
    header_block_id: u32,
    iter_count: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GlobalContextDef {
    iter_id: u32,
    parent_iter_id: u32,
    depth: u32,
    header_id: u32,
    iter_count: i64,
}

#[derive(Default)]
struct Counters {
    bytes_decoded: AtomicU64,
    records_parsed: AtomicU64,
    records_indexed: AtomicU64,
    contexts_indexed: AtomicU64,
    runs_written: AtomicU64,
    run_groups_written: AtomicU64,
    rows_written: AtomicU64,
}

struct SqlitePublication {
    target: PathBuf,
    staging: PathBuf,
    published: bool,
}

impl SqlitePublication {
    fn new(target: &Path) -> io::Result<Self> {
        let parent = target
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let file_name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "trace-index SQLite output must have a file name",
                )
            })?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let staging = parent.join(format!(
            ".{file_name}.partial.{}.{stamp}",
            std::process::id()
        ));
        Ok(Self {
            target: target.to_path_buf(),
            staging,
            published: false,
        })
    }

    fn publish(&mut self) -> io::Result<()> {
        fs::rename(&self.staging, &self.target)?;
        self.published = true;
        Ok(())
    }
}

impl Drop for SqlitePublication {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.staging);
        }
    }
}

pub fn build_trace_index_sqlite(
    raw_paths: &[PathBuf],
    sqlite_path: &Path,
    bench: Option<&str>,
) -> io::Result<BuildResult> {
    if raw_paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no raw trace paths supplied",
        ));
    }
    if let Some(parent) = sqlite_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let kinds = IndexKinds::from_env()?;
    let flush_members = parser_flush_members();
    let keep_runs = keep_run_files();
    let counters = Arc::new(Counters::default());
    let sampler_stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let counters = Arc::clone(&counters);
        let stop = Arc::clone(&sampler_stop);
        thread::spawn(move || sampler_loop(counters, stop))
    };

    let started = Instant::now();
    let mut publication = SqlitePublication::new(sqlite_path)?;
    let mut run_writer = RunFileWriter::create(&publication.staging)?;
    let mut sqlite = SqliteTraceIndexWriter::create(&publication.staging, bench)?;
    eprintln!(
        "[trace-index] building {} from {} raw trace file(s) via sorted runs",
        sqlite_path.display(),
        raw_paths.len()
    );

    let build_result: io::Result<BuildResult> = (|| {
        // Each raw file is one complete input execution. We assign every
        // execution a disjoint slice of the iter-id space so observations from
        // different executions never share a co-occurrence bucket:
        //   iter_base            -> this execution's straight-line (iter_id==0) id
        //   iter_base + k (k>=1) -> this execution's loop-iteration k
        // The execution therefore occupies [iter_base, iter_base + max_raw_iter];
        // the next execution starts one past that. Reserving a NON-ZERO id for
        // the straight-line records is the fix for the cross-trace fabrication
        // bug: previously every execution's iter_id==0 records collapsed into a
        // single shared bucket, so a tuple at a non-loop pc could pair var X from
        // one input run with var Y from another.
        let mut next_used: u32 = 0;
        for (path_index, path) in raw_paths.iter().enumerate() {
            let execution_id = u32::try_from(path_index + 1).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "trace execution id space exhausted",
                )
            })?;
            let iter_base = next_used.checked_add(1).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "trace iteration id space exhausted",
                )
            })?;
            let max_raw_iter = process_raw_file(
                path,
                execution_id,
                iter_base,
                kinds,
                flush_members,
                &mut run_writer,
                &mut sqlite,
                Arc::clone(&counters),
            )?;
            next_used = iter_base.checked_add(max_raw_iter).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "trace iteration context id offset overflow",
                )
            })?;
        }

        let records = counters.records_indexed.load(Ordering::Relaxed);
        let merge_fan_in = merge_fan_in()?;
        eprintln!(
            "[trace-index] merging {} sorted run file(s) with fan-in {}",
            format_commas(run_writer.run_paths.len() as u64),
            merge_fan_in,
        );
        let final_run_paths = reduce_runs_to_fan_in(
            &run_writer.run_paths,
            &run_writer.dir,
            merge_fan_in,
            !keep_runs,
        )?;
        sqlite.set_meta("source", "raw_log")?;
        sqlite.set_meta("builder", "native_sqlite_sorted_runs")?;
        sqlite.set_meta("raw_files", raw_paths.len().to_string().as_str())?;
        sqlite.set_meta("records", records.to_string().as_str())?;
        // Both lanes index; the key stays for the index meta contract.
        sqlite.set_meta("skipped_shadow", "0")?;
        sqlite.set_meta(
            "contexts",
            counters
                .contexts_indexed
                .load(Ordering::Relaxed)
                .to_string()
                .as_str(),
        )?;
        sqlite.set_meta("index_kinds", kinds.meta_value())?;

        let rows = merge_runs_into_sqlite(
            &final_run_paths,
            &run_writer.names,
            &mut sqlite,
            Arc::clone(&counters),
        )?;
        if !keep_runs {
            remove_run_files(&final_run_paths)?;
        }
        sqlite.insert_contexts(&run_writer.contexts, &run_writer.names)?;
        let rows = sqlite.finish(rows)?;

        Ok(BuildResult {
            path: sqlite_path.display().to_string(),
            rows,
            records,
            raw_files: raw_paths.len() as u64,
            skipped_shadow: 0,
            contexts: counters.contexts_indexed.load(Ordering::Relaxed),
        })
    })();

    if !keep_runs {
        let _ = run_writer.cleanup();
    }

    sampler_stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();

    let result = build_result?;
    publication.publish()?;
    let elapsed = started.elapsed().as_secs_f64().max(1e-3);
    eprintln!(
        "[trace-index] DONE {} rows, {} records in {:.1}s ({:.1}k rec/s)",
        format_commas(result.rows),
        format_commas(result.records),
        elapsed,
        result.records as f64 / elapsed / 1000.0
    );
    Ok(result)
}

impl IndexKinds {
    fn meta_value(self) -> &'static str {
        match (self.pc, self.block, self.op) {
            (true, true, true) => "pc,block,op",
            (true, true, false) => "pc,block",
            (true, false, true) => "pc,op",
            (false, true, true) => "block,op",
            (true, false, false) => "pc",
            (false, true, false) => "block",
            (false, false, true) => "op",
            (false, false, false) => "",
        }
    }
}

fn parser_flush_members() -> u64 {
    std::env::var("SWOOSH_TRACE_INDEX_RUN_MEMBERS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PARSER_FLUSH_MEMBERS)
}

fn keep_run_files() -> bool {
    matches!(
        std::env::var("SWOOSH_TRACE_INDEX_KEEP_RUNS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn merge_fan_in() -> io::Result<usize> {
    let soft_limit = soft_nofile_limit()?;
    let configured = match std::env::var("SWOOSH_TRACE_INDEX_MERGE_FAN_IN") {
        Ok(raw) => raw
            .parse::<usize>()
            .ok()
            .filter(|value| *value >= 2)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "SWOOSH_TRACE_INDEX_MERGE_FAN_IN must be an integer of at least 2",
                )
            })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_MAX_MERGE_FAN_IN,
        Err(error) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid SWOOSH_TRACE_INDEX_MERGE_FAN_IN: {error}"),
            ))
        }
    };
    merge_fan_in_for_limit(soft_limit, configured)
}

fn merge_fan_in_for_limit(soft_limit: u64, configured: usize) -> io::Result<usize> {
    let safe_limit = soft_limit.saturating_sub(MERGE_FD_HEADROOM);
    if safe_limit < 2 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "RLIMIT_NOFILE soft limit {soft_limit} leaves fewer than two merge inputs \
                 after reserving {MERGE_FD_HEADROOM} descriptors"
            ),
        ));
    }
    let safe_limit = usize::try_from(safe_limit).unwrap_or(usize::MAX);
    Ok(configured.min(safe_limit))
}

#[cfg(unix)]
fn soft_nofile_limit() -> io::Result<u64> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let result = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    if limit.rlim_cur == libc::RLIM_INFINITY {
        return Ok(u64::MAX);
    }
    Ok(limit.rlim_cur as u64)
}

#[cfg(not(unix))]
fn soft_nofile_limit() -> io::Result<u64> {
    Ok((DEFAULT_MAX_MERGE_FAN_IN as u64) + MERGE_FD_HEADROOM)
}

fn process_raw_file(
    path: &Path,
    execution_id: u32,
    iter_base: u32,
    kinds: IndexKinds,
    flush_members: u64,
    run_writer: &mut RunFileWriter,
    sqlite: &mut SqliteTraceIndexWriter,
    counters: Arc<Counters>,
) -> io::Result<u32> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mmap = unsafe { Mmap::map(&file)? };
    #[cfg(unix)]
    {
        let _ = mmap.advise(memmap2::Advice::Sequential);
    }
    let mmap = Arc::new(mmap);
    let frame_ranges = scan_frame_ranges_exact(&mmap)?;
    if frame_ranges.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: no zstd frames", path.display()),
        ));
    }

    let (var_names, block_names) = parse_header_from_frame0(&mmap[frame_ranges[0].clone()])?;
    let var_names = Arc::new(var_names);
    let block_names = Arc::new(block_names);

    let record_frames: Vec<FrameJob> = frame_ranges
        .into_iter()
        .enumerate()
        .skip(1)
        .map(|(frame_ordinal, range)| {
            let frame_ordinal = u32::try_from(frame_ordinal).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "too many raw trace frames")
            })?;
            Ok(FrameJob {
                frame_ordinal,
                range,
            })
        })
        .collect::<io::Result<_>>()?;

    eprintln!(
        "[trace-index] reading {} ({:.2} GB compressed, {} frames, iter_base={})",
        path.display(),
        file_size as f64 / 1024.0 / 1024.0 / 1024.0,
        record_frames.len(),
        iter_base
    );

    let (chunk_tx, chunk_rx) = bounded::<DecodedChunk>(256);
    let (batch_tx, batch_rx) = bounded::<ParserOutput>(64);
    let max_raw_iter = Arc::new(AtomicU32::new(0));

    let mut parser_handles = Vec::with_capacity(NUM_PARSERS);
    for _ in 0..NUM_PARSERS {
        let rx = chunk_rx.clone();
        let tx = batch_tx.clone();
        let var_names = Arc::clone(&var_names);
        let block_names = Arc::clone(&block_names);
        let counters = Arc::clone(&counters);
        let max_raw_iter = Arc::clone(&max_raw_iter);
        parser_handles.push(thread::spawn(move || {
            parser_loop(
                rx,
                tx,
                var_names,
                block_names,
                execution_id,
                iter_base,
                kinds,
                flush_members,
                counters,
                max_raw_iter,
            )
        }));
    }
    drop(chunk_rx);
    drop(batch_tx);

    let (frame_tx, frame_rx) = bounded::<FrameJob>(record_frames.len().max(1));
    for job in record_frames.iter().cloned() {
        frame_tx
            .send(job)
            .expect("frame channel capacity matches job count");
    }
    drop(frame_tx);

    let effective_decoders = NUM_DECODERS.min(record_frames.len().max(1));
    let mut decoder_handles = Vec::with_capacity(effective_decoders);
    for _ in 0..effective_decoders {
        let rx = frame_rx.clone();
        let tx = chunk_tx.clone();
        let mmap = Arc::clone(&mmap);
        let counters = Arc::clone(&counters);
        decoder_handles.push(thread::spawn(move || decoder_loop(rx, tx, mmap, counters)));
    }
    drop(frame_rx);
    drop(chunk_tx);

    sqlite.begin()?;
    while let Ok(output) = batch_rx.recv() {
        match output {
            ParserOutput::Aggregate(batch) => {
                run_writer.write_batch(batch)?;
                counters
                    .runs_written
                    .store(run_writer.run_paths.len() as u64, Ordering::Relaxed);
                counters
                    .run_groups_written
                    .store(run_writer.groups_written, Ordering::Relaxed);
            }
            ParserOutput::Ordered(batch) => sqlite.insert_ordered_batch(batch)?,
        }
    }

    for handle in decoder_handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "trace-index decoder panicked",
                ))
            }
        }
    }
    for handle in parser_handles {
        if handle.join().is_err() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "trace-index parser panicked",
            ));
        }
    }
    sqlite.commit()?;
    Ok(max_raw_iter.load(Ordering::Relaxed))
}

#[derive(Clone)]
struct FrameJob {
    frame_ordinal: u32,
    range: std::ops::Range<usize>,
}

fn scan_frame_ranges_exact(data: &[u8]) -> io::Result<Vec<std::ops::Range<usize>>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let size = zstd::zstd_safe::find_frame_compressed_size(&data[offset..])
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        if size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd frame reported zero compressed size",
            ));
        }
        let end = offset.checked_add(size).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "zstd frame size overflow")
        })?;
        if end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "zstd frame extends past end of file",
            ));
        }
        out.push(offset..end);
        offset = end;
    }
    Ok(out)
}

fn parse_header_from_frame0(frame_bytes: &[u8]) -> io::Result<(Vec<String>, Vec<String>)> {
    let mut reader = zstd::stream::read::Decoder::new(Cursor::new(frame_bytes))?;
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("raw log: bad magic {:?}", magic),
        ));
    }
    let mut version = [0u8; 1];
    reader.read_exact(&mut version)?;
    if version[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("raw log: unsupported version {}", version[0]),
        ));
    }
    let var_names = read_name_table(&mut reader)?;
    let block_names = read_name_table(&mut reader)?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw log: frame 0 must contain only the SWRL header",
        ));
    }
    Ok((var_names, block_names))
}

fn read_name_table<R: Read>(reader: &mut R) -> io::Result<Vec<String>> {
    let mut count_buf = [0u8; 4];
    reader.read_exact(&mut count_buf)?;
    let count = u32::from_le_bytes(count_buf) as usize;
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        let mut len_buf = [0u8; 2];
        reader.read_exact(&mut len_buf)?;
        let len = u16::from_le_bytes(len_buf) as usize;
        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes)?;
        names.push(String::from_utf8_lossy(&bytes).into_owned());
    }
    Ok(names)
}

fn decoder_loop(
    frame_rx: Receiver<FrameJob>,
    chunk_tx: Sender<DecodedChunk>,
    mmap: Arc<Mmap>,
    counters: Arc<Counters>,
) -> io::Result<()> {
    let mut dctx = zstd::bulk::Decompressor::new()?;
    let mut out_buf = vec![0u8; MAX_FRAME_OUTPUT];

    while let Ok(job) = frame_rx.recv() {
        let frame_ordinal = job.frame_ordinal;
        let out_len = dctx
            .decompress_to_buffer(&mmap[job.range], &mut out_buf)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        if out_len % RECORD_SIZE != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw log frame has trailing partial record: {} bytes",
                    out_len % RECORD_SIZE
                ),
            ));
        }
        let mut offset = 0;
        while offset < out_len {
            let take = CHUNK_BYTES.min(out_len - offset);
            let take_aligned = (take / RECORD_SIZE) * RECORD_SIZE;
            let end = offset + take_aligned;
            counters
                .bytes_decoded
                .fetch_add(take_aligned as u64, Ordering::Relaxed);
            let first_record_ordinal = u32::try_from(offset / RECORD_SIZE).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "raw trace frame has too many records",
                )
            })?;
            if chunk_tx
                .send(DecodedChunk {
                    frame_ordinal,
                    first_record_ordinal,
                    bytes: out_buf[offset..end].to_vec(),
                })
                .is_err()
            {
                return Ok(());
            }
            offset = end;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parser_loop(
    chunk_rx: Receiver<DecodedChunk>,
    batch_tx: Sender<ParserOutput>,
    var_names: Arc<Vec<String>>,
    block_names: Arc<Vec<String>>,
    execution_id: u32,
    iter_base: u32,
    kinds: IndexKinds,
    flush_members: u64,
    counters: Arc<Counters>,
    max_raw_iter: Arc<AtomicU32>,
) {
    let n_vars = var_names.len();
    let n_blocks = block_names.len();
    let mut values: FxHashMap<IndexKey, FxHashSet<[u8; 12]>> = FxHashMap::default();
    let mut contexts: Vec<LocalContextDef> = Vec::new();
    let mut buffered_members = 0u64;
    let members_per_record = kinds.per_record_members();

    while let Ok(chunk) = chunk_rx.recv() {
        let mut parsed_this_chunk = 0u64;
        let mut indexed_this_chunk = 0u64;
        let mut contexts_this_chunk = 0u64;
        let mut ordered_visits: FxHashMap<u32, OrderedVisitChunk> = FxHashMap::default();
        let mut ordered_scalars: FxHashMap<u32, OrderedScalarChunk> = FxHashMap::default();
        let mut off = 0usize;
        while off + RECORD_SIZE <= chunk.bytes.len() {
            let record_ordinal = chunk
                .first_record_ordinal
                .checked_add((off / RECORD_SIZE) as u32)
                .expect("decoded trace chunk record ordinal overflow");
            let event_seq = ((chunk.frame_ordinal as u64) << 32) | record_ordinal as u64;
            let kind = chunk.bytes[off];
            let var_id = u32::from_le_bytes([
                chunk.bytes[off + 1],
                chunk.bytes[off + 2],
                chunk.bytes[off + 3],
                chunk.bytes[off + 4],
            ]);
            let pc = u32::from_le_bytes([
                chunk.bytes[off + 5],
                chunk.bytes[off + 6],
                chunk.bytes[off + 7],
                chunk.bytes[off + 8],
            ]);
            let block_id = u32::from_le_bytes([
                chunk.bytes[off + 9],
                chunk.bytes[off + 10],
                chunk.bytes[off + 11],
                chunk.bytes[off + 12],
            ]);
            let value = i64::from_le_bytes([
                chunk.bytes[off + 13],
                chunk.bytes[off + 14],
                chunk.bytes[off + 15],
                chunk.bytes[off + 16],
                chunk.bytes[off + 17],
                chunk.bytes[off + 18],
                chunk.bytes[off + 19],
                chunk.bytes[off + 20],
            ]);
            let iter_id = u32::from_le_bytes([
                chunk.bytes[off + 21],
                chunk.bytes[off + 22],
                chunk.bytes[off + 23],
                chunk.bytes[off + 24],
            ]);
            off += RECORD_SIZE;
            parsed_this_chunk += 1;

            if kind == OP_ITER_CONTEXT {
                if var_id > 0 && (block_id as usize) < n_blocks {
                    max_raw_iter.fetch_max(var_id, Ordering::Relaxed);
                    if pc > 0 {
                        max_raw_iter.fetch_max(pc, Ordering::Relaxed);
                    }
                    contexts.push(LocalContextDef {
                        iter_id: offset_nonzero(var_id, iter_base),
                        parent_iter_id: offset_nonzero(pc, iter_base),
                        depth: iter_id,
                        header_block_id: block_id,
                        iter_count: value,
                    });
                    contexts_this_chunk += 1;
                    if contexts.len() >= 4096 {
                        send_batch(
                            &batch_tx,
                            &var_names,
                            &block_names,
                            &mut values,
                            &mut contexts,
                        );
                    }
                }
                continue;
            }

            if kind == OP_PRE_PC {
                push_ordered_visit(&mut ordered_visits, pc, event_seq);
                continue;
            }

            let scalar_event = matches!(kind, OP_INITIAL_SCALAR | OP_WRITE | OP_UNKNOWN_WRITE);
            if scalar_event {
                if var_id as usize >= n_vars {
                    continue;
                }
                // The traced program is the self-composition: the shadow lane
                // is an ordinary input (the runner completes every declared
                // shadow formal before executing), so `.shadow` scalars index
                // exactly like base ones.  A differing-lane pair trace is the
                // cheap refutation instrument; dropping the shadow half here
                // made it blind (aead #34: 0 of 102,799 timelines were shadow).
                push_ordered_scalar(&mut ordered_scalars, var_id, event_seq, kind, pc, value);
                if kind != OP_WRITE {
                    continue;
                }
            } else if kind != OP_READ {
                // Unknown event kinds never acquire aggregate-value semantics.
                continue;
            }

            if var_id as usize >= n_vars || block_id as usize >= n_blocks {
                continue;
            }
            // Shadow-lane reads index like base-lane reads (see above); the
            // `skipped_shadow` counter stays in the index meta as an
            // always-zero typed fact so the contract keys are unchanged.

            // Loop records get this execution's per-iteration id; straight-line
            // records (iter_id==0) get the execution's reserved base id — NOT a
            // shared 0 — so co-occurrence tuples never cross executions.
            let eff_iter = if iter_id > 0 {
                max_raw_iter.fetch_max(iter_id, Ordering::Relaxed);
                offset_nonzero(iter_id, iter_base)
            } else {
                iter_base
            };
            let mut member = [0u8; 12];
            member[..8].copy_from_slice(&(value as u64).to_le_bytes());
            member[8..].copy_from_slice(&eff_iter.to_le_bytes());

            if kinds.pc {
                push_member(
                    &mut values,
                    IndexKey {
                        kind: KIND_PC,
                        var_id,
                        loc: pc,
                        op: 0,
                    },
                    member,
                );
            }
            if kinds.block {
                push_member(
                    &mut values,
                    IndexKey {
                        kind: KIND_BLOCK,
                        var_id,
                        loc: block_id,
                        op: 0,
                    },
                    member,
                );
            }
            if kinds.op {
                push_member(
                    &mut values,
                    IndexKey {
                        kind: KIND_OP,
                        var_id,
                        loc: pc,
                        op: kind,
                    },
                    member,
                );
            }
            indexed_this_chunk += 1;
            buffered_members += members_per_record;

            if buffered_members >= flush_members {
                send_batch(
                    &batch_tx,
                    &var_names,
                    &block_names,
                    &mut values,
                    &mut contexts,
                );
                buffered_members = 0;
            }
        }

        if !ordered_visits.is_empty() || !ordered_scalars.is_empty() {
            let mut visits: Vec<_> = ordered_visits.into_values().collect();
            visits.sort_unstable_by_key(|chunk| chunk.pc);
            let mut scalar_timelines: Vec<_> = ordered_scalars.into_values().collect();
            scalar_timelines.sort_unstable_by_key(|chunk| chunk.var_id);
            let _ = batch_tx.send(ParserOutput::Ordered(OrderedBatch {
                execution_id,
                var_names: Arc::clone(&var_names),
                visits,
                scalar_timelines,
            }));
        }

        counters
            .records_parsed
            .fetch_add(parsed_this_chunk, Ordering::Relaxed);
        counters
            .records_indexed
            .fetch_add(indexed_this_chunk, Ordering::Relaxed);
        counters
            .contexts_indexed
            .fetch_add(contexts_this_chunk, Ordering::Relaxed);
    }
    send_batch(
        &batch_tx,
        &var_names,
        &block_names,
        &mut values,
        &mut contexts,
    );
}

#[inline]
fn offset_nonzero(iter_id: u32, iter_offset: u32) -> u32 {
    if iter_id == 0 {
        0
    } else {
        iter_id
            .checked_add(iter_offset)
            .expect("trace iteration context id overflow")
    }
}

#[inline]
fn push_member(
    values: &mut FxHashMap<IndexKey, FxHashSet<[u8; 12]>>,
    key: IndexKey,
    member: [u8; 12],
) {
    values.entry(key).or_default().insert(member);
}

#[inline]
fn push_ordered_visit(visits: &mut FxHashMap<u32, OrderedVisitChunk>, pc: u32, event_seq: u64) {
    let chunk = visits.entry(pc).or_insert_with(|| OrderedVisitChunk {
        pc,
        first_event_seq: event_seq,
        last_event_seq: event_seq,
        event_count: 0,
        events: Vec::new(),
    });
    chunk.last_event_seq = event_seq;
    chunk.event_count += 1;
    chunk.events.extend_from_slice(&event_seq.to_le_bytes());
}

#[inline]
fn push_ordered_scalar(
    timelines: &mut FxHashMap<u32, OrderedScalarChunk>,
    var_id: u32,
    event_seq: u64,
    kind: u8,
    pc: u32,
    value: i64,
) {
    let chunk = timelines
        .entry(var_id)
        .or_insert_with(|| OrderedScalarChunk {
            var_id,
            first_event_seq: event_seq,
            last_event_seq: event_seq,
            event_count: 0,
            events: Vec::new(),
        });
    chunk.last_event_seq = event_seq;
    chunk.event_count += 1;
    chunk.events.extend_from_slice(&event_seq.to_le_bytes());
    chunk.events.push(kind);
    chunk.events.extend_from_slice(&pc.to_le_bytes());
    chunk.events.extend_from_slice(&value.to_le_bytes());
}

fn send_batch(
    batch_tx: &Sender<ParserOutput>,
    var_names: &Arc<Vec<String>>,
    block_names: &Arc<Vec<String>>,
    values: &mut FxHashMap<IndexKey, FxHashSet<[u8; 12]>>,
    contexts: &mut Vec<LocalContextDef>,
) {
    if values.is_empty() && contexts.is_empty() {
        return;
    }
    let mut batch_values = FxHashMap::default();
    std::mem::swap(values, &mut batch_values);
    let mut batch_contexts = Vec::new();
    std::mem::swap(contexts, &mut batch_contexts);
    let _ = batch_tx.send(ParserOutput::Aggregate(ParsedBatch {
        var_names: Arc::clone(var_names),
        block_names: Arc::clone(block_names),
        values: batch_values,
        contexts: batch_contexts,
    }));
}

#[derive(Default)]
struct NameInterner {
    vars: Vec<String>,
    var_ids: FxHashMap<String, u32>,
    blocks: Vec<String>,
    block_ids: FxHashMap<String, u32>,
}

impl NameInterner {
    fn intern_var(&mut self, name: &str) -> u32 {
        if let Some(id) = self.var_ids.get(name) {
            return *id;
        }
        let id = self.vars.len() as u32;
        self.vars.push(name.to_owned());
        self.var_ids.insert(name.to_owned(), id);
        id
    }

    fn intern_block(&mut self, name: &str) -> u32 {
        if let Some(id) = self.block_ids.get(name) {
            return *id;
        }
        let id = self.blocks.len() as u32;
        self.blocks.push(name.to_owned());
        self.block_ids.insert(name.to_owned(), id);
        id
    }

    fn var_name(&self, id: u32) -> io::Result<&str> {
        self.vars
            .get(id as usize)
            .map(String::as_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad global var id"))
    }

    fn block_name(&self, id: u32) -> io::Result<&str> {
        self.blocks
            .get(id as usize)
            .map(String::as_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad global block id"))
    }
}

struct RunGroup {
    key: GlobalKey,
    members: Vec<[u8; 12]>,
}

struct RunFileWriter {
    dir: PathBuf,
    run_paths: Vec<PathBuf>,
    names: NameInterner,
    contexts: FxHashMap<u32, GlobalContextDef>,
    groups_written: u64,
}

impl RunFileWriter {
    fn create(sqlite_path: &Path) -> io::Result<Self> {
        let base_dir = std::env::var("SWOOSH_TRACE_INDEX_TMPDIR")
            .map(PathBuf::from)
            .ok()
            .or_else(|| sqlite_path.parent().map(Path::to_path_buf))
            .unwrap_or_else(std::env::temp_dir);
        fs::create_dir_all(&base_dir)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let sqlite_name = sqlite_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("trace_index.sqlite");
        let dir = base_dir.join(format!(
            ".{sqlite_name}.runs.{}.{}",
            std::process::id(),
            stamp
        ));
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            run_paths: Vec::new(),
            names: NameInterner::default(),
            contexts: FxHashMap::default(),
            groups_written: 0,
        })
    }

    fn write_batch(&mut self, batch: ParsedBatch) -> io::Result<()> {
        for ctx in batch.contexts {
            if ctx.header_block_id as usize >= batch.block_names.len() {
                continue;
            }
            let header_id = self
                .names
                .intern_block(&batch.block_names[ctx.header_block_id as usize]);
            let global = GlobalContextDef {
                iter_id: ctx.iter_id,
                parent_iter_id: ctx.parent_iter_id,
                depth: ctx.depth,
                header_id,
                iter_count: ctx.iter_count,
            };
            if let Some(existing) = self.contexts.get(&global.iter_id) {
                if existing != &global {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("conflicting trace context id {}", global.iter_id),
                    ));
                }
            } else {
                self.contexts.insert(global.iter_id, global);
            }
        }
        if batch.values.is_empty() {
            return Ok(());
        }
        let mut groups = Vec::with_capacity(batch.values.len());
        for (key, members) in batch.values {
            if members.is_empty() {
                continue;
            }
            let var_id = self.names.intern_var(&batch.var_names[key.var_id as usize]);
            let loc = if key.kind == KIND_BLOCK {
                self.names
                    .intern_block(&batch.block_names[key.loc as usize])
            } else {
                key.loc
            };
            let mut members: Vec<[u8; 12]> = members.into_iter().collect();
            members.sort_unstable();
            members.dedup();
            groups.push(RunGroup {
                key: GlobalKey {
                    kind: key.kind,
                    var_id,
                    loc,
                    op: key.op,
                },
                members,
            });
        }
        if groups.is_empty() {
            return Ok(());
        }
        groups.sort_unstable_by_key(|group| group.key);
        let groups = merge_adjacent_groups(groups);
        let path = self
            .dir
            .join(format!("run_{:08}.swir", self.run_paths.len()));
        write_run_file(&path, &groups)?;
        self.groups_written += groups.len() as u64;
        self.run_paths.push(path);
        Ok(())
    }

    fn cleanup(&self) -> io::Result<()> {
        if self.dir.exists() {
            fs::remove_dir_all(&self.dir)?;
        }
        Ok(())
    }
}

fn merge_adjacent_groups(groups: Vec<RunGroup>) -> Vec<RunGroup> {
    let mut out: Vec<RunGroup> = Vec::with_capacity(groups.len());
    for group in groups {
        if let Some(last) = out.last_mut() {
            if last.key == group.key {
                last.members.extend(group.members);
                last.members.sort_unstable();
                last.members.dedup();
                continue;
            }
        }
        out.push(group);
    }
    out
}

fn write_run_file(path: &Path, groups: &[RunGroup]) -> io::Result<()> {
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, File::create(path)?);
    writer.write_all(RUN_MAGIC)?;
    writer.write_all(&[RUN_VERSION])?;
    for group in groups {
        write_run_group(&mut writer, group)?;
    }
    writer.flush()
}

fn write_run_group<W: Write>(writer: &mut W, group: &RunGroup) -> io::Result<()> {
    write_run_key(writer, group.key)?;
    writer.write_all(&(group.members.len() as u64).to_le_bytes())?;
    for member in &group.members {
        writer.write_all(member)?;
    }
    Ok(())
}

fn write_run_key<W: Write>(writer: &mut W, key: GlobalKey) -> io::Result<()> {
    writer.write_all(&[key.kind])?;
    writer.write_all(&key.var_id.to_le_bytes())?;
    writer.write_all(&key.loc.to_le_bytes())?;
    writer.write_all(&[key.op])?;
    Ok(())
}

struct RunCursor {
    path: PathBuf,
    reader: BufReader<File>,
    current: Option<RunGroup>,
    previous_key: Option<GlobalKey>,
}

impl RunCursor {
    fn open(path: &Path) -> io::Result<Self> {
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, File::open(path)?);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != RUN_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: bad run magic", path.display()),
            ));
        }
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != RUN_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: unsupported run version {}", path.display(), version[0]),
            ));
        }
        let mut cursor = Self {
            path: path.to_path_buf(),
            reader,
            current: None,
            previous_key: None,
        };
        cursor.advance()?;
        Ok(cursor)
    }

    fn advance(&mut self) -> io::Result<()> {
        let next = read_run_group(&mut self.reader)?;
        if let Some(group) = next.as_ref() {
            if self
                .previous_key
                .is_some_and(|previous| group.key <= previous)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{}: run keys are not in strict ascending order",
                        self.path.display()
                    ),
                ));
            }
            self.previous_key = Some(group.key);
        }
        self.current = next;
        Ok(())
    }
}

fn read_run_group<R: Read>(reader: &mut R) -> io::Result<Option<RunGroup>> {
    let mut key_buf = [0u8; RUN_KEY_SIZE];
    if !read_exact_or_eof(reader, &mut key_buf)? {
        return Ok(None);
    }
    let key = GlobalKey {
        kind: key_buf[0],
        var_id: u32::from_le_bytes([key_buf[1], key_buf[2], key_buf[3], key_buf[4]]),
        loc: u32::from_le_bytes([key_buf[5], key_buf[6], key_buf[7], key_buf[8]]),
        op: key_buf[9],
    };
    let mut count_buf = [0u8; 8];
    reader.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf);
    if count > (usize::MAX as u64) / 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "run group member count is too large",
        ));
    }
    let mut members = Vec::with_capacity(count as usize);
    let mut member = [0u8; 12];
    for _ in 0..count {
        reader.read_exact(&mut member)?;
        members.push(member);
    }
    Ok(Some(RunGroup { key, members }))
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut offset = 0;
    while offset < buf.len() {
        let n = reader.read(&mut buf[offset..])?;
        if n == 0 {
            if offset == 0 {
                return Ok(false);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "partial run record",
            ));
        }
        offset += n;
    }
    Ok(true)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct HeapItem {
    key: GlobalKey,
    cursor_idx: usize,
}

fn reduce_runs_to_fan_in(
    run_paths: &[PathBuf],
    run_dir: &Path,
    fan_in: usize,
    reclaim_inputs: bool,
) -> io::Result<Vec<PathBuf>> {
    if fan_in < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "trace-index merge fan-in must be at least 2",
        ));
    }
    let mut current = run_paths.to_vec();
    let mut pass = 0usize;
    while current.len() > fan_in {
        pass += 1;
        let batch_count = current.len().div_ceil(fan_in);
        eprintln!(
            "[trace-index] merge pass {}: {} input run(s) -> {} run(s)",
            pass,
            format_commas(current.len() as u64),
            format_commas(batch_count as u64),
        );
        let mut next = Vec::with_capacity(batch_count);
        for (batch_index, batch) in current.chunks(fan_in).enumerate() {
            if batch.len() == 1 {
                next.push(batch[0].clone());
                continue;
            }
            let output = run_dir.join(format!("merge_{pass:04}_{batch_index:08}.swir"));
            merge_runs_into_run_atomically(batch, &output)?;
            if reclaim_inputs {
                remove_run_files(batch)?;
            }
            next.push(output);
        }
        current = next;
    }
    Ok(current)
}

fn remove_run_files(run_paths: &[PathBuf]) -> io::Result<()> {
    for path in run_paths {
        fs::remove_file(path)?;
    }
    Ok(())
}

struct PartialRunPublication {
    path: PathBuf,
    published: bool,
}

impl PartialRunPublication {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            published: false,
        }
    }
}

impl Drop for PartialRunPublication {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn merge_runs_into_run_atomically(run_paths: &[PathBuf], output: &Path) -> io::Result<()> {
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("merged.swir");
    let partial = output.with_file_name(format!(".{file_name}.partial"));
    if partial.exists() {
        fs::remove_file(&partial)?;
    }
    let mut publication = PartialRunPublication::new(partial);
    {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&publication.path)?;
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
        writer.write_all(RUN_MAGIC)?;
        writer.write_all(&[RUN_VERSION])?;
        merge_sorted_run_groups(run_paths, |group| write_run_group(&mut writer, group))?;
        writer.flush()?;
    }
    fs::rename(&publication.path, output)?;
    publication.published = true;
    Ok(())
}

fn merge_runs_into_sqlite(
    run_paths: &[PathBuf],
    names: &NameInterner,
    sqlite: &mut SqliteTraceIndexWriter,
    counters: Arc<Counters>,
) -> io::Result<u64> {
    let mut rows = 0u64;
    sqlite.begin()?;
    merge_sorted_run_groups(run_paths, |group| {
        sqlite.insert_members(group.key, &group.members, names)?;
        rows += 1;
        counters.rows_written.store(rows, Ordering::Relaxed);
        if rows % 100_000 == 0 {
            sqlite.commit()?;
            sqlite.begin()?;
        }
        Ok(())
    })?;
    sqlite.commit()?;
    Ok(rows)
}

fn merge_sorted_run_groups<F>(run_paths: &[PathBuf], mut emit: F) -> io::Result<u64>
where
    F: FnMut(&RunGroup) -> io::Result<()>,
{
    let mut cursors = Vec::with_capacity(run_paths.len());
    let mut heap = BinaryHeap::new();
    for path in run_paths {
        let cursor_idx = cursors.len();
        let cursor = RunCursor::open(path)?;
        if let Some(group) = cursor.current.as_ref() {
            heap.push(Reverse(HeapItem {
                key: group.key,
                cursor_idx,
            }));
        }
        cursors.push(cursor);
    }

    let mut groups = 0u64;
    while let Some(Reverse(item)) = heap.pop() {
        let key = item.key;
        let mut members = Vec::new();
        consume_heap_item(item, &mut cursors, &mut heap, &mut members)?;
        while let Some(Reverse(next)) = heap.peek().copied() {
            if next.key != key {
                break;
            }
            let next = heap.pop().expect("peeked heap item").0;
            consume_heap_item(next, &mut cursors, &mut heap, &mut members)?;
        }
        members.sort_unstable();
        members.dedup();
        emit(&RunGroup { key, members })?;
        groups += 1;
    }
    Ok(groups)
}

fn consume_heap_item(
    item: HeapItem,
    cursors: &mut [RunCursor],
    heap: &mut BinaryHeap<Reverse<HeapItem>>,
    members: &mut Vec<[u8; 12]>,
) -> io::Result<()> {
    let cursor = &mut cursors[item.cursor_idx];
    let group = cursor
        .current
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty run cursor"))?;
    if group.key != item.key {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "run cursor key mismatch",
        ));
    }
    members.extend(group.members);
    cursor.advance()?;
    if let Some(next_group) = cursor.current.as_ref() {
        heap.push(Reverse(HeapItem {
            key: next_group.key,
            cursor_idx: item.cursor_idx,
        }));
    }
    Ok(())
}

fn sampler_loop(counters: Arc<Counters>, stop: Arc<AtomicBool>) {
    let start = Instant::now();
    let mut last_bytes = 0u64;
    let mut last_records = 0u64;
    let mut last_t = start;
    while !stop.load(Ordering::Relaxed) {
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
        let records = counters.records_parsed.load(Ordering::Relaxed);
        let indexed = counters.records_indexed.load(Ordering::Relaxed);
        let runs = counters.runs_written.load(Ordering::Relaxed);
        let groups = counters.run_groups_written.load(Ordering::Relaxed);
        let rows = counters.rows_written.load(Ordering::Relaxed);
        eprintln!(
            "[trace-index] {:>13} parsed | {:>13} indexed | decode {:>6.1} MB/s | parse {:>7.1}k rec/s | runs {:>5} groups {:>9} rows {:>9} | {:.1} min",
            format_commas(records),
            format_commas(indexed),
            (bytes - last_bytes) as f64 / dt / 1024.0 / 1024.0,
            (records - last_records) as f64 / dt / 1000.0,
            format_commas(runs),
            format_commas(groups),
            format_commas(rows),
            total_dt / 60.0,
        );
        last_bytes = bytes;
        last_records = records;
        last_t = now;
    }
}

struct SqliteTraceIndexWriter {
    db: *mut sqlite3,
    insert_stmt: *mut sqlite3_stmt,
    context_stmt: *mut sqlite3_stmt,
    pre_visit_stmt: *mut sqlite3_stmt,
    scalar_timeline_stmt: *mut sqlite3_stmt,
    meta_stmt: *mut sqlite3_stmt,
}

impl SqliteTraceIndexWriter {
    fn create(path: &Path, bench: Option<&str>) -> io::Result<Self> {
        let mut db: *mut sqlite3 = ptr::null_mut();
        let path_text = path.to_string_lossy();
        let path_c = std::ffi::CString::new(path_text.as_bytes()).map_err(invalid_nul)?;
        let rc = unsafe { sqlite3_open(path_c.as_ptr(), &mut db) };
        if rc != SQLITE_OK {
            let err = sqlite_error(db, "sqlite3_open");
            if !db.is_null() {
                unsafe {
                    sqlite3_close(db);
                }
            }
            return Err(err);
        }

        let mut writer = Self {
            db,
            insert_stmt: ptr::null_mut(),
            context_stmt: ptr::null_mut(),
            pre_visit_stmt: ptr::null_mut(),
            scalar_timeline_stmt: ptr::null_mut(),
            meta_stmt: ptr::null_mut(),
        };
        writer.exec(
            "
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            CREATE TABLE IF NOT EXISTS trace_evidence (
                kind TEXT NOT NULL,
                var TEXT NOT NULL,
                loc TEXT NOT NULL,
                op TEXT NOT NULL DEFAULT '',
                members BLOB NOT NULL,
                PRIMARY KEY (kind, var, loc, op)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS trace_iteration_contexts (
                iter_id INTEGER PRIMARY KEY,
                parent_iter_id INTEGER NOT NULL,
                depth INTEGER NOT NULL,
                header TEXT NOT NULL,
                iter_count INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS trace_pre_visits (
                pc INTEGER NOT NULL,
                execution_id INTEGER NOT NULL,
                first_event_seq INTEGER NOT NULL,
                last_event_seq INTEGER NOT NULL,
                event_count INTEGER NOT NULL,
                events BLOB NOT NULL,
                PRIMARY KEY (pc, execution_id, first_event_seq)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS trace_scalar_timelines (
                var TEXT NOT NULL,
                execution_id INTEGER NOT NULL,
                first_event_seq INTEGER NOT NULL,
                last_event_seq INTEGER NOT NULL,
                event_count INTEGER NOT NULL,
                events BLOB NOT NULL,
                PRIMARY KEY (var, execution_id, first_event_seq)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS trace_index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            ",
        )?;
        writer.insert_stmt = writer.prepare(
            "INSERT INTO trace_evidence(kind, var, loc, op, members) \
             VALUES(?, ?, ?, ?, ?)",
        )?;
        writer.context_stmt = writer.prepare(
            "INSERT OR REPLACE INTO trace_iteration_contexts\
             (iter_id, parent_iter_id, depth, header, iter_count) \
             VALUES(?, ?, ?, ?, ?)",
        )?;
        writer.pre_visit_stmt = writer.prepare(
            "INSERT INTO trace_pre_visits\
             (pc, execution_id, first_event_seq, last_event_seq, event_count, events) \
             VALUES(?, ?, ?, ?, ?, ?)",
        )?;
        writer.scalar_timeline_stmt = writer.prepare(
            "INSERT INTO trace_scalar_timelines\
             (var, execution_id, first_event_seq, last_event_seq, event_count, events) \
             VALUES(?, ?, ?, ?, ?, ?)",
        )?;
        writer.meta_stmt =
            writer.prepare("INSERT OR REPLACE INTO trace_index_meta(key, value) VALUES(?, ?)")?;
        writer.set_meta("schema_version", TRACE_INDEX_VERSION)?;
        writer.set_meta("ordered_trace_version", ORDERED_TRACE_VERSION)?;
        writer.set_meta("ordered_trace_authority", "awaiting_completion_manifest")?;
        writer.set_meta(
            "ordered_sequence_encoding",
            "u64:(physical_frame_ordinal<<32)|record_ordinal",
        )?;
        writer.set_meta("ordered_visit_encoding", "le:u64-event-seq")?;
        writer.set_meta(
            "ordered_scalar_encoding",
            "le:u64-seq,u8-kind,u32-pc,i64-value",
        )?;
        if let Some(bench) = bench {
            writer.set_meta("benchmark", bench)?;
        }
        Ok(writer)
    }

    fn exec(&self, sql: &str) -> io::Result<()> {
        let sql_c = std::ffi::CString::new(sql).map_err(invalid_nul)?;
        let mut errmsg: *mut c_char = ptr::null_mut();
        let rc =
            unsafe { sqlite3_exec(self.db, sql_c.as_ptr(), None, ptr::null_mut(), &mut errmsg) };
        if rc != SQLITE_OK {
            let msg = if errmsg.is_null() {
                sqlite_error(self.db, "sqlite3_exec")
            } else {
                let text = unsafe { CStr::from_ptr(errmsg) }
                    .to_string_lossy()
                    .into_owned();
                unsafe {
                    sqlite3_free(errmsg as *mut c_void);
                }
                io::Error::new(io::ErrorKind::Other, text)
            };
            return Err(msg);
        }
        Ok(())
    }

    fn insert_contexts(
        &mut self,
        contexts: &FxHashMap<u32, GlobalContextDef>,
        names: &NameInterner,
    ) -> io::Result<()> {
        if contexts.is_empty() {
            return Ok(());
        }
        let mut ordered: Vec<GlobalContextDef> = contexts.values().copied().collect();
        ordered.sort_unstable_by_key(|ctx| ctx.iter_id);
        self.begin()?;
        for (idx, ctx) in ordered.iter().enumerate() {
            let header = names.block_name(ctx.header_id)?;
            unsafe {
                bind_i64(self.context_stmt, 1, ctx.iter_id as i64)?;
                bind_i64(self.context_stmt, 2, ctx.parent_iter_id as i64)?;
                bind_i64(self.context_stmt, 3, ctx.depth as i64)?;
                bind_text(self.context_stmt, 4, header)?;
                bind_i64(self.context_stmt, 5, ctx.iter_count)?;
                step_done(self.db, self.context_stmt)?;
                reset_and_clear(self.db, self.context_stmt)?;
            }
            if (idx + 1) % 100_000 == 0 {
                self.commit()?;
                self.begin()?;
            }
        }
        self.commit()
    }

    fn insert_ordered_batch(&mut self, batch: OrderedBatch) -> io::Result<()> {
        for visit in batch.visits {
            if visit.events.len() != visit.event_count as usize * ORDERED_VISIT_MEMBER_SIZE {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "packed PRE-visit chunk has an invalid size",
                ));
            }
            unsafe {
                bind_i64(self.pre_visit_stmt, 1, visit.pc as i64)?;
                bind_i64(self.pre_visit_stmt, 2, batch.execution_id as i64)?;
                bind_i64(
                    self.pre_visit_stmt,
                    3,
                    ordered_seq_as_i64(visit.first_event_seq)?,
                )?;
                bind_i64(
                    self.pre_visit_stmt,
                    4,
                    ordered_seq_as_i64(visit.last_event_seq)?,
                )?;
                bind_i64(self.pre_visit_stmt, 5, visit.event_count as i64)?;
                bind_blob(self.pre_visit_stmt, 6, &visit.events)?;
                step_done(self.db, self.pre_visit_stmt)?;
                reset_and_clear(self.db, self.pre_visit_stmt)?;
            }
        }
        for timeline in batch.scalar_timelines {
            if timeline.events.len() != timeline.event_count as usize * ORDERED_SCALAR_MEMBER_SIZE {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "packed scalar-timeline chunk has an invalid size",
                ));
            }
            let var = batch
                .var_names
                .get(timeline.var_id as usize)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bad ordered scalar var id")
                })?;
            unsafe {
                bind_text(self.scalar_timeline_stmt, 1, var)?;
                bind_i64(self.scalar_timeline_stmt, 2, batch.execution_id as i64)?;
                bind_i64(
                    self.scalar_timeline_stmt,
                    3,
                    ordered_seq_as_i64(timeline.first_event_seq)?,
                )?;
                bind_i64(
                    self.scalar_timeline_stmt,
                    4,
                    ordered_seq_as_i64(timeline.last_event_seq)?,
                )?;
                bind_i64(self.scalar_timeline_stmt, 5, timeline.event_count as i64)?;
                bind_blob(self.scalar_timeline_stmt, 6, &timeline.events)?;
                step_done(self.db, self.scalar_timeline_stmt)?;
                reset_and_clear(self.db, self.scalar_timeline_stmt)?;
            }
        }
        Ok(())
    }

    fn prepare(&self, sql: &str) -> io::Result<*mut sqlite3_stmt> {
        let sql_c = std::ffi::CString::new(sql).map_err(invalid_nul)?;
        let mut stmt: *mut sqlite3_stmt = ptr::null_mut();
        let rc =
            unsafe { sqlite3_prepare_v2(self.db, sql_c.as_ptr(), -1, &mut stmt, ptr::null_mut()) };
        if rc != SQLITE_OK {
            return Err(sqlite_error(self.db, "sqlite3_prepare_v2"));
        }
        Ok(stmt)
    }

    fn begin(&self) -> io::Result<()> {
        self.exec("BEGIN IMMEDIATE")
    }

    fn commit(&self) -> io::Result<()> {
        self.exec("COMMIT")
    }

    fn insert_members(
        &mut self,
        key: GlobalKey,
        members: &[[u8; 12]],
        names: &NameInterner,
    ) -> io::Result<()> {
        let kind = sqlite_kind(key.kind);
        let var = names.var_name(key.var_id)?;
        let loc;
        let loc_ref: &str = if key.kind == KIND_BLOCK {
            names.block_name(key.loc)?
        } else {
            loc = key.loc.to_string();
            loc.as_str()
        };
        let op;
        let op_ref: &str = if key.kind == KIND_OP {
            op = (key.op as char).to_string();
            op.as_str()
        } else {
            ""
        };
        let mut blob = Vec::with_capacity(members.len() * 12);
        for member in members {
            blob.extend_from_slice(member);
        }
        unsafe {
            bind_text(self.insert_stmt, 1, kind)?;
            bind_text(self.insert_stmt, 2, var)?;
            bind_text(self.insert_stmt, 3, loc_ref)?;
            bind_text(self.insert_stmt, 4, op_ref)?;
            bind_blob(self.insert_stmt, 5, &blob)?;
            step_done(self.db, self.insert_stmt)?;
            reset_and_clear(self.db, self.insert_stmt)?;
        }
        Ok(())
    }

    fn set_meta(&mut self, key: &str, value: &str) -> io::Result<()> {
        unsafe {
            bind_text(self.meta_stmt, 1, key)?;
            bind_text(self.meta_stmt, 2, value)?;
            step_done(self.db, self.meta_stmt)?;
            reset_and_clear(self.db, self.meta_stmt)?;
        }
        Ok(())
    }

    fn finish(mut self, rows: u64) -> io::Result<u64> {
        self.exec(
            "CREATE INDEX IF NOT EXISTS trace_evidence_var_kind \
             ON trace_evidence (var, kind, op)",
        )?;
        unsafe {
            sqlite3_finalize(self.insert_stmt);
            sqlite3_finalize(self.context_stmt);
            sqlite3_finalize(self.pre_visit_stmt);
            sqlite3_finalize(self.scalar_timeline_stmt);
            sqlite3_finalize(self.meta_stmt);
            self.insert_stmt = ptr::null_mut();
            self.context_stmt = ptr::null_mut();
            self.pre_visit_stmt = ptr::null_mut();
            self.scalar_timeline_stmt = ptr::null_mut();
            self.meta_stmt = ptr::null_mut();
            if sqlite3_close(self.db) != SQLITE_OK {
                return Err(sqlite_error(self.db, "sqlite3_close"));
            }
            self.db = ptr::null_mut();
        }
        Ok(rows)
    }
}

impl Drop for SqliteTraceIndexWriter {
    fn drop(&mut self) {
        unsafe {
            if !self.insert_stmt.is_null() {
                sqlite3_finalize(self.insert_stmt);
            }
            if !self.context_stmt.is_null() {
                sqlite3_finalize(self.context_stmt);
            }
            if !self.pre_visit_stmt.is_null() {
                sqlite3_finalize(self.pre_visit_stmt);
            }
            if !self.scalar_timeline_stmt.is_null() {
                sqlite3_finalize(self.scalar_timeline_stmt);
            }
            if !self.meta_stmt.is_null() {
                sqlite3_finalize(self.meta_stmt);
            }
            if !self.db.is_null() {
                sqlite3_close(self.db);
            }
        }
    }
}

fn ordered_seq_as_i64(seq: u64) -> io::Result<i64> {
    i64::try_from(seq).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "ordered trace sequence exceeds SQLite INTEGER range",
        )
    })
}

fn sqlite_kind(kind: u8) -> &'static str {
    match kind {
        KIND_PC => "pc",
        KIND_BLOCK => "block",
        KIND_OP => "op",
        _ => "unknown",
    }
}

fn invalid_nul(err: std::ffi::NulError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, err.to_string())
}

fn format_commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let chars: Vec<char> = s.chars().collect();
    for (i, ch) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*ch);
    }
    out
}

#[allow(non_camel_case_types)]
enum sqlite3 {}
#[allow(non_camel_case_types)]
enum sqlite3_stmt {}

const SQLITE_OK: c_int = 0;
const SQLITE_DONE: c_int = 101;

type SqliteCallback =
    Option<unsafe extern "C" fn(*mut c_void, c_int, *mut *mut c_char, *mut *mut c_char) -> c_int>;
type SqliteDestructor = Option<unsafe extern "C" fn(*mut c_void)>;

#[link(name = "sqlite3")]
unsafe extern "C" {
    fn sqlite3_open(filename: *const c_char, pp_db: *mut *mut sqlite3) -> c_int;
    fn sqlite3_close(db: *mut sqlite3) -> c_int;
    fn sqlite3_errmsg(db: *mut sqlite3) -> *const c_char;
    fn sqlite3_free(ptr: *mut c_void);
    fn sqlite3_exec(
        db: *mut sqlite3,
        sql: *const c_char,
        callback: SqliteCallback,
        arg: *mut c_void,
        errmsg: *mut *mut c_char,
    ) -> c_int;
    fn sqlite3_prepare_v2(
        db: *mut sqlite3,
        sql: *const c_char,
        nbyte: c_int,
        pp_stmt: *mut *mut sqlite3_stmt,
        tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_finalize(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_step(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_reset(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_clear_bindings(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_bind_text(
        stmt: *mut sqlite3_stmt,
        idx: c_int,
        value: *const c_char,
        n: c_int,
        destructor: SqliteDestructor,
    ) -> c_int;
    fn sqlite3_bind_blob(
        stmt: *mut sqlite3_stmt,
        idx: c_int,
        value: *const c_void,
        n: c_int,
        destructor: SqliteDestructor,
    ) -> c_int;
    fn sqlite3_bind_int64(stmt: *mut sqlite3_stmt, idx: c_int, value: i64) -> c_int;
}

fn sqlite_error(db: *mut sqlite3, context: &str) -> io::Error {
    if db.is_null() {
        return io::Error::new(
            io::ErrorKind::Other,
            format!("{context}: null sqlite handle"),
        );
    }
    let msg = unsafe { CStr::from_ptr(sqlite3_errmsg(db)) }
        .to_string_lossy()
        .into_owned();
    io::Error::new(io::ErrorKind::Other, format!("{context}: {msg}"))
}

unsafe fn bind_text(stmt: *mut sqlite3_stmt, idx: c_int, value: &str) -> io::Result<()> {
    let rc = sqlite3_bind_text(
        stmt,
        idx,
        value.as_ptr() as *const c_char,
        value.len() as c_int,
        None,
    );
    if rc != SQLITE_OK {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("sqlite3_bind_text failed with code {rc}"),
        ));
    }
    Ok(())
}

unsafe fn bind_blob(stmt: *mut sqlite3_stmt, idx: c_int, value: &[u8]) -> io::Result<()> {
    let rc = sqlite3_bind_blob(
        stmt,
        idx,
        value.as_ptr() as *const c_void,
        value.len() as c_int,
        None,
    );
    if rc != SQLITE_OK {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("sqlite3_bind_blob failed with code {rc}"),
        ));
    }
    Ok(())
}

unsafe fn bind_i64(stmt: *mut sqlite3_stmt, idx: c_int, value: i64) -> io::Result<()> {
    let rc = sqlite3_bind_int64(stmt, idx, value);
    if rc != SQLITE_OK {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("sqlite3_bind_int64 failed with code {rc}"),
        ));
    }
    Ok(())
}

unsafe fn step_done(db: *mut sqlite3, stmt: *mut sqlite3_stmt) -> io::Result<()> {
    let rc = sqlite3_step(stmt);
    if rc != SQLITE_DONE {
        return Err(sqlite_error(db, "sqlite3_step"));
    }
    Ok(())
}

unsafe fn reset_and_clear(db: *mut sqlite3, stmt: *mut sqlite3_stmt) -> io::Result<()> {
    let rc = sqlite3_reset(stmt);
    if rc != SQLITE_OK {
        return Err(sqlite_error(db, "sqlite3_reset"));
    }
    let rc = sqlite3_clear_bindings(stmt);
    if rc != SQLITE_OK {
        return Err(sqlite_error(db, "sqlite3_clear_bindings"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "swoosh-trace-index-{label}-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn member(value: i64, iter_id: u32) -> [u8; 12] {
        let mut member = [0u8; 12];
        member[..8].copy_from_slice(&(value as u64).to_le_bytes());
        member[8..].copy_from_slice(&iter_id.to_le_bytes());
        member
    }

    fn sample_runs(dir: &Path, count: usize) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for index in 0..count {
            let groups = vec![
                RunGroup {
                    key: GlobalKey {
                        kind: KIND_PC,
                        var_id: 0,
                        loc: 10,
                        op: 0,
                    },
                    members: vec![
                        member((index % 7) as i64, (index % 5) as u32),
                        member(index as i64, index as u32),
                    ],
                },
                RunGroup {
                    key: GlobalKey {
                        kind: KIND_OP,
                        var_id: 0,
                        loc: 10,
                        op: OP_WRITE,
                    },
                    members: vec![member(-(index as i64), index as u32)],
                },
            ];
            let path = dir.join(format!("run_{index:08}.swir"));
            write_run_file(&path, &groups).unwrap();
            paths.push(path);
        }
        paths
    }

    #[test]
    fn bounded_merge_is_byte_identical_across_fan_in_limits() {
        let baseline_dir = test_dir("baseline");
        let bounded_dir = test_dir("bounded");
        let baseline_runs = sample_runs(&baseline_dir, 23);
        let bounded_runs = sample_runs(&bounded_dir, 23);

        let baseline = baseline_dir.join("final.swir");
        merge_runs_into_run_atomically(&baseline_runs, &baseline).unwrap();

        let reduced = reduce_runs_to_fan_in(&bounded_runs, &bounded_dir, 3, false).unwrap();
        assert!(reduced.len() <= 3);
        let bounded = bounded_dir.join("final.swir");
        merge_runs_into_run_atomically(&reduced, &bounded).unwrap();

        assert_eq!(fs::read(&baseline).unwrap(), fs::read(&bounded).unwrap());
        fs::remove_dir_all(baseline_dir).unwrap();
        fs::remove_dir_all(bounded_dir).unwrap();
    }

    #[test]
    fn bounded_merge_reclaims_consumed_temporary_runs() {
        let dir = test_dir("reclaim");
        let input_runs = sample_runs(&dir, 9);
        let reduced = reduce_runs_to_fan_in(&input_runs, &dir, 3, true).unwrap();

        assert_eq!(reduced.len(), 3);
        assert!(reduced.iter().all(|path| path.exists()));
        assert!(input_runs.iter().all(|path| !path.exists()));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn corrupt_run_cleans_partial_output_and_publishes_nothing() {
        let dir = test_dir("corrupt");
        let mut runs = sample_runs(&dir, 2);
        let corrupt = dir.join("corrupt.swir");
        fs::write(&corrupt, b"not-a-run").unwrap();
        runs.push(corrupt);

        let output = dir.join("merged.swir");
        let error = merge_runs_into_run_atomically(&runs, &output).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(!output.exists());
        assert!(!dir.join(".merged.swir.partial").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unwind_cleans_partial_run_publication() {
        let dir = test_dir("unwind");
        let partial = dir.join(".merged.swir.partial");
        let result = std::panic::catch_unwind({
            let partial = partial.clone();
            move || {
                let _publication = PartialRunPublication::new(partial.clone());
                fs::write(&partial, b"partial").unwrap();
                panic!("simulated interruption");
            }
        });
        assert!(result.is_err());
        assert!(!partial.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_build_preserves_existing_sqlite_and_cleans_staging() {
        let dir = test_dir("sqlite-publication");
        let raw = dir.join("corrupt.trace.raw.zst");
        let sqlite = dir.join("index.sqlite");
        fs::write(&raw, b"not-a-trace").unwrap();
        fs::write(&sqlite, b"previous-complete-index").unwrap();

        let error = build_trace_index_sqlite(&[raw], &sqlite, Some("test")).unwrap_err();
        assert!(
            error.kind() == io::ErrorKind::InvalidData
                || error.kind() == io::ErrorKind::UnexpectedEof
        );
        assert_eq!(fs::read(&sqlite).unwrap(), b"previous-complete-index");
        let staged: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(".index.sqlite.partial."))
            })
            .collect();
        assert!(staged.is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn merge_fan_in_reserves_descriptors_and_clips_configuration() {
        assert_eq!(merge_fan_in_for_limit(1_024, 256).unwrap(), 256);
        assert_eq!(merge_fan_in_for_limit(64, 256).unwrap(), 32);
        assert_eq!(merge_fan_in_for_limit(64, 8).unwrap(), 8);
        assert!(merge_fan_in_for_limit(33, 8).is_err());
    }
}
