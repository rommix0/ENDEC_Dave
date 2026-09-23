//! The original ARM engine, wrapped as a reference implementation.
//!
//! This is the only crate that links `loqrs`. Its job is to answer one
//! question for every sentence in the corpus: *what does the real engine do?*
//! — as PCM, and as the intermediate streams the engine can be asked to print.
//!
//! # How the intermediate streams are obtained
//!
//! No patching and no breakpoints. Every stage already has an `XxxWrite` that
//! serialises its stream to text, and the instance parameters that turn them on
//! ship in the binary — `PlainLesOut`, `PlainFonOut`, `TraceChunks`,
//! `TraceWordTranscription`, `TraceSelection`, `TracePhdAtm`. The engine writes
//! them to the session's `TraceFile`, which is pointed at a path the host VFS
//! serves as a sink, so the bytes come back in-process.
//!
//! `TraceFile` is used rather than `LogFile` deliberately: `loqrs`'s session
//! scaffold already writes `"LogFile" = "stderr"` and appends extra lines after
//! it, so overriding `LogFile` would depend on whether the engine's INI reader
//! takes the first or the last duplicate key. `TraceFile` is never set by the
//! scaffold, so there is no duplicate and no ambiguity.
//!
//! Whether a given switch actually routes to `TraceFile` is an empirical
//! question per switch; [`Oracle::probe`] answers it by turning each one on
//! alone and reporting what came back.

use std::path::{Path, PathBuf};

use loqhost::tts::{Engine, EngineConfig};
use loqhost::vfs::Backing;
use loqng::Stage;

/// Guest path the trace sink is mounted at.
pub const TRACE_PATH: &str = "/loq/trace.txt";

/// The VFS slot name `loqrs` gives the guest's stdout.
const STDOUT_SLOT: &str = "<stdout>";

/// The instance parameters that make the engine print its own intermediate
/// representation, paired with the stage each one reports on.
///
/// `Top` has no dump switch of its own; its output is what `Les` receives.
pub const TRACE_SWITCHES: &[(&str, &str, Option<Stage>)] = &[
    ("PlainLesOut", "YES", Some(Stage::Les)),
    ("PlainFonOut", "YES", Some(Stage::Fon)),
    ("TraceChunks", "YES", Some(Stage::Les)),
    ("TraceWordTranscription", "YES", Some(Stage::Fon)),
    ("TraceSelection", "YES", Some(Stage::Cat)),
    ("TracePhdAtm", "YES", None),
];

/// Spellings of "on" to try when probing a switch.
///
/// The engine matches some boolean parameters with `ELQFindValueFromKey`
/// against packed tables — `ytsYTS` at `0x9a0b8`, `nfNF` at `0x9a1ec`,
/// `aulAUL` at `0x9a768` — which read as first-letter sets rather than whole
/// words, so a single character may be what it wants.
pub const PROBE_VALUES: &[&str] = &["YES", "yes", "Y", "y", "1", "TRUE", "ON", "t", "s"];

#[derive(Debug, Clone)]
pub struct OracleConfig {
    pub lib_dir: PathBuf,
    pub data_dir: PathBuf,
    pub voice: String,
    /// Speech-database coding module; `loqmsx` for the ENDEC Dave bank.
    pub endec: String,
    pub sample_rate: u32,
    /// Instance parameters set after `open`, on top of the engine's defaults.
    pub params: Vec<(String, String)>,
    /// Stages to dump via `ttsSetOutput`.
    pub dump_stages: Vec<Stage>,
    /// Let `loqrs` patch its native replacements over the codec's hottest
    /// routines.
    ///
    /// **`xtask verify` must turn this OFF.** Six routines are patched —
    /// `fir2x`, `iir32`, `iir16`, `lsp2lpc`, `sigmul` and `unpack` — so with
    /// it on, calling those addresses runs `loqrs`'s Rust, and a comparison
    /// against `loqng`'s Rust proves only that one transcription matches
    /// another. The ARM is the authority, so the interpreter has to actually
    /// interpret. Everything else (capture, replay) wants it on for speed.
    pub native: bool,
}

impl OracleConfig {
    pub fn new(lib_dir: impl AsRef<Path>, data_dir: impl AsRef<Path>) -> Self {
        OracleConfig {
            lib_dir: lib_dir.as_ref().to_path_buf(),
            data_dir: data_dir.as_ref().to_path_buf(),
            voice: "Dave".to_string(),
            endec: "loqmsx".to_string(),
            sample_rate: 16000,
            params: Vec::new(),
            dump_stages: Vec::new(),
            native: true,
        }
    }

    /// Run the codec's hot routines as interpreted ARM rather than as
    /// `loqrs`'s native replacements. Slower, and the only honest setting for
    /// a differential test — see [`OracleConfig::native`].
    pub fn interpreted(mut self) -> Self {
        self.native = false;
        self
    }

    /// Dump every stage's stream through `ttsSetOutput`.
    pub fn with_all_stage_dumps(mut self) -> Self {
        self.dump_stages = Stage::ALL.to_vec();
        self
    }

    pub fn with_stage_dump(mut self, stage: Stage) -> Self {
        self.dump_stages.push(stage);
        self
    }

    /// Turn on every stream dump the engine offers.
    pub fn with_all_traces(mut self) -> Self {
        for (k, v, _) in TRACE_SWITCHES {
            self.params.push((k.to_string(), v.to_string()));
        }
        self
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.push((key.to_string(), value.to_string()));
        self
    }
}

/// One sentence's worth of reference output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Capture {
    pub text: String,
    /// 16-bit signed little-endian mono PCM at the configured rate.
    pub audio: Vec<u8>,
    /// The engine's event log: `TTSEVT_WORDTRANSCRIPTION`, `INPUT`, audio
    /// bookkeeping. Present without any switch being set.
    pub trace: String,
    /// The guest's stdout. Empty in practice — the engine never writes there —
    /// but captured so that stays a measurement rather than an assumption.
    pub stdout: String,
    /// Per-stage stream dumps, for each stage opened with `ttsSetOutput`.
    pub stages: Vec<(Stage, String)>,
}

impl Capture {
    /// Samples, not bytes.
    pub fn samples(&self) -> usize {
        self.audio.len() / 2
    }

    pub fn duration_secs(&self, rate: u32) -> f64 {
        if rate == 0 {
            return 0.0;
        }
        self.samples() as f64 / rate as f64
    }

    /// Write to a directory as three plain files, so a capture is inspectable
    /// with `cat` and diffable with `git diff`.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("text.txt"), self.text.as_bytes())?;
        std::fs::write(dir.join("audio.raw"), &self.audio)?;
        std::fs::write(dir.join("trace.txt"), self.trace.as_bytes())?;
        std::fs::write(dir.join("stdout.txt"), self.stdout.as_bytes())?;
        for (stage, dump) in &self.stages {
            std::fs::write(dir.join(format!("stage-{stage}.txt")), dump.as_bytes())?;
        }
        Ok(())
    }

    pub fn load(dir: &Path) -> std::io::Result<Capture> {
        Ok(Capture {
            text: String::from_utf8_lossy(&std::fs::read(dir.join("text.txt"))?).into_owned(),
            audio: std::fs::read(dir.join("audio.raw"))?,
            trace: String::from_utf8_lossy(&std::fs::read(dir.join("trace.txt"))?).into_owned(),
            // Added after the first captures were written, so a corpus from
            // before it existed still loads.
            stdout: std::fs::read(dir.join("stdout.txt"))
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_default(),
            stages: Stage::ALL
                .into_iter()
                .filter_map(|s| {
                    std::fs::read(dir.join(format!("stage-{s}.txt")))
                        .ok()
                        .map(|b| (s, String::from_utf8_lossy(&b).into_owned()))
                })
                .collect(),
        })
    }

    /// The dump for one stage, if it was captured.
    pub fn stage(&self, stage: Stage) -> Option<&str> {
        self.stages
            .iter()
            .find(|(s, _)| *s == stage)
            .map(|(_, d)| d.as_str())
    }
}

pub struct Oracle {
    engine: Engine,
    rate: u32,
    /// Bytes already taken from the trace sink, so each capture gets only its
    /// own share.
    trace_mark: usize,
    stdout_mark: usize,
    /// One entry per stage opened with `set_stage_output`, holding how many
    /// bytes of its dump have already been drained.
    stage_marks: Vec<(Stage, usize)>,
}

/// Guest path a stage's dump is written to.
fn stage_path(stage: Stage) -> String {
    format!("/loq/stage-{stage}.txt")
}

impl Oracle {
    /// Load the engine and open a session. This is the expensive call — it
    /// indexes the voice bank — and it is paid once per process, not per
    /// sentence.
    pub fn open(cfg: &OracleConfig) -> Result<Oracle, String> {
        let mut ecfg = EngineConfig::new(&cfg.lib_dir, &cfg.data_dir);
        ecfg.session_extra
            .push(format!("\"TraceFile\" = \"{TRACE_PATH}\""));
        ecfg.native = cfg.native;

        let mut engine = Engine::new(&ecfg)?;
        // Must exist before the engine opens the file, or the write falls
        // through to the host filesystem.
        engine.host.vfs.add_sink_path(TRACE_PATH);
        divert_stdout(&mut engine);

        engine.open(&cfg.voice, &cfg.endec, cfg.sample_rate)?;
        for (k, v) in &cfg.params {
            engine.set_instance_param(k, v)?;
        }

        let trace_mark = engine.host.vfs.sink_len(TRACE_PATH);
        let mut o = Oracle {
            engine,
            rate: cfg.sample_rate,
            trace_mark,
            stdout_mark: 0,
            stage_marks: Vec::new(),
        };
        o.stdout_mark = o.stdout_len();
        for stage in &cfg.dump_stages {
            o.set_stage_output(*stage)?;
        }
        Ok(o)
    }

    pub fn sample_rate(&self) -> u32 {
        self.rate
    }

    /// Speak one sentence and collect everything the engine emitted for it.
    pub fn capture(&mut self, text: &str) -> Result<Capture, String> {
        let audio = self.engine.speak(text)?;
        let trace = self.take_trace();
        let stdout = self.take_stdout();
        let stages = self.take_stage_dumps();
        Ok(Capture {
            text: text.to_string(),
            audio,
            trace,
            stdout,
            stages,
        })
    }

    /// Ask the engine to dump a stage's stream to a file, and capture it.
    ///
    /// `ttsSetOutput(instance, module, writer, sink)` is what actually drives
    /// the `XxxWrite` serialisers — not the `Plain*Out` parameters, which are
    /// read by `ELQSetInstanceValue` but never reach the writer table. The
    /// table lookup at `LoqTTS6.so+0x22344` is called from exactly one place,
    /// `ELQSetOutput+0x6c`.
    ///
    /// Passing `writer = 0` would take the stage's default sink, but that
    /// table (`LoqTTS6.so+0x22398`) is all nulls, and `ELQSetOutput` calls the
    /// result without checking — a null jump, not an error. The engine expects
    /// the caller to supply the sink, and exports one: `ELQDefaultOutputFunction`
    /// is a single entry point with three roles keyed on its arguments —
    /// `(path, 0)` opens and returns a handle, `(text, handle)` writes, and
    /// `(0, handle)` closes.
    ///
    /// The path is registered as a VFS sink first, so the bytes land in memory
    /// rather than on the host filesystem.
    pub fn set_stage_output(&mut self, stage: Stage) -> Result<(), String> {
        let path = stage_path(stage);
        self.engine.host.vfs.add_sink_path(&path);

        let instance = self
            .engine
            .instance()
            .ok_or_else(|| "no session is open".to_string())?;
        let sink = self
            .engine
            .machine
            .lookup("ELQDefaultOutputFunction")
            .ok_or_else(|| {
                "this engine build does not export ELQDefaultOutputFunction".to_string()
            })?;
        let module = self.engine.alloc_cstring(stage.as_str())?;
        let dest = self.engine.alloc_cstring(&path)?;

        let rc = self
            .engine
            .call_api("ttsSetOutput", &[instance, module, sink, dest])?;
        if rc != 0 {
            return Err(format!("ttsSetOutput({stage}) failed with 0x{rc:08x}"));
        }
        self.stage_marks.push((stage, 0));
        Ok(())
    }

    /// Drain every stage dump opened by [`set_stage_output`].
    fn take_stage_dumps(&mut self) -> Vec<(Stage, String)> {
        let mut out = Vec::new();
        for i in 0..self.stage_marks.len() {
            let (stage, from) = self.stage_marks[i];
            let path = stage_path(stage);
            let now = self.engine.host.vfs.sink_len(&path);
            if now > from {
                let bytes = self.engine.host.vfs.sink_since(&path, from);
                out.push((stage, String::from_utf8_lossy(&bytes).into_owned()));
            }
            self.stage_marks[i].1 = now;
        }
        out
    }

    fn stdout_len(&self) -> usize {
        self.engine
            .host
            .vfs
            .files
            .iter()
            .flatten()
            .find(|f| f.path == STDOUT_SLOT)
            .and_then(|f| match &f.backing {
                Backing::Sink { data } => Some(data.len()),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Bytes the guest wrote to stdout since the last drain.
    fn take_stdout(&mut self) -> String {
        let from = self.stdout_mark;
        let slot = self
            .engine
            .host
            .vfs
            .files
            .iter()
            .flatten()
            .find(|f| f.path == STDOUT_SLOT)
            .and_then(|f| match &f.backing {
                Backing::Sink { data } => Some(data.clone()),
                _ => None,
            });
        let Some(data) = slot else {
            return String::new();
        };
        self.stdout_mark = data.len();
        if data.len() <= from {
            return String::new();
        }
        String::from_utf8_lossy(&data[from..]).into_owned()
    }

    /// Bytes written to the trace sink since the last time it was drained.
    fn take_trace(&mut self) -> String {
        let vfs = &mut self.engine.host.vfs;
        let now = vfs.sink_len(TRACE_PATH);
        if now <= self.trace_mark {
            self.trace_mark = now;
            return String::new();
        }
        let bytes = vfs.sink_since(TRACE_PATH, self.trace_mark);
        self.trace_mark = now;
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Escape hatch for anything this wrapper does not expose.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    // ---- calling individual guest routines -------------------------------
    //
    // This is how a Rust port is proved bit-exact: run the ARM function and
    // the Rust one over the same inputs and compare. It needs no patching and
    // no change to `loqrs` — `Host::call`, `Host::heap` and `Machine` are all
    // public already.
    //
    // A module is only in memory once something has loaded it, so `loqmsx.so`
    // requires an open session. That is what `Oracle::open` already does.

    /// Load address of a module, or `None` if nothing has loaded it.
    pub fn module_base(&self, name: &str) -> Option<u32> {
        self.engine.machine.module_base(name)
    }

    /// Reserve guest memory. Not freed; a harness process is short-lived.
    pub fn alloc(&mut self, bytes: u32) -> Result<u32, String> {
        let a = self
            .engine
            .host
            .heap
            .malloc(&mut self.engine.machine.mem, bytes.max(4));
        if a == 0 {
            return Err(format!("out of guest memory asking for {bytes} bytes"));
        }
        self.engine
            .machine
            .mem
            .fill(a, bytes as usize, 0)
            .map_err(|e| e.to_string())?;
        Ok(a)
    }

    /// Call `module + offset`, returning r0.
    ///
    /// Arguments beyond the fourth go on the stack, which `Machine::setup_call`
    /// already handles — so a 6- or 15-argument routine is callable too.
    pub fn call_at(&mut self, module: &str, offset: u32, args: &[u32]) -> Result<u32, String> {
        let base = self
            .module_base(module)
            .ok_or_else(|| format!("{module} is not loaded; open a session first"))?;
        self.engine
            .host
            .call(&mut self.engine.machine, base + offset, args)
    }

    pub fn write_i16s(&mut self, addr: u32, vals: &[i16]) -> Result<(), String> {
        for (i, v) in vals.iter().enumerate() {
            self.engine
                .machine
                .mem
                .write_u16(addr + (i as u32) * 2, *v as u16)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn read_i16s(&self, addr: u32, n: usize) -> Result<Vec<i16>, String> {
        (0..n)
            .map(|i| {
                self.engine
                    .machine
                    .mem
                    .read_u16(addr + (i as u32) * 2)
                    .map(|v| v as i16)
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn write_i32s(&mut self, addr: u32, vals: &[i32]) -> Result<(), String> {
        for (i, v) in vals.iter().enumerate() {
            self.engine
                .machine
                .mem
                .write_u32(addr + (i as u32) * 4, *v as u32)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn read_i32s(&self, addr: u32, n: usize) -> Result<Vec<i32>, String> {
        (0..n)
            .map(|i| {
                self.engine
                    .machine
                    .mem
                    .read_u32(addr + (i as u32) * 4)
                    .map(|v| v as i32)
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn write_bytes(&mut self, addr: u32, vals: &[u8]) -> Result<(), String> {
        self.engine
            .machine
            .mem
            .write_bytes(addr, vals)
            .map_err(|e| e.to_string())
    }

    pub fn read_bytes(&self, addr: u32, n: usize) -> Result<Vec<u8>, String> {
        self.engine
            .machine
            .mem
            .read_bytes(addr, n)
            .map_err(|e| e.to_string())
    }

    pub fn write_u32(&mut self, addr: u32, v: u32) -> Result<(), String> {
        self.engine
            .machine
            .mem
            .write_u32(addr, v)
            .map_err(|e| e.to_string())
    }

    pub fn read_u32(&self, addr: u32) -> Result<u32, String> {
        self.engine
            .machine
            .mem
            .read_u32(addr)
            .map_err(|e| e.to_string())
    }

    pub fn write_u8(&mut self, addr: u32, v: u8) -> Result<(), String> {
        self.engine
            .machine
            .mem
            .write_u8(addr, v)
            .map_err(|e| e.to_string())
    }

    /// Write a struct's worth of words at once, for building guest arguments.
    pub fn write_u32s(&mut self, addr: u32, vals: &[u32]) -> Result<(), String> {
        for (i, v) in vals.iter().enumerate() {
            self.write_u32(addr + (i as u32) * 4, *v)?;
        }
        Ok(())
    }

    /// Determine, empirically, which switches produce output and where.
    ///
    /// Each switch is turned on alone against the same sentence and the trace
    /// sink is measured. A switch that yields nothing either routes somewhere
    /// else — `LogFile`, or straight to stdout via `ELQGetStdout` — or needs a
    /// value other than `YES`. This is a diagnostic, not part of capture.
    pub fn probe(cfg: &OracleConfig, text: &str) -> Result<Vec<ProbeResult>, String> {
        let mut out = Vec::new();

        let base = OracleConfig {
            params: Vec::new(),
            ..cfg.clone()
        };
        let mut o = Oracle::open(&base)?;
        let quiet = o.capture(text)?;
        drop(o);

        for (key, _, stage) in TRACE_SWITCHES {
            // The engine parses some boolean parameters through
            // `ELQFindValueFromKey` against compact tables like `ytsYTS` and
            // `nfNF`, so what counts as "on" is not obvious. Try the plausible
            // spellings and keep whichever produces the most output.
            let mut best = ProbeResult {
                key: key.to_string(),
                stage: *stage,
                trace_bytes: quiet.trace.len(),
                stdout_bytes: quiet.stdout.len(),
                baseline_trace: quiet.trace.len(),
                baseline_stdout: quiet.stdout.len(),
                error: None,
            };

            for value in PROBE_VALUES {
                let one = OracleConfig {
                    params: vec![(key.to_string(), value.to_string())],
                    ..cfg.clone()
                };
                let got = Oracle::open(&one).and_then(|mut o| {
                    let c = o.capture(text)?;
                    Ok((c.trace.len(), c.stdout.len()))
                });
                match got {
                    Ok((t, s)) => {
                        if t + s > best.trace_bytes + best.stdout_bytes {
                            best.trace_bytes = t;
                            best.stdout_bytes = s;
                            best.error = None;
                        }
                    }
                    Err(e) => {
                        if !best.produced_output() {
                            best.error = Some(e);
                        }
                    }
                }
            }
            out.push(best);
        }
        Ok(out)
    }
}

/// Point the guest's stdout at an in-memory sink.
///
/// `loqrs` gives stdout a `Backing::Stdout` that writes straight through to the
/// host, which is right for a CLI and useless for capture. Both the slot list
/// and the backing enum are public, so this needs no change to `loqrs` — the
/// slot is found by the path name it was installed under.
fn divert_stdout(engine: &mut Engine) {
    for f in engine.host.vfs.files.iter_mut().flatten() {
        if f.path == STDOUT_SLOT {
            f.backing = Backing::Sink { data: Vec::new() };
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub key: String,
    pub stage: Option<Stage>,
    /// Trace bytes produced with this switch on and nothing else.
    pub trace_bytes: usize,
    /// Stdout bytes produced with this switch on and nothing else.
    pub stdout_bytes: usize,
    /// Trace bytes produced with no switches at all.
    pub baseline_trace: usize,
    /// Stdout bytes produced with no switches at all.
    pub baseline_stdout: usize,
    pub error: Option<String>,
}

impl ProbeResult {
    pub fn produced_output(&self) -> bool {
        self.error.is_none() && (self.extra_trace() > 0 || self.extra_stdout() > 0)
    }

    pub fn extra_trace(&self) -> usize {
        self.trace_bytes.saturating_sub(self.baseline_trace)
    }

    pub fn extra_stdout(&self) -> usize {
        self.stdout_bytes.saturating_sub(self.baseline_stdout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_round_trips_through_a_directory() {
        let dir = std::env::temp_dir().join("loqng-capture-test");
        let _ = std::fs::remove_dir_all(&dir);
        let c = Capture {
            text: "Testing one two three.".to_string(),
            audio: vec![1, 2, 3, 4],
            trace: "* EVENT: TTSEVT_TEXT\n".to_string(),
            stdout: String::new(),
            stages: vec![
                (Stage::Top, "Testing one two three.\n".to_string()),
                (Stage::Fon, "th\t  79\t 95\t13107\tD F\n".to_string()),
            ],
        };
        c.save(&dir).unwrap();

        let back = Capture::load(&dir).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.stage(Stage::Top), Some("Testing one two three.\n"));
        assert_eq!(back.stage(Stage::Cat), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn duration_comes_from_sample_count() {
        let c = Capture {
            audio: vec![0; 32000],
            ..Default::default()
        };
        assert_eq!(c.samples(), 16000);
        assert!((c.duration_secs(16000) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn every_switch_names_a_real_stage_or_none() {
        for (key, value, stage) in TRACE_SWITCHES {
            assert!(!key.is_empty());
            assert_eq!(*value, "YES");
            if let Some(s) = stage {
                assert!(loqng::Stage::ALL.contains(s));
            }
        }
    }
}
