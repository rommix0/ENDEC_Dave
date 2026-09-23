//! Brings up the Loquendo engine inside the emulator and exposes it to Rust.

use std::path::{Path, PathBuf};

use armemu::Machine;

use crate::libc::Host;

/// Guest-side layout. These paths exist only inside the emulator, so they are
/// stable no matter where the real files live on the host.
pub const GUEST_ROOT: &str = "/loq";
pub const GUEST_LIB: &str = "/loq/lib";
pub const GUEST_DATA: &str = "/loq/data";
pub const GUEST_SESSION: &str = "/loq/default.session";
pub const GUEST_LICENSE: &str = "/loq/LicenseCode.txt";
pub const GUEST_OUT: &str = "/loq/out.raw";

/// The driver the engine ships with. It is the documented way to drive the
/// public tts* API, and using it keeps us byte-for-byte with the original.
pub const DRIVER_NAME: &str = "tts";

/// Every Loquendo 6 voice bank in this family is 16 kHz.
pub const SAMPLE_RATE: u32 = 16000;

/// The engine only emits audio once it has seen a sentence end, so text that
/// stops without one is silently swallowed.
pub const SENTENCE_END: [char; 5] = ['.', ';', ':', '!', '?'];

/// Does this line already end a sentence, looking past any closing quote or
/// bracket?
fn ends_sentence(line: &str) -> bool {
    line.chars()
        .rev()
        .find(|c| !matches!(c, '"' | '\'' | ')' | ']' | '}' | '»' | '”' | '’'))
        .is_some_and(|c| SENTENCE_END.contains(&c))
}

/// Punctuate every line that needs it, so a caller who forgets still gets
/// audio. Applied inside the engine layer, not the command line, so every
/// caller of `speak` and `synthesize` is covered.
pub fn terminate(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for line in text.lines() {
        // Indentation in a script file carries no meaning for the engine.
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push_str(line);
        if !ends_sentence(line) {
            out.push('.');
        }
        out.push('\n');
    }
    out
}

/// Encode for the engine's `InputTextCoding=ansi`: one byte per character.
/// Anything outside Latin-1 has no representation and becomes a question mark.
pub fn to_ansi(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| if (c as u32) < 0x100 { c as u8 } else { b'?' })
        .collect()
}

pub struct EngineConfig {
    /// Host directory holding LoqTTS6.so and the other engine modules.
    pub lib_dir: PathBuf,
    /// Host directory holding the voice tree (the engine's DataPath).
    pub data_dir: PathBuf,
    /// The ARM `tts` driver binary.
    pub driver: PathBuf,
    /// Contents of LicenseCode.txt.
    pub license: Option<String>,
    /// Extra lines appended to the generated session file.
    pub session_extra: Vec<String>,
    pub trace_calls: bool,
    pub trace_vfs: bool,
    pub trace_branches: bool,
    pub breakpoints: Vec<u32>,
    pub watches: Vec<u32>,
    pub dumps: Vec<String>,
    pub trace_api: bool,
    pub profile: bool,
    /// Where to cache the post-`open` snapshot. `None` disables it.
    pub snapshot_dir: Option<PathBuf>,
    /// Replace the codec's hottest routines with native implementations.
    pub native: bool,
}

impl EngineConfig {
    pub fn new(lib_dir: impl AsRef<Path>, data_dir: impl AsRef<Path>) -> Self {
        let lib_dir = lib_dir.as_ref().to_path_buf();
        EngineConfig {
            driver: lib_dir.join(DRIVER_NAME),
            lib_dir,
            data_dir: data_dir.as_ref().to_path_buf(),
            license: None,
            session_extra: Vec::new(),
            trace_calls: false,
            trace_vfs: false,
            trace_branches: false,
            breakpoints: Vec::new(),
            watches: Vec::new(),
            dumps: Vec::new(),
            trace_api: false,
            profile: false,
            snapshot_dir: None,
            native: true,
        }
    }
}

/// What to synthesise and how.
pub struct Request {
    pub text: String,
    pub voice: String,
    /// Speech-database coding module; "loqmsx" for the ENDEC Dave bank.
    pub endec: String,
    /// Extra arguments passed straight to the driver.
    pub extra: Vec<String>,
}

impl Request {
    pub fn new(text: impl Into<String>) -> Self {
        Request {
            text: text.into(),
            voice: "Dave".to_string(),
            endec: "loqmsx".to_string(),
            extra: Vec::new(),
        }
    }
}

/// Live engine handles for the direct API.
struct Handles {
    session: u32,
    instance: u32,
    voice: u32,
    scratch: u32,
    /// Guest allocations kept alive for as long as the session is.
    owned: Vec<u32>,
    /// The previous utterance's text, freed when the next one starts.
    last_text: u32,
}

/// Guest routines replaced by host implementations: module, offset, and the
/// host call to run instead. Adding one means adding a matching arm to
/// `Host::dispatch`.
pub const NATIVE_PATCHES: &[(&str, u32, &str)] = &[
    ("loqmsx.so", 0x2c44, "loqmsx:fir2x"),
    ("loqmsx.so", 0x14ac, "loqmsx:unpack"),
    ("loqmsx.so", 0x3c98, "loqmsx:sigmul"),
    ("loqmsx.so", 0x6500, "loqmsx:lsp2lpc"),
    ("loqmsx.so", 0x22c4, "loqmsx:iir32"),
    ("loqmsx.so", 0x3cd0, "loqmsx:iir16"),
    ("LoqTTS6.so", 0x29f1c, "loqtts:ctxcost"),
    ("LoqTTS6.so", 0x2b014, "loqtts:cands"),
];

pub struct Engine {
    pub machine: Machine,
    pub host: Host,
    driver: PathBuf,
    driver_module: Option<usize>,
    handles: Option<Handles>,
}

impl Engine {
    pub fn new(cfg: &EngineConfig) -> Result<Engine, String> {
        let mut machine = Machine::new();
        let mut host = Self::scaffold(cfg);
        Self::apply_run_flags(cfg, &mut machine, &mut host);
        host.init_data(&mut machine);

        let core = format!("{GUEST_LIB}/LoqTTS6.so");
        let data = host.vfs.read_file(&core).ok_or_else(|| {
            format!(
                "LoqTTS6.so was not found in {} and none is embedded in this build",
                cfg.lib_dir.display()
            )
        })?;
        let core_idx = machine.load("LoqTTS6.so", &data)?;

        if cfg.trace_api {
            let api: Vec<(String, u32)> = machine.modules[0]
                .exported_names()
                .filter(|n| n.starts_with("tts"))
                .map(|n| (n.to_string(), 0))
                .collect();
            for (name, _) in api {
                if let Some(addr) = machine.lookup(&name) {
                    machine.cpu.breakpoints.push(addr);
                    host.api_names.insert(addr, name);
                }
            }
            eprintln!("[api] tracing {} entry points", host.api_names.len());
        }

        let stubs = machine.link(|m, sym| host.resolve_for_link(m, sym))?;
        if cfg.trace_calls && !stubs.is_empty() {
            eprintln!("[link] host-stubbed {} symbols", stubs.len());
        }
        // The core is loaded directly rather than through `dlopen`, so it does
        // not pass the patch point the coding modules do.
        host.patch_native(&mut machine, core_idx)?;
        Ok(Engine {
            machine,
            host,
            driver: cfg.driver.clone(),
            driver_module: None,
            handles: None,
        })
    }

    /// Mounts, overlays and the generated session file: everything about a
    /// session that costs no guest execution. The snapshot restore path uses
    /// this too, and brings its own guest state.
    fn scaffold(cfg: &EngineConfig) -> Host {
        let mut host = Host::new(armemu::HEAP_BASE, armemu::HEAP_LIMIT);
        host.native = cfg.native;
        host.dumps = cfg.dumps.clone();
        host.lib_dir = cfg.lib_dir.clone();

        host.vfs.mount(GUEST_LIB, &cfg.lib_dir);
        host.vfs.mount(GUEST_DATA, &cfg.data_dir);
        host.vfs.mount(GUEST_ROOT, &cfg.lib_dir);
        host.vfs.cwd = GUEST_ROOT.to_string();
        host.vfs.add_sink_path(GUEST_OUT);

        let mut session = format!(
            "\"DataPath\" = \"{GUEST_DATA}\"\n\"LibraryPath\" = \"{GUEST_LIB}\"\n\"LogFile\" = \"stderr\"\n"
        );
        if cfg.license.is_some() {
            session.push_str(&format!("\"LicenseFile\" = \"{GUEST_LICENSE}\"\n"));
        }
        for line in &cfg.session_extra {
            session.push_str(line);
            session.push('\n');
        }
        // Anything not present in lib_dir is served from the binary itself.
        for (name, bytes) in crate::embedded::MODULES {
            if !cfg.lib_dir.join(name).is_file() {
                host.vfs
                    .add_overlay(&format!("{GUEST_LIB}/{name}"), bytes.to_vec());
            }
        }

        host.vfs.add_overlay(GUEST_SESSION, session.into_bytes());
        if let Some(lic) = &cfg.license {
            host.vfs.add_overlay(GUEST_LICENSE, lic.as_bytes().to_vec());
        }
        host
    }

    /// Tracing and profiling belong to this run, never to a snapshot. Every
    /// saved thread context carries its own copy, so they all get set.
    fn apply_run_flags(cfg: &EngineConfig, machine: &mut Machine, host: &mut Host) {
        host.trace_calls = cfg.trace_calls;
        host.native = cfg.native;
        // `scaffold` sets this too, but a snapshot restore brings its own
        // host, so without this line `--dump` prints nothing on the second
        // and every later run.
        host.dumps = cfg.dumps.clone();
        host.vfs.trace = cfg.trace_vfs;
        machine.cpu.history = cfg.trace_branches;
        machine.cpu.breakpoints = cfg.breakpoints.clone();
        machine.cpu.profile = cfg.profile;
        machine.mem.watch = cfg.watches.clone();
        machine.mem.watch_shadow = vec![0; cfg.watches.len()];
        for t in machine.threads.iter_mut() {
            t.cpu.history = cfg.trace_branches;
            t.cpu.profile = cfg.profile;
            t.cpu.breakpoints = cfg.breakpoints.clone();
        }
        machine.clear_samples();
    }

    /// Synthesise `req.text`, returning 16 kHz 16-bit mono little-endian PCM.
    ///
    /// The engine loads its voice bank as part of this call, so pass several
    /// sentences in one request rather than building a second `Engine`.
    pub fn synthesize(&mut self, req: &Request) -> Result<Vec<u8>, String> {
        let text = terminate(&req.text);
        if text.is_empty() {
            return Err("there is nothing to speak".to_string());
        }
        self.host.vfs.set_stdin(to_ansi(&text));
        self.host.vfs.clear_sinks();

        let mut args = vec![
            format!("-I{}", req.endec),
            format!("-v{}", req.voice),
            format!("-Df={GUEST_OUT}"),
        ];
        args.extend(req.extra.iter().cloned());

        let code = self.run_driver(&args)?;
        if code != 0 {
            return Err(format!("the engine driver returned 0x{code:08x}"));
        }
        self.host
            .vfs
            .take_sink(GUEST_OUT)
            .ok_or_else(|| "the engine produced no audio".to_string())
    }

    /// Copy a string into guest memory, NUL terminated.
    fn cstring(&mut self, s: &str) -> Result<u32, String> {
        self.cbytes(s.as_bytes())
    }

    /// Copy raw bytes into guest memory, NUL terminated.
    fn cbytes(&mut self, bytes: &[u8]) -> Result<u32, String> {
        let addr = self
            .host
            .heap
            .malloc(&mut self.machine.mem, bytes.len() as u32 + 1);
        if addr == 0 {
            return Err("out of guest memory".to_string());
        }
        self.machine
            .mem
            .write_cstr(addr, bytes)
            .map_err(|e| e.to_string())?;
        Ok(addr)
    }

    /// Call an exported engine function by name and return its result code.
    fn api(&mut self, name: &str, args: &[u32]) -> Result<u32, String> {
        let addr = self
            .machine
            .lookup(name)
            .ok_or_else(|| format!("{name} is not exported by this engine build"))?;
        self.host.call(&mut self.machine, addr, args)
    }

    /// Same, but a non-zero result is an error.
    fn api_ok(&mut self, name: &str, args: &[u32]) -> Result<(), String> {
        let rc = self.api(name, args)?;
        if rc != 0 {
            return Err(format!("{name} failed with 0x{rc:08x}"));
        }
        Ok(())
    }

    /// Open a session, instance and voice, so `speak` can be called repeatedly.
    ///
    /// This is the expensive step: it loads and indexes the voice bank.
    pub fn open(&mut self, voice: &str, endec: &str, sample_rate: u32) -> Result<(), String> {
        if self.handles.is_some() {
            return Ok(());
        }
        let mut owned = Vec::new();

        // Two output words plus a scratch block for ttsSetModularStructure.
        let out = self.host.heap.malloc(&mut self.machine.mem, 16);
        let scratch = self.host.heap.malloc(&mut self.machine.mem, 512);
        if out == 0 || scratch == 0 {
            return Err("out of guest memory".to_string());
        }
        let _ = self.machine.mem.fill(out, 16, 0);
        let _ = self.machine.mem.fill(scratch, 512, 0);
        owned.push(out);
        owned.push(scratch);

        let empty = self.cstring("")?;
        owned.push(empty);

        self.api_ok("ttsNewSession", &[out, empty])?;
        let session = self.machine.mem.read_u32(out).map_err(|e| e.to_string())?;

        self.api_ok("ttsNewInstance", &[out + 4, session, empty])?;
        let instance = self
            .machine
            .mem
            .read_u32(out + 4)
            .map_err(|e| e.to_string())?;

        for (k, v) in [
            ("MultiSpacePause", "NO"),
            ("MaxParPause", "0"),
            ("InputTextCoding", "ansi"),
        ] {
            let kp = self.cstring(k)?;
            let vp = self.cstring(v)?;
            owned.push(kp);
            owned.push(vp);
            self.api_ok("ttsSetInstanceParam", &[instance, kp, vp])?;
        }

        let vname = self.cstring(voice)?;
        let coding = self.cstring(endec)?;
        owned.push(vname);
        owned.push(coding);
        self.api_ok(
            "ttsNewVoice",
            &[out + 8, instance, vname, sample_rate, coding],
        )?;
        let voice_h = self
            .machine
            .mem
            .read_u32(out + 8)
            .map_err(|e| e.to_string())?;

        let device = self.cstring("LoqAudioFile")?;
        let file = self.cstring(GUEST_OUT)?;
        let linear = self.cstring("l")?;
        owned.push(device);
        owned.push(file);
        owned.push(linear);
        self.host.vfs.add_sink_path(GUEST_OUT);
        self.api_ok("ttsSetAudio", &[instance, device, file, linear, 0])?;

        let top = self.cstring("top")?;
        let acu = self.cstring("acu")?;
        owned.push(top);
        owned.push(acu);
        self.api_ok("ttsSetModularStructure", &[instance, top, acu, scratch])?;

        self.handles = Some(Handles {
            session,
            instance,
            voice: voice_h,
            scratch,
            owned,
            last_text: 0,
        });
        Ok(())
    }

    /// The open instance handle, or `None` before `open`.
    pub fn instance(&self) -> Option<u32> {
        self.handles.as_ref().map(|h| h.instance)
    }

    /// Copy a string into guest memory and keep it alive for the session.
    ///
    /// The engine stores some of the pointers it is handed rather than copying
    /// the bytes, so a caller cannot free these itself.
    pub fn alloc_cstring(&mut self, s: &str) -> Result<u32, String> {
        let p = self.cstring(s)?;
        if let Some(h) = self.handles.as_mut() {
            h.owned.push(p);
        }
        Ok(p)
    }

    /// Call an exported engine function by name and return its result code.
    ///
    /// The engine exports 67 `tts*` entry points and this wrapper binds a
    /// handful; this reaches the rest without adding a method per call.
    pub fn call_api(&mut self, name: &str, args: &[u32]) -> Result<u32, String> {
        self.api(name, args)
    }

    /// Set an engine instance parameter on an already-open session.
    ///
    /// `open` sets the three the driver needs; this exposes the rest, of which
    /// the engine has dozens — `PlainLesOut`, `PlainFonOut`, `TraceSelection`,
    /// `SpellingLevel`, `DefaultNumberType` and so on. The key and value are
    /// kept alive for as long as the session is, because the engine stores the
    /// pointers rather than copying in every case.
    pub fn set_instance_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        let instance = self
            .instance()
            .ok_or_else(|| "no session is open; call open() first".to_string())?;
        let kp = self.cstring(key)?;
        let vp = self.cstring(value)?;
        if let Some(h) = self.handles.as_mut() {
            h.owned.push(kp);
            h.owned.push(vp);
        }
        self.api_ok("ttsSetInstanceParam", &[instance, kp, vp])
    }

    /// Speak one utterance on an already-open session.
    ///
    /// Unlike `synthesize`, this keeps the voice bank loaded, so the second and
    /// later calls cost only their synthesis time.
    pub fn speak(&mut self, text: &str) -> Result<Vec<u8>, String> {
        if self.handles.is_none() {
            return Err("no session is open; call open() first".to_string());
        }
        let instance = self.handles.as_ref().unwrap().instance;

        // The engine has finished with the previous utterance by now.
        let stale = self.handles.as_ref().unwrap().last_text;
        if stale != 0 {
            self.host.heap.free(stale);
        }

        let punctuated = terminate(text);
        if punctuated.is_empty() {
            return Err("there is nothing to speak".to_string());
        }
        let tp = self.cbytes(&to_ansi(punctuated.trim_end()))?;
        self.handles.as_mut().unwrap().last_text = tp;

        let before = self.host.vfs.sink_len(GUEST_OUT);
        // The driver calls ttsRead(h, text, 1, 0, 1, 2).
        self.api_ok("ttsRead", &[instance, tp, 1, 0, 1, 2])?;
        self.host.pump(&mut self.machine)?;

        let pcm = self.host.vfs.sink_since(GUEST_OUT, before);
        if pcm.is_empty() {
            return Err("the engine produced no audio for this utterance".to_string());
        }
        Ok(pcm)
    }

    /// Tear the session down. Dropping the `Engine` is also fine.
    pub fn close(&mut self) -> Result<(), String> {
        let Some(h) = self.handles.take() else {
            return Ok(());
        };
        let _ = self.api("ttsDeleteVoice", &[h.voice]);
        let _ = self.api("ttsDeleteInstance", &[h.instance]);
        let _ = self.api("ttsDeleteSession", &[h.session]);
        for a in h.owned {
            self.host.heap.free(a);
        }
        if h.last_text != 0 {
            self.host.heap.free(h.last_text);
        }
        let _ = h.scratch;
        Ok(())
    }

    /// Run the ARM `tts` driver with arbitrary arguments (diagnostics).
    pub fn run_driver(&mut self, args: &[String]) -> Result<u32, String> {
        let idx = match self.driver_module {
            Some(i) => i,
            None => {
                let guest = format!("{GUEST_LIB}/{DRIVER_NAME}");
                let data = match std::fs::read(&self.driver) {
                    Ok(d) => d,
                    Err(_) => self.host.vfs.read_file(&guest).ok_or_else(|| {
                        format!(
                            "{}: no `tts` driver on disk or embedded",
                            self.driver.display()
                        )
                    })?,
                };
                let i = self.machine.load(DRIVER_NAME, &data)?;
                let stubs = self
                    .machine
                    .link(|m, sym| self.host.resolve_for_link(m, sym))?;
                if self.host.trace_calls && !stubs.is_empty() {
                    eprintln!("[link] driver stubbed: {}", stubs.join(", "));
                }
                self.driver_module = Some(i);
                i
            }
        };

        let main = self.machine.modules[idx]
            .symbol("main")
            .ok_or("the driver has no `main` symbol")?;

        let mut argv_strings = vec![DRIVER_NAME.to_string()];
        argv_strings.extend(args.iter().cloned());
        let argv = self.write_argv(&argv_strings)?;
        let envp = self.write_argv(&[])?;

        self.host.call(
            &mut self.machine,
            main,
            &[argv_strings.len() as u32, argv, envp],
        )
    }

    fn write_argv(&mut self, items: &[String]) -> Result<u32, String> {
        let mut ptrs = Vec::new();
        for s in items {
            let addr = self.machine.alloc_hostdata(s.len() as u32 + 1);
            self.machine
                .mem
                .write_cstr(addr, s.as_bytes())
                .map_err(|e| e.to_string())?;
            ptrs.push(addr);
        }
        let array = self.machine.alloc_hostdata((ptrs.len() as u32 + 1) * 4);
        for (i, p) in ptrs.iter().enumerate() {
            self.machine
                .mem
                .write_u32(array + (i as u32) * 4, *p)
                .map_err(|e| e.to_string())?;
        }
        self.machine
            .mem
            .write_u32(array + (ptrs.len() as u32) * 4, 0)
            .map_err(|e| e.to_string())?;
        Ok(array)
    }

    pub fn modules(&self) -> String {
        self.machine
            .modules
            .iter()
            .map(|m| format!("  {:<20} base 0x{:08x} end 0x{:08x}", m.name, m.base, m.end))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Reset the sample histogram, to profile one phase at a time.
    pub fn profile_reset(&mut self) {
        self.machine.clear_samples();
    }

    pub fn profile_report(&self, top: usize) -> String {
        self.machine.profile_report(top)
    }

    pub fn profile_raw(&self) -> String {
        self.machine.profile_raw()
    }

    pub fn stats(&self) -> String {
        fn pct(part: u64, whole: u64) -> f64 {
            if whole == 0 {
                0.0
            } else {
                100.0 * part as f64 / whole as f64
            }
        }

        // "in compiled blocks", not "native": a compiled block still calls
        // back into the interpreter for every instruction the backend could
        // not emit, and those are counted here too.
        let bails = armemu::block::bail_count();
        let mut out = format!(
            "{} instructions ({:.1}% in compiled blocks, {:.1}% native{}), \
             {} KiB guest RAM, {} allocations (peak {} KiB)",
            self.machine.cpu.ran(),
            pct(self.machine.cpu.jitted, self.machine.cpu.ran()),
            pct(self.machine.cpu.jit_native, self.machine.cpu.ran()),
            if bails == 0 {
                String::new()
            } else {
                format!(", {bails} memory bails")
            },
            self.machine.mem.resident_bytes() / 1024,
            self.host.heap.allocations,
            self.host.heap.peak / 1024
        );

        // What the callbacks inside compiled blocks are, in execution order of
        // importance: the priority list for what the backend should emit next.
        let rejected = armemu::block::rejected();
        let total: u64 = rejected.iter().sum();
        if total > 0 {
            let mut rows: Vec<(&str, u64)> = armemu::block::WHY
                .iter()
                .zip(rejected.iter())
                .map(|((name, _), n)| (*name, *n))
                .filter(|(_, n)| *n > 0)
                .collect();
            rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            out.push_str("\n  callbacks by reason (share of all instructions):");
            for (name, n) in rows {
                out.push_str(&format!(
                    "\n    {name:<26} {:>5.1}%",
                    pct(n, self.machine.cpu.ran())
                ));
            }
        }
        out
    }
}

/// Wrap raw 16-bit mono PCM in a RIFF/WAVE header.
pub fn wav(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() + 44);
    let data_len = pcm.len() as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpunctuated_lines_get_a_full_stop() {
        assert_eq!(terminate("Tornado warning"), "Tornado warning.\n");
        assert_eq!(terminate("Take cover now"), "Take cover now.\n");
    }

    #[test]
    fn existing_punctuation_is_left_alone() {
        for t in ["Already done.", "Really?", "Listen!", "Note:", "Wait;"] {
            assert_eq!(terminate(t), format!("{t}\n"), "{t} was altered");
        }
    }

    #[test]
    fn a_closing_quote_does_not_hide_the_full_stop() {
        assert_eq!(
            terminate("He said \"take cover.\""),
            "He said \"take cover.\"\n"
        );
        assert_eq!(terminate("(see below)"), "(see below).\n");
    }

    #[test]
    fn every_line_is_terminated_and_blanks_are_dropped() {
        assert_eq!(
            terminate("First line\n\n  Second line  \nThird."),
            "First line.\nSecond line.\nThird.\n"
        );
    }

    #[test]
    fn trailing_whitespace_is_not_mistaken_for_missing_punctuation() {
        assert_eq!(terminate("Done.   "), "Done.\n");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(terminate("   \n\n  "), "");
    }

    #[test]
    fn ansi_encoding_is_one_byte_per_character() {
        assert_eq!(to_ansi("abc"), b"abc");
        assert_eq!(to_ansi("caf\u{e9}"), vec![b'c', b'a', b'f', 0xe9]);
        assert_eq!(
            to_ansi("\u{4e2d}"),
            b"?",
            "no Latin-1 form, so a placeholder"
        );
    }
}

// SNAPSHOT-MARK
use armemu::snap::{self, Fnv};
use armemu::{Reader, Snap, SnapResult, Writer};

impl Snap for Handles {
    fn save(&self, w: &mut Writer) {
        w.u32(self.session);
        w.u32(self.instance);
        w.u32(self.voice);
        w.u32(self.scratch);
        w.put(&self.owned);
        w.u32(self.last_text);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(Handles {
            session: r.u32()?,
            instance: r.u32()?,
            voice: r.u32()?,
            scratch: r.u32()?,
            owned: r.get()?,
            last_text: r.u32()?,
        })
    }
}

/// Identifies the inputs a snapshot was built from, so one built against
/// different engine modules or voice data is never loaded.
///
/// Voice data is covered by relative path, size and first 4 KiB rather than in
/// full: hashing 40 MB on every start would cost more than the snapshot saves.
/// Editing a voice file in place without changing its length or its head is
/// the one change this will not notice.
pub fn fingerprint(cfg: &EngineConfig, voice: &str, endec: &str, sample_rate: u32) -> u64 {
    let mut h = Fnv::new();
    h.write_u64(snap::VERSION as u64);
    h.write_str(voice);
    h.write_str(endec);
    h.write_u64(sample_rate as u64);
    h.write_str(&cfg.driver.to_string_lossy());
    h.write_str(cfg.license.as_deref().unwrap_or(""));
    for line in &cfg.session_extra {
        h.write_str(line);
    }
    // The native veneers are written into guest memory during `open`, so a
    // snapshot taken with them is not interchangeable with one taken without.
    h.write_u64(cfg.native as u64);
    for (module, off, name) in NATIVE_PATCHES {
        h.write_str(module);
        h.write_u64(*off as u64);
        h.write_str(name);
    }

    // Whatever would actually be loaded: the file in lib_dir when there is
    // one, otherwise the copy built into this binary.
    for (name, bytes) in crate::embedded::MODULES {
        h.write_str(name);
        match std::fs::read(cfg.lib_dir.join(name)) {
            Ok(on_disk) => h.write(&on_disk),
            Err(_) => h.write(bytes),
        }
    }

    let mut rel = Vec::new();
    walk(&cfg.data_dir, String::new(), &mut rel);
    rel.sort();
    for r in &rel {
        h.write_str(r);
        let path = cfg.data_dir.join(r);
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        h.write_u64(meta.len());
        h.write(&head_of(&path, 4096));
    }
    h.finish()
}

fn walk(dir: &Path, prefix: String, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let rel = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        match e.file_type() {
            Ok(t) if t.is_dir() => walk(&e.path(), rel, out),
            Ok(_) => out.push(rel),
            Err(_) => {}
        }
    }
}

fn head_of(path: &Path, n: u64) -> Vec<u8> {
    use std::io::Read as _;
    let mut buf = Vec::new();
    if let Ok(f) = std::fs::File::open(path) {
        let _ = f.take(n).read_to_end(&mut buf);
    }
    buf
}

/// Replace `path` in one step, so a second process never reads a half-written
/// snapshot and a crash never leaves a truncated one behind.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

impl Engine {
    /// Serialise a machine that has already been through `open`.
    pub fn snapshot(&self, fingerprint: u64) -> Vec<u8> {
        let mut w = Writer::new();
        w.raw(&snap::MAGIC);
        w.u32(snap::VERSION);
        w.u64(fingerprint);
        w.put(&self.machine);
        self.host.save_state(&mut w);
        w.put(&self.driver);
        w.put(&self.driver_module);
        w.put(&self.handles);
        w.buf
    }

    /// Rebuild an engine from `bytes`, which must carry `fingerprint`.
    pub fn from_snapshot(
        cfg: &EngineConfig,
        fingerprint: u64,
        bytes: &[u8],
    ) -> Result<Engine, String> {
        let mut r = Reader::new(bytes);
        if r.take(snap::MAGIC.len())? != snap::MAGIC {
            return Err("not a loqdave snapshot".to_string());
        }
        let version = r.u32()?;
        if version != snap::VERSION {
            return Err(format!(
                "snapshot format is {version}, this build reads {}",
                snap::VERSION
            ));
        }
        let saved = r.u64()?;
        if saved != fingerprint {
            return Err("it was built from different engine modules or voice data".to_string());
        }

        let mut machine: Machine = r.get()?;
        let mut host = Self::scaffold(cfg);
        host.restore_state(&mut r)?;
        let driver = r.get()?;
        let driver_module = r.get()?;
        let handles = r.get()?;
        Self::apply_run_flags(cfg, &mut machine, &mut host);
        Ok(Engine {
            machine,
            host,
            driver,
            driver_module,
            handles,
        })
    }
}

/// Open a session, reusing a cached snapshot when one matches.
///
/// The first call pays the full voice-bank load and writes the snapshot; later
/// calls read it back instead, which is the difference between seconds and
/// milliseconds. The cache is keyed by fingerprint, so a stale snapshot is
/// simply never found and there is nothing to invalidate.
pub fn open_cached(
    cfg: &EngineConfig,
    voice: &str,
    endec: &str,
    sample_rate: u32,
) -> Result<Engine, String> {
    let Some(dir) = cfg.snapshot_dir.clone() else {
        let mut engine = Engine::new(cfg)?;
        engine.open(voice, endec, sample_rate)?;
        return Ok(engine);
    };

    let fp = fingerprint(cfg, voice, endec, sample_rate);
    let path = dir.join(format!("loqdave-{fp:016x}.snap"));
    if let Ok(bytes) = std::fs::read(&path) {
        match Engine::from_snapshot(cfg, fp, &bytes) {
            Ok(engine) => return Ok(engine),
            Err(why) => eprintln!("[snapshot] ignoring {}: {why}", path.display()),
        }
    }

    let mut engine = Engine::new(cfg)?;
    engine.open(voice, endec, sample_rate)?;
    let _ = std::fs::create_dir_all(&dir);
    if let Err(e) = write_atomic(&path, &engine.snapshot(fp)) {
        eprintln!("[snapshot] {} was not written: {e}", path.display());
    }
    Ok(engine)
}
