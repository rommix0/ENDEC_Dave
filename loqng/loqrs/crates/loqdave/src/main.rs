//! loqdave - synthesise speech with a Loquendo 6 voice bank. There is no port
//! of the engine: the original ARM engine runs on an in-process interpreter.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use loqhost::tts::{open_cached, wav, SAMPLE_RATE};
use loqhost::{Engine, EngineConfig};

struct Args {
    lib_dir: PathBuf,
    lib_given: bool,
    data_dir: PathBuf,
    data_given: bool,
    driver: Option<PathBuf>,
    license: Option<PathBuf>,
    voice: String,
    endec: String,
    text: Option<String>,
    file: Option<PathBuf>,
    out: Option<PathBuf>,
    raw: bool,
    lines: bool,
    snapshot_dir: Option<PathBuf>,
    no_snapshot: bool,
    no_native: bool,
    block_align: usize,
    stats: bool,
    trace_calls: bool,
    trace_vfs: bool,
    trace_branches: bool,
    trace_api: bool,
    profile: bool,
    profile_raw: Option<PathBuf>,
    breakpoints: Vec<u32>,
    watches: Vec<u32>,
    dumps: Vec<String>,
    passthrough: Vec<String>,
}

const USAGE: &str = "loqdave - Loquendo TTS on an in-process ARM interpreter

usage:
    loqdave [options] [text...] [outfile]

A trailing argument ending in .wav, .raw or .pcm is the output file, so the
familiar shape works:

    loqdave \"Tornado warning.\" alert.wav

Everything else bare is text to speak. `-o` overrides, and `-` means stdout.
A .raw or .pcm destination implies --raw.

paths (all optional; the engine itself is built into this binary):
    --data <dir>       voice data directory, the engine's DataPath.
                       Default: $LOQ_DATA, then ./data, then /opt/loq/data
    --lib <dir>        load the engine modules from here instead of from
                       this binary. Default: $LOQ_LIB, ./lib, /opt/loq/lib

options:
    --file <path>      read the text to speak from this file (UTF-8, or
                       Latin-1 when that is what it turns out to be)
    --voice <name>     voice to speak with            (default: Dave)
    --endec <name>     speech-database coding module  (default: loqmsx)
    --driver <file>    the ARM `tts` binary           (default: <lib>/tts)
    -o, --out <file>   write audio here; overrides the positional form
                       (default: the trailing positional, else stdout)
    --raw              emit headerless PCM instead of a WAV
    --lines            accepted and ignored; speaking each input line as its
                       own utterance on one session is now the default
    --snapshot <dir>   cache the loaded session here, so later runs skip the
                       voice-bank load. Default: $LOQ_SNAPSHOT, else the
                       system temporary directory
    --no-snapshot      always load the voice bank from scratch
    --no-native        interpret the codec's hot routines instead of running
                       the native replacements (for A/B and for debugging)
    --block-align <n>  drop a trailing partial n-byte block, as the ENDEC does
                       (use 4096 to match the device byte for byte)
    --stats            report interpreter statistics on stderr
    -h, --help         this text

Text comes from --file, or the bare arguments, or stdin. Sentences are
punctuated for you when they are not already, because the engine only emits
audio once it has seen a sentence end.

diagnostics:
    --trace-calls      log every call the guest makes into the host
    --trace-vfs        log guest file, directory and mmap activity
    --trace-branches   keep a control-transfer history for crash reports
    --trace-api        log every Loquendo tts* call with its arguments
    --profile          report where guest instructions were spent
    --profile-raw <f>  write the synthesis samples to <f> as module+offset
                       counts, for attributing them with an external symbol
                       table (the coding modules are stripped)
    --break <hex>      dump registers when the guest reaches this address
    --watch <hex>      report the instruction that changes this guest word
    --dump <spec>      hex dump at each breakpoint, e.g. r6+0x11d0:0x20
    -- <args...>       run the ARM driver with these arguments instead

examples:
    loqdave \"This is a test.\" hello.wav
    loqdave \"Tornado warning.\" | aplay -r 16000 -f S16_LE
    echo 'Tornado warning.' | loqdave alert.wav
    loqdave --file input.txt out.wav
    loqdave --file script.txt --lines script.wav
    loqdave -- -h
";

fn hex(v: &str) -> Result<u32, String> {
    u32::from_str_radix(v.trim_start_matches("0x"), 16).map_err(|e| e.to_string())
}

/// First existing path from an environment variable and then a list of
/// conventional locations. Keeps the common case free of flags.
fn discover(env_var: &str, candidates: &[&str]) -> Option<PathBuf> {
    if let Some(v) = std::env::var_os(env_var) {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

/// Does this bare argument name an output file rather than words to speak?
///
/// Only a recognised audio extension counts, plus `-` for stdout. Anything
/// looser starts mistaking text for filenames, and `-o` is always there.
fn looks_like_output(s: &str) -> bool {
    if s == "-" {
        return true;
    }
    let lower = s.to_ascii_lowercase();
    [".wav", ".raw", ".pcm"].iter().any(|e| lower.ends_with(e))
}

/// Read a script. UTF-8 when it is valid, otherwise Latin-1, which is what a
/// Windows-authored alert file usually is and what the engine's `ansi` input
/// coding expects.
fn read_text_file(path: &PathBuf) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    match String::from_utf8(bytes) {
        Ok(s) => Ok(s),
        Err(e) => Ok(e.into_bytes().iter().map(|b| *b as char).collect()),
    }
}

fn parse() -> Result<Args, String> {
    parse_from(std::env::args().skip(1))
}

fn parse_from(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args {
        lib_dir: PathBuf::new(),
        lib_given: false,
        data_dir: PathBuf::from("data"),
        data_given: false,
        driver: None,
        license: None,
        voice: "Dave".to_string(),
        endec: "loqmsx".to_string(),
        text: None,
        file: None,
        out: None,
        raw: false,
        lines: false,
        snapshot_dir: None,
        no_snapshot: false,
        no_native: false,
        block_align: 0,
        stats: false,
        trace_calls: false,
        trace_vfs: false,
        trace_branches: false,
        trace_api: false,
        profile: false,
        profile_raw: None,
        breakpoints: Vec::new(),
        watches: Vec::new(),
        dumps: Vec::new(),
        passthrough: Vec::new(),
    };
    let mut words: Vec<String> = Vec::new();
    let mut it = args;

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--lib" => {
                a.lib_dir = it.next().ok_or("--lib needs a directory")?.into();
                a.lib_given = true;
            }
            "--data" => {
                a.data_dir = it.next().ok_or("--data needs a directory")?.into();
                a.data_given = true;
            }
            "--file" => a.file = Some(it.next().ok_or("--file needs a path")?.into()),
            "--driver" => a.driver = Some(it.next().ok_or("--driver needs a file")?.into()),
            "--license" => a.license = Some(it.next().ok_or("--license needs a file")?.into()),
            "--voice" => a.voice = it.next().ok_or("--voice needs a name")?,
            "--endec" => a.endec = it.next().ok_or("--endec needs a name")?,
            "-o" | "--out" => a.out = Some(it.next().ok_or("--out needs a file")?.into()),
            "--raw" => a.raw = true,
            "--lines" => a.lines = true,
            "--snapshot" => {
                a.snapshot_dir = Some(it.next().ok_or("--snapshot needs a directory")?.into())
            }
            "--no-snapshot" => a.no_snapshot = true,
            "--no-native" => a.no_native = true,
            "--block-align" => {
                let v = it.next().ok_or("--block-align needs a byte count")?;
                a.block_align = v.parse().map_err(|e| format!("--block-align: {e}"))?;
            }
            "--stats" => a.stats = true,
            "--trace-calls" => a.trace_calls = true,
            "--trace-vfs" => a.trace_vfs = true,
            "--trace-branches" => a.trace_branches = true,
            "--trace-api" => a.trace_api = true,
            "--profile" => a.profile = true,
            "--profile-raw" => {
                a.profile = true;
                a.profile_raw = Some(it.next().ok_or("--profile-raw needs a file")?.into());
            }
            "--break" => a
                .breakpoints
                .push(hex(&it.next().ok_or("--break needs an address")?)?),
            "--watch" => a
                .watches
                .push(hex(&it.next().ok_or("--watch needs an address")?)?),
            "--dump" => a.dumps.push(it.next().ok_or("--dump needs a spec")?),
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--" => {
                a.passthrough.extend(it);
                break;
            }
            other if other.starts_with("--") => return Err(format!("unknown option `{other}`")),
            other => words.push(other.to_string()),
        }
    }
    // Text first, output file last: `loqdave "Tornado warning." alert.wav`.
    // An explicit -o wins, and then every bare word is text.
    if a.out.is_none() {
        if let Some(last) = words.last() {
            if looks_like_output(last) {
                let f = words.pop().expect("checked above");
                a.out = Some(PathBuf::from(f));
            }
        }
    }
    if !words.is_empty() {
        a.text = Some(words.join(" "));
    }
    if a.file.is_some() && a.text.is_some() {
        return Err("--file and inline text cannot both be given".to_string());
    }

    // A .raw or .pcm destination means headerless samples.
    if !a.raw {
        if let Some(p) = &a.out {
            let name = p.to_string_lossy().to_ascii_lowercase();
            a.raw = name.ends_with(".raw") || name.ends_with(".pcm");
        }
    }

    // Nothing on the command line? Fall back to the usual places, so an
    // unpacked directory or the container image works with no flags at all.
    if !a.data_given {
        if let Some(p) = discover("LOQ_DATA", &["data", "/opt/loq/data"]) {
            a.data_dir = p;
        }
    }
    if a.license.is_none() {
        a.license = discover(
            "LOQ_LICENSE",
            &["LicenseCode.txt", "/opt/loq/LicenseCode.txt"],
        );
    }
    if !a.lib_given {
        if let Some(p) = discover("LOQ_LIB", &["lib", "/opt/loq/lib"]) {
            a.lib_dir = p;
        }
    }
    Ok(a)
}

/// Where the post-load snapshot is cached. On by default, because the whole
/// point is that an ordinary invocation is fast; the temporary directory is
/// writable in a container and survives between runs.
fn snapshot_dir(a: &Args) -> Option<PathBuf> {
    if a.no_snapshot {
        return None;
    }
    if let Some(d) = &a.snapshot_dir {
        return Some(d.clone());
    }
    match std::env::var_os("LOQ_SNAPSHOT") {
        Some(v) if v.is_empty() => None,
        Some(v) => Some(PathBuf::from(v)),
        None => Some(std::env::temp_dir()),
    }
}

fn build_config(a: &Args) -> Result<EngineConfig, String> {
    let mut cfg = EngineConfig::new(&a.lib_dir, &a.data_dir);
    if let Some(d) = &a.driver {
        cfg.driver = d.clone();
    }
    cfg.trace_calls = a.trace_calls;
    cfg.trace_vfs = a.trace_vfs;
    cfg.trace_branches = a.trace_branches || !a.breakpoints.is_empty();
    cfg.breakpoints = a.breakpoints.clone();
    cfg.watches = a.watches.clone();
    cfg.dumps = a.dumps.clone();
    cfg.trace_api = a.trace_api;
    cfg.profile = a.profile;
    cfg.snapshot_dir = snapshot_dir(a);
    cfg.native = !a.no_native;
    if let Some(p) = &a.license {
        cfg.license =
            Some(std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?);
    }
    Ok(cfg)
}

fn run(a: &Args) -> Result<(), String> {
    let cfg = build_config(a)?;

    if !a.passthrough.is_empty() {
        let mut engine = Engine::new(&cfg)?;
        let code = engine.run_driver(&a.passthrough)?;
        if a.stats {
            eprintln!("loqdave: driver returned {code}; {}", engine.stats());
        }
        return Ok(());
    }

    let text = match (&a.file, &a.text) {
        (Some(f), _) => read_text_file(f)?,
        (None, Some(t)) => t.clone(),
        (None, None) => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("reading stdin: {e}"))?;
            s
        }
    };
    if text.trim().is_empty() {
        return Err("no text to speak".to_string());
    }

    // The session API, not the vendor driver: it produces byte-identical
    // audio (verified over the whole regression set, single- and multi-line)
    // and it is the path a cached snapshot can restore.
    let started = std::time::Instant::now();
    let mut engine = open_cached(&cfg, &a.voice, &a.endec, SAMPLE_RATE)?;
    if a.stats {
        eprintln!(
            "loqdave: engine ready in {:.3} s",
            started.elapsed().as_secs_f64()
        );
    }

    let result = speak_lines(&mut engine, a, &text);
    if a.stats {
        eprintln!("loqdave: {}", engine.stats());
    }
    let mut pcm = match result {
        Ok(p) => p,
        Err(e) => {
            eprintln!("loqdave: modules:\n{}", engine.modules());
            return Err(e);
        }
    };

    if a.block_align > 1 {
        let keep = pcm.len() / a.block_align * a.block_align;
        if keep < pcm.len() && a.stats {
            eprintln!(
                "loqdave: dropped {} bytes to align to {}",
                pcm.len() - keep,
                a.block_align
            );
        }
        pcm.truncate(keep);
    }

    if a.stats {
        let secs = (pcm.len() as f64) / 2.0 / f64::from(SAMPLE_RATE);
        eprintln!("loqdave: {secs:.3} s of audio");
    }

    let bytes = if a.raw { pcm } else { wav(&pcm, SAMPLE_RATE) };
    match &a.out {
        Some(p) if p.as_os_str() != "-" => {
            std::fs::write(p, &bytes).map_err(|e| format!("{}: {e}", p.display()))?
        }
        _ => std::io::stdout()
            .write_all(&bytes)
            .map_err(|e| format!("writing stdout: {e}"))?,
    }
    Ok(())
}

/// Speak every non-empty line on one session and concatenate the audio.
fn speak_lines(engine: &mut Engine, a: &Args, text: &str) -> Result<Vec<u8>, String> {
    let started = std::time::Instant::now();
    engine.open(&a.voice, &a.endec, SAMPLE_RATE)?;
    if a.stats {
        eprintln!(
            "loqdave: session open in {:.2} s",
            started.elapsed().as_secs_f64()
        );
    }

    if a.profile {
        eprintln!("loqdave: voice-bank load profile:
{}", engine.profile_report(12));
        engine.profile_reset();
    }

    let mut all = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let t = std::time::Instant::now();
        let pcm = engine.speak(line)?;
        if a.stats {
            let elapsed = t.elapsed().as_secs_f64();
            let secs = pcm.len() as f64 / 2.0 / f64::from(SAMPLE_RATE);
            eprintln!(
                "loqdave: {secs:.3} s of audio in {elapsed:.2} s ({:.1}x real time)  {}",
                secs / elapsed.max(1e-9),
                &line[..line.len().min(44)]
            );
        }
        all.extend_from_slice(&pcm);
    }
    if a.profile {
        eprintln!("loqdave: synthesis profile:
{}", engine.profile_report(12));
    }
    if let Some(p) = &a.profile_raw {
        std::fs::write(p, engine.profile_raw())
            .map_err(|e| format!("{}: {e}", p.display()))?;
        eprintln!("loqdave: raw synthesis samples written to {}", p.display());
    }
    engine.close()?;
    if all.is_empty() {
        return Err("nothing was spoken".to_string());
    }
    Ok(all)
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("loqdave: {e}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("loqdave: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Args {
        parse_from(args.iter().map(|s| s.to_string())).expect("parse")
    }

    #[test]
    fn trailing_audio_path_becomes_the_output() {
        let a = parse(&["Tornado warning.", "alert.wav"]);
        assert_eq!(a.text.as_deref(), Some("Tornado warning."));
        assert_eq!(a.out, Some(PathBuf::from("alert.wav")));
        assert!(!a.raw, "a .wav destination should stay a WAV");
    }

    #[test]
    fn unquoted_words_still_join_into_one_sentence() {
        let a = parse(&["Tornado", "warning", "now.", "alert.wav"]);
        assert_eq!(a.text.as_deref(), Some("Tornado warning now."));
        assert_eq!(a.out, Some(PathBuf::from("alert.wav")));
    }

    #[test]
    fn ordinary_words_are_never_mistaken_for_a_filename() {
        let a = parse(&["Hello", "World"]);
        assert_eq!(a.text.as_deref(), Some("Hello World"));
        assert_eq!(a.out, None, "no audio extension, so nothing is an output");
    }

    #[test]
    fn raw_and_pcm_destinations_imply_headerless_output() {
        assert!(parse(&["Testing.", "out.raw"]).raw);
        assert!(parse(&["Testing.", "out.pcm"]).raw);
        assert!(parse(&["-o", "out.RAW", "Testing."]).raw);
    }

    #[test]
    fn explicit_out_flag_wins_and_leaves_positionals_as_text() {
        let a = parse(&["-o", "chosen.wav", "Speak", "this.wav"]);
        assert_eq!(a.out, Some(PathBuf::from("chosen.wav")));
        assert_eq!(a.text.as_deref(), Some("Speak this.wav"));
    }

    #[test]
    fn a_lone_audio_path_leaves_the_text_to_stdin() {
        let a = parse(&["alert.wav"]);
        assert_eq!(a.out, Some(PathBuf::from("alert.wav")));
        assert_eq!(a.text, None, "text should then come from stdin");
    }

    #[test]
    fn dash_means_stdout() {
        let a = parse(&["Testing.", "-"]);
        assert_eq!(a.out, Some(PathBuf::from("-")));
        assert_eq!(a.text.as_deref(), Some("Testing."));
    }

    #[test]
    fn file_input_leaves_the_positional_as_the_output() {
        let a = parse(&["--file", "input.txt", "out.wav"]);
        assert_eq!(a.file, Some(PathBuf::from("input.txt")));
        assert_eq!(a.out, Some(PathBuf::from("out.wav")));
        assert_eq!(a.text, None, "the text comes from the file");
    }

    #[test]
    fn file_and_inline_text_together_are_rejected() {
        let r = parse_from(
            ["--file", "in.txt", "Speak", "this."]
                .iter()
                .map(|s| s.to_string()),
        );
        assert!(r.is_err(), "ambiguous input should be refused, not guessed");
    }

    #[test]
    fn driver_passthrough_is_untouched() {
        let a = parse(&["--", "-h", "-vDave"]);
        assert_eq!(a.passthrough, vec!["-h", "-vDave"]);
        assert_eq!(a.text, None);
    }
}
