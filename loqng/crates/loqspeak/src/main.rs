//! `loqspeak` — the Loquendo TTS 6 engine as one native binary.
//!
//! No ARM, no interpreter, no emulator, and — when the voice tree is built
//! in — no files beside the executable. The engine is 180,590 ARM
//! instructions translated to Rust; its output is byte-identical to the
//! original on every capture in the corpus.
//!
//! ```text
//! loqspeak "Hello there." -o hello.wav
//! loqspeak --file script.txt -o out.wav
//! type script.txt | loqspeak --file - -o - > out.wav
//! loqspeak --serve < requests.tsv
//! ```

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

mod engine;
mod wav;

use engine::{Engine, Source};

const USAGE: &str = "\
loqspeak — Loquendo TTS 6, natively

USAGE:
    loqspeak [text] [options]
    loqspeak --file <path>|- [options]
    loqspeak --serve

OPTIONS:
    -o, --out <path>     WAV output; `-` is standard output  [speech.wav]
    -f, --file <path>    Read the text from a file; `-` is standard input
        --raw            Write headerless 16-bit mono PCM instead of a WAV
        --voice <name>   Voice                                [Dave]
        --rate <hz>      Sample rate                          [16000]
        --serve          One request per line on standard input:
                         `<out path><TAB><text>`, one status line per reply
        --lib <dir>      Engine modules, instead of the built-in tree
        --data <dir>     Voice tree, instead of the built-in tree
    -q, --quiet          Only errors
    -h, --help           This
        --version        Version and what is built in

Text is read as Latin-1, not UTF-8: the engine is driven with
`InputTextCoding=ansi`, which is one byte per character.
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("loqspeak: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    text: Option<String>,
    file: Option<String>,
    out: String,
    raw: bool,
    voice: String,
    rate: u32,
    serve: bool,
    lib: Option<PathBuf>,
    data: Option<PathBuf>,
    quiet: bool,
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(());
    }
    if argv.iter().any(|a| a == "--version") {
        println!("loqspeak {}", env!("CARGO_PKG_VERSION"));
        println!(
            "engine: {} modules, {} exports, translated from ARM",
            loqng::eng::MODULES.len(),
            loqng::eng::EXPORTS.len()
        );
        if loqng_voice::present() {
            println!(
                "voice tree: {} files, {:.1} MB built in",
                loqng_voice::FILES.len(),
                loqng_voice::bytes() as f64 / (1024.0 * 1024.0)
            );
        } else {
            println!("voice tree: none built in — pass --lib and --data");
        }
        return Ok(());
    }

    let a = parse(&argv)?;
    let src = source(&a)?;

    if a.serve {
        return serve(&src, &a);
    }

    let text = read_text(&a)?;
    if text.trim().is_empty() {
        return Err("no text to speak (try --help)".into());
    }

    let t0 = std::time::Instant::now();
    let mut eng = Engine::open(&src, &a.voice, a.rate)?;
    let load = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let pcm = eng.say(&text)?;
    let spoke = t1.elapsed().as_secs_f64();
    if pcm.is_empty() {
        let log = eng.log();
        return Err(format!(
            "the engine produced no audio{}",
            if log.is_empty() {
                String::new()
            } else {
                format!("\n{}", log.trim_end())
            }
        ));
    }

    let bytes = if a.raw {
        pcm.clone()
    } else {
        wav::wav(&pcm, a.rate)
    };
    write_out(&a.out, &bytes)?;

    if !a.quiet {
        let secs = pcm.len() as f64 / 2.0 / a.rate as f64;
        eprintln!(
            "{} characters -> {secs:.2} s of audio in {spoke:.2} s \
                   ({:.0}x real time); engine ready in {load:.2} s",
            text.len(),
            secs / spoke.max(1e-9)
        );
    }
    Ok(())
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        text: None,
        file: None,
        out: "speech.wav".into(),
        raw: false,
        voice: "Dave".into(),
        rate: 16000,
        serve: false,
        lib: None,
        data: None,
        quiet: false,
    };
    let mut k = 0;
    while k < argv.len() {
        let arg = argv[k].as_str();
        let mut next = |what: &str| -> Result<String, String> {
            k += 1;
            argv.get(k)
                .cloned()
                .ok_or_else(|| format!("{what} needs a value"))
        };
        match arg {
            "-o" | "--out" => a.out = next("--out")?,
            "-f" | "--file" => a.file = Some(next("--file")?),
            "--voice" => a.voice = next("--voice")?,
            "--rate" => {
                let v = next("--rate")?;
                a.rate = v.parse().map_err(|_| format!("--rate {v}: not a number"))?;
            }
            "--lib" => a.lib = Some(PathBuf::from(next("--lib")?)),
            "--data" => a.data = Some(PathBuf::from(next("--data")?)),
            "--raw" => a.raw = true,
            "--serve" => a.serve = true,
            "-q" | "--quiet" => a.quiet = true,
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown option `{other}` (try --help)"));
            }
            other => a.text = Some(other.to_string()),
        }
        k += 1;
    }
    Ok(a)
}

/// The built-in tree unless a directory was named, and a clear error if
/// neither is available.
fn source(a: &Args) -> Result<Source, String> {
    match (&a.lib, &a.data) {
        (None, None) if loqng_voice::present() => Ok(Source::BuiltIn),
        (None, None) => Err("this build carries no voice tree; pass \
                             --lib <dir> --data <dir>"
            .into()),
        (Some(l), Some(d)) => Ok(Source::Disk(l.clone(), d.clone())),
        _ => Err("--lib and --data go together".into()),
    }
}

/// Latin-1, one byte per character, because that is what the engine is told
/// it will get. A UTF-8 file is decoded as UTF-8 first when it is valid, so
/// the common case still reads correctly.
fn latin1(raw: Vec<u8>) -> String {
    match String::from_utf8(raw) {
        Ok(s) => s,
        Err(e) => e.into_bytes().iter().map(|&b| b as char).collect(),
    }
}

fn read_text(a: &Args) -> Result<String, String> {
    let raw = match a.file.as_deref() {
        Some("-") => {
            let mut v = Vec::new();
            std::io::stdin()
                .read_to_end(&mut v)
                .map_err(|e| format!("standard input: {e}"))?;
            v
        }
        Some(p) => std::fs::read(p).map_err(|e| format!("{p}: {e}"))?,
        None => return Ok(a.text.clone().unwrap_or_default().trim().to_string()),
    };
    Ok(latin1(raw).trim().to_string())
}

fn write_out(path: &str, bytes: &[u8]) -> Result<(), String> {
    if path == "-" {
        let mut o = std::io::stdout().lock();
        o.write_all(bytes)
            .and_then(|_| o.flush())
            .map_err(|e| format!("standard output: {e}"))
    } else {
        std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
    }
}

/// Keep the engine warm and answer one request per line.
///
/// A service pays 0.2 s to load the voice; doing that per utterance is the
/// difference between 6 ms and 200 ms for a short phrase. Input is
/// `<out path><TAB><text>`; output is one line per request, `ok <path>
/// <samples> <seconds>` or `error <message>`, flushed immediately.
fn serve(src: &Source, a: &Args) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let mut eng = Engine::open(src, &a.voice, a.rate)?;
    if !a.quiet {
        eprintln!(
            "loqspeak ready in {:.2} s: {} at {} Hz, {}",
            t0.elapsed().as_secs_f64(),
            a.voice,
            a.rate,
            if matches!(src, Source::BuiltIn) {
                "built-in voice tree"
            } else {
                "voice tree from disk"
            }
        );
    }

    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| format!("standard input: {e}"))?;
    let out = std::io::stdout();
    let mut out = out.lock();

    for line in latin1(input).lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (path, text) = match line.split_once('\t') {
            Some((p, t)) => (p.trim(), t),
            None => {
                writeln!(out, "error no tab in request").ok();
                out.flush().ok();
                continue;
            }
        };
        let t = std::time::Instant::now();
        let reply = eng.say(text).and_then(|pcm| {
            if pcm.is_empty() {
                return Err("no audio".into());
            }
            let bytes = if a.raw {
                pcm.clone()
            } else {
                wav::wav(&pcm, a.rate)
            };
            write_out(path, &bytes)?;
            Ok(pcm.len() / 2)
        });
        match reply {
            Ok(n) => writeln!(out, "ok {path} {n} {:.3}", t.elapsed().as_secs_f64()).ok(),
            Err(e) => writeln!(out, "error {e}").ok(),
        };
        out.flush().ok();
    }
    Ok(())
}
