//! `xtask engrun` — run the translated engine and make it speak.
//!
//! This is the end of the port: no ARM, no interpreter, no oracle. The engine
//! is [`loqng::eng`] (translated by `tools/engxlate.py`) running on
//! [`loqng::engine::Runtime`], reading nothing but the voice tree.
//!
//! The call sequence is the documented one, taken from `loqrs`'s
//! `Engine::open`, which is byte-identical to the vendor driver:
//!
//! ```text
//! ttsNewSession(&h, "")            ttsNewVoice(&h, inst, voice, rate, endec)
//! ttsNewInstance(&h, session, "")  ttsSetAudio(inst, "LoqAudioFile", f, "l", 0)
//! ttsSetInstanceParam(...)         ttsSetModularStructure(inst, "top", "acu", buf)
//! ttsRead(inst, text, 1, 0, 1, 2)
//! ```
//!
//! ```text
//! cargo run -p xtask --release -- engrun "text"            [--out speech.raw]
//! cargo run -p xtask --release -- engrun --file script.txt [--out speech.raw]
//! type script.txt | cargo run -p xtask --release -- engrun --file -
//! ```
//!
//! A command line is capped near 32,767 characters on Windows, which is an
//! OS limit and not the engine's — `--file` has no such ceiling, and `-`
//! reads standard input.

use loqng::engine::{Runtime, GUEST_OUT};

use crate::Paths;

const RATE: u32 = 16000;

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|k| args.get(k + 1).cloned());
    let from = args
        .iter()
        .position(|a| a == "--file")
        .and_then(|k| args.get(k + 1).cloned());

    let text = match from.as_deref() {
        Some("-") => {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("reading standard input: {e}"))?;
            s
        }
        Some(p) => {
            // Latin-1 rather than UTF-8: the engine is told
            // `InputTextCoding=ansi`, so it wants one byte per character.
            let raw = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            match String::from_utf8(raw) {
                Ok(s) => s,
                Err(e) => e.into_bytes().iter().map(|&b| b as char).collect(),
            }
        }
        None => {
            let mut k = 0;
            let mut said = None;
            while k < args.len() {
                match args[k].as_str() {
                    "--out" | "--file" => k += 1,
                    a if a.starts_with("--") => {}
                    a => {
                        said = Some(a.to_string());
                        break;
                    }
                }
                k += 1;
            }
            said.unwrap_or_else(|| "Testing one two three.".to_string())
        }
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("no text to speak".into());
    }

    println!("lib  {}", paths.lib_dir.display());
    println!("data {}", paths.data_dir.display());
    let (mut rt, mut m) = Runtime::new(&paths.lib_dir, &paths.data_dir);
    println!(
        "image mapped, {} exports, {} constructors",
        loqng::eng::EXPORTS.len(),
        loqng::eng::INIT.len()
    );

    let t0 = std::time::Instant::now();
    rt.init(&mut m);
    println!("constructors ran in {:.2} s", t0.elapsed().as_secs_f64());

    let api =
        |rt: &mut Runtime, m: &mut loqng::xrt::Mem, name: &str, a: &[u32]| -> Result<u32, String> {
            let at = rt
                .lookup(name)
                .ok_or_else(|| format!("{name} is not exported by the engine"))?;
            let rc = rt.call(m, at, a);
            if rc != 0 {
                if !rt.log.is_empty() {
                    println!("\nengine log:\n{}", String::from_utf8_lossy(&rt.log));
                }
                return Err(format!("{name} returned 0x{rc:08x}"));
            }
            Ok(rc)
        };

    // Four output words plus the scratch block `ttsSetModularStructure` wants.
    let out_words = rt.alloc(16);
    let scratch = rt.alloc(512);
    let empty = rt.cstring(&mut m, "");

    let t0 = std::time::Instant::now();
    api(&mut rt, &mut m, "ttsNewSession", &[out_words, empty])?;
    let session = m.r32(out_words);
    println!("session 0x{session:08x}");

    api(
        &mut rt,
        &mut m,
        "ttsNewInstance",
        &[out_words + 4, session, empty],
    )?;
    let inst = m.r32(out_words + 4);
    println!("instance 0x{inst:08x}");

    for (k, v) in [
        ("MultiSpacePause", "NO"),
        ("MaxParPause", "0"),
        ("InputTextCoding", "ansi"),
    ] {
        let kp = rt.cstring(&mut m, k);
        let vp = rt.cstring(&mut m, v);
        api(&mut rt, &mut m, "ttsSetInstanceParam", &[inst, kp, vp])?;
    }

    let vname = rt.cstring(&mut m, "Dave");
    let endec = rt.cstring(&mut m, "loqmsx");
    api(
        &mut rt,
        &mut m,
        "ttsNewVoice",
        &[out_words + 8, inst, vname, RATE, endec],
    )?;
    println!("voice loaded in {:.2} s", t0.elapsed().as_secs_f64());

    let device = rt.cstring(&mut m, "LoqAudioFile");
    let file = rt.cstring(&mut m, GUEST_OUT);
    let linear = rt.cstring(&mut m, "l");
    api(
        &mut rt,
        &mut m,
        "ttsSetAudio",
        &[inst, device, file, linear, 0],
    )?;

    let top = rt.cstring(&mut m, "top");
    let acu = rt.cstring(&mut m, "acu");
    api(
        &mut rt,
        &mut m,
        "ttsSetModularStructure",
        &[inst, top, acu, scratch],
    )?;

    // The engine emits nothing until it sees a sentence end.
    let mut say = text.clone();
    if !say.trim_end().ends_with(['.', '!', '?', ';', ':']) {
        say.push('.');
    }
    println!("speaking {} characters", say.len());
    let tp = rt.cstring(&mut m, &say);
    let t0 = std::time::Instant::now();
    api(&mut rt, &mut m, "ttsRead", &[inst, tp, 1, 0, 1, 2])?;
    // `ttsRead` queues the utterance; the worker renders it.
    rt.pump();
    let secs = t0.elapsed().as_secs_f64();

    let audio = rt.sink(GUEST_OUT);
    let shown: String = say.chars().take(60).collect();
    println!(
        "\nspoke {shown:?}{} in {secs:.2} s",
        if say.chars().count() > 60 { " ..." } else { "" }
    );
    println!(
        "{} worker thread(s) requested, {} waits on a condition variable",
        rt.deferred.len(),
        rt.cond_waits
    );
    println!(
        "{} bytes = {} samples = {:.2} s of audio at {RATE} Hz",
        audio.len(),
        audio.len() / 2,
        audio.len() as f64 / 2.0 / RATE as f64
    );
    if !rt.log.is_empty() {
        println!("\nengine log:\n{}", String::from_utf8_lossy(&rt.log));
    }
    if let Some(p) = out {
        std::fs::write(&p, &audio).map_err(|e| format!("{p}: {e}"))?;
        println!("wrote {p}");
    }
    if audio.is_empty() {
        return Err("the engine produced no audio".into());
    }
    Ok(())
}
