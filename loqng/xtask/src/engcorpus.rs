//! `xtask engcorpus` — speak the whole corpus natively and diff the audio.
//!
//! The acceptance gate for the port: every captured sentence, spoken by the
//! translated engine on its own runtime, byte-identical to what the ARM
//! engine produced. Nothing here touches the oracle — it reads the captures
//! off disk and compares.
//!
//! ```text
//! cargo run -p xtask --release -- engcorpus [--limit N]
//! ```
//!
//! Each sentence gets a fresh [`Runtime`], because a session that has spoken
//! is not in its initial state and the point is to reproduce the capture, not
//! to measure warm throughput.

use loqng::engine::{Runtime, GUEST_OUT};

use crate::capture::corpus_dirs;
use crate::Paths;

const RATE: u32 = 16000;

fn speak(paths: &Paths, text: &str) -> Result<Vec<u8>, String> {
    let (mut rt, mut m) = Runtime::new(&paths.lib_dir, &paths.data_dir);
    rt.init(&mut m);

    let api =
        |rt: &mut Runtime, m: &mut loqng::xrt::Mem, name: &str, a: &[u32]| -> Result<(), String> {
            let at = rt.lookup(name).ok_or_else(|| format!("no {name}"))?;
            let rc = rt.call(m, at, a);
            if rc != 0 {
                return Err(format!("{name} returned 0x{rc:08x}"));
            }
            Ok(())
        };

    let out = rt.alloc(16);
    let scratch = rt.alloc(512);
    let empty = rt.cstring(&mut m, "");
    api(&mut rt, &mut m, "ttsNewSession", &[out, empty])?;
    let session = m.r32(out);
    api(
        &mut rt,
        &mut m,
        "ttsNewInstance",
        &[out + 4, session, empty],
    )?;
    let inst = m.r32(out + 4);
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
        &[out + 8, inst, vname, RATE, endec],
    )?;
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

    let tp = rt.cstring(&mut m, text);
    api(&mut rt, &mut m, "ttsRead", &[inst, tp, 1, 0, 1, 2])?;
    rt.pump();
    Ok(rt.sink(GUEST_OUT))
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let limit: usize = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|k| args.get(k + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let dirs = corpus_dirs(&paths.corpus)?;
    println!("{} captures under {}", dirs.len(), paths.corpus.display());

    let (mut ok, mut bad, mut missing) = (0usize, 0usize, 0usize);
    let mut samples_ok = 0u64;
    let t0 = std::time::Instant::now();
    for dir in dirs.iter().take(limit) {
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        let text = match std::fs::read_to_string(dir.join("text.txt")) {
            Ok(t) => t.trim().to_string(),
            Err(_) => {
                missing += 1;
                continue;
            }
        };
        let want = match std::fs::read(dir.join("audio.raw")) {
            Ok(v) => v,
            Err(_) => {
                missing += 1;
                continue;
            }
        };
        let got = match speak(paths, &text) {
            Ok(v) => v,
            Err(e) => {
                bad += 1;
                println!("  {name}  {text:?}\n    failed: {e}");
                continue;
            }
        };
        if got == want {
            ok += 1;
            samples_ok += got.len() as u64 / 2;
            continue;
        }
        bad += 1;
        let at = got.iter().zip(want.iter()).position(|(a, b)| a != b);
        println!("  {name}  {text:?}");
        println!(
            "    native {} bytes, capture {} bytes, first difference {:?}",
            got.len(),
            want.len(),
            at
        );
    }

    println!(
        "\n{ok} exact, {bad} differ, {missing} skipped in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    println!(
        "{samples_ok} samples = {:.1} s of audio byte-identical to the \
              ARM engine",
        samples_ok as f64 / RATE as f64
    );
    if bad > 0 {
        return Err(format!("{bad} capture(s) differ"));
    }
    println!("\nBIT-EXACT: the native engine reproduces the ARM engine");
    Ok(())
}
