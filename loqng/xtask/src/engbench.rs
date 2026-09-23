//! `xtask engbench` — exact timings, scaling, and the length ceiling.
//!
//! "About nine times real time" is an average over a corpus and hides the
//! shape. What matters for a service is two numbers: the **fixed cost of an
//! utterance** and the **marginal cost per second of audio**. This measures
//! both, over phrase lengths from one word to tens of thousands of
//! characters, and then finds where the engine actually stops.
//!
//! ```text
//! cargo run -p xtask --release -- engbench            # timings and scaling
//! cargo run -p xtask --release -- engbench --limit    # how long can a phrase be
//! ```
//!
//! Every timing is on a **warm** engine — one voice load, many utterances —
//! because that is how a service runs. The voice load is reported separately.

use loqng::engine::{Runtime, GUEST_OUT};
use loqng::xrt::Mem;

use crate::Paths;

const RATE: f64 = 16000.0;

/// Boxed, and boxed *before* the engine runs, because a worker thread holds
/// raw pointers to the runtime and its memory. Moving either after the
/// thread exists leaves it pointing at the old address.
pub(crate) struct Engine {
    rt: Box<Runtime>,
    m: Box<Mem>,
    inst: u32,
}

impl Engine {
    pub(crate) fn open(paths: &Paths) -> Result<(Engine, f64), String> {
        let t0 = std::time::Instant::now();
        let (rt, m) = Runtime::new(&paths.lib_dir, &paths.data_dir);
        let (mut rt, mut m) = (Box::new(rt), Box::new(m));
        rt.init(&mut m);

        let call = |rt: &mut Runtime, m: &mut Mem, name: &str, a: &[u32]| -> Result<(), String> {
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
        call(&mut rt, &mut m, "ttsNewSession", &[out, empty])?;
        let session = m.r32(out);
        call(
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
            call(&mut rt, &mut m, "ttsSetInstanceParam", &[inst, kp, vp])?;
        }
        let vname = rt.cstring(&mut m, "Dave");
        let endec = rt.cstring(&mut m, "loqmsx");
        call(
            &mut rt,
            &mut m,
            "ttsNewVoice",
            &[out + 8, inst, vname, 16000, endec],
        )?;
        let device = rt.cstring(&mut m, "LoqAudioFile");
        let file = rt.cstring(&mut m, GUEST_OUT);
        let linear = rt.cstring(&mut m, "l");
        call(
            &mut rt,
            &mut m,
            "ttsSetAudio",
            &[inst, device, file, linear, 0],
        )?;
        let top = rt.cstring(&mut m, "top");
        let acu = rt.cstring(&mut m, "acu");
        call(
            &mut rt,
            &mut m,
            "ttsSetModularStructure",
            &[inst, top, acu, scratch],
        )?;
        let load = t0.elapsed().as_secs_f64();
        Ok((Engine { rt, m, inst }, load))
    }

    /// Speak once. Returns `(wall seconds, bytes of audio)`.
    pub(crate) fn say(&mut self, text: &str) -> Result<(f64, usize), String> {
        let tp = self.rt.cstring(&mut self.m, text);
        let at = self.rt.lookup("ttsRead").ok_or("no ttsRead")?;
        let inst = self.inst;
        let t0 = std::time::Instant::now();
        let rc = self.rt.call(&mut self.m, at, &[inst, tp, 1, 0, 1, 2]);
        self.rt.pump();
        let dt = t0.elapsed().as_secs_f64();
        if rc != 0 {
            return Err(format!("ttsRead returned 0x{rc:08x}"));
        }
        Ok((dt, self.rt.take_sink(GUEST_OUT).len()))
    }
}

/// Real sentences, so the front end does real work rather than repeating one
/// cached lookup. Taken from the corpus texts.
const SENTENCES: &[&str] = &[
    "Testing one two three.",
    "The quick brown fox jumps over the lazy dog.",
    "He finished third out of twenty one.",
    "Please hold while I transfer your call to the next available agent.",
    "The repair cost one thousand dollars and the part alone was eighty nine.",
    "No, not at all, though it may seem that way at first glance.",
    "Email support at example dot com or visit the website today.",
    "She walked down the street and noticed the shop had closed early.",
];

pub(crate) fn build(chars: usize) -> String {
    let mut s = String::new();
    let mut k = 0;
    while s.len() < chars {
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(SENTENCES[k % SENTENCES.len()]);
        k += 1;
    }
    s
}

/// One enormous sentence: no full stop until the very end, so the engine
/// cannot flush early and must hold the whole thing.
fn one_sentence(chars: usize) -> String {
    let mut s = String::from("The list contains");
    let words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    ];
    let mut k = 0;
    while s.len() < chars {
        s.push(' ');
        s.push_str(words[k % words.len()]);
        k += 1;
    }
    s.push('.');
    s
}

fn run_limit(paths: &Paths, single: bool, budget: f64, from: usize) -> Result<(), String> {
    println!(
        "\n{} — doubling until it fails",
        if single {
            "ONE SENTENCE (no early flush)"
        } else {
            "MANY SENTENCES"
        }
    );
    println!(
        "  {:>9}  {:>9}  {:>10}  {:>9}  {:>11}  {}",
        "chars", "audio s", "wall s", "x real", "audio s/1k", "result"
    );
    let mut chars = from;
    let mut last_ok = 0usize;
    let mut last_ratio = 0.0f64;
    while chars <= 8 << 20 {
        let text = if single {
            one_sentence(chars)
        } else {
            build(chars)
        };
        // A fresh engine each time: a 4 MB utterance would otherwise exhaust
        // the guest heap through text that the engine keeps pointers into.
        let (mut e, _) = Engine::open(paths)?;
        let n = text.len();
        match e.say(&text) {
            Ok((dt, bytes)) => {
                let audio = bytes as f64 / 2.0 / RATE;
                let ratio = audio * 1000.0 / n as f64;
                // A drop here means the engine swallowed text rather than
                // failing: the same input producing proportionally less
                // audio is truncation, and it would be silent otherwise.
                let note = if last_ratio > 0.0 && ratio < last_ratio * 0.9 {
                    "TRUNCATED?"
                } else {
                    "ok"
                };
                println!(
                    "  {n:>9}  {audio:>9.2}  {dt:>10.3}  {:>9.2}  \
                          {ratio:>11.3}  {note}",
                    if dt > 0.0 { audio / dt } else { 0.0 }
                );
                if bytes == 0 {
                    println!("  no audio at {n} characters — stopping");
                    break;
                }
                last_ok = n;
                last_ratio = ratio;
                if dt > budget {
                    println!(
                        "  stopping on a time budget, not a failure: \
                              {n} characters still works"
                    );
                    break;
                }
            }
            Err(e) => {
                println!(
                    "  {n:>9}  {:>9}  {:>10}  {:>9}  {:>11}  FAILED: {e}",
                    "-", "-", "-", "-"
                );
                break;
            }
        }
        chars *= 2;
    }
    println!("  largest that worked: {last_ok} characters");
    Ok(())
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--limit") {
        let budget: f64 = args
            .iter()
            .position(|a| a == "--budget")
            .and_then(|k| args.get(k + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(45.0);
        let from: usize = args
            .iter()
            .position(|a| a == "--from")
            .and_then(|k| args.get(k + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        run_limit(paths, false, budget, from)?;
        run_limit(paths, true, budget, from)?;
        return Ok(());
    }

    let (mut e, load) = Engine::open(paths)?;
    println!("voice load: {load:.3} s (once per process)\n");

    // The first few are literal, because building from whole sentences
    // cannot produce anything shorter than one sentence — and the shortest
    // utterance is exactly what measures the fixed cost.
    let sizes: &[(&str, &str, usize)] = &[
        ("one word", "Yes.", 0),
        ("two words", "Thank you.", 0),
        ("tiny", "Please hold the line.", 0),
        ("short", "", 60),
        ("medium", "", 250),
        ("long", "", 1000),
        ("extra long", "", 4000),
        ("huge", "", 16000),
    ];

    // A first utterance warms caches the rest benefit from; measuring it
    // with the others would blame its cost on whichever size ran first.
    let _ = e.say("Warming up.")?;

    println!(
        "  {:<11} {:>7} {:>9} {:>10} {:>10} {:>8} {:>10}",
        "size", "chars", "audio s", "wall ms", "best ms", "x real", "ms/audio s"
    );
    let mut points: Vec<(f64, f64)> = Vec::new();
    for (name, literal, chars) in sizes {
        let text = if literal.is_empty() {
            build(*chars)
        } else {
            literal.to_string()
        };
        let reps = if text.len() > 2000 { 2 } else { 5 };
        let mut times = Vec::new();
        let mut bytes = 0;
        for _ in 0..reps {
            let (dt, b) = e.say(&text)?;
            times.push(dt);
            bytes = b;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best = times[0];
        let median = times[times.len() / 2];
        let audio = bytes as f64 / 2.0 / RATE;
        points.push((audio, best));
        println!(
            "  {name:<11} {:>7} {audio:>9.3} {:>10.2} {:>10.2} {:>8.2} {:>10.2}",
            text.len(),
            median * 1e3,
            best * 1e3,
            if best > 0.0 { audio / best } else { 0.0 },
            if audio > 0.0 { best * 1e3 / audio } else { 0.0 }
        );
    }

    // Least squares on wall = fixed + marginal * audio.
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let marginal = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let fixed = (sy - marginal * sx) / n;

    let floor = points.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    println!(
        "\nwall = {:.1} ms fixed + {:.1} ms per second of audio (least squares)",
        fixed * 1e3,
        marginal * 1e3
    );
    println!(
        "marginal rate {:.1}x real time; shortest utterance measured \
              {:.1} ms end to end",
        1.0 / marginal,
        floor * 1e3
    );
    println!(
        "\nlinearity check (a flat last column means it scales with \
              length and nothing worse):"
    );
    println!("  {:>9}  {:>12}", "audio s", "ms/audio s");
    for (audio, best) in &points {
        if *audio > 0.0 {
            println!("  {audio:>9.3}  {:>12.2}", best * 1e3 / audio);
        }
    }
    Ok(())
}
