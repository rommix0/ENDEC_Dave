//! `xtask replay` — diff a Rust stage against the saved corpus.
//!
//! The loop every phase after 0 lives in: load each capture, run the Rust
//! stage over the same input, and report the first byte that differs along
//! with which record and field it lands in.
//!
//! No stage is implemented yet, so this reports what the corpus holds and what
//! it would be compared against. That is deliberate: the command exists and is
//! wired up before there is anything to run through it, so the first stage to
//! land has somewhere to be tested.

use loqng::Stage;
use loqng_oracle::Capture;

use crate::capture::corpus_dirs;
use crate::Paths;

pub fn run(p: &Paths, stage: Option<String>) -> Result<(), String> {
    let Some(name) = stage else {
        return Err("which stage? one of: top les fon cat sig acu".to_string());
    };
    let stage = Stage::parse(&name)
        .ok_or_else(|| format!("`{name}` is not a stage; try top les fon cat sig acu"))?;

    let dirs = corpus_dirs(&p.corpus)?;
    if dirs.is_empty() {
        return Err(format!(
            "no captures in {} — run `xtask capture` first",
            p.corpus.display()
        ));
    }

    let mut total_audio = 0usize;
    let mut dumped = 0usize;
    let mut dump_bytes = 0usize;
    let mut phones = 0usize;
    for d in &dirs {
        let c = Capture::load(d).map_err(|e| format!("{}: {e}", d.display()))?;
        total_audio += c.audio.len();
        if let Some(dump) = c.stage(stage) {
            if !dump.is_empty() {
                dumped += 1;
                dump_bytes += dump.len();
                if stage == Stage::Fon {
                    phones += loqng::stream::parse_fon_dump(dump).len();
                }
            }
        }
    }

    println!("stage:   {stage} (LoqTTS6.so+0x{:x})", stage.run_addr());
    println!("corpus:  {} captures in {}", dirs.len(), p.corpus.display());
    println!("         {:.1} MB of audio", total_audio as f64 / 1e6);
    println!(
        "         {dumped}/{} carry a {stage} dump, {:.1} kB total",
        dirs.len(),
        dump_bytes as f64 / 1e3
    );
    if stage == Stage::Fon {
        println!("         {phones} phone records parsed");
    }
    println!();

    if dumped == 0 {
        println!("No {stage} dump in the corpus. Re-run `xtask capture` — the dumps come");
        println!("from ttsSetOutput and are written per stage as `stage-{stage}.txt`.");
        return Ok(());
    }

    match stage {
        Stage::Cat | Stage::Sig | Stage::Acu => {
            println!("Track A: the gate is byte-identical PCM against `audio.raw`,");
            println!("with `stage-{stage}.txt` as the per-stage cross-check.");
        }
        Stage::Top | Stage::Les | Stage::Fon => {
            println!("Track B: the gate is a byte-identical text diff against");
            println!("`stage-{stage}.txt`.");
        }
    }
    println!();
    println!("`loqng::{stage}` is not implemented yet — see PLAN.md for its phase.");
    Ok(())
}
