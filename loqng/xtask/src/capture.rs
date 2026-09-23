//! `xtask capture` — speak the corpus through the ARM oracle and save it.
//!
//! One session is opened and every sentence is spoken on it, so the expensive
//! voice-bank load is paid once rather than per sentence.
//!
//! Each case lands in its own directory named by a hash of its text, so a
//! capture is stable across runs and reorderings, and re-running only rewrites
//! what changed.

use std::path::{Path, PathBuf};

use loqng_oracle::{Oracle, OracleConfig};

use crate::{flag, Paths};

/// Sentences that exercise the parts of `Les` and `Fon` that break in
/// production: numbers in every form, dates, currency, acronyms, addresses and
/// abbreviations. Kept small enough to run in seconds; `--texts` supplies the
/// full 50,000-sentence set.
const BUILTIN: &[&str] = &[
    "Testing one two three.",
    "The quick brown fox jumped over the lazy dog.",
    "She sells sea shells by the sea shore.",
    // Integers, ordinals, ranges.
    "There were 7 people, 42 chairs and 1,058 empty seats.",
    "He finished 3rd out of 21, ahead of the 4th and 5th place runners.",
    "Count from 11 to 19, then 20, 30, 40 and 100.",
    "The population reached 8,100,000,000 in 2024.",
    "Chapter 4, section 17, paragraph 3.",
    // Currency.
    "The repair cost $1,247.50 and the part alone was $89.",
    "She paid 247 pounds for it, or about 310 euros.",
    "A total of $0.99, plus $12 shipping.",
    // Dates and times.
    "The meeting is on March 3rd, 2026 at 4:30 PM.",
    "It happened on 11/12/2001, a Tuesday.",
    "Set your clocks back at 2:00 AM on November 1st.",
    "From 1939 to 1945, and again in 1968.",
    // Phone numbers and identifiers.
    "Call 555-0142 or 1-800-555-0199 for details.",
    "The tracking number is 1Z999AA10123456784.",
    // Acronyms and abbreviations.
    "The NOAA and the NWS issued a joint statement.",
    "Dr. Smith of St. Mary's Hospital on Rd. 9 said so.",
    "The CEO of the FBI met the head of NASA at 10 a.m.",
    "Approx. 12 in. of snow fell, i.e. about 30 cm.",
    // Addresses.
    "Send it to 1600 Pennsylvania Ave NW, Washington, DC 20500.",
    "They live at 42 Elm St., Apt. 3B.",
    // Symbols and mixed content.
    "The rate rose 3.5% to 12.75%, a 20-year high.",
    "Temperatures of -4 degrees and +18 degrees were recorded.",
    "Use the A/B test, and/or the control group.",
    "Email support@example.com or visit www.example.com today.",
    // Punctuation and structure.
    "Wait -- what happened? Nothing; everything is fine!",
    "He said \"take cover.\" Then the line went dead.",
    "One (1) item, two (2) items, three (3) items.",
    // Alert-style text, the shape this voice is actually used for.
    "The National Weather Service has issued a Tornado Warning.",
    "A Severe Thunderstorm Warning remains in effect until 9:15 PM.",
    "This is a test of the Emergency Alert System. This is only a test.",
    "Residents of Jefferson County should seek shelter immediately.",
    "The warning covers Adams, Boone and Clark counties until 11 PM.",
    // Long and short.
    "Yes.",
    "No, not at all, though it may seem that way at first glance to anyone who has not been following the matter closely for the last several years.",
];

pub fn run(p: &Paths, args: &[String]) -> Result<(), String> {
    let texts = match flag(args, "--texts") {
        Some(f) => read_texts(Path::new(&f))?,
        None => BUILTIN.iter().map(|s| s.to_string()).collect(),
    };
    let limit = flag(args, "--limit")
        .map(|s| s.parse::<usize>().map_err(|e| format!("--limit: {e}")))
        .transpose()?
        .unwrap_or(usize::MAX);

    let texts: Vec<String> = texts.into_iter().take(limit).collect();
    if texts.is_empty() {
        return Err("nothing to capture".to_string());
    }

    let out = p.corpus.join("captures");
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

    println!("opening the ARM engine (this indexes the voice bank; ~2.5 s)");
    let cfg = OracleConfig::new(&p.lib_dir, &p.data_dir).with_all_stage_dumps();
    let started = std::time::Instant::now();
    let mut oracle = Oracle::open(&cfg)?;
    println!("ready in {:.2} s", started.elapsed().as_secs_f64());
    println!();

    let rate = oracle.sample_rate();
    let mut audio_secs = 0.0;
    let mut traced = 0usize;
    let run = std::time::Instant::now();

    for (i, text) in texts.iter().enumerate() {
        let c = oracle.capture(text)?;
        audio_secs += c.duration_secs(rate);
        if !c.trace.is_empty() {
            traced += 1;
        }
        let dir = out.join(case_name(i, text));
        c.save(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let dumps: Vec<String> = c
            .stages
            .iter()
            .filter(|(_, d)| !d.is_empty())
            .map(|(s, d)| format!("{s}:{}", d.len()))
            .collect();
        println!(
            "  [{:>3}/{}] {:>6.2}s  {:<28} {}",
            i + 1,
            texts.len(),
            c.duration_secs(rate),
            dumps.join(" "),
            short(text)
        );
    }

    let wall = run.elapsed().as_secs_f64();
    println!();
    println!(
        "{} captures, {:.1} s of audio in {:.1} s ({:.1}x real time)",
        texts.len(),
        audio_secs,
        wall,
        if wall > 0.0 { audio_secs / wall } else { 0.0 }
    );
    println!("{traced}/{} produced trace output", texts.len());
    println!("written to {}", out.display());

    if traced == 0 {
        println!();
        println!("No stage dumps came back. Track A (audio) is still gated, but");
        println!("Track B needs these — run `xtask probe` to find out why.");
    }
    Ok(())
}

/// Capture directories, sorted, so replay order is stable.
pub fn corpus_dirs(corpus: &Path) -> Result<Vec<PathBuf>, String> {
    let root = corpus.join("captures");
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("text.txt").is_file())
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn read_texts(f: &Path) -> Result<Vec<String>, String> {
    let raw = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
    // Same rule `loqdave` uses: UTF-8 when it is, Latin-1 when it is not.
    let text = match String::from_utf8(raw.clone()) {
        Ok(s) => s,
        Err(_) => raw.iter().map(|b| *b as char).collect(),
    };
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

/// `0007-3f2a9c1b` — ordered for reading, hashed for stability.
fn case_name(i: usize, text: &str) -> String {
    format!("{:04}-{:08x}", i, fnv1a(text.as_bytes()))
}

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in bytes {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn short(s: &str) -> String {
    let one: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if one.chars().count() <= 52 {
        return one;
    }
    let cut: String = one.chars().take(49).collect();
    format!("{cut}...")
}
