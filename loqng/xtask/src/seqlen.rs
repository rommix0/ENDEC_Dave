//! `xtask seqlen` — measure how long a `SEQUENS` unit's output really is.
//!
//! The gating unknown for `SEQUENS`. `CONFINI` gives the source segment and
//! `DURATA` the requested duration, and **neither predicts the output length**:
//! across the corpus `CONFINI` is exact on some captures and 1419 samples too
//! long on others, while `DURATA * rate / 1000` is close but off by tens of
//! samples. A pitch-synchronous synthesiser lays down whole periods, so the
//! length should be quantised to them — but guessing the quantisation from two
//! wrong models is how a plausible-and-wrong answer gets written down.
//!
//! So measure it instead of modelling it.
//!
//! # How
//!
//! Every non-`SEQUENS` unit renders byte-exactly, which makes them landmarks.
//! Walk the unit list keeping two cursors — `mine` into the local render and
//! `theirs` into the captured audio. They advance together across modelled
//! units. At a run of `SEQUENS` units the local length is a guess, so instead
//! of trusting it, take a distinctive slice from the **middle** of the next
//! modelled unit (the middle, because the first and last 32 samples may carry
//! a crossfade) and search the captured audio for it. Where it lands gives the
//! true end of the `SEQUENS` run, and subtraction gives its length.
//!
//! That yields ground truth per run with no reverse engineering at all, and it
//! is what any length model has to reproduce.
//!
//! # Reading the output
//!
//! `confini` and `durata` are the two candidate lengths; `actual` is measured.
//! A run of several `SEQUENS` units only yields a total, so single-unit runs
//! are the useful rows — they are marked `[1]`.

use std::fs;

use loqng_codec::reader::{FlatFrames, Unit};
use loqng_codec::window::SIG_COSENO;

use crate::capture::corpus_dirs;
use crate::concat::{parse_cat, render_multi, CatRecord, Placed};
use crate::voicebank::Banks;
use crate::{flag, Paths};

/// Samples taken from the middle of a landmark unit when searching.
const PROBE: usize = 48;

/// Find `needle` in `hay[from..]`, returning the absolute index.
fn find(hay: &[i16], needle: &[i16], from: usize, window: usize) -> Option<usize> {
    if needle.is_empty() || from >= hay.len() {
        return None;
    }
    let end = (from + window + needle.len()).min(hay.len());
    let region = &hay[from..end];
    region
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| from + p)
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let limit: usize = flag(args, "--limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let dirs = corpus_dirs(&paths.corpus)?;
    let mut rows = 0usize;
    let mut confini_exact = 0usize;
    let mut durata_exact = 0usize;
    let mut singles = 0usize;

    println!(
        "{:<16} {:>3} {:>5} {:>8} {:>8} {:>8}  {:>7} {:>7}  {}",
        "capture", "run", "units", "confini", "durata", "actual", "d(conf)", "d(dur)", "F0 / phone"
    );

    for d in dirs.iter().take(limit) {
        let cat_path = d.join("stage-cat.txt");
        let wav_path = d.join("audio.raw");
        if !cat_path.is_file() || !wav_path.is_file() {
            continue;
        }
        let cat = fs::read_to_string(&cat_path).map_err(|e| e.to_string())?;
        let recs = parse_cat(&cat);
        if recs.is_empty() {
            continue;
        }
        let names: Vec<String> = recs.iter().map(|r| r.preloaded.clone()).collect();
        let banks = Banks::open_all(&paths.data_dir, &names)?;
        let rate = banks.get(0).map(|b| b.rate()).unwrap_or(16000);

        let kept: Vec<&CatRecord> = recs
            .iter()
            .filter(|r| {
                r.confini
                    .map(|(a, b)| Unit::from_confini(a, b, rate).count > 0)
                    .unwrap_or(false)
            })
            .collect();
        let mut placed: Vec<Placed> = Vec::with_capacity(kept.len());
        for r in kept.iter() {
            let (a, b) = r.confini.unwrap();
            placed.push(Placed {
                unit: Unit::from_confini(a, b, rate),
                bank: banks.index_of(&r.preloaded).unwrap_or(0),
                join: r.algo == "CONCATENAZIONE",
            });
        }

        let srcs: Vec<FlatFrames> = banks.loaded.iter().map(|b| b.frames()).collect();
        let mut readers = banks.readers();
        let mine = render_multi(&mut readers, &srcs, &placed, &SIG_COSENO)?;

        let want_bytes = fs::read(&wav_path).map_err(|e| e.to_string())?;
        let want: Vec<i16> = want_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Offsets of each unit in the local render.
        let mut my_off = Vec::with_capacity(placed.len() + 1);
        let mut acc = 0usize;
        for p in placed.iter() {
            my_off.push(acc);
            acc += p.unit.count;
        }
        my_off.push(acc);

        let label = d
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut theirs = 0usize;
        let mut i = 0usize;
        let mut run_no = 0usize;

        while i < placed.len() {
            if kept[i].algo != "SEQUENS" {
                theirs += placed[i].unit.count;
                i += 1;
                continue;
            }

            // A maximal run of SEQUENS units.
            let start = i;
            let mut confini_total = 0usize;
            let mut durata_ms = 0.0f64;
            while i < placed.len() && kept[i].algo == "SEQUENS" {
                confini_total += placed[i].unit.count;
                durata_ms += kept[i].durata;
                i += 1;
            }
            let nunits = i - start;
            let durata_total = (durata_ms * rate as f64 / 1000.0).round() as usize;

            // Locate the end of the run in the captured audio.
            let actual = if i >= placed.len() {
                // The run ends the utterance.
                Some(want.len().saturating_sub(theirs))
            } else {
                let lm = my_off[i];
                let lmlen = placed[i].unit.count;
                if lmlen < PROBE + 64 || lm + lmlen > mine.len() {
                    None
                } else {
                    let mid = lm + lmlen / 2 - PROBE / 2;
                    let needle = &mine[mid..mid + PROBE];
                    let offset_into_unit = mid - lm;
                    find(&want, needle, theirs, confini_total + durata_total + 4096)
                        .map(|p| p - offset_into_unit - theirs)
                }
            };

            run_no += 1;
            rows += 1;
            if nunits == 1 {
                singles += 1;
            }
            match actual {
                Some(a) => {
                    if a == confini_total {
                        confini_exact += 1;
                    }
                    if a == durata_total {
                        durata_exact += 1;
                    }
                    println!(
                        "{label:<16} {run_no:>3} {:>5} {confini_total:>8} {durata_total:>8} \
                         {a:>8}  {:>+7} {:>+7}  {} {}{}",
                        nunits,
                        a as i64 - confini_total as i64,
                        a as i64 - durata_total as i64,
                        kept[start].f0.0,
                        kept[start].fonema,
                        if nunits == 1 { "  [1]" } else { "" }
                    );
                    theirs += a;
                }
                None => {
                    println!(
                        "{label:<16} {run_no:>3} {:>5} {confini_total:>8} {durata_total:>8} \
                         {:>8}  {:>7} {:>7}  {} {}",
                        nunits, "?", "-", "-", kept[start].f0.0, kept[start].fonema
                    );
                    theirs += confini_total;
                }
            }
        }
    }

    println!();
    println!("{rows} SEQUENS runs measured ({singles} of them a single unit)");
    println!("  CONFINI predicts the length on {confini_exact}/{rows}");
    println!("  DURATA  predicts the length on {durata_exact}/{rows}");
    Ok(())
}
