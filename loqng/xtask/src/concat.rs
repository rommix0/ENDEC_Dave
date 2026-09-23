//! `xtask concat` — render a captured utterance from its Cat unit list.
//!
//! This closes the back half of the pipeline. The corpus holds, per sentence,
//! the engine's own `Cat` stage dump and the audio it produced. `Cat` names
//! the units it selected as `CONFINI` boundaries — **times in seconds inside
//! the voice bank** — so given a bit-exact bank decoder the audio is fully
//! determined:
//!
//! ```text
//! stage-cat.txt  ->  units  ->  BankReader  ->  concatenate  ->  audio.raw
//! ```
//!
//! # Status: byte-exact except `SEQUENS`
//!
//! All 37 captures reproduce **byte-identically up to their first `SEQUENS`
//! unit** — 1,326,513 samples, about 83 seconds. That proves the codec in its
//! real usage rather than only on frame 0, the `CONFINI` interpretation, the
//! seek/preroll rule in `driver.rs`, the output delay, and the concatenation
//! smoothing.
//!
//! # The three algorithms, as established
//!
//! ```text
//! COPIA           plain copy of a bank segment, nothing applied
//! CONCATENAZIONE  a copy PLUS a 64-sample overlap-add with the next unit
//! SEQUENS         prosody-modified; NOT yet implemented
//! ```
//!
//! **`ALGO` is what selects the smoothing, not whether the segments happen to
//! be adjacent in the bank.** Keying off contiguity instead looks right — a
//! contiguous join needs no smoothing, surely — and it is wrong: contiguous
//! `CONCATENAZIONE` joins are smoothed too. Because both legs then read the
//! *same* samples, the crossfade is a near-identity, differing only where the
//! two window weights fail to sum to 32767. See
//! [`loqng_codec::reader::render_joined`].
//!
//! **Zero-length units must be dropped first.** 62 of the corpus's 3,732
//! units have `CONFINI` endpoints that round to the same sample — exactly the
//! 62 carrying `NMARKERS = 0`. Left in place they break the join chain: each
//! neighbour pairs with the empty unit instead of with the other, so the fade
//! never happens. This one fix took the corpus from 26/37 to 37/37.
//!
//! **Banks are per unit.** `PRELOADED` changes within an utterance, and each
//! bank gets its own decoder — see [`crate::voicebank`].
//!
//! # What `SEQUENS` still needs
//!
//! `DURATA` and `F0` are non-trivial **only** on `SEQUENS` units — 204 of
//! 3,732, and every one carries a duration and an F0 pair. So `SEQUENS` means
//! "generated with prosody applied", and needs the pitch-synchronous
//! machinery `notes/stage-sig.md` §5.6 describes.
//!
//! Because a stretched `SEQUENS` unit changes the output length, everything
//! after the first one shifts and compares as different — which is why the
//! honest metric here is the matching prefix rather than a whole-utterance
//! diff.
//!
//! # What the interpretation rests on
//!
//! Before any code was written, `tools/catunits.py` checked the hypothesis on
//! three captures: summing `round((t1 - t0) * 16000)` over every unit
//! reproduced the captured sample count **exactly** — 27017, 43585 and 38638,
//! ratio 1.0000.

use std::fs;
use std::path::{Path, PathBuf};

use loqng_codec::reader::{crossfade, BankReader, FlatFrames, Unit, JOIN_HALF};
use loqng_codec::window::SIG_COSENO;

use crate::capture::corpus_dirs;
use crate::voicebank::Banks;
use crate::{flag, Paths};

/// One record of a `stage-cat.txt` dump, reduced to what matters here.
#[derive(Debug, Clone)]
pub struct CatRecord {
    pub algo: String,
    pub preloaded: String,
    pub fonema: String,
    pub confini: Option<(f64, f64)>,
    pub durata: f64,
    pub f0: (f64, f64),
    pub guadagno: (f64, f64),
}

/// Parse the `BEGIN`/`END` records of a Cat dump.
pub fn parse_cat(text: &str) -> Vec<CatRecord> {
    let mut out = Vec::new();
    let mut cur: Option<CatRecord> = None;
    for line in text.lines() {
        let line = line.trim();
        if line == "BEGIN" {
            cur = Some(CatRecord {
                algo: String::new(),
                preloaded: String::new(),
                fonema: String::new(),
                confini: None,
                durata: 0.0,
                f0: (0.0, 0.0),
                guadagno: (1.0, 1.0),
            });
            continue;
        }
        if line == "END" {
            if let Some(r) = cur.take() {
                out.push(r);
            }
            continue;
        }
        let Some(r) = cur.as_mut() else { continue };
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim();
        let pair = || -> (f64, f64) {
            let mut it = v.split_whitespace();
            let a = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let b = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            (a, b)
        };
        match k {
            "ALGO" => r.algo = v.to_string(),
            "PRELOADED" => r.preloaded = v.to_string(),
            "FONEMA" => r.fonema = v.to_string(),
            "CONFINI" => {
                let (a, b) = pair();
                r.confini = Some((a, b));
            }
            "DURATA" => r.durata = v.parse().unwrap_or(0.0),
            "F0" => r.f0 = pair(),
            "GUADAGNO" => r.guadagno = pair(),
            _ => {}
        }
    }
    out
}

/// A unit resolved against its bank.
#[derive(Debug, Clone, Copy)]
pub struct Placed {
    pub unit: Unit,
    pub bank: usize,
    pub join: bool,
}

/// Render a unit list whose units may come from different banks.
///
/// Each bank has its own reader, so switching banks mid-utterance does not
/// disturb the one being left — the engine preloads both and keeps them open.
/// A join between units from *different* banks crossfades across the two
/// voices, which is what the boundary arithmetic implies and what the corpus
/// appears to show.
pub fn render_multi(
    readers: &mut [BankReader],
    srcs: &[FlatFrames],
    placed: &[Placed],
    win: &[i16],
) -> Result<Vec<i16>, String> {
    let half = JOIN_HALF;
    let n = 2 * half;
    let total: usize = placed.iter().map(|p| p.unit.count).sum();
    let mut out: Vec<i16> = Vec::with_capacity(total);
    let mut pending: Option<Vec<i16>> = None;

    for (i, p) in placed.iter().enumerate() {
        let at = out.len();
        let (r, s) = (
            readers
                .get_mut(p.bank)
                .ok_or_else(|| format!("unit {i}: no reader for bank {}", p.bank))?,
            srcs.get(p.bank)
                .ok_or_else(|| format!("unit {i}: no frames for bank {}", p.bank))?,
        );
        r.read(s, p.unit.start, p.unit.count, &mut out)
            .map_err(|e| format!("unit {i}: {e:?}"))?;

        if let Some(q) = pending.take() {
            for (k, v) in q.iter().enumerate() {
                if let Some(slot) = out.get_mut(at + k) {
                    *slot = *v;
                }
            }
        }

        let Some(next) = placed.get(i + 1) else {
            continue;
        };
        if !p.join || p.unit.count < half || next.unit.count < half || next.unit.start < half as u64
        {
            continue;
        }

        // Left leg: this unit's last `half` samples plus `half` past its end.
        let mut left: Vec<i16> = out[out.len() - half..].to_vec();
        {
            let r = readers.get_mut(p.bank).unwrap();
            let s = &srcs[p.bank];
            r.read(s, p.unit.start + p.unit.count as u64, half, &mut left)
                .map_err(|e| format!("unit {i} tail: {e:?}"))?;
        }

        // Right leg: `half` before the next unit's start, plus its first `half`.
        let mut right: Vec<i16> = Vec::with_capacity(n);
        {
            let r = readers.get_mut(next.bank).unwrap();
            let s = &srcs[next.bank];
            r.read(s, next.unit.start - half as u64, n, &mut right)
                .map_err(|e| format!("unit {i} head: {e:?}"))?;
        }

        let blend = crossfade(win, &left, &right, n);
        let tail = out.len() - half;
        out[tail..].copy_from_slice(&blend[..half]);
        pending = Some(blend[half..].to_vec());
    }

    Ok(out)
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let limit: usize = flag(args, "--limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let write_wav = flag(args, "--out").map(PathBuf::from);
    let only = flag(args, "--only");
    let verbose = args.iter().any(|a| a == "--verbose");

    let dirs = corpus_dirs(&paths.corpus)?;
    if dirs.is_empty() {
        return Err(format!(
            "no captures in {} — run `xtask capture` first",
            paths.corpus.display()
        ));
    }

    let mut exact = 0usize;
    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut total_prefix = 0usize;
    let mut total_upto = 0usize;
    let mut prefix_to_sequens = 0usize;
    let mut first_bad: Option<String> = None;
    let mut banks_seen = String::new();

    for d in dirs.iter().take(limit) {
        if let Some(pat) = only.as_ref() {
            let nm = d
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if !nm.contains(pat.as_str()) {
                continue;
            }
        }
        let cat_path = d.join("stage-cat.txt");
        let wav_path = d.join("audio.raw");
        if !cat_path.is_file() || !wav_path.is_file() {
            skipped += 1;
            continue;
        }
        let cat =
            fs::read_to_string(&cat_path).map_err(|e| format!("{}: {e}", cat_path.display()))?;
        let recs = parse_cat(&cat);
        if recs.is_empty() {
            skipped += 1;
            continue;
        }

        // Load every bank this utterance names, each with its own decoder.
        let names: Vec<String> = recs.iter().map(|r| r.preloaded.clone()).collect();
        let banks = Banks::open_all(&paths.data_dir, &names)?;
        if banks_seen.is_empty() {
            banks_seen = banks.summary();
        }
        let rate = banks.get(0).map(|b| b.rate()).unwrap_or(16000);

        // Drop zero-length units before anything else touches the chain.
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
                bank: banks
                    .index_of(&r.preloaded)
                    .ok_or_else(|| format!("no bank loaded for {}", r.preloaded))?,
                join: r.algo == "CONCATENAZIONE",
            });
        }

        let srcs: Vec<FlatFrames> = banks.loaded.iter().map(|b| b.frames()).collect();
        let mut readers = banks.readers();
        let mine = render_multi(&mut readers, &srcs, &placed, &SIG_COSENO)?;

        let want_bytes = fs::read(&wav_path).map_err(|e| format!("{}: {e}", wav_path.display()))?;
        let want: Vec<i16> = want_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        checked += 1;
        let label = d
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        // Output offset of the first unit this renderer does not model.
        let mut first_unmodelled = None;
        let mut acc = 0usize;
        for (ui, p) in placed.iter().enumerate() {
            if kept[ui].algo == "SEQUENS" && first_unmodelled.is_none() {
                first_unmodelled = Some(acc);
            }
            acc += p.unit.count;
        }
        let prefix = mine
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(mine.len().min(want.len()));
        total_prefix += prefix;
        total_upto += first_unmodelled.unwrap_or(mine.len());
        if Some(prefix) == first_unmodelled || prefix >= first_unmodelled.unwrap_or(usize::MAX) {
            prefix_to_sequens += 1;
        }

        let n = mine.len().min(want.len());
        let ndiff = mine.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        if mine == want {
            exact += 1;
            println!("  {label}  {:6} samples  BIT-EXACT", mine.len());
        } else {
            let seq = first_unmodelled.unwrap_or(usize::MAX);
            let verdict = if prefix >= seq {
                "clean to SEQUENS"
            } else {
                "DIFFERS EARLY  "
            };
            println!(
                "  {label}  {:6} samples  {verdict}  {ndiff:6} differ, first at {prefix:6}, \
                 SEQUENS at {:6}, lengths {} vs {}",
                mine.len(),
                if seq == usize::MAX { 0 } else { seq },
                mine.len(),
                want.len()
            );

            if verbose && first_bad.is_none() {
                let mut msg = String::new();
                let mut at = 0usize;
                for (ui, p) in placed.iter().enumerate() {
                    let end = (at + p.unit.count).min(n);
                    if end > at {
                        let bad = (at..end).filter(|&k| mine[k] != want[k]).count();
                        if bad > 0 {
                            let maxd = (at..end)
                                .map(|k| (mine[k] as i32 - want[k] as i32).abs())
                                .max()
                                .unwrap_or(0);
                            msg.push_str(&format!(
                                "    unit {ui:3} {:<14} {:<8} out[{at:6}..{end:6}) \
                                 {bad:5}/{:<5} differ, max |d| {maxd}\n",
                                kept[ui].algo,
                                banks.get(p.bank).map(|b| b.name.as_str()).unwrap_or("?"),
                                p.unit.count
                            ));
                        }
                    }
                    at = end;
                }
                first_bad = Some(msg);
            }
        }

        if let Some(p) = write_wav.as_ref() {
            write_wav_file(p, rate, &mine)?;
            println!("    wrote {}", p.display());
        }
    }

    println!();
    if !banks_seen.is_empty() {
        println!("banks: {banks_seen}");
    }
    println!("{exact}/{checked} utterances byte-identical end to end ({skipped} skipped)");
    println!("{prefix_to_sequens}/{checked} match exactly up to their first SEQUENS unit");
    println!(
        "{total_prefix} of {total_upto} samples before the first SEQUENS are exact ({:.2}%)",
        100.0 * total_prefix as f64 / total_upto.max(1) as f64
    );
    if let Some(b) = first_bad {
        println!();
        print!("{b}");
    }
    if exact < checked {
        return Err("not byte-identical".to_string());
    }
    Ok(())
}

fn write_wav_file(path: &Path, rate: u32, pcm: &[i16]) -> Result<(), String> {
    let data_len = (pcm.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + pcm.len() * 2);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        v.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(path, &v).map_err(|e| format!("{}: {e}", path.display()))
}
