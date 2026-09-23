//! `xtask join` — recover the concatenation smoothing at a unit boundary.
//!
//! `xtask concat` renders a captured utterance by hard-cutting bank segments
//! together and gets byte-identical audio **except** for runs of up to 64
//! samples centred on unit boundaries — thirty-two before the join and
//! thirty-two after. So the engine blends across the join. This recovers how,
//! by solving for the weights.
//!
//! **It answered the question and also misled.** Run against the first
//! capture it only ever shows *discontinuous* joins, because those are the
//! ones with a visible difference, and that made "smoothing happens where the
//! bank position jumps" look obvious. It is wrong: the smoothing is selected
//! by `ALGO == CONCATENAZIONE`, and contiguous joins get it too — they just
//! look almost unchanged, because both legs then read the same samples and the
//! crossfade is a near-identity. The filter on `contiguous` below is retained
//! deliberately: a contiguous join has `L == R`, so `(W-L)/(R-L)` divides by
//! zero and recovers nothing. Keep the filter, distrust the generalisation.
//!
//! # The experiment
//!
//! For one boundary, four 64-sample windows centred on it:
//!
//! ```text
//! L   the LEFT unit's bank samples, continued 32 past its end
//! R   the RIGHT unit's bank samples, started 32 before its start
//! M   what `concat` produces: L's first half hard-cut to R's second half
//! W   what the engine produced (audio.raw)
//! ```
//!
//! If the blend is a weighted crossfade then `W[i] = (1-w[i])*L[i] +
//! w[i]*R[i]`, and `w` is recoverable per sample wherever `L[i] != R[i]`:
//!
//! ```text
//! w[i] = (W[i] - L[i]) / (R[i] - L[i])
//! ```
//!
//! A clean ramp from 0 to 1 means a linear crossfade; a curve means something
//! else; garbage means it is not a crossfade at all and the smoothing is
//! doing something else entirely (overlap-add at a pitch mark, say).
//!
//! Printing `w` is the point. **Do not fit a model to it here** — read the
//! numbers first.

use std::fs;
use std::path::PathBuf;

use loqng_codec::reader::{BankReader, FlatFrames, Unit};
use loqng_data::bank::{BankHeader, BankParams, BankTables, TableSpan};

use crate::capture::corpus_dirs;
use crate::concat::parse_cat;
use crate::{flag, Paths};

const DEFAULT_BANK: &str = "EnglishUs/Dave/Dave-19200.16000.loqmsx.bin";
/// Half-width of the window the differences occupy.
const HALF: usize = 32;

fn signed(b: &[u8]) -> Vec<i8> {
    b.iter().map(|&v| v as i8).collect()
}

fn span(raw: &[u8], s: &TableSpan, what: &str) -> Result<Vec<u8>, String> {
    s.slice(raw)
        .map(|v| v.to_vec())
        .ok_or_else(|| format!("{what}: span is outside the preamble"))
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let bank_path = flag(args, "--bank")
        .map(PathBuf::from)
        .unwrap_or_else(|| paths.data_dir.join(DEFAULT_BANK));
    let want_joins: usize = flag(args, "--joins")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let name = bank_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bank".to_string());
    let raw = fs::read(&bank_path).map_err(|e| format!("{}: {e}", bank_path.display()))?;
    let header = BankHeader::parse(&name, &raw).map_err(|e| e.to_string())?;
    let tables = BankTables::parse(&name, &raw, &header).map_err(|e| e.to_string())?;
    let params = BankParams::derive(&name, &raw, &tables).map_err(|e| e.to_string())?;

    let idx = span(&raw, &tables.idx, "context array")?;
    let nb1 = signed(&span(&raw, &tables.nb_cb1, "nb lsp 1")?);
    let nb2 = signed(&span(&raw, &tables.nb_cb2, "nb lsp 2")?);
    let nb3 = signed(&span(&raw, &tables.nb_cb3, "nb lsp 3")?);
    let nbi = signed(&span(&raw, &tables.nb_innov, "nb innovation")?);
    let sb1 = signed(&span(&raw, &tables.sb_cb1, "sb lsp 1")?);
    let sb2 = signed(&span(&raw, &tables.sb_cb2, "sb lsp 2")?);
    let sbi = signed(&span(&raw, &tables.sb_innov, "sb innovation")?);
    let mk_voice = || loqng_codec::decoder::Voice {
        ctx: &idx,
        nb_lsp: [&nb1, &nb2, &nb3],
        nb_innov: &nbi,
        sb_lsp: [&sb1, &sb2],
        sb_innov: &sbi,
        nb_innov_log2: (14 - params.nb.innov_shift) as u8,
        sb_innov_log2: (14 - params.sb.innov_shift) as u8,
    };

    let frame_bytes = header.frame_bytes() as usize;
    let coded = raw
        .get(header.data_offset as usize..)
        .ok_or_else(|| "the bank has no payload".to_string())?;
    let src = FlatFrames::new(coded, frame_bytes);
    let rate = header.output_rate;

    let dirs = corpus_dirs(&paths.corpus)?;
    let d = dirs.first().ok_or_else(|| "no captures".to_string())?;
    let cat = fs::read_to_string(d.join("stage-cat.txt")).map_err(|e| e.to_string())?;
    let recs = parse_cat(&cat);
    let want_bytes = fs::read(d.join("audio.raw")).map_err(|e| e.to_string())?;
    let want: Vec<i16> = want_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let units: Vec<Unit> = recs
        .iter()
        .filter_map(|r| r.confini)
        .map(|(a, b)| Unit::from_confini(a, b, rate))
        .collect();
    let algos: Vec<&str> = recs
        .iter()
        .filter(|r| r.confini.is_some())
        .map(|r| r.algo.as_str())
        .collect();

    println!(
        "{}: {} units",
        d.file_name().unwrap().to_string_lossy(),
        units.len()
    );
    println!("window is {} samples either side of the join", HALF);
    println!();

    // Walk the boundaries, keeping the running output offset.
    let mut out_at = 0usize;
    let mut shown = 0usize;
    for i in 0..units.len().saturating_sub(1) {
        let left = units[i];
        let right = units[i + 1];
        out_at += left.count;
        let contiguous = left.start + left.count as u64 == right.start;
        if contiguous {
            continue;
        }
        if out_at < HALF || out_at + HALF > want.len() {
            continue;
        }
        if shown >= want_joins {
            break;
        }
        shown += 1;

        // L: the left unit continued past its end.
        let mut reader = BankReader::new(mk_voice(), tables.xor_key);
        let mut l: Vec<i16> = Vec::new();
        reader
            .read(
                &src,
                left.start + left.count as u64 - HALF as u64,
                2 * HALF,
                &mut l,
            )
            .map_err(|e| format!("{e:?}"))?;

        // R: the right unit started early.
        let mut reader = BankReader::new(mk_voice(), tables.xor_key);
        let mut r: Vec<i16> = Vec::new();
        reader
            .read(&src, right.start - HALF as u64, 2 * HALF, &mut r)
            .map_err(|e| format!("{e:?}"))?;

        let w = &want[out_at - HALF..out_at + HALF];

        println!(
            "join {i}->{}  ({} -> {})  output sample {out_at}",
            i + 1,
            algos.get(i).unwrap_or(&"?"),
            algos.get(i + 1).unwrap_or(&"?")
        );
        println!("   k     L      R      W    (W-L)/(R-L)");
        for k in 0..2 * HALF {
            let lv = l[k] as f64;
            let rv = r[k] as f64;
            let wv = w[k] as f64;
            let frac = if (rv - lv).abs() > 8.0 {
                format!("{:8.4}", (wv - lv) / (rv - lv))
            } else {
                "     --".to_string()
            };
            let cut = if k == HALF { " <- join" } else { "" };
            println!(
                "  {:3} {:6} {:6} {:6}  {}{}",
                k as i32 - HALF as i32,
                l[k],
                r[k],
                w[k],
                frac,
                cut
            );
        }
        println!();
    }

    if shown == 0 {
        return Err("no discontinuous joins found in the first capture".to_string());
    }
    Ok(())
}
