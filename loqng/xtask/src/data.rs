//! `xtask data` — read every data file in the voice tree and report.
//!
//! This is the regression test for `loqng-data` against real vendor files
//! rather than the small literals in its unit tests. A format that is only
//! partly understood reports what it managed, not a pass.

use std::path::Path;

use loqng_data::{bank::BankHeader, descriptor::Descriptor, lexicon::Lexicon, sde::Sde, text};

use crate::Paths;

pub fn run(p: &Paths) -> Result<(), String> {
    println!("voice tree: {}", p.data_dir.display());
    println!();

    let mut files = Vec::new();
    walk(&p.data_dir, &mut files)?;
    files.sort();

    let mut ok = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for f in &files {
        let rel = f
            .strip_prefix(&p.data_dir)
            .unwrap_or(f)
            .to_string_lossy()
            .replace('\\', "/");
        let ext = f
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let summary = match ext.as_str() {
            "vde" | "lde" | "session" => read_descriptor(f),
            "lex" => read_lexicon(f),
            "sde" => read_sde(f),
            "bin" => read_bank(f),
            "phd" => read_phd(f),
            "atm" => read_atm(f),
            "__never" => {
                skipped += 1;
                println!("  {rel:<48} -- not implemented (Phase 1)");
                continue;
            }
            _ => continue,
        };

        match summary {
            Ok(s) => {
                ok += 1;
                println!("  {rel:<48} ok  {s}");
            }
            Err(e) => {
                failed += 1;
                println!("  {rel:<48} FAIL  {e}");
            }
        }
    }

    println!();
    println!("{ok} read, {skipped} not implemented, {failed} failed");
    if failed > 0 {
        return Err(format!("{failed} file(s) failed to parse"));
    }
    Ok(())
}

fn read_descriptor(f: &Path) -> Result<String, String> {
    let raw = std::fs::read(f).map_err(|e| e.to_string())?;
    let name = f.file_name().unwrap_or_default().to_string_lossy();
    let d = Descriptor::parse_bytes(&name, &raw).map_err(|e| e.to_string())?;
    let inc = d.includes();
    let extra = if inc.is_empty() {
        String::new()
    } else {
        format!(", includes {}", inc.join(" "))
    };
    Ok(format!("{} entries{extra}", d.entries.len()))
}

fn read_lexicon(f: &Path) -> Result<String, String> {
    let raw = std::fs::read(f).map_err(|e| e.to_string())?;
    let name = f.file_name().unwrap_or_default().to_string_lossy();
    let encoded = text::looks_encoded(&raw);
    let l = Lexicon::parse_bytes(&name, &raw).map_err(|e| e.to_string())?;
    let cs = l.entries.iter().filter(|e| e.case_sensitive).count();
    Ok(format!(
        "\"{}\", {} entries ({cs} case-sensitive){}",
        l.name,
        l.len(),
        if encoded { ", high-bit" } else { ", plain" }
    ))
}

fn read_sde(f: &Path) -> Result<String, String> {
    let raw = std::fs::read(f).map_err(|e| e.to_string())?;
    let name = f.file_name().unwrap_or_default().to_string_lossy();
    let s = Sde::parse_bytes(&name, &raw).map_err(|e| e.to_string())?;
    let (u, l, r) = s.tally();
    let agree = if s.counts_agree() {
        "counts agree"
    } else {
        "COUNTS DISAGREE"
    };

    // The decoded records must chain: a record without a `!` starts exactly
    // where the previous one ended.
    let units = s.units();
    let explicit = units.iter().filter(|x| x.explicit_start).count();
    let chain = if s.continuity_holds() {
        "chain ok"
    } else {
        "CHAIN BROKEN"
    };
    // Every recording's first unit anchors, so these must be equal.
    let anchors = if explicit == u {
        "anchors=utterances"
    } else {
        "ANCHOR MISMATCH"
    };

    Ok(format!(
        "v{}, {u} utterances, {l}+{r} halves, {agree}, {} units {chain}, {anchors}",
        s.version().unwrap_or(-1),
        units.len()
    ))
}

fn read_atm(f: &Path) -> Result<String, String> {
    let raw = std::fs::read(f).map_err(|e| e.to_string())?;
    let name = f.file_name().unwrap_or_default().to_string_lossy();
    let a = loqng_data::atm::Atm::parse_bytes(&name, &raw).map_err(|e| e.to_string())?;

    // Two checks the format makes possible: the header states the file's own
    // length, and the symbol section must end exactly where the transition
    // table starts.
    let consistent = if a.self_consistent(raw.len()) {
        "consistent"
    } else {
        "INCONSISTENT"
    };
    let alphabet: String = a.symbols.graphemes.iter().take(5).cloned().collect();
    Ok(format!(
        "{} graphemes ({alphabet}...), {} phonemes, class {}x{}, {} states / {} transitions, {consistent}",
        a.symbols.graphemes.len(),
        a.symbols.phonemes.len(),
        a.symbols.classes.len(),
        a.symbols.class_width,
        a.transitions.states.len(),
        a.transitions.records.len()
    ))
}

fn read_phd(f: &Path) -> Result<String, String> {
    let raw = std::fs::read(f).map_err(|e| e.to_string())?;
    let name = f.file_name().unwrap_or_default().to_string_lossy();
    let p = loqng_data::phd::Phd::parse_bytes(&name, &raw).map_err(|e| e.to_string())?;

    // Lookup is a binary search, so an unsorted file would break the engine's
    // own dictionary. Report it rather than assuming.
    let order = match p.first_unsorted() {
        None => "sorted".to_string(),
        Some((i, a, b)) => format!("UNSORTED at {i}: {a:?} > {b:?}"),
    };
    // Prove the search works on the real data, not just the unit fixtures.
    let probe = p
        .entries
        .get(p.len() / 2)
        .map(|e| {
            if p.lookup(&e.word) == Some(e.transcription.as_str()) {
                "search ok".to_string()
            } else {
                format!("SEARCH MISSED {:?}", e.word)
            }
        })
        .unwrap_or_else(|| "empty".to_string());

    Ok(format!(
        "{} entries, {}, {order}, {probe}",
        p.len(),
        if p.delta_encoded { "delta" } else { "plain" }
    ))
}

fn read_bank(f: &Path) -> Result<String, String> {
    let mut raw = vec![0u8; loqng_data::bank::HEADER_LEN];
    read_head(f, &mut raw)?;
    let name = f.file_name().unwrap_or_default().to_string_lossy();
    let h = BankHeader::parse(&name, &raw).map_err(|e| e.to_string())?;
    let len = std::fs::metadata(f).map_err(|e| e.to_string())?.len() as usize;

    // The tag is in the file name, so it is checkable rather than merely
    // plausible: Dave-19200.16000.loqmsx.bin must give 19200.16000.
    let tag = h.rate_tag();
    let named = name.contains(&tag);

    // The bitrate predicts a frame size; the file length measures one. They
    // have to agree, and that is what identifies word 0 as a bitrate.
    let sizes = match h.frame_bytes_measured(len) {
        Some(m) if m == h.frame_bytes() => format!("{m} B/frame confirmed"),
        Some(m) => format!("SIZE MISMATCH header={} measured={m}", h.frame_bytes()),
        None => "PAYLOAD NOT A WHOLE NUMBER OF FRAMES".to_string(),
    };

    // The preamble's seven codebooks must end exactly on the payload. That is
    // the whole layout checking itself: a misread count lands anywhere else.
    let mut pre = vec![0u8; h.data_offset as usize];
    read_head(f, &mut pre)?;
    let tables = match loqng_data::bank::BankTables::parse(&name, &pre, &h) {
        Ok(t) => {
            let p = match loqng_data::bank::BankParams::derive(&name, &pre, &t) {
                Ok(p) => format!(
                    "nb {}:{:?}+{}sh, sb {}:{:?}+{}sh",
                    p.nb.order,
                    p.nb.lsp_nbits,
                    p.nb.innov_shape_bits,
                    p.sb.order,
                    p.sb.lsp_nbits,
                    p.sb.innov_shape_bits
                ),
                Err(e) => format!("PARAMS {e}"),
            };
            format!("{} ctx, key {:#04x}, {p}", t.contexts, t.xor_key)
        }
        Err(e) => format!("TABLES {e}"),
    };

    let secs = h.duration_secs();
    Ok(format!(
        "{tag} {}, {} frames, {sizes}, {:.0}m{:02.0}s, {tables}",
        if named {
            "matches name"
        } else {
            "NAME MISMATCH"
        },
        h.frames,
        (secs / 60.0).floor(),
        secs % 60.0
    ))
}

/// Read only the head of a file; the banks are up to 21 MB.
fn read_head(f: &Path, buf: &mut [u8]) -> Result<(), String> {
    use std::io::Read;
    let mut fh = std::fs::File::open(f).map_err(|e| e.to_string())?;
    fh.read_exact(buf).map_err(|e| e.to_string())
}

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in entries {
        let e = e.map_err(|e| e.to_string())?;
        let p = e.path();
        if p.is_dir() {
            walk(&p, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}
