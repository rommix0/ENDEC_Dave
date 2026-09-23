//! Loading voice banks by the name `Cat` uses for them.
//!
//! `stage-cat.txt` names a bank per unit:
//!
//! ```text
//! PRELOADED = EnglishUs/Dave/Dave
//! PRELOADED = EnglishUs/Dave/DaveAU
//! ```
//!
//! **This is not constant across an utterance.** In the corpus the `SEQUENS`
//! units come from `DaveAU` (216 references) while everything else comes from
//! `Dave` (3,516), so a renderer that loads one bank reads the `SEQUENS` units
//! out of the wrong file entirely.
//!
//! The name maps to a file by appending the rate tag: `EnglishUs/Dave/DaveAU`
//! becomes `EnglishUs/Dave/DaveAU-19200.16000.loqmsx.bin`. The tag is not
//! assumed — [`resolve`] globs `<stem>-*.loqmsx.bin`, which also keeps
//! `Dave` from matching `DaveAU` because the hyphen is part of the pattern.
//!
//! # Each bank needs its own decoder
//!
//! The LSP and innovation codebooks are per voice, and the decoder carries
//! excitation history, `mem_sp`, `interp_qlpc` and the QMF memories across
//! frames. So a bank cannot share a [`BankReader`] with another bank, and
//! switching banks mid-utterance must not reset the one being left — the
//! engine preloads both and keeps them open. [`Banks`] holds one reader per
//! bank for exactly that reason.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use loqng_codec::decoder::Voice;
use loqng_codec::reader::{BankReader, CodedFrames, FlatFrames};
use loqng_data::bank::{BankHeader, BankParams, BankTables, TableSpan};

/// One bank's bytes and everything derived from its preamble.
pub struct LoadedBank {
    pub name: String,
    pub raw: Vec<u8>,
    pub header: BankHeader,
    pub tables: BankTables,
    pub params: BankParams,
    idx: Vec<u8>,
    nb: [Vec<i8>; 3],
    sb: [Vec<i8>; 2],
    nb_innov: Vec<i8>,
    sb_innov: Vec<i8>,
}

fn signed(b: &[u8]) -> Vec<i8> {
    b.iter().map(|&v| v as i8).collect()
}

fn span(raw: &[u8], s: &TableSpan, what: &str) -> Result<Vec<u8>, String> {
    s.slice(raw)
        .map(|v| v.to_vec())
        .ok_or_else(|| format!("{what}: span is outside the preamble"))
}

/// Find the `.bin` for a `PRELOADED` name under the voice tree.
pub fn resolve(data_dir: &Path, preloaded: &str) -> Result<PathBuf, String> {
    let rel = preloaded.replace('/', std::path::MAIN_SEPARATOR_STR);
    let full = data_dir.join(&rel);
    let dir = full
        .parent()
        .ok_or_else(|| format!("{preloaded}: no parent directory"))?;
    let stem = full
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| format!("{preloaded}: no file name"))?;

    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut hits: Vec<PathBuf> = Vec::new();
    for e in entries.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with(&format!("{stem}-")) && n.ends_with(".loqmsx.bin") {
            hits.push(e.path());
        }
    }
    hits.sort();
    hits.into_iter()
        .next()
        .ok_or_else(|| format!("no <{stem}-*.loqmsx.bin> in {}", dir.display()))
}

impl LoadedBank {
    pub fn open(path: &Path) -> Result<LoadedBank, String> {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "bank".to_string());
        let raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let header = BankHeader::parse(&name, &raw).map_err(|e| e.to_string())?;
        let tables = BankTables::parse(&name, &raw, &header).map_err(|e| e.to_string())?;
        let params = BankParams::derive(&name, &raw, &tables).map_err(|e| e.to_string())?;

        let idx = span(&raw, &tables.idx, "context array")?;
        let nb = [
            signed(&span(&raw, &tables.nb_cb1, "nb lsp 1")?),
            signed(&span(&raw, &tables.nb_cb2, "nb lsp 2")?),
            signed(&span(&raw, &tables.nb_cb3, "nb lsp 3")?),
        ];
        let sb = [
            signed(&span(&raw, &tables.sb_cb1, "sb lsp 1")?),
            signed(&span(&raw, &tables.sb_cb2, "sb lsp 2")?),
        ];
        let nb_innov = signed(&span(&raw, &tables.nb_innov, "nb innovation")?);
        let sb_innov = signed(&span(&raw, &tables.sb_innov, "sb innovation")?);

        Ok(LoadedBank {
            name,
            raw,
            header,
            tables,
            params,
            idx,
            nb,
            sb,
            nb_innov,
            sb_innov,
        })
    }

    pub fn voice(&self) -> Voice<'_> {
        Voice {
            ctx: &self.idx,
            nb_lsp: [&self.nb[0], &self.nb[1], &self.nb[2]],
            nb_innov: &self.nb_innov,
            sb_lsp: [&self.sb[0], &self.sb[1]],
            sb_innov: &self.sb_innov,
            nb_innov_log2: (14 - self.params.nb.innov_shift) as u8,
            sb_innov_log2: (14 - self.params.sb.innov_shift) as u8,
        }
    }

    pub fn frames(&self) -> FlatFrames<'_> {
        let at = self.header.data_offset as usize;
        let coded = self.raw.get(at..).unwrap_or(&[]);
        FlatFrames::new(coded, self.header.frame_bytes() as usize)
    }

    pub fn rate(&self) -> u32 {
        self.header.output_rate
    }
}

/// Every bank an utterance refers to, each with its own decoder.
pub struct Banks {
    pub loaded: Vec<LoadedBank>,
    by_name: BTreeMap<String, usize>,
}

impl Banks {
    /// Load every distinct `PRELOADED` name, in first-seen order.
    pub fn open_all(data_dir: &Path, names: &[String]) -> Result<Banks, String> {
        let mut loaded = Vec::new();
        let mut by_name = BTreeMap::new();
        for n in names {
            if by_name.contains_key(n) {
                continue;
            }
            let path = resolve(data_dir, n)?;
            by_name.insert(n.clone(), loaded.len());
            loaded.push(LoadedBank::open(&path)?);
        }
        Ok(Banks { loaded, by_name })
    }

    pub fn index_of(&self, preloaded: &str) -> Option<usize> {
        self.by_name.get(preloaded).copied()
    }

    pub fn get(&self, i: usize) -> Option<&LoadedBank> {
        self.loaded.get(i)
    }

    /// A reader per bank, in the same order as [`Banks::loaded`].
    ///
    /// Separate readers because each bank has its own codebooks *and* its own
    /// running filter state; sharing one would corrupt both.
    pub fn readers(&self) -> Vec<BankReader<'_>> {
        self.loaded
            .iter()
            .map(|b| BankReader::new(b.voice(), b.tables.xor_key))
            .collect()
    }

    pub fn summary(&self) -> String {
        self.loaded
            .iter()
            .map(|b| format!("{} ({} frames)", b.name, b.frames().frames()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}
