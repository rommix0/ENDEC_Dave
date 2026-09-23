//! `.atm` — the grapheme-to-phoneme automaton.
//!
//! 1.3 MB for `EnglishUs6.9.atm`. The plan is to port the *interpreter* and
//! keep the vendor data (`PLAN.md` §5, Phase 1), so what matters is the
//! structure, not a byte-for-byte spec of the transition table.
//!
//! Everything here is **big-endian**, like the `.bin` bank and unlike the rest
//! of the engine.
//!
//! Recovered from `ELQLoadAutomaEx` (Ghidra `0x91ca4`) and its three section
//! parsers, `FUN_00090f30`, `FUN_0009104c` and `FUN_00091250`.
//!
//! # Source prefix
//!
//! **The path carries a one-character prefix** selecting how the data is read;
//! the real filename is `path + 1`. `ELQLoadAutomaEx` `strchr`s `path[0]`
//! through a series of tables to pick:
//!
//! | mode | source |
//! |---|---|
//! | 0 | stdio, read on demand |
//! | 1 | memory-mapped |
//! | 2 | inside a `.bin` container |
//! | 3 | slurped whole into malloc |
//! | `0xffff` | unrecognised prefix — hard error |
//!
//! # Layout
//!
//! ```text
//! 0       Header        12 bytes, holds the other section offsets
//! hdr.b   Params        24 bytes, counts and identifiers
//! hdr.c   Symbols       graphemes, phonemes, and a class table
//! hdr.d   Transitions   the FST proper — not yet parsed
//! ```
//!
//! ## Header (12 bytes at 0)
//!
//! ```text
//! +0  u8      -> struct +0x00
//! +1  u8      -> struct +0x02
//! +2  u8      -> struct +0x04
//! +3  u16 BE  params_off      -> struct +0x98
//! +5  u16 BE  symbols_off     -> struct +0x9c
//! +7  u16 BE  transitions_off -> struct +0xa0
//! +9  u24 BE  file_size       -> struct +0xa4
//! ```
//!
//! `EnglishUs6.9.atm` gives `12, 36, 1314, 1311323` — and 1,311,323 is the
//! file's exact size on disk, which is the cheapest possible sanity check.
//!
//! ## Params (24 bytes at `params_off`)
//!
//! ```text
//! +0  u8      -> +0x0c        +12 u8      -> +0x20
//! +1  u8      -> +0x0e        +13 u8      grapheme_count -> +0x24
//! +2  u8      -> +0x10        +14 u8      -> +0x26
//! +3  u8      record size, 24 -> +0x22     +15 u8      -> +0x28
//! +4  u32 BE  -> +0x18        +16 u8      -> +0x2a
//! +8  u32 BE  -> +0x1c        +17 u16 BE  -> +0x2c
//!                             +19 u16 BE  phoneme_count -> +0x30
//!                             +21 u24 BE  -> +0x34
//! ```
//!
//! The two `u32`s at `+4` and `+8` are **the same value** in every file seen
//! (`0x4178d11d` for `EnglishUs6.9`). That is the repeated pair visible in a
//! hex dump. It is not identified — plausibly a checksum stored twice, or a
//! created/modified timestamp pair; as a Unix time it would be 2004-10-19.
//! Unconfirmed, so it is exposed raw and not named.
//!
//! ## Symbols (at `symbols_off`)
//!
//! An 11-byte header, then three tables:
//!
//! ```text
//! +0  u8      -> +0x38
//! +1  u8      -> +0x3a
//! +2  u8      -> +0x3c
//! +3  u16 BE  section length, INCLUDING these 11 bytes -> +0x44
//! +5  u8      -> +0x48
//! +6  u8      -> +0x4c
//! +7  u8      -> +0x50
//! +8  u8      -> +0x54
//! +9  u16 BE  grapheme pool size in bytes
//! ```
//!
//! then `grapheme_count` entries, then a `u16 BE` phoneme pool size, then
//! `phoneme_count` entries, then a `u8` class-table width and a
//! `phoneme_count x width` table of bytes.
//!
//! Each string entry is `len_minus_1` followed by `len_minus_1 + 1` bytes,
//! **NUL included**. So `01 61 00` is the one-character grapheme `a`.
//!
//! For `EnglishUs6.9`: length 1278, so the section runs 36..1314 — exactly the
//! header's `transitions_off`. Pool size 93 = 31 graphemes x 3 bytes. The
//! graphemes are `'` then `a`..`z` and a few more.
//!
//! ## Transitions (at `transitions_off`)
//!
//! A 5-byte header, then a state table, then the transition records:
//!
//! ```text
//! +0  u8      -> +0x6c
//! +1  u8      -> +0x6e
//! +2  u8      -> +0x70
//! +3  u8      descriptor stride, 4 in every file seen -> +0x7c
//! +4  u8      -> +0x80
//!
//! then `state_count` descriptors of `stride` bytes:
//!     u24 BE  byte offset into the record array (cumulative)
//!     u8      transition count for this state
//!
//! then the records, 4 bytes each:
//!     b0  bit 7 = flag, bits 0..6 = symbol index
//!     b1  u16 BE target state
//!     b2
//!     b3  auxiliary byte
//! ```
//!
//! **This accounts for the whole file, exactly, in both vendor files.** The
//! descriptor offsets are cumulative in *bytes* (`count * 4`), and the final
//! total equals the number of bytes left after the state table:
//!
//! | | states | transitions | state table | records |
//! |---|---:|---:|---:|---:|
//! | EnglishUs | 62,247 | 265,254 | 248,988 B | 1,061,016 B |
//! | EnglishTE | 36,214 | 142,339 | 144,856 B | 569,356 B |
//!
//! Both sum to EOF with zero slack, and all 98,461 descriptors across the two
//! files agree with the running total.
//!
//! The loader expands this into a dense `state_count x column_count` table of
//! 16-byte cells, but only on the stdio path. Mapped, bin and slurped modes —
//! which is what the engine actually uses — keep pointers to the state table
//! (`+0x84`) and the record array (`+0x88`) and walk them in place. This
//! parser follows the sparse form, since the dense one is 62,247 x 62 x 16 =
//! 62 MB for a 1.3 MB file.
//!
//! On the flag bit: a cell's `+0xc` defaults to 1 and is cleared only when the
//! record's high bit is **clear**, which also writes `0xffffffff` to `+4`. So
//! high-bit-clear is the special case, not the common one. What it means is
//! `ELQDriveAutomaEx` territory, not the loader's.
//!
//! # Walking it
//!
//! `ELQDriveAutomaEx` (`0x92b98`) is the entry point, and it is small:
//!
//! ```text
//! n = tokenize(auto, word, symbuf)             FUN_0009211c
//! for i in 0..n:
//!     ph = walk(auto, arg, symbuf, n, i, prev) FUN_000928f0
//!     if 0 <= ph < phoneme_count:
//!         out.push(ph)
//!         prev = class_of(auto + 0x38, ph)     FUN_00092570
//! emit(auto, out, ...)                         FUN_00092a4c
//! ```
//!
//! The bounds check against `phoneme_count` is what identifies the return as a
//! phoneme index. `FUN_000928f0` is the actual per-position search and is
//! Phase 5/6 work — this crate parses the data, it does not run the automaton.
//!
//! ## A discrepancy worth knowing about
//!
//! The mapped/bin/slurped path skips an entry whose `len_minus_1` is `0xff`,
//! treating it as empty. **The stdio path does not** — its loop is a
//! `do/while`, so it always copies at least one byte. The engine loads these
//! mapped in practice, so the `0xff` check is what is implemented here, but
//! the two branches of the original genuinely disagree.

use crate::{Error, Result};

/// Source mode from the automaton's `+0x8c`, selected by the path's first
/// character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceMode {
    Stdio,
    Mapped,
    BinContainer,
    Slurped,
}

impl SourceMode {
    pub fn from_code(code: u16) -> Option<SourceMode> {
        match code {
            0 => Some(SourceMode::Stdio),
            1 => Some(SourceMode::Mapped),
            2 => Some(SourceMode::BinContainer),
            3 => Some(SourceMode::Slurped),
            _ => None,
        }
    }
}

/// Split a vendor automaton path into its source prefix and real filename.
pub fn split_source_prefix(path: &str) -> Option<(char, &str)> {
    let mut cs = path.chars();
    let prefix = cs.next()?;
    Some((prefix, cs.as_str()))
}

/// The 12-byte file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub f0: u8,
    pub f1: u8,
    pub f2: u8,
    pub params_off: u32,
    pub symbols_off: u32,
    pub transitions_off: u32,
    /// The file's own length, per the header.
    pub file_size: u32,
}

/// The 24-byte parameter block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub record_len: u8,
    pub grapheme_count: u16,
    pub phoneme_count: u16,
    /// The identical pair at `+4` and `+8`. Not identified.
    pub tag: (u32, u32),
    /// `u16 BE` at `+17` -> `+0x2c`. The width of the automaton's transition
    /// space; `FUN_00091828` uses it as the column count when it expands the
    /// dense table. 62 in both vendor files.
    pub column_count: u16,
    /// `u24 BE` at `+21` -> `+0x34`. The number of states.
    pub state_count: u32,
    /// Fields whose meaning is not established, in file order:
    /// `[0]..[2]` are bytes 0..2, then bytes 12, 14, 15, 16.
    pub unknown_bytes: [u8; 7],
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Symbols {
    /// One string per grapheme, NUL stripped.
    pub graphemes: Vec<String>,
    /// One string per phoneme, NUL stripped.
    pub phonemes: Vec<String>,
    /// `phoneme_count` rows of `class_width` bytes.
    pub classes: Vec<Vec<u8>>,
    pub class_width: usize,
    /// Declared length of the whole section, including its 11-byte header.
    pub section_len: u32,
}

/// One state: where its transitions start and how many it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateDesc {
    /// Byte offset into the record array, cumulative across states.
    pub record_off: u32,
    pub count: u8,
}

/// One 4-byte transition record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    /// Low 7 bits of byte 0 — an index into the grapheme/class space.
    pub symbol: u8,
    /// Bit 7 of byte 0. Clear is the special case; see the module docs.
    pub high_bit: bool,
    /// Target state, `u16 BE` from bytes 1 and 2.
    pub target: u16,
    /// Byte 3. Not identified.
    pub aux: u8,
}

#[derive(Debug, Clone, Default)]
pub struct Transitions {
    pub f0: u8,
    pub f1: u8,
    pub f2: u8,
    pub desc_stride: usize,
    pub f4: u8,
    pub states: Vec<StateDesc>,
    pub records: Vec<Transition>,
}

#[derive(Debug, Clone)]
pub struct Atm {
    pub header: Header,
    pub params: Params,
    pub symbols: Symbols,
    pub transitions: Transitions,
}

impl Atm {
    pub fn parse_bytes(name: &str, raw: &[u8]) -> Result<Atm> {
        let header = parse_header(name, raw)?;
        let params = parse_params(name, raw, header.params_off as usize)?;
        let symbols = parse_symbols(name, raw, header.symbols_off as usize, &params)?;
        let transitions = parse_transitions(name, raw, header.transitions_off as usize, &params)?;
        Ok(Atm {
            header,
            params,
            symbols,
            transitions,
        })
    }

    /// Transitions leaving one state.
    pub fn state_transitions(&self, state: usize) -> &[Transition] {
        let Some(d) = self.transitions.states.get(state) else {
            return &[];
        };
        let start = d.record_off as usize / 4;
        let end = (start + d.count as usize).min(self.transitions.records.len());
        &self.transitions.records[start.min(end)..end]
    }

    /// Every structural check the format makes possible, all of which held on
    /// both vendor files — so a failure here is a finding about the data.
    ///
    /// * the header states the file's own length;
    /// * the symbol section ends exactly where the transition table begins;
    /// * state descriptor offsets are a running total of `count * 4`;
    /// * that total consumes the file to the last byte.
    pub fn self_consistent(&self, actual_len: usize) -> bool {
        if self.header.file_size as usize != actual_len {
            return false;
        }
        if self.header.symbols_off + self.symbols.section_len != self.header.transitions_off {
            return false;
        }
        let mut cum = 0u32;
        for d in &self.transitions.states {
            if d.record_off != cum {
                return false;
            }
            cum += d.count as u32 * 4;
        }
        cum as usize == self.transitions.records.len() * 4
    }
}

pub fn parse_transitions(
    name: &str,
    raw: &[u8],
    at: usize,
    params: &Params,
) -> Result<Transitions> {
    need(name, raw, at, 5, "transitions header")?;
    let desc_stride = raw[at + 3] as usize;
    if desc_stride < 4 {
        return Err(Error::new(
            name,
            0,
            format!("descriptor stride {desc_stride} is too small for a u24 offset plus a count"),
        ));
    }

    let n_states = params.state_count as usize;
    let table = at + 5;
    need(name, raw, table, n_states * desc_stride, "state table")?;

    let mut states = Vec::with_capacity(n_states);
    let mut total = 0usize;
    for s in 0..n_states {
        let d = &raw[table + s * desc_stride..][..desc_stride];
        // The count is the LAST byte of the descriptor, whatever the stride.
        let count = d[desc_stride - 1];
        states.push(StateDesc {
            record_off: be24(d, 0),
            count,
        });
        total += count as usize;
    }

    let recs = table + n_states * desc_stride;
    need(name, raw, recs, total * 4, "transition records")?;
    let mut records = Vec::with_capacity(total);
    for i in 0..total {
        let r = &raw[recs + i * 4..][..4];
        records.push(Transition {
            symbol: r[0] & 0x7f,
            high_bit: r[0] & 0x80 != 0,
            target: be16(r, 1) as u16,
            aux: r[3],
        });
    }

    Ok(Transitions {
        f0: raw[at],
        f1: raw[at + 1],
        f2: raw[at + 2],
        desc_stride,
        f4: raw[at + 4],
        states,
        records,
    })
}

fn need(name: &str, raw: &[u8], at: usize, n: usize, what: &str) -> Result<()> {
    if at + n > raw.len() {
        return Err(Error::new(
            name,
            0,
            format!(
                "{what} needs {n} bytes at 0x{at:x}, file is {} long",
                raw.len()
            ),
        ));
    }
    Ok(())
}

fn be16(b: &[u8], i: usize) -> u32 {
    ((b[i] as u32) << 8) | b[i + 1] as u32
}

fn be24(b: &[u8], i: usize) -> u32 {
    ((b[i] as u32) << 16) | ((b[i + 1] as u32) << 8) | b[i + 2] as u32
}

fn be32(b: &[u8], i: usize) -> u32 {
    ((b[i] as u32) << 24) | ((b[i + 1] as u32) << 16) | ((b[i + 2] as u32) << 8) | b[i + 3] as u32
}

pub fn parse_header(name: &str, raw: &[u8]) -> Result<Header> {
    need(name, raw, 0, 12, "header")?;
    Ok(Header {
        f0: raw[0],
        f1: raw[1],
        f2: raw[2],
        params_off: be16(raw, 3),
        symbols_off: be16(raw, 5),
        transitions_off: be16(raw, 7),
        file_size: be24(raw, 9),
    })
}

pub fn parse_params(name: &str, raw: &[u8], at: usize) -> Result<Params> {
    need(name, raw, at, 24, "params")?;
    let b = &raw[at..at + 24];
    Ok(Params {
        record_len: b[3],
        grapheme_count: b[13] as u16,
        phoneme_count: be16(b, 19) as u16,
        tag: (be32(b, 4), be32(b, 8)),
        column_count: be16(b, 17) as u16,
        state_count: be24(b, 21),
        unknown_bytes: [b[0], b[1], b[2], b[12], b[14], b[15], b[16]],
    })
}

pub fn parse_symbols(name: &str, raw: &[u8], at: usize, params: &Params) -> Result<Symbols> {
    need(name, raw, at, 11, "symbols header")?;
    let section_len = be16(raw, at + 3);
    let mut p = at + 11;

    let graphemes = read_table(
        name,
        raw,
        &mut p,
        params.grapheme_count as usize,
        "graphemes",
    )?;

    // The phoneme pool size sits between the two tables and is not needed,
    // since each entry carries its own length.
    need(name, raw, p, 2, "phoneme pool size")?;
    p += 2;

    let phonemes = read_table(name, raw, &mut p, params.phoneme_count as usize, "phonemes")?;

    need(name, raw, p, 1, "class width")?;
    let class_width = raw[p] as usize;
    p += 1;

    let rows = params.phoneme_count as usize;
    need(name, raw, p, rows * class_width, "class table")?;
    let mut classes = Vec::with_capacity(rows);
    for _ in 0..rows {
        classes.push(raw[p..p + class_width].to_vec());
        p += class_width;
    }

    Ok(Symbols {
        graphemes,
        phonemes,
        classes,
        class_width,
        section_len,
    })
}

/// `count` entries of `len_minus_1` followed by `len_minus_1 + 1` bytes.
///
/// The stored bytes include the NUL terminator, which is stripped here. A
/// `len_minus_1` of `0xff` is an empty entry and consumes no payload.
fn read_table(
    name: &str,
    raw: &[u8],
    p: &mut usize,
    count: usize,
    what: &str,
) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        need(name, raw, *p, 1, what)?;
        let lm1 = raw[*p];
        *p += 1;
        if lm1 == 0xff {
            out.push(String::new());
            continue;
        }
        let n = lm1 as usize + 1;
        need(name, raw, *p, n, what)?;
        let bytes = &raw[*p..*p + n];
        *p += n;
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        out.push(bytes[..end].iter().map(|b| *b as char).collect());
        let _ = i;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real first 59 bytes of `EnglishUs6.9.atm`.
    const HEAD: &[u8] = &[
        0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x24, 0x05, 0x22, 0x14, 0x02, 0x5b, // header
        0x01, 0x00, 0x00, 0x18, 0x41, 0x78, 0xd1, 0x1d, 0x41, 0x78, 0xd1, 0x1d, // params
        0x12, 0x1f, 0x00, 0x38, 0x16, 0x00, 0x3e, 0x00, 0x4f, 0x00, 0xf3, 0x27, 0x01, 0x00, 0x00,
        0x04, 0xfe, 0x1b, 0x1c, 0x1d, 0x1e, 0x00, 0x5d, // symbols hdr
        0x01, 0x27, 0x00, 0x01, 0x61, 0x00, 0x01, 0x62, 0x00, 0x01, 0x63, 0x00,
    ];

    #[test]
    fn header_offsets_match_the_real_file() {
        let h = parse_header("x.atm", HEAD).unwrap();
        assert_eq!(h.params_off, 12);
        assert_eq!(h.symbols_off, 36);
        assert_eq!(h.transitions_off, 1314);
        // EnglishUs6.9.atm is exactly this many bytes on disk.
        assert_eq!(h.file_size, 1_311_323);
    }

    #[test]
    fn params_give_the_symbol_counts() {
        let h = parse_header("x.atm", HEAD).unwrap();
        let p = parse_params("x.atm", HEAD, h.params_off as usize).unwrap();
        assert_eq!(p.record_len, 24);
        assert_eq!(p.grapheme_count, 31);
        assert_eq!(p.phoneme_count, 79);
        // The pair really is identical; that is the whole observation.
        assert_eq!(p.tag.0, p.tag.1);
        assert_eq!(p.tag.0, 0x4178_d11d);
    }

    #[test]
    fn the_symbol_section_ends_where_transitions_begin() {
        let h = parse_header("x.atm", HEAD).unwrap();
        let p = parse_params("x.atm", HEAD, h.params_off as usize).unwrap();
        // Only the section header is needed for the length.
        let len = be16(HEAD, h.symbols_off as usize + 3);
        assert_eq!(len, 1278);
        assert_eq!(h.symbols_off + len, h.transitions_off);
        let _ = p;
    }

    #[test]
    fn string_entries_are_length_prefixed_and_nul_terminated() {
        let mut p = 47; // just past the 11-byte symbols header
        let g = read_table("x.atm", HEAD, &mut p, 4, "graphemes").unwrap();
        assert_eq!(g, vec!["'", "a", "b", "c"]);
        // 4 entries x 3 bytes.
        assert_eq!(p, 47 + 12);
    }

    #[test]
    fn an_ff_length_is_an_empty_entry() {
        let raw = [0xffu8, 0x01, b'a', 0x00];
        let mut p = 0;
        let t = read_table("x.atm", &raw, &mut p, 2, "t").unwrap();
        assert_eq!(t, vec!["", "a"]);
        assert_eq!(p, 4);
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        assert!(parse_header("x.atm", &HEAD[..8]).is_err());
        assert!(parse_params("x.atm", HEAD, 1200).is_err());
    }

    /// Two states, 2 and 1 transitions, laid out exactly as a vendor file.
    fn tiny_transitions() -> (Vec<u8>, Params) {
        let mut v = vec![0x01, 0x00, 0x00, 0x04, 0x04]; // 5-byte header, stride 4
                                                        // state 0: offset 0, 2 transitions
        v.extend_from_slice(&[0x00, 0x00, 0x00, 0x02]);
        // state 1: offset 8 (= 2 * 4), 1 transition
        v.extend_from_slice(&[0x00, 0x00, 0x08, 0x01]);
        // records
        v.extend_from_slice(&[0x85, 0x00, 0x2a, 0x07]); // high bit set, sym 5, -> 42
        v.extend_from_slice(&[0x03, 0x01, 0x00, 0x00]); // high bit clear, sym 3, -> 256
        v.extend_from_slice(&[0x7f, 0xff, 0xff, 0x01]); // sym 127, -> 65535
        let params = Params {
            record_len: 24,
            grapheme_count: 0,
            phoneme_count: 0,
            tag: (0, 0),
            column_count: 62,
            state_count: 2,
            unknown_bytes: [0; 7],
        };
        (v, params)
    }

    #[test]
    fn transition_records_decode() {
        let (raw, p) = tiny_transitions();
        let t = parse_transitions("x.atm", &raw, 0, &p).unwrap();
        assert_eq!(t.desc_stride, 4);
        assert_eq!(t.states.len(), 2);
        assert_eq!(t.records.len(), 3);

        assert_eq!(t.records[0].symbol, 5);
        assert!(t.records[0].high_bit);
        assert_eq!(t.records[0].target, 42);
        assert_eq!(t.records[0].aux, 7);

        // The high bit must not leak into the symbol index.
        assert_eq!(t.records[1].symbol, 3);
        assert!(!t.records[1].high_bit);
        assert_eq!(t.records[1].target, 256);
        assert_eq!(t.records[2].target, 65535);
    }

    #[test]
    fn descriptor_offsets_are_bytes_not_record_indices() {
        let (raw, p) = tiny_transitions();
        let t = parse_transitions("x.atm", &raw, 0, &p).unwrap();
        // State 1 starts after two 4-byte records, so 8, not 2.
        assert_eq!(t.states[0].record_off, 0);
        assert_eq!(t.states[1].record_off, 8);
        assert_eq!(t.states[1].count, 1);
    }

    #[test]
    fn state_transitions_slices_by_state() {
        let (raw, p) = tiny_transitions();
        let atm = Atm {
            header: parse_header("x.atm", HEAD).unwrap(),
            params: p.clone(),
            symbols: Symbols::default(),
            transitions: parse_transitions("x.atm", &raw, 0, &p).unwrap(),
        };
        assert_eq!(atm.state_transitions(0).len(), 2);
        assert_eq!(atm.state_transitions(1).len(), 1);
        assert_eq!(atm.state_transitions(1)[0].target, 65535);
        // Out of range is empty, not a panic.
        assert!(atm.state_transitions(99).is_empty());
    }

    #[test]
    fn a_truncated_record_array_is_an_error() {
        let (mut raw, p) = tiny_transitions();
        raw.truncate(raw.len() - 2);
        assert!(parse_transitions("x.atm", &raw, 0, &p).is_err());
    }

    #[test]
    fn source_modes_cover_zero_to_three_only() {
        assert_eq!(SourceMode::from_code(0), Some(SourceMode::Stdio));
        assert_eq!(SourceMode::from_code(3), Some(SourceMode::Slurped));
        assert_eq!(SourceMode::from_code(4), None);
        assert_eq!(SourceMode::from_code(0xffff), None);
    }

    #[test]
    fn the_prefix_is_one_character_and_the_rest_is_the_path() {
        let (p, rest) = split_source_prefix("mEnglishUs/EnglishUs6.9.atm").unwrap();
        assert_eq!(p, 'm');
        assert_eq!(rest, "EnglishUs/EnglishUs6.9.atm");
        assert_eq!(split_source_prefix(""), None);
    }
}
