//! `.phd` — the phonetic dictionary: word to phonetic transcription.
//!
//! Recovered from `ELQAddPhd` (Ghidra `0x8cd40`) and `ELQPhdBinfind`
//! (`0x8d0a8`). The whole format:
//!
//! ```text
//! 01 01                     optional magic; present => delta-encoded
//! <body>                    "word=transcription\n" records, ASCII-sorted
//! ```
//!
//! If and only if the first two bytes are `01 01`, they are dropped and the
//! rest is **delta-decoded in place**, each byte against the already-decoded
//! one before it:
//!
//! ```text
//! plain[0] = raw[0]
//! plain[i] = raw[i] - plain[i - 1]        (wrapping, u8)
//! ```
//!
//! `EnglishUs6.9.phd` starts `01 01 61 88 9a b0 9d a8 8d 8e 69 9a 84 6b ...`,
//! which decodes to `a's=\`HEI z\naaron'...` — so the first entry is the word
//! `a's` with the transcription `` `HEI z ``.
//!
//! The loader counts `=` bytes to size the entry array, then splits the body
//! into `{key, value}` pairs at the **first** `=` in each record. A record with
//! no `=` is a hard error in the engine, and is one here.
//!
//! Lookup is a binary search (`ELQPhdBinfind`), so the file must already be
//! sorted. ARM's `char` is unsigned, so `strcmp` orders by unsigned byte, which
//! is what Rust's `[u8]` comparison does — the orders agree with no special
//! handling.

use crate::{Error, Result};

/// The two bytes that mark a delta-encoded body.
pub const DELTA_MAGIC: [u8; 2] = [0x01, 0x01];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhdEntry {
    pub word: String,
    pub transcription: String,
}

#[derive(Debug, Clone, Default)]
pub struct Phd {
    pub entries: Vec<PhdEntry>,
    /// Whether the source carried the delta magic.
    pub delta_encoded: bool,
}

impl Phd {
    pub fn parse_bytes(name: &str, raw: &[u8]) -> Result<Phd> {
        let delta_encoded = raw.len() >= 2 && raw[..2] == DELTA_MAGIC;
        let body = if delta_encoded {
            delta_decode(&raw[2..])
        } else {
            raw.to_vec()
        };

        let mut entries = Vec::new();
        for (i, rec) in body.split(|b| *b == b'\n').enumerate() {
            if rec.is_empty() {
                continue;
            }
            let Some(eq) = rec.iter().position(|b| *b == b'=') else {
                return Err(Error::new(
                    name,
                    i + 1,
                    format!("record has no `=`: {:?}", latin1(&rec[..rec.len().min(32)])),
                ));
            };
            entries.push(PhdEntry {
                word: latin1(&rec[..eq]),
                transcription: latin1(&rec[eq + 1..]),
            });
        }

        Ok(Phd {
            entries,
            delta_encoded,
        })
    }

    /// Binary search, as `ELQPhdBinfind` does it.
    ///
    /// Returns `None` rather than scanning if the file turns out not to be
    /// sorted — the engine would miss the entry too, and quietly finding it
    /// here would hide a real difference.
    pub fn lookup(&self, word: &str) -> Option<&str> {
        let key = word.as_bytes();
        let (mut lo, mut hi) = (0i64, self.entries.len() as i64 - 1);
        while lo <= hi {
            let mid = ((lo + hi) / 2) as usize;
            match self.entries[mid].word.as_bytes().cmp(key) {
                std::cmp::Ordering::Less => lo = mid as i64 + 1,
                std::cmp::Ordering::Greater => hi = mid as i64 - 1,
                std::cmp::Ordering::Equal => return Some(&self.entries[mid].transcription),
            }
        }
        None
    }

    /// The first pair that is out of order, if any.
    ///
    /// Worth checking once per file: an unsorted `.phd` silently breaks the
    /// engine's own lookups, so a failure here is a finding about the data.
    pub fn first_unsorted(&self) -> Option<(usize, &str, &str)> {
        self.entries.windows(2).enumerate().find_map(|(i, w)| {
            if w[0].word.as_bytes() > w[1].word.as_bytes() {
                Some((i, w[0].word.as_str(), w[1].word.as_str()))
            } else {
                None
            }
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// `plain[i] = raw[i] - plain[i - 1]`, sequential and wrapping.
///
/// The engine does this in place over the buffer it just read, so each
/// subtraction uses the **decoded** predecessor, not the raw one.
pub fn delta_decode(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    let mut prev = 0u8;
    for (i, b) in raw.iter().enumerate() {
        let v = if i == 0 { *b } else { b.wrapping_sub(prev) };
        out.push(v);
        prev = v;
    }
    out
}

/// Inverse of [`delta_decode`], for round-trip tests.
pub fn delta_encode(plain: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(plain.len());
    let mut prev = 0u8;
    for (i, b) in plain.iter().enumerate() {
        out.push(if i == 0 { *b } else { b.wrapping_add(prev) });
        prev = *b;
    }
    out
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|b| *b as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real first bytes of `EnglishUs6.9.phd`.
    const HEAD: &[u8] = &[
        0x01, 0x01, 0x61, 0x88, 0x9a, 0xb0, 0x9d, 0xa8, 0x8d, 0x8e, 0x69, 0x9a, 0x84, 0x6b,
    ];

    #[test]
    fn decodes_the_real_file_head() {
        let plain = delta_decode(&HEAD[2..]);
        assert_eq!(latin1(&plain), "a's=`HEI z\na");
    }

    #[test]
    fn delta_round_trips() {
        let plain = b"a's=`HEI z\naaron's=`AI r H@ n z\n";
        assert_eq!(delta_decode(&delta_encode(plain)), plain);
    }

    fn sample() -> Vec<u8> {
        let plain = b"a's=`HEI z\naaron=`AI r H@ n\nzebra=`z i: b r @\n";
        let mut v = DELTA_MAGIC.to_vec();
        v.extend_from_slice(&delta_encode(plain));
        v
    }

    #[test]
    fn parses_a_delta_encoded_file() {
        let p = Phd::parse_bytes("x.phd", &sample()).unwrap();
        assert!(p.delta_encoded);
        assert_eq!(p.len(), 3);
        assert_eq!(p.entries[0].word, "a's");
        assert_eq!(p.entries[0].transcription, "`HEI z");
    }

    #[test]
    fn a_plain_file_needs_no_magic() {
        let p = Phd::parse_bytes("x.phd", b"aaron=`AI r H@ n\n").unwrap();
        assert!(!p.delta_encoded);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn binary_search_finds_every_entry() {
        let p = Phd::parse_bytes("x.phd", &sample()).unwrap();
        assert_eq!(p.lookup("a's"), Some("`HEI z"));
        assert_eq!(p.lookup("aaron"), Some("`AI r H@ n"));
        assert_eq!(p.lookup("zebra"), Some("`z i: b r @"));
        assert_eq!(p.lookup("nothing"), None);
        assert_eq!(p.first_unsorted(), None);
    }

    #[test]
    fn only_the_first_equals_splits_a_record() {
        let p = Phd::parse_bytes("x.phd", b"a=b=c\n").unwrap();
        assert_eq!(p.entries[0].word, "a");
        assert_eq!(p.entries[0].transcription, "b=c");
    }

    #[test]
    fn a_record_without_an_equals_is_an_error() {
        assert!(Phd::parse_bytes("x.phd", b"aaron\n").is_err());
    }
}
