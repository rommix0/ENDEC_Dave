//! `.sde` — the signal descriptor: what the voice bank's recordings are and
//! where each half-unit sits inside them.
//!
//! High-bit-encoded text (see [`crate::text`]), one record per line.
//! `Dave.sde` is 2,723,902 bytes and 229,489 lines:
//!
//! ```text
//! 11,2                                    two counts, meaning not yet known
//! VERSION=6
//! N=6374,124064,222632                    record counts, see below
//! U=# # @ @                               a recording's phonetic transcription
//! L# !'E #-% " "                          left half of a unit
//! R# #>E . "                              right half
//! ...
//! S=I                                     479 of these, meaning not yet known
//! ```
//!
//! The counts in `N=` check out against the file: 6,374 `U=` lines, and
//! 111,316 `L` plus 111,316 `R` = 222,632. The middle number, 124,064, is not
//! a line count and is unexplained.
//!
//! # The numeric encoding is base 93
//!
//! Recovered from `FUN_000385ec` (Ghidra `0x385ec`), which Ghidra decompiles
//! as returning `void` — it drops the accumulator entirely. The disassembly is
//! twelve instructions and unambiguous:
//!
//! ```asm
//! cmp  r2,#0x21          ; stop on any byte <= '!'
//! sub  r3,r2,#0x22       ; digit = c - '"'
//! add  r3,r1,r1,lsl #1   ; acc*3
//! rsb  r1,r3,r3,lsl #5   ; (acc*3)*31 = acc*93
//! add  r1,r1,r2          ; acc = acc*93 + digit
//! ```
//!
//! So `'"'` is 0, `'#'` is 1, up to `'~'` = 92, most significant digit first,
//! and any byte at or below `'!'` ends the number. That is why a field can be
//! one, two or three characters: it is just magnitude, not a tagged encoding.
//!
//! # Unit records
//!
//! From `FUN_0003c75c` (`0x3c75c`), the field decoder the `.sde` reader calls.
//! After the `L`/`R` letter a record reads:
//!
//! ```text
//! <index> [!<start>] <length> <skipped> <extra>
//! ```
//!
//! * `index` is the phoneme's position within its recording.
//! * A `!` prefix gives an **explicit** start. Without it the start is the
//!   previous record's end — units are contiguous.
//! * `end = start + length`, and the reader treats `end < start` as an error.
//! * One token is skipped, then `extra` is read. The pair is packed as
//!   `(length << 10) | (extra & 0xffff)`, which the engine unpacks with `>> 10`.
//!
//! `L` marks the left half of a diphone and `R` the right; the reader places
//! them at `slot = (base + index) * 2`, `+1` for `R`. That is the same
//! even/odd half encoding the unit-selection cost function uses.
//!
//! # Verified against the vendor files
//!
//! Every invariant holds on all three Dave banks:
//!
//! | | `N=` | records | explicit starts | continuity breaks |
//! |---|---|---:|---:|---:|
//! | Dave | 6374, 124064, 222632 | 222,632 | 6,374 | 0 |
//! | DaveAU | 388, 5229, 8906 | 8,906 | 388 | 0 |
//! | DaveGilded | 576, 5977, 9650 | 9,650 | 576 | 0 |
//!
//! The explicit-start count equals the utterance count exactly — the first
//! unit of each recording anchors, and all 222,632 of Dave's records chain
//! from there without a single break.
//!
//! `N=` is `[utterances, phonemes, half-unit records]`, where the phoneme
//! count splits the `U=` transcription on **space and `-`**, not space alone:
//! Dave's 101,111 space-separated tokens become 124,064 once `T$-Dg` and
//! `n-n` are split, which is exactly `N[1]`.

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdeRecord {
    /// The bare `11,2` first line.
    Header(Vec<i64>),
    /// `VERSION=6`.
    Version(i64),
    /// `N=6374,124064,222632`.
    Counts(Vec<i64>),
    /// `U=<transcription>` — the phonetic transcription of one recording.
    Utterance(String),
    /// `L<index> [!<start>] <length> <skipped> <extra>` — left half of a unit.
    ///
    /// `tokens` is everything after the `L`, split on spaces and still in
    /// base-93 text. Decoding needs the running end from the previous record,
    /// so it happens in [`Sde::units`] rather than here.
    Left { tokens: Vec<String> },
    /// `R<index> ...` — right half.
    Right { tokens: Vec<String> },
    /// `S=I`, 479 of them in `Dave.sde`. Purpose unknown.
    Section(String),
    /// A line matching none of the above, kept rather than dropped so a parse
    /// is never silently lossy.
    Other(String),
}

#[derive(Debug, Clone, Default)]
pub struct Sde {
    pub records: Vec<SdeRecord>,
}

impl Sde {
    pub fn parse(name: &str, text: &str) -> Result<Sde> {
        let mut records = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            records.push(parse_line(name, i + 1, line)?);
        }
        Ok(Sde { records })
    }

    pub fn parse_bytes(name: &str, raw: &[u8]) -> Result<Sde> {
        Sde::parse(name, &crate::text::decode_auto(raw))
    }

    pub fn version(&self) -> Option<i64> {
        self.records.iter().find_map(|r| match r {
            SdeRecord::Version(v) => Some(*v),
            _ => None,
        })
    }

    /// The `N=` counts, if present.
    pub fn counts(&self) -> Option<&[i64]> {
        self.records.iter().find_map(|r| match r {
            SdeRecord::Counts(v) => Some(v.as_slice()),
            _ => None,
        })
    }

    pub fn utterances(&self) -> impl Iterator<Item = &str> {
        self.records.iter().filter_map(|r| match r {
            SdeRecord::Utterance(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// `(utterances, left halves, right halves)`.
    pub fn tally(&self) -> (usize, usize, usize) {
        let mut t = (0, 0, 0);
        for r in &self.records {
            match r {
                SdeRecord::Utterance(_) => t.0 += 1,
                SdeRecord::Left { .. } => t.1 += 1,
                SdeRecord::Right { .. } => t.2 += 1,
                _ => {}
            }
        }
        t
    }

    /// Decode every `L`/`R` record, chaining implicit starts.
    ///
    /// A record without a `!` begins where the previous one ended, so this has
    /// to run in file order over the whole file — an individual record cannot
    /// be decoded on its own.
    pub fn units(&self) -> Vec<Unit> {
        let mut out = Vec::new();
        let mut end = 0u32;
        for r in &self.records {
            let (side, tokens) = match r {
                SdeRecord::Left { tokens } => (Side::Left, tokens),
                SdeRecord::Right { tokens } => (Side::Right, tokens),
                _ => continue,
            };
            if tokens.len() < 3 {
                continue;
            }
            let index = decode_num(&tokens[0]);
            let mut p = 1;
            let (start, explicit_start) = match tokens[1].strip_prefix('!') {
                Some(v) => {
                    p = 2;
                    (decode_num(v), true)
                }
                None => (end, false),
            };
            let length = decode_num(&tokens[p]);
            // One token is skipped between the length and the extra.
            let extra = tokens.get(p + 2).map(|t| decode_num(t)).unwrap_or(0);
            end = start.wrapping_add(length);
            out.push(Unit {
                side,
                index,
                start,
                length,
                extra,
                explicit_start,
            });
        }
        out
    }

    /// Every record chains from the previous one, except where a `!` anchors.
    ///
    /// Held on all three Dave banks across 241,188 records with zero breaks,
    /// so a failure is a finding about the data.
    pub fn continuity_holds(&self) -> bool {
        let mut end = 0u32;
        for u in self.units() {
            if !u.explicit_start && u.start != end {
                return false;
            }
            end = u.end();
        }
        true
    }

    /// Whether the `N=` header agrees with what was actually read.
    ///
    /// The first count is the utterance total and the third is left+right.
    /// The second is not a line count and is not checked.
    pub fn counts_agree(&self) -> bool {
        let Some(n) = self.counts() else {
            return false;
        };
        if n.len() < 3 {
            return false;
        }
        let (u, l, r) = self.tally();
        n[0] == u as i64 && n[2] == (l + r) as i64
    }
}

/// Which half of a diphone a record describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// One decoded half-unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unit {
    pub side: Side,
    /// Phoneme position within its recording.
    pub index: u32,
    pub start: u32,
    pub length: u32,
    /// The value packed into the low bits alongside the length.
    pub extra: u32,
    /// Whether the record carried a `!` start rather than chaining.
    pub explicit_start: bool,
}

impl Unit {
    pub fn end(&self) -> u32 {
        self.start + self.length
    }

    /// The engine's packed second word: `(length << 10) | (extra & 0xffff)`.
    ///
    /// It unpacks the length with `>> 10`, so the two genuinely share bits and
    /// a large `extra` would corrupt the length. Reproduced as written.
    pub fn packed(&self) -> u32 {
        (self.length << 10) | (self.extra & 0xffff)
    }

    /// Slot in the engine's unit array: `(base + index) * 2`, `+1` for a right
    /// half. Same even/odd diphone encoding the selection cost uses.
    pub fn slot(&self, base: u32) -> u32 {
        (base + self.index) * 2 + if self.side == Side::Right { 1 } else { 0 }
    }
}

/// Decode one base-93 field, as `FUN_000385ec` does.
///
/// Digits run from `'"'` = 0 to `'~'` = 92, most significant first. Any byte
/// at or below `'!'` terminates, which is how the `!` explicit-start marker
/// and the field separator both end a number.
pub fn decode_num(s: &str) -> u32 {
    let mut acc: u32 = 0;
    for b in s.bytes() {
        if b <= 0x21 {
            break;
        }
        acc = acc.wrapping_mul(93).wrapping_add((b - 0x22) as u32);
    }
    acc
}

/// Encode back to base 93, for round-trip tests.
pub fn encode_num(mut v: u32) -> String {
    if v == 0 {
        return "\"".to_string();
    }
    let mut out = Vec::new();
    while v > 0 {
        out.push(b'"' + (v % 93) as u8);
        v /= 93;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

fn parse_line(file: &str, line: usize, s: &str) -> Result<SdeRecord> {
    if let Some(v) = s.strip_prefix("VERSION=") {
        let n = v
            .trim()
            .parse::<i64>()
            .map_err(|_| Error::new(file, line, format!("bad VERSION `{v}`")))?;
        return Ok(SdeRecord::Version(n));
    }
    if let Some(v) = s.strip_prefix("N=") {
        return Ok(SdeRecord::Counts(numbers(file, line, v)?));
    }
    if let Some(v) = s.strip_prefix("U=") {
        return Ok(SdeRecord::Utterance(v.to_string()));
    }
    if let Some(v) = s.strip_prefix("S=") {
        return Ok(SdeRecord::Section(v.to_string()));
    }

    let mut cs = s.chars();
    match (cs.next(), cs.next()) {
        // Only the L/R letter is consumed. What follows is the index token,
        // which is base-93 and therefore multi-character once a recording has
        // more than 93 phonemes — treating it as a single tag char silently
        // corrupts every long utterance.
        (Some(side @ ('L' | 'R')), Some(_)) => {
            let tokens: Vec<String> = s[side.len_utf8()..]
                .split(' ')
                .filter(|f| !f.is_empty())
                .map(str::to_string)
                .collect();
            Ok(if side == 'L' {
                SdeRecord::Left { tokens }
            } else {
                SdeRecord::Right { tokens }
            })
        }
        // The first line is a bare comma-separated pair with no tag.
        _ if line == 1 && s.contains(',') => Ok(SdeRecord::Header(numbers(file, line, s)?)),
        _ => Ok(SdeRecord::Other(s.to_string())),
    }
}

fn numbers(file: &str, line: usize, s: &str) -> Result<Vec<i64>> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<i64>()
                .map_err(|_| Error::new(file, line, format!("bad number `{p}`")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SDE: &str = concat!(
        "11,2\n",
        "VERSION=6\n",
        "N=2,7,4\n",
        "U=# # @ @\n",
        "L# !'E #-% \" \"\n",
        "R# #>E . \"\n",
        "U=# `i: T$-Dg #\n",
        "L$ $l) 0 \"\n",
        "R, ##7 $ n\n",
        "S=I\n",
    );

    #[test]
    fn splits_every_record_kind() {
        let s = Sde::parse("Dave.sde", SDE).unwrap();
        assert_eq!(s.version(), Some(6));
        assert_eq!(s.counts(), Some(&[2i64, 7, 4][..]));
        assert_eq!(s.tally(), (2, 2, 2));
        assert!(matches!(s.records[0], SdeRecord::Header(_)));
        assert!(matches!(s.records.last(), Some(SdeRecord::Section(_))));
    }

    #[test]
    fn keeps_unit_fields_verbatim() {
        let s = Sde::parse("Dave.sde", SDE).unwrap();
        let SdeRecord::Left { tokens } = &s.records[4] else {
            panic!("expected a left half");
        };
        // The index token comes first and is NOT a single tag character.
        assert_eq!(tokens, &["#", "!'E", "#-%", "\"", "\""]);
    }

    #[test]
    fn the_hash_prefixed_three_char_field_is_one_field() {
        let s = Sde::parse("Dave.sde", SDE).unwrap();
        let SdeRecord::Right { tokens } = &s.records[8] else {
            panic!("expected a right half");
        };
        assert_eq!(tokens, &[",", "##7", "$", "n"]);
    }

    #[test]
    fn counts_are_checked_against_the_body() {
        let s = Sde::parse("Dave.sde", SDE).unwrap();
        // 2 utterances and 2+2 halves, against the declared N=2,7,4.
        assert!(s.counts_agree());
    }

    #[test]
    fn base93_digits_start_at_double_quote() {
        assert_eq!(decode_num("\""), 0);
        assert_eq!(decode_num("#"), 1);
        assert_eq!(decode_num("0"), 14);
        assert_eq!(decode_num("~"), 92);
        // Values checked by hand against the real Dave.sde.
        assert_eq!(decode_num("'E"), 500);
        assert_eq!(decode_num("#-%"), 9675);
        assert_eq!(decode_num("$l)"), 24187);
    }

    #[test]
    fn anything_at_or_below_bang_terminates_a_number() {
        // This is how both the field separator and the `!` marker end a value.
        assert_eq!(decode_num("!"), 0);
        assert_eq!(decode_num("#-% "), decode_num("#-%"));
        assert_eq!(decode_num("#!junk"), 1);
    }

    #[test]
    fn numbers_round_trip() {
        for v in [0u32, 1, 92, 93, 500, 9675, 24187, 853_161_782] {
            assert_eq!(decode_num(&encode_num(v)), v);
        }
    }

    /// The real first four unit records of `Dave.sde`.
    const UNITS: &str = concat!(
        "11,2\nVERSION=6\nN=1,1,4\nU=# # @ @\n",
        "L# !'E #-% \" \"\n",
        "R# #>E . \"\n",
        "L$ $l) 0 \"\n",
        "R$ %&A @ \"\n",
    );

    #[test]
    fn units_decode_to_the_values_verified_against_the_vendor_file() {
        let u = Sde::parse("Dave.sde", UNITS).unwrap().units();
        assert_eq!(u.len(), 4);
        assert_eq!((u[0].index, u[0].start, u[0].length), (1, 500, 9675));
        assert!(u[0].explicit_start);
        assert_eq!((u[1].index, u[1].start, u[1].length), (1, 10175, 11288));
        assert_eq!((u[2].index, u[2].start, u[2].length), (2, 21463, 24187));
        assert_eq!((u[3].index, u[3].start, u[3].length), (2, 45650, 26350));
    }

    #[test]
    fn implicit_starts_chain_from_the_previous_end() {
        let u = Sde::parse("Dave.sde", UNITS).unwrap().units();
        for w in u.windows(2) {
            assert!(!w[1].explicit_start);
            assert_eq!(w[1].start, w[0].end());
        }
        assert!(Sde::parse("Dave.sde", UNITS).unwrap().continuity_holds());
    }

    #[test]
    fn left_and_right_take_even_and_odd_slots() {
        let u = Sde::parse("Dave.sde", UNITS).unwrap().units();
        assert_eq!(u[0].side, Side::Left);
        assert_eq!(u[1].side, Side::Right);
        assert_eq!(u[0].slot(0), 2);
        assert_eq!(u[1].slot(0), 3);
    }

    #[test]
    fn the_packed_word_unpacks_with_a_shift_of_ten() {
        let u = Sde::parse("Dave.sde", UNITS).unwrap().units();
        assert_eq!(u[0].packed() >> 10, u[0].length);
    }

    #[test]
    fn nothing_is_silently_dropped() {
        let s = Sde::parse("x.sde", "11,2\nwhat is this\n").unwrap();
        assert_eq!(s.records.len(), 2);
        assert!(matches!(s.records[1], SdeRecord::Other(_)));
    }
}
