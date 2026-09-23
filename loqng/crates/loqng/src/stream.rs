//! The stream that passes between stages.
//!
//! The `Fon` -> `Cat` contract is the one boundary already recovered, and it
//! came for free: `FonWrite` (`LoqTTS6.so+0x45010`) serialises the phonetic
//! stream to text for the `PlainFonOut` instance parameter, so its field
//! accesses spell out the record layout.
//!
//! From the decompilation, a record is **0x10 bytes**:
//!
//! ```text
//! +0  u16   field_a   \
//! +2  u16   field_b    >  passed to sprintf as three unsigned values
//! +4  u16   field_c   /
//! +6  u8    phoneme   index into the language's phone table
//! +7  u8    proslab   prosodic label
//! +8  u32   text      pointer to a tag string, or null
//! +12       ...       4 bytes not touched by FonWrite
//! ```
//!
//! and the array ends at the first record whose `phoneme` is zero — `FonWrite`
//! loops `while (*(char *)(rec + i * 0x10 + 6) != '\0')`.
//!
//! # What the three `u16`s are
//!
//! A real dump, from `ttsSetOutput(instance, "fon", ELQDefaultOutputFunction,
//! path)` on "Testing one two three.":
//!
//! ```text
//! th	  79	 95	13107	D F
//! `E	 100	106	13107	D P
//! s	  90	122	13107	D P
//! ...
//! `i:	 196	 93	13107	D W
//! @	 500	 73	13107	D W
//! ```
//!
//! Three things line up, so these are named rather than left as `a`/`b`/`c`:
//!
//! * The stage computes exactly three per-phone quantities, and the Italian
//!   names say which: `FonDurata` (duration, `0x46eb0`), `FonTono` (pitch,
//!   `0x516f0`) and `FonGuadagno` (gain, `0x514b4`).
//! * Column 1 ranges 59..500 and the 500 is the trailing `@` pause — that is
//!   milliseconds of duration.
//! * Column 2 ranges 73..122, which is Hz for a male voice.
//! * Column 3 is **13107 on every phone**, which is [`GAIN_UNITY`] — 100%.
//!
//! The first two were circumstantial when written and have since been
//! confirmed against `FonDurata` and `FonTono` directly.
//!
//! **The third was read wrongly for a while and is worth flagging.** 13107 /
//! 16384 = 0.7999, so the column was taken for a gain of 0.8 in Q14. It is
//! not: `FonGuadagno` assembles 13107 as `3 * 17 * 257` and applies it as
//! `13107 * percent / 100`, making it a percentage scale whose unity is 13107
//! and whose ceiling is 26214 (200%). Two independent reverse-engineering
//! passes agree. A tidy-looking ratio is not evidence of a fixed-point format.
//!
//! `proslab` is decoded for display by `FonNum2ProsLab` (`0x40698`) and encoded
//! by `FonProsLab2Num` (`0x40618`). The four values that `FonWrite` treats
//! specially when choosing a separator — 6, 12, 13, 14 — are the ones flagged
//! in [`ProsLabel::is_boundary`].

/// One phone in the stream `Fon` hands to `Cat`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Phone {
    /// `+0`. Milliseconds. See the module docs for why this is duration.
    pub duration_ms: u16,
    /// `+2`. Hertz.
    pub f0_hz: u16,
    /// `+4`. Gain, where [`GAIN_UNITY`] is 100%. **Not Q14** — see below.
    pub gain: u16,
    /// `+6`. Index into the language's phone table; zero terminates the stream.
    pub phoneme: u8,
    /// `+7`.
    pub proslab: ProsLabel,
    /// The tag text at `+8`, when the record carried one.
    pub text: Option<String>,
    /// `+0xc`. Four bytes zeroed on reset whose writer has not been found.
    ///
    /// The record is 0x10 bytes and the fields above account for only 0xc of
    /// it. Carried so the layout is complete and a round-trip stays faithful.
    pub reserved: u32,
}

/// The gain value meaning 100%.
///
/// **This is not a Q14 fixed-point number**, despite 13107/16384 landing on a
/// suspiciously tidy 0.8. `FonGuadagno` converts a percentage with
/// `13107 * n / 100`, and 13107 is assembled in the code as `n * 3 * 17 * 257`
/// — so the scale is "13107 per 100%", the step is 1310 (10%) and the ceiling
/// is 26214 (200%).
///
/// Reading it as Q14 puts every unmodified phone at a gain of 0.8 instead of
/// 1.0 and makes the percentage arithmetic irreproducible. Two independent
/// reverse-engineering passes — one through `FonGuadagno`, one through
/// `ELQStr2Guad` and `SigRead`'s field initialisation — agree on 100%.
pub const GAIN_UNITY: u16 = 13107;

/// One step of the gain scale: 10%.
pub const GAIN_STEP: u16 = 1310;

/// One line of a `fon` dump, which is what `FonWrite` prints.
///
/// The numeric fields are the record's, but the phoneme and the labels have
/// been through `ELQNum2Fonema`, `ELQNum2Stile` and `FonNum2ProsLab`, so they
/// are names here and bytes in [`Phone`]. Mapping between the two needs the
/// language module's phone table and is Phase 6 work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DumpedPhone {
    pub phoneme: String,
    pub duration_ms: u16,
    pub f0_hz: u16,
    pub gain_q14: u16,
    /// `ELQNum2Stile` of the record's style byte.
    pub style: String,
    /// `FonNum2ProsLab` of `+7`.
    pub proslab: String,
}

/// Parse a `fon` stage dump.
///
/// `FonWrite` emits one record per line as
/// `<phoneme>\t<duration>\t<f0>\t<gain>\t<style> <proslab>`, with the numbers
/// space-padded. A line that does not have all five fields is skipped rather
/// than failing the parse, because the dump also carries the broad/narrow
/// transcription variant, which runs the phonemes together on one line.
pub fn parse_fon_dump(dump: &str) -> Vec<DumpedPhone> {
    let mut out = Vec::new();
    for line in dump.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            continue;
        }
        let (Ok(duration_ms), Ok(f0_hz), Ok(gain_q14)) = (
            f[1].trim().parse::<u16>(),
            f[2].trim().parse::<u16>(),
            f[3].trim().parse::<u16>(),
        ) else {
            continue;
        };
        let mut labels = f[4].split_whitespace();
        out.push(DumpedPhone {
            phoneme: f[0].to_string(),
            duration_ms,
            f0_hz,
            gain_q14,
            style: labels.next().unwrap_or("").to_string(),
            proslab: labels.next().unwrap_or("").to_string(),
        });
    }
    out
}

/// Bytes per record in the guest's array.
pub const PHONE_RECORD_LEN: usize = 0x10;

/// A prosodic label, kept as the raw byte because the mapping to names lives
/// in `FonNum2ProsLab` and is not transcribed yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProsLabel(pub u8);

impl ProsLabel {
    /// The labels `FonWrite` separates with a different string.
    ///
    /// It picks one separator for 6, 12, 13 and 14 and another for everything
    /// else, which is the only structure the serialiser exposes about them.
    pub fn is_boundary(self) -> bool {
        matches!(self.0, 6 | 12 | 13 | 14)
    }
}

/// The phone array between `Fon` and `Cat`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhoneStream {
    pub phones: Vec<Phone>,
}

impl PhoneStream {
    pub fn new() -> Self {
        PhoneStream::default()
    }

    pub fn len(&self) -> usize {
        self.phones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.phones.is_empty()
    }

    /// Read the array out of a flat copy of guest memory.
    ///
    /// Stops at the terminator or at the end of `raw`, whichever comes first,
    /// so a truncated capture yields what it has instead of failing. Pointer
    /// fields cannot be followed from a flat copy, so `text` is always `None`
    /// here; the capture side resolves them.
    pub fn from_records(raw: &[u8]) -> PhoneStream {
        let mut phones = Vec::new();
        for rec in raw.chunks_exact(PHONE_RECORD_LEN) {
            let phoneme = rec[6];
            if phoneme == 0 {
                break;
            }
            phones.push(Phone {
                duration_ms: u16::from_le_bytes([rec[0], rec[1]]),
                f0_hz: u16::from_le_bytes([rec[2], rec[3]]),
                gain: u16::from_le_bytes([rec[4], rec[5]]),
                phoneme,
                proslab: ProsLabel(rec[7]),
                text: None,
                reserved: u32::from_le_bytes([rec[12], rec[13], rec[14], rec[15]]),
            });
        }
        PhoneStream { phones }
    }

    /// Lay the array back out as the guest stores it, terminator included.
    ///
    /// The pointer at `+8` is written as zero — it is a guest address and has
    /// no meaning here — so this round-trips [`from_records`] but is not
    /// byte-identical to a guest buffer that carried tag text. The four bytes
    /// at `+0xc` are preserved rather than zeroed, since nothing is known
    /// about what writes them.
    pub fn to_records(&self) -> Vec<u8> {
        let mut out = vec![0u8; (self.phones.len() + 1) * PHONE_RECORD_LEN];
        for (i, p) in self.phones.iter().enumerate() {
            let r = &mut out[i * PHONE_RECORD_LEN..][..PHONE_RECORD_LEN];
            r[0..2].copy_from_slice(&p.duration_ms.to_le_bytes());
            r[2..4].copy_from_slice(&p.f0_hz.to_le_bytes());
            r[4..6].copy_from_slice(&p.gain.to_le_bytes());
            r[6] = p.phoneme;
            r[7] = p.proslab.0;
            r[12..16].copy_from_slice(&p.reserved.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two records then a terminator, laid out as the guest would.
    fn raw() -> Vec<u8> {
        let mut v = vec![0u8; 3 * PHONE_RECORD_LEN];
        v[0..2].copy_from_slice(&100u16.to_le_bytes());
        v[2..4].copy_from_slice(&200u16.to_le_bytes());
        v[4..6].copy_from_slice(&300u16.to_le_bytes());
        v[6] = 42;
        v[7] = 12;
        v[16..18].copy_from_slice(&7u16.to_le_bytes());
        v[22] = 9;
        v[23] = 1;
        v
    }

    #[test]
    fn reads_records_until_the_terminator() {
        let s = PhoneStream::from_records(&raw());
        assert_eq!(s.len(), 2);
        assert_eq!(s.phones[0].duration_ms, 100);
        assert_eq!(s.phones[0].f0_hz, 200);
        assert_eq!(s.phones[0].gain, 300);
        assert_eq!(s.phones[0].phoneme, 42);
        assert_eq!(s.phones[1].phoneme, 9);
    }

    /// The first three lines of a real `fon` dump for "Testing one two three."
    const FON_DUMP: &str = concat!(
        "th\t  79\t 95\t13107\tD F\n",
        "`E\t 100\t106\t13107\tD P\n",
        "@\t 500\t 73\t13107\tD W\n",
    );

    #[test]
    fn parses_a_real_fon_dump() {
        let p = parse_fon_dump(FON_DUMP);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].phoneme, "th");
        assert_eq!(p[0].duration_ms, 79);
        assert_eq!(p[0].f0_hz, 95);
        assert_eq!(p[0].style, "D");
        assert_eq!(p[0].proslab, "F");
        assert_eq!(p[1].phoneme, "`E");
        // The trailing pause is the longest thing in the stream.
        assert_eq!(p[2].phoneme, "@");
        assert_eq!(p[2].duration_ms, 500);
    }

    /// Every phone in an unmodified dump carries 100%.
    ///
    /// This test used to assert `13107 / 16384 == 0.8` and call that "where
    /// the name comes from". That was a coincidence read as a fact: the value
    /// is a percentage scale, not Q14. `FonGuadagno` builds it as
    /// `3 * 17 * 257 == 13107` and divides by 100.
    #[test]
    fn every_phone_carries_unity_gain() {
        for p in parse_fon_dump(FON_DUMP) {
            assert_eq!(p.gain_q14, GAIN_UNITY);
        }
        assert_eq!(3 * 17 * 257, GAIN_UNITY as i32);
        assert_eq!(GAIN_UNITY / 10, GAIN_STEP); // 1310.7 truncates to 1310
                                                // 200% is the documented ceiling.
        assert_eq!(GAIN_UNITY as u32 * 2, 26214);
    }

    #[test]
    fn non_record_lines_are_skipped_not_fatal() {
        let mixed = format!("th `E s d I N\n{FON_DUMP}");
        assert_eq!(parse_fon_dump(&mixed).len(), 3);
    }

    #[test]
    fn a_zero_phoneme_ends_the_stream_even_mid_buffer() {
        let mut v = raw();
        v[22] = 0;
        assert_eq!(PhoneStream::from_records(&v).len(), 1);
    }

    #[test]
    fn records_round_trip() {
        let s = PhoneStream::from_records(&raw());
        assert_eq!(PhoneStream::from_records(&s.to_records()), s);
    }

    #[test]
    fn boundary_labels_are_the_four_fonwrite_singles_out() {
        for n in [6u8, 12, 13, 14] {
            assert!(ProsLabel(n).is_boundary());
        }
        for n in [0u8, 1, 5, 7, 11, 15, 255] {
            assert!(!ProsLabel(n).is_boundary());
        }
    }

    #[test]
    fn a_truncated_buffer_yields_what_it_has() {
        let v = raw();
        let s = PhoneStream::from_records(&v[..PHONE_RECORD_LEN + 4]);
        assert_eq!(s.len(), 1);
    }
}
