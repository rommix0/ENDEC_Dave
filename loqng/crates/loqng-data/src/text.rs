//! The `.lex` and `.sde` obfuscation.
//!
//! Both are ordinary Latin-1 text with bit 7 set on every byte. `Dave.sde`
//! starts `b1 b1 ac b2 0a`, which is `11,2\n`; `EnglishUs6.9.lex` starts
//! `c5 ee e7 ec e9 f3 e8` = `English`. Clearing the high bit is the whole
//! transform — there is no key and no per-byte state.
//!
//! Bytes that already have bit 7 clear are left alone rather than rejected:
//! `Dave.sde` embeds `\n` as a literal `0x0a`, so the encoding is not applied
//! uniformly even within one file.

/// Decode a high-bit-set file into Latin-1 text.
pub fn decode(raw: &[u8]) -> String {
    raw.iter().map(|b| (b & 0x7f) as char).collect()
}

/// Re-encode, for round-trip tests and for writing a modified lexicon.
///
/// Newlines stay bare, matching what the vendor files actually contain.
pub fn encode(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| {
            let b = if (c as u32) < 0x100 { c as u8 } else { b'?' };
            if b == b'\n' || b == b'\r' {
                b
            } else {
                b | 0x80
            }
        })
        .collect()
}

/// Whether a file looks high-bit-encoded, so a caller can accept either form.
///
/// Plain ASCII text has no high bytes at all; an encoded file is almost
/// entirely high bytes. The midpoint separates them with enormous margin.
pub fn looks_encoded(raw: &[u8]) -> bool {
    let sample = &raw[..raw.len().min(4096)];
    if sample.is_empty() {
        return false;
    }
    let high = sample.iter().filter(|b| **b & 0x80 != 0).count();
    high * 2 > sample.len()
}

/// Decode if encoded, otherwise read as Latin-1.
pub fn decode_auto(raw: &[u8]) -> String {
    if looks_encoded(raw) {
        decode(raw)
    } else {
        raw.iter().map(|b| *b as char).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_sde_magic() {
        assert_eq!(decode(&[0xb1, 0xb1, 0xac, 0xb2, 0x0a]), "11,2\n");
    }

    #[test]
    fn decodes_the_lex_magic() {
        let raw = [0xc5, 0xee, 0xe7, 0xec, 0xe9, 0xf3, 0xe8];
        assert_eq!(decode(&raw), "English");
    }

    #[test]
    fn round_trips() {
        let s = "11,2\nVERSION=6\nN=1,2,3\n";
        assert_eq!(decode(&encode(s)), s);
    }

    #[test]
    fn detects_plain_text() {
        assert!(!looks_encoded(b"; a comment\n\"Key\" = \"Value\"\n"));
        assert!(looks_encoded(&[0xc5, 0xee, 0xe7, 0xec]));
    }
}
