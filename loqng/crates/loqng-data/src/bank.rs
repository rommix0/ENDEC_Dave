//! `.bin` — the coded unit bank, e.g. `Dave-19200.16000.loqmsx.bin` (21 MB).
//!
//! Big-endian `u32` header, which is worth noting because everything else the
//! engine touches is little-endian: this file is byte-order-independent on
//! purpose so one bank serves both host orders (compare `ELQIsIntelByteOrder`
//! at `LoqTTS6.so+0x906e8`).
//!
//! Comparing the three banks in the Dave tree separates the constants from the
//! per-voice values:
//!
//! ```text
//! word:        0   1   2     3   4       5      6  7       8   9    10  11
//! Dave      19200  10 320 16000  48  426582  12654  2  657106  96 1280  80
//! DaveAU    19200  10 320 16000  48   19958  18045  2  238482  91 1280  80
//! DaveGilded 19200 10 320 16000  48   26224  17731  2  259148  97 1280  80
//! ```
//!
//! What that establishes:
//!
//! * **word 0 = 19200 is the BITRATE in bits per second**, not a sample rate.
//!   Word 3 = 16000 is the output rate, and word 2 = 320 samples is a 20 ms
//!   frame — also Speex's wideband frame size. So the file name
//!   `Dave-19200.16000.loqmsx.bin` reads `<bitrate>.<sample rate>`.
//! * **word 8 is the offset where coded frames begin**, and **word 5 is the
//!   frame count**. Those two plus the file length give the coded frame size,
//!   and it comes out at exactly 48 bytes on all three banks:
//!
//! ```text
//! (21133042 - 657106) / 426582 = 48.0000     Dave
//! ( 1196466 - 238482) /  19958 = 48.0000     DaveAU
//! ( 1517900 - 259148) /  26224 = 48.0000     DaveGilded
//! ```
//!
//!   and 48 bytes x 8 x (16000 / 320) = **19,200 bps**, which closes the loop
//!   back on word 0. That is what identifies word 0 as a bitrate: an exact
//!   three-way agreement between the header, the file length and the name.
//!
//! * **word 4 = 48 is the coded frame size in bytes**, read as a `u16` at
//!   byte `0x12`. An earlier note here called it the header length and said it
//!   only coincided with the frame size; that was wrong — the decoder reads it
//!   as `bytesPerFrame`, and the measured 48 above is the same number.
//! * At 320 samples a frame, 426,582 frames is 2 h 22 m for Dave against
//!   6 m 39 s for DaveAU, matching their 21 MB and 1.2 MB sizes.
//! * **word 9 is the context count `N`** — 96 for Dave, 91 for DaveAU, 97 for
//!   DaveGilded. Every per-voice codebook is stored `N` times over, once per
//!   context, and [`BankTables`] slices them on it.
//! * **word 6's low byte is the XOR key** (`0x6e` for Dave). The rest of that
//!   word is still unidentified, and `raw` exposes it.
//!
//! # The preamble carries the codebooks
//!
//! Everything between byte `0x7c` and `data_offset` is per-voice codebook
//! data: a per-frame context index followed by seven codebooks. See
//! [`BankTables`]. That is what makes this bank undecodable by stock Speex,
//! and it is the other half of the three customisations documented in
//! `loqng-codec`.
//!
//! The decoder reads a frame with
//! `ELQBinGetBuffer(bin, frame_bytes * index + data_offset + bias, frame_bytes, 1, 0)`,
//! so the payload really is a flat array of fixed-size coded frames.

use crate::{Error, Result};

/// Bytes of fixed header, from word 4.
pub const HEADER_LEN: usize = 48;

/// Words in the fixed header.
pub const HEADER_WORDS: usize = HEADER_LEN / 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankHeader {
    /// Word 0. **Bitrate in bits per second**, not a sample rate.
    pub bitrate_bps: u32,
    /// Word 3. Sample rate the bank decodes to.
    pub output_rate: u32,
    /// Word 2. Samples per coded frame.
    pub frame_samples: u32,
    /// Word 4. **The coded frame size in bytes**, which the decoder reads as a
    /// `u16` at byte `0x12`. See [`BankHeader::frame_bytes_stated`].
    pub header_len: u32,
    /// Word 5. Coded frames in the bank.
    pub frames: u32,
    /// Word 8. Byte offset where the coded frames begin.
    pub data_offset: u32,
    /// The whole fixed header, including the words not yet identified.
    pub raw: [u32; HEADER_WORDS],
}

impl BankHeader {
    pub fn parse(name: &str, raw: &[u8]) -> Result<BankHeader> {
        if raw.len() < HEADER_LEN {
            return Err(Error::new(
                name,
                0,
                format!(
                    "{} bytes is shorter than a {HEADER_LEN}-byte header",
                    raw.len()
                ),
            ));
        }
        let mut w = [0u32; HEADER_WORDS];
        for (i, slot) in w.iter_mut().enumerate() {
            let b = &raw[i * 4..i * 4 + 4];
            *slot = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }

        let h = BankHeader {
            bitrate_bps: w[0],
            output_rate: w[3],
            frame_samples: w[2],
            header_len: w[4],
            frames: w[5],
            data_offset: w[8],
            raw: w,
        };

        // A little-endian read of this file yields 19200 as 0x004b0000; the
        // rates are the cheapest way to catch a byte-order mistake early.
        if h.output_rate == 0 || h.output_rate > 192_000 {
            return Err(Error::new(
                name,
                0,
                format!(
                    "implausible output rate {} — wrong byte order?",
                    h.output_rate
                ),
            ));
        }
        Ok(h)
    }

    /// Duration of the whole bank, from the frame count and frame size.
    pub fn duration_secs(&self) -> f64 {
        if self.output_rate == 0 {
            return 0.0;
        }
        (self.frames as f64) * (self.frame_samples as f64) / (self.output_rate as f64)
    }

    /// The tag the file name carries: `<bitrate>.<sample rate>`.
    ///
    /// `Dave-19200.16000.loqmsx.bin` must give `19200.16000`, which is how a
    /// caller can check a bank against the descriptor that named it.
    pub fn rate_tag(&self) -> String {
        format!("{}.{}", self.bitrate_bps, self.output_rate)
    }

    /// Frames per second, from the sample rate and frame size.
    pub fn frame_rate(&self) -> u32 {
        if self.frame_samples == 0 {
            return 0;
        }
        self.output_rate / self.frame_samples
    }

    /// Coded bytes per frame, from the bitrate. 48 on every Dave bank.
    ///
    /// The decoder reads exactly this many bytes per frame, so it is the
    /// stride of the payload array.
    pub fn frame_bytes(&self) -> u32 {
        let fps = self.frame_rate();
        if fps == 0 {
            return 0;
        }
        self.bitrate_bps / fps / 8
    }

    /// The same figure measured from the file, as a cross-check on the header.
    pub fn frame_bytes_measured(&self, file_len: usize) -> Option<u32> {
        let payload = (file_len as u64).checked_sub(self.data_offset as u64)?;
        if self.frames == 0 || payload % self.frames as u64 != 0 {
            return None;
        }
        u32::try_from(payload / self.frames as u64).ok()
    }

    /// Whether the header, the file length and the bitrate all agree.
    ///
    /// They do on all three Dave banks, exactly. That three-way agreement is
    /// what identifies word 0 as a bitrate rather than a sample rate.
    pub fn consistent_with(&self, file_len: usize) -> bool {
        self.frame_bytes_measured(file_len) == Some(self.frame_bytes())
    }

    /// The coded frame size as the header states it: a `u16` at byte `0x12`,
    /// which is the low half of word 4.
    ///
    /// [`BankHeader::frame_bytes`] derives the same number from the bitrate.
    /// They agree on every Dave bank, which is why either can be trusted.
    pub fn frame_bytes_stated(&self) -> u32 {
        self.raw[4] & 0xffff
    }

    /// Word 9: how many contexts every per-voice codebook is stored for.
    pub fn contexts(&self) -> u32 {
        self.raw[9]
    }

    /// The XOR key applied to every coded frame, from the low byte of word 6.
    ///
    /// Zero means the bank is not scrambled.
    pub fn xor_key(&self) -> u8 {
        (self.raw[6] & 0xff) as u8
    }
}

/// Speex's `SIG_SHIFT`. Codebook bytes are scaled by `SIG_SHIFT - log2(norm)`,
/// where `norm` is the per-band normalisation in the header.
pub const SIG_SHIFT: i32 = 14;

/// Where the codebook tables begin, immediately after the parameter block.
pub const TABLES_OFFSET: usize = 0x7c;

/// One codebook, stored once per context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableSpan {
    /// Byte offset in the file.
    pub offset: usize,
    /// Bytes per context.
    pub per_context: usize,
    /// Total bytes: `per_context * contexts`.
    pub len: usize,
}

impl TableSpan {
    /// The whole table.
    pub fn slice<'a>(&self, raw: &'a [u8]) -> Option<&'a [u8]> {
        raw.get(self.offset..self.offset + self.len)
    }

    /// One context's codebook, which is what an unquantiser is handed.
    pub fn context<'a>(&self, raw: &'a [u8], ctx: u32) -> Option<&'a [u8]> {
        let at = self
            .offset
            .checked_add(self.per_context.checked_mul(ctx as usize)?)?;
        raw.get(at..at.checked_add(self.per_context)?)
    }
}

/// The per-voice codebooks in the `.bin` preamble.
///
/// # Layout
///
/// Tables run back to back from [`TABLES_OFFSET`] and **end exactly at
/// `data_offset`**, which is a strong self-check: if the running offset lands
/// anywhere else, one of the counts was misread. [`BankTables::parse`]
/// enforces it.
///
/// ```text
/// 0x7c   idx        frames bytes            per-FRAME context selector
///        nb_cb1     hdr[0x28] * contexts    NB LSP stage 1
///        nb_cb2     hdr[0x2c] * contexts    NB LSP stage 2 (low split)
///        nb_cb3     hdr[0x30] * contexts    NB LSP stage 3 (high split)
///        nb_innov   hdr[0x34] * contexts    NB innovation
///        sb_cb1     hdr[0x38] * contexts    SB LSP stage 1
///        sb_cb2     hdr[0x3c] * contexts    SB LSP stage 2
///        sb_innov   hdr[0x48] * contexts    SB innovation
/// ```
///
/// For Dave those counts are 1280 (128 entries x order 10), 80 (16 x 5),
/// 80 (16 x 5), 320 (64 x 5), 256 (32 x 8), 64 (8 x 8) and 320 (32 x 10) —
/// which is where the stage counts in `loqng-codec`'s `lsp::unquant` come
/// from.
///
/// # `idx` is indexed by FRAME, not by voice
///
/// This is the part worth not re-deriving. The unquantisers take what looks
/// like a codebook-offset argument and use it as
/// `sel = *(u8 *)(state[+0x00] + arg)`; `state[+0x00]` is this table and the
/// argument is **the frame number**. So the codebook changes from frame to
/// frame across the bank, not just from voice to voice. A port that hoists the
/// selection out of the frame loop decodes the wrong codebook almost
/// everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankTables {
    pub contexts: u32,
    pub xor_key: u8,
    /// Per-frame context selector, one byte per frame.
    pub idx: TableSpan,
    pub nb_cb1: TableSpan,
    pub nb_cb2: TableSpan,
    pub nb_cb3: TableSpan,
    pub nb_innov: TableSpan,
    pub sb_cb1: TableSpan,
    pub sb_cb2: TableSpan,
    pub sb_innov: TableSpan,
    /// `SIG_SHIFT - log2(norm)`: how far a codebook byte is shifted up.
    pub nb_innov_shift: i32,
    pub sb_innov_shift: i32,
}

impl BankTables {
    pub fn parse(name: &str, raw: &[u8], h: &BankHeader) -> Result<BankTables> {
        let need = h.data_offset as usize;
        if raw.len() < need {
            return Err(Error::new(
                name,
                0,
                format!(
                    "{} bytes is shorter than the {need}-byte preamble",
                    raw.len()
                ),
            ));
        }

        let be32 = |at: usize| -> u32 {
            let b = &raw[at..at + 4];
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        };

        let contexts = h.contexts() as usize;
        let mut off = TABLES_OFFSET;
        let span = |per_context: usize, off: &mut usize| -> TableSpan {
            let s = TableSpan {
                offset: *off,
                per_context,
                len: per_context * contexts,
            };
            *off += s.len;
            s
        };

        let idx = TableSpan {
            offset: off,
            per_context: h.frames as usize,
            len: h.frames as usize,
        };
        off += idx.len;

        let nb_cb1 = span(be32(0x28) as usize, &mut off);
        let nb_cb2 = span(be32(0x2c) as usize, &mut off);
        let nb_cb3 = span(be32(0x30) as usize, &mut off);
        let nb_innov = span(be32(0x34) as usize, &mut off);
        let sb_cb1 = span(be32(0x38) as usize, &mut off);
        let sb_cb2 = span(be32(0x3c) as usize, &mut off);
        let sb_innov = span(be32(0x48) as usize, &mut off);

        // The whole point of the layout: it has to land exactly on the payload.
        if off != need {
            return Err(Error::new(
                name,
                0,
                format!("tables end at {off}, but the payload starts at {need}"),
            ));
        }

        Ok(BankTables {
            contexts: h.contexts(),
            xor_key: h.xor_key(),
            idx,
            nb_cb1,
            nb_cb2,
            nb_cb3,
            nb_innov,
            sb_cb1,
            sb_cb2,
            sb_innov,
            nb_innov_shift: SIG_SHIFT - ilog2(be32(0x4c)),
            sb_innov_shift: SIG_SHIFT - ilog2(be32(0x50)),
        })
    }

    /// The context a given frame decodes with.
    pub fn context_of(&self, raw: &[u8], frame: u32) -> Option<u8> {
        raw.get(self.idx.offset + frame as usize).copied()
    }
}

/// Floor log2, matching the decoder's own shift-until-one loop.
fn ilog2(mut x: u32) -> i32 {
    let mut n = 0;
    while x > 1 {
        x >>= 1;
        n += 1;
    }
    n
}

/// The span the LSP ladder covers, before the codebooks refine it.
///
/// Narrowband is order 10 with a step of 2048, wideband order 8 with a step of
/// 2560 — and `10 * 2048 == 8 * 2560 == 20480`. So the step is not a free
/// parameter: it is this span divided by the order, which is why the two
/// `.bss` globals the module reads hold 2048 and 2560.
pub const LADDER_SPAN: u32 = 20480;

/// Additive base of the narrowband LSP ladder.
pub const LADDER_BASE_NB: i32 = 0x800;

/// Additive base of the wideband LSP ladder. **Not the same as narrowband.**
pub const LADDER_BASE_SB: i32 = 0x1800;

/// One band's decode parameters, all derived from the bank header.
///
/// Nothing here is a magic constant except the two ladder bases: every index
/// width follows from `per_context / count` being a power of two, and the
/// ladder step from [`LADDER_SPAN`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandParams {
    /// LSP order: 10 narrowband, 8 wideband.
    pub order: usize,
    /// Ladder step, `LADDER_SPAN / order`.
    pub spacing: u16,
    /// Ladder base.
    pub base: i32,
    /// Index width per LSP stage.
    pub lsp_nbits: Vec<u32>,
    /// Coefficients each LSP stage refines; stage 0 is the whole order.
    pub lsp_counts: Vec<usize>,
    /// First coefficient each LSP stage refines.
    pub lsp_starts: Vec<usize>,
    /// Innovation subvector size.
    pub innov_subvect: usize,
    /// Innovation index width.
    pub innov_shape_bits: u32,
    /// How far an innovation codebook byte is shifted up.
    pub innov_shift: i32,
}

/// Both bands' decode parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankParams {
    pub nb: BandParams,
    pub sb: BandParams,
}

impl BankParams {
    /// Derive both bands from the header and the table sizes.
    ///
    /// The header carries the orders and splits directly:
    ///
    /// ```text
    /// word 21 nb order   word 22 nb split_lo   word 23 nb split_hi
    /// word 24 sb order   word 25 sb split_2
    /// word 28 nb innovation subvector   word 29 sb innovation subvector
    /// ```
    /// `raw` must cover at least the parameter block, i.e. [`TABLES_OFFSET`]
    /// bytes. [`BankHeader::raw`] only holds the first twelve words, so these
    /// fields have to come off the file.
    pub fn derive(name: &str, raw: &[u8], t: &BankTables) -> Result<BankParams> {
        if raw.len() < TABLES_OFFSET {
            return Err(Error::new(
                name,
                0,
                format!(
                    "{} bytes is shorter than the {TABLES_OFFSET}-byte parameter block",
                    raw.len()
                ),
            ));
        }
        let w = |i: usize| -> usize {
            let b = &raw[i * 4..i * 4 + 4];
            u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize
        };

        let nb_order = w(21);
        let sb_order = w(24);
        if nb_order == 0 || sb_order == 0 {
            return Err(Error::new(
                name,
                0,
                "bank states a zero LSP order".to_string(),
            ));
        }

        let bits = |span: &TableSpan, count: usize, what: &str| -> Result<u32> {
            if count == 0 || span.per_context % count != 0 {
                return Err(Error::new(
                    name,
                    0,
                    format!(
                        "{what}: {} bytes does not divide by {count}",
                        span.per_context
                    ),
                ));
            }
            let entries = span.per_context / count;
            if !entries.is_power_of_two() {
                return Err(Error::new(
                    name,
                    0,
                    format!("{what}: {entries} entries is not a power of two"),
                ));
            }
            Ok(entries.trailing_zeros())
        };

        let nb_slo = w(22);
        let nb_shi = w(23);
        let sb_split2 = w(25);
        let nb_sub = w(28);
        let sb_sub = w(29);

        let nb = BandParams {
            order: nb_order,
            spacing: (LADDER_SPAN / nb_order as u32) as u16,
            base: LADDER_BASE_NB,
            lsp_nbits: vec![
                bits(&t.nb_cb1, nb_order, "nb lsp stage 1")?,
                bits(&t.nb_cb2, nb_slo, "nb lsp stage 2")?,
                bits(&t.nb_cb3, nb_shi, "nb lsp stage 3")?,
            ],
            lsp_counts: vec![nb_order, nb_slo, nb_shi],
            lsp_starts: vec![0, 0, nb_slo],
            innov_subvect: nb_sub,
            innov_shape_bits: bits(&t.nb_innov, nb_sub, "nb innovation")?,
            innov_shift: t.nb_innov_shift,
        };

        let sb = BandParams {
            order: sb_order,
            spacing: (LADDER_SPAN / sb_order as u32) as u16,
            base: LADDER_BASE_SB,
            lsp_nbits: vec![
                bits(&t.sb_cb1, sb_order, "sb lsp stage 1")?,
                bits(&t.sb_cb2, sb_split2, "sb lsp stage 2")?,
            ],
            lsp_counts: vec![sb_order, sb_split2],
            lsp_starts: vec![0, 0],
            innov_subvect: sb_sub,
            innov_shape_bits: bits(&t.sb_innov, sb_sub, "sb innovation")?,
            innov_shift: t.sb_innov_shift,
        };

        Ok(BankParams { nb, sb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(words: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        for w in words {
            v.extend_from_slice(&w.to_be_bytes());
        }
        v
    }

    const DAVE: [u32; 12] = [
        19200, 10, 320, 16000, 48, 426_582, 12654, 2, 657_106, 96, 1280, 80,
    ];

    /// The real file lengths, which the header has to agree with.
    const DAVE_LEN: usize = 21_133_042;

    #[test]
    fn reads_the_dave_header() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        assert_eq!(h.bitrate_bps, 19200);
        assert_eq!(h.output_rate, 16000);
        assert_eq!(h.frame_samples, 320);
        assert_eq!(h.header_len, 48);
        assert_eq!(h.frames, 426_582);
        assert_eq!(h.data_offset, 657_106);
    }

    #[test]
    fn word_zero_is_a_bitrate_not_a_sample_rate() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        // 320 samples at 16 kHz is a 20 ms frame, so 50 frames a second.
        assert_eq!(h.frame_rate(), 50);
        // 19200 bps / 50 / 8 = 48 bytes a frame.
        assert_eq!(h.frame_bytes(), 48);
        // And the file length says the same, independently.
        assert_eq!(h.frame_bytes_measured(DAVE_LEN), Some(48));
        assert!(h.consistent_with(DAVE_LEN));
    }

    #[test]
    fn a_file_length_that_disagrees_is_caught() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        assert!(!h.consistent_with(DAVE_LEN + 1));
        assert_eq!(h.frame_bytes_measured(1000), None);
    }

    #[test]
    fn rate_tag_matches_the_file_name() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        assert_eq!(h.rate_tag(), "19200.16000");
    }

    #[test]
    fn duration_is_hours_for_the_full_bank() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        // 426582 * 320 / 16000 = 8531.64 s.
        assert!((h.duration_secs() - 8531.64).abs() < 0.01);
    }

    #[test]
    fn little_endian_is_rejected_rather_than_believed() {
        let mut v = Vec::new();
        for w in DAVE {
            v.extend_from_slice(&w.to_le_bytes());
        }
        assert!(BankHeader::parse("Dave.bin", &v).is_err());
    }

    #[test]
    fn a_truncated_file_is_an_error() {
        assert!(BankHeader::parse("Dave.bin", &[0u8; 20]).is_err());
    }

    #[test]
    fn word_four_states_the_frame_size_the_bitrate_predicts() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        assert_eq!(h.frame_bytes_stated(), 48);
        assert_eq!(h.frame_bytes_stated(), h.frame_bytes());
    }

    #[test]
    fn the_context_count_and_xor_key_come_out_of_words_nine_and_six() {
        let h = BankHeader::parse("Dave.bin", &header(&DAVE)).unwrap();
        assert_eq!(h.contexts(), 96);
        // 12654 = 0x316e, so the key is the low byte.
        assert_eq!(h.xor_key(), 0x6e);
    }

    #[test]
    fn the_norm_shifts_are_floor_log2() {
        // Header carries 128 and 256; SIG_SHIFT - log2 gives 7 and 6.
        assert_eq!(SIG_SHIFT - ilog2(128), 7);
        assert_eq!(SIG_SHIFT - ilog2(256), 6);
        assert_eq!(ilog2(1), 0);
        assert_eq!(ilog2(0), 0);
    }

    /// The seven codebooks have to land exactly on `data_offset`. Build a
    /// preamble whose counts do that, then move one and check it is caught.
    #[test]
    fn tables_must_end_exactly_on_the_payload() {
        const CONTEXTS: u32 = 4;
        const FRAMES: u32 = 10;
        let counts = [8usize, 4, 4, 6, 5, 3, 7];
        let total: usize = counts.iter().map(|c| c * CONTEXTS as usize).sum();
        let data_offset = TABLES_OFFSET + FRAMES as usize + total;

        let mut raw = vec![0u8; data_offset];
        let mut put = |at: usize, v: u32| raw[at..at + 4].copy_from_slice(&v.to_be_bytes());
        put(0, 19200);
        put(0x08, 320);
        put(0x0c, 16000);
        put(0x10, 48);
        put(0x14, FRAMES);
        put(0x20, data_offset as u32);
        put(0x24, CONTEXTS);
        for (i, c) in counts.iter().take(6).enumerate() {
            put(0x28 + i * 4, *c as u32);
        }
        put(0x48, counts[6] as u32);
        put(0x4c, 128);
        put(0x50, 256);

        let h = BankHeader::parse("t.bin", &raw).unwrap();
        let t = BankTables::parse("t.bin", &raw, &h).unwrap();
        assert_eq!(t.contexts, CONTEXTS);
        assert_eq!(t.idx.len, FRAMES as usize);
        assert_eq!(t.nb_cb1.offset, TABLES_OFFSET + FRAMES as usize);
        assert_eq!(t.sb_innov.offset + t.sb_innov.len, data_offset);
        assert_eq!(t.nb_innov_shift, 7);
        assert_eq!(t.sb_innov_shift, 6);

        // One count too large and the run no longer lands on the payload.
        let mut bad = raw.clone();
        bad[0x2b] = counts[1] as u8 + 1;
        let h2 = BankHeader::parse("t.bin", &bad).unwrap();
        assert!(BankTables::parse("t.bin", &bad, &h2).is_err());
    }

    /// The Dave banks' real parameter block, so the derivation is pinned to
    /// values read off the shipped files rather than to a synthetic case.
    #[test]
    fn dave_parameters_derive_from_the_header_alone() {
        const CONTEXTS: u32 = 96;
        const FRAMES: u32 = 8;
        // per-context sizes, in table order.
        let counts = [1280usize, 80, 80, 320, 256, 64, 320];
        let total: usize = counts.iter().map(|c| c * CONTEXTS as usize).sum();
        let data_offset = TABLES_OFFSET + FRAMES as usize + total;

        let mut raw = vec![0u8; data_offset];
        let mut put = |at: usize, v: u32| raw[at..at + 4].copy_from_slice(&v.to_be_bytes());
        put(0x00, 19200);
        put(0x08, 320);
        put(0x0c, 16000);
        put(0x10, 48);
        put(0x14, FRAMES);
        put(0x18, 12654);
        put(0x20, data_offset as u32);
        put(0x24, CONTEXTS);
        for (i, c) in counts.iter().take(6).enumerate() {
            put(0x28 + i * 4, *c as u32);
        }
        put(0x48, counts[6] as u32);
        put(0x4c, 128);
        put(0x50, 256);
        // orders and splits
        put(0x54, 10);
        put(0x58, 5);
        put(0x5c, 5);
        put(0x60, 8);
        put(0x64, 8);
        put(0x70, 5);
        put(0x74, 10);

        let h = BankHeader::parse("Dave.bin", &raw).unwrap();
        let t = BankTables::parse("Dave.bin", &raw, &h).unwrap();
        let p = BankParams::derive("Dave.bin", &raw, &t).unwrap();

        // These match `loqrs/sapi/loq_lsp.c`'s hand-written constants exactly,
        // but nothing here was copied from them.
        assert_eq!(p.nb.order, 10);
        assert_eq!(p.nb.lsp_nbits, vec![7, 4, 4]);
        assert_eq!(p.nb.lsp_counts, vec![10, 5, 5]);
        assert_eq!(p.nb.lsp_starts, vec![0, 0, 5]);
        assert_eq!(p.nb.spacing, 2048);
        assert_eq!(p.nb.base, 0x800);
        assert_eq!(p.nb.innov_subvect, 5);
        assert_eq!(p.nb.innov_shape_bits, 6);
        assert_eq!(p.nb.innov_shift, 7);

        assert_eq!(p.sb.order, 8);
        assert_eq!(p.sb.lsp_nbits, vec![5, 3]);
        assert_eq!(p.sb.lsp_counts, vec![8, 8]);
        assert_eq!(p.sb.spacing, 2560);
        assert_eq!(p.sb.base, 0x1800);
        assert_eq!(p.sb.innov_subvect, 10);
        assert_eq!(p.sb.innov_shape_bits, 5);
        assert_eq!(p.sb.innov_shift, 6);

        // The two ladders cover the same span; that is why the step is not a
        // free parameter.
        assert_eq!(p.nb.order as u32 * p.nb.spacing as u32, LADDER_SPAN);
        assert_eq!(p.sb.order as u32 * p.sb.spacing as u32, LADDER_SPAN);
    }

    #[test]
    fn a_codebook_that_is_not_a_power_of_two_is_rejected() {
        // 1280 / 10 = 128 is fine; 1290 / 10 = 129 is not a valid index width.
        let span = TableSpan {
            offset: 0,
            per_context: 1290,
            len: 1290,
        };
        let t = BankTables {
            contexts: 1,
            xor_key: 0,
            idx: span,
            nb_cb1: span,
            nb_cb2: span,
            nb_cb3: span,
            nb_innov: span,
            sb_cb1: span,
            sb_cb2: span,
            sb_innov: span,
            nb_innov_shift: 7,
            sb_innov_shift: 6,
        };
        let mut raw = vec![0u8; TABLES_OFFSET];
        raw[0x54..0x58].copy_from_slice(&10u32.to_be_bytes());
        raw[0x60..0x64].copy_from_slice(&8u32.to_be_bytes());
        assert!(BankParams::derive("t.bin", &raw, &t).is_err());
    }

    #[test]
    fn a_context_slice_steps_by_its_own_stride() {
        let span = TableSpan {
            offset: 10,
            per_context: 4,
            len: 12,
        };
        let raw: Vec<u8> = (0..30).collect();
        assert_eq!(span.context(&raw, 0).unwrap(), &[10, 11, 12, 13]);
        assert_eq!(span.context(&raw, 2).unwrap(), &[18, 19, 20, 21]);
        assert_eq!(span.slice(&raw).unwrap().len(), 12);
    }
}
