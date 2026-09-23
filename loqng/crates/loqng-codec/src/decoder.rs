//! `nb_decode` and `sb_decode` at frame level — the top of the codec.
//!
//! # Why this is transcription and not reverse engineering
//!
//! `loqrs/sapi/` holds a complete C reimplementation of this module that is
//! **proven bit-exact against the ARM decoder**, and it is built on *stock*
//! Speex 1.2beta1. `loqrs/speex_mod/modes.c.patch` is the entire diff: twelve
//! insertions, six deletions, one file. It repoints two decode submodes at
//! Loquendo's own unquantisers and changes nothing else.
//!
//! So Loquendo's frame loop **is** Speex's frame loop. Only three things
//! differ, and all three are already ported:
//!
//! | difference | where it lives |
//! |---|---|
//! | per-voice LSP codebook | [`crate::lsp::unquant`] |
//! | per-voice innovation codebook | [`crate::innov::unquant`] |
//! | no output highpass | simply not called here |
//!
//! Everything else — the gain ladders, the 3-tap pitch predictor, the QMF
//! synthesis bank, the LSP interpolation and margin — is stock, which is
//! exactly why the rebuilt shim decodes correctly using the stock tables in
//! [`crate::tables`].
//!
//! # The configuration, from the proven shim
//!
//! `loqmsx_dll.c::make_decoder` sets, and nothing else:
//!
//! ```text
//! SPEEX_SET_ENH               0   on the wideband state
//! SPEEX_SET_SUBMODE_ENCODING  0   on both states
//! SPEEX_SET_LOW_MODE          5   narrowband submode 5
//! SPEEX_SET_HIGH_MODE         2   wideband submode 2
//! SPEEX_SET_HIGHPASS          0   on both states
//! ```
//!
//! `SUBMODE_ENCODING = 0` is the load-bearing one: it means the submode is
//! **forced, never read from the bitstream**. Every submode-header parse, the
//! wideband-skip loop and the in-band request handling in upstream
//! `nb_decode` are therefore dead, and the frame's very first bit is LSP data.
//! A port that keeps the header parse consumes bits that are not there and
//! produces noise.
//!
//! # The bit budget proves the parameter set
//!
//! Dave's coded frame is 48 bytes = 384 bits. Adding up the fields this
//! decoder reads:
//!
//! ```text
//! narrowband   LSP 7 + 4 + 4                            =  15
//!              frame gain (qe)                          =   5
//!              4 subframes of (pitch 7 + gain 7
//!                              + subframe gain 3
//!                              + innovation 8 * 6)      = 260
//! wideband     LSP 5 + 3                                =   8
//!              4 subframes of (qgc 4 + innovation 4 * 5) =  96
//!                                                         ----
//!                                                          384
//! ```
//!
//! That is an exact fit, and it is the cheapest end-to-end check available:
//! any wrong field width shows up as a non-multiple of 8 here long before any
//! audio is produced. [`tests::the_frame_budget_is_exactly_forty_eight_bytes`]
//! asserts it against the constants this file actually uses.
//!
//! # Two one-subframe delays, in both bands
//!
//! Easy to miss and audible if missed. In `nb_decode` the output is taken as
//! `out[i] = exc[i - subframeSize]`, and the synthesis filter runs on
//! `interp_qlpc` — the *previous* subframe's coefficients — which is only
//! replaced by this subframe's `ak` **after** filtering. `sb_decode` does the
//! same through its 40-sample `exc_buf`. The `pi_gain` handed to the wideband
//! band is likewise computed from the lagged coefficients.
//!
//! # Traps in reading the upstream source
//!
//! * **`iir_mem2` is selected by `PRECISION16`, not `FIXED_POINT`.** The build
//!   defines only `FIXED_POINT`, so the version that compiles is the *second*
//!   one in `filters.c` — `SATURATE(x, 805306368)` with `MAC16_32_Q15` — not
//!   the 16-bit one that appears first. [`crate::filters::iir32`] matches the
//!   compiled arm.
//! * **`fixed_arm4.h` overrides exactly three macros**: `MULT16_32_Q14`,
//!   `MULT16_32_Q15` and `DIV32_16`. `PDIV32_16` is *not* among them, so it
//!   keeps the generic definition and performs a plain C division rather than
//!   the restoring division in [`crate::math::div15`]. Using `div15` for both
//!   would be wrong; see [`pdiv32_16`].
//! * Reading these files with the `#if` lines stripped silently merges the
//!   fixed- and floating-point arms of nearly every expression.

use crate::bits::Bits;
use crate::filters::{iir16, iir32, signal_mul};
use crate::innov::{SplitCbParams, VoiceCodebook};
use crate::lpc::lsp_to_lpc;
use crate::lsp::{self, LspStage};
use crate::ltp::{self, LtpParams};
use crate::math::compute_rms16;
use crate::nb::fold_innovation;
use crate::sb::qmf_combine;
use crate::tables::{EXC_GAIN_SCAL3, GAIN_CDBK_NB, GC_QUANT_BOUND, OL_GAIN, QMF_H0, QMF_H1};

/// Narrowband frame length, in samples.
pub const NB_FRAME: usize = 160;
/// Narrowband subframe length.
pub const NB_SUBFRAME: usize = 40;
/// Subframes per frame, both bands.
pub const SUBFRAMES: usize = 4;
/// Narrowband LPC order.
pub const NB_ORDER: usize = 10;
/// Lowest pitch lag, added to the 7-bit pitch field.
pub const PITCH_START: i32 = 17;
/// Highest pitch lag. Only used to size the history.
pub const PITCH_END: i32 = 144;

/// Wideband LPC order. **Not the same as [`NB_ORDER`].**
pub const SB_ORDER: usize = 8;
/// Samples per output frame: two bands at 160 each.
pub const FULL_FRAME: usize = 2 * NB_FRAME;
/// QMF synthesis filter length.
pub const QMF_ORDER: usize = 64;

/// LSP ladder spacing, narrowband.
///
/// The module reads this from `.bss` at [`crate::lsp::SPACING_ADDR_NB`] rather
/// than baking it in, so it is in principle per-voice. This is the value Dave
/// uses, taken from `loq_lsp.c`'s `NB_STEP` in the bit-exact shim.
pub const NB_SPACING: u16 = 2048;

/// LSP ladder spacing, wideband (`loq_lsp.c`'s `SB_STEP`).
///
/// Both ladders span the same 20480, as `loqng_data::bank::LADDER_SPAN` says:
/// 10 x 2048 narrowband, 8 x 2560 wideband.
pub const SB_SPACING: u16 = 2560;

/// `28406`, the Q15 factor applied to the frame gain ladder.
pub const OL_GAIN_FACTOR: i16 = 28406;

/// `28626`, the Q15 factor applied to the high-band gain ladder.
pub const GC_FACTOR: i16 = 28626;

/// Narrowband excitation history, in samples: `2*max_pitch + subframe + 12`.
const NB_HIST: usize = 2 * PITCH_END as usize + NB_SUBFRAME + 12;
/// Total narrowband excitation buffer.
const NB_EXC_BUF: usize = NB_FRAME + NB_HIST;
/// Index of `exc[0]` within that buffer.
const NB_EXC_OFF: usize = 2 * PITCH_END as usize + NB_SUBFRAME + 6;

/// `split_cb_nb`: 8 subvectors of 5, 6-bit shape index, no sign bit.
pub const NB_SPLIT_CB: SplitCbParams = SplitCbParams {
    subvect_size: 5,
    nb_subvect: 8,
    shape_bits: 6,
    have_sign: false,
};

/// `split_cb_high_lbr`: 4 subvectors of 10, 5-bit shape index, no sign bit.
pub const SB_SPLIT_CB: SplitCbParams = SplitCbParams {
    subvect_size: 10,
    nb_subvect: 4,
    shape_bits: 5,
    have_sign: false,
};

/// `ltp_params_nb`: 7-bit pitch, 7-bit gain index.
pub const NB_LTP: LtpParams = LtpParams {
    pitch_bits: 7,
    gain_bits: 7,
};

/// What can go wrong. The live path for a well-formed bank produces none of
/// these; they exist so a malformed bank fails loudly instead of quietly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The coded frame ran out of bits.
    Truncated,
    /// The frame index has no entry in the voice's context array.
    NoContext(usize),
    /// The output slice is not [`FULL_FRAME`] samples.
    BadOutputLen(usize),
}

/// The per-voice codebooks, borrowed out of a parsed bank preamble.
///
/// Field order matches the on-disk table order, which
/// `loqng_data::bank::BankTables` recovers: context array, then the three
/// narrowband LSP stages, the narrowband innovation, the two wideband LSP
/// stages, and the wideband innovation.
#[derive(Debug, Clone, Copy)]
pub struct Voice<'a> {
    /// Frame index to codebook context. One byte per frame in the bank.
    pub ctx: &'a [u8],
    /// Narrowband LSP stages, 7 / 4 / 4 bits.
    pub nb_lsp: [&'a [i8]; 3],
    /// Narrowband innovation codebook.
    pub nb_innov: &'a [i8],
    /// Wideband LSP stages, 5 / 3 bits.
    pub sb_lsp: [&'a [i8]; 2],
    /// Wideband innovation codebook.
    pub sb_innov: &'a [i8],
    /// `log2` of the narrowband innovation normalisation (header `+0x4c`).
    ///
    /// Dave stores 128, so this is 7 and codebook bytes are shifted up by
    /// `SIG_SHIFT - 7 = 7`. Stock Speex would use 5.
    pub nb_innov_log2: u8,
    /// `log2` of the wideband innovation normalisation (header `+0x50`).
    /// Dave stores 256, so this is 8.
    pub sb_innov_log2: u8,
}

impl<'a> Voice<'a> {
    /// The codebook context for a frame, which selects every per-voice table.
    pub fn context(&self, frame: usize) -> Result<u8, DecodeError> {
        self.ctx
            .get(frame)
            .copied()
            .ok_or(DecodeError::NoContext(frame))
    }

    fn nb_codebook(&self) -> VoiceCodebook<'a> {
        VoiceCodebook {
            cb: self.nb_innov,
            entry_bytes: NB_SPLIT_CB.subvect_size,
            shift: self.nb_innov_log2,
        }
    }

    fn sb_codebook(&self) -> VoiceCodebook<'a> {
        VoiceCodebook {
            cb: self.sb_innov,
            entry_bytes: SB_SPLIT_CB.subvect_size,
            shift: self.sb_innov_log2,
        }
    }
}

/// `MULT16_32_Q15`, as `fixed_arm4.h` defines it: the full 64-bit product
/// shifted right by 15, truncated to 32 bits.
pub fn mult16_32_q15(x: i16, y: i32) -> i32 {
    (((x as i64).wrapping_mul(y as i64)) >> 15) as i32
}

/// `PDIV32_16` — a **rounding C division**, not the restoring division.
///
/// `fixed_arm4.h` overrides `DIV32_16` but leaves this alone, so it compiles
/// to whatever the toolchain emits for `/` and truncates toward zero. Both
/// operands are narrowed to 16 bits first, exactly as the macro writes it.
///
/// The divisor cannot be zero on the live path — it is `(82 + rh) >> 5` with
/// `rh` seeded at `LPC_SCALING` — so the guard here is a safety net rather
/// than reproduced behaviour. C would simply trap.
pub fn pdiv32_16(a: i32, b: i32) -> i16 {
    let b16 = b as i16;
    if b16 == 0 {
        return 0;
    }
    let num = a.wrapping_add((b16 >> 1) as i32);
    num.wrapping_div(b16 as i32) as i16
}

/// The narrowband decoder: Speex `DecState`, trimmed to the reachable fields.
#[derive(Debug, Clone)]
pub struct NbDecoder {
    exc_buf: Vec<i16>,
    interp_qlpc: Vec<i16>,
    old_qlsp: Vec<i16>,
    mem_sp: Vec<i32>,
    pi_gain: Vec<i32>,
    first: bool,
}

impl Default for NbDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NbDecoder {
    pub fn new() -> Self {
        NbDecoder {
            exc_buf: vec![0i16; NB_EXC_BUF],
            interp_qlpc: vec![0i16; NB_ORDER],
            old_qlsp: vec![0i16; NB_ORDER],
            mem_sp: vec![0i32; NB_ORDER],
            pi_gain: vec![0i32; SUBFRAMES],
            first: true,
        }
    }

    /// `SPEEX_GET_EXC`: this frame's excitation, which the wideband decoder
    /// reads to size its own high-band gain.
    pub fn excitation(&self) -> &[i16] {
        &self.exc_buf[NB_EXC_OFF..NB_EXC_OFF + NB_FRAME]
    }

    /// `SPEEX_GET_PI_GAIN`: the per-subframe analysis gain at `w = pi`.
    pub fn pi_gain(&self) -> &[i32] {
        &self.pi_gain
    }

    /// Decode one narrowband frame into `out` ([`NB_FRAME`] samples).
    pub fn decode(
        &mut self,
        voice: &Voice,
        frame: usize,
        bits: &mut Bits,
        out: &mut [i16],
    ) -> Result<(), DecodeError> {
        if out.len() < NB_FRAME {
            return Err(DecodeError::BadOutputLen(out.len()));
        }
        let ctx = voice.context(frame)?;

        // Slide the excitation history down by one frame.
        self.exc_buf.copy_within(NB_FRAME..NB_FRAME + NB_HIST, 0);

        let mut qlsp = vec![0i16; NB_ORDER];
        let stages = [
            LspStage::first(voice.nb_lsp[0], 7, NB_ORDER),
            LspStage::split(voice.nb_lsp[1], 4, 5, 0),
            LspStage::split(voice.nb_lsp[2], 4, 5, 5),
        ];
        lsp::unquant(
            &mut qlsp,
            NB_ORDER,
            NB_SPACING,
            lsp::BASE_NB,
            &stages,
            ctx,
            bits,
        );

        if self.first {
            self.old_qlsp.copy_from_slice(&qlsp);
        }

        // Frame gain. The width is 5 because lpcSize is 10, not 15.
        let qe = bits.unpack(crate::nb::ol_gain_bits(NB_ORDER as u32) as i32) as usize;
        let ol_gain = mult16_32_q15(OL_GAIN_FACTOR, OL_GAIN[qe & 31]);

        let cbk = voice.nb_codebook();
        let mut innov = vec![0i32; NB_SUBFRAME];
        let mut exc32 = vec![0i32; NB_SUBFRAME];

        for sub in 0..SUBFRAMES {
            let offset = NB_SUBFRAME * sub;
            let at = NB_EXC_OFF + offset;

            for v in self.exc_buf[at..at + NB_SUBFRAME].iter_mut() {
                *v = 0;
            }

            // 3-tap adaptive codebook. The history is everything already
            // written, which is why this borrows the prefix.
            let (_pitch, _gains) = {
                let (hist, _) = self.exc_buf.split_at(at);
                ltp::unquant(
                    hist,
                    &mut exc32,
                    PITCH_START,
                    &NB_LTP,
                    &GAIN_CDBK_NB,
                    NB_SUBFRAME,
                    bits,
                    0,
                )
            };

            // Sub-frame gain correction: have_subframe_gain is 3 for submode 5.
            let q_energy = bits.unpack(3) as usize;
            let ener = crate::filters::mult16_32_q14(EXC_GAIN_SCAL3[q_energy & 7], ol_gain);

            for v in innov.iter_mut() {
                *v = 0;
            }
            crate::innov::unquant(&cbk, ctx, &NB_SPLIT_CB, bits, &mut innov);
            let scaled = innov.clone();
            signal_mul(&scaled, &mut innov, ener, NB_SUBFRAME as i32);

            for i in 0..NB_SUBFRAME {
                self.exc_buf[at + i] = fold_innovation(innov[i], exc32[i]);
            }
        }

        if bits.overflowed() {
            return Err(DecodeError::Truncated);
        }

        // The output trails the excitation by one subframe.
        let from = NB_EXC_OFF - NB_SUBFRAME;
        for i in 0..NB_FRAME {
            out[i] = self.exc_buf[from + i];
        }

        let mut interp = vec![0i16; NB_ORDER];
        let mut ak = vec![0i16; NB_ORDER];
        for sub in 0..SUBFRAMES {
            let offset = NB_SUBFRAME * sub;

            lsp::interpolate(
                &self.old_qlsp,
                &qlsp,
                &mut interp,
                NB_ORDER,
                sub as i32,
                SUBFRAMES as i32,
            );
            lsp::enforce_margin(&mut interp, crate::nb::NB_LSP_MARGIN as i16);
            lsp_to_lpc(&interp, &mut ak, NB_ORDER as i32);

            // Computed from the *previous* subframe's coefficients.
            let mut pi_g: i32 = LPC_SCALING;
            for i in (0..NB_ORDER).step_by(2) {
                pi_g = pi_g
                    .wrapping_add(self.interp_qlpc[i + 1] as i32)
                    .wrapping_sub(self.interp_qlpc[i] as i32);
            }
            self.pi_gain[sub] = pi_g;

            let src: Vec<i16> = out[offset..offset + NB_SUBFRAME].to_vec();
            iir16(
                &src,
                &self.interp_qlpc,
                NB_SUBFRAME as i32,
                NB_ORDER as i32,
                &mut out[offset..offset + NB_SUBFRAME],
                &mut self.mem_sp,
            );

            self.interp_qlpc.copy_from_slice(&ak);
        }

        // Loquendo disables the output highpass, so nothing runs here.

        self.old_qlsp.copy_from_slice(&qlsp);
        self.first = false;
        Ok(())
    }
}

/// `LPC_SCALING`, the seed for the analysis-gain accumulators.
const LPC_SCALING: i32 = 8192;

/// The wideband decoder: Speex `SBDecState` plus its narrowband half.
#[derive(Debug, Clone)]
pub struct SbDecoder {
    /// The low band, decoded first and used to drive the high band.
    pub low: NbDecoder,
    x0d: Vec<i32>,
    high: Vec<i32>,
    y0: Vec<i32>,
    y1: Vec<i32>,
    g0_mem: Vec<i32>,
    g1_mem: Vec<i32>,
    exc: Vec<i32>,
    exc_buf: Vec<i32>,
    qlsp: Vec<i16>,
    old_qlsp: Vec<i16>,
    interp_qlsp: Vec<i16>,
    interp_qlpc: Vec<i16>,
    mem_sp: Vec<i32>,
    first: bool,
}

impl Default for SbDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SbDecoder {
    pub fn new() -> Self {
        SbDecoder {
            low: NbDecoder::new(),
            x0d: vec![0i32; NB_FRAME],
            high: vec![0i32; FULL_FRAME],
            y0: vec![0i32; FULL_FRAME],
            y1: vec![0i32; FULL_FRAME],
            g0_mem: vec![0i32; QMF_ORDER],
            g1_mem: vec![0i32; QMF_ORDER],
            exc: vec![0i32; NB_FRAME],
            exc_buf: vec![0i32; NB_SUBFRAME],
            qlsp: vec![0i16; SB_ORDER],
            old_qlsp: vec![0i16; SB_ORDER],
            interp_qlsp: vec![0i16; SB_ORDER],
            interp_qlpc: vec![0i16; SB_ORDER],
            mem_sp: vec![0i32; 2 * SB_ORDER],
            first: true,
        }
    }

    /// Decode one coded frame into `out` ([`FULL_FRAME`] samples at 16 kHz).
    ///
    /// `frame` is the absolute frame index in the bank; it selects the
    /// codebook context for **both** bands.
    pub fn decode(
        &mut self,
        voice: &Voice,
        frame: usize,
        bits: &mut Bits,
        out: &mut [i16],
    ) -> Result<(), DecodeError> {
        if out.len() < FULL_FRAME {
            return Err(DecodeError::BadOutputLen(out.len()));
        }
        let ctx = voice.context(frame)?;

        let mut low = vec![0i16; NB_FRAME];
        self.low.decode(voice, frame, bits, &mut low)?;
        for i in 0..NB_FRAME {
            self.x0d[i] = (low[i] as i32) << 14;
        }

        let low_pi_gain: Vec<i32> = self.low.pi_gain().to_vec();
        let low_exc: Vec<i16> = self.low.excitation().to_vec();

        for v in self.exc.iter_mut() {
            *v = 0;
        }

        let stages = [
            LspStage::first(voice.sb_lsp[0], 5, SB_ORDER),
            LspStage::split(voice.sb_lsp[1], 3, SB_ORDER, 0),
        ];
        lsp::unquant(
            &mut self.qlsp,
            SB_ORDER,
            SB_SPACING,
            lsp::BASE_SB,
            &stages,
            ctx,
            bits,
        );

        if self.first {
            self.old_qlsp.copy_from_slice(&self.qlsp);
        }

        let cbk = voice.sb_codebook();
        let mut ak = vec![0i16; SB_ORDER];

        for sub in 0..SUBFRAMES {
            let offset = NB_SUBFRAME * sub;

            lsp::interpolate(
                &self.old_qlsp,
                &self.qlsp,
                &mut self.interp_qlsp,
                SB_ORDER,
                sub as i32,
                SUBFRAMES as i32,
            );
            lsp::enforce_margin(&mut self.interp_qlsp, crate::sb::LSP_MARGIN as i16);
            lsp_to_lpc(&self.interp_qlsp, &mut ak, SB_ORDER as i32);

            // Response ratio between the bands at 4 kHz, from the lagged
            // coefficients.
            let mut rh: i32 = LPC_SCALING;
            for i in (0..SB_ORDER).step_by(2) {
                rh = rh
                    .wrapping_add(self.interp_qlpc[i + 1] as i32)
                    .wrapping_sub(self.interp_qlpc[i] as i32);
            }
            let rl = low_pi_gain[sub];
            let filter_ratio = pdiv32_16(rl.wrapping_add(82) << 2, (82 + rh) >> 5);

            let qgc = bits.unpack(4) as usize;
            let el = compute_rms16(&low_exc[offset..], NB_SUBFRAME);
            let gc = mult16_32_q15(GC_FACTOR, GC_QUANT_BOUND[qgc & 15] as i32) as i16;

            // subframeSize is 40, so the `*= 1.4142` arm for 80 is dead.
            let scale = (pdiv32_16((gc as i32) << 8, filter_ratio as i32) as i32)
                .wrapping_mul(el.wrapping_add(1))
                << 6;

            let sub_exc = &mut self.exc[offset..offset + NB_SUBFRAME];
            for v in sub_exc.iter_mut() {
                *v = 0;
            }
            crate::innov::unquant(&cbk, ctx, &SB_SPLIT_CB, bits, sub_exc);
            let raw = sub_exc.to_vec();
            signal_mul(&raw, sub_exc, scale, NB_SUBFRAME as i32);

            // One-subframe delay, the wideband twin of the narrowband one.
            let src: Vec<i32> = self.exc_buf.clone();
            iir32(
                &src,
                &self.interp_qlpc,
                NB_SUBFRAME as i32,
                SB_ORDER as i32,
                &mut self.high[offset..offset + NB_SUBFRAME],
                &mut self.mem_sp,
            );
            self.exc_buf
                .copy_from_slice(&self.exc[offset..offset + NB_SUBFRAME]);
            self.interp_qlpc.copy_from_slice(&ak);
        }

        if bits.overflowed() {
            return Err(DecodeError::Truncated);
        }

        crate::filters::fir2x_run(
            &self.x0d,
            &QMF_H0,
            &mut self.y0,
            FULL_FRAME as i32,
            QMF_ORDER as i32,
            &mut self.g0_mem,
        );
        crate::filters::fir2x_run(
            &self.high,
            &QMF_H1,
            &mut self.y1,
            FULL_FRAME as i32,
            QMF_ORDER as i32,
            &mut self.g1_mem,
        );

        for i in 0..FULL_FRAME {
            out[i] = qmf_combine(self.y0[i], self.y1[i]);
        }

        self.old_qlsp.copy_from_slice(&self.qlsp);
        self.first = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bits the narrowband half reads from one frame.
    fn nb_frame_bits() -> usize {
        let lsp = 7 + 4 + 4;
        let gain = 5;
        let per_sub = NB_LTP.pitch_bits as usize
            + NB_LTP.gain_bits as usize
            + 3
            + NB_SPLIT_CB.nb_subvect * NB_SPLIT_CB.shape_bits as usize;
        lsp + gain + SUBFRAMES * per_sub
    }

    /// Bits the wideband half reads from one frame.
    fn sb_frame_bits() -> usize {
        let lsp = 5 + 3;
        let per_sub = 4 + SB_SPLIT_CB.nb_subvect * SB_SPLIT_CB.shape_bits as usize;
        lsp + SUBFRAMES * per_sub
    }

    #[test]
    fn the_frame_budget_is_exactly_forty_eight_bytes() {
        // Dave's coded frame. Any wrong field width breaks this before any
        // audio is produced, which makes it the cheapest check in the crate.
        assert_eq!(nb_frame_bits(), 280);
        assert_eq!(sb_frame_bits(), 104);
        let total = nb_frame_bits() + sb_frame_bits();
        assert_eq!(total, 384);
        assert_eq!(total % 8, 0);
        assert_eq!(total / 8, 48);
    }

    #[test]
    fn the_excitation_buffer_lines_up_with_the_history_shift() {
        // The shift moves NB_HIST samples down by one frame, and exc[0] must
        // land far enough in that a full subframe of lookback exists.
        assert_eq!(NB_EXC_BUF, NB_FRAME + NB_HIST);
        assert_eq!(NB_EXC_BUF, 500);
        assert_eq!(NB_EXC_OFF, 334);
        assert!(NB_EXC_OFF >= NB_SUBFRAME);
        // The deepest read is a pitch lag of PITCH_END from the last subframe.
        let deepest = NB_EXC_OFF + 3 * NB_SUBFRAME;
        assert!(deepest >= 2 * PITCH_END as usize);
    }

    #[test]
    fn the_two_ladders_span_the_same_range() {
        // 10 * 2048 == 8 * 2560 == LADDER_SPAN.
        assert_eq!(NB_ORDER * NB_SPACING as usize, 20480);
        assert_eq!(SB_ORDER * SB_SPACING as usize, 20480);
    }

    #[test]
    fn pdiv_rounds_then_truncates_toward_zero() {
        // Rounding is by adding half the divisor, which for a negative
        // numerator pulls toward zero rather than away from it.
        assert_eq!(pdiv32_16(10, 4), 3);
        assert_eq!(pdiv32_16(-10, 4), -2);
        assert_eq!(pdiv32_16(0, 7), 0);
        // Not the restoring division: div15 would give a Q0 magnitude with no
        // rounding term at all.
        assert_eq!(pdiv32_16(100, 8), 13);
    }

    #[test]
    fn pdiv_survives_a_zero_divisor_instead_of_trapping() {
        assert_eq!(pdiv32_16(1234, 0), 0);
        // A divisor whose low 16 bits are zero narrows to zero too.
        assert_eq!(pdiv32_16(1234, 0x1_0000), 0);
    }

    #[test]
    fn q15_multiply_keeps_the_full_product() {
        assert_eq!(
            mult16_32_q15(OL_GAIN_FACTOR, OL_GAIN[0]),
            (28406i64 * 18900) as i32 >> 15
        );
        // A product that overflows 32 bits must still shift from 64.
        assert_eq!(
            mult16_32_q15(32767, 0x7fff_ffff),
            ((32767i64 * 0x7fff_ffff) >> 15) as i32
        );
    }

    #[test]
    fn a_fresh_decoder_starts_silent_and_first() {
        let d = NbDecoder::new();
        assert!(d.first);
        assert!(d.excitation().iter().all(|&v| v == 0));
        assert_eq!(d.pi_gain().len(), SUBFRAMES);
        let s = SbDecoder::new();
        assert_eq!(s.g0_mem.len(), QMF_ORDER);
        assert_eq!(s.mem_sp.len(), 2 * SB_ORDER);
    }

    #[test]
    fn a_short_output_slice_is_refused() {
        let mut d = SbDecoder::new();
        let v = Voice {
            ctx: &[0],
            nb_lsp: [&[], &[], &[]],
            nb_innov: &[],
            sb_lsp: [&[], &[]],
            sb_innov: &[],
            nb_innov_log2: 7,
            sb_innov_log2: 8,
        };
        let data = [0u8; 48];
        let mut bits = Bits::new(&data);
        let mut out = [0i16; 16];
        assert_eq!(
            d.decode(&v, 0, &mut bits, &mut out),
            Err(DecodeError::BadOutputLen(16))
        );
    }

    #[test]
    fn a_frame_past_the_context_array_is_refused() {
        let mut d = SbDecoder::new();
        let v = Voice {
            ctx: &[0, 1, 2],
            nb_lsp: [&[], &[], &[]],
            nb_innov: &[],
            sb_lsp: [&[], &[]],
            sb_innov: &[],
            nb_innov_log2: 7,
            sb_innov_log2: 8,
        };
        let data = [0u8; 48];
        let mut bits = Bits::new(&data);
        let mut out = [0i16; FULL_FRAME];
        assert_eq!(
            d.decode(&v, 9, &mut bits, &mut out),
            Err(DecodeError::NoContext(9))
        );
    }
}
