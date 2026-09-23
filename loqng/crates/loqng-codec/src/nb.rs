//! `nb_decode` — the narrowband CELP decoder, `loqmsx.so+0x8690`.
//! **10.2% of synthesis**, 4,864 bytes.
//!
//! This was the last unidentified hot function, and the one `PORTING.md`
//! mislabelled as `0x864c` at ~5.5%. `0x864c` is a 68-byte destructor that
//! takes no samples; the samples belong here, in its neighbour.
//!
//! `sb_decode` reaches it through a 20-byte dispatcher at `0x107b8`
//! (`mode->dec`), which is why that address looked like the decoder from the
//! call site but never shows up in a profile.
//!
//! # Shape
//!
//! ```text
//! memmove(exc_buf, exc_buf + frame_size, history)      // slide the history
//! submode->lsp_unquant(lsp, lpc_size, bits, ...)       // vtable +0x14
//! if lost_frame: attenuate exc by a factor from |lsp - old_lsp|
//! if first_frame or lost: old_lsp = lsp
//!
//! if submode->lbr_pitch != -1:
//!     pitch = unpack(bits, 7) + min_pitch               // open-loop pitch
//! if submode->forced_pitch_gain:
//!     pitch_coef = gain_table[unpack(bits, 4)]
//!
//! ol_gain = gain_table[unpack(bits, lpc_size == 15 ? 4 : 5)]
//!
//! for each subframe:
//!     submode->ltp_unquant(exc, exc2, start, end, pitch_coef, par, nsf,
//!                          &pitch, &gains, bits, stack, count_lost,
//!                          offset, last_gain, cdbk_offset)   // vtable +0x1c
//!     gain = ol_gain scaled by an optional per-subframe index
//!     submode->innovation_unquant(..., 0)                    // vtable +0x28
//!     signal_mul(innov, innov, gain, nsf)
//!     exc2[i] = (innov[i] + exc[i] * 2 + 0x2000) >> 14
//!     if submode->double_codebook:
//!         innovation_unquant(..., 1)                         // second pass
//!         signal_mul(innov2, innov2, gain * K, nsf)
//!         exc2[i] += (innov2[i] + 0x2000) >> 14
//!
//! for each subframe:                                          // synthesis
//!     lsp_interpolate(old_lsp, lsp, interp, lpc_size, sub, nb_subframes)
//!     lsp_enforce_margin(interp, lpc_size, 16)
//!     lsp_to_lpc(interp, ak, lpc_size, stack)
//!     iir16(out, ak, out, nsf, lpc_size, mem, stack)
//! ```
//!
//! # Four things worth not re-deriving
//!
//! **1. The LSP margin differs between the bands.** `nb_decode` calls
//! `lsp_enforce_margin` with **16**; `sb_decode` calls it with **410**. Using
//! one value for both is a silent, plausible-sounding bug.
//!
//! **2. The gain index width depends on the LPC order.** `lpc_size == 15`
//! reads 4 bits, anything else reads 5. Get it wrong and every subsequent bit
//! in the frame is misaligned.
//!
//! **3. There is an optional second innovation pass.** When the submode's
//! `double_codebook` flag is set, `innovation_unquant` runs again with its last
//! argument `1` instead of `0`, into a separate buffer that is then scaled and
//! added. Skipping it consumes the wrong number of bits.
//!
//! **4. The pitch gain history wraps at 3.** `state[0x68]` indexes a three-slot
//! ring at `state+0x62`, reset to 0 once it exceeds 2. It feeds the next
//! frame's `last_pitch_gain`, so the ring's phase is observable in the output.
//!
//! # Dave's live path is a small fraction of `nb_decode`
//!
//! Cross-read against upstream `libspeex/nb_celp.c` (the 1.2beta1 tree is on
//! disk at `loqrs/speexb1/`). Most of the function is dead for this voice:
//!
//! | region | why it is dead |
//! |---|---|
//! | wideband-skip + submode header parse | `SPEEX_SET_SUBMODE_ENCODING` is 0, so the submode is forced, not read |
//! | `nb_decode_lost` and the null-submode arm | the bank never loses a frame |
//! | open-loop pitch read | `lbr_pitch == -1` |
//! | open-loop pitch-gain read | `forced_pitch_gain == 0` |
//! | the `submodeID == 1` vocoder arm | Dave is submode 5 |
//! | the `double_codebook` second pass | 0 for submode 5 |
//! | DTX | only reachable from submode 1 |
//! | `speex_rand` comfort noise | unreachable in the shipped module — no caller and no pointer in any section |
//!
//! What is left, per frame:
//!
//! ```text
//! shift the excitation history down by frameSize
//! lsp_unquant(qlsp, lpcSize, bits, params, frame_index)   // 3 stages for Dave
//! qe = unpack(bits, 5); ol_gain = f(qe)
//! for sub in 0..nbSubframes:
//!     zero exc[sub]
//!     ltp_unquant(...)            -> pitch, 3 gains, exc32
//!     innovation_unquant(...)     -> innov
//!     signal_mul(innov, innov, ener)
//!     exc[i] = (innov[i] + 2 * exc32[i] + 0x2000) >> 14     // SIG_SHIFT 14
//!     interpolate LSPs, lsp_to_lpc, iir_mem16 synthesis
//! ```
//!
//! [`fold_innovation`] is that combination, and it agrees with upstream
//! independently of how it was originally derived.
//!
//! ## A trap in reading the upstream source
//!
//! `nb_celp.c` carries fixed-point and floating-point arms of nearly every
//! expression behind `#ifdef FIXED_POINT`. Reading it with the preprocessor
//! lines stripped **silently merges both arms**, so two contradictory versions
//! of the same statement appear one after another — two consecutive
//! `signal_mul` calls, for instance, only one of which is compiled. Always
//! read this file with its `#if` context intact. The shipped module is the
//! fixed-point build.
//!
//! # Packet-loss concealment
//!
//! When the previous frame was lost, the excitation is scaled by a factor
//! derived from the summed absolute difference between the new and old LSPs —
//! a larger spectral jump attenuates harder. The bank never loses frames, so
//! this path should be unreachable for `loqng`, but it writes to `state[0x48]`
//! and so must not be entered accidentally.

/// `lsp_enforce_margin`'s margin in the **narrowband** decoder.
///
/// Deliberately named apart from the wideband one, which is 410.
pub const NB_LSP_MARGIN: i32 = 16;

/// The wideband margin, for contrast. See [`crate::sb::LSP_MARGIN`].
pub const SB_LSP_MARGIN: i32 = 410;

/// Bits in the open-loop pitch field.
pub const PITCH_BITS: u32 = 7;

/// Bits in the forced pitch-gain index, when the submode uses one.
pub const PITCH_GAIN_BITS: u32 = 4;

/// Slots in the pitch gain ring at `state + 0x62`.
pub const GAIN_HISTORY: usize = 3;

/// Width of the frame's gain index, which depends on the LPC order.
pub fn ol_gain_bits(lpc_size: u32) -> u32 {
    if lpc_size == 15 {
        4
    } else {
        5
    }
}

/// Fold the innovation into the excitation: `(innov + exc * 2 + 0x2000) >> 14`.
///
/// Wrapping, matching the ARM adds, and truncating to 16 bits as the store
/// does.
pub fn fold_innovation(innov: i32, exc: i32) -> i16 {
    let v = innov.wrapping_add(exc.wrapping_mul(2)).wrapping_add(0x2000) >> 14;
    v as i16
}

/// Add the second codebook pass: `exc2 += (innov2 + 0x2000) >> 14`.
pub fn add_second_pass(exc2: i16, innov2: i32) -> i16 {
    (exc2 as i32).wrapping_add(innov2.wrapping_add(0x2000) >> 14) as i16
}

/// Advance the three-slot pitch gain ring.
pub fn next_gain_slot(slot: usize) -> usize {
    let n = slot + 1;
    if n > GAIN_HISTORY - 1 {
        0
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_bands_use_different_lsp_margins() {
        // The whole point: these are not interchangeable.
        assert_eq!(NB_LSP_MARGIN, 16);
        assert_eq!(SB_LSP_MARGIN, 410);
        assert_ne!(NB_LSP_MARGIN, SB_LSP_MARGIN);
        assert_eq!(SB_LSP_MARGIN, crate::sb::LSP_MARGIN);
    }

    #[test]
    fn gain_width_hinges_on_an_lpc_order_of_fifteen() {
        assert_eq!(ol_gain_bits(15), 4);
        assert_eq!(ol_gain_bits(10), 5);
        assert_eq!(ol_gain_bits(8), 5);
        assert_eq!(ol_gain_bits(16), 5);
    }

    #[test]
    fn folding_rounds_before_shifting() {
        // +0x2000 is half of 1 << 14.
        assert_eq!(fold_innovation(0, 0), 0);
        assert_eq!(fold_innovation(0x2000, 0), 1);
        assert_eq!(fold_innovation(0x1fff, 0), 0);
        // The excitation is DOUBLED before folding, so the threshold on `exc`
        // is half what it is on `innov`: 0x1000, not 0x2000.
        assert_eq!(fold_innovation(0, 0x0fff), 0);
        assert_eq!(fold_innovation(0, 0x1000), 1);
        assert_eq!(fold_innovation(0, 0x3000), 2);
    }

    #[test]
    fn the_second_pass_accumulates_rather_than_replacing() {
        let first = fold_innovation(0x8000, 0);
        assert_eq!(first, 2);
        assert_eq!(add_second_pass(first, 0x8000), 4);
        // A zero second pass still rounds up by half.
        assert_eq!(add_second_pass(first, 0), 2);
    }

    #[test]
    fn the_gain_ring_wraps_at_three() {
        assert_eq!(next_gain_slot(0), 1);
        assert_eq!(next_gain_slot(1), 2);
        assert_eq!(next_gain_slot(2), 0);
    }
}
