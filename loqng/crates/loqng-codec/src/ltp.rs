//! `pitch_unquant_3tap` — the 3-tap long-term (pitch) predictor,
//! `loqmsx.so+0x7414`. **9.1% of synthesis.**
//!
//! # It is stock, and it is not what the earlier note assumed
//!
//! `loqrs/PORTING.md` listed this as UNIDENTIFIED, reasoning that it
//! "dereferences arg 6 (`seed` in the stock signature) as a struct with a
//! pointer at +0 — matches neither stock `split_cb_shape_sign_unquant` nor
//! `loq_innov_unquant`".
//!
//! That comparison was against the wrong callback. This is the **LTP**
//! unquantiser (`submode->ltp_unquant`), not an innovation unquantiser, and in
//! its signature that argument is `par` — an `ltp_params`, whose first member
//! really is a pointer. Nothing here is customised.
//!
//! # The argument list, off the prologue
//!
//! Eight pushed registers plus `sub sp, sp, #0x10` puts the caller's stack
//! arguments at `sp+0x30` onward. Fifteen arguments with `par` sixth and
//! `cdbk_offset` last is Speex's `ltp_unquant` function-pointer type, and the
//! order matches upstream exactly. Upstream calls the first two `exc` and
//! `exc32`: the `i16` excitation and the `i32` accumulator. They are named
//! `exc2` and `exc` here, which is confusing enough that an earlier version of
//! this note claimed they were swapped relative to upstream. **They are not.**
//!
//! ```text
//! r0        exc2             excitation history, i16, indexed NEGATIVELY
//!                            (upstream's `exc`)
//! r1        exc              output, i32 per sample; memset to 0 first
//!                            (upstream's `exc32`)
//! r2        start            added to the decoded pitch
//! r3        end              (unread)
//! sp+0x30   pitch_coef       (unread)
//! sp+0x34   par              ltp_params
//! sp+0x38   nsf              samples in this subframe
//! sp+0x3c   pitch_val        int *, receives start + pitch
//! sp+0x40   gain_val         i16 *, receives the three UNSCALED taps
//! sp+0x44   bits             SpeexBits
//! sp+0x48   stack            (unread)
//! sp+0x4c   count_lost       (unread)
//! sp+0x50   subframe_offset  (unread)
//! sp+0x54   last_pitch_gain  (unread)
//! sp+0x58   cdbk_offset      selects the gain codebook
//! ```
//!
//! Ghidra recovers only ten of the fifteen and reads the last as an
//! uninitialised local, which is why it looked unfamiliar.
//!
//! # `ltp_params`
//!
//! ```text
//! +0x00  ptr  gain_cdbk     signed bytes, 4 per entry
//! +0x04  int  gain_bits
//! +0x08  int  pitch_bits
//! ```
//!
//! # What it does
//!
//! ```text
//! base     = gain_cdbk + (cdbk_offset << gain_bits) * 4
//! pitch    = unpack(bits, pitch_bits)
//! gain_idx = unpack(bits, gain_bits)
//! g[i]     = 32 + (i8) base[gain_idx * 4 + i]        for i in 0..3
//! gain_val[0..3] = g                                  // reported UNSCALED
//! *pitch_val = p = start + pitch
//! g[i]    *= 128                                      // then scaled, as i16
//! exc[0..nsf] = 0
//! for k in 0..3:
//!     lag1 = p + 1 - k
//!     lag2 = 2 * p + 1 - k
//!     n1 = min(nsf, lag1)
//!     n2 = min(nsf, lag2)
//!     for i in 0 .. n1: exc[i] += g[2-k] * exc2[i - lag1]
//!     for i in n1 .. n2: exc[i] += g[2-k] * exc2[i - lag2]
//! ```
//!
//! Five details worth not re-deriving:
//!
//! * **The codebook stride is 4, not 3.** Entries are three taps plus a fourth
//!   byte the decoder never reads. `gain_cdbk_nb` at `0x12c2e` and
//!   `gain_cdbk_lbr` at `0x12e2e` are `0x200` apart, which is 128 entries x 4.
//! * **`gain_val` receives the unscaled gains**, before the `* 128`. The
//!   caller uses them for its own bookkeeping, so scaling first would be
//!   observable.
//! * **The scaled gains are stored back as 16-bit**, so `(32 + b) * 128`
//!   wraps in an `i16` rather than widening. It does not overflow for any
//!   byte — `(32+127)*128 = 20352` — but the truncation is real code.
//! * **The two inner loops are one predictor, not two.** When the pitch lag is
//!   shorter than the subframe the predictor would reference its own output,
//!   so the tail is taken from one period further back instead. Collapsing
//!   them into a single loop changes the result whenever `p < nsf`.
//! * **The bounds are signed.** `movge` is a signed compare, so a lag that
//!   comes out negative produces no iterations at all rather than wrapping
//!   into a huge loop. The i32 arithmetic here reproduces that.

use crate::bits::Bits;

/// `ltp_params`, the struct arg 5 points at. The codebook pointer at `+0x00`
/// is passed separately here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtpParams {
    pub gain_bits: u32,
    pub pitch_bits: u32,
}

/// Bytes per gain codebook entry: three taps plus one unused.
pub const GAIN_CDBK_STRIDE: usize = 4;

/// The bias added to every stored codebook byte.
pub const GAIN_BIAS: i16 = 32;

/// The factor the gains are scaled by before prediction.
pub const GAIN_SCALE: i16 = 128;

/// Index of the first byte of a codebook entry.
pub fn cdbk_index(cdbk_offset: u32, gain_bits: u32, gain_idx: u32) -> usize {
    let base = (cdbk_offset.wrapping_shl(gain_bits) as usize).wrapping_mul(GAIN_CDBK_STRIDE);
    base.wrapping_add(gain_idx as usize * GAIN_CDBK_STRIDE)
}

/// The three taps for a codebook entry, unscaled, as `gain_val` receives them.
pub fn gains(cdbk: &[i8], at: usize) -> [i16; 3] {
    let mut g = [0i16; 3];
    for (i, slot) in g.iter_mut().enumerate() {
        *slot = GAIN_BIAS.wrapping_add(cdbk.get(at + i).copied().unwrap_or(0) as i16);
    }
    g
}

/// Run the 3-tap predictor into an existing buffer, which is zeroed first.
///
/// `hist` is the excitation history: `hist[hist.len() - n]` is `exc2[-n]`, so
/// it holds everything before the current subframe.
///
/// `pitch` is the value written to `*pitch_val`, i.e. `start + pitch` already
/// added. `gains_unscaled` are the taps as they come out of the codebook;
/// scaling by 128 happens here, exactly as the original does it.
pub fn predict_into(
    hist: &[i16],
    pitch: i32,
    gains_unscaled: [i16; 3],
    nsf: usize,
    exc: &mut [i32],
) {
    for slot in exc.iter_mut().take(nsf) {
        *slot = 0;
    }

    let g: [i16; 3] = [
        gains_unscaled[0].wrapping_mul(GAIN_SCALE),
        gains_unscaled[1].wrapping_mul(GAIN_SCALE),
        gains_unscaled[2].wrapping_mul(GAIN_SCALE),
    ];
    let nsf = nsf as i32;

    for k in 0..3i32 {
        let tap = g[(2 - k) as usize] as i32;
        let lag1 = pitch.wrapping_add(1).wrapping_sub(k);
        let lag2 = pitch.wrapping_mul(2).wrapping_add(1).wrapping_sub(k);
        let n1 = nsf.min(lag1);
        let n2 = nsf.min(lag2);

        for i in 0..n1 {
            accumulate(exc, hist, i, lag1, tap);
        }
        for i in n1..n2 {
            accumulate(exc, hist, i, lag2, tap);
        }
    }
}

/// Run the 3-tap predictor, allocating the output.
pub fn predict(hist: &[i16], pitch: i32, gains_unscaled: [i16; 3], nsf: usize) -> Vec<i32> {
    let mut exc = vec![0i32; nsf];
    predict_into(hist, pitch, gains_unscaled, nsf, &mut exc);
    exc
}

fn accumulate(exc: &mut [i32], hist: &[i16], i: i32, lag: i32, tap: i32) {
    let Some(h) = sample(hist, i, lag) else {
        return;
    };
    let Ok(idx) = usize::try_from(i) else { return };
    if let Some(slot) = exc.get_mut(idx) {
        *slot = slot.wrapping_add(tap.wrapping_mul(h as i32));
    }
}

/// `exc2[i - lag]`, where index 0 is the first sample of the current subframe
/// and history runs backwards from the end of `hist`.
///
/// `lag - i <= 0` would be the current subframe's own output rather than
/// history, so it is rejected. Both loops in [`predict_into`] bound `i < lag`,
/// so they never ask for it — but the guard belongs here, not in the caller.
fn sample(hist: &[i16], i: i32, lag: i32) -> Option<i16> {
    let back = lag.checked_sub(i)?;
    if back <= 0 {
        return None;
    }
    hist.len().checked_sub(back as usize).map(|p| hist[p])
}

/// The whole routine: read the bitstream, look the gains up, predict.
///
/// Returns `(*pitch_val, gain_val)` — the gains as the caller receives them,
/// which is **before** the `* 128`.
#[allow(clippy::too_many_arguments)]
pub fn unquant(
    hist: &[i16],
    exc: &mut [i32],
    start: i32,
    par: &LtpParams,
    cdbk: &[i8],
    nsf: usize,
    bits: &mut Bits,
    cdbk_offset: u32,
) -> (i32, [i16; 3]) {
    let pitch = bits.unpack(par.pitch_bits as i32) as i32;
    let gain_idx = bits.unpack(par.gain_bits as i32);
    let g = gains(cdbk, cdbk_index(cdbk_offset, par.gain_bits, gain_idx));
    let pitch_val = start.wrapping_add(pitch);
    predict_into(hist, pitch_val, g, nsf, exc);
    (pitch_val, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_entries_are_four_bytes_apart() {
        // gain_cdbk_nb 0x12c2e .. gain_cdbk_lbr 0x12e2e is 0x200 = 128 * 4.
        assert_eq!(0x12e2e - 0x12c2e, 128 * GAIN_CDBK_STRIDE);
        assert_eq!(cdbk_index(0, 7, 0), 0);
        assert_eq!(cdbk_index(0, 7, 1), 4);
        // cdbk_offset steps a whole codebook of (1 << gain_bits) entries.
        assert_eq!(cdbk_index(1, 7, 0), 128 * 4);
    }

    #[test]
    fn gains_are_biased_by_thirty_two() {
        let cdbk: Vec<i8> = vec![0, 1, -1, 99, -32, 127, -128, 0];
        assert_eq!(gains(&cdbk, 0), [32, 33, 31]);
        assert_eq!(gains(&cdbk, 4), [0, 159, -96]);
    }

    #[test]
    fn scaled_gains_still_fit_an_i16() {
        // (32 + 127) * 128 and (32 - 128) * 128 both fit, so the original's
        // short array is not silently overflowing.
        assert_eq!((32i16 + 127) * GAIN_SCALE, 20352);
        assert_eq!((32i16 - 128) * GAIN_SCALE, -12288);
    }

    #[test]
    fn a_long_pitch_uses_only_the_first_segment() {
        // pitch >= nsf, so lag1 > nsf and the second loop never runs.
        let hist: Vec<i16> = (0..64).collect();
        let out = predict(&hist, 40, [32, 32, 32], 16);
        assert_eq!(out.len(), 16);
        // Every tap contributes, so nothing is zero.
        assert!(out.iter().all(|v| *v != 0));
    }

    #[test]
    fn a_short_pitch_crosses_into_the_second_segment() {
        // pitch < nsf forces the tail to come from one period further back.
        let hist: Vec<i16> = (0..64).collect();
        let short = predict(&hist, 5, [32, 32, 32], 16);

        // A single-loop predictor would read exc2[i - lag1] past the subframe
        // start for i >= lag1; the real one switches to lag2. Build that wrong
        // version and check it actually differs, so the test has teeth.
        let mut wrong = vec![0i32; 16];
        for k in 0..3i32 {
            let tap = (32i16.wrapping_mul(GAIN_SCALE)) as i32;
            let lag1 = 5 + 1 - k;
            for i in 0..16i32 {
                if let Some(h) = sample(&hist, i, lag1) {
                    wrong[i as usize] = wrong[i as usize].wrapping_add(tap * h as i32);
                }
            }
        }
        assert_ne!(short, wrong);
    }

    #[test]
    fn output_is_zeroed_before_accumulating() {
        let hist: Vec<i16> = vec![0; 64];
        let mut exc = vec![99i32; 8];
        predict_into(&hist, 20, [32, 32, 32], 8, &mut exc);
        assert_eq!(exc, vec![0i32; 8]);
    }

    #[test]
    fn a_negative_lag_produces_no_iterations() {
        // Signed bounds: min(nsf, lag) is negative, so neither loop runs and
        // the output stays zero rather than wrapping into a huge loop.
        let hist: Vec<i16> = (0..64).collect();
        assert_eq!(predict(&hist, -10, [32, 32, 32], 8), vec![0i32; 8]);
    }
}
