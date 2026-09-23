//! A native replacement for the hottest function in the coding module.
//!
//! Profiling attributed **38.6% of all synthesis instructions** to a single
//! routine at `loqmsx.so+0x2c44` — 60.9% of the codec's own time. It is a 2x
//! polyphase interpolating FIR: for every pair of input samples it emits four
//! outputs, two per polyphase branch, carrying `ord-1` samples of history
//! across calls.
//!
//! Everything it does is integer, so a faithful port is exact rather than
//! approximate. The arithmetic below mirrors the ARM one for one: products of
//! two 16-bit taps (so never wider than 2^30), each shifted right by two
//! before accumulation, summed into a wrapping 32-bit accumulator.
//!
//! The original allocates its scratch on the stack and indexes it with a
//! 4-byte stride while storing 16-bit values, so only every other 16-bit slot
//! is live. That is reproduced here as a plain `i16` array, one entry per
//! 4-byte slot, which is the same thing without the holes.

/// Reusable buffers, so the hot path does not allocate. These routines are
/// called millions of times per utterance.
#[derive(Default)]
pub struct Scratch {
    /// Reversed, rescaled input followed by the saved history.
    pub taps: Vec<i16>,
    pub coef: Vec<i16>,
    pub out: Vec<i32>,
    pub out16: Vec<i16>,
    pub input: Vec<i32>,
    pub input16: Vec<i16>,
    pub mem: Vec<i32>,
}

/// The codec saturates to +/-32767, not to the full 16-bit range.
pub const SAT_HIGH: i32 = 32767;
pub const SAT_LOW: i32 = -32767;

/// The 32-bit variant clamps an order of magnitude wider, leaving headroom for
/// the feedback term it adds next.
const WIDE: i32 = 0x3000_0000;

#[inline(always)]
fn clamp_wide(v: i32) -> i32 {
    if v > WIDE {
        WIDE
    } else if v < -WIDE {
        -WIDE
    } else {
        v
    }
}

/// How many history entries the routine carries in `mem`.
///
/// The guest loops `for (i = 0; i < ord - 1; i += 2)`, so this is
/// `ceil((ord-1)/2)`, which for a non-negative `ord` is `ord / 2`.
#[inline]
pub fn history_len(ord: i32) -> usize {
    if ord <= 1 {
        0
    } else {
        (ord / 2) as usize
    }
}

/// Slots the guest's `alloca` provides: `((ord + n) * 2) & ~3` bytes, used
/// four bytes at a time.
#[inline]
pub fn scratch_len(n: i32, ord: i32) -> usize {
    let bytes = (n.saturating_add(ord)).saturating_mul(2) & !3;
    (bytes.max(0) / 4) as usize
}

/// Out-of-range taps read uninitialised stack in the original. Reading zero is
/// the deterministic choice; if the engine ever depended on the leftovers the
/// byte-exactness regression is what would catch it.
#[inline(always)]
fn tap(s: &[i16], idx: isize) -> i32 {
    if idx < 0 {
        return 0;
    }
    match s.get(idx as usize) {
        Some(v) => *v as i32,
        None => 0,
    }
}

#[inline(always)]
fn coef_at(c: &[i16], idx: usize) -> i32 {
    match c.get(idx) {
        Some(v) => *v as i32,
        None => 0,
    }
}

/// `s` holds the reversed input and history; `coef` the interleaved polyphase
/// taps. Writes `n` outputs, four at a time.
pub fn fir2x(s: &[i16], coef: &[i16], n: i32, ord: i32, out: &mut [i32]) {
    if n <= 0 {
        return;
    }
    let half = (n / 2) as isize;
    let ord = ord.max(0) as usize;

    let mut i = 0usize;
    let mut k = 0isize;
    while i < n as usize {
        // The window slides two input samples for every four outputs.
        let sigma = half - 2 * k;
        let mut acc0 = 0i32;
        let mut acc1 = 0i32;
        let mut acc2 = 0i32;
        let mut acc3 = 0i32;
        let mut prev = tap(s, sigma - 2);

        let mut j = 0usize;
        while j < ord {
            let sb = sigma + (j / 2) as isize;
            let a = tap(s, sb - 1);
            let b = tap(s, sb);
            let c0 = coef_at(coef, j);
            let c1 = coef_at(coef, j + 1);
            let c2 = coef_at(coef, j + 2);
            let c3 = coef_at(coef, j + 3);

            acc0 = acc0
                .wrapping_add((a.wrapping_mul(c0)) >> 2)
                .wrapping_add((b.wrapping_mul(c2)) >> 2);
            acc1 = acc1
                .wrapping_add((a.wrapping_mul(c1)) >> 2)
                .wrapping_add((b.wrapping_mul(c3)) >> 2);
            acc2 = acc2
                .wrapping_add((prev.wrapping_mul(c0)) >> 2)
                .wrapping_add((a.wrapping_mul(c2)) >> 2);
            acc3 = acc3
                .wrapping_add((prev.wrapping_mul(c1)) >> 2)
                .wrapping_add((a.wrapping_mul(c3)) >> 2);

            prev = b;
            j += 4;
        }

        // The tail is written unconditionally by the original too, so `n` is
        // expected to be a multiple of four; clip rather than panic if not.
        for (slot, v) in [acc0, acc1, acc2, acc3].iter().enumerate() {
            if let Some(o) = out.get_mut(i + slot) {
                *o = *v;
            }
        }
        i += 4;
        k += 1;
    }
}

/// `loqmsx.so+0x22c4`: an all-pole synthesis filter over 32-bit samples.
///
/// The history is halved on the way in and doubled on the way out, which is
/// how the original buys a guard bit for the accumulation in between. Each
/// tap product is formed in 64 bits and shifted right by 15.
pub fn iir32(x: &[i32], den: &[i16], n: i32, ord: i32, out: &mut [i32], mem: &mut [i32]) {
    let ord = ord.max(0) as usize;
    for m in mem.iter_mut().take(ord) {
        *m >>= 1;
    }

    for i in 0..n.max(0) as usize {
        let mut v = clamp_wide(*x.get(i).unwrap_or(&0));
        v = clamp_wide(v.wrapping_add(mem.first().copied().unwrap_or(0).wrapping_mul(4)));

        let neg = -(v as i64);
        // Writes trail the reads by one slot, so this is safe in place.
        for j in 0..ord.saturating_sub(1) {
            let p = neg.wrapping_mul(coef_at(den, j) as i64);
            mem[j] = ((p >> 15) as i32).wrapping_add(mem[j + 1]);
        }
        if ord > 0 {
            let p = neg.wrapping_mul(coef_at(den, ord - 1) as i64);
            mem[ord - 1] = (p >> 15) as i32;
        }
        if let Some(o) = out.get_mut(i) {
            *o = v;
        }
    }

    for m in mem.iter_mut().take(ord) {
        *m = m.wrapping_shl(1);
    }
}

/// `loqmsx.so+0x3cd0`: the same filter over 16-bit samples.
///
/// The feedback term arrives rounded down from the 32-bit history, the result
/// saturates to +/-32767, and the taps are driven by its negation.
pub fn iir16(x: &[i16], den: &[i16], n: i32, ord: i32, out: &mut [i16], mem: &mut [i32]) {
    let ord = ord.max(0) as usize;

    for i in 0..n.max(0) as usize {
        let fb = mem.first().copied().unwrap_or(0).wrapping_add(0x1000) >> 13;
        let v = (*x.get(i).unwrap_or(&0) as i32).wrapping_add(fb);
        let y = v.clamp(SAT_LOW, SAT_HIGH) as i16;
        let neg = -(y as i32);

        for j in 0..ord.saturating_sub(1) {
            mem[j] = coef_at(den, j).wrapping_mul(neg).wrapping_add(mem[j + 1]);
        }
        if ord > 0 {
            mem[ord - 1] = neg.wrapping_mul(coef_at(den, ord - 1));
        }
        if let Some(o) = out.get_mut(i) {
            *o = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_length_matches_the_guest_loop() {
        // for (i = 0; i < ord - 1; i += 2) -> ceil((ord-1)/2)
        for ord in 0..64i32 {
            let mut expect = 0usize;
            let mut i = 0i32;
            while i < ord - 1 {
                expect += 1;
                i += 2;
            }
            assert_eq!(history_len(ord), expect, "ord = {ord}");
        }
    }

    #[test]
    fn scratch_is_as_big_as_the_guest_alloca() {
        // ((ord + n) * 2) & ~3 bytes, four bytes per slot.
        assert_eq!(scratch_len(160, 16), 88);
        assert_eq!(scratch_len(0, 0), 0);
        assert!(scratch_len(160, 16) >= 160 / 2 + history_len(16));
    }

    #[test]
    fn an_impulse_reproduces_the_taps() {
        // With a single unit input the output is the tap set, interleaved by
        // polyphase branch and scaled by the two right shifts.
        let ord = 4;
        let n = 8;
        let mut s = vec![0i16; scratch_len(n, ord)];
        s[(n / 2 - 1) as usize] = 1 << 10;
        let coef = [100i16, 200, 300, 400];
        let mut out = vec![0i32; n as usize];
        fir2x(&s, &coef, n, ord, &mut out);
        assert_eq!(out[0], (1024 * 100) >> 2);
        assert_eq!(out[1], (1024 * 200) >> 2);
        assert!(out.iter().any(|v| *v != 0));
    }

    #[test]
    fn zero_input_gives_zero_output() {
        let s = vec![0i16; 64];
        let coef = vec![1234i16; 16];
        let mut out = vec![7i32; 32];
        fir2x(&s, &coef, 32, 16, &mut out);
        assert!(out.iter().all(|v| *v == 0));
    }

    #[test]
    fn iir16_saturates_at_the_codec_limit_not_the_type_limit() {
        // The literal pool holds 0x7fff and 0xffff8001, so the low clamp is
        // -32767; -32768 would be a different filter.
        let den = [0i16; 2];
        let x = [i16::MAX, i16::MIN];
        let mut out = vec![0i16; 2];
        // 0x10001000 >> 13 is 32768, so the first sample is pushed over.
        let mut mem = vec![0x1000_0000i32, 0];
        iir16(&x, &den, 2, 2, &mut out, &mut mem);
        assert_eq!(out[0], 32767);
        assert_eq!(out[1], -32767, "-32768 must clamp up to -32767");
    }

    #[test]
    fn iir16_feedback_wraps_the_way_the_guest_add_does() {
        // ARM `add` wraps, so a history near i32::MAX turns the feedback
        // negative rather than saturating. Faithful, not accidental.
        let den = [0i16; 1];
        let mut out = vec![0i16; 1];
        let mut mem = vec![i32::MAX];
        iir16(&[0i16], &den, 1, 1, &mut out, &mut mem);
        assert_eq!(out[0], -32767);
    }

    #[test]
    fn iir32_drains_its_history_through_the_feedback_term() {
        // Zero taps leave only the shift-down, the x4 feedback from mem[0],
        // and the shift-up, so the history walks out through the output.
        let x = vec![0i32; 8];
        let den = [0i16; 4];
        let mut out = vec![0i32; 8];
        let mut mem = vec![8i32, 16, 24, 32];
        iir32(&x, &den, 8, 4, &mut out, &mut mem);
        assert_eq!(out, vec![16, 32, 48, 64, 0, 0, 0, 0]);
        assert_eq!(mem, vec![0, 0, 0, 0]);
    }

    #[test]
    fn iir32_clamps_to_the_wide_limit() {
        let x = [i32::MAX, i32::MIN];
        let den = [0i16; 2];
        let mut out = vec![0i32; 2];
        let mut mem = vec![0i32; 2];
        iir32(&x, &den, 2, 2, &mut out, &mut mem);
        assert_eq!(out, vec![0x3000_0000, -0x3000_0000]);
    }

    #[test]
    fn nothing_is_written_for_an_empty_block() {
        let s = vec![1i16; 16];
        let coef = vec![1i16; 4];
        let mut out = vec![9i32; 4];
        fir2x(&s, &coef, 0, 4, &mut out);
        assert_eq!(out, vec![9; 4]);
    }
}
