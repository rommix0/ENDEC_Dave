//! The four signal-path filters, and the fixed-point primitive they share.
//!
//! These already existed as bit-exact Rust in `loqrs`
//! (`crates/loqhost/src/msx.rs` and `libc.rs`), where they run as native
//! patches over the interpreter. They are brought across here so `loqng` can
//! decode without the interpreter, and then re-proved on their own terms:
//! `loqrs` validates them by whole-utterance audio diff, which is a strong
//! end-to-end gate but tells you nothing about which input broke.
//! `xtask verify` calls each one directly.
//!
//! | routine | offset | what |
//! |---|---|---|
//! | [`fir2x`] | `0x2c44` | polyphase QMF synthesis, 2x upsampling |
//! | [`iir32`] | `0x22c4` | all-pole synthesis filter, 32-bit |
//! | [`iir16`] | `0x3cd0` | the same filter, 16-bit |
//! | [`signal_mul`] | `0x3c98` | scale a 32-bit signal by a Q14 gain |
//!
//! # `MULT16_32_Q14` is the ARM4 one
//!
//! The module was built with `libspeex/fixed_arm4.h`, which overrides exactly
//! three macros. [`mult16_32_q14`] is one of them: a full `smull` and a 64-bit
//! shift, **not** `fixed_generic.h`'s split into two 16x16 multiplies. The two
//! disagree whenever the 32-bit operand exceeds 16 bits, which is most of the
//! time on a signal path.
//!
//! `signal_mul` is the standing example of the disassembly overruling the C:
//! stock Speex truncates through `EXTRACT16` first, and the shipped code has
//! no `sxth` at all.

/// `MULT16_32_Q14`, as `fixed_arm4.h` defines it and the module compiles it.
#[inline(always)]
pub fn mult16_32_q14(x: i16, y: i32) -> i32 {
    (((y as i64).wrapping_mul(x as i64)) >> 14) as i32
}

/// The codec saturates to +/-32767, not to the full 16-bit range.
pub const SAT_HIGH: i32 = 32767;
/// The low clamp is -32767. `-32768` would be a different filter.
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

/// How many history entries [`iir32`] and [`iir16`] carry in `mem`.
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
/// the deterministic choice; a byte-exactness regression is what would catch
/// it if the engine ever depended on the leftovers.
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

/// `loqmsx.so+0x2c44`: polyphase QMF synthesis.
///
/// `s` holds the reversed input and history; `coef` the interleaved polyphase
/// taps. Writes `n` outputs, four at a time, sliding the window two input
/// samples per group.
pub fn fir2x(s: &[i16], coef: &[i16], n: i32, ord: i32, out: &mut [i32]) {
    if n <= 0 {
        return;
    }
    let half = (n / 2) as isize;
    let ord = ord.max(0) as usize;

    let mut i = 0usize;
    let mut k = 0isize;
    while i < n as usize {
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
/// how the original buys a guard bit for the accumulation in between. Each tap
/// product is formed in 64 bits and shifted right by 15.
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

/// [`fir2x`] with the guest's own marshalling, so it can be called and
/// verified as the whole routine rather than as a kernel.
///
/// The guest builds its tap buffer on the stack: `n/2` input samples written
/// **newest first** and rescaled by `2^14` with rounding, followed by the
/// saved history. Afterwards the newest samples become the next call's
/// history.
///
/// `mem` is the guest's history array, which has an **8-byte stride with the
/// sample at +4** — so in `i32` units the entry for `k` is `mem[2*k + 1]`.
/// The other word is not touched.
///
/// The last group of taps is read whole even when `ord` is not a multiple of
/// four, exactly as the unrolled guest loop does.
pub fn fir2x_run(x: &[i32], coef: &[i16], out: &mut [i32], n: i32, ord: i32, mem: &mut [i32]) {
    let half = if n > 0 { (n / 2) as usize } else { 0 };
    let hist = history_len(ord);
    let slots = scratch_len(n, ord).max(half + hist);

    let mut s = vec![0i16; slots];
    for k in 0..half {
        let v = x.get(half - 1 - k).copied().unwrap_or(0);
        s[k] = (v.wrapping_add(0x2000) >> 14) as i16;
    }
    for k in 0..hist {
        s[half + k] = mem.get(2 * k + 1).copied().unwrap_or(0) as u16 as i16;
    }

    let taps = (ord.max(0) as usize).div_ceil(4) * 4;
    let mut c = vec![0i16; taps];
    for (j, slot) in c.iter_mut().enumerate() {
        *slot = coef.get(j).copied().unwrap_or(0);
    }

    fir2x(&s, &c, n, ord, out);

    for k in 0..hist {
        if let Some(slot) = mem.get_mut(2 * k + 1) {
            *slot = s[k] as i32;
        }
    }
}

/// `loqmsx.so+0x3c98`: scale a 32-bit signal by a Q14 gain.
///
/// Argument order matches the guest: `(x, y, scale, len)`.
///
/// **Transcribed from the disassembly, not the C.** Stock fixed-point Speex
/// computes `SHL32(MULT16_32_Q14(EXTRACT16(SHR32(x[i],7)), scale), 7)`, where
/// `EXTRACT16` truncates to 16 bits. The shipped code has no `sxth`: it takes
/// the full 64-bit product and shifts. The hardware wins.
///
/// The original is a do-while, so `len == 0` would wrap. Every call site
/// passes a positive length, so this uses a zero-trip loop rather than
/// reproducing that.
pub fn signal_mul(x: &[i32], y: &mut [i32], scale: i32, len: i32) {
    for i in 0..len.max(0) as usize {
        let v = *x.get(i).unwrap_or(&0);
        let p = (scale as i64).wrapping_mul((v >> 7) as i64);
        let out = ((p >> 14) as i32).wrapping_shl(7);
        if let Some(o) = y.get_mut(i) {
            *o = out;
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
        assert_eq!(scratch_len(160, 16), 88);
        assert_eq!(scratch_len(0, 0), 0);
        assert!(scratch_len(160, 16) >= 160 / 2 + history_len(16));
    }

    #[test]
    fn an_impulse_reproduces_the_taps() {
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
    fn iir16_saturates_at_the_codec_limit_not_the_type_limit() {
        let den = [0i16; 2];
        let x = [i16::MAX, i16::MIN];
        let mut out = vec![0i16; 2];
        let mut mem = vec![0x1000_0000i32, 0];
        iir16(&x, &den, 2, 2, &mut out, &mut mem);
        assert_eq!(out[0], 32767);
        assert_eq!(out[1], -32767, "-32768 must clamp up to -32767");
    }

    #[test]
    fn iir32_drains_its_history_through_the_feedback_term() {
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

    /// The ARM4 macro keeps the whole 32-bit operand; the generic one would
    /// have truncated it to 16 bits first. Pin the difference so a "cleanup"
    /// back to the stock form fails loudly.
    #[test]
    fn mult16_32_q14_does_not_truncate_its_wide_operand() {
        let wide = 0x0012_3456i32;
        assert_eq!(mult16_32_q14(3, wide), ((wide as i64 * 3) >> 14) as i32);
        let truncated = wide as i16 as i32;
        assert_ne!(
            mult16_32_q14(3, wide),
            ((truncated as i64 * 3) >> 14) as i32
        );
    }

    #[test]
    fn signal_mul_keeps_the_low_seven_bits_out_of_the_product() {
        // x >> 7 then << 7: the bottom seven bits of the input never
        // contribute, so inputs differing only there give the same output.
        let a = [0x0001_2380i32];
        let b = [0x0001_23ffi32];
        let mut ya = [0i32];
        let mut yb = [0i32];
        signal_mul(&a, &mut ya, 0x4000, 1);
        signal_mul(&b, &mut yb, 0x4000, 1);
        assert_eq!(ya, yb);
    }
}
