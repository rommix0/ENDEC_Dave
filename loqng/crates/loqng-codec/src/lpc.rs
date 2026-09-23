//! `lsp_to_lpc` — `loqmsx.so+0x6500`. Turns line spectral pairs back into the
//! synthesis filter's coefficients.
//!
//! Speex's fixed-point `lsp_to_lpc` (`libspeex/lsp.c`) with [`spx_cos`] inlined
//! by the ARM compiler, which is why the shipped routine makes no calls at all.
//! Brought across from `loqrs`, where it runs as a native patch, and re-proved
//! here directly rather than through whole-utterance audio.
//!
//! # Why it needs no guest scratch
//!
//! The `xp`/`xq`/`freqn` working set is `ALLOC`'d from the caller's scratch
//! arena, and `stack` is passed **by value**, so nothing left behind in it is
//! observable. Only `ak` is output. That lets the whole cascade run in host
//! memory and touch the caller's buffers only to read `freq` and write `ak`.
//!
//! # Two things the obvious reading gets wrong
//!
//! Both were found by running this against interpreted ARM. Neither shows up
//! in whole-utterance audio, because real LSPs never reach the region where
//! they differ — which is exactly why they survived so long.
//!
//! * **The output rule is a magnitude test, not a clamp.** See
//!   [`saturate_ak`]. Coefficients between 32768 and 65535 are stored
//!   *wrapped*, not saturated.
//! * **`MULT16_16_P13` does not truncate its operands**, despite the name.
//!   See [`mult16_16_p13`].

use crate::filters::mult16_32_q14;

/// `MULT16_16_P13(a, b)` = `(4096 + a * b) >> 13`.
///
/// **The operands are NOT truncated to 16 bits.** The name says `16_16`, and
/// `fixed_generic.h` writes it as a product of two `spx_word16_t`, but the
/// shipped code has no `sxth` anywhere in the inlined `spx_cos` — every step
/// is a plain 32-bit `mul`. Truncating the operands agrees for any in-range
/// LSP (`0 ..= LSP_MAX`, where `LSP_MAX - x` also fits) and diverges outside
/// it, so the difference hides until something feeds it an angle it should
/// not have.
#[inline(always)]
pub fn mult16_16_p13(a: i32, b: i32) -> i32 {
    4096i32.wrapping_add(a.wrapping_mul(b)) >> 13
}

/// Speex's `spx_cos` (`libspeex/math_approx.c`).
///
/// A fourth-order polynomial on the first quadrant, reflected about
/// `LSP_MAX / 2`. The reflection point 12868 and the mirror constant 25736 are
/// the module's fixed-point pi and half of it.
pub fn spx_cos(x: i16) -> i16 {
    const K1: i32 = 8192;
    const K2: i32 = -4096;
    const K3: i32 = 340;
    const K4: i32 = -10;

    let poly = |x2: i32| {
        mult16_16_p13(
            x2,
            K2.wrapping_add(mult16_16_p13(x2, K3.wrapping_add(mult16_16_p13(K4, x2)))),
        )
    };

    // Nothing here is narrowed to 16 bits. The only truncation in the shipped
    // routine is the `ldrsh` that reads the finished value back off the
    // stack, which is the `as i16` on the way out.
    if (x as i32) < 12868 {
        let x2 = mult16_16_p13(x as i32, x as i32);
        K1.wrapping_add(poly(x2)) as i16
    } else {
        let xx = 25736i32 - x as i32;
        let x2 = mult16_16_p13(xx, xx);
        (-K1).wrapping_sub(poly(x2)) as i16
    }
}

/// `ANGLE2X(a)` = `SHL16(spx_cos(a), 2)`, stored back into an `i16` — so it
/// wraps rather than saturating.
#[inline]
pub fn angle2x(a: i16) -> i16 {
    ((spx_cos(a) as i32) << 2) as i16
}

/// How a coefficient reaches `ak`, which is **not** a two-sided clamp.
///
/// The shipped code tests the *magnitude*, not the value:
///
/// ```asm
/// movs ip, r2          ; a
/// rsbmi ip, ip, #0     ; ip = |a|
/// rsbmi r1, r1, #0     ; r1 = -32767 when a < 0
/// lsrs ip, ip, #0x10   ; Z set iff |a| < 65536
/// moveq r1, r2         ; ...then keep a as it is
/// strh r1, [lr, #-2]
/// ```
///
/// So anything with `|a| < 65536` is stored **truncated to 16 bits, wrapping**
/// — a coefficient of 40000 comes out as -25536 — and only `|a| >= 65536`
/// saturates, to `+/-32767` following the sign of `a`.
///
/// A two-sided clamp at `+/-32767` is the obvious reading and it is wrong:
/// it disagrees on the whole band from 32768 to 65535. Real coefficients stay
/// well inside that, which is why an implementation carrying the wrong rule
/// can still produce identical audio for hours.
#[inline]
pub fn saturate_ak(a: i32) -> i16 {
    let mag = if a < 0 { a.wrapping_neg() } else { a };
    if (mag as u32) >> 16 == 0 {
        a as i16
    } else if a < 0 {
        -32767
    } else {
        32767
    }
}

/// Expand `lpcrdr` line spectral pairs into `lpcrdr` LPC coefficients.
///
/// `freq` is the LSP set, `ak` receives the coefficients. Both are `lpcrdr`
/// long; anything shorter is left alone rather than panicking.
pub fn lsp_to_lpc(freq: &[i16], ak: &mut [i16], lpcrdr: i32) {
    /// The impulse the cascade is driven with, in Q21.
    const QIMP: i32 = 21;

    let n = lpcrdr.max(0) as usize;
    let m = (lpcrdr >> 1).max(0) as usize;
    let row = (lpcrdr + 3).max(0) as usize;
    if n == 0 || m == 0 {
        return;
    }

    let mut freqn = vec![0i16; n];
    for (i, f) in freqn.iter_mut().enumerate() {
        *f = angle2x(freq.get(i).copied().unwrap_or(0));
    }

    let mut xp = vec![0i32; (m + 1) * row];
    let mut xq = vec![0i32; (m + 1) * row];
    let xin: i32 = 1 << (QIMP - 1);

    for i in 0..=m {
        let b = i * row;
        xp[b + 1] = 0;
        xp[b + 2] = xin;
        xp[b + 2 + 2 * i] = xin;
        xq[b + 1] = 0;
        xq[b + 2] = xin;
        xq[b + 2 + 2 * i] = xin;
    }

    xp[row + 3] = mult16_32_q14(freqn[0], xp[2]).wrapping_neg();
    xq[row + 3] = mult16_32_q14(freqn[1], xq[2]).wrapping_neg();

    let (mut xout1, mut xout2) = (0i32, 0i32);

    for i in 1..m {
        let (c, nx) = (i * row, (i + 1) * row);
        let mut j = 1usize;
        while j < 2 * (i + 1) - 1 {
            let mult = mult16_32_q14(freqn[2 * i], xp[c + j + 1]);
            xp[nx + j + 2] = xp[c + j + 2].wrapping_sub(mult).wrapping_add(xp[c + j]);
            let mult = mult16_32_q14(freqn[2 * i + 1], xq[c + j + 1]);
            xq[nx + j + 2] = xq[c + j + 2].wrapping_sub(mult).wrapping_add(xq[c + j]);
            j += 1;
        }
        // Last column: xp[i][j+2] and xq[i][j+2] are known zero.
        let mult = mult16_32_q14(freqn[2 * i], xp[c + j + 1]);
        xp[nx + j + 2] = xp[c + j].wrapping_sub(mult);
        let mult = mult16_32_q14(freqn[2 * i + 1], xq[c + j + 1]);
        xq[nx + j + 2] = xq[c + j].wrapping_sub(mult);
    }

    let last = m * row;
    let shift = QIMP - 13;
    for j in 1..=n {
        let p = xp[last + j + 2];
        let q = xq[last + j + 2];
        let sum = p.wrapping_add(xout1).wrapping_add(q).wrapping_sub(xout2);
        let a = sum.wrapping_add(1 << (shift - 1)) >> shift;
        xout1 = p;
        xout2 = q;
        if let Some(o) = ak.get_mut(j - 1) {
            *o = saturate_ak(a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_is_plus_one_at_zero_and_minus_one_at_pi() {
        // Q13, so 8192 is 1.0. The approximation is exact at both ends.
        assert_eq!(spx_cos(0), 8192);
        assert_eq!(spx_cos(25736), -8192);
    }

    #[test]
    fn cos_is_near_zero_at_the_quarter_turn() {
        // 25736/2 = 12868 is the reflection point, i.e. pi/2.
        assert!(spx_cos(12868).abs() < 64, "{}", spx_cos(12868));
    }

    #[test]
    fn cos_descends_across_the_first_quadrant() {
        let mut prev = spx_cos(0);
        for a in (0..12868).step_by(97) {
            let v = spx_cos(a as i16);
            assert!(v <= prev + 1, "not monotonic at {a}: {prev} -> {v}");
            prev = v;
        }
    }

    #[test]
    fn a_short_output_is_left_alone_rather_than_panicking() {
        let freq: Vec<i16> = (0..10).map(|i| 2000 + i * 2000).collect();
        let mut ak = [0i16; 3];
        lsp_to_lpc(&freq, &mut ak, 10);
        // Only the first three coefficients could be written.
        assert!(ak.iter().any(|v| *v != 0));
    }

    /// The band from 32768 to 65535 wraps rather than saturating. A two-sided
    /// clamp would return 32767 for all of it, which is the reading this
    /// codebase carried until it was checked against the hardware.
    #[test]
    fn coefficients_below_the_magnitude_limit_wrap_instead_of_clamping() {
        assert_eq!(saturate_ak(0), 0);
        assert_eq!(saturate_ak(32767), 32767);
        assert_eq!(saturate_ak(40000), 40000u16 as i16);
        assert_eq!(saturate_ak(40000), -25536);
        assert_eq!(saturate_ak(65535), -1);
        assert_eq!(saturate_ak(-40000), (-40000i32) as i16);
    }

    #[test]
    fn only_a_magnitude_of_65536_or_more_saturates() {
        assert_eq!(saturate_ak(65536), 32767);
        assert_eq!(saturate_ak(-65536), -32767);
        assert_eq!(saturate_ak(i32::MAX), 32767);
        // wrapping_neg of i32::MIN stays negative, and still saturates low.
        assert_eq!(saturate_ak(i32::MIN), -32767);
    }

    #[test]
    fn the_p13_multiply_keeps_operands_wider_than_sixteen_bits() {
        let wide = 40000i32;
        assert_eq!(mult16_16_p13(wide, wide), (4096 + wide * wide) >> 13);
        // Truncating first would fold 40000 to -25536 and change the sign of
        // nothing here, but it does change the magnitude.
        let truncated = wide as i16 as i32;
        assert_ne!(mult16_16_p13(wide, 3), (4096 + truncated * 3) >> 13);
    }

    #[test]
    fn a_zero_order_call_writes_nothing() {
        let mut ak = [7i16; 4];
        lsp_to_lpc(&[], &mut ak, 0);
        assert_eq!(ak, [7; 4]);
    }
}
