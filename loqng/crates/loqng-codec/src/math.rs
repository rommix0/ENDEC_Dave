//! `spx_sqrt` — fixed-point square root, `loqmsx.so+0x7fb8`.
//!
//! A pure leaf, which makes it the right place to start porting: it can be
//! proved bit-exact against the guest on its own, with no state to set up.
//! `compute_rms16` (`0x1d90`) is built on it.
//!
//! # Shape
//!
//! Normalise into a window, evaluate a cubic in Q14, denormalise:
//!
//! ```text
//! if x <= 0: return 0
//! k = 0
//! if x > 0xffffff { x >>= 10; k  = 5 }
//! if x > 0x0fffff { x >>=  6; k += 3 }
//! if x > 0x03ffff { x >>=  4; k += 2 }
//! if x > 0x007fff { x >>=  2; k += 1 }
//! if x > 0x003fff { x >>=  2; k += 1 }
//! while x <= 0xfff { x *= 4; k -= 1 }
//! n = (i16) x
//! v = horner(n)                       // see below
//! if v >= 0x3fff { v = 0x3fff }
//! v = if k < 1 { v >> (-k & 0xff) } else { v << (k & 0xff) }
//! return (v << 9) >> 16
//! ```
//!
//! Note the five normalisation tests are **sequential, not exclusive** — each
//! `if` sees the value the previous one left. A `match` on the original
//! magnitude gives different shifts.
//!
//! # The Horner chain truncates at every step
//!
//! ```text
//! t = i16((n * 0x1077) >> 14) - 0x3153      -> i16
//! t = i16((t * n) >> 14) + 0x52b5           -> i16
//! t = i16((t * n) >> 14) + 0x0e32           -> i16
//! ```
//!
//! The decompiler writes each truncation as `* 0x10000 >> 0x10`, which is a
//! wrapping shift into the top half and an arithmetic shift back — in other
//! words, keep the low 16 bits and sign-extend. Carrying full precision
//! between the terms gives different answers, so every step narrows.
//!
//! Every multiply and shift here wraps rather than saturating, matching ARM.

/// Value the normalisation loop shifts up towards.
const NORM_FLOOR: i32 = 0xfff;

/// Ceiling applied to the polynomial before denormalising.
const SAT: i32 = 0x3fff;

/// Keep the low 16 bits and sign-extend, as `* 0x10000 >> 0x10` does.
#[inline]
fn trunc16(v: i32) -> i32 {
    v as i16 as i32
}

/// `spx_sqrt`, bit-exact.
pub fn spx_sqrt(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let mut x = x;
    let mut k: i32 = 0;

    // Sequential, each seeing the previous one's result.
    if x > 0x00ff_ffff {
        x >>= 10;
        k = 5;
    }
    if x > 0x000f_ffff {
        x >>= 6;
        k += 3;
    }
    if x > 0x0003_ffff {
        x >>= 4;
        k += 2;
    }
    if x > 0x0000_7fff {
        x >>= 2;
        k += 1;
    }
    if x > 0x0000_3fff {
        x >>= 2;
        k += 1;
    }
    while x <= NORM_FLOOR {
        x = x.wrapping_mul(4);
        k -= 1;
    }

    let n = x as i16 as i32;
    let mut v = trunc16(n.wrapping_mul(0x1077) >> 14).wrapping_sub(0x3153);
    v = trunc16(trunc16(v).wrapping_mul(n) >> 14).wrapping_add(0x52b5);
    v = trunc16(trunc16(v).wrapping_mul(n) >> 14).wrapping_add(0x0e32);
    v = trunc16(v);

    if v >= SAT {
        v = SAT;
    }
    v = if k < 1 {
        v >> ((-k) & 0xff)
    } else {
        ((v as u32) << (k & 0xff)) as i32
    };
    (((v as u32) << 9) as i32) >> 16
}

/// The module's inlined 15-bit restoring division, producing a Q14 quotient.
///
/// It appears at least three times — twice in `sb_decode` and once in
/// `lsp_interpolate` — always fully unrolled as fourteen conditional subtracts
/// followed by a fifteenth that sets the low bit without updating the
/// remainder. Sign is handled by XOR-ing the operands up front, taking
/// magnitudes, and negating the 16-bit quotient at the end.
///
/// Everything wraps: `den * 0x4000` overflows routinely and the original lets
/// it, so this does too.
pub fn div15(num: i32, den: i32) -> i16 {
    let sign_neg = (num ^ den) < 0;
    let mut n = if num < 0 { num.wrapping_neg() } else { num };
    let d = if den < 0 { den.wrapping_neg() } else { den };

    let mut q: u16 = 0;
    let mut bit: u32 = 0x4000;
    while bit >= 2 {
        let t = n.wrapping_add(d.wrapping_mul(bit as i32).wrapping_neg());
        if t >= 0 {
            q |= bit as u16;
            n = t;
        }
        bit >>= 1;
    }
    // The last step sets the bit but does not write back the remainder.
    if n.wrapping_sub(d) >= 0 {
        q |= 1;
    }
    let q = if sign_neg { q.wrapping_neg() } else { q };
    q as i16
}

/// `compute_rms16` — `loqmsx.so+0x1d90`. 4.9% of synthesis.
///
/// Root-mean-square of a 16-bit block, with the precision juggling that makes
/// it work in fixed point: find the peak, shift small blocks up so the squares
/// keep their bits, sum, divide, [`spx_sqrt`], shift back.
///
/// Three things to know:
///
/// * **The peak starts at 10, not 0.** A silent block therefore does not
///   divide by zero, and its result is whatever the floor produces.
/// * **The loops step by four with no remainder pass**, so `len` must be a
///   multiple of 4. The original reads past the end otherwise; this returns
///   early rather than reproducing that.
/// * **The two branches scale differently.** Below a peak of `0x4000` the
///   samples are shifted *up* by 0..3 and the result shifted back by
///   `3 - shift`; at or above it they are halved first and the result is
///   scaled by 16 instead.
pub fn compute_rms16(x: &[i16], len: usize) -> i32 {
    if len > x.len() || len % 4 != 0 {
        return 0;
    }

    let mut max: i32 = 10;
    for v in x.iter().take(len) {
        let a = (*v as i32).abs();
        if max < a {
            max = a;
        }
    }

    let mut sum: i32 = 0;
    if max < 0x4000 {
        let shift: u32 = if max < 0x800 {
            3
        } else if max < 0x1000 {
            2
        } else if max < 0x2000 {
            1
        } else {
            0
        };
        for c in x[..len].chunks_exact(4) {
            let mut acc: i32 = 0;
            for s in c {
                let v = ((*s as i32) << shift) as i16 as i32;
                acc = acc.wrapping_add(v.wrapping_mul(v));
            }
            sum = sum.wrapping_add(acc >> 6);
        }
        let r = spx_sqrt(sum / len as i32);
        ((r as u32) << (3 - shift)) as i32 as i16 as i32
    } else {
        for c in x[..len].chunks_exact(4) {
            let mut acc: i32 = 0;
            for s in c {
                let v = ((*s as i32) + 1) >> 1;
                acc = acc.wrapping_add(v.wrapping_mul(v));
            }
            sum = sum.wrapping_add(acc >> 6);
        }
        let r = spx_sqrt(sum / len as i32);
        (((r as u32) << 0x14) as i32) >> 0x10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_positive_input_is_zero() {
        assert_eq!(spx_sqrt(0), 0);
        assert_eq!(spx_sqrt(-1), 0);
        assert_eq!(spx_sqrt(i32::MIN), 0);
    }

    /// The result is Q-scaled, not a plain integer root, so the useful
    /// invariant is that it tracks `sqrt` up to a *constant* factor. What that
    /// factor is, and whether every bit matches, is settled by `xtask verify`
    /// against the guest — not by an expectation written here.
    #[test]
    fn it_tracks_sqrt_up_to_a_constant_factor() {
        let mut ratios = Vec::new();
        for x in [10_000i32, 1_000_000, 4_000_000, 100_000_000] {
            let got = spx_sqrt(x) as f64;
            assert!(got > 0.0, "sqrt({x}) returned {got}");
            ratios.push(got / (x as f64).sqrt());
        }
        let first = ratios[0];
        for r in &ratios {
            assert!(
                (r - first).abs() / first < 0.05,
                "scaling is not constant: {ratios:?}"
            );
        }
    }

    #[test]
    fn it_is_monotonic_over_a_wide_range() {
        let mut prev = 0;
        let mut x = 1i32;
        while x < 0x4000_0000 {
            let v = spx_sqrt(x);
            assert!(v >= prev, "dropped at {x}: {v} < {prev}");
            prev = v;
            x = x.saturating_mul(3) / 2 + 1;
        }
    }

    #[test]
    fn it_never_panics_on_extremes() {
        for x in [
            1,
            0xfff,
            0x1000,
            0x3fff,
            0x4000,
            0x7fff,
            0x8000,
            0x3ffff,
            0x40000,
            0xfffff,
            0x100000,
            0xffffff,
            0x1000000,
            i32::MAX,
        ] {
            let _ = spx_sqrt(x);
        }
    }

    #[test]
    fn the_horner_chain_narrows_at_each_step() {
        // If the truncation were dropped, a large intermediate would survive
        // and change the result. Pin that trunc16 really is a 16-bit wrap.
        assert_eq!(trunc16(0x1_0000), 0);
        assert_eq!(trunc16(0x0_8000), -32768);
        assert_eq!(trunc16(-0x1_0001), -1);
    }
}
