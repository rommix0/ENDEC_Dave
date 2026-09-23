//! `sb_decode` — the wideband sub-band CELP decoder, `loqmsx.so+0xc5bc`.
//!
//! 25.0% of synthesis time and the largest single routine in the module
//! (2,852 bytes). It is Speex's `sb_decode` from `sb_celp.c`, and **every leaf
//! it calls is now ported**:
//!
//! | call | what | where |
//! |---|---|---|
//! | `0x207b8` | low-band `nb_decode` | [`crate::decoder::NbDecoder`] |
//! | `0x16a70` | `lsp_interpolate` | [`crate::lsp::interpolate`] |
//! | `0x169bc` | `lsp_enforce_margin` | [`crate::lsp::enforce_margin`] |
//! | `0x16500` | `lsp_to_lpc` | [`crate::lpc::lsp_to_lpc`] |
//! | `0x114ac` | `speex_bits_unpack_unsigned` | [`crate::bits::Bits::unpack`] |
//! | `0x13c98` | `signal_mul` | [`crate::filters::signal_mul`] |
//! | `0x122c4` | `iir_mem` 32-bit | [`crate::filters::iir32`] |
//! | `0x12c44` | QMF synthesis FIR | [`crate::filters::fir2x_run`] |
//! | `0x11d90` | RMS/energy of the excitation | [`crate::math::compute_rms16`] |
//! | submode`+0x14` | `lsp_unquant`, indirect | [`crate::lsp::unquant`] |
//! | submode`+0x28` | `innovation_unquant`, indirect | [`crate::innov::unquant`] |
//!
//! The frame loop that wires them together is [`crate::decoder::SbDecoder`].
//!
//! # Shape
//!
//! ```text
//! low = nb_decode(nb_state, bits, low_pcm)      // 0x207b8
//! for i in 0..frame_size: x0d[i] = low_pcm[i] << 14
//! decoder_ctl(nb_state, 0x67, &scratch)
//! if low != 0 : return low
//! submode = submode_from_frame_bytes(state[10])
//! if submodes[submode] == NULL : return -1
//! decoder_ctl(nb_state, 100)                    // no payload
//! decoder_ctl(nb_state, 0x65, high_pcm)
//! submodes[submode].lsp_unquant(lsp, lpc_size, bits, ...)
//! for each subframe:
//!     lsp_interpolate(prev_lsp, lsp, interp, lpc_size, sub, nb_subframes)
//!     lsp_enforce_margin(interp, lpc_size, 410)
//!     lsp_to_lpc(interp, ak, lpc_size, ...)
//!     ... gain, innovation_unquant, signal_mul, iir ...
//! fir2x(x0d, tbl_lo, out0, frame_size, 64, mem0)   // QMF synthesis
//! fir2x(x1d, tbl_hi, out1, frame_size, 64, mem1)
//! for i: out[i] = clamp((out0[i] - out1[i] + 0x1000) >> 13, -32767, 32767)
//! ```
//!
//! # Constants, from the literal pool at `0xd0b4`
//!
//! ```text
//! 0x108     GOT slot holding the mode pointer the submode switch compares
//! 410       lsp_enforce_margin's LSP_MARGIN -- this is what identifies 0x169bc
//! 32767     output saturation, positive
//! -32767    output saturation, negative -- NOT -32768
//! 28626     0x6fd2, a Q15 gain factor (0.8736)
//! 0x66666667  the division magic, used as described below
//! ```
//!
//! Three more are GOT-relative table pointers: the gain table indexed by the
//! 5-bit unpack, and the two QMF coefficient tables handed to `fir2x`.
//!
//! **The saturation is asymmetric**: `+32767` and `-32767`, not `-32768`. That
//! is the same quirk `loqrs` documented in `iir16`, so it is a property of this
//! module rather than a one-off, and a port that uses `i16::MIN` will differ on
//! the loudest samples.
//!
//! # The division trap
//!
//! Somewhere in this neighbourhood the decompiler shows:
//!
//! ```text
//! ((int)(0x66666667 * x >> 0x21) - (x >> 0x1f)) >> 5
//! ```
//!
//! `(x * 0x66666667) >> 33` is `x / 5`, and the `- (x >> 31)` makes that
//! truncate toward zero. The `>> 5` that follows is an **arithmetic shift**,
//! which floors. So the whole expression is `trunc(x / 5) >> 5`, and that is
//! **not** `x / 160` for negative `x`. Reproduce the expression, not the
//! intent. [`div_trunc5_shr5`] does.
//!
//! **Corrected: this is not the high-band gain scaling.** An earlier version
//! of this note attributed the expression to the gain path. Upstream
//! `sb_celp.c` scales the high band with `PDIV32_16`, whose divisor is
//! `filter_ratio` — a *runtime value*, which no compiler can turn into a magic
//! multiply. There is no division by 5 or by 160 anywhere in `sb_decode`'s
//! arithmetic.
//!
//! What `0x66666667` with a `>> 5` tail actually is: a compiled `x / 160` on a
//! target with no divide instruction. 160 is `frameSize`, so this belongs to
//! **frame-index arithmetic** — `position / frameSize` in the streaming pump —
//! and not to the codec at all. The helper below stays because it is correct
//! and tested, but nothing in this module should call it.
//!
//! # Submode selection
//!
//! `state[10]` is compared against a set of byte counts to pick one of five
//! submodes. **48 — this bank's coded frame size — selects submode 3.**
//!
//! The values fall into pairs plus singletons:
//!
//! ```text
//! submode 1:  0x1d (29)  0x13 (19)
//! submode 2:  0x27 (39)  0x18 (24)
//! submode 3:  0x38 (56)  0x30 (48)   <- Dave
//! submode 4:  0x52 (82)  0x42 (66)
//! submode 5:  0x66 (102)
//! ```
//!
//! A second group — `0x21 0x17 0x2b 0x1c`, `0x3c 0x34`, `0x56 0x46`, `0x6a` —
//! is **conditional on the low-band mode pointer** at `nb_state->mode + 0x28`
//! matching the GOT slot at offset `0x108`. When it matches, those pick
//! submodes 2/3/4/5 respectively; when it does not, the decoder switches to a
//! different nb state (`nb_state[1]`) and takes submode 1 or 2. Anything not in
//! either group leaves the previous submode in place rather than erroring.
//!
//! That conditional branch is **not** fully pinned down yet — the decompiler's
//! control flow through it is tangled and it does not matter for a 48-byte
//! bank, which lands in the unconditional group. It is flagged here so nobody
//! assumes the simple table is the whole story.

/// Output saturation limits. Deliberately asymmetric, as the module is.
pub const OUT_MAX: i32 = 32767;
pub const OUT_MIN: i32 = -32767;

/// `lsp_enforce_margin`'s margin, which is what identifies that call.
pub const LSP_MARGIN: i32 = 410;

/// The Q15 factor at `0xd0d8`.
pub const HIGH_GAIN_Q15: i32 = 28626;

/// Coded frame sizes that select each submode, unconditionally.
///
/// Indexed by `submode - 1`, in **this module's own numbering** — see
/// [`submode_from_frame_bytes`], which has the whole story.
pub const SUBMODE_FRAME_BYTES: [&[u8]; 5] = [
    &[0x1d, 0x13],
    &[0x27, 0x18],
    &[0x38, 0x30],
    &[0x52, 0x42],
    &[0x66],
];

/// Pick a submode from the coded frame size, for the unconditional group.
///
/// Returns `None` for a size that is only reachable through the mode-pointer
/// test, or not a known size at all — in which case the original leaves the
/// previous submode in place rather than failing.
///
/// # There are TWO submode numberings and they differ by one
///
/// This cost a round trip of wrong "corrections", so it is worth stating
/// plainly. `xtask refpcm` reads the live values out of the decoder states:
///
/// ```text
/// this module's arrays        stock Speex modes.c        parameters
/// nb submodeID 6        ==    nb_submode5          split_cb {5, 8, 6}, 280 bits
/// sb submodeID 3        ==    wb_submode2          split_cb {10, 4, 5}
/// ```
///
/// So **48 coded bytes selects submode 3 here**, which is what the original
/// reverse engineering said, and `submode - 1` indexing is correct.
///
/// The trap is `loqmsx_dll.c::make_decoder`, which sets
/// `SPEEX_SET_LOW_MODE = 5` and `SPEEX_SET_HIGH_MODE = 2`. Those are right —
/// for the shim, which is built on *stock* `modes.c`. Applying them to this
/// module selects the wrong submode: forcing nb 5 lands on a submode with
/// `split_cb {8, 5, 7}` and 216 bits a frame (stock `nb_submode4`), and the
/// bitstream then misaligns by 64 bits a frame. The shim's own comment,
/// `/* SB submode 2 (Dave SB3) */`, is the tell that two numberings are in
/// play.
///
/// **Never force the submode when driving this module.** The coded frame size
/// is a constructor argument to `speex_decoder_init` precisely so the module
/// can select the pair itself, and it gets it right.
///
/// **Entry 0 is not verified.** A 19- or 29-byte frame maps to submode 1 here,
/// and no bank on hand uses those sizes, so nothing has exercised it.
pub fn submode_from_frame_bytes(bytes: u8) -> Option<u8> {
    SUBMODE_FRAME_BYTES
        .iter()
        .position(|set| set.contains(&bytes))
        .map(|i| i as u8 + 1)
}

/// `trunc(x / 5) >> 5`, exactly as the module computes it.
///
/// Written out rather than as `x / 160`, because the truncating divide and the
/// flooring shift compose differently from a single division for negative
/// inputs.
pub fn div_trunc5_shr5(x: i32) -> i32 {
    let q5 = ((x as i64 * 0x6666_6667i64) >> 33) as i32 - (x >> 31);
    q5 >> 5
}

/// Combine the two QMF band outputs into the final PCM sample.
///
/// `(lo - hi + 0x1000) >> 13`, saturated to the module's asymmetric limits.
pub fn qmf_combine(lo: i32, hi: i32) -> i16 {
    let v = (lo.wrapping_sub(hi).wrapping_add(0x1000)) >> 13;
    v.clamp(OUT_MIN, OUT_MAX) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dave_selects_submode_three_in_this_modules_numbering() {
        // Confirmed by reading SBDecState+0x84 live: `xtask refpcm` prints
        // "sb submodes[] at 0x0027d44c, submodeID 3". The equivalent stock
        // submode is wb_submode2 -- a different number for the same thing.
        assert_eq!(submode_from_frame_bytes(48), Some(3));
        assert_eq!(submode_from_frame_bytes(0x30), Some(3));
    }

    #[test]
    fn every_unconditional_size_maps_to_its_submode() {
        for (i, set) in SUBMODE_FRAME_BYTES.iter().enumerate() {
            for b in set.iter() {
                assert_eq!(submode_from_frame_bytes(*b), Some(i as u8 + 1));
            }
        }
        // A size only reachable through the mode-pointer test.
        assert_eq!(submode_from_frame_bytes(0x21), None);
        assert_eq!(submode_from_frame_bytes(0), None);
    }

    #[test]
    fn no_submode_exceeds_the_three_bit_field() {
        // SB_SUBMODE_BITS is 3, so an eight-slot array. Anything outside
        // that would index past the array the original actually has.
        for set in SUBMODE_FRAME_BYTES.iter() {
            for b in set.iter() {
                let m = submode_from_frame_bytes(*b).unwrap();
                assert!(m < 8, "submode {m} does not fit SB_SUBMODE_BITS");
            }
        }
    }

    #[test]
    fn saturation_is_asymmetric() {
        // A port using i16::MIN would differ here, on the loudest samples.
        assert_eq!(qmf_combine(i32::MIN / 4, 0), OUT_MIN as i16);
        assert_eq!(qmf_combine(i32::MAX / 4, 0), OUT_MAX as i16);
        assert_ne!(OUT_MIN, i16::MIN as i32);
    }

    #[test]
    fn qmf_rounds_before_shifting() {
        // The +0x1000 is half of 1 << 13, so this is round-half-up.
        assert_eq!(qmf_combine(0, 0), 0);
        assert_eq!(qmf_combine(0x1000, 0), 1);
        assert_eq!(qmf_combine(0x0fff, 0), 0);
        // The high band subtracts.
        assert_eq!(qmf_combine(0x4000, 0x2000), 1);
    }

    #[test]
    fn the_division_matches_a_reference_over_its_range() {
        // Reference: truncating /5 then an arithmetic >>5, composed as written.
        for x in [
            -100_000i32,
            -1441,
            -161,
            -160,
            -5,
            -1,
            0,
            1,
            5,
            160,
            161,
            100_000,
        ] {
            let want = ((x as i64 / 5) as i32) >> 5;
            assert_eq!(div_trunc5_shr5(x), want, "x = {x}");
        }
    }

    #[test]
    fn the_division_is_not_a_plain_divide_by_160() {
        // If these were the same, writing `x / 160` would be safe. They are
        // not, which is the whole point of spelling it out.
        let differs = (-2000..0).any(|x| div_trunc5_shr5(x) != x / 160);
        assert!(differs);
    }
}
