//! `lsp_enforce_margin` — `loqmsx.so+0x69bc`. 2.5% of synthesis.
//!
//! Keeps line spectral pairs separated and inside the valid range, so the
//! filter they become stays stable. A pure leaf over one buffer, so it can be
//! proved bit-exact against the guest on its own.
//!
//! **The margin differs between the two bands**: `nb_decode` passes 16,
//! `sb_decode` passes 410. See [`crate::nb::NB_LSP_MARGIN`].
//!
//! # Shape
//!
//! ```text
//! if lsp[0]     < margin            { lsp[0]     = margin }
//! if lsp[len-1] > LSP_MAX - margin  { lsp[len-1] = LSP_MAX - margin }
//! for i in 1 .. len-1:
//!     if lsp[i] < lsp[i-1] + margin { lsp[i] = lsp[i-1] + margin }
//!     if lsp[i] > lsp[i+1] - margin { lsp[i] = (lsp[i] >> 1)
//!                                             + ((lsp[i+1] - margin) >> 1) }
//! ```
//!
//! Three details that a careless port loses:
//!
//! * **The averaging is `(a >> 1) + (b >> 1)`, not `(a + b) >> 1`.** Those
//!   differ by one whenever both operands are odd, and LSPs routinely are.
//! * **The comparisons are made in 32 bits, the stores in 16.** `lsp[i-1] +
//!   margin` is compared without wrapping but assigned with it, so a value
//!   near the top of the range can compare one way and store another.
//! * **The second test sees the value the first one just wrote.** They are two
//!   sequential clamps on the same element, not a choice between two.
//!
//! `LSP_MAX` is 25736, the module's fixed-point pi.

/// The upper end of the LSP range: fixed-point pi.
pub const LSP_MAX: i32 = 0x6488;

/// Clamp LSPs to at least `margin` apart and inside the range, in place.
///
/// A slice shorter than 3 gets only the two endpoint clamps, as the original
/// returns early.
pub fn enforce_margin(lsp: &mut [i16], margin: i16) {
    let len = lsp.len();
    if len == 0 {
        return;
    }
    let m = margin as i32;

    if (lsp[0] as i32) < m {
        lsp[0] = margin;
    }
    let top = (LSP_MAX - m) as i16;
    if lsp[len - 1] > top {
        lsp[len - 1] = top;
    }
    if len < 3 {
        return;
    }

    for i in 1..len - 1 {
        // Compared in 32 bits, stored in 16.
        let cur = lsp[i] as i32;
        let floor = lsp[i - 1] as i32 + m;
        if cur < floor {
            lsp[i] = (lsp[i - 1]).wrapping_add(margin);
        }

        // Re-read: this sees whatever the clamp above left behind.
        let cur = lsp[i];
        let ceil = lsp[i + 1] as i32 - m;
        if ceil < cur as i32 {
            // (a >> 1) + (b >> 1), NOT (a + b) >> 1.
            lsp[i] = ((ceil >> 1) as i16).wrapping_add(cur >> 1);
        }
    }
}

/// `lsp_interpolate` — `loqmsx.so+0x6a70`. 2.8% of synthesis.
///
/// Blends the previous frame's LSPs towards the current frame's, in
/// proportion to how far through the frame this subframe is:
///
/// ```text
/// t = div15((subframe + 1) * 0x4000, nb_subframes)      // Q14 weight
/// out[i] = i16((new[i] * t          + 0x2000) >> 14)
///        + i16((old[i] * (0x4000-t) + 0x2000) >> 14)
/// ```
///
/// Two details: the weight comes from the module's own
/// [`crate::math::div15`], not a C divide, and **each half is truncated to 16
/// bits before they are added**, with the sum itself wrapping as an `i16`.
/// Summing in 32 bits and narrowing once gives different answers.
pub fn interpolate(
    old: &[i16],
    new: &[i16],
    out: &mut [i16],
    len: usize,
    subframe: i32,
    nb_subframes: i32,
) {
    let t = crate::math::div15(
        (subframe.wrapping_add(1)).wrapping_mul(0x4000),
        nb_subframes,
    ) as i32;
    let inv = (0x4000i32 - t) as i16 as i32;

    let n = len.min(old.len()).min(new.len()).min(out.len());
    for i in 0..n {
        let a = ((new[i] as i32).wrapping_mul(t).wrapping_add(0x2000) >> 14) as i16;
        let b = ((old[i] as i32).wrapping_mul(inv).wrapping_add(0x2000) >> 14) as i16;
        out[i] = a.wrapping_add(b);
    }
}

/// `lsp_unquant` — **the third Loquendo customisation**, reached only through
/// the submode struct at `+0x14`.
///
/// # There are four of them, not one
///
/// The module ships four variants, differing only in how many codebook stages
/// they apply and which decoder-state fields they read:
///
/// | entry | stages | band | used by |
/// |---|---:|---|---|
/// | `0xbda0` | 1 | narrowband | nb submodes 0–3 |
/// | `0xbe50` | 2 | narrowband | nb submode 4 |
/// | `0xbf64` | 3 | narrowband | **nb submode 5 — what Dave uses** |
/// | `0xc0e0` | 2 | wideband | all three wb submodes, incl. **wb submode 2** |
///
/// Counting `bl 0x14ac` in each is what separates them: one `unpack` per
/// stage. Dave forces NB submode 5 and SB submode 2, so `0xbf64` and `0xc0e0`
/// are the two that actually decode this voice.
///
/// This matches `loqrs/sapi/loq_lsp.c`, an independently written C shim that
/// is already proven bit-exact against the real decoder: its
/// `loq_nb_lsp_unquant` has three stages and `loq_sb_lsp_unquant` has two.
///
/// # The algorithm, uniform across all four
///
/// ```text
/// for i in 0..order: lsp[i] = i * spacing + base
/// sel = *(u8 *)(state[+0x00] + cdbk_offset)
/// for each stage:
///     cb  = stage.base + ((stage.count * sel) << stage.nbits)
///     id  = unpack(bits, stage.nbits)
///     for i in 0..stage.count:
///         lsp[stage.start + i] += (i8) cb[id * stage.count + i] << stage.shift
/// ```
///
/// Stage 0 covers all `order` coefficients and shifts by **5**; every later
/// stage refines a split and shifts by **4**. The narrowband third stage
/// starts at `split_lo` rather than 0.
///
/// Signature (5 arguments; caller arguments start at `sp+0x14` for `0xbda0`'s
/// 5-register push, `sp+0x24` for the 9-register ones):
///
/// ```text
/// r0 lsp, r1 order, r2 bits, r3 state, sp+N cdbk_offset
/// ```
///
/// # State fields, per band
///
/// The two bands use entirely separate field sets, and the narrowband set is
/// a strict prefix pattern — a 1-stage variant reads only the first column.
///
/// ```text
///            stage nbits          stage count         codebook base
/// narrow     +0x08 +0x0c +0x10    order, +0x30, +0x34   +0x54 +0x58 +0x5c
/// wide       +0x1c +0x20          order, +0x44          +0x68 +0x6c
/// ```
///
/// Three things worth not re-deriving:
///
/// * **`spacing` is not a constant**, and there are *two* of them. Stock Speex
///   uses `LSP_LINEAR(i)` = `(i+1) << 11`. Here the multiplier is a `u16` read
///   from `.bss`: narrowband at `0x1d8d8`, wideband at `0x1d8da`, reached
///   GOT-relative as `[sl, #0x134]` and `[sl, #0x136]`. Both are set at run
///   time, so a port must read them rather than bake in 2048. That is why
///   `.bss` is 8 bytes. See [`SPACING_ADDR_NB`] and [`SPACING_ADDR_SB`].
/// * **The ladder's additive base differs per band too**: `0x800` narrowband
///   (`add r1, r6, #0x800`), `0x1800` wideband (`add r1, r2, #0x1800`). Using
///   the narrowband one for both puts every wideband coefficient 4096 low —
///   which is exactly how this was caught. See [`BASE_NB`] and [`BASE_SB`],
///   and `loq_lsp.c`'s separate `NB_OFF` / `SB_OFF`.
/// * **The codebook selector is the same mechanism as the innovation one**,
///   through the pointer at `state[+0x00]`, but the stride is
///   `count << nbits` rather than `entry_bytes << shape_bits`.
/// * **Both shifts are ARM register shifts**, so an `nbits` of 32 or more
///   gives zero. [`crate::innov::arm_lsl`] reproduces that.
/// **This is the `.bss` address of the spacing word, not the spacing.** The
/// values Dave actually uses are [`crate::decoder::NB_SPACING`] (2048) and
/// [`crate::decoder::SB_SPACING`] (2560), recovered from the bit-exact shim.
pub const SPACING_ADDR_NB: u32 = 0x1d8d8;

/// The wideband linear-spacing global, two bytes above [`SPACING_ADDR_NB`].
/// Also an address; see the note there.
pub const SPACING_ADDR_SB: u32 = 0x1d8da;

/// Additive base of the narrowband LSP ladder.
pub const BASE_NB: i32 = 0x800;

/// Additive base of the wideband LSP ladder. **Not the same as [`BASE_NB`].**
pub const BASE_SB: i32 = 0x1800;

/// One refinement stage of the LSP quantiser.
#[derive(Debug, Clone, Copy)]
pub struct LspStage<'a> {
    /// The whole codebook array for this stage, before the per-voice offset.
    pub cb: &'a [i8],
    /// Index width in bits.
    pub nbits: u32,
    /// How many coefficients this stage refines.
    pub count: usize,
    /// Index of the first coefficient it refines.
    pub start: usize,
    /// Left shift applied to each codebook byte: 5 for stage 0, else 4.
    pub shift: u32,
}

impl<'a> LspStage<'a> {
    /// Stage 0: the full-order stage, shifted by 5.
    pub fn first(cb: &'a [i8], nbits: u32, order: usize) -> Self {
        LspStage {
            cb,
            nbits,
            count: order,
            start: 0,
            shift: 5,
        }
    }

    /// A refinement stage, shifted by 4.
    pub fn split(cb: &'a [i8], nbits: u32, count: usize, start: usize) -> Self {
        LspStage {
            cb,
            nbits,
            count,
            start,
            shift: 4,
        }
    }
}

/// Decode one frame's LSPs into `lsp`.
///
/// Pass one, two or three stages to reproduce whichever variant applies; see
/// the table above.
pub fn unquant(
    lsp: &mut [i16],
    order: usize,
    spacing: u16,
    base: i32,
    stages: &[LspStage],
    selector: u8,
    bits: &mut crate::bits::Bits,
) {
    for i in 0..order.min(lsp.len()) {
        let v = (i as i32).wrapping_mul(spacing as i32).wrapping_add(base);
        lsp[i] = v as i16;
    }

    for st in stages {
        let off = crate::innov::arm_lsl(st.count.wrapping_mul(selector as usize) as i32, st.nbits)
            as u32 as usize;
        let cb = st.cb.get(off..).unwrap_or(&[]);
        let id = bits.unpack(st.nbits as i32) as usize;

        for i in 0..st.count {
            let at = id.wrapping_mul(st.count).wrapping_add(i);
            let c = cb.get(at).copied().unwrap_or(0) as i32;
            if let Some(slot) = lsp.get_mut(st.start.wrapping_add(i)) {
                *slot = slot.wrapping_add(crate::innov::arm_lsl(c, st.shift) as i16);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lsp_base_is_linear_in_the_spacing_global() {
        // With spacing 2048 the base is stock Speex's LSP_LINEAR: (i+1) << 11.
        let mut lsp = [0i16; 4];
        let buf = [0u8; 4];
        let mut b = crate::bits::Bits::new(&buf);
        unquant(&mut lsp, 4, 2048, BASE_NB, &[], 0, &mut b);
        for (i, v) in lsp.iter().enumerate() {
            assert_eq!(*v as i32, (i as i32 + 1) << 11);
        }
    }

    /// Stage 0 shifts by 5 and covers every coefficient; a split stage shifts
    /// by 4 and only touches its own window. Getting the two shifts the same
    /// way round is the easiest thing to lose in a port.
    #[test]
    fn a_split_stage_refines_only_its_window() {
        let cb0: Vec<i8> = vec![1; 64];
        let cb1: Vec<i8> = vec![1; 64];
        let mut lsp = [0i16; 6];
        let buf = [0u8; 8];
        let mut b = crate::bits::Bits::new(&buf);
        unquant(
            &mut lsp,
            6,
            0,
            BASE_NB,
            &[LspStage::first(&cb0, 3, 6), LspStage::split(&cb1, 3, 2, 4)],
            0,
            &mut b,
        );
        // Stage 0 adds 1 << 5 everywhere; the split adds 1 << 4 at 4 and 5.
        assert_eq!(lsp[0], 0x800 + 32);
        assert_eq!(lsp[3], 0x800 + 32);
        assert_eq!(lsp[4], 0x800 + 32 + 16);
        assert_eq!(lsp[5], 0x800 + 32 + 16);
    }

    #[test]
    fn endpoints_are_clamped_into_range() {
        let mut lsp = [0i16, 5000, 10000, 30000];
        enforce_margin(&mut lsp, 16);
        assert_eq!(lsp[0], 16);
        assert_eq!(lsp[3], (LSP_MAX - 16) as i16);
    }

    /// The two clamps are sequential and the second can pull an element back
    /// below what the first set, so "every gap is at least `margin`" is NOT an
    /// invariant of this routine. What is stable is that crowded values move
    /// apart from where they started; exact outputs are settled by
    /// `xtask verify` against the guest.
    #[test]
    fn crowded_pairs_move_apart() {
        let mut lsp = [100i16, 101, 102, 20000];
        let before = lsp;
        enforce_margin(&mut lsp, 410);
        assert_ne!(lsp, before);
        assert!(lsp[1] > before[1], "{lsp:?}");
        assert!(lsp[2] > before[2], "{lsp:?}");
    }

    #[test]
    fn averaging_halves_each_operand_separately() {
        // Both odd: (a>>1)+(b>>1) is one less than (a+b)>>1.
        let a = 101i32;
        let b = 203i32;
        assert_eq!((a >> 1) + (b >> 1), 151);
        assert_eq!((a + b) >> 1, 152);
        // So a port using the wrong form is off by one here.
        assert_ne!((a >> 1) + (b >> 1), (a + b) >> 1);
    }

    #[test]
    fn a_short_buffer_gets_only_the_endpoint_clamps() {
        let mut two = [0i16, 30000];
        enforce_margin(&mut two, 16);
        assert_eq!(two, [16, (LSP_MAX - 16) as i16]);

        let mut one = [0i16];
        enforce_margin(&mut one, 16);
        // The single element is both first and last; the last clamp wins only
        // if it is above the ceiling, which 16 is not.
        assert_eq!(one, [16]);

        let mut none: [i16; 0] = [];
        enforce_margin(&mut none, 16);
    }

    #[test]
    fn a_zero_margin_leaves_an_ordered_set_alone() {
        let mut lsp = [1000i16, 2000, 3000, 4000];
        let before = lsp;
        enforce_margin(&mut lsp, 0);
        assert_eq!(lsp, before);
    }
}
