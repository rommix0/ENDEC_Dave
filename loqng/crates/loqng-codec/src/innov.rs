//! `split_cb_shape_sign_unquant` — innovation (fixed codebook) unquantiser,
//! `loqmsx.so+0x153c`. **9.2% of synthesis**, and one of Loquendo's three
//! customisations.
//!
//! `loqrs/PORTING.md` guessed this name for `0x7414`, which turned out to be
//! the pitch predictor. The name was right; the address was not. It is here.
//!
//! # The signature is not stock
//!
//! Stock Speex's `innovation_unquant_func` is six arguments,
//! `(exc, par, nsf, bits, stack, seed)`. The shipped one takes **eight**, with
//! the decoder state first and a band selector added, read straight off the
//! prologue (8 pushed registers plus `sub sp, sp, #0xc` puts caller arguments
//! at `sp+0x2c`):
//!
//! ```text
//! r0        state            decoder state; the codebook comes out of this
//! r1        exc              output, i32 per sample
//! r2        par              split_cb_params
//! r3        band             0 = low, non-zero = high
//! sp+0x2c   (unread)
//! sp+0x30   bits             SpeexBits
//! sp+0x34   stack            scratch; ind[] then signs[], each 4-aligned
//! sp+0x38   cdbk_offset      byte index of the per-voice selector
//! ```
//!
//! # The customisation: a per-voice codebook
//!
//! Stock Speex takes the shape codebook straight out of the parameter struct —
//! `split_cb_params.shape_cb`, the third member. **This function never reads
//! that member.** It computes the codebook address instead, from the decoder
//! state, with a different set of fields per band:
//!
//! ```text
//! band 0:  cb    = state[+0x60] + state[+0x38] * ((1 << shape_bits) * sel)
//!          shift = state.byte[+0x84]
//! band 1:  cb    = state[+0x64] + state[+0x3c] * ((1 << shape_bits) * sel)
//!          shift = state.byte[+0x85]
//!
//! where    sel   = *(u8 *)(state[+0x00] + cdbk_offset)
//! ```
//!
//! So `+0x60`/`+0x64` are the bases of per-voice codebook arrays, `+0x38`/
//! `+0x3c` are the bytes per entry, and a byte fetched through the pointer at
//! `+0x00` picks which codebook. That is the "per-voice innovation codebook"
//! the module is known to add, and it is why a stock Speex decoder cannot read
//! this bank.
//!
//! The two bands use adjacent single bytes at `+0x84` and `+0x85` for their
//! scaling shift, which is also not stock.
//!
//! # `split_cb_params`
//!
//! Confirmed against the fields this function actually touches:
//!
//! ```text
//! +0x00  subvect_size
//! +0x04  nb_subvect
//! +0x08  shape_cb      <- IGNORED here; see above
//! +0x0c  shape_bits
//! +0x10  have_sign
//! ```
//!
//! # What it does
//!
//! ```text
//! for i in 0..nb_subvect:
//!     signs[i] = have_sign ? unpack(bits, 1) : 0
//!     ind[i]   = unpack(bits, shape_bits)
//! for i in 0..nb_subvect:
//!     for j in 0..subvect_size:
//!         v = (i8) cb[ind[i] * subvect_size + j] << (14 - shift)
//!         exc[i * subvect_size + j] = if signs[i] == 0 { v } else { -v }
//! ```
//!
//! Four details that a careless port loses:
//!
//! * **Two passes.** Every sign and index is read from the bitstream before
//!   any of them is expanded. Within the read loop the sign comes first, then
//!   the index; interleaving the reads with the expansion would consume the
//!   bits in the same order but is fragile to get right, and the original is
//!   unambiguous about it.
//! * **A sign bit of 1 means negative.** The original reaches that branch
//!   through a compare against a constant rather than a direct test, but the
//!   effect is this.
//! * **The scaling is an ARM register shift**, so an amount of 32 or more
//!   yields zero rather than wrapping or panicking. `14 - shift` is computed
//!   in 32 bits over a byte loaded with `ldrb`, so any `shift > 14` lands
//!   there. See [`arm_lsl`].
//! * **The codebook is indexed by `subvect_size`, not by the `entry_bytes`
//!   field** that scales the per-voice offset. The two are equal for the banks
//!   in hand, which is exactly why mixing them up would go unnoticed.

use crate::bits::Bits;

/// `split_cb_params`, as this function reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitCbParams {
    pub subvect_size: usize,
    pub nb_subvect: usize,
    pub shape_bits: u32,
    pub have_sign: bool,
}

/// Which sub-band's state fields to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    Low,
    High,
}

impl Band {
    /// State offset of this band's codebook base pointer.
    pub fn cb_base_offset(self) -> u32 {
        match self {
            Band::Low => 0x60,
            Band::High => 0x64,
        }
    }

    /// State offset of this band's bytes-per-entry word.
    pub fn entry_bytes_offset(self) -> u32 {
        match self {
            Band::Low => 0x38,
            Band::High => 0x3c,
        }
    }

    /// State offset of this band's scaling shift byte.
    pub fn shift_offset(self) -> u32 {
        match self {
            Band::Low => 0x84,
            Band::High => 0x85,
        }
    }
}

/// The per-voice codebook for one band, as resolved out of decoder state.
#[derive(Debug, Clone, Copy)]
pub struct VoiceCodebook<'a> {
    /// The band's whole codebook array, from `state[+0x60]` or `state[+0x64]`.
    pub cb: &'a [i8],
    /// Bytes per entry, from `state[+0x38]` or `state[+0x3c]`.
    pub entry_bytes: usize,
    /// Scaling shift, from `state.byte[+0x84]` or `state.byte[+0x85]`.
    pub shift: u8,
}

/// ARM `lsl` by a register: the amount is the low byte, and 32 or more gives
/// zero. Rust's `<<` panics there instead, so it cannot be used directly.
pub fn arm_lsl(v: i32, amount: u32) -> i32 {
    let a = amount & 0xff;
    if a >= 32 {
        0
    } else {
        ((v as u32) << a) as i32
    }
}

/// Byte offset of a per-voice codebook, relative to its band's base pointer.
pub fn codebook_offset(entry_bytes: usize, shape_bits: u32, selector: u8) -> usize {
    entry_bytes.wrapping_mul((1usize << shape_bits).wrapping_mul(selector as usize))
}

/// Read every sign and index for a frame, in bitstream order.
///
/// Returns `(signs, ind)`.
pub fn read_indices(par: &SplitCbParams, bits: &mut Bits) -> (Vec<u32>, Vec<u32>) {
    let mut signs = vec![0u32; par.nb_subvect];
    let mut ind = vec![0u32; par.nb_subvect];
    for i in 0..par.nb_subvect {
        signs[i] = if par.have_sign { bits.unpack(1) } else { 0 };
        ind[i] = bits.unpack(par.shape_bits as i32);
    }
    (signs, ind)
}

/// Expand the innovation, given signs and indices already read.
///
/// `cb` is the per-voice codebook, already offset to the right voice.
pub fn expand(
    cb: &[i8],
    par: &SplitCbParams,
    shift: u32,
    signs: &[u32],
    ind: &[u32],
    exc: &mut [i32],
) {
    let up = 14u32.wrapping_sub(shift);
    for i in 0..par.nb_subvect {
        let negate = signs.get(i).copied().unwrap_or(0) != 0;
        let base = (ind.get(i).copied().unwrap_or(0) as usize).wrapping_mul(par.subvect_size);
        for j in 0..par.subvect_size {
            let Some(c) = cb.get(base.wrapping_add(j)) else {
                continue;
            };
            let v = arm_lsl(*c as i32, up);
            let out = i * par.subvect_size + j;
            if let Some(slot) = exc.get_mut(out) {
                *slot = if negate { v.wrapping_neg() } else { v };
            }
        }
    }
}

/// The whole routine: resolve the per-voice codebook, read the bitstream, and
/// expand into `exc`.
///
/// `selector` is the byte the original fetches through `state[+0x00]` at
/// `cdbk_offset`.
pub fn unquant(
    cbk: &VoiceCodebook,
    selector: u8,
    par: &SplitCbParams,
    bits: &mut Bits,
    exc: &mut [i32],
) {
    let off = codebook_offset(cbk.entry_bytes, par.shape_bits, selector);
    let cb = cbk.cb.get(off..).unwrap_or(&[]);
    let (signs, ind) = read_indices(par, bits);
    expand(cb, par, cbk.shift as u32, &signs, &ind, exc);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn par() -> SplitCbParams {
        SplitCbParams {
            subvect_size: 5,
            nb_subvect: 8,
            shape_bits: 6,
            have_sign: false,
        }
    }

    #[test]
    fn per_voice_offset_steps_a_whole_codebook() {
        // split_cb_nb is {5, 8, exc_5_64_table, 6, 0}: 64 entries of 5 bytes.
        assert_eq!(codebook_offset(5, 6, 0), 0);
        assert_eq!(codebook_offset(5, 6, 1), 5 * 64);
        assert_eq!(codebook_offset(5, 6, 3), 3 * 5 * 64);
    }

    #[test]
    fn entries_are_shifted_up_by_fourteen_minus_shift() {
        let cb: Vec<i8> = (0..64 * 5).map(|i| (i % 7) as i8 - 3).collect();
        let p = par();
        let mut exc = vec![0i32; p.nb_subvect * p.subvect_size];
        expand(&cb, &p, 0, &[0; 8], &[0; 8], &mut exc);
        // shift 0 means << 14.
        assert_eq!(exc[0], (cb[0] as i32) << 14);
        assert_eq!(exc[1], (cb[1] as i32) << 14);
    }

    #[test]
    fn a_shift_above_fourteen_yields_zero_rather_than_panicking() {
        // 14 - 20 is negative; ARM takes the low byte, 0xfa = 250 >= 32.
        assert_eq!(arm_lsl(-128, 14u32.wrapping_sub(20)), 0);
        assert_eq!(arm_lsl(1, 31), i32::MIN);
        assert_eq!(arm_lsl(1, 32), 0);
        // The low byte is what counts, so 256 is a shift of zero.
        assert_eq!(arm_lsl(3, 256), 3);
    }

    #[test]
    fn a_sign_bit_negates_the_whole_subvector() {
        let cb: Vec<i8> = (0..64 * 5).map(|i| (i % 7) as i8 - 3).collect();
        let p = par();
        let mut plus = vec![0i32; p.nb_subvect * p.subvect_size];
        let mut minus = vec![0i32; p.nb_subvect * p.subvect_size];
        expand(&cb, &p, 2, &[0; 8], &[1; 8], &mut plus);
        expand(&cb, &p, 2, &[1; 8], &[1; 8], &mut minus);
        for (a, b) in plus.iter().zip(minus.iter()) {
            assert_eq!(*a, -*b);
        }
    }

    #[test]
    fn each_subvector_uses_its_own_index() {
        let cb: Vec<i8> = (0..64 * 5).map(|i| i as i8).collect();
        let p = par();
        let mut exc = vec![0i32; p.nb_subvect * p.subvect_size];
        let ind = [0u32, 1, 2, 3, 4, 5, 6, 7];
        expand(&cb, &p, 14, &[0; 8], &ind, &mut exc);
        // shift 14 means << 0, so the values are the raw codebook bytes.
        for (i, want) in ind.iter().enumerate() {
            assert_eq!(exc[i * 5], cb[*want as usize * 5] as i32);
        }
    }

    #[test]
    fn a_short_codebook_is_skipped_rather_than_panicking() {
        let p = par();
        let mut exc = vec![0i32; p.nb_subvect * p.subvect_size];
        expand(&[1i8, 2, 3], &p, 0, &[0; 8], &[0; 8], &mut exc);
        assert_eq!(exc[0], 1 << 14);
        // Past the end of the codebook, untouched.
        assert_eq!(exc[4], 0);
    }

    #[test]
    fn signs_and_indices_alternate_in_the_bitstream() {
        // have_sign with shape_bits 6 means 7 bits per subvector: s iiiiii.
        // "1 000001" then "0 000010", packed MSB first across the byte break.
        let p = SplitCbParams {
            subvect_size: 5,
            nb_subvect: 2,
            shape_bits: 6,
            have_sign: true,
        };
        let buf = [0b1000_0010u8, 0b0000_1000, 0x00];
        let mut b = Bits::new(&buf);
        let (signs, ind) = read_indices(&p, &mut b);
        assert_eq!(signs, vec![1, 0]);
        assert_eq!(ind, vec![1, 2]);
        assert_eq!(b.position(), 14);
    }
}
