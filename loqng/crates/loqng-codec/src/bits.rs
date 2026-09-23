//! `speex_bits_unpack_unsigned` — `loqmsx.so+0x14ac`. The bitstream reader
//! every decode path goes through, one bit at a time.
//!
//! Untouched upstream Speex, so this is a transcription of `libspeex/bits.c`
//! confirmed field for field against the disassembly. `loqrs` already patches
//! it natively in `crates/loqhost/src/libc.rs`, and that patch is the reference
//! this agrees with.
//!
//! # `SpeexBits`, as the guest lays it out
//!
//! ```text
//! +0x00  char *chars
//! +0x04  int   nbBits
//! +0x08  int   charPtr
//! +0x0c  int   bitPtr
//! +0x10  int   owner
//! +0x14  int   overflow
//! ```
//!
//! Two details a rewrite loses:
//!
//! * **`overflow` is sticky.** Once set, every subsequent read returns 0
//!   without advancing the cursors, so a truncated frame degrades quietly
//!   instead of running off the buffer.
//! * **The overflow test runs before every read**, including reads that would
//!   have fit. It compares `charPtr*8 + bitPtr + nb` against `nbBits`, so a
//!   read ending exactly on `nbBits` is fine and one bit past is not.

/// Byte size of the guest `SpeexBits` struct.
pub const SPEEX_BITS_LEN: u32 = 0x18;

/// Field offsets in the guest struct, for a harness that has to build one.
pub mod field {
    pub const CHARS: u32 = 0x00;
    pub const NB_BITS: u32 = 0x04;
    pub const CHAR_PTR: u32 = 0x08;
    pub const BIT_PTR: u32 = 0x0c;
    pub const OWNER: u32 = 0x10;
    pub const OVERFLOW: u32 = 0x14;
}

/// A bitstream cursor over a borrowed buffer.
#[derive(Debug, Clone)]
pub struct Bits<'a> {
    pub chars: &'a [u8],
    pub nb_bits: i32,
    pub char_ptr: i32,
    pub bit_ptr: i32,
    pub overflow: i32,
}

impl<'a> Bits<'a> {
    /// A cursor at the start of `chars`, with `nbBits` set to the whole buffer.
    pub fn new(chars: &'a [u8]) -> Self {
        Bits {
            chars,
            nb_bits: (chars.len() as i32).wrapping_mul(8),
            char_ptr: 0,
            bit_ptr: 0,
            overflow: 0,
        }
    }

    /// Read `nb` bits, most significant first.
    pub fn unpack(&mut self, nb: i32) -> u32 {
        if (self.char_ptr.wrapping_shl(3))
            .wrapping_add(self.bit_ptr)
            .wrapping_add(nb)
            > self.nb_bits
        {
            self.overflow = 1;
        }
        if self.overflow != 0 {
            return 0;
        }

        let mut d: u32 = 0;
        for _ in 0..nb.max(0) {
            let byte = self.chars.get(self.char_ptr as usize).copied().unwrap_or(0) as u32;
            d = (d << 1) | ((byte >> (7 - self.bit_ptr)) & 1);
            self.bit_ptr += 1;
            if self.bit_ptr == 8 {
                self.bit_ptr = 0;
                self.char_ptr += 1;
            }
        }
        d
    }

    /// Bits consumed so far.
    pub fn position(&self) -> i32 {
        self.char_ptr.wrapping_shl(3).wrapping_add(self.bit_ptr)
    }

    /// Whether a read has run past the end.
    ///
    /// Sticky: once set, every later `unpack` returns 0 without advancing, so
    /// a caller can decode a whole frame and test this once at the end rather
    /// than after each field.
    pub fn overflowed(&self) -> bool {
        self.overflow != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_come_out_most_significant_first() {
        let buf = [0b1011_0010u8, 0b0100_0001];
        let mut b = Bits::new(&buf);
        assert_eq!(b.unpack(1), 1);
        assert_eq!(b.unpack(3), 0b011);
        assert_eq!(b.unpack(4), 0b0010);
        assert_eq!(b.unpack(8), 0b0100_0001);
    }

    #[test]
    fn a_read_spanning_a_byte_boundary_is_continuous() {
        let buf = [0xffu8, 0x00];
        let mut b = Bits::new(&buf);
        assert_eq!(b.unpack(4), 0xf);
        assert_eq!(b.unpack(8), 0b1111_0000);
        assert_eq!(b.position(), 12);
    }

    #[test]
    fn overflow_is_sticky_and_returns_zero() {
        let buf = [0xffu8];
        let mut b = Bits::new(&buf);
        assert_eq!(b.unpack(8), 0xff);
        // One bit past nbBits: latches, and never recovers.
        assert_eq!(b.unpack(1), 0);
        assert_eq!(b.overflow, 1);
        b.char_ptr = 0;
        b.bit_ptr = 0;
        assert_eq!(b.unpack(1), 0);
    }

    #[test]
    fn a_read_ending_exactly_on_nb_bits_still_fits() {
        let buf = [0xa5u8];
        let mut b = Bits::new(&buf);
        assert_eq!(b.unpack(8), 0xa5);
        assert_eq!(b.overflow, 0);
    }
}
