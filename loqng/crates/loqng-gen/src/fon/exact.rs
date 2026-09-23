//! The broad-to-narrow mapper, exact by construction.
//!
//! `fonema_us.rs` is `FonemaLargo2Stretto_US` translated one ARM instruction
//! to one Rust statement by `tools/fonxlate.py`. This module is what that code
//! runs against: the four flags, a flat memory holding the structures the
//! caller passes, and native `strcmp`/`strncmp`/`strstr`/`strlen`.
//!
//! No ARM is executed. The memory holds only data: the caller's arrays, a
//! stack, and the module's own tables (`fonema_us_image.rs`) at their ELF
//! vaddrs, which is where the translated code's GOT arithmetic expects them.
//!
//! The context layout is the one `xtask fonprobe` builds for the real function:
//!
//! ```text
//! ctx[0x04]  out[]    16-byte records, the narrow code at +6
//! ctx[0x10]  ph[]     8-byte records {u32 word, u8 code}
//! ctx[0x28]  phrase   +0x04 word count, +0x18 first entry
//! ctx[0x2c]  elems    20-byte records, +0x04 spelling, +0x12 a flag byte
//! ctx[0x40]  cfg      +0x30c the style; Dave is 3
//! ```

use super::fonema_us::fonema_largo2stretto_us;
use super::fonema_us_image::IMAGE;
use super::narrow::Phone;

const HEAP: u32 = 0x1000_0000;
const STACK: u32 = 0x2000_0000;
const STACK_SIZE: u32 = 0x4000;
const STYLE: u32 = 3;

/// Records past the end of the phrase that read as zero. The function looks
/// ahead without a length check, so the array must be padded; see `narrow`.
const PAD: usize = 16;

#[derive(Clone, Copy, Default)]
pub struct Flags {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
}

impl Flags {
    pub fn sub(a: u32, b: u32) -> (u32, Flags) {
        let r = a.wrapping_sub(b);
        (
            r,
            Flags {
                n: r >> 31 != 0,
                z: r == 0,
                c: a >= b,
                v: ((a ^ b) & (a ^ r)) >> 31 != 0,
            },
        )
    }

    pub fn add(a: u32, b: u32) -> (u32, Flags) {
        let (r, c) = a.overflowing_add(b);
        (
            r,
            Flags {
                n: r >> 31 != 0,
                z: r == 0,
                c,
                v: (!(a ^ b) & (a ^ r)) >> 31 != 0,
            },
        )
    }

    /// N and Z from the result, C from the shifter, V unchanged.
    pub fn logic(r: u32, c: bool, v: bool) -> Flags {
        Flags {
            n: r >> 31 != 0,
            z: r == 0,
            c,
            v,
        }
    }
}

pub struct Mem {
    heap: Vec<u8>,
    stack: Vec<u8>,
}

impl Mem {
    fn new() -> Self {
        Mem {
            heap: Vec::new(),
            stack: vec![0; STACK_SIZE as usize],
        }
    }

    pub fn sp(&self) -> u32 {
        STACK + STACK_SIZE - 16
    }

    fn alloc(&mut self, n: u32) -> u32 {
        let at = HEAP + self.heap.len() as u32;
        self.heap
            .resize(self.heap.len() + ((n as usize + 7) & !7), 0);
        at
    }

    #[inline]
    pub fn r8(&self, a: u32) -> u8 {
        if a >= STACK && a < STACK + STACK_SIZE {
            return self.stack[(a - STACK) as usize];
        }
        if a >= HEAP && ((a - HEAP) as usize) < self.heap.len() {
            return self.heap[(a - HEAP) as usize];
        }
        let k = IMAGE.partition_point(|(s, _)| *s <= a);
        if k > 0 {
            let (s, b) = IMAGE[k - 1];
            if ((a - s) as usize) < b.len() {
                return b[(a - s) as usize];
            }
        }
        panic!("FonemaLargo2Stretto_US read unmapped 0x{a:08x}");
    }

    #[inline]
    pub fn r32(&self, a: u32) -> u32 {
        debug_assert!(a & 3 == 0, "unaligned ldr at 0x{a:08x}");
        u32::from_le_bytes([self.r8(a), self.r8(a + 1), self.r8(a + 2), self.r8(a + 3)])
    }

    #[inline]
    pub fn w8(&mut self, a: u32, v: u8) {
        if a >= STACK && a < STACK + STACK_SIZE {
            self.stack[(a - STACK) as usize] = v;
        } else if a >= HEAP && ((a - HEAP) as usize) < self.heap.len() {
            self.heap[(a - HEAP) as usize] = v;
        } else {
            panic!("FonemaLargo2Stretto_US wrote unmapped 0x{a:08x}");
        }
    }

    #[inline]
    pub fn w32(&mut self, a: u32, v: u32) {
        for (k, b) in v.to_le_bytes().into_iter().enumerate() {
            self.w8(a + k as u32, b);
        }
    }

    pub fn strcmp(&self, a: u32, b: u32) -> u32 {
        self.strncmp(a, b, u32::MAX)
    }

    pub fn strncmp(&self, a: u32, b: u32, n: u32) -> u32 {
        for k in 0..n {
            let (x, y) = (self.r8(a + k), self.r8(b + k));
            if x != y || x == 0 {
                return (x as i32 - y as i32) as u32;
            }
        }
        0
    }

    pub fn strlen(&self, a: u32) -> u32 {
        let mut k = 0;
        while self.r8(a + k) != 0 {
            k += 1;
        }
        k
    }

    pub fn strstr(&self, hay: u32, needle: u32) -> u32 {
        let n = self.strlen(needle);
        let h = self.strlen(hay);
        if n > h {
            return 0;
        }
        for s in 0..=h - n {
            if (0..n).all(|k| self.r8(hay + s + k) == self.r8(needle + k)) {
                return hay + s;
            }
        }
        0
    }
}

#[cfg(feature = "fon-cover")]
static HITS: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

#[cfg(feature = "fon-cover")]
pub fn hit(pc: u32) {
    let mut h = HITS.lock().unwrap();
    if h.is_empty() {
        h.resize(super::fonema_us_blocks::BLOCKS.len(), 0);
    }
    if let Ok(k) = super::fonema_us_blocks::BLOCKS.binary_search(&pc) {
        h[k] = h[k].saturating_add(1);
    }
}

/// `(block, times entered)` for every block, since the process started.
#[cfg(feature = "fon-cover")]
pub fn coverage() -> Vec<(u32, u32)> {
    let h = HITS.lock().unwrap();
    super::fonema_us_blocks::BLOCKS
        .iter()
        .enumerate()
        .map(|(k, &b)| (b, h.get(k).copied().unwrap_or(0)))
        .collect()
}

/// Every printable string in the image: the spellings the mapper compares
/// against, for tests that want its string arms to fire.
pub fn literals() -> Vec<String> {
    let mut out = Vec::new();
    for (_s, b) in IMAGE {
        for part in b.split(|&c| c == 0) {
            if !part.is_empty() && part.iter().all(|c| c.is_ascii_graphic()) {
                out.push(String::from_utf8_lossy(part).into_owned());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Map one phrase from broad to narrow, as `LoqTTS6` drives the mapper: one
/// call per broad phone, `i` from the return value, `j` from `*p4`.
///
/// Returns the narrow codes and, for each, the broad index that produced it.
pub fn narrow(phones: &[Phone], words: &[&str], elem12: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let mut m = Mem::new();
    let n = phones.len() + PAD;
    let ctx = m.alloc(0x100);
    let out = m.alloc(n as u32 * 16 + 64);
    let ph = m.alloc(n as u32 * 8);
    let phrase = m.alloc(28 * 4);
    let elems = m.alloc(words.len().max(1) as u32 * 20);
    let cfg = m.alloc(0x400);
    let p4 = m.alloc(4);

    m.w32(ctx + 0x04, out);
    m.w32(ctx + 0x10, ph);
    m.w32(ctx + 0x28, phrase);
    m.w32(ctx + 0x2C, elems);
    m.w32(ctx + 0x40, cfg);
    m.w32(cfg + 0x30C, STYLE);
    m.w32(phrase + 0x04, words.len() as u32);

    for (k, w) in words.iter().enumerate() {
        let s = m.alloc(w.len() as u32 + 1);
        for (q, &b) in w.as_bytes().iter().enumerate() {
            m.w8(s + q as u32, b);
        }
        let e = elems + k as u32 * 20;
        m.w32(e + 0x04, s);
        m.w8(e + 0x12, elem12.get(k).copied().unwrap_or(0));
    }
    for (k, p) in phones.iter().enumerate() {
        m.w32(ph + k as u32 * 8, p.word);
        m.w8(ph + k as u32 * 8 + 4, p.code);
    }

    let (mut i, mut last_j) = (0u32, 0u32);
    let mut owner = Vec::new();
    while (i as usize) < phones.len() {
        let before = i;
        let pos = phones[i as usize].word;
        m.w32(m.sp(), ph);
        i = fonema_largo2stretto_us(&mut m, [ctx, pos, i, p4]);
        last_j = m.r32(p4);
        while (owner.len() as u32) < last_j {
            owner.push(before);
        }
    }
    let codes = (0..last_j).map(|k| m.r8(out + k * 16 + 6)).collect();
    (codes, owner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phones(v: &[(u32, u8)]) -> Vec<Phone> {
        v.iter().map(|&(word, code)| Phone { word, code }).collect()
    }

    /// Broad and narrow arrays read out of guest memory during a synthesis
    /// (`xtask fonprobe`'s first case). The medial `t` of "testing" loses its
    /// aspiration after `s`.
    #[test]
    fn testing_one_two_three_matches_the_engine() {
        let ph = phones(&[
            (0, 0x2F),
            (0, 0x55),
            (0, 0x45),
            (0, 0x2F),
            (0, 0x11),
            (0, 0x52),
            (1, 0x23),
            (1, 0x06),
            (1, 0x50),
            (2, 0x2F),
            (2, 0x14),
            (3, 0x4A),
            (3, 0x27),
            (3, 0x0F),
            (4, 0x03),
        ]);
        let words = ["testing", "one", "two", "three", "."];
        let (got, owner) = narrow(&ph, &words, &[0; 5]);
        assert_eq!(
            got,
            [
                0x2F, 0x55, 0x45, 0x2E, 0x11, 0x52, 0x23, 0x06, 0x50, 0x2F, 0x14, 0x4A, 0x27, 0x0F,
                0x03
            ]
        );
        assert_eq!(owner, (0..15).collect::<Vec<u32>>());
    }

    /// Rule `0x2822c`, the one the rule table under-fired 22 times: the
    /// word-final `th` of "finished" becomes the unreleased `Hut` (slot 7).
    #[test]
    fn finished_matches_the_engine() {
        let ph = phones(&[
            (0, 0x4D),
            (0, 0x0E),
            (1, 0x42),
            (1, 0x10),
            (1, 0x50),
            (1, 0x11),
            (1, 0x48),
            (1, 0x2F),
            (2, 0x4A),
            (2, 0x0C),
            (2, 0x38),
            (3, 0x1B),
            (3, 0x2F),
            (4, 0x18),
            (4, 0x43),
            (5, 0x2F),
            (5, 0x23),
            (5, 0x55),
            (5, 0x50),
            (5, 0x2F),
            (5, 0x0E),
            (6, 0x23),
            (6, 0x06),
            (6, 0x50),
            (7, 0x02),
        ]);
        let words = ["he", "finished", "third", "out", "of", "twenty", "one", ","];
        let (got, _owner) = narrow(&ph, &words, &[0; 8]);
        assert_eq!(
            got,
            [
                0x4D, 0x0E, 0x42, 0x10, 0x50, 0x11, 0x48, 0x30, 0x4A, 0x0C, 0x38, 0x1B, 0x34, 0x18,
                0x44, 0x2F, 0x23, 0x55, 0x50, 0x34, 0x0E, 0x23, 0x06, 0x50, 0x02
            ]
        );
    }
}
