//! Binary snapshots, so an expensive guest initialisation is paid once.
//!
//! Bringing the voice bank up costs ~300 million guest instructions, and the
//! result is just bytes: guest pages, registers, and the host-side bookkeeping
//! that describes them. Writing those out and reading them back turns a
//! multi-second start into a file read.
//!
//! The format is little-endian. It is a cache, not an interchange format:
//! correctness comes from the caller refusing to restore a snapshot that was
//! not built from the same inputs, not from the encoding being portable.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;
use std::path::PathBuf;

pub const MAGIC: [u8; 8] = *b"LOQSNAP\x00";
pub const VERSION: u32 = 1;

pub type SnapResult<T> = Result<T, String>;

pub struct Writer {
    pub buf: Vec<u8>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub fn new() -> Self {
        Writer {
            buf: Vec::with_capacity(1 << 20),
        }
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Length-prefixed, and byte-for-byte what the generic `Vec<u8>` encoding
    /// produces; this one just does it a block at a time.
    pub fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }

    /// No length prefix; the reader has to know the size.
    pub fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    pub fn put<T: Snap>(&mut self, v: &T) {
        v.save(self);
    }
}

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn take(&mut self, n: usize) -> SnapResult<&'a [u8]> {
        if self.remaining() < n {
            return Err(format!(
                "snapshot truncated: wanted {n} bytes at offset {}, {} left",
                self.pos,
                self.remaining()
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> SnapResult<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> SnapResult<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> SnapResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> SnapResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn bytes(&mut self) -> SnapResult<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }

    pub fn get<T: Snap>(&mut self) -> SnapResult<T> {
        T::load(self)
    }

    /// A count read from the stream, rejected if it could not possibly be
    /// backed by the bytes that are left. Every encodable type costs at least
    /// one byte, so this bounds allocation without knowing the element type.
    pub fn count(&mut self) -> SnapResult<usize> {
        let n = self.u32()? as usize;
        if n > self.remaining() {
            return Err(format!(
                "snapshot claims {n} elements with only {} bytes left",
                self.remaining()
            ));
        }
        Ok(n)
    }
}

pub trait Snap: Sized {
    fn save(&self, w: &mut Writer);
    fn load(r: &mut Reader) -> SnapResult<Self>;
}

macro_rules! scalar {
    ($t:ty, $put:ident, $get:ident) => {
        impl Snap for $t {
            fn save(&self, w: &mut Writer) {
                w.$put(*self);
            }
            fn load(r: &mut Reader) -> SnapResult<Self> {
                r.$get()
            }
        }
    };
}

scalar!(u8, u8, u8);
scalar!(u16, u16, u16);
scalar!(u32, u32, u32);
scalar!(u64, u64, u64);

impl Snap for i32 {
    fn save(&self, w: &mut Writer) {
        w.u32(*self as u32);
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(r.u32()? as i32)
    }
}

impl Snap for usize {
    fn save(&self, w: &mut Writer) {
        w.u64(*self as u64);
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(r.u64()? as usize)
    }
}

impl Snap for bool {
    fn save(&self, w: &mut Writer) {
        w.u8(*self as u8);
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(r.u8()? != 0)
    }
}

/// By bits, so a restored machine compares equal to the one that was saved.
impl Snap for f64 {
    fn save(&self, w: &mut Writer) {
        w.u64(self.to_bits());
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(f64::from_bits(r.u64()?))
    }
}

impl Snap for String {
    fn save(&self, w: &mut Writer) {
        w.bytes(self.as_bytes());
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        let b = r.bytes()?;
        String::from_utf8(b.to_vec()).map_err(|e| format!("snapshot has a bad string: {e}"))
    }
}

impl Snap for PathBuf {
    fn save(&self, w: &mut Writer) {
        w.bytes(self.to_string_lossy().as_bytes());
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(PathBuf::from(String::load(r)?))
    }
}

impl<T: Snap> Snap for Option<T> {
    fn save(&self, w: &mut Writer) {
        match self {
            None => w.u8(0),
            Some(v) => {
                w.u8(1);
                v.save(w);
            }
        }
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        match r.u8()? {
            0 => Ok(None),
            1 => Ok(Some(T::load(r)?)),
            t => Err(format!("snapshot has option tag {t}")),
        }
    }
}

impl<A: Snap, B: Snap> Snap for (A, B) {
    fn save(&self, w: &mut Writer) {
        self.0.save(w);
        self.1.save(w);
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok((A::load(r)?, B::load(r)?))
    }
}

impl<A: Snap, B: Snap, C: Snap> Snap for (A, B, C) {
    fn save(&self, w: &mut Writer) {
        self.0.save(w);
        self.1.save(w);
        self.2.save(w);
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok((A::load(r)?, B::load(r)?, C::load(r)?))
    }
}

impl<T: Snap> Snap for Vec<T> {
    fn save(&self, w: &mut Writer) {
        w.u32(self.len() as u32);
        for v in self {
            v.save(w);
        }
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        let n = r.count()?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(T::load(r)?);
        }
        Ok(v)
    }
}

impl<K: Snap + Eq + Hash, V: Snap> Snap for HashMap<K, V> {
    fn save(&self, w: &mut Writer) {
        w.u32(self.len() as u32);
        for (k, v) in self {
            k.save(w);
            v.save(w);
        }
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        let n = r.count()?;
        let mut m = HashMap::with_capacity(n);
        for _ in 0..n {
            let k = K::load(r)?;
            m.insert(k, V::load(r)?);
        }
        Ok(m)
    }
}

impl<K: Snap + Ord, V: Snap> Snap for BTreeMap<K, V> {
    fn save(&self, w: &mut Writer) {
        w.u32(self.len() as u32);
        for (k, v) in self {
            k.save(w);
            v.save(w);
        }
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        let n = r.count()?;
        let mut m = BTreeMap::new();
        for _ in 0..n {
            let k = K::load(r)?;
            m.insert(k, V::load(r)?);
        }
        Ok(m)
    }
}

impl<T: Snap + Ord> Snap for BTreeSet<T> {
    fn save(&self, w: &mut Writer) {
        w.u32(self.len() as u32);
        for v in self {
            v.save(w);
        }
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        let n = r.count()?;
        let mut s = BTreeSet::new();
        for _ in 0..n {
            s.insert(T::load(r)?);
        }
        Ok(s)
    }
}

impl<T: Snap + Copy + Default, const N: usize> Snap for [T; N] {
    fn save(&self, w: &mut Writer) {
        for v in self {
            v.save(w);
        }
    }
    fn load(r: &mut Reader) -> SnapResult<Self> {
        let mut a = [T::default(); N];
        for slot in a.iter_mut() {
            *slot = T::load(r)?;
        }
        Ok(a)
    }
}

/// FNV-1a. Snapshot fingerprints only need to notice that an input changed,
/// and this avoids a dependency for the sake of sixty-four bits.
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv {
    pub fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }

    pub fn write_str(&mut self, s: &str) {
        self.write(s.as_bytes());
        self.write(&[0]);
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}
