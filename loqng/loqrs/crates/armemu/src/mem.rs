//! Guest address space: flat 32-bit, lazily mapped in 64 KiB pages.

use std::fmt;

pub const PAGE_BITS: u32 = 16;
pub const PAGE_SIZE: usize = 1 << PAGE_BITS;
pub const PAGE_MASK: u32 = (PAGE_SIZE as u32) - 1;
pub const NUM_PAGES: usize = 1 << (32 - PAGE_BITS);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    Unmapped(u32),
    Overlap(u32, u32),
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Unmapped(a) => write!(f, "unmapped guest address 0x{a:08x}"),
            Fault::Overlap(a, n) => write!(f, "map overlaps existing pages at 0x{a:08x}+0x{n:x}"),
        }
    }
}

pub type MemResult<T> = Result<T, Fault>;

pub struct Mem {
    /// Owning storage; `None` means the page is not mapped.
    pages: Vec<Option<Box<[u8]>>>,
    /// Hot-path lookup table mirroring `pages`; null means unmapped.
    ptrs: Vec<*mut u8>,
    resident: usize,
    /// Debug watchpoints: addresses whose word value is polled each step.
    pub watch: Vec<u32>,
    pub watch_shadow: Vec<u32>,
}

// The raw pointers alias only into `pages`, which never moves while a page is live.
unsafe impl Send for Mem {}

impl Default for Mem {
    fn default() -> Self {
        Self::new()
    }
}

impl Mem {
    pub fn new() -> Self {
        Mem {
            pages: (0..NUM_PAGES).map(|_| None).collect(),
            ptrs: vec![std::ptr::null_mut(); NUM_PAGES],
            resident: 0,
            watch: Vec::new(),
            watch_shadow: Vec::new(),
        }
    }

    /// Bytes currently committed.
    pub fn resident_bytes(&self) -> usize {
        self.resident * PAGE_SIZE
    }

    #[inline(always)]
    fn page_of(addr: u32) -> usize {
        (addr >> PAGE_BITS) as usize
    }

    /// `addr >> PAGE_BITS` cannot exceed NUM_PAGES for a 32-bit address, so
    /// the bounds check this skips can never fire.
    #[inline(always)]
    fn page_ptr(&self, addr: u32) -> *mut u8 {
        unsafe { *self.ptrs.get_unchecked(Self::page_of(addr)) }
    }

    /// Address of the page-pointer table, for compiled code that indexes it
    /// directly instead of calling back in.
    ///
    /// The table is allocated once at its full size and never grows, so this
    /// address is stable for the life of this `Mem` — which is what makes it
    /// safe to bake into emitted code. Entries change as pages are mapped and
    /// unmapped; the table itself does not move. Blocks compiled against one
    /// `Mem` must not be reused with another, which [`BlockCache`] enforces.
    ///
    /// [`BlockCache`]: crate::block::BlockCache
    #[inline]
    pub fn ptrs_base(&self) -> usize {
        self.ptrs.as_ptr() as usize
    }

    #[inline(always)]
    pub fn is_mapped(&self, addr: u32) -> bool {
        !self.ptrs[Self::page_of(addr)].is_null()
    }

    /// The host pointer backing `addr`'s page, or null when unmapped.
    ///
    /// Instruction fetch holds one of these across a whole page of straight-
    /// line code instead of resolving per instruction. Only the page base is
    /// cached, never the bytes, so a guest store still shows through.
    #[inline(always)]
    pub(crate) fn page_base(&self, addr: u32) -> *const u8 {
        self.page_ptr(addr)
    }

    fn commit(&mut self, page: usize) {
        if self.ptrs[page].is_null() {
            let mut b = vec![0u8; PAGE_SIZE].into_boxed_slice();
            self.ptrs[page] = b.as_mut_ptr();
            self.pages[page] = Some(b);
            self.resident += 1;
        }
    }

    /// Map `[addr, addr+len)`, committing every page it touches. Idempotent.
    pub fn map(&mut self, addr: u32, len: u32) {
        if len == 0 {
            return;
        }
        let first = Self::page_of(addr);
        let last = Self::page_of(addr.wrapping_add(len - 1));
        for p in first..=last {
            self.commit(p);
        }
    }

    /// Release the pages fully covered by `[addr, addr+len)`.
    pub fn unmap(&mut self, addr: u32, len: u32) {
        if len == 0 {
            return;
        }
        let first = ((addr + PAGE_MASK) >> PAGE_BITS) as usize;
        let end = addr.wrapping_add(len) >> PAGE_BITS;
        for p in first..(end as usize) {
            if self.pages[p].take().is_some() {
                self.ptrs[p] = std::ptr::null_mut();
                self.resident -= 1;
            }
        }
    }

    #[inline(always)]
    pub fn read_u8(&self, addr: u32) -> MemResult<u8> {
        let p = self.page_ptr(addr);
        if p.is_null() {
            return Err(Fault::Unmapped(addr));
        }
        Ok(unsafe { *p.add((addr & PAGE_MASK) as usize) })
    }

    #[inline(always)]
    pub fn write_u8(&mut self, addr: u32, v: u8) -> MemResult<()> {
        let p = self.page_ptr(addr);
        if p.is_null() {
            return Err(Fault::Unmapped(addr));
        }
        unsafe { *p.add((addr & PAGE_MASK) as usize) = v };
        Ok(())
    }

    #[inline(always)]
    pub fn read_u16(&self, addr: u32) -> MemResult<u16> {
        let off = (addr & PAGE_MASK) as usize;
        if off <= PAGE_SIZE - 2 {
            let p = self.page_ptr(addr);
            if p.is_null() {
                return Err(Fault::Unmapped(addr));
            }
            Ok(u16::from_le(unsafe {
                (p.add(off) as *const u16).read_unaligned()
            }))
        } else {
            Ok(u16::from_le_bytes([
                self.read_u8(addr)?,
                self.read_u8(addr.wrapping_add(1))?,
            ]))
        }
    }

    #[inline(always)]
    pub fn write_u16(&mut self, addr: u32, v: u16) -> MemResult<()> {
        let b = v.to_le_bytes();
        self.write_u8(addr, b[0])?;
        self.write_u8(addr.wrapping_add(1), b[1])
    }

    #[inline(always)]
    pub fn read_u32(&self, addr: u32) -> MemResult<u32> {
        let off = (addr & PAGE_MASK) as usize;
        if off <= PAGE_SIZE - 4 {
            let p = self.page_ptr(addr);
            if p.is_null() {
                return Err(Fault::Unmapped(addr));
            }
            // One load rather than four: the guest is little-endian, and
            // `from_le` compiles away on the hosts we target.
            Ok(u32::from_le(unsafe {
                (p.add(off) as *const u32).read_unaligned()
            }))
        } else {
            Ok(u32::from_le_bytes([
                self.read_u8(addr)?,
                self.read_u8(addr.wrapping_add(1))?,
                self.read_u8(addr.wrapping_add(2))?,
                self.read_u8(addr.wrapping_add(3))?,
            ]))
        }
    }

    #[inline(always)]
    pub fn write_u32(&mut self, addr: u32, v: u32) -> MemResult<()> {
        let off = (addr & PAGE_MASK) as usize;
        if off <= PAGE_SIZE - 4 {
            let p = self.page_ptr(addr);
            if p.is_null() {
                return Err(Fault::Unmapped(addr));
            }
            unsafe {
                (p.add(off) as *mut u32).write_unaligned(v.to_le());
            }
            Ok(())
        } else {
            let b = v.to_le_bytes();
            for (i, byte) in b.iter().enumerate() {
                self.write_u8(addr.wrapping_add(i as u32), *byte)?;
            }
            Ok(())
        }
    }

    pub fn read_bytes(&self, addr: u32, len: usize) -> MemResult<Vec<u8>> {
        let mut out = vec![0u8; len];
        self.read_into(addr, &mut out)?;
        Ok(out)
    }

    pub fn read_into(&self, addr: u32, dst: &mut [u8]) -> MemResult<()> {
        let mut a = addr;
        let mut i = 0;
        while i < dst.len() {
            let off = (a & PAGE_MASK) as usize;
            let p = self.page_ptr(a);
            if p.is_null() {
                return Err(Fault::Unmapped(a));
            }
            let n = (PAGE_SIZE - off).min(dst.len() - i);
            unsafe { std::ptr::copy_nonoverlapping(p.add(off), dst[i..].as_mut_ptr(), n) };
            i += n;
            a = a.wrapping_add(n as u32);
        }
        Ok(())
    }

    pub fn write_bytes(&mut self, addr: u32, src: &[u8]) -> MemResult<()> {
        let mut a = addr;
        let mut i = 0;
        while i < src.len() {
            let off = (a & PAGE_MASK) as usize;
            let p = self.page_ptr(a);
            if p.is_null() {
                return Err(Fault::Unmapped(a));
            }
            let n = (PAGE_SIZE - off).min(src.len() - i);
            unsafe { std::ptr::copy_nonoverlapping(src[i..].as_ptr(), p.add(off), n) };
            i += n;
            a = a.wrapping_add(n as u32);
        }
        Ok(())
    }

    pub fn fill(&mut self, addr: u32, len: usize, val: u8) -> MemResult<()> {
        let mut a = addr;
        let mut i = 0;
        while i < len {
            let off = (a & PAGE_MASK) as usize;
            let p = self.page_ptr(a);
            if p.is_null() {
                return Err(Fault::Unmapped(a));
            }
            let n = (PAGE_SIZE - off).min(len - i);
            unsafe { std::ptr::write_bytes(p.add(off), val, n) };
            i += n;
            a = a.wrapping_add(n as u32);
        }
        Ok(())
    }

    /// NUL-terminated guest string as raw bytes (without the terminator).
    pub fn read_cstr(&self, addr: u32) -> MemResult<Vec<u8>> {
        let mut out = Vec::new();
        let mut a = addr;
        loop {
            let b = self.read_u8(a)?;
            if b == 0 {
                return Ok(out);
            }
            out.push(b);
            a = a.wrapping_add(1);
        }
    }

    pub fn read_cstring_lossy(&self, addr: u32) -> MemResult<String> {
        Ok(String::from_utf8_lossy(&self.read_cstr(addr)?).into_owned())
    }

    /// Write `s` plus a NUL terminator.
    pub fn write_cstr(&mut self, addr: u32, s: &[u8]) -> MemResult<()> {
        self.write_bytes(addr, s)?;
        self.write_u8(addr.wrapping_add(s.len() as u32), 0)
    }
}

use crate::snap::{Reader, Snap, SnapResult, Writer};

/// Pages are stored sparsely and zero-run encoded. Most of a live machine is
/// stack that was never touched and heap that was calloc'd, so this is the
/// difference between a 40 MB file and a small one.
impl Snap for Mem {
    fn save(&self, w: &mut Writer) {
        w.u32(self.resident as u32);
        for (i, page) in self.pages.iter().enumerate() {
            let Some(bytes) = page else { continue };
            w.u32(i as u32);
            let len_at = w.buf.len();
            w.u32(0);

            let mut pos = 0;
            while pos < PAGE_SIZE {
                let zeros = bytes[pos..].iter().take_while(|b| **b == 0).count();
                w.u32(zeros as u32);
                pos += zeros;
                if pos >= PAGE_SIZE {
                    break;
                }
                let lit = bytes[pos..].iter().take_while(|b| **b != 0).count();
                w.u32(lit as u32);
                w.raw(&bytes[pos..pos + lit]);
                pos += lit;
            }

            let len = (w.buf.len() - len_at - 4) as u32;
            w.buf[len_at..len_at + 4].copy_from_slice(&len.to_le_bytes());
        }
        w.put(&self.watch);
        w.put(&self.watch_shadow);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        let mut m = Mem::new();
        let n = r.count()?;
        for _ in 0..n {
            let index = r.u32()? as usize;
            if index >= NUM_PAGES {
                return Err(format!("snapshot has page index {index}"));
            }
            let len = r.u32()? as usize;
            let body = r.take(len)?;
            let mut rd = Reader::new(body);

            m.commit(index);
            let page = m.pages[index].as_mut().unwrap();
            let mut pos = 0;
            while pos < PAGE_SIZE {
                let zeros = rd.u32()? as usize;
                pos = pos
                    .checked_add(zeros)
                    .filter(|p| *p <= PAGE_SIZE)
                    .ok_or_else(|| format!("snapshot page {index} overruns on zeros"))?;
                if pos >= PAGE_SIZE {
                    break;
                }
                let lit = rd.u32()? as usize;
                let end = pos
                    .checked_add(lit)
                    .filter(|p| *p <= PAGE_SIZE)
                    .ok_or_else(|| format!("snapshot page {index} overruns on literals"))?;
                page[pos..end].copy_from_slice(rd.take(lit)?);
                pos = end;
            }
        }
        m.watch = r.get()?;
        m.watch_shadow = r.get()?;
        Ok(m)
    }
}
