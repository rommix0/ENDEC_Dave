//! Guest heap. Bookkeeping lives on the host side, so no headers are written
//! into guest memory and a stray guest write cannot corrupt the allocator.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use armemu::Mem;

const ALIGN: u32 = 8;

pub struct Heap {
    base: u32,
    limit: u32,
    brk: u32,
    live: HashMap<u32, u32>,
    /// addr -> size, kept sorted so neighbouring frees can coalesce.
    free_by_addr: BTreeMap<u32, u32>,
    /// (size, addr), for best fit.
    free_by_size: BTreeSet<(u32, u32)>,
    pub peak: u32,
    pub allocations: u64,
}

impl Heap {
    pub fn new(base: u32, limit: u32) -> Self {
        Heap {
            base,
            limit,
            brk: base,
            live: HashMap::new(),
            free_by_addr: BTreeMap::new(),
            free_by_size: BTreeSet::new(),
            peak: 0,
            allocations: 0,
        }
    }

    pub fn in_heap(&self, addr: u32) -> bool {
        addr >= self.base && addr < self.limit
    }

    pub fn size_of(&self, addr: u32) -> Option<u32> {
        self.live.get(&addr).copied()
    }

    pub fn used(&self) -> u32 {
        self.brk - self.base
    }

    fn take_free(&mut self, addr: u32, size: u32) {
        self.free_by_addr.remove(&addr);
        self.free_by_size.remove(&(size, addr));
    }

    fn put_free(&mut self, addr: u32, size: u32) {
        self.free_by_addr.insert(addr, size);
        self.free_by_size.insert((size, addr));
    }

    pub fn malloc(&mut self, mem: &mut Mem, bytes: u32) -> u32 {
        let size = ((bytes.max(1)) + ALIGN - 1) & !(ALIGN - 1);
        self.allocations += 1;

        // Best fit among free blocks.
        if let Some(&(bsize, baddr)) = self.free_by_size.range((size, 0)..).next() {
            self.take_free(baddr, bsize);
            let leftover = bsize - size;
            if leftover >= 32 {
                self.put_free(baddr + size, leftover);
                self.live.insert(baddr, size);
            } else {
                self.live.insert(baddr, bsize);
            }
            return baddr;
        }

        if self.brk.checked_add(size).is_none() || self.brk + size > self.limit {
            return 0;
        }
        let addr = self.brk;
        mem.map(addr, size);
        self.brk += size;
        self.peak = self.peak.max(self.brk - self.base);
        self.live.insert(addr, size);
        addr
    }

    pub fn calloc(&mut self, mem: &mut Mem, n: u32, each: u32) -> u32 {
        let total = (n as u64) * (each as u64);
        if total > u32::MAX as u64 {
            return 0;
        }
        let a = self.malloc(mem, total as u32);
        if a != 0 {
            let _ = mem.fill(a, total as usize, 0);
        }
        a
    }

    pub fn free(&mut self, addr: u32) -> bool {
        if addr == 0 {
            return true;
        }
        let Some(size) = self.live.remove(&addr) else {
            return false;
        };

        let mut start = addr;
        let mut len = size;

        // Coalesce with the block that ends where this one starts.
        if let Some((&paddr, &psize)) = self.free_by_addr.range(..start).next_back() {
            if paddr + psize == start {
                self.take_free(paddr, psize);
                start = paddr;
                len += psize;
            }
        }
        // ...and with the one that starts where this one ends.
        if let Some((&naddr, &nsize)) = self.free_by_addr.range(start + len..).next() {
            if naddr == start + len {
                self.take_free(naddr, nsize);
                len += nsize;
            }
        }

        if start + len == self.brk {
            self.brk = start;
        } else {
            self.put_free(start, len);
        }
        true
    }

    pub fn realloc(&mut self, mem: &mut Mem, addr: u32, bytes: u32) -> u32 {
        if addr == 0 {
            return self.malloc(mem, bytes);
        }
        if bytes == 0 {
            self.free(addr);
            return 0;
        }
        let Some(old) = self.live.get(&addr).copied() else {
            return 0;
        };
        let want = ((bytes.max(1)) + ALIGN - 1) & !(ALIGN - 1);
        if want <= old {
            if old - want >= 32 {
                self.live.insert(addr, want);
                self.put_free(addr + want, old - want);
            }
            return addr;
        }
        // Grow in place if the next block is free and large enough.
        if let Some((&naddr, &nsize)) = self.free_by_addr.range(addr + old..).next() {
            if naddr == addr + old && old + nsize >= want {
                self.take_free(naddr, nsize);
                let total = old + nsize;
                let leftover = total - want;
                if leftover >= 32 {
                    self.put_free(addr + want, leftover);
                    self.live.insert(addr, want);
                } else {
                    self.live.insert(addr, total);
                }
                return addr;
            }
        }
        if addr + old == self.brk && addr + want <= self.limit {
            mem.map(addr, want);
            self.brk = addr + want;
            self.peak = self.peak.max(self.brk - self.base);
            self.live.insert(addr, want);
            return addr;
        }

        let new = self.malloc(mem, bytes);
        if new == 0 {
            return 0;
        }
        if let Ok(data) = mem.read_bytes(addr, old as usize) {
            let _ = mem.write_bytes(new, &data);
        }
        self.free(addr);
        new
    }
}

// SNAPSHOT-MARK
use armemu::{Reader, Snap, SnapResult, Writer};

impl Snap for Heap {
    fn save(&self, w: &mut Writer) {
        w.u32(self.base);
        w.u32(self.limit);
        w.u32(self.brk);
        w.put(&self.live);
        w.put(&self.free_by_addr);
        w.put(&self.free_by_size);
        w.u32(self.peak);
        w.u64(self.allocations);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(Heap {
            base: r.u32()?,
            limit: r.u32()?,
            brk: r.u32()?,
            live: r.get()?,
            free_by_addr: r.get()?,
            free_by_size: r.get()?,
            peak: r.u32()?,
            allocations: r.u64()?,
        })
    }
}
