//! ELF32 ARM loader: enough of a dynamic linker to bring up the Loquendo
//! shared objects without a guest ld.so.

use std::collections::HashMap;

use crate::mem::{Mem, PAGE_SIZE};

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;

const DT_NEEDED: u32 = 1;
const DT_PLTRELSZ: u32 = 2;
const DT_HASH: u32 = 4;
const DT_STRTAB: u32 = 5;
const DT_SYMTAB: u32 = 6;
const DT_RELA: u32 = 7;
const DT_RELASZ: u32 = 8;
const DT_INIT: u32 = 12;
const DT_FINI: u32 = 13;
const DT_SONAME: u32 = 14;
const DT_REL: u32 = 17;
const DT_RELSZ: u32 = 18;
const DT_PLTREL: u32 = 20;
const DT_JMPREL: u32 = 23;
const DT_INIT_ARRAY: u32 = 25;
const DT_FINI_ARRAY: u32 = 26;
const DT_INIT_ARRAYSZ: u32 = 27;
const DT_FINI_ARRAYSZ: u32 = 28;

const R_ARM_NONE: u32 = 0;
const R_ARM_PC24: u32 = 1;
const R_ARM_ABS32: u32 = 2;
const R_ARM_REL32: u32 = 3;
const R_ARM_COPY: u32 = 20;
const R_ARM_GLOB_DAT: u32 = 21;
const R_ARM_JUMP_SLOT: u32 = 22;
const R_ARM_RELATIVE: u32 = 23;

#[derive(Debug, Clone)]
pub struct Sym {
    pub name: String,
    pub value: u32,
    pub size: u32,
    pub info: u8,
    pub shndx: u16,
}

#[derive(Debug, Clone, Copy)]
struct Rel {
    offset: u32,
    sym: u32,
    kind: u32,
}

pub struct Module {
    pub name: String,
    pub base: u32,
    pub end: u32,
    pub needed: Vec<String>,
    pub soname: Option<String>,
    pub exports: HashMap<String, u32>,
    /// Names from a non-dynamic .symtab (present in the sample executables).
    pub statics: HashMap<String, u32>,
    pub init: Option<u32>,
    pub fini: Option<u32>,
    pub init_array: Vec<u32>,
    syms: Vec<Sym>,
    rels: Vec<Rel>,
    plt_rels: Vec<Rel>,
}

impl Module {
    pub fn symbol(&self, name: &str) -> Option<u32> {
        self.exports
            .get(name)
            .or_else(|| self.statics.get(name))
            .copied()
    }

    /// Names this module references but does not define.
    pub fn undefined_symbols(&self) -> impl Iterator<Item = &str> {
        self.syms
            .iter()
            .filter(|s| s.shndx == 0 && !s.name.is_empty())
            .map(|s| s.name.as_str())
    }

    pub fn exported_names(&self) -> impl Iterator<Item = &str> {
        self.exports.keys().map(|s| s.as_str())
    }

    /// Address -> nearest preceding exported symbol, for diagnostics.
    pub fn describe(&self, addr: u32) -> Option<(String, u32)> {
        let mut best: Option<(&str, u32)> = None;
        for s in &self.syms {
            if s.shndx == 0 || s.value == 0 {
                continue;
            }
            let a = self.base.wrapping_add(s.value);
            if a <= addr && best.map_or(true, |(_, b)| a > b) {
                best = Some((&s.name, a));
            }
        }
        best.map(|(n, a)| (n.to_string(), addr - a))
    }
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Allocates guest addresses for successive modules.
pub struct Loader {
    next_base: u32,
}

impl Loader {
    pub fn new(first_base: u32) -> Self {
        Loader {
            next_base: first_base,
        }
    }

    pub fn load(&mut self, mem: &mut Mem, name: &str, data: &[u8]) -> Result<Module, String> {
        if data.len() < 52 || &data[0..4] != b"\x7fELF" {
            return Err(format!("{name}: not an ELF file"));
        }
        if data[4] != 1 || data[5] != 1 {
            return Err(format!("{name}: expected 32-bit little-endian ELF"));
        }
        if u16le(data, 18) != 40 {
            return Err(format!("{name}: not EM_ARM"));
        }
        let e_type = u16le(data, 16);
        let e_phoff = u32le(data, 28) as usize;
        let e_phentsize = u16le(data, 42) as usize;
        let e_phnum = u16le(data, 44) as usize;

        let mut lo = u32::MAX;
        let mut hi = 0u32;
        for i in 0..e_phnum {
            let p = e_phoff + i * e_phentsize;
            if u32le(data, p) != PT_LOAD {
                continue;
            }
            let vaddr = u32le(data, p + 8);
            let memsz = u32le(data, p + 20);
            lo = lo.min(vaddr);
            hi = hi.max(vaddr.wrapping_add(memsz));
        }
        if lo == u32::MAX {
            return Err(format!("{name}: no PT_LOAD segments"));
        }

        let base = if e_type == 2 { 0 } else { self.next_base };
        let span = (hi - lo) as usize;
        let rounded = (span + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
        if e_type != 2 {
            self.next_base = self
                .next_base
                .wrapping_add(rounded as u32 + PAGE_SIZE as u32);
        }

        mem.map(base.wrapping_add(lo), rounded as u32 + PAGE_SIZE as u32);
        for i in 0..e_phnum {
            let p = e_phoff + i * e_phentsize;
            if u32le(data, p) != PT_LOAD {
                continue;
            }
            let offset = u32le(data, p + 4) as usize;
            let vaddr = base.wrapping_add(u32le(data, p + 8));
            let filesz = u32le(data, p + 16) as usize;
            let memsz = u32le(data, p + 20) as usize;
            if offset + filesz > data.len() {
                return Err(format!("{name}: segment {i} runs past end of file"));
            }
            mem.write_bytes(vaddr, &data[offset..offset + filesz])
                .map_err(|e| format!("{name}: {e}"))?;
            if memsz > filesz {
                mem.fill(vaddr + filesz as u32, memsz - filesz, 0)
                    .map_err(|e| format!("{name}: {e}"))?;
            }
        }

        // Locate PT_DYNAMIC and walk it out of guest memory (post-load, so the
        // addresses are already based).
        let mut dyn_addr = None;
        for i in 0..e_phnum {
            let p = e_phoff + i * e_phentsize;
            if u32le(data, p) == PT_DYNAMIC {
                dyn_addr = Some(base.wrapping_add(u32le(data, p + 8)));
            }
        }

        let mut m = Module {
            name: name.to_string(),
            base,
            end: base.wrapping_add(hi),
            needed: Vec::new(),
            soname: None,
            exports: HashMap::new(),
            statics: HashMap::new(),
            init: None,
            fini: None,
            init_array: Vec::new(),
            syms: Vec::new(),
            rels: Vec::new(),
            plt_rels: Vec::new(),
        };

        // .symtab entries are appended only after .dynsym has filled indices
        // 0..n, because relocations index the dynamic table by position.
        let Some(dyn_addr) = dyn_addr else {
            read_symtab(data, base, &mut m);
            return Ok(m);
        };

        let mut strtab = 0u32;
        let mut symtab = 0u32;
        let mut hash = 0u32;
        let mut rel = 0u32;
        let mut relsz = 0u32;
        let mut jmprel = 0u32;
        let mut pltrelsz = 0u32;
        let mut pltrel_kind = DT_REL;
        let mut needed_off = Vec::new();
        let mut soname_off = None;
        let mut init_array = 0u32;
        let mut init_arraysz = 0u32;

        let mut a = dyn_addr;
        loop {
            let tag = mem.read_u32(a).map_err(|e| format!("{name}: {e}"))?;
            let val = mem.read_u32(a + 4).map_err(|e| format!("{name}: {e}"))?;
            if tag == 0 {
                break;
            }
            match tag {
                DT_NEEDED => needed_off.push(val),
                DT_SONAME => soname_off = Some(val),
                DT_STRTAB => strtab = val,
                DT_SYMTAB => symtab = val,
                DT_HASH => hash = val,
                DT_REL => rel = val,
                DT_RELSZ => relsz = val,
                DT_JMPREL => jmprel = val,
                DT_PLTRELSZ => pltrelsz = val,
                DT_PLTREL => pltrel_kind = val,
                DT_INIT => m.init = Some(base.wrapping_add(val)),
                DT_FINI => m.fini = Some(base.wrapping_add(val)),
                DT_INIT_ARRAY => init_array = val,
                DT_INIT_ARRAYSZ => init_arraysz = val,
                DT_RELA | DT_RELASZ | DT_FINI_ARRAY | DT_FINI_ARRAYSZ => {}
                _ => {}
            }
            a += 8;
        }
        if pltrel_kind == DT_RELA {
            return Err(format!("{name}: RELA PLT relocations are not supported"));
        }

        // Some objects give DT_ addresses already based, some as link-time
        // vaddrs; normalise by checking whether the value falls inside the image.
        let fix = |v: u32| -> u32 {
            if v == 0 {
                0
            } else if v >= base && v < m.end {
                v
            } else {
                base.wrapping_add(v)
            }
        };
        let strtab = fix(strtab);
        let symtab = fix(symtab);
        let hash = fix(hash);
        let rel_a = fix(rel);
        let jmprel_a = fix(jmprel);
        let init_array_a = fix(init_array);

        let read_str = |mem: &Mem, off: u32| -> String {
            mem.read_cstring_lossy(strtab.wrapping_add(off))
                .unwrap_or_default()
        };

        for off in needed_off {
            m.needed.push(read_str(mem, off));
        }
        if let Some(off) = soname_off {
            m.soname = Some(read_str(mem, off));
        }

        // Symbol count comes from the hash table's nchain field.
        let nsyms = if hash != 0 {
            mem.read_u32(hash + 4).map_err(|e| format!("{name}: {e}"))?
        } else {
            0
        };
        for i in 0..nsyms {
            let o = symtab + i * 16;
            let st_name = mem.read_u32(o).map_err(|e| format!("{name}: {e}"))?;
            let st_value = mem.read_u32(o + 4).map_err(|e| format!("{name}: {e}"))?;
            let st_size = mem.read_u32(o + 8).map_err(|e| format!("{name}: {e}"))?;
            let st_info = mem.read_u8(o + 12).map_err(|e| format!("{name}: {e}"))?;
            let st_shndx = mem.read_u16(o + 14).map_err(|e| format!("{name}: {e}"))?;
            let nm = read_str(mem, st_name);
            if st_shndx != 0 && !nm.is_empty() {
                m.exports.insert(nm.clone(), base.wrapping_add(st_value));
            }
            m.syms.push(Sym {
                name: nm,
                value: st_value,
                size: st_size,
                info: st_info,
                shndx: st_shndx,
            });
        }

        let read_rels = |addr: u32, size: u32| -> Result<Vec<Rel>, String> {
            let mut out = Vec::new();
            let mut o = addr;
            let end = addr + size;
            while o < end {
                let offset = mem.read_u32(o).map_err(|e| format!("{name}: {e}"))?;
                let info = mem.read_u32(o + 4).map_err(|e| format!("{name}: {e}"))?;
                out.push(Rel {
                    offset,
                    sym: info >> 8,
                    kind: info & 0xFF,
                });
                o += 8;
            }
            Ok(out)
        };
        if rel_a != 0 {
            m.rels = read_rels(rel_a, relsz)?;
        }
        if jmprel_a != 0 {
            m.plt_rels = read_rels(jmprel_a, pltrelsz)?;
        }

        if init_array_a != 0 {
            for i in 0..(init_arraysz / 4) {
                let f = mem
                    .read_u32(init_array_a + i * 4)
                    .map_err(|e| format!("{name}: {e}"))?;
                if f != 0 && f != u32::MAX {
                    m.init_array.push(f);
                }
            }
        }

        read_symtab(data, base, &mut m);
        Ok(m)
    }
}

/// Apply a module's relocations. `resolve` supplies addresses for symbols this
/// module does not define itself.
pub fn relocate<F>(mem: &mut Mem, m: &Module, mut resolve: F) -> Result<Vec<String>, String>
where
    F: FnMut(&str) -> Option<u32>,
{
    let mut unresolved = Vec::new();
    let all = m.rels.iter().chain(m.plt_rels.iter());
    for r in all {
        let p = m.base.wrapping_add(r.offset);
        let sym = m.syms.get(r.sym as usize);

        // A COPY relocation must take the value from whoever else defines the
        // symbol; the executable's own definition is the destination.
        let is_copy = r.kind == R_ARM_COPY;
        let value = match sym {
            None => 0,
            Some(s) if s.shndx != 0 && !is_copy => m.base.wrapping_add(s.value),
            Some(s) if s.name.is_empty() => 0,
            Some(s) => match resolve(&s.name) {
                Some(v) => v,
                None => {
                    let weak = (s.info >> 4) == 2;
                    if !weak && !unresolved.contains(&s.name) {
                        unresolved.push(s.name.clone());
                    }
                    0
                }
            },
        };

        match r.kind {
            R_ARM_NONE => {}
            R_ARM_RELATIVE => {
                let a = mem.read_u32(p).map_err(|e| format!("{}: {e}", m.name))?;
                mem.write_u32(p, m.base.wrapping_add(a))
                    .map_err(|e| format!("{}: {e}", m.name))?;
            }
            R_ARM_ABS32 => {
                let a = mem.read_u32(p).map_err(|e| format!("{}: {e}", m.name))?;
                mem.write_u32(p, value.wrapping_add(a))
                    .map_err(|e| format!("{}: {e}", m.name))?;
            }
            R_ARM_REL32 => {
                let a = mem.read_u32(p).map_err(|e| format!("{}: {e}", m.name))?;
                mem.write_u32(p, value.wrapping_add(a).wrapping_sub(p))
                    .map_err(|e| format!("{}: {e}", m.name))?;
            }
            R_ARM_GLOB_DAT | R_ARM_JUMP_SLOT => {
                mem.write_u32(p, value)
                    .map_err(|e| format!("{}: {e}", m.name))?;
            }
            R_ARM_COPY => {
                if let Some(s) = sym {
                    if value == 0 || value == p {
                        continue;
                    }
                    let size = if s.size == 0 { 4 } else { s.size };
                    let bytes = mem
                        .read_bytes(value, size as usize)
                        .map_err(|e| format!("{}: {e}", m.name))?;
                    mem.write_bytes(p, &bytes)
                        .map_err(|e| format!("{}: {e}", m.name))?;
                }
            }
            R_ARM_PC24 => {
                let a = mem.read_u32(p).map_err(|e| format!("{}: {e}", m.name))?;
                let addend = (((a & 0x00FF_FFFF) << 8) as i32 >> 6) as u32;
                let off = value.wrapping_add(addend).wrapping_sub(p);
                let imm = (off >> 2) & 0x00FF_FFFF;
                mem.write_u32(p, (a & 0xFF00_0000) | imm)
                    .map_err(|e| format!("{}: {e}", m.name))?;
            }
            other => return Err(format!("{}: unsupported relocation type {other}", m.name)),
        }
    }
    Ok(unresolved)
}

/// Pull names out of a non-dynamic .symtab, straight from the file image.
fn read_symtab(data: &[u8], base: u32, m: &mut Module) {
    if data.len() < 52 {
        return;
    }
    let e_shoff = u32le(data, 32) as usize;
    let e_shentsize = u16le(data, 46) as usize;
    let e_shnum = u16le(data, 48) as usize;
    let e_shstrndx = u16le(data, 50) as usize;
    if e_shoff == 0 || e_shnum == 0 || e_shoff + e_shnum * e_shentsize > data.len() {
        return;
    }

    let sec = |i: usize| -> (u32, u32, u32, u32, u32) {
        let o = e_shoff + i * e_shentsize;
        (
            u32le(data, o),      // name
            u32le(data, o + 4),  // type
            u32le(data, o + 16), // offset
            u32le(data, o + 20), // size
            u32le(data, o + 24), // link
        )
    };
    if e_shstrndx >= e_shnum {
        return;
    }
    let shstr_off = sec(e_shstrndx).2 as usize;
    let name_at = |off: usize| -> String {
        let start = shstr_off + off;
        let end = data[start..].iter().position(|c| *c == 0).unwrap_or(0) + start;
        String::from_utf8_lossy(&data[start..end]).into_owned()
    };

    for i in 0..e_shnum {
        let (nm, ty, off, size, link) = sec(i);
        if ty != 2 || name_at(nm as usize) != ".symtab" {
            continue;
        }
        if link as usize >= e_shnum {
            continue;
        }
        let str_off = sec(link as usize).2 as usize;
        let count = (size / 16) as usize;
        for k in 0..count {
            let o = off as usize + k * 16;
            if o + 16 > data.len() {
                break;
            }
            let st_name = u32le(data, o) as usize;
            let st_value = u32le(data, o + 4);
            let st_size = u32le(data, o + 8);
            let st_info = data[o + 12];
            let st_shndx = u16le(data, o + 14);
            if st_shndx == 0 || st_name == 0 {
                continue;
            }
            let start = str_off + st_name;
            if start >= data.len() {
                continue;
            }
            let end = data[start..].iter().position(|c| *c == 0).unwrap_or(0) + start;
            let name = String::from_utf8_lossy(&data[start..end]).into_owned();
            if name.is_empty() {
                continue;
            }
            m.statics
                .entry(name.clone())
                .or_insert(base.wrapping_add(st_value));
            if (st_info & 0xF) == 2 {
                m.syms.push(Sym {
                    name,
                    value: st_value,
                    size: st_size,
                    info: st_info,
                    shndx: st_shndx,
                });
            }
        }
    }
}

use crate::snap::{Reader, Snap, SnapResult, Writer};

impl Snap for Sym {
    fn save(&self, w: &mut Writer) {
        w.put(&self.name);
        w.u32(self.value);
        w.u32(self.size);
        w.u8(self.info);
        w.u16(self.shndx);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(Sym {
            name: r.get()?,
            value: r.u32()?,
            size: r.u32()?,
            info: r.u8()?,
            shndx: r.u16()?,
        })
    }
}

impl Snap for Loader {
    fn save(&self, w: &mut Writer) {
        w.u32(self.next_base);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(Loader {
            next_base: r.u32()?,
        })
    }
}

/// Relocations are deliberately not carried. A module in a snapshot has
/// already been bound, and `Machine::link` never revisits it; a module loaded
/// afterwards resolves against `exports`, which is kept.
impl Snap for Module {
    fn save(&self, w: &mut Writer) {
        w.put(&self.name);
        w.u32(self.base);
        w.u32(self.end);
        w.put(&self.needed);
        w.put(&self.soname);
        w.put(&self.exports);
        w.put(&self.statics);
        w.put(&self.init);
        w.put(&self.fini);
        w.put(&self.init_array);
        w.put(&self.syms);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(Module {
            name: r.get()?,
            base: r.u32()?,
            end: r.u32()?,
            needed: r.get()?,
            soname: r.get()?,
            exports: r.get()?,
            statics: r.get()?,
            init: r.get()?,
            fini: r.get()?,
            init_array: r.get()?,
            syms: r.get()?,
            rels: Vec::new(),
            plt_rels: Vec::new(),
        })
    }
}
