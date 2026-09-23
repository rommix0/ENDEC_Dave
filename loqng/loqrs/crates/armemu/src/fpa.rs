//! FPA11 floating-point coprocessor (cp1/cp2), as GCC 3.3 emits it for ARM OABI.
//!
//! Two layout facts drive this file:
//!   * doubles in memory are word-swapped (most significant word at the lower
//!     address) - the classic ARM OABI "mixed endian" double;
//!   * LFM/SFM move 12 bytes per register and are only ever used to spill and
//!     restore f4-f7, so the extended slot uses our own private layout.

use crate::cpu::{Cpu, Stop};
use crate::mem::Mem;

const PREC_S: u32 = 0;
const PREC_D: u32 = 1;
const PREC_E: u32 = 2;

/// FPA immediate constants selected by the Fm field when bit 3 is set.
const CONSTANTS: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 10.0];

/// Marker stored in the spare word of an extended slot.
const EXT_TAG: u32 = 0x464C_4F51;

#[derive(Clone)]
pub struct Fpa {
    pub f: [f64; 8],
    pub fpsr: u32,
    /// Counts instructions we approximate, for diagnostics.
    pub directed_rounding_ops: u64,
}

impl Default for Fpa {
    fn default() -> Self {
        Self::new()
    }
}

impl Fpa {
    pub fn new() -> Self {
        Fpa {
            f: [0.0; 8],
            fpsr: 0,
            directed_rounding_ops: 0,
        }
    }
}

#[inline]
fn round_to(prec: u32, x: f64) -> f64 {
    if prec == PREC_S {
        x as f32 as f64
    } else {
        x
    }
}

impl Cpu {
    /// LDF/STF (cp1) and LFM/SFM (cp2).
    pub(crate) fn exec_cpdt(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        let cp = (insn >> 8) & 0xF;
        if cp != 1 && cp != 2 {
            return Err(Stop::Undefined { pc, insn });
        }
        let p = insn & 0x0100_0000 != 0;
        let u = insn & 0x0080_0000 != 0;
        let w = insn & 0x0020_0000 != 0;
        let load = insn & 0x0010_0000 != 0;
        let rn = (insn >> 16) & 0xF;
        let fd = ((insn >> 12) & 7) as usize;
        let sel = (((insn >> 22) & 1) << 1) | ((insn >> 15) & 1);

        let base = self.r[rn as usize];
        let offset = (insn & 0xFF) * 4;
        let offaddr = if u {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
        let mut addr = if p { offaddr } else { base };

        let fault = |f| Stop::Fault { pc, fault: f };

        if cp == 1 {
            match sel {
                PREC_S => {
                    if load {
                        let bits = mem.read_u32(addr).map_err(fault)?;
                        self.fpa.f[fd] = f32::from_bits(bits) as f64;
                    } else {
                        mem.write_u32(addr, (self.fpa.f[fd] as f32).to_bits())
                            .map_err(fault)?;
                    }
                }
                PREC_D => {
                    if load {
                        let hi = mem.read_u32(addr).map_err(fault)?;
                        let lo = mem.read_u32(addr.wrapping_add(4)).map_err(fault)?;
                        self.fpa.f[fd] = f64::from_bits(((hi as u64) << 32) | lo as u64);
                    } else {
                        let bits = self.fpa.f[fd].to_bits();
                        mem.write_u32(addr, (bits >> 32) as u32).map_err(fault)?;
                        mem.write_u32(addr.wrapping_add(4), bits as u32)
                            .map_err(fault)?;
                    }
                }
                PREC_E => {
                    if load {
                        self.fpa.f[fd] = read_ext(mem, addr).map_err(fault)?;
                    } else {
                        write_ext(mem, addr, self.fpa.f[fd]).map_err(fault)?;
                    }
                }
                _ => return Err(Stop::Undefined { pc, insn }),
            }
        } else {
            // LFM/SFM: {bit22,bit15} gives the register count, 0 meaning four.
            let count = if sel == 0 { 4 } else { sel as usize };
            for i in 0..count {
                let reg = (fd + i) & 7;
                if load {
                    self.fpa.f[reg] = read_ext(mem, addr).map_err(fault)?;
                } else {
                    write_ext(mem, addr, self.fpa.f[reg]).map_err(fault)?;
                }
                addr = addr.wrapping_add(12);
            }
        }

        if !p || w {
            self.set_reg(rn, offaddr);
        }
        Ok(())
    }

    /// Arithmetic (CPDO) on cp1.
    pub(crate) fn exec_cpdo(&mut self, insn: u32, pc: u32) -> Result<(), Stop> {
        if (insn >> 8) & 0xF != 1 {
            return Err(Stop::Undefined { pc, insn });
        }
        let prec = (((insn >> 19) & 1) << 1) | ((insn >> 7) & 1);
        if prec == 3 {
            return Err(Stop::Undefined { pc, insn });
        }
        if (insn >> 5) & 3 != 0 {
            self.fpa.directed_rounding_ops += 1;
        }
        let fd = ((insn >> 12) & 7) as usize;
        let fn_ = ((insn >> 16) & 7) as usize;
        let m = if insn & 8 != 0 {
            CONSTANTS[(insn & 7) as usize]
        } else {
            self.fpa.f[(insn & 7) as usize]
        };
        let sel = (insn >> 20) & 0xF;
        let monadic = insn & 0x0000_8000 != 0;

        let res = if monadic {
            match sel {
                0x0 => m,         // MVF
                0x1 => -m,        // MNF
                0x2 => m.abs(),   // ABS
                0x3 => m.trunc(), // RND (round to integral, toward zero here)
                0x4 => m.sqrt(),  // SQT
                0x5 => m.log10(), // LOG
                0x6 => m.ln(),    // LGN
                0x7 => m.exp(),   // EXP
                0x8 => m.sin(),   // SIN
                0x9 => m.cos(),   // COS
                0xA => m.tan(),   // TAN
                0xB => m.asin(),  // ASN
                0xC => m.acos(),  // ACS
                0xD => m.atan(),  // ATN
                0xE => m.round(), // URD
                _ => m,           // NRM
            }
        } else {
            let n = self.fpa.f[fn_];
            match sel {
                0x0 => n + m,      // ADF
                0x1 => n * m,      // MUF
                0x2 => n - m,      // SUF
                0x3 => m - n,      // RSF
                0x4 => n / m,      // DVF
                0x5 => m / n,      // RDF
                0x6 => n.powf(m),  // POW
                0x7 => m.powf(n),  // RPW
                0x8 => n % m,      // RMF
                0x9 => n * m,      // FML (fast multiply, single only)
                0xA => n / m,      // FDV
                0xB => m / n,      // FRD
                0xC => n.atan2(m), // POL
                _ => return Err(Stop::Undefined { pc, insn }),
            }
        };

        self.fpa.f[fd] = round_to(prec, res);
        Ok(())
    }

    /// Register transfers and comparisons (CPRT) on cp1.
    pub(crate) fn exec_cprt(&mut self, insn: u32, pc: u32) -> Result<(), Stop> {
        if (insn >> 8) & 0xF != 1 {
            return Err(Stop::Undefined { pc, insn });
        }
        let sel = (insn >> 20) & 0xF;
        let rd = (insn >> 12) & 0xF;

        match sel {
            0x0 => {
                // FLT: integer -> float register (Fn field).
                let prec = (((insn >> 19) & 1) << 1) | ((insn >> 7) & 1);
                let fnr = ((insn >> 16) & 7) as usize;
                let v = self.r[rd as usize] as i32 as f64;
                self.fpa.f[fnr] = round_to(prec, v);
            }
            0x1 => {
                // FIX: float -> integer, honouring the instruction's rounding mode.
                let fm = (insn & 7) as usize;
                let x = self.fpa.f[fm];
                let r = match (insn >> 5) & 3 {
                    0 => round_half_even(x),
                    1 => x.ceil(),
                    2 => x.floor(),
                    _ => x.trunc(),
                };
                let clamped = if r.is_nan() {
                    0
                } else if r >= 2147483647.0 {
                    i32::MAX
                } else if r <= -2147483648.0 {
                    i32::MIN
                } else {
                    r as i32
                };
                self.set_reg(rd, clamped as u32);
            }
            0x2 => self.fpa.fpsr = self.r[rd as usize],
            0x3 => {
                let v = self.fpa.fpsr;
                self.set_reg(rd, v);
            }
            0x4 | 0x5 => {
                // FPCR is privileged and unused by compiler output.
                if sel == 0x5 {
                    self.set_reg(rd, 0);
                }
            }
            0x9 | 0xB | 0xD | 0xF => {
                let a = self.fpa.f[((insn >> 16) & 7) as usize];
                let mut b = if insn & 8 != 0 {
                    CONSTANTS[(insn & 7) as usize]
                } else {
                    self.fpa.f[(insn & 7) as usize]
                };
                if sel == 0xB || sel == 0xF {
                    b = -b; // CNF/CNFE compare against the negated operand
                }
                if a.is_nan() || b.is_nan() {
                    self.n = false;
                    self.z = false;
                    self.c = true;
                    self.v = true;
                } else {
                    self.n = a < b;
                    self.z = a == b;
                    self.c = a >= b;
                    self.v = false;
                }
            }
            _ => return Err(Stop::Undefined { pc, insn }),
        }
        Ok(())
    }
}

#[inline]
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

fn read_ext(mem: &Mem, addr: u32) -> Result<f64, crate::mem::Fault> {
    let lo = mem.read_u32(addr)?;
    let hi = mem.read_u32(addr.wrapping_add(4))?;
    Ok(f64::from_bits(((hi as u64) << 32) | lo as u64))
}

fn write_ext(mem: &mut Mem, addr: u32, v: f64) -> Result<(), crate::mem::Fault> {
    let bits = v.to_bits();
    mem.write_u32(addr, bits as u32)?;
    mem.write_u32(addr.wrapping_add(4), (bits >> 32) as u32)?;
    mem.write_u32(addr.wrapping_add(8), EXT_TAG)
}

use crate::snap::{Reader, Snap, SnapResult, Writer};

impl Snap for Fpa {
    fn save(&self, w: &mut Writer) {
        w.put(&self.f);
        w.u32(self.fpsr);
        w.u64(self.directed_rounding_ops);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(Fpa {
            f: r.get()?,
            fpsr: r.u32()?,
            directed_rounding_ops: r.u64()?,
        })
    }
}
