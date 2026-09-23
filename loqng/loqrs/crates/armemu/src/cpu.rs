//! ARMv5TE interpreter (ARM state only; the Loquendo binaries contain no Thumb).

use crate::fpa::Fpa;
use std::collections::HashMap;

use crate::block::{BlockCache, Op};
use crate::mem::{Fault, Mem};

/// PC values in this page trap out to the host as call index `(pc - HOSTCALL_BASE) / 4`.
pub const HOSTCALL_BASE: u32 = 0xE100_0000;
pub const HOSTCALL_END: u32 = 0xE110_0000;
/// Returning here ends a `call()`.
pub const MAGIC_RETURN: u32 = 0xE0FF_FFF0;
/// A spawned thread returns here when its start routine finishes.
pub const MAGIC_THREAD_EXIT: u32 = 0xE0FF_FFE0;
/// Control transfers kept for post-mortem when `history` is on.
pub const BRANCH_HISTORY: usize = 96;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    HostCall(u32),
    Halted,
    ThreadExit,
    Breakpoint(u32),
    Watch { pc: u32, addr: u32, value: u32 },
    Undefined { pc: u32, insn: u32 },
    Swi { pc: u32, imm: u32 },
    Fault { pc: u32, fault: Fault },
    Budget,
}

#[derive(Clone)]
pub struct Cpu {
    pub r: [u32; 16],
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
    pub fpa: Fpa,
    pub icount: u64,
    /// Recent (from, to) control transfers; only filled when `history` is on.
    pub branches: Vec<(u32, u32)>,
    pub history: bool,
    pub branch_head: usize,
    pub breakpoints: Vec<u32>,
    /// Sampled PC histogram, filled when `profile` is on.
    pub profile: bool,
    pub samples: HashMap<u32, u64>,
    /// Instructions executed inside compiled blocks. Carried across thread
    /// switches by Machine::switch_to, like icount.
    pub jitted: u64,
    /// Of those, the ones that ran as emitted code rather than as a callback
    /// into the interpreter. This is the number the cost model turns on:
    /// `jitted` counts callbacks too, and a callback is slightly dearer than
    /// interpreting the same instruction outright.
    pub jit_native: u64,
    /// `icount` when this process took over, which for loqdave means when it
    /// restored the snapshot.
    ///
    /// `icount` is cumulative across the snapshot and `jitted` is not, so a
    /// coverage figure taken against `icount` silently counts the engine
    /// initialisation that ran in whichever process *built* the snapshot. That
    /// made a 99.6% look like 77.9%.
    pub icount_base: u64,
    bp_skip: u32,
    pub(crate) next: u32,
    pub(crate) pc_written: bool,
    /// Set by the compiled-block callback when the interpreter stopped;
    /// carries the `Stop` out through a return value that cannot hold one.
    /// Not snapshotted: transient, and never live across a block boundary.
    pub(crate) jit_stop: Option<Stop>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            r: [0; 16],
            n: false,
            z: false,
            c: false,
            v: false,
            fpa: Fpa::new(),
            icount: 0,
            branches: Vec::new(),
            history: false,
            branch_head: 0,
            breakpoints: Vec::new(),
            profile: false,
            samples: HashMap::new(),
            jitted: 0,
            jit_native: 0,
            icount_base: 0,
            bp_skip: u32::MAX,
            next: 0,
            pc_written: false,
            jit_stop: None,
        }
    }

    /// Instructions run since [`Cpu::icount_base`] — this process's own work,
    /// which is what a coverage figure has to be taken against.
    #[inline]
    pub fn ran(&self) -> u64 {
        self.icount - self.icount_base
    }

    #[inline(always)]
    pub fn set_reg(&mut self, i: u32, v: u32) {
        self.r[i as usize] = v;
        if i == 15 {
            self.pc_written = true;
        }
    }

    #[inline(always)]
    fn cond(&self, c: u32) -> bool {
        match c {
            0x0 => self.z,
            0x1 => !self.z,
            0x2 => self.c,
            0x3 => !self.c,
            0x4 => self.n,
            0x5 => !self.n,
            0x6 => self.v,
            0x7 => !self.v,
            0x8 => self.c && !self.z,
            0x9 => !self.c || self.z,
            0xA => self.n == self.v,
            0xB => self.n != self.v,
            0xC => !self.z && (self.n == self.v),
            0xD => self.z || (self.n != self.v),
            _ => true,
        }
    }

    #[inline(always)]
    fn set_nz(&mut self, v: u32) {
        self.n = (v & 0x8000_0000) != 0;
        self.z = v == 0;
    }

    /// Barrel shifter for register operands. Returns (value, carry_out).
    #[inline(always)]
    fn shift_reg(&self, insn: u32) -> (u32, bool) {
        let rm = (insn & 0xF) as usize;
        let typ = (insn >> 5) & 3;
        let by_reg = (insn & 0x10) != 0;
        let mut val = self.r[rm];
        let amount = if by_reg {
            // Rm reads as pc+12 when shifted by a register.
            if rm == 15 {
                val = val.wrapping_add(4);
            }
            self.r[((insn >> 8) & 0xF) as usize] & 0xFF
        } else {
            (insn >> 7) & 0x1F
        };

        if by_reg {
            if amount == 0 {
                return (val, self.c);
            }
            return match typ {
                0 => {
                    if amount < 32 {
                        (val << amount, (val >> (32 - amount)) & 1 != 0)
                    } else if amount == 32 {
                        (0, val & 1 != 0)
                    } else {
                        (0, false)
                    }
                }
                1 => {
                    if amount < 32 {
                        (val >> amount, (val >> (amount - 1)) & 1 != 0)
                    } else if amount == 32 {
                        (0, val & 0x8000_0000 != 0)
                    } else {
                        (0, false)
                    }
                }
                2 => {
                    if amount < 32 {
                        (
                            ((val as i32) >> amount) as u32,
                            (val >> (amount - 1)) & 1 != 0,
                        )
                    } else {
                        let s = val & 0x8000_0000 != 0;
                        (if s { 0xFFFF_FFFF } else { 0 }, s)
                    }
                }
                _ => {
                    let a = amount & 31;
                    if a == 0 {
                        (val, val & 0x8000_0000 != 0)
                    } else {
                        (val.rotate_right(a), (val >> (a - 1)) & 1 != 0)
                    }
                }
            };
        }

        match typ {
            0 => {
                if amount == 0 {
                    (val, self.c)
                } else {
                    (val << amount, (val >> (32 - amount)) & 1 != 0)
                }
            }
            1 => {
                if amount == 0 {
                    (0, val & 0x8000_0000 != 0)
                } else {
                    (val >> amount, (val >> (amount - 1)) & 1 != 0)
                }
            }
            2 => {
                if amount == 0 {
                    let s = val & 0x8000_0000 != 0;
                    (if s { 0xFFFF_FFFF } else { 0 }, s)
                } else {
                    (
                        ((val as i32) >> amount) as u32,
                        (val >> (amount - 1)) & 1 != 0,
                    )
                }
            }
            _ => {
                if amount == 0 {
                    // RRX
                    let cin = self.c as u32;
                    ((val >> 1) | (cin << 31), val & 1 != 0)
                } else {
                    (val.rotate_right(amount), (val >> (amount - 1)) & 1 != 0)
                }
            }
        }
    }

    #[inline(always)]
    fn operand2(&self, insn: u32) -> (u32, bool) {
        if insn & 0x0200_0000 != 0 {
            let imm = insn & 0xFF;
            let rot = ((insn >> 8) & 0xF) * 2;
            if rot == 0 {
                (imm, self.c)
            } else {
                let v = imm.rotate_right(rot);
                (v, v & 0x8000_0000 != 0)
            }
        } else {
            self.shift_reg(insn)
        }
    }

    pub fn run(&mut self, mem: &mut Mem, budget: u64) -> Stop {
        // The debug facilities cost four branches per instruction. Hoist the
        // decision out of the loop and let the monomorphised copy drop them.
        let debugging =
            self.profile || self.history || !self.breakpoints.is_empty() || !mem.watch.is_empty();
        if debugging {
            self.run_inner::<true>(mem, budget)
        } else {
            self.run_inner::<false>(mem, budget)
        }
    }

    /// Execute through the decoded-block cache.
    ///
    /// Semantically identical to `run_inner::<false>`: same `icount`, same PC
    /// bookkeeping, same stop conditions. What changes is where the work
    /// happens — the fetch, the `cond` and class extraction and the three PC
    /// guards are paid once per block instead of once per instruction.
    ///
    /// The debug facilities are deliberately absent: [`Cpu::run`] routes to the
    /// plain interpreter whenever profiling, tracing, breakpoints or
    /// watchpoints are active, so none of their behaviour changes and this
    /// path stays free of the branches they cost.
    pub fn run_cached(&mut self, mem: &mut Mem, blocks: &mut BlockCache, budget: u64) -> Stop {
        let mut left = budget;
        while left > 0 {
            let pc = self.r[15];
            if pc == MAGIC_RETURN {
                return Stop::Halted;
            }
            if pc == MAGIC_THREAD_EXIT {
                return Stop::ThreadExit;
            }
            if (HOSTCALL_BASE..HOSTCALL_END).contains(&pc) {
                return Stop::HostCall((pc - HOSTCALL_BASE) / 4);
            }

            let block = blocks.get(mem, pc);
            if block.is_empty() {
                // Nothing decodable here — an unmapped PC. Hand the rest of the
                // budget to the interpreter so the fault is reported exactly as
                // it always was.
                return self.run_inner::<false>(mem, left);
            }

            // A compiled block runs in full: native code where it could be
            // emitted, a callback into the interpreter everywhere else. It
            // returns how many guest instructions actually ran, which is short
            // of the block length only when it exited early.
            if let Some(c) = &block.code {
                if c.covered as u64 <= left {
                    let regs = self.r.as_mut_ptr();
                    let cpu = self as *mut Cpu as *mut u8;
                    let memp = mem as *mut Mem as *mut u8;
                    // Low half: instructions run. High half: how many of
                    // them were emitted code rather than a callback.
                    let ret = unsafe { c.run(regs, cpu, memp) };
                    let ran = ret & 0xFFFF_FFFF;
                    self.icount += ran;
                    self.jitted += ran;
                    self.jit_native += ret >> 32;
                    left -= ran;
                    if let Some(s) = self.jit_stop.take() {
                        return s;
                    }
                    // r15 is already where it belongs, set either by the
                    // epilogue or by the callback that left the block.
                    continue;
                }
            }

            let mut at = pc;
            for op in block.ops.iter() {
                let Op::Raw { insn } = *op;
                self.icount += 1;
                self.r[15] = at.wrapping_add(8);
                self.next = at.wrapping_add(4);
                self.pc_written = false;
                if let Err(s) = self.exec(insn, at, mem) {
                    self.r[15] = at;
                    return s;
                }
                if !self.pc_written {
                    self.r[15] = self.next;
                }
                left -= 1;
                // The static terminator analysis only sizes blocks; THIS is the
                // correctness guarantee. A PC that did not simply advance by
                // four leaves the block, so a missed terminator costs a short
                // block and never wrong behaviour.
                if self.r[15] != at.wrapping_add(4) || left == 0 {
                    break;
                }
                at = at.wrapping_add(4);
            }
        }
        Stop::Budget
    }

    fn run_inner<const DEBUG: bool>(&mut self, mem: &mut Mem, budget: u64) -> Stop {
        // Straight-line code fetches thousands of instructions from one page,
        // so resolve the page once and index into it until the PC leaves.
        let mut fetch_page = u32::MAX;
        let mut fetch_base: *const u8 = std::ptr::null();

        for _ in 0..budget {
            let pc = self.r[15];
            if pc == MAGIC_RETURN {
                return Stop::Halted;
            }
            if pc == MAGIC_THREAD_EXIT {
                return Stop::ThreadExit;
            }
            if (HOSTCALL_BASE..HOSTCALL_END).contains(&pc) {
                return Stop::HostCall((pc - HOSTCALL_BASE) / 4);
            }
            if DEBUG && !self.breakpoints.is_empty() {
                if pc == self.bp_skip {
                    self.bp_skip = u32::MAX;
                } else if self.breakpoints.contains(&pc) {
                    self.bp_skip = pc;
                    return Stop::Breakpoint(pc);
                }
            }
            let off = (pc & crate::mem::PAGE_MASK) as usize;
            let insn = if off <= crate::mem::PAGE_SIZE - 4 {
                let page = pc >> crate::mem::PAGE_BITS;
                if page != fetch_page {
                    fetch_base = mem.page_base(pc);
                    if fetch_base.is_null() {
                        return Stop::Fault {
                            pc,
                            fault: Fault::Unmapped(pc),
                        };
                    }
                    fetch_page = page;
                }
                u32::from_le(unsafe { (fetch_base.add(off) as *const u32).read_unaligned() })
            } else {
                // Only reachable from a misaligned PC, which means the guest
                // has already gone wrong; let the slow path report it.
                match mem.read_u32(pc) {
                    Ok(i) => i,
                    Err(f) => return Stop::Fault { pc, fault: f },
                }
            };
            self.icount += 1;
            if DEBUG && self.profile && self.icount & 0xFF == 0 {
                *self.samples.entry(pc).or_insert(0) += 1;
            }
            self.r[15] = pc.wrapping_add(8);
            self.next = pc.wrapping_add(4);
            self.pc_written = false;

            if let Err(s) = self.exec(insn, pc, mem) {
                self.r[15] = pc;
                return s;
            }
            if !self.pc_written {
                self.r[15] = self.next;
            }
            if DEBUG && !mem.watch.is_empty() {
                for i in 0..mem.watch.len() {
                    let a = mem.watch[i];
                    let now = mem.read_u32(a).unwrap_or(0);
                    if now != mem.watch_shadow[i] {
                        mem.watch_shadow[i] = now;
                        return Stop::Watch {
                            pc,
                            addr: a,
                            value: now,
                        };
                    }
                }
            }
            if DEBUG && self.history && self.r[15] != pc.wrapping_add(4) {
                let to = self.r[15];
                if self.branches.len() < BRANCH_HISTORY {
                    self.branches.push((pc, to));
                } else {
                    self.branches[self.branch_head] = (pc, to);
                }
                self.branch_head = (self.branch_head + 1) % BRANCH_HISTORY;
            }
        }
        Stop::Budget
    }

    #[inline(always)]
    pub(crate) fn exec(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        let cond = insn >> 28;
        if cond == 0xF {
            // v5 unconditional space: BLX (immediate) is the only form GCC 3.3 emits.
            if insn & 0xFE00_0000 == 0xFA00_0000 {
                let off = (((insn & 0x00FF_FFFF) as i32) << 8 >> 8) as u32;
                let h = (insn >> 24) & 1;
                self.r[14] = pc.wrapping_add(4);
                self.next = pc
                    .wrapping_add(8)
                    .wrapping_add(off << 2)
                    .wrapping_add(h << 1);
                return Ok(());
            }
            // PLD and friends: no architectural effect here.
            return Ok(());
        }
        if !self.cond(cond) {
            return Ok(());
        }

        let class = (insn >> 25) & 7;
        match class {
            0 | 1 => self.exec_dp_and_misc(insn, pc, mem),
            2 | 3 => self.exec_ldr_str(insn, pc, mem),
            4 => self.exec_ldm_stm(insn, pc, mem),
            5 => {
                let off = (((insn & 0x00FF_FFFF) as i32) << 8 >> 8) as u32;
                if insn & 0x0100_0000 != 0 {
                    self.r[14] = pc.wrapping_add(4);
                }
                self.next = pc.wrapping_add(8).wrapping_add(off << 2);
                Ok(())
            }
            6 => self.exec_cpdt(insn, pc, mem),
            _ => {
                if insn & 0x0100_0000 != 0 {
                    return Err(Stop::Swi {
                        pc,
                        imm: insn & 0x00FF_FFFF,
                    });
                }
                if insn & 0x10 != 0 {
                    self.exec_cprt(insn, pc)
                } else {
                    self.exec_cpdo(insn, pc)
                }
            }
        }
    }

    fn exec_dp_and_misc(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        // Multiplies, BX/BLX, CLZ, halfword/signed loads and PSR transfers live
        // in the data-processing encoding space; peel them off first.
        if insn & 0x0200_0000 == 0 {
            if insn & 0x0FFF_FFF0 == 0x012F_FF10 {
                let t = self.r[(insn & 0xF) as usize];
                self.next = t & !1;
                return Ok(());
            }
            if insn & 0x0FFF_FFF0 == 0x012F_FF30 {
                let t = self.r[(insn & 0xF) as usize];
                self.r[14] = pc.wrapping_add(4);
                self.next = t & !1;
                return Ok(());
            }
            if insn & 0x0FF0_00F0 == 0x0160_0010 {
                let v = self.r[(insn & 0xF) as usize];
                self.set_reg((insn >> 12) & 0xF, v.leading_zeros());
                return Ok(());
            }
            if insn & 0x0FC0_00F0 == 0x0000_0090 {
                return self.exec_mul(insn);
            }
            if insn & 0x0F80_00F0 == 0x0080_0090 {
                return self.exec_mull(insn);
            }
            if insn & 0x0FB0_0FF0 == 0x0100_0090 {
                return self.exec_swp(insn, pc, mem);
            }
            if insn & 0x0E00_0090 == 0x0000_0090 && insn & 0x60 != 0 {
                return self.exec_halfword(insn, pc, mem);
            }
            if insn & 0x0FBF_0FFF == 0x010F_0000 {
                let v = self.cpsr();
                self.set_reg((insn >> 12) & 0xF, v);
                return Ok(());
            }
            if insn & 0x0FB0_FFF0 == 0x0120_F000 {
                let v = self.r[(insn & 0xF) as usize];
                self.set_cpsr(v);
                return Ok(());
            }
            if insn & 0x0F90_0090 == 0x0100_0080 || insn & 0x0F90_0FF0 == 0x0100_0050 {
                return self.exec_dsp(insn);
            }
        } else if insn & 0x0FB0_0000 == 0x0320_0000 {
            // MSR immediate / hint space.
            return Ok(());
        }

        let op = (insn >> 21) & 0xF;
        let s = insn & 0x0010_0000 != 0;
        let rn = (insn >> 16) & 0xF;
        let rd = (insn >> 12) & 0xF;
        let (op2, shift_c) = self.operand2(insn);
        let a = self.r[rn as usize];

        let (result, write, carry, overflow) = match op {
            0x0 => (a & op2, true, shift_c, self.v),
            0x1 => (a ^ op2, true, shift_c, self.v),
            0x2 => sub_flags(a, op2, true),
            0x3 => sub_flags(op2, a, true),
            0x4 => add_flags(a, op2, false),
            0x5 => add_flags(a, op2, self.c),
            0x6 => sub_flags(a, op2, self.c),
            0x7 => sub_flags(op2, a, self.c),
            0x8 => (a & op2, false, shift_c, self.v),
            0x9 => (a ^ op2, false, shift_c, self.v),
            0xA => {
                let (r, _, c, v) = sub_flags(a, op2, true);
                (r, false, c, v)
            }
            0xB => {
                let (r, _, c, v) = add_flags(a, op2, false);
                (r, false, c, v)
            }
            0xC => (a | op2, true, shift_c, self.v),
            0xD => (op2, true, shift_c, self.v),
            0xE => (a & !op2, true, shift_c, self.v),
            _ => (!op2, true, shift_c, self.v),
        };

        if s {
            self.set_nz(result);
            self.c = carry;
            self.v = overflow;
        }
        if write {
            self.set_reg(rd, result);
            if rd == 15 {
                self.next = result;
            }
        }
        Ok(())
    }

    fn exec_mul(&mut self, insn: u32) -> Result<(), Stop> {
        let rd = (insn >> 16) & 0xF;
        let rn = (insn >> 12) & 0xF;
        let rs = (insn >> 8) & 0xF;
        let rm = insn & 0xF;
        let mut res = self.r[rm as usize].wrapping_mul(self.r[rs as usize]);
        if insn & 0x0020_0000 != 0 {
            res = res.wrapping_add(self.r[rn as usize]);
        }
        self.set_reg(rd, res);
        if insn & 0x0010_0000 != 0 {
            self.set_nz(res);
        }
        Ok(())
    }

    fn exec_mull(&mut self, insn: u32) -> Result<(), Stop> {
        let rdhi = (insn >> 16) & 0xF;
        let rdlo = (insn >> 12) & 0xF;
        let rs = self.r[((insn >> 8) & 0xF) as usize];
        let rm = self.r[(insn & 0xF) as usize];
        let signed = insn & 0x0040_0000 != 0;
        let accumulate = insn & 0x0020_0000 != 0;

        let mut res: u64 = if signed {
            ((rm as i32 as i64).wrapping_mul(rs as i32 as i64)) as u64
        } else {
            (rm as u64).wrapping_mul(rs as u64)
        };
        if accumulate {
            let acc = ((self.r[rdhi as usize] as u64) << 32) | self.r[rdlo as usize] as u64;
            res = res.wrapping_add(acc);
        }
        self.set_reg(rdlo, res as u32);
        self.set_reg(rdhi, (res >> 32) as u32);
        if insn & 0x0010_0000 != 0 {
            self.n = res & 0x8000_0000_0000_0000 != 0;
            self.z = res == 0;
        }
        Ok(())
    }

    /// QADD/QSUB and the SMLAxy family. GCC 3.3 emits these only rarely.
    fn exec_dsp(&mut self, insn: u32) -> Result<(), Stop> {
        let op = (insn >> 21) & 3;
        if insn & 0x90 == 0x50 {
            let rm = self.r[(insn & 0xF) as usize] as i32;
            let rn = self.r[((insn >> 16) & 0xF) as usize] as i32;
            let res = match op {
                0 => rm.saturating_add(rn),
                1 => rm.saturating_sub(rn),
                2 => rm.saturating_add(rn.saturating_mul(2)),
                _ => rm.saturating_sub(rn.saturating_mul(2)),
            };
            self.set_reg((insn >> 12) & 0xF, res as u32);
            return Ok(());
        }
        let x = (insn >> 5) & 1;
        let y = (insn >> 6) & 1;
        let rm = self.r[(insn & 0xF) as usize];
        let rs = self.r[((insn >> 8) & 0xF) as usize];
        let a = if x == 0 {
            rm as u16 as i16
        } else {
            (rm >> 16) as u16 as i16
        } as i32;
        let b = if y == 0 {
            rs as u16 as i16
        } else {
            (rs >> 16) as u16 as i16
        } as i32;
        let rn = (insn >> 12) & 0xF;
        let rd = (insn >> 16) & 0xF;
        match op {
            0 => {
                let prod = a.wrapping_mul(b);
                self.set_reg(rd, prod.wrapping_add(self.r[rn as usize] as i32) as u32);
            }
            1 => {
                let prod = if insn & 0x20 != 0 {
                    a.wrapping_mul(b)
                } else {
                    ((self.r[(insn & 0xF) as usize] as i32 as i64 * rs as i32 as i64) >> 16) as i32
                };
                self.set_reg(rd, prod as u32);
            }
            2 => {
                let acc = ((self.r[rd as usize] as u64) << 32) | self.r[rn as usize] as u64;
                let res = (acc as i64).wrapping_add(a.wrapping_mul(b) as i64) as u64;
                self.set_reg(rn, res as u32);
                self.set_reg(rd, (res >> 32) as u32);
            }
            _ => {
                self.set_reg(rd, a.wrapping_mul(b) as u32);
            }
        }
        Ok(())
    }

    fn exec_swp(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        let addr = self.r[((insn >> 16) & 0xF) as usize];
        let rm = self.r[(insn & 0xF) as usize];
        let rd = (insn >> 12) & 0xF;
        if insn & 0x0040_0000 != 0 {
            let old = mem
                .read_u8(addr)
                .map_err(|f| Stop::Fault { pc, fault: f })?;
            mem.write_u8(addr, rm as u8)
                .map_err(|f| Stop::Fault { pc, fault: f })?;
            self.set_reg(rd, old as u32);
        } else {
            let old = mem
                .read_u32(addr)
                .map_err(|f| Stop::Fault { pc, fault: f })?;
            mem.write_u32(addr, rm)
                .map_err(|f| Stop::Fault { pc, fault: f })?;
            self.set_reg(rd, old);
        }
        Ok(())
    }

    fn exec_halfword(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        let p = insn & 0x0100_0000 != 0;
        let u = insn & 0x0080_0000 != 0;
        let w = insn & 0x0020_0000 != 0;
        let load = insn & 0x0010_0000 != 0;
        let rn = (insn >> 16) & 0xF;
        let rd = (insn >> 12) & 0xF;

        let offset = if insn & 0x0040_0000 != 0 {
            ((insn >> 4) & 0xF0) | (insn & 0xF)
        } else {
            self.r[(insn & 0xF) as usize]
        };

        let base = self.r[rn as usize];
        let offaddr = if u {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
        let addr = if p { offaddr } else { base };

        let kind = (insn >> 5) & 3;
        if load {
            let val = match kind {
                1 => mem
                    .read_u16(addr)
                    .map_err(|f| Stop::Fault { pc, fault: f })? as u32,
                2 => mem
                    .read_u8(addr)
                    .map_err(|f| Stop::Fault { pc, fault: f })? as i8 as i32
                    as u32,
                _ => mem
                    .read_u16(addr)
                    .map_err(|f| Stop::Fault { pc, fault: f })? as i16 as i32
                    as u32,
            };
            if !p || w {
                self.set_reg(rn, offaddr);
            }
            self.set_reg(rd, val);
            if rd == 15 {
                self.next = val;
            }
        } else {
            // Only STRH exists here (kind == 1); STRD/LDRD are v5TE and unused by GCC 3.3.
            let v = self.r[rd as usize] as u16;
            mem.write_u16(addr, v)
                .map_err(|f| Stop::Fault { pc, fault: f })?;
            if !p || w {
                self.set_reg(rn, offaddr);
            }
        }
        Ok(())
    }

    fn exec_ldr_str(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        if insn & 0x0200_0000 != 0 && insn & 0x10 != 0 {
            return Err(Stop::Undefined { pc, insn });
        }
        let p = insn & 0x0100_0000 != 0;
        let u = insn & 0x0080_0000 != 0;
        let byte = insn & 0x0040_0000 != 0;
        let w = insn & 0x0020_0000 != 0;
        let load = insn & 0x0010_0000 != 0;
        let rn = (insn >> 16) & 0xF;
        let rd = (insn >> 12) & 0xF;

        let offset = if insn & 0x0200_0000 != 0 {
            self.shift_reg(insn).0
        } else {
            insn & 0xFFF
        };
        let base = self.r[rn as usize];
        let offaddr = if u {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
        let addr = if p { offaddr } else { base };

        if load {
            let val = if byte {
                mem.read_u8(addr)
                    .map_err(|f| Stop::Fault { pc, fault: f })? as u32
            } else {
                // Unaligned word loads rotate, as on ARMv5.
                let raw = mem
                    .read_u32(addr & !3)
                    .map_err(|f| Stop::Fault { pc, fault: f })?;
                raw.rotate_right((addr & 3) * 8)
            };
            if !p || w {
                self.set_reg(rn, offaddr);
            }
            self.set_reg(rd, val);
            if rd == 15 {
                self.next = val & !1;
            }
        } else {
            let val = if rd == 15 {
                pc.wrapping_add(8)
            } else {
                self.r[rd as usize]
            };
            if byte {
                mem.write_u8(addr, val as u8)
                    .map_err(|f| Stop::Fault { pc, fault: f })?;
            } else {
                mem.write_u32(addr & !3, val)
                    .map_err(|f| Stop::Fault { pc, fault: f })?;
            }
            if !p || w {
                self.set_reg(rn, offaddr);
            }
        }
        Ok(())
    }

    fn exec_ldm_stm(&mut self, insn: u32, pc: u32, mem: &mut Mem) -> Result<(), Stop> {
        let p = insn & 0x0100_0000 != 0;
        let u = insn & 0x0080_0000 != 0;
        let w = insn & 0x0020_0000 != 0;
        let load = insn & 0x0010_0000 != 0;
        let rn = (insn >> 16) & 0xF;
        let list = insn & 0xFFFF;
        let count = list.count_ones();
        let base = self.r[rn as usize];

        // Compute the lowest transfer address; registers always go low-to-high.
        let start = if u {
            if p {
                base.wrapping_add(4)
            } else {
                base
            }
        } else if p {
            base.wrapping_sub(count * 4)
        } else {
            base.wrapping_sub(count * 4).wrapping_add(4)
        };
        let writeback = if u {
            base.wrapping_add(count * 4)
        } else {
            base.wrapping_sub(count * 4)
        };

        let mut addr = start;
        if load {
            // Base writeback happens before the loads land, matching ARM's ordering
            // when the base register is itself in the list.
            if w {
                self.set_reg(rn, writeback);
            }
            for i in 0..16u32 {
                if list & (1 << i) == 0 {
                    continue;
                }
                let v = mem
                    .read_u32(addr)
                    .map_err(|f| Stop::Fault { pc, fault: f })?;
                self.set_reg(i, v);
                if i == 15 {
                    self.next = v & !1;
                }
                addr = addr.wrapping_add(4);
            }
        } else {
            for i in 0..16u32 {
                if list & (1 << i) == 0 {
                    continue;
                }
                let v = if i == 15 {
                    pc.wrapping_add(8)
                } else if i == rn && w && list & ((1 << i) - 1) != 0 {
                    writeback
                } else {
                    self.r[i as usize]
                };
                mem.write_u32(addr, v)
                    .map_err(|f| Stop::Fault { pc, fault: f })?;
                addr = addr.wrapping_add(4);
            }
            if w {
                self.set_reg(rn, writeback);
            }
        }
        Ok(())
    }

    pub fn cpsr(&self) -> u32 {
        // User mode, ARM state, interrupts enabled.
        ((self.n as u32) << 31)
            | ((self.z as u32) << 30)
            | ((self.c as u32) << 29)
            | ((self.v as u32) << 28)
            | 0x10
    }

    pub fn set_cpsr(&mut self, v: u32) {
        self.n = v & 0x8000_0000 != 0;
        self.z = v & 0x4000_0000 != 0;
        self.c = v & 0x2000_0000 != 0;
        self.v = v & 0x1000_0000 != 0;
    }
}

#[inline(always)]
fn add_flags(a: u32, b: u32, carry_in: bool) -> (u32, bool, bool, bool) {
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(carry_in as u32);
    let carry = c1 || c2;
    let overflow = ((a ^ s2) & (b ^ s2) & 0x8000_0000) != 0;
    (s2, true, carry, overflow)
}

#[inline(always)]
fn sub_flags(a: u32, b: u32, carry_in: bool) -> (u32, bool, bool, bool) {
    // a - b - !carry_in, i.e. ARM's SUB/SBC with carry as NOT-borrow.
    let nb = !b;
    let (s1, c1) = a.overflowing_add(nb);
    let (s2, c2) = s1.overflowing_add(carry_in as u32);
    let carry = c1 || c2;
    let overflow = ((a ^ b) & (a ^ s2) & 0x8000_0000) != 0;
    (s2, true, carry, overflow)
}

use crate::snap::{Reader, Snap, SnapResult, Writer};

impl Snap for Cpu {
    fn save(&self, w: &mut Writer) {
        w.put(&self.r);
        w.put(&self.n);
        w.put(&self.z);
        w.put(&self.c);
        w.put(&self.v);
        w.put(&self.fpa);
        w.u64(self.icount);
        w.put(&self.branches);
        w.put(&self.history);
        w.put(&self.branch_head);
        w.put(&self.breakpoints);
        w.put(&self.profile);
        w.put(&self.samples);
        w.u32(self.bp_skip);
        w.u32(self.next);
        w.put(&self.pc_written);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        let mut cpu = Cpu {
            r: r.get()?,
            n: r.get()?,
            z: r.get()?,
            c: r.get()?,
            v: r.get()?,
            fpa: r.get()?,
            icount: r.u64()?,
            branches: r.get()?,
            history: r.get()?,
            branch_head: r.get()?,
            breakpoints: r.get()?,
            profile: r.get()?,
            samples: r.get()?,
            bp_skip: r.u32()?,
            next: r.u32()?,
            pc_written: r.get()?,
            // Not snapshotted: a statistic, not guest state. Adding it to the
            // wire format would invalidate every existing snapshot.
            jitted: 0,
            jit_native: 0,
            icount_base: 0,
            jit_stop: None,
        };
        cpu.icount_base = cpu.icount;
        Ok(cpu)
    }
}
