//! Basic-block cache: decode each guest instruction once, not once per execution.
//!
//! The interpreter's per-instruction cost is dominated by work that does not
//! depend on the CPU state: fetching the word, extracting `cond`, extracting
//! the class, dispatching on it, and re-extracting the operand fields inside
//! the handler. All of that is a pure function of the instruction, so it is
//! done once here and the result cached against the guest PC.
//!
//! Blocks end at the first instruction that can move the PC anywhere other
//! than `pc + 4`. That analysis is only an optimisation: the executor also
//! checks after every instruction that the PC really did advance by four, and
//! leaves the block if it did not. So a missed terminator costs a short block,
//! never wrong behaviour — which is what lets the cache be introduced without
//! re-deriving the whole instruction set up front.
//!
//! Cache invalidation is deliberately blunt: [`BlockCache::flush`] drops
//! everything, and is called wherever guest code can change (native veneer
//! patching, module load, snapshot restore). The engine's modules are fixed
//! images that do not rewrite themselves, so nothing finer is needed; if that
//! assumption ever breaks, the byte-exactness harness is what catches it.

use std::rc::Rc;

use crate::mem::Mem;

/// A decoded guest instruction.
///
/// `Raw` defers to the interpreter's own handler, so an unspecialised opcode
/// still benefits from the cached fetch and the hoisted PC guards. Specialised
/// variants are added as profiling justifies them.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    /// Not specialised: hand `insn` to `Cpu::exec`.
    Raw { insn: u32 },
}

/// A straight-line run of decoded instructions starting at `start`.
pub struct Block {
    pub start: u32,
    pub ops: Vec<Op>,
    /// Native code for the longest emittable PREFIX of `ops`, if any.
    ///
    /// A prefix rather than the whole block: every block ends at a control
    /// transfer by construction, and those are not emitted yet, so requiring
    /// full coverage would compile almost nothing. The interpreter picks up at
    /// `covered`.
    pub code: Option<Compiled>,
}

impl Block {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Longest block we will build. Bounds translation latency for a cold run and
/// keeps one runaway scan from walking an entire module.
const MAX_BLOCK: usize = 256;

/// True when `insn` can write the PC, i.e. it ends a block.
///
/// Conservative by construction: anything uncertain is treated as a terminator.
fn terminates(insn: u32) -> bool {
    let cond = insn >> 28;
    if cond == 0xF {
        // The v5 unconditional space that GCC 3.3 emits here is BLX(imm).
        return insn & 0xFE00_0000 == 0xFA00_0000;
    }
    let class = (insn >> 25) & 7;
    match class {
        // Data processing: terminates when it writes r15, and BX/BLX live in
        // this encoding space too.
        0 | 1 => {
            if insn & 0x0FFF_FFF0 == 0x012F_FF10 || insn & 0x0FFF_FFF0 == 0x012F_FF30 {
                return true;
            }
            (insn >> 12) & 0xF == 15
        }
        // Single data transfer: terminates when the destination is r15.
        2 | 3 => (insn >> 12) & 0xF == 15,
        // Block transfer: terminates when r15 is in the register list.
        4 => insn & 0x8000 != 0,
        // B / BL.
        5 => true,
        // Coprocessor data transfer never writes r15.
        6 => false,
        // SWI and coprocessor register transfers: treat as terminators.
        _ => true,
    }
}

/// Decode one instruction. Everything is `Raw` for now; this is the seam where
/// specialised variants get added.
#[inline]
fn decode(insn: u32) -> Op {
    Op::Raw { insn }
}

/// Build the block starting at `pc`.
///
/// Stops at the first terminator, at a page boundary (the next page may not be
/// mapped, and fetching it here would fault at translation time rather than
/// where the guest would have), or at `MAX_BLOCK`.
pub fn translate(mem: &Mem, pc: u32) -> Block {
    let mut ops = Vec::new();
    let mut at = pc;
    while ops.len() < MAX_BLOCK {
        // Do not cross a page boundary: the next page's mapping is not our
        // business at translation time.
        if (at & crate::mem::PAGE_MASK) as usize > crate::mem::PAGE_SIZE - 4 {
            break;
        }
        let insn = match mem.read_u32(at) {
            Ok(i) => i,
            // Unmapped: stop short and let the interpreter fault at the right
            // place, with the right PC.
            Err(_) => break,
        };
        ops.push(decode(insn));
        if terminates(insn) {
            break;
        }
        at = at.wrapping_add(4);
    }
    let code = compile(&ops, pc, mem.ptrs_base());
    if std::env::var_os("LOQ_JIT_STATS").is_some() {
        let (nat, tot) = code
            .as_ref()
            .map(|c| (c.native, c.covered))
            .unwrap_or((0, ops.len()));
        eprintln!(
            "[jit] block 0x{pc:08x} {tot:3} ops, {nat:3} native ({:.0}%){}",
            if tot == 0 {
                0.0
            } else {
                100.0 * nat as f64 / tot as f64
            },
            if code.is_some() { "" } else { "  REJECTED" }
        );
    }
    Block {
        start: pc,
        ops,
        code,
    }
}

/// Guest PC -> decoded block.
///
/// Direct-mapped rather than hashed. Block exits land every few instructions
/// in branchy code, so the lookup is hot: hashing a `u32` and probing a
/// `HashMap` measured *slower* than the interpreter it replaced. Indexing an
/// array by the PC bits and comparing one tag is a handful of instructions.
///
/// A collision simply replaces the resident block, which costs a retranslation
/// and nothing else — there is no correctness content in the cache.
pub struct BlockCache {
    slots: Box<[Option<Rc<Block>>]>,
    /// Address of the page-pointer table the resident blocks were compiled
    /// against. Emitted memory accesses bake it in as an immediate, so blocks
    /// from one `Mem` must never run against another; checking it here makes
    /// that structural rather than a comment. It changes at most once per
    /// `Mem`, so the flush it triggers is not a hot path.
    ptrs: usize,
    /// Blocks translated since the last flush.
    pub translated: u64,
    /// Block lookups served from a resident block.
    pub hits: u64,
}

/// 64K slots: covers the engine's hot code many times over, and costs 512 KiB
/// of pointers on a 64-bit host.
const CACHE_BITS: usize = 16;
const CACHE_SIZE: usize = 1 << CACHE_BITS;
const CACHE_MASK: usize = CACHE_SIZE - 1;

impl Default for BlockCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockCache {
    pub fn new() -> Self {
        BlockCache {
            slots: vec![None; CACHE_SIZE].into_boxed_slice(),
            ptrs: 0,
            translated: 0,
            hits: 0,
        }
    }

    #[inline(always)]
    fn slot(pc: u32) -> usize {
        (pc >> 2) as usize & CACHE_MASK
    }

    /// Fetch the block at `pc`, translating it if it is not resident.
    ///
    /// Returns an `Rc` so the caller can execute it while holding `&mut Cpu`.
    #[inline]
    pub fn get(&mut self, mem: &Mem, pc: u32) -> Rc<Block> {
        let ptrs = mem.ptrs_base();
        if ptrs != self.ptrs {
            self.flush();
            self.ptrs = ptrs;
        }
        let i = Self::slot(pc);
        if let Some(b) = &self.slots[i] {
            if b.start == pc {
                self.hits += 1;
                return b.clone();
            }
        }
        self.translated += 1;
        let b = Rc::new(translate(mem, pc));
        self.slots[i] = Some(b.clone());
        b
    }

    /// Instructions executed per block entry, for tuning. Diagnostic only.
    pub fn report(&self, icount: u64) {
        let entries = self.hits + self.translated;
        if entries == 0 {
            return;
        }
        eprintln!(
            "[block] {} entries ({} translated, {} hits), {:.1} insns/entry",
            entries,
            self.translated,
            self.hits,
            icount as f64 / entries as f64
        );
    }

    /// Drop every cached block. Call wherever guest code may have changed.
    pub fn flush(&mut self) {
        for s in self.slots.iter_mut() {
            *s = None;
        }
    }
}

// ---------------------------------------------------------------- backend --

#[cfg(all(unix, target_arch = "x86_64"))]
mod backend {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::Op;
    use crate::x64::{Emitter, ExecBuf, SET_C, SET_NC, SET_O, SET_S, SET_Z};

    /// Whether to keep the rejection histogram. Read once, at first use.
    static STATS: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("LOQ_JIT_STATS").is_some());

    /// Emitted memory accesses hard-code the 64 KiB page geometry: `shr 16`
    /// for the page index and `movzx cx` for the offset. Both would be silently
    /// wrong if the address space were ever re-tiled.
    const _: () = assert!(crate::mem::PAGE_BITS == 16);
    const _: () = assert!(crate::mem::NUM_PAGES == 1 << 16);

    /// Native code for a block prefix, plus how many guest instructions it runs.
    pub struct Compiled {
        buf: ExecBuf,
        /// Guest instructions in the block.
        pub covered: usize,
        /// How many of them are native rather than callbacks.
        pub native: usize,
    }

    impl Compiled {
        /// Run the block. The low half of the result is how many guest
        /// instructions executed, which is `covered` unless it exited early;
        /// the high half is how many of those ran natively.
        ///
        /// # Safety
        /// `regs` must point at the 16-word guest register file, and `cpu` and
        /// `mem` at the live `Cpu` and `Mem` the callback will operate on.
        #[inline(always)]
        pub unsafe fn run(&self, regs: *mut u32, cpu: *mut u8, mem: *mut u8) -> u64 {
            (self.buf.as_fn())(regs, cpu, mem)
        }
    }

    /// Interpreter callback for one instruction inside a compiled block.
    ///
    /// Reproduces `run_inner`'s per-instruction contract exactly: r15 reads as
    /// `pc + 8`, `next` is the fall-through, and the PC is only advanced when
    /// the instruction did not write it.
    ///
    /// Returns 0 when the PC simply advanced, 1 when the block must be left —
    /// either the PC went elsewhere, or the interpreter stopped, in which case
    /// the `Stop` is parked on the `Cpu` because a `u64` cannot carry one.
    unsafe extern "sysv64" fn step(cpu: *mut u8, mem: *mut u8, insn: u32, pc: u32) -> u64 {
        if *STATS {
            REJECTED[classify(insn) as usize].fetch_add(1, Ordering::Relaxed);
        }
        let cpu = &mut *(cpu as *mut crate::cpu::Cpu);
        let mem = &mut *(mem as *mut crate::mem::Mem);
        cpu.r[15] = pc.wrapping_add(8);
        cpu.next = pc.wrapping_add(4);
        cpu.pc_written = false;
        if let Err(s) = cpu.exec(insn, pc, mem) {
            cpu.r[15] = pc;
            cpu.jit_stop = Some(s);
            return 1;
        }
        if !cpu.pc_written {
            cpu.r[15] = cpu.next;
        }
        u64::from(cpu.r[15] != pc.wrapping_add(4))
    }

    // ------------------------------------------------- rejection histogram
    //
    // Every instruction the backend could not emit passes through `step`, so
    // classifying there is exact and execution-weighted — no sampling, and no
    // per-instruction cost on the fast path, because the instruction is
    // already paying for a call. Instructions in blocks that failed the cost
    // model are not counted: the question this answers is what to emit next
    // to empty the callbacks out of the blocks we do compile.

    /// The first reason the backend's acceptance test turned an instruction
    /// down. Ordered as the test itself checks, so "fixing this unlocks that
    /// share" reads correctly.
    #[derive(Clone, Copy)]
    pub enum Why {
        UncondSpace,
        Branch,
        Shifted,
        SetsFlags,
        Mul32,
        Mul64,
        Halfword,
        MemShifted,
        MemPc,
        BlockTransfer,
        Coproc,
        Other,
    }

    pub const WHY: [(&str, Why); 12] = [
        ("v5 unconditional space", Why::UncondSpace),
        ("branch", Why::Branch),
        ("shift amount in a register", Why::Shifted),
        ("sets flags", Why::SetsFlags),
        ("multiply, 32-bit", Why::Mul32),
        ("multiply, 64-bit", Why::Mul64),
        ("halfword / signed transfer", Why::Halfword),
        ("ldr/str, shifted offset", Why::MemShifted),
        ("ldr/str, touches pc", Why::MemPc),
        ("ldm/stm", Why::BlockTransfer),
        ("coprocessor", Why::Coproc),
        ("other", Why::Other),
    ];

    static REJECTED: [AtomicU64; 12] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    /// Counts per [`WHY`] slot. Diagnostic; only filled when stats are on.
    pub fn rejected() -> [u64; 12] {
        std::array::from_fn(|i| REJECTED[i].load(Ordering::Relaxed))
    }

    fn classify(insn: u32) -> Why {
        if insn >> 28 == 0xF {
            return Why::UncondSpace;
        }
        match (insn >> 25) & 7 {
            0 => {
                if insn & 0x0FFF_FFF0 == 0x012F_FF10 || insn & 0x0FFF_FFF0 == 0x012F_FF30 {
                    Why::Branch // bx / blx
                } else if insn & 0x90 == 0x90 {
                    // The same masks `exec_dp_and_misc` peels these off with.
                    if insn & 0x0FC0_00F0 == 0x0000_0090 {
                        Why::Mul32
                    } else if insn & 0x0F80_00F0 == 0x0080_0090 {
                        Why::Mul64
                    } else if insn & 0x0E00_0090 == 0x0000_0090 && insn & 0x60 != 0 {
                        Why::Halfword
                    } else {
                        Why::Other
                    }
                } else if insn & (1 << 20) != 0 {
                    Why::SetsFlags
                } else if insn & 0x10 != 0 {
                    Why::Shifted
                } else {
                    Why::Other
                }
            }
            1 => {
                if insn & (1 << 20) != 0 {
                    Why::SetsFlags
                } else {
                    Why::Other
                }
            }
            2 | 3 => {
                if (insn >> 25) & 1 != 0 && (insn >> 4) & 0xFF != 0 {
                    Why::MemShifted
                } else {
                    Why::MemPc
                }
            }
            4 => Why::BlockTransfer,
            5 => Why::Branch,
            6 => Why::Coproc,
            _ => Why::Other,
        }
    }

    /// Times a memory fast path gave up and fell back to the interpreter.
    static BAILS: AtomicU64 = AtomicU64::new(0);

    /// How often an inlined access bailed. A fast path that bails often is a
    /// fast path that is not paying for itself, so this is worth watching
    /// whenever the emitted set widens. Diagnostic only.
    pub fn bail_count() -> u64 {
        BAILS.load(Ordering::Relaxed)
    }

    /// The slow-path stub of an inlined memory access: the same callback as
    /// [`step`], counted.
    ///
    /// The fast path jumps here having touched nothing but host scratch
    /// registers, so re-executing the whole instruction is correct.
    unsafe extern "sysv64" fn step_slow(cpu: *mut u8, mem: *mut u8, insn: u32, pc: u32) -> u64 {
        BAILS.fetch_add(1, Ordering::Relaxed);
        step(cpu, mem, insn, pc)
    }

    /// ARM data-processing opcodes this backend emits.
    const AND: u32 = 0x0;
    const EOR: u32 = 0x1;
    const SUB: u32 = 0x2;
    const RSB: u32 = 0x3;
    const ADD: u32 = 0x4;
    const TST: u32 = 0x8;
    const TEQ: u32 = 0x9;
    const CMP: u32 = 0xA;
    const CMN: u32 = 0xB;
    const ORR: u32 = 0xC;
    const MOV: u32 = 0xD;
    const BIC: u32 = 0xE;
    const MVN: u32 = 0xF;

    /// Decode the rotated 8-bit immediate of a data-processing instruction.
    #[inline]
    fn dp_imm(insn: u32) -> u32 {
        (insn & 0xFF).rotate_right(((insn >> 8) & 0xF) * 2)
    }

    /// Emit one instruction, or return false if this backend cannot.
    ///
    /// Deliberately narrow. Only the immediate form (class 1) is accepted,
    /// because the register form shares its encoding space with multiplies,
    /// BX, CLZ, PSR transfers and halfword loads, and telling those apart is
    /// not worth it before the common case is proven.
    ///
    /// The caller has already emitted the condition test, so `cond` is not
    /// this function's business.
    ///
    /// Excluded on purpose:
    /// * the S bit — needs NZCV computed and written back;
    /// * `rd == 15` — a PC write ends the block anyway;
    /// * `rn == 15` — reading the PC yields `pc + 8`, which the interpreter
    ///   maintains per instruction and compiled code does not.
    fn emit(e: &mut Emitter, insn: u32) -> bool {
        let class = (insn >> 25) & 7;
        if class == 0 {
            // The register form shares this space with multiplies, BX/BLX,
            // CLZ, PSR transfers and halfword/signed loads. Peel every one of
            // them off before trusting the data-processing reading.
            if insn & 0x0FFF_FFF0 == 0x012F_FF10 || insn & 0x0FFF_FFF0 == 0x012F_FF30 {
                return false; // BX / BLX
            }
            if insn & 0x90 == 0x90 {
                return false; // multiply, or halfword/signed transfer
            }
            if insn & 0x10 != 0 {
                return false; // shift amount taken from a register
            }
        } else if class != 1 {
            return false;
        }
        if insn & (1 << 20) != 0 {
            return false; // sets flags
        }
        let opc = (insn >> 21) & 0xF;
        let rn = ((insn >> 16) & 0xF) as u8;
        let rd = ((insn >> 12) & 0xF) as u8;
        if rd == 15 || rn == 15 {
            return false;
        }
        if class == 1 {
            let imm = dp_imm(insn);
            match opc {
                MOV => e.store_imm(rd, imm),
                MVN => e.store_imm(rd, !imm),
                RSB => {
                    e.mov_eax(imm);
                    e.sub_reg(rn);
                    e.store_reg(rd);
                }
                AND | EOR | SUB | ADD | ORR | BIC => {
                    e.load_reg(rn);
                    match opc {
                        AND => e.and_eax(imm),
                        EOR => e.xor_eax(imm),
                        SUB => e.sub_eax(imm),
                        ADD => e.add_eax(imm),
                        ORR => e.or_eax(imm),
                        BIC => e.and_eax(!imm),
                        _ => unreachable!(),
                    }
                    e.store_reg(rd);
                }
                // ADC/SBC/RSC need carry in; TST/TEQ/CMP/CMN are flag-only.
                _ => return false,
            }
            return true;
        }

        let rm = (insn & 0xF) as u8;
        if rm == 15 {
            return false; // reading the PC needs pc + 8
        }

        if (insn >> 4) & 0xFF != 0 {
            // Shifted register operand. op2 is built in `ecx` first, so the
            // opcodes below take it from there instead of from the register
            // file — which is why they do not simply share the code above.
            let typ = (insn >> 5) & 3;
            let amount = (insn >> 7) & 0x1F;
            if typ == 3 && amount == 0 {
                return false; // rrx needs the carry flag
            }
            if !matches!(opc, AND | EOR | SUB | RSB | ADD | ORR | MOV | BIC | MVN) {
                return false; // checked before emitting, not after
            }
            e.load_ecx(rm);
            e.shift_ecx(typ, amount);
            match opc {
                MOV => e.store_ecx(rd),
                MVN => {
                    e.not_ecx();
                    e.store_ecx(rd);
                }
                BIC => {
                    e.not_ecx();
                    e.load_reg(rn);
                    e.and_eax_ecx();
                    e.store_reg(rd);
                }
                RSB => {
                    e.mov_eax_ecx();
                    e.sub_reg(rn);
                    e.store_reg(rd);
                }
                AND | EOR | SUB | ADD | ORR => {
                    e.load_reg(rn);
                    match opc {
                        AND => e.and_eax_ecx(),
                        EOR => e.xor_eax_ecx(),
                        SUB => e.sub_eax_ecx(),
                        ADD => e.add_eax_ecx(),
                        ORR => e.or_eax_ecx(),
                        _ => unreachable!(),
                    }
                    e.store_reg(rd);
                }
                _ => unreachable!(),
            }
            return true;
        }

        // Unshifted register operand: op2 is simply Rm.
        match opc {
            MOV => {
                e.load_reg(rm);
                e.store_reg(rd);
            }
            MVN => {
                e.load_reg(rm);
                e.not_eax();
                e.store_reg(rd);
            }
            BIC => {
                e.load_reg(rm);
                e.not_eax();
                e.and_reg(rn);
                e.store_reg(rd);
            }
            RSB => {
                e.load_reg(rm);
                e.sub_reg(rn);
                e.store_reg(rd);
            }
            AND | EOR | SUB | ADD | ORR => {
                e.load_reg(rn);
                match opc {
                    AND => e.and_reg(rm),
                    EOR => e.xor_reg(rm),
                    SUB => e.sub_reg(rm),
                    ADD => e.add_reg(rm),
                    ORR => e.or_reg(rm),
                    _ => unreachable!(),
                }
                e.store_reg(rd);
            }
            _ => return false,
        }
        true
    }

    // --------------------------------------------------------- conditions
    //
    // Flag *generation* is still left to the interpreter — the S bit is not
    // emitted — but that does not stop conditional instructions being emitted,
    // because whoever set the flags stored them on the `Cpu` where compiled
    // code can read them. Conditionals were the single largest class of
    // callback before this: 11.9% of all instructions executed.

    const OFF_N: u32 = std::mem::offset_of!(crate::cpu::Cpu, n) as u32;
    const OFF_Z: u32 = std::mem::offset_of!(crate::cpu::Cpu, z) as u32;
    const OFF_C: u32 = std::mem::offset_of!(crate::cpu::Cpu, c) as u32;
    const OFF_V: u32 = std::mem::offset_of!(crate::cpu::Cpu, v) as u32;

    /// `jz` / `jnz`, as the second byte of a `0F 8x` near jump.
    const JZ: u8 = 0x84;
    const JNZ: u8 = 0x85;

    /// Emit the test for ARM condition `cond`, jumping past the body when it
    /// fails. Returns the sites to patch to just after the body.
    ///
    /// `None` for the `0xF` encoding space, which is not a condition at all.
    /// Rust guarantees a `bool` is 0 or 1, so the flags compare as plain bytes.
    fn emit_cond(e: &mut Emitter, cond: u32) -> Option<Vec<usize>> {
        // Skip the body when the flag is clear (the condition wanted it set).
        fn on_flag(e: &mut Emitter, off: u32, want_set: bool) -> Vec<usize> {
            e.cmp_flag(off);
            vec![e.jcc(if want_set { JZ } else { JNZ })]
        }
        // Skip the body when `n == v` differs from `want_equal`.
        fn on_nv(e: &mut Emitter, want_equal: bool) -> usize {
            e.mov_al_flag(OFF_N);
            e.cmp_al_flag(OFF_V);
            e.jcc(if want_equal { JNZ } else { JZ })
        }

        Some(match cond {
            0x0 => on_flag(e, OFF_Z, true),  // eq
            0x1 => on_flag(e, OFF_Z, false), // ne
            0x2 => on_flag(e, OFF_C, true),  // cs
            0x3 => on_flag(e, OFF_C, false), // cc
            0x4 => on_flag(e, OFF_N, true),  // mi
            0x5 => on_flag(e, OFF_N, false), // pl
            0x6 => on_flag(e, OFF_V, true),  // vs
            0x7 => on_flag(e, OFF_V, false), // vc
            // hi: c && !z — skip if either half fails.
            0x8 => {
                e.cmp_flag(OFF_C);
                let a = e.jcc(JZ);
                e.cmp_flag(OFF_Z);
                let b = e.jcc(JNZ);
                vec![a, b]
            }
            // ls: !c || z — skip only if c is set and z is clear.
            0x9 => {
                e.cmp_flag(OFF_C);
                let run = e.jcc(JZ);
                e.cmp_flag(OFF_Z);
                let skip = e.jcc(JZ);
                e.patch(run);
                vec![skip]
            }
            0xA => vec![on_nv(e, true)],  // ge: n == v
            0xB => vec![on_nv(e, false)], // lt: n != v
            // gt: !z && n == v
            0xC => {
                e.cmp_flag(OFF_Z);
                let a = e.jcc(JNZ);
                let b = on_nv(e, true);
                vec![a, b]
            }
            // le: z || n != v
            0xD => {
                e.cmp_flag(OFF_Z);
                let run = e.jcc(JNZ);
                let skip = on_nv(e, false);
                e.patch(run);
                vec![skip]
            }
            0xE => Vec::new(), // al
            _ => return None,  // the v5 unconditional space
        })
    }

    /// A deferred slow-path stub for one inlined memory access.
    struct Bail {
        /// rel32 sites in the fast path that jump here.
        sites: Vec<usize>,
        insn: u32,
        pc: u32,
        /// Instructions to report on return — this one included.
        ///
        /// Reporting the count *before* it would be a hang, not a slowdown:
        /// the caller would re-enter at the same PC, translate a block opening
        /// with the same access, and bail again forever. The stub executes the
        /// instruction so the block always makes progress.
        count: u32,
        /// How many of the instructions before this one ran natively. This
        /// one did not: it is about to go through the interpreter.
        native: u32,
    }

    /// The offset added to (or subtracted from) the base register.
    enum Off {
        Imm(u32),
        Reg(u8),
        /// `Rm` shifted by an immediate. Kept apart from `Reg` so the common
        /// unshifted case stays one instruction.
        Shifted(u8, u32, u32),
    }

    /// Emit one single data transfer: `ldr`/`str`, word or byte, with an
    /// immediate or unshifted register offset.
    ///
    /// Returns the patch sites that must be pointed at this instruction's
    /// slow-path stub, or `None` if the form is not emitted.
    ///
    /// The guest architecture's own masking is what keeps this short. A word
    /// access goes to `addr & !3` and a page is 64 KiB, so an aligned word
    /// never straddles two pages; a byte access never can either. There is
    /// therefore no page-crossing check anywhere — only a null page pointer,
    /// and, for word loads, an alignment test, because an unaligned word load
    /// rotates on ARMv5 and that is left to the interpreter.
    ///
    /// The caller has already emitted the condition test, so `cond` is not
    /// this function's business.
    ///
    /// Excluded on purpose:
    /// * `rd == 15` — a PC write, and a PC read for `str`;
    /// * a write-back to `r15` — emitted code must never touch the PC;
    /// * shifted register offsets — not emitted yet, like every other shift.
    fn emit_mem(e: &mut Emitter, insn: u32, pc: u32, ptrs: usize) -> Option<Vec<usize>> {
        let class = (insn >> 25) & 7;
        if class != 2 && class != 3 {
            return None;
        }
        let reg_off = class == 3;
        if reg_off && insn & 0x10 != 0 {
            return None; // undefined in this encoding space
        }

        let pre = insn & 0x0100_0000 != 0;
        let up = insn & 0x0080_0000 != 0;
        let byte = insn & 0x0040_0000 != 0;
        let load = insn & 0x0010_0000 != 0;
        let rn = ((insn >> 16) & 0xF) as u8;
        let rd = ((insn >> 12) & 0xF) as u8;
        // Matches the interpreter exactly, which means LDRT/STRT (`!pre && w`)
        // write back like the post-indexed form they are encoded as.
        let writeback = !pre || insn & 0x0020_0000 != 0;

        if rd == 15 || (rn == 15 && writeback) {
            return None;
        }

        let off = if reg_off {
            let rm = (insn & 0xF) as u8;
            if rm == 15 {
                return None; // reads pc + 8
            }
            if (insn >> 4) & 0xFF == 0 {
                Off::Reg(rm)
            } else {
                let typ = (insn >> 5) & 3;
                let amount = (insn >> 7) & 0x1F;
                if typ == 3 && amount == 0 {
                    return None; // rrx needs the carry flag
                }
                Off::Shifted(rm, typ, amount)
            }
        } else {
            Off::Imm(insn & 0xFFF)
        };

        // Nothing below can fail, so nothing is emitted until here.
        let mut bails = Vec::new();

        // ecx = the address accessed; esi = the write-back value, if any.
        emit_addr(e, rn, pc, pre, up, writeback, off);

        if load && !byte {
            bails.push(e.bail_unaligned());
        }
        bails.push(e.page_lookup(ptrs));

        if load {
            // A word load reached here aligned, so the offset needs no mask.
            if byte {
                e.mem_load_byte();
            } else {
                e.mem_load_word();
            }
            // The write-back lands before the destination, so `ldr rN, [rN], #k`
            // keeps the loaded value — which is what the interpreter does.
            if writeback {
                e.store_esi(rn);
            }
            e.store_reg(rd);
        } else {
            if !byte {
                e.align_offset();
            }
            // The stored value is read before the write-back, so clobbering
            // the address register with it here is safe.
            e.load_ecx(rd);
            if byte {
                e.mem_store_byte();
            } else {
                e.mem_store_word();
            }
            if writeback {
                e.store_esi(rn);
            }
        }
        Some(bails)
    }

    /// Emit a flag-setting data-processing instruction.
    ///
    /// The flags are read straight back out of x86's own: `sets`/`setz` for N
    /// and Z, and for an arithmetic op `seto` for V and `setc` for C — except
    /// after a subtraction, where ARM's C is "not borrow" and x86's CF is the
    /// borrow, so it is `setae`. A logical op leaves V alone and takes C from
    /// the barrel shifter, which for an unshifted operand means leaving that
    /// alone too.
    ///
    /// Every PSR transfer, multiply, swap and halfword form sharing this
    /// encoding space has the S bit clear or bit 4 set, so requiring S and
    /// rejecting bit 4 excludes all of them.
    ///
    /// Excluded on purpose:
    /// * `adc`/`sbc`/`rsc` — they need the carry *in*, which means getting the
    ///   guest's C into x86's CF first;
    /// * a logical op with a shifted operand — its C is the shifter's carry
    ///   out rather than the ALU's, which is separate work.
    fn emit_s(e: &mut Emitter, insn: u32) -> bool {
        let class = (insn >> 25) & 7;
        if class == 0 {
            if insn & 0x10 != 0 {
                return false; // multiply, halfword, swp, bx, register shift
            }
        } else if class != 1 {
            return false;
        }
        if insn & 0x0010_0000 == 0 {
            return false; // flags untouched; `emit` handles those
        }

        let opc = (insn >> 21) & 0xF;
        let rn = ((insn >> 16) & 0xF) as u8;
        let rd = ((insn >> 12) & 0xF) as u8;
        let arith = matches!(opc, SUB | RSB | ADD | CMP | CMN);
        if !arith && !matches!(opc, AND | EOR | ORR | BIC | MOV | MVN | TST | TEQ) {
            return false; // adc / sbc / rsc
        }
        let writes = !matches!(opc, TST | TEQ | CMP | CMN);
        let reads_rn = !matches!(opc, MOV | MVN);
        if (writes && rd == 15) || (reads_rn && rn == 15) {
            return false;
        }

        // op2 into ecx, plus what a logical op should do to C. An arithmetic
        // op takes C from the ALU and ignores this.
        let mut set_carry = None;
        if class == 1 {
            let v = dp_imm(insn);
            if (insn >> 8) & 0xF != 0 {
                // A rotated immediate carries out its top bit — a constant.
                set_carry = Some(v >> 31 != 0);
            }
            e.mov_ecx(v);
        } else {
            let rm = (insn & 0xF) as u8;
            if rm == 15 {
                return false; // reads pc + 8
            }
            let typ = (insn >> 5) & 3;
            let amount = (insn >> 7) & 0x1F;
            let shifted = (insn >> 4) & 0xFF != 0;
            if shifted && (!arith || (typ == 3 && amount == 0)) {
                return false;
            }
            e.load_ecx(rm);
            if shifted {
                e.shift_ecx(typ, amount);
            }
        }

        // The result in eax, with x86's flags set by the operation itself.
        match opc {
            ADD | CMN => {
                e.load_reg(rn);
                e.add_eax_ecx();
            }
            SUB | CMP => {
                e.load_reg(rn);
                e.sub_eax_ecx();
            }
            RSB => {
                e.mov_eax_ecx();
                e.sub_reg(rn);
            }
            AND | TST => {
                e.load_reg(rn);
                e.and_eax_ecx();
            }
            EOR | TEQ => {
                e.load_reg(rn);
                e.xor_eax_ecx();
            }
            ORR => {
                e.load_reg(rn);
                e.or_eax_ecx();
            }
            BIC => {
                e.not_ecx();
                e.load_reg(rn);
                e.and_eax_ecx();
            }
            // `mov` and `not` leave the flags alone, so N and Z need drawing
            // out with an explicit test.
            MOV => {
                e.mov_eax_ecx();
                e.test_eax();
            }
            _ => {
                e.mov_eax_ecx();
                e.not_eax();
                e.test_eax();
            }
        }

        e.setcc_flag(SET_S, OFF_N);
        e.setcc_flag(SET_Z, OFF_Z);
        if arith {
            let c = if matches!(opc, SUB | RSB | CMP) {
                SET_NC
            } else {
                SET_C
            };
            e.setcc_flag(c, OFF_C);
            e.setcc_flag(SET_O, OFF_V);
        } else if let Some(b) = set_carry {
            e.mov_flag_imm(OFF_C, b);
        }
        if writes {
            e.store_reg(rd);
        }
        true
    }

    /// Emit the address computation every transfer shares: the address in
    /// `ecx`, and the write-back value in `esi` when there is one.
    ///
    /// A shifted offset is materialised in `edi`, which nothing else in a fast
    /// path uses.
    fn emit_addr(e: &mut Emitter, rn: u8, pc: u32, pre: bool, up: bool, writeback: bool, off: Off) {
        match rn {
            15 => e.mov_ecx(pc.wrapping_add(8)), // literal pool; a constant here
            _ => e.load_ecx(rn),
        }
        if let Off::Shifted(rm, typ, amount) = off {
            e.load_edi(rm);
            e.shift_edi(typ, amount);
        }
        if pre {
            match off {
                Off::Imm(v) if up => e.add_ecx(v),
                Off::Imm(v) => e.sub_ecx(v),
                Off::Reg(rm) if up => e.add_ecx_reg(rm),
                Off::Reg(rm) => e.sub_ecx_reg(rm),
                Off::Shifted(..) if up => e.add_ecx_edi(),
                Off::Shifted(..) => e.sub_ecx_edi(),
            }
            if writeback {
                e.mov_esi_ecx();
            }
        } else {
            // Post-indexed: the access uses the base and the write-back the
            // sum, and `!pre` already implies a write-back.
            e.mov_esi_ecx();
            match off {
                Off::Imm(v) if up => e.add_esi(v),
                Off::Imm(v) => e.sub_esi(v),
                Off::Reg(rm) if up => e.add_esi_reg(rm),
                Off::Reg(rm) => e.sub_esi_reg(rm),
                Off::Shifted(..) if up => e.add_esi_edi(),
                Off::Shifted(..) => e.sub_esi_edi(),
            }
        }
    }

    /// Emit a halfword or signed transfer: `ldrh`, `strh`, `ldrsb`, `ldrsh`.
    ///
    /// This is the one access that can straddle a page — a word is aligned
    /// before use and a byte cannot cross at all, but a halfword at offset
    /// 0xFFFF has its second byte in the next page — so it is the one fast
    /// path that checks for it. It does not align the address: the interpreter
    /// reads the two bytes where they lie, and so does an unaligned 16-bit x86
    /// load.
    ///
    /// `ldrd`/`strd` share this encoding space and are left to the
    /// interpreter; GCC 3.3 does not emit them.
    fn emit_half(e: &mut Emitter, insn: u32, pc: u32, ptrs: usize) -> Option<Vec<usize>> {
        if insn & 0x0E00_0090 != 0x0000_0090 || insn & 0x60 == 0 {
            return None;
        }
        let pre = insn & 0x0100_0000 != 0;
        let up = insn & 0x0080_0000 != 0;
        let load = insn & 0x0010_0000 != 0;
        let rn = ((insn >> 16) & 0xF) as u8;
        let rd = ((insn >> 12) & 0xF) as u8;
        let writeback = !pre || insn & 0x0020_0000 != 0;
        let kind = (insn >> 5) & 3;

        if rd == 15 || (rn == 15 && writeback) {
            return None;
        }
        if !load && kind != 1 {
            return None; // ldrd / strd
        }
        let off = if insn & 0x0040_0000 != 0 {
            // An 8-bit immediate, split across two nibbles.
            Off::Imm(((insn >> 4) & 0xF0) | (insn & 0xF))
        } else {
            let rm = (insn & 0xF) as u8;
            if rm == 15 {
                return None; // reads pc + 8
            }
            Off::Reg(rm)
        };

        let mut bails = Vec::new();
        emit_addr(e, rn, pc, pre, up, writeback, off);
        bails.push(e.page_lookup(ptrs));
        if kind != 2 {
            bails.push(e.bail_page_cross()); // two bytes, so it might cross
        }

        if load {
            match kind {
                1 => e.mem_load_half(),
                2 => e.mem_load_sbyte(),
                _ => e.mem_load_shalf(),
            }
            if writeback {
                e.store_esi(rn);
            }
            e.store_reg(rd);
        } else {
            e.load_ecx(rd);
            e.mem_store_half();
            if writeback {
                e.store_esi(rn);
            }
        }
        Some(bails)
    }

    /// Emit `mul` / `mla`. Only N and Z are written when S is set: the
    /// interpreter leaves C and V alone, as ARMv5 permits.
    fn emit_mul(e: &mut Emitter, insn: u32) -> bool {
        if insn & 0x0FC0_00F0 != 0x0000_0090 {
            return false;
        }
        let rd = ((insn >> 16) & 0xF) as u8;
        let rn = ((insn >> 12) & 0xF) as u8;
        let rs = ((insn >> 8) & 0xF) as u8;
        let rm = (insn & 0xF) as u8;
        if rd == 15 || rn == 15 || rs == 15 || rm == 15 {
            return false;
        }
        e.load_reg(rm);
        e.imul_reg(rs);
        if insn & 0x0020_0000 != 0 {
            e.add_reg(rn); // mla
        }
        if insn & 0x0010_0000 != 0 {
            e.test_eax();
            e.setcc_flag(SET_S, OFF_N);
            e.setcc_flag(SET_Z, OFF_Z);
        }
        e.store_reg(rd);
        true
    }

    /// Emit `umull` / `smull` / `umlal` / `smlal`.
    ///
    /// A 32x32 multiply widened to 64 bits is one x86 instruction, so the
    /// work here is getting the operands extended the right way round and the
    /// result back into two guest registers. Both halves of the accumulator
    /// are read before either is written, which is what the interpreter does.
    fn emit_mull(e: &mut Emitter, insn: u32) -> bool {
        if insn & 0x0F80_00F0 != 0x0080_0090 {
            return false;
        }
        let rdhi = ((insn >> 16) & 0xF) as u8;
        let rdlo = ((insn >> 12) & 0xF) as u8;
        let rs = ((insn >> 8) & 0xF) as u8;
        let rm = (insn & 0xF) as u8;
        if rdhi == 15 || rdlo == 15 || rs == 15 || rm == 15 {
            return false;
        }
        if insn & 0x0040_0000 != 0 {
            e.movsxd_rax(rm);
            e.movsxd_rcx(rs);
        } else {
            // A 32-bit load zero-extends into the full register, which is
            // exactly the unsigned widening this wants.
            e.load_reg(rm);
            e.load_ecx(rs);
        }
        e.imul_rax_rcx();
        if insn & 0x0020_0000 != 0 {
            e.load_ecx(rdhi);
            e.shl_rcx_32();
            e.load_edx(rdlo);
            e.or_rcx_rdx();
            e.add_rax_rcx();
        }
        if insn & 0x0010_0000 != 0 {
            // N and Z come from the whole 64-bit result.
            e.test_rax();
            e.setcc_flag(SET_S, OFF_N);
            e.setcc_flag(SET_Z, OFF_Z);
        }
        e.store_reg(rdlo);
        e.shr_rax_32();
        e.store_reg(rdhi);
        true
    }

    /// Emit `b` / `bl`, whose target is a constant.
    ///
    /// A branch is always the last instruction in a block — `terminates` says
    /// so and `translate` stops there — so it can write `r15` and leave
    /// directly, taking its own epilogue with it. The caller's condition test
    /// covers the not-taken case: control falls past this into the ordinary
    /// end-of-block epilogue, which stores the fall-through PC.
    ///
    /// This is the one place emitted code touches `r15`, and it is why the
    /// caller must not then overwrite it.
    fn emit_branch(e: &mut Emitter, insn: u32, pc: u32, insns: u32, native: u32) -> bool {
        if (insn >> 25) & 7 != 5 {
            return false;
        }
        let off = (((insn & 0x00FF_FFFF) as i32) << 8 >> 8) as u32;
        if insn & 0x0100_0000 != 0 {
            e.store_imm(14, pc.wrapping_add(4)); // bl: lr = the next instruction
        }
        e.store_imm(15, pc.wrapping_add(8).wrapping_add(off << 2));
        e.epilogue(insns, native);
        true
    }

    /// Compile a whole block: native code where possible, a callback into the
    /// interpreter everywhere else.
    ///
    /// Compiling a prefix instead was tried and measured 0% coverage: blocks
    /// start at branch targets, which open with a load, `push`, compare or
    /// conditional, so the emittable prefix was almost always empty. Covering
    /// the whole block makes coverage track instruction coverage instead.
    ///
    /// `ptrs` is the address of the guest page table, baked into every inlined
    /// memory access. [`BlockCache`] refuses to reuse blocks across a change to
    /// it.
    ///
    /// [`BlockCache`]: super::BlockCache
    pub fn compile(ops: &[Op], start: u32, ptrs: usize) -> Option<Compiled> {
        let n = ops.len();
        if n == 0 {
            return None;
        }
        // A callback is dearer than interpreting the same instruction — same
        // work, behind a call — so a block only pays for itself once enough of
        // it is native. These are host instructions, counted off the emitted
        // sequences and the interpreter's own dispatch; only their ratio
        // matters, and it is worth re-deriving whenever the emitted set widens.
        const INTERP: usize = 26; // one instruction through `Cpu::exec`
        const CALLBACK: usize = 28; // the same, reached through a call
        const DP: usize = 3; // an emitted data-processing op or branch
        const MEM: usize = 13; // an emitted load or store, fast path
        const MUL: usize = 8; // an emitted multiply
        const ENTRY: usize = 15; // prologue and epilogue, once per entry

        let mut e = Emitter::new();
        e.prologue();
        // (patch site, instructions run, of which native). An exit at `i` ran
        // the instructions before it plus the one that diverted, and the
        // diverting instruction was a callback or a bail — never native.
        let mut exits: Vec<(usize, u32, u32)> = Vec::new();
        let mut bails: Vec<Bail> = Vec::new();
        let mut native = 0usize;
        // Host instructions the block is expected to cost, against `INTERP`
        // per instruction for the interpreter. See the check after the loop.
        let mut cost = ENTRY;
        // Set when a branch wrote `r15` and left on its own: the end-of-block
        // store must not then clobber it, and for an unconditional branch the
        // fall-through is unreachable.
        let mut branched_uncond = false;

        for (i, op) in ops.iter().enumerate() {
            let Op::Raw { insn } = *op;
            let pc = start.wrapping_add(4 * i as u32);
            let native_before = native as u32;

            // The condition test goes down first and the body is tried after
            // it, because whether the body is emittable is not known until it
            // has been decoded. If it is not, the buffer is rewound and the
            // instruction becomes an ordinary callback, so a rejected
            // instruction leaves nothing behind.
            let rewind = e.len();
            if let Some(skips) = emit_cond(&mut e, insn >> 28) {
                let body = if emit(&mut e, insn) || emit_s(&mut e, insn) {
                    native += 1;
                    cost += DP;
                    true
                } else if emit_mul(&mut e, insn) || emit_mull(&mut e, insn) {
                    native += 1;
                    cost += MUL;
                    true
                } else if let Some(sites) =
                    emit_mem(&mut e, insn, pc, ptrs).or_else(|| emit_half(&mut e, insn, pc, ptrs))
                {
                    native += 1;
                    cost += MEM;
                    bails.push(Bail {
                        sites,
                        insn,
                        pc,
                        count: i as u32 + 1,
                        native: native_before,
                    });
                    true
                } else if i + 1 == n && emit_branch(&mut e, insn, pc, n as u32, native_before + 1) {
                    native += 1;
                    cost += DP;
                    branched_uncond = skips.is_empty();
                    true
                } else {
                    false
                };
                if body {
                    // A failed condition lands here: past the body, on to the
                    // next instruction.
                    for site in skips {
                        e.patch(site);
                    }
                    continue;
                }
                e.truncate(rewind);
            }
            e.call_step(step as usize, insn, pc);
            cost += CALLBACK;
            exits.push((e.test_jnz(), i as u32 + 1, native_before));
        }

        if cost >= n * INTERP {
            return None;
        }

        // Fell off the end: r15 is the instruction after the block. Native
        // instructions never touch r15 — except an emitted branch, which took
        // its own exit above, so reaching here means it was not taken and the
        // fall-through PC is right. A callback that returned 0 already
        // advanced r15 to exactly here, so this is correct either way.
        if !branched_uncond {
            e.store_imm(15, start.wrapping_add(4 * n as u32));
            e.epilogue(n as u32, native as u32);
        }
        for (at, cnt, nat) in exits {
            e.patch(at);
            e.epilogue(cnt, nat);
        }
        // Slow-path stubs, out of line past every exit: they run the whole
        // instruction through the interpreter and leave the block.
        for b in bails {
            for site in b.sites {
                e.patch(site);
            }
            e.call_step(step_slow as usize, b.insn, b.pc);
            e.epilogue(b.count, b.native);
        }

        ExecBuf::new(&e.finish()).ok().map(|buf| Compiled {
            buf,
            covered: n,
            native,
        })
    }
}

#[cfg(not(all(unix, target_arch = "x86_64")))]
mod backend {
    use super::Op;

    /// Placeholder on targets without a backend; nothing is ever compiled.
    pub struct Compiled {
        pub covered: usize,
        pub native: usize,
    }

    impl Compiled {
        /// # Safety
        /// Never called: `compile` always returns `None` here.
        pub unsafe fn run(&self, _regs: *mut u32, _cpu: *mut u8, _mem: *mut u8) -> u64 {
            0
        }
    }

    pub fn compile(_ops: &[Op], _start: u32, _ptrs: usize) -> Option<Compiled> {
        None
    }

    /// No fast path here, so nothing ever bails.
    pub fn bail_count() -> u64 {
        0
    }

    /// Nothing is compiled here, so nothing is rejected either.
    #[derive(Clone, Copy)]
    pub enum Why {
        Other,
    }

    pub const WHY: [(&str, Why); 1] = [("other", Why::Other)];

    pub fn rejected() -> [u64; 1] {
        [0]
    }
}

pub use backend::{bail_count, compile, rejected, Compiled, Why, WHY};
