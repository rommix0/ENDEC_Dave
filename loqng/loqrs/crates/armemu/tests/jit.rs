//! Differential test: the JIT must leave the machine in exactly the state the
//! interpreter would.
//!
//! The audio regression is the stronger correctness gate — byte-identical
//! output over real synthesis exercises millions of instructions — but it only
//! covers paths the engine happens to take. It never produced a single
//! unaligned or unmapped access, so the memory fast path's bail stubs ran
//! zero times. Those paths are reached deliberately here.
//!
//! Every case asserts that compiled code actually ran. Without that a block
//! rejected by the cost model would interpret both times and pass vacuously.

use armemu::{Machine, Stop, MAGIC_RETURN};

/// Where the assembled block goes.
const CODE: u32 = 0x0010_0000;
/// Scratch the block loads from and stores to.
const DATA: u32 = 0x0020_0000;
/// Deliberately never mapped.
const UNMAPPED: u32 = 0x5000_0000;

/// `bx lr`, which returns to [`MAGIC_RETURN`] and halts the machine.
const BX_LR: u32 = 0xE12F_FF1E;

struct Outcome {
    regs: [u32; 16],
    flags: (bool, bool, bool, bool),
    data: Vec<u8>,
    stop: Stop,
    /// Instructions that ran inside a compiled block.
    jitted: u64,
}

/// Assemble `insns`, run them, and report the final state.
///
/// The block is followed by `bx lr` and then eight literal words, so a test
/// can exercise pc-relative loads without executing its own data.
fn run(insns: &[u32], jit: bool, flags: (bool, bool, bool, bool)) -> Outcome {
    let mut m = Machine::new();
    m.jit = jit;
    m.mem.map(CODE, 4096);
    // Two pages, so a halfword can be made to straddle the boundary.
    m.mem.map(DATA, 2 * 0x1_0000);

    for (i, insn) in insns.iter().enumerate() {
        m.mem.write_u32(CODE + 4 * i as u32, *insn).unwrap();
    }
    let end = CODE + 4 * insns.len() as u32;
    m.mem.write_u32(end, BX_LR).unwrap();
    // Literal 0 is MAGIC_RETURN so a test that clobbers `lr` can reload it
    // and still halt; the rest are just recognisable.
    m.mem.write_u32(end + 4, MAGIC_RETURN).unwrap();
    for k in 1..8u32 {
        m.mem.write_u32(end + 4 + 4 * k, 0xDEAD_0000 | k).unwrap();
    }

    // A recognisable pattern: a load that reads the wrong address, or reads
    // the right one with the wrong rotation, produces a visibly wrong word.
    for i in 0..256u32 {
        m.mem.write_u32(DATA + 4 * i, 0xA0B0_0000 | i).unwrap();
    }

    m.cpu.r[0] = DATA;
    m.cpu.r[1] = DATA + 0x40;
    m.cpu.r[2] = 4;
    m.cpu.r[3] = 0x1234_5678;
    m.cpu.r[4] = DATA + 1; // unaligned on purpose
    m.cpu.r[5] = 0x8000_0001;
    m.cpu.r[6] = UNMAPPED;
    m.cpu.r[7] = DATA + 0xFFFF; // the last byte of the first page
                                // A base well inside the mapped page, so a negative offset still lands
                                // somewhere real.
    m.cpu.r[9] = DATA + 0x200;
    // The edges that carry and overflow turn on.
    m.cpu.r[8] = 0;
    m.cpu.r[10] = 0xFFFF_FFFF;
    m.cpu.r[11] = 0x8000_0000;
    m.cpu.r[12] = 0x7FFF_FFFF;
    m.cpu.r[14] = MAGIC_RETURN;
    m.cpu.r[15] = CODE;
    // Flags are set here rather than by a preceding `cmp` so that every
    // combination is reachable, including the ones a comparison cannot
    // produce on its own.
    (m.cpu.n, m.cpu.z, m.cpu.c, m.cpu.v) = flags;

    let stop = m.run(10_000);
    Outcome {
        regs: m.cpu.r,
        flags: (m.cpu.n, m.cpu.z, m.cpu.c, m.cpu.v),
        data: {
            // The scratch window, plus the bytes either side of the page
            // boundary that a straddling halfword touches.
            let mut d = m.mem.read_bytes(DATA, 1024).unwrap();
            d.extend(m.mem.read_bytes(DATA + 0xFFF0, 64).unwrap());
            d
        },
        stop,
        jitted: m.cpu.jitted,
    }
}

/// Run `insns` both ways and require the results to agree.
#[track_caller]
fn same(insns: &[u32]) {
    same_with(insns, (false, false, false, false));
}

#[track_caller]
fn same_with(insns: &[u32], flags: (bool, bool, bool, bool)) {
    let a = run(insns, false, flags);
    let b = run(insns, true, flags);
    assert!(
        b.jitted > 0,
        "nothing was compiled, so this compared the interpreter with itself"
    );
    assert_eq!(a.stop, b.stop, "stop reason differs (flags {flags:?})");
    for (i, (x, y)) in a.regs.iter().zip(b.regs.iter()).enumerate() {
        assert_eq!(
            x, y,
            "r{i} differs with flags {flags:?}: interpreter 0x{x:08x}, jit 0x{y:08x}"
        );
    }
    assert_eq!(a.flags, b.flags, "flags differ (started {flags:?})");
    assert!(a.data == b.data, "guest memory differs (flags {flags:?})");
}

#[test]
fn word_transfer_immediate_offset() {
    same(&[
        0xE590_1004, // ldr r1, [r0, #4]
        0xE590_2010, // ldr r2, [r0, #16]
        0xE580_1020, // str r1, [r0, #32]
        0xE510_3008, // ldr r3, [r0, #-8]   (U = 0)
        0xE500_2004, // str r2, [r0, #-4]
    ]);
}

#[test]
fn byte_transfer() {
    same(&[
        0xE5D0_1001, // ldrb r1, [r0, #1]
        0xE5D0_2003, // ldrb r2, [r0, #3]
        0xE5C0_1011, // strb r1, [r0, #17]
        0xE5C0_2012, // strb r2, [r0, #18]
        0xE5D0_3011, // ldrb r3, [r0, #17]
    ]);
}

#[test]
fn write_back_pre_and_post_indexed() {
    same(&[
        0xE4B0_1004, // ldr r1, [r0], #4     post-indexed
        0xE5B0_2008, // ldr r2, [r0, #8]!    pre-indexed with write-back
        0xE4A0_3004, // str r3, [r0], #4     post-indexed store
        0xE5A0_1010, // str r1, [r0, #16]!
    ]);
}

/// `ldr rN, [rN], #k` — the loaded value must win over the write-back, which
/// is the one ordering the emitted sequence could plausibly get backwards.
#[test]
fn write_back_into_the_loaded_register() {
    same(&[
        0xE490_0004, // ldr r0, [r0], #4
        0xE590_1000, // ldr r1, [r0]
        0xE591_2000, // ldr r2, [r1]
    ]);
}

#[test]
fn register_offset() {
    same(&[
        0xE790_1002, // ldr r1, [r0, r2]
        0xE780_1002, // str r1, [r0, r2]
        0xE710_3002, // ldr r3, [r0, -r2]
        0xE7D0_5002, // ldrb r5, [r0, r2]
    ]);
}

/// The fast path bails on these: an unaligned word load rotates on ARMv5 and
/// the emitted sequence does not, so it hands the whole instruction back to
/// the interpreter.
#[test]
fn unaligned_word_load_rotates() {
    same(&[
        0xE594_1000, // ldr r1, [r4]        r4 = DATA + 1
        0xE594_2001, // ldr r2, [r4, #1]    DATA + 2
        0xE594_3002, // ldr r3, [r4, #2]    DATA + 3
        0xE590_5001, // ldr r5, [r0, #1]
        0xE580_1004, // str r1, [r0, #4]    keep a result in memory too
    ]);
}

/// An unaligned word store does not rotate — it truncates the address — so it
/// stays on the fast path. Worth pinning separately from the load.
#[test]
fn unaligned_word_store_truncates() {
    same(&[
        0xE584_3000, // str r3, [r4]        r4 = DATA + 1
        0xE584_3001, // str r3, [r4, #1]
        0xE590_1000, // ldr r1, [r0]
        0xE590_2004, // ldr r2, [r0, #4]
    ]);
}

#[test]
fn pc_relative_literal_load() {
    // Five instructions, then `bx lr` at index 5 and literals from index 6.
    // For the instruction at index i, `[pc, #off]` reads CODE + 4i + 8 + off.
    same(&[
        0xE59F_1010, // ldr r1, [pc, #16]   -> index 6
        0xE59F_2010, // ldr r2, [pc, #16]   -> index 7
        0xE59F_3010, // ldr r3, [pc, #16]   -> index 8
        0xE580_1000, // str r1, [r0]
        0xE580_2004, // str r2, [r0, #4]
    ]);
}

/// A faulting access must report the same stop, at the same PC, either way.
#[test]
fn unmapped_access_faults_identically() {
    same(&[
        0xE590_1000, // ldr r1, [r0]      fine
        0xE596_2000, // ldr r2, [r6]      r6 is unmapped
        0xE580_2008, // str r2, [r0, #8]  never reached
    ]);
}

#[test]
fn unmapped_store_faults_identically() {
    same(&[
        0xE590_1000, // ldr r1, [r0]
        0xE586_1000, // str r1, [r6]
        0xE590_2000, // ldr r2, [r0]
    ]);
}

/// Every condition code against every flag combination.
///
/// Compiled code reads the flags the interpreter left on the `Cpu`, so the
/// initial state is set directly rather than with a preceding comparison —
/// that is the only way to reach combinations like "n set, v set, z set".
#[test]
fn every_condition_matches_the_interpreter() {
    for cond in 0x0..=0xEu32 {
        let c = cond << 28;
        for bits in 0..16u32 {
            same_with(
                &[
                    c | 0x0590_1004, // ldr<c> r1, [r0, #4]
                    c | 0x03A0_2001, // mov<c> r2, #1
                    c | 0x0580_2008, // str<c> r2, [r0, #8]
                    c | 0x0490_5004, // ldr<c> r5, [r0], #4   (write-back)
                    0xE590_3000,     // ldr  r3, [r0]
                    0xE580_300C,     // str  r3, [r0, #12]
                ],
                (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0),
            );
        }
    }
}

/// A conditional access that also has to bail: the slow-path stub re-runs the
/// whole instruction, condition included, so it must not run a body the
/// condition had already ruled out.
#[test]
fn conditional_unaligned_load() {
    for cond in [0x0u32, 0x1, 0x8, 0x9, 0xC, 0xD] {
        let c = cond << 28;
        for bits in 0..16u32 {
            same_with(
                &[
                    c | 0x0594_1000, // ldr<c> r1, [r4]    r4 = DATA + 1
                    c | 0x0594_2002, // ldr<c> r2, [r4, #2]
                    0xE590_3000,     // ldr r3, [r0]
                    0xE580_1010,     // str r1, [r0, #16]
                    0xE580_2014,     // str r2, [r0, #20]
                ],
                (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0),
            );
        }
    }
}

/// `<op> rd, rn, rm, <shift> #n` for every opcode, shift type and amount.
///
/// `ror #0` is RRX, which needs the carry flag and is left to the
/// interpreter; the surrounding instructions keep the block worth compiling
/// so the case is still covered rather than skipped.
#[test]
fn shifted_register_operands() {
    // and eor sub rsb add orr mov bic mvn
    const OPCODES: [u32; 9] = [0x0, 0x1, 0x2, 0x3, 0x4, 0xC, 0xD, 0xE, 0xF];
    for opc in OPCODES {
        for typ in 0..4u32 {
            for amount in 0..32u32 {
                // r1 = r3 <op> (r5 shifted)
                let dp = 0xE000_0000
                    | (opc << 21)
                    | (3 << 16)
                    | (1 << 12)
                    | (amount << 7)
                    | (typ << 5)
                    | 5;
                same(&[
                    dp,
                    0xE580_1040, // str r1, [r0, #64]
                    0xE590_2040, // ldr r2, [r0, #64]
                    0xE282_6001, // add r6, r2, #1
                    0xE580_6044, // str r6, [r0, #68]
                ]);
            }
        }
    }
}

/// `ldr`/`str rd, [rn, rm, <shift> #n]`, every shift type and amount.
///
/// Some rotations put the address outside the mapped page. That is not a
/// problem to avoid — it is the fault path, and both sides must take it at the
/// same instruction with the same reason.
#[test]
fn shifted_memory_offsets() {
    for typ in 0..4u32 {
        for amount in 0..32u32 {
            for up in [0u32, 1] {
                let ldr = 0xE000_0000
                    | (3 << 25)
                    | (1 << 24) // pre-indexed
                    | (up << 23)
                    | (1 << 20) // load
                    | (9 << 16) // rn = r9, mid-page
                    | (1 << 12) // rd = r1
                    | (amount << 7)
                    | (typ << 5)
                    | 2; // rm = r2
                let str_ = (ldr & !(1 << 20)) | (3 << 12); // str r3, [...]
                same(&[
                    0xE590_5000, // ldr r5, [r0]   keeps the block compiled
                    ldr,
                    str_,
                    0xE580_5048, // str r5, [r0, #72]
                ]);
            }
        }
    }
}

/// The same shifted offsets with write-back, where the offset feeds both the
/// access and the new base.
#[test]
fn shifted_memory_offsets_with_write_back() {
    for typ in 0..4u32 {
        for amount in [0u32, 1, 2, 7, 31] {
            for pre in [0u32, 1] {
                let base = 0xE000_0000
                    | (3 << 25)
                    | (pre << 24)
                    | (1 << 23) // up
                    | (pre << 21) // W, meaningful only when pre-indexed
                    | (1 << 20)
                    | (9 << 16)
                    | (1 << 12)
                    | (amount << 7)
                    | (typ << 5)
                    | 2;
                same(&[
                    0xE590_5000, // ldr r5, [r0]
                    base,
                    0xE580_1050, // str r1, [r0, #80]
                    0xE580_9054, // str r9, [r0, #84]   the written-back base
                ]);
            }
        }
    }
}

/// `b<cond>`, taken and not taken, against every flag combination.
///
/// The branch is emitted as a constant store to `r15` followed by the block's
/// own exit — the one place compiled code writes the PC — so the not-taken
/// path has to reach the ordinary end-of-block epilogue and get the
/// fall-through PC instead.
#[test]
fn conditional_branch() {
    for cond in 0x0..=0xEu32 {
        for bits in 0..16u32 {
            same_with(
                &[
                    0xE590_1000,                    // ldr r1, [r0]
                    0xE580_1004,                    // str r1, [r0, #4]
                    (cond << 28) | 0x0A00_0000 | 1, // b<cond> -> index 5
                    0xE3A0_2001,                    // mov r2, #1
                    0xE580_2008,                    // str r2, [r0, #8]
                    0xE3A0_3002,                    // mov r3, #2
                    0xE580_300C,                    // str r3, [r0, #12]
                ],
                (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0),
            );
        }
    }
}

/// `bl<cond>` also writes `lr`, and only when taken.
///
/// The block reloads `lr` from the literal pool before returning, so a taken
/// branch still halts instead of running away.
#[test]
fn conditional_branch_with_link() {
    for cond in 0x0..=0xEu32 {
        for bits in 0..16u32 {
            same_with(
                &[
                    0xE590_1000,                // ldr r1, [r0]
                    0xE580_1004,                // str r1, [r0, #4]
                    (cond << 28) | 0x0B00_0000, // bl<cond> -> index 4
                    0xE3A0_2001,                // mov r2, #1
                    0xE580_E010,                // str lr, [r0, #16]
                    0xE59F_E000,                // ldr lr, [pc, #0]  = MAGIC_RETURN
                ],
                (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0),
            );
        }
    }
}

/// A backward branch, which is what a loop actually looks like.
#[test]
fn backward_branch_loop() {
    same(&[
        0xE3A0_2005, // mov r2, #5
        0xE590_1000, // ldr r1, [r0]        <- loop top, index 1
        0xE081_1002, // add r1, r1, r2
        0xE580_1000, // str r1, [r0]
        0xE242_2001, // sub r2, r2, #1
        0xE352_0000, // cmp r2, #0
        0x1AFF_FFF9, // bne -> index 1
        0xE580_1020, // str r1, [r0, #32]
    ]);
}

/// Operands chosen for the edges carry and overflow turn on.
const EDGES: [u32; 6] = [3, 5, 8, 10, 11, 12];

/// Two starting flag states, enough to tell "left alone" from "cleared".
const SEEDS: [(bool, bool, bool, bool); 2] =
    [(false, false, false, false), (true, true, true, true)];

/// Every data-processing opcode with the S bit, register operand.
///
/// `adc`/`sbc`/`rsc` are in the sweep although the backend does not emit them:
/// they have to keep agreeing, and they are the ones that read C rather than
/// write it.
#[test]
fn flag_setting_register_form() {
    for opc in 0x0..=0xFu32 {
        for rn in EDGES {
            for rm in EDGES {
                let dp = 0xE000_0000 | (opc << 21) | (1 << 20) | (rn << 16) | (1 << 12) | rm;
                for seed in SEEDS {
                    same_with(
                        &[
                            dp,
                            0xE580_1060, // str r1, [r0, #96]
                            0xE590_2060, // ldr r2, [r0, #96]
                            0xE580_2064, // str r2, [r0, #100]
                        ],
                        seed,
                    );
                }
            }
        }
    }
}

/// The same with a rotated immediate, whose carry-out is a constant the
/// backend folds in rather than computes.
#[test]
fn flag_setting_immediate_form() {
    for opc in 0x0..=0xFu32 {
        for rn in EDGES {
            for rot in [0u32, 1, 2, 8, 15] {
                for imm in [0u32, 1, 0x80, 0xFF] {
                    let dp = 0xE200_0000
                        | (opc << 21)
                        | (1 << 20)
                        | (rn << 16)
                        | (1 << 12)
                        | (rot << 8)
                        | imm;
                    for seed in SEEDS {
                        same_with(
                            &[
                                dp,
                                0xE580_1060, // str r1, [r0, #96]
                                0xE590_2060, // ldr r2, [r0, #96]
                                0xE580_2064, // str r2, [r0, #100]
                            ],
                            seed,
                        );
                    }
                }
            }
        }
    }
}

/// A shifted operand with the S bit. The backend takes these only for
/// arithmetic, where C comes from the ALU; a logical one would need the
/// shifter's carry-out and stays a callback.
#[test]
fn flag_setting_shifted_form() {
    for opc in 0x0..=0xFu32 {
        for typ in 0..4u32 {
            for amount in [0u32, 1, 7, 31] {
                let dp = 0xE000_0000
                    | (opc << 21)
                    | (1 << 20)
                    | (11 << 16)
                    | (1 << 12)
                    | (amount << 7)
                    | (typ << 5)
                    | 12;
                for seed in SEEDS {
                    same_with(
                        &[
                            dp,
                            0xE580_1060, // str r1, [r0, #96]
                            0xE590_2060, // ldr r2, [r0, #96]
                            0xE580_2064, // str r2, [r0, #100]
                        ],
                        seed,
                    );
                }
            }
        }
    }
}

/// Flags set by compiled code and consumed by compiled code in the same block
/// — the round trip through the guest's flag bytes.
#[test]
fn flags_round_trip_within_a_block() {
    for rn in EDGES {
        for rm in EDGES {
            same(&[
                0xE050_0000 | (rn << 16) | (1 << 12) | rm, // subs r1, rn, rm
                0x0590_2000,                               // ldreq r2, [r0]
                0xB590_3004,                               // ldrlt r3, [r0, #4]
                0x2580_1068,                               // strcs r1, [r0, #104]
                0xE580_106C,                               // str r1, [r0, #108]
                0xE580_2070,                               // str r2, [r0, #112]
                0xE580_3074,                               // str r3, [r0, #116]
            ]);
        }
    }
}

/// `ldrh` / `ldrsb` / `ldrsh` / `strh`, every addressing form.
#[test]
fn halfword_and_signed_transfers() {
    for kind in [1u32, 2, 3] {
        for load in [0u32, 1] {
            if load == 0 && kind != 1 {
                continue; // only strh exists
            }
            for imm in [0u32, 1] {
                for pre in [0u32, 1] {
                    for up in [0u32, 1] {
                        // Write-back is only encodable as W when pre-indexed;
                        // post-indexing always writes back.
                        for w in [0u32, 1] {
                            let (offhi, offlo) = if imm == 1 { (0, 6) } else { (0, 2) };
                            let t = 0xE000_0000
                                | (pre << 24)
                                | (up << 23)
                                | (imm << 22)
                                | ((w & pre) << 21)
                                | (load << 20)
                                | (9 << 16) // rn = r9, mid-page
                                | (1 << 12) // rd = r1
                                | (offhi << 8)
                                | 0x90
                                | (kind << 5)
                                | offlo;
                            same(&[
                                0xE590_5000, // ldr r5, [r0]   keeps the block compiled
                                t,
                                0xE580_1078, // str r1, [r0, #120]
                                0xE580_907C, // str r9, [r0, #124]  the base after
                            ]);
                        }
                    }
                }
            }
        }
    }
}

/// A halfword whose second byte is in the next page. The interpreter splits
/// those into two byte accesses, so the fast path hands them over — a case the
/// engine itself never once produces.
#[test]
fn halfword_straddling_a_page() {
    for kind in [1u32, 3] {
        // ldr<kind> r1, [r7]   with r7 at the last byte of the page
        let load = 0xE000_0000
            | (1 << 24)
            | (1 << 23)
            | (1 << 22)
            | (1 << 20)
            | (7 << 16)
            | (1 << 12)
            | 0x90
            | (kind << 5);
        same(&[
            0xE590_5000, // ldr r5, [r0]
            load,
            0xE580_1078, // str r1, [r0, #120]
        ]);
    }
    // strh across the boundary, which the interpreter writes a byte at a time.
    let store =
        0xE000_0000 | (1 << 24) | (1 << 23) | (1 << 22) | (7 << 16) | (3 << 12) | 0x90 | (1 << 5);
    same(&[
        0xE590_5000, // ldr r5, [r0]
        store,
        0xE580_507C, // str r5, [r0, #124]
    ]);
}

/// `mul` / `mla`, with and without the S bit.
#[test]
fn multiply_32() {
    for a in [0u32, 1] {
        for sbit in [0u32, 1] {
            for rm in EDGES {
                for rs in EDGES {
                    let mul = 0xE000_0000
                        | (a << 21)
                        | (sbit << 20)
                        | (1 << 16) // rd = r1
                        | (2 << 12) // rn = r2, the accumulator
                        | (rs << 8)
                        | 0x90
                        | rm;
                    for seed in SEEDS {
                        same_with(
                            &[
                                mul,
                                0xE580_1080, // str r1, [r0, #128]
                                0xE590_3080, // ldr r3, [r0, #128]
                                0xE580_3084, // str r3, [r0, #132]
                            ],
                            seed,
                        );
                    }
                }
            }
        }
    }
}

/// `umull` / `smull` / `umlal` / `smlal`.
///
/// The accumulating forms read both halves of the destination pair before
/// writing either, so the sweep includes them with the pair pre-loaded.
#[test]
fn multiply_64() {
    for signed in [0u32, 1] {
        for a in [0u32, 1] {
            for sbit in [0u32, 1] {
                for rm in EDGES {
                    for rs in EDGES {
                        let mull = 0xE080_0000
                            | (signed << 22)
                            | (a << 21)
                            | (sbit << 20)
                            | (1 << 16) // rdhi = r1
                            | (2 << 12) // rdlo = r2
                            | (rs << 8)
                            | 0x90
                            | rm;
                        same_with(
                            &[
                                mull,
                                0xE580_1088, // str r1, [r0, #136]
                                0xE580_208C, // str r2, [r0, #140]
                                0xE590_3088, // ldr r3, [r0, #136]
                            ],
                            SEEDS[1],
                        );
                    }
                }
            }
        }
    }
}
