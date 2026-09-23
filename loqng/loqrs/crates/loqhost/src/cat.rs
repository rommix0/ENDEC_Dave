//! A native replacement for the unit-selection context cost in `LoqTTS6.so`.
//!
//! After the codec kernels moved out of the way, profiling put ~35% of every
//! remaining guest instruction inside one routine at `LoqTTS6.so+0x29f1c`, and
//! another ~19% in the candidate loop that calls it. It has no symbol — the
//! nearest preceding one is `CatCloseGlob`, which ends well before it — so it
//! is named here for what it does.
//!
//! It scores how well a candidate unit's phonetic context matches the target's,
//! walking right then left through the context and accumulating table-driven
//! penalties with decreasing weight. Everything is integer and every table is
//! in guest memory, so this is a transcription rather than a model of it.
//!
//! `glob+0x774` selects which half of a diphone is being matched, and its low
//! bit picks between two weightings that are otherwise the same shape. The odd
//! case looks one position further to the right and one less to the left.

use armemu::Mem;

/// `.rodata` offset within `LoqTTS6.so` of the pairwise class-penalty table,
/// recovered from the literal pool at `0x2a89c`: the routine forms
/// `pc + 0x91000` and then subtracts `0x2c404`.
pub(crate) const COST_TABLE: u32 = 0x8_EB2C;

/// Context positions at or above this distance count as a mismatch.
const MISMATCH: u32 = 0x1E;

/// Returned instead of a score when the candidate's class disqualifies it.
const PENALTY_HARD: i32 = 5_000_000;
const PENALTY_SOFT: i32 = 2_500_000;

#[inline(always)]
pub(crate) fn b(mem: &Mem, addr: u32) -> Result<u32, String> {
    mem.read_u8(addr).map(u32::from).map_err(|e| e.to_string())
}

#[inline(always)]
pub(crate) fn h(mem: &Mem, addr: u32) -> Result<u32, String> {
    mem.read_u16(addr).map(u32::from).map_err(|e| e.to_string())
}

#[inline(always)]
pub(crate) fn w(mem: &Mem, addr: u32) -> Result<u32, String> {
    mem.read_u32(addr).map_err(|e| e.to_string())
}

/// `cost[y * 16 + x]`.
#[inline(always)]
fn cost(mem: &Mem, table: u32, x: u32, y: u32) -> Result<i32, String> {
    let idx = x.wrapping_add(y.wrapping_mul(0x10));
    Ok(w(mem, table.wrapping_add(idx.wrapping_mul(4)))? as i32)
}

/// Whether the fixed-weight left walk continues past a position scoring `v`.
///
/// Below the threshold it always stops; at or above it the original stops
/// testing the distance and starts testing how much context is left, which is
/// why this needs both.
#[inline(always)]
fn keep_walking(v: u32, idx: u32, step: u32) -> bool {
    let (mut gt, mut eq) = (v > 0x1C, v == 0x1D);
    if v > 0x1D {
        gt = idx > step - 1;
        eq = idx == step;
    }
    gt && !eq
}

/// Everything the walks need that does not change within one call.
struct Ctx {
    table: u32,
    cand: u32,
    tgt: u32,
    dist: u32,
    stride: u32,
    unit: u32,
    idx: u32,
}

/// One position of left context, `step` places back.
///
/// The slot off the front of the candidate has no predecessor to compare, so
/// it uses a sentinel distance row and contributes no class penalty.
fn left_step(
    mem: &Mem,
    c: &Ctx,
    step: u32,
    weight: i32,
    score: &mut i32,
    left: &mut i32,
) -> Result<u32, String> {
    let boundary = c.unit == step.wrapping_sub(1);
    let prev = c
        .cand
        .wrapping_add(c.unit.wrapping_sub(step).wrapping_mul(0x10));
    let row = if boundary {
        c.stride.wrapping_mul(2)
    } else {
        c.stride.wrapping_mul(b(mem, prev.wrapping_add(6))?)
    };

    let ti = c.idx.wrapping_sub(step).wrapping_mul(4);
    let v = b(
        mem,
        c.dist
            .wrapping_add(row)
            .wrapping_add(b(mem, c.tgt.wrapping_add(ti))?),
    )?;
    if !boundary {
        *score += cost(
            mem,
            c.table,
            b(mem, prev.wrapping_add(7))?,
            b(mem, c.tgt.wrapping_add(ti.wrapping_add(1)))?,
        )? * weight;
    }
    *left += (v as i32) * weight;
    Ok(v)
}

/// One position of right context, `step` places forward.
fn right_step(
    mem: &Mem,
    c: &Ctx,
    step: u32,
    weight: i32,
    score: &mut i32,
    target: &mut i32,
) -> Result<u32, String> {
    let next = c
        .cand
        .wrapping_add(c.unit.wrapping_add(step).wrapping_mul(0x10));
    let ti = c.idx.wrapping_add(step).wrapping_mul(4);
    let v = b(
        mem,
        c.dist
            .wrapping_add(c.stride.wrapping_mul(b(mem, next.wrapping_add(6))?))
            .wrapping_add(b(mem, c.tgt.wrapping_add(ti))?),
    )?;
    *score += cost(
        mem,
        c.table,
        b(mem, next.wrapping_add(7))?,
        b(mem, c.tgt.wrapping_add(ti.wrapping_add(1)))?,
    )? * weight;
    *target += (v as i32) * weight;
    Ok(v)
}

/// `LoqTTS6.so+0x29f1c`. Returns the candidate's score and writes the
/// concatenation half of it to `glob+0x918`, as the original does.
#[allow(clippy::too_many_arguments)]
pub fn context_cost(
    mem: &mut Mem,
    base: u32,
    glob: u32,
    tgt: u32,
    cand: u32,
    cls: u32,
    rec: u32,
    idx: u32,
) -> Result<i32, String> {
    let pos = h(mem, glob.wrapping_add(0x774))?;
    let odd = pos & 1 != 0;
    let count = h(mem, rec.wrapping_add(4))?;
    let phones = w(mem, glob.wrapping_add(0x91C))?;

    let c = Ctx {
        table: base.wrapping_add(COST_TABLE),
        cand,
        tgt,
        dist: w(mem, phones.wrapping_add(8))?,
        stride: w(mem, phones.wrapping_add(4))?,
        unit: pos >> 1,
        idx,
    };
    let at = cand.wrapping_add(c.unit.wrapping_mul(0x10));

    // The position itself contributes a class penalty but no distance.
    let mut score = cost(
        mem,
        c.table,
        b(mem, at.wrapping_add(7))?,
        b(mem, tgt.wrapping_add(idx.wrapping_mul(4).wrapping_add(1)))?,
    )? * 10;
    let mut target: i32 = 1;
    let mut left: i32 = 0;
    let mut back: u32 = 0;
    let depth: u32;

    // Right context: fixed weights, then an open-ended tail at weight 1.
    let right: &[i32] = if odd { &[10, 6, 3] } else { &[6, 3] };
    if idx + 1 < count {
        let mut stopped = None;
        for (i, weight) in right.iter().enumerate() {
            let step = i as u32 + 1;
            let v = right_step(mem, &c, step, *weight, &mut score, &mut target)?;
            if v < MISMATCH || count <= idx + step + 1 {
                stopped = Some(step + 1);
                break;
            }
        }
        match stopped {
            Some(d) => depth = d,
            None => {
                let mut k = right.len() as u32;
                loop {
                    let step = k + 1;
                    let v = right_step(mem, &c, step, 1, &mut score, &mut target)?;
                    if v < MISMATCH || idx + step + 1 >= count {
                        break;
                    }
                    k = step;
                }
                depth = k + 2;
            }
        }
    } else {
        depth = 1;
    }

    // Left context: same shape, but the tail's stop test and the value it
    // leaves in `back` differ between the two halves.
    if idx != 0 {
        let fixed: &[i32] = if odd { &[6, 3] } else { &[10, 4, 2] };
        let mut stopped = false;
        for (i, weight) in fixed.iter().enumerate() {
            let step = i as u32 + 1;
            let v = left_step(mem, &c, step, *weight, &mut score, &mut left)?;
            if !keep_walking(v, idx, step) || c.unit < step {
                back = step;
                stopped = true;
                break;
            }
        }
        if !stopped {
            let mut n = fixed.len() as u32 + 1;
            loop {
                // The odd half reports the position it stopped on; the even
                // half reports the one after it.
                let step = if odd { n } else { n };
                let v = left_step(mem, &c, step, 1, &mut score, &mut left)?;
                back = if odd { step } else { step + 1 };
                let room = if odd { back + 1 <= idx } else { back <= idx };
                let have = if odd { back <= c.unit } else { step <= c.unit };
                n = step + 1;
                if !(v > 0x1D && room && have) {
                    break;
                }
            }
        }
    }

    // Both halves are scaled by the candidate's own phone weight, but which
    // of them gets it depends on where the walks stopped.
    let phone = b(
        mem,
        w(mem, w(mem, glob.wrapping_add(0x40))?)?
            .wrapping_add(b(mem, at.wrapping_add(6))?.wrapping_mul(0x18))
            .wrapping_add(5),
    )? as i32;

    if odd {
        left = left.wrapping_mul(phone) / 10;
        if depth == 2 {
            score = score.wrapping_mul(phone) / 10;
        }
    } else if back == 1 {
        score = score.wrapping_mul(phone) / 10;
    }

    target = target.wrapping_add(left);

    let rc = b(mem, rec.wrapping_add(6))?;
    if rc == b'I' as u32 {
        if (cls == b'X' as u32 || cls == b'I' as u32)
            && idx == back
            && b(mem, tgt.wrapping_add(4))? == b(mem, cand.wrapping_add(6))?
        {
            score = score.wrapping_mul(3) / 2;
        } else {
            score /= 3;
            target /= 3;
        }
    } else if rc != b'D' as u32 {
        score /= 10;
        target /= 10;
    }

    mem.write_u32(glob.wrapping_add(0x918), score as u32)
        .map_err(|e| e.to_string())?;

    // A candidate sitting at a phrase boundary whose class does not line up
    // with the target's is rejected outright rather than scored.
    let tail = cand.wrapping_add(depth.wrapping_mul(0x10));
    let scored = depth + 1 < count || b(mem, tgt)? != 2 || b(mem, tail.wrapping_add(6))? != 0 || {
        let p = b(mem, tail.wrapping_sub(10))?;
        p != b(mem, tgt.wrapping_add(count.wrapping_mul(4).wrapping_sub(4)))?
            || p.wrapping_sub(2) > 1
    };
    if scored {
        return Ok(target.wrapping_add(score));
    }

    let same_class = [
        (b'D', b'd'),
        (b'I', b'i'),
        (b'E', b'e'),
        (b'F', b'f'),
        (b'X', b'x'),
    ]
    .iter()
    .any(|(u, l)| cls == *u as u32 && rc == *l as u32);
    if rc == cls || same_class {
        return Ok(PENALTY_HARD);
    }
    if (rc == b'e' as u32 && cls == b'F' as u32) || (rc == b'f' as u32 && cls == b'E' as u32) {
        return Ok(PENALTY_SOFT);
    }
    Ok(target.wrapping_add(score))
}
