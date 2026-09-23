//! Runtime for code translated from the engine by `tools/armxlate.py`.
//!
//! A translated function is ordinary Rust: one statement per original
//! instruction, over the sixteen registers and four flags in [`Cpu`] and the
//! memory in [`Mem`]. Nothing here decodes or interprets an instruction. The
//! module's own data -- tables, the GOT, globals -- sits in `Mem` at its ELF
//! vaddr, because that is where the translated address arithmetic expects it.
//!
//! Imports the runtime can do itself (`memcpy`, `strcmp`, ...) are here;
//! everything else, and every indirect call, goes to a [`Host`].

pub mod mem;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
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

    pub fn adc(a: u32, b: u32, cin: bool) -> (u32, Flags) {
        let wide = a as u64 + b as u64 + cin as u64;
        let r = wide as u32;
        (
            r,
            Flags {
                n: r >> 31 != 0,
                z: r == 0,
                c: wide >> 32 != 0,
                v: (!(a ^ b) & (a ^ r)) >> 31 != 0,
            },
        )
    }

    /// `a - b - !cin`, which ARM computes as `a + !b + cin`.
    pub fn sbc(a: u32, b: u32, cin: bool) -> (u32, Flags) {
        let (r, mut f) = Flags::adc(a, !b, cin);
        f.v = ((a ^ b) & (a ^ r)) >> 31 != 0;
        (r, f)
    }

    /// A logical result: N and Z from it, C from the shifter, V unchanged.
    pub fn logic(self, r: u32, c: bool) -> Flags {
        Flags {
            n: r >> 31 != 0,
            z: r == 0,
            c,
            v: self.v,
        }
    }
}

#[cfg(feature = "cover")]
static HITS: std::sync::Mutex<Option<std::collections::HashMap<u32, u64>>> =
    std::sync::Mutex::new(None);

/// Ordered block entries. Two translations of the same ARM function must
/// produce the same sequence, so the first place they differ is the first
/// branch one of them got wrong — which is how a translator bug gets found.
#[cfg(feature = "cover")]
static TRACE: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

#[cfg(feature = "cover")]
static TRACING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The last blocks entered, always recorded under `cover`. A fault in
/// translated code says nothing about where it came from, and this is the
/// cheapest thing that does: the tail names the guest function.
#[cfg(feature = "cover")]
static RING: std::sync::Mutex<([(u32, [u32; 16]); 64], usize)> =
    std::sync::Mutex::new(([(0, [0; 16]); 64], 0));

/// The last blocks entered, oldest first.
pub fn recent() -> Vec<(u32, [u32; 16])> {
    #[cfg(feature = "cover")]
    {
        let r = RING.lock().unwrap();
        let n = r.1;
        let take = n.min(64);
        return (0..take).map(|k| r.0[(n - take + k) % 64]).collect();
    }
    #[cfg(not(feature = "cover"))]
    Vec::new()
}

pub(crate) fn where_from() -> String {
    let r = recent();
    if r.is_empty() {
        return String::from("\n  (rebuild with `--features xrt-cover` to see where from)");
    }
    let mut s = String::from("\n  last blocks entered, oldest first:");
    for chunk in r.chunks(8) {
        s.push_str("\n   ");
        for (a, _) in chunk {
            s.push_str(&format!(" {a:08x}"));
        }
    }
    let (at, regs) = r[r.len() - 1];
    s.push_str(&format!("\n  registers on entry to {at:08x}:"));
    for (k, v) in regs.iter().enumerate() {
        if k % 4 == 0 {
            s.push_str("\n   ");
        }
        s.push_str(&format!(" r{k:<2}={v:08x}"));
    }
    s
}

/// A profile of block entries, cheap enough to believe.
///
/// `cover`'s counter takes three mutexes per block, which is fine for finding
/// a wrong branch and useless for finding where the time goes: it changes the
/// shape of what it measures. This is a direct-mapped array of counters over
/// the guest text, so a block entry costs an index, a compare and an add.
///
/// The counters are shared, not per-thread, and incremented without
/// synchronisation. The engine's threads pass a baton and only one of them
/// runs at a time (see `engine::sched`), so there is no race to lose; even if
/// there were, a dropped count in a profile is not a wrong answer.
#[cfg(feature = "prof")]
pub mod prof {
    use std::cell::UnsafeCell;

    /// The four module images together span less than 4 MB of guest text.
    pub const BASE: u32 = 0x0010_0000;
    pub const SLOTS: usize = 0x0040_0000 / 4;

    struct Counts(UnsafeCell<[u64; SLOTS]>);
    unsafe impl Sync for Counts {}

    static COUNT: Counts = Counts(UnsafeCell::new([0; SLOTS]));

    #[inline(always)]
    pub fn tick(pc: u32) {
        let k = (pc.wrapping_sub(BASE) >> 2) as usize;
        if k < SLOTS {
            unsafe { *(COUNT.0.get() as *mut u64).add(k) += 1 };
        }
    }

    /// Every block that was entered, and how often, clearing the counters.
    pub fn take() -> Vec<(u32, u64)> {
        let p = COUNT.0.get() as *mut u64;
        let mut out = Vec::new();
        for k in 0..SLOTS {
            let v = unsafe { *p.add(k) };
            if v != 0 {
                unsafe { *p.add(k) = 0 };
                out.push((BASE + (k as u32) * 4, v));
            }
        }
        out
    }
}

/// Count a block entry, the cheap way. See [`prof`].
#[cfg(feature = "prof")]
#[inline(always)]
pub fn hit(pc: u32, _regs: &[u32; 16]) {
    prof::tick(pc);
}

/// Count a block entry. Only with `--features xrt-cover`.
#[cfg(all(feature = "cover", not(feature = "prof")))]
pub fn hit(pc: u32, regs: &[u32; 16]) {
    {
        let mut r = RING.lock().unwrap();
        let n = r.1;
        r.0[n % 64] = (pc, *regs);
        r.1 = n + 1;
    }
    let mut h = HITS.lock().unwrap();
    *h.get_or_insert_with(Default::default)
        .entry(pc)
        .or_default() += 1;
    if TRACING.load(std::sync::atomic::Ordering::Relaxed) {
        TRACE.lock().unwrap().push(pc);
    }
}

/// Start an ordered trace, discarding any previous one.
#[cfg(feature = "cover")]
pub fn trace_start() {
    TRACE.lock().unwrap().clear();
    TRACING.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cover")]
pub fn trace_take() -> Vec<u32> {
    TRACING.store(false, std::sync::atomic::Ordering::Relaxed);
    std::mem::take(&mut *TRACE.lock().unwrap())
}

/// Times each block was entered, since the process started.
#[cfg(feature = "cover")]
pub fn hits(pc: u32) -> u64 {
    HITS.lock()
        .unwrap()
        .as_ref()
        .and_then(|h| h.get(&pc).copied())
        .unwrap_or(0)
}

#[derive(Clone, Default, Debug)]
pub struct Cpu {
    pub r: [u32; 16],
    pub f: Flags,
    /// FPA11 register file. The engine uses 379 floating-point instructions
    /// in total, all of them here.
    pub fp: [f64; 8],
    pub fpsr: u32,
}

// ---- FPA11, as GCC emits it for ARM OABI ---------------------------------
//
// Mirrors `loqrs` `crates/armemu/src/fpa.rs`, which is proved bit-exact
// against the ARM oracle. Two layout facts drive it: doubles in memory are
// word-swapped (most significant word at the lower address), and LFM/SFM
// move 12 bytes per register.

/// Immediates the Fm field selects when bit 3 is set.
const FPA_CONST: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 10.0];

/// Marker in the spare word of an extended slot, matching `loqrs`.
const EXT_TAG: u32 = 0x464C_4F51;

#[inline]
fn round_to(prec: u32, x: f64) -> f64 {
    if prec == 0 {
        x as f32 as f64
    } else {
        x
    }
}

#[inline]
fn operand(c: &Cpu, fm: usize, konst: i32) -> f64 {
    if konst >= 0 {
        FPA_CONST[konst as usize]
    } else {
        c.fp[fm]
    }
}

pub fn fpa_do(
    c: &mut Cpu,
    fd: usize,
    fnr: usize,
    fm: usize,
    konst: i32,
    sel: u32,
    monadic: bool,
    prec: u32,
) {
    let b = operand(c, fm, konst);
    let res = if monadic {
        match sel {
            0x0 => b,
            0x1 => -b,
            0x2 => b.abs(),
            0x3 => b.trunc(),
            0x4 => b.sqrt(),
            0x5 => b.log10(),
            0x6 => b.ln(),
            0x7 => b.exp(),
            0x8 => b.sin(),
            0x9 => b.cos(),
            0xA => b.tan(),
            0xB => b.asin(),
            0xC => b.acos(),
            0xD => b.atan(),
            0xE => b.round(),
            _ => b,
        }
    } else {
        let a = c.fp[fnr];
        match sel {
            0x0 => a + b,
            0x1 => a * b,
            0x2 => a - b,
            0x3 => b - a,
            0x4 => a / b,
            0x5 => b / a,
            0x6 => a.powf(b),
            0x7 => b.powf(a),
            0x8 => a % b,
            0x9 => a * b,
            0xA => a / b,
            0xB => b / a,
            0xC => a.atan2(b),
            _ => panic!("FPA opcode {sel:#x}"),
        }
    };
    c.fp[fd] = round_to(prec, res);
}

pub fn fpa_cmp(c: &mut Cpu, fnr: usize, fm: usize, konst: i32, negate: bool) {
    let a = c.fp[fnr];
    let b = operand(c, fm, konst);
    let b = if negate { -b } else { b };
    c.f = if a.is_nan() || b.is_nan() {
        Flags {
            n: false,
            z: false,
            c: true,
            v: true,
        }
    } else {
        Flags {
            n: a < b,
            z: a == b,
            c: a >= b,
            v: false,
        }
    };
}

pub fn fpa_flt(c: &mut Cpu, fnr: usize, v: u32, prec: u32) {
    c.fp[fnr] = round_to(prec, v as i32 as f64);
}

pub fn fpa_fix(c: &mut Cpu, fm: usize, round: u32) -> u32 {
    let x = c.fp[fm];
    let r = match round {
        0 => round_half_even(x),
        1 => x.ceil(),
        2 => x.floor(),
        _ => x.trunc(),
    };
    let v = if r.is_nan() {
        0
    } else if r >= 2147483647.0 {
        i32::MAX
    } else if r <= -2147483648.0 {
        i32::MIN
    } else {
        r as i32
    };
    v as u32
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

pub fn fpa_ldf(c: &mut Cpu, m: &mut Mem, fd: usize, prec: u32, a: u32) {
    c.fp[fd] = match prec {
        0 => f32::from_bits(m.r32(a)) as f64,
        1 => f64::from_bits(((m.r32(a) as u64) << 32) | m.r32(a + 4) as u64),
        _ => read_ext(m, a),
    };
}

pub fn fpa_stf(c: &mut Cpu, m: &mut Mem, fd: usize, prec: u32, a: u32) {
    let v = c.fp[fd];
    match prec {
        0 => m.w32(a, (v as f32).to_bits()),
        1 => {
            let bits = v.to_bits();
            m.w32(a, (bits >> 32) as u32);
            m.w32(a + 4, bits as u32);
        }
        _ => write_ext(m, a, v),
    }
}

pub fn fpa_lfm(c: &mut Cpu, m: &mut Mem, fd: usize, count: usize, a: u32) {
    for i in 0..count {
        c.fp[(fd + i) & 7] = read_ext(m, a + 12 * i as u32);
    }
}

pub fn fpa_sfm(c: &mut Cpu, m: &mut Mem, fd: usize, count: usize, a: u32) {
    for i in 0..count {
        let v = c.fp[(fd + i) & 7];
        write_ext(m, a + 12 * i as u32, v);
    }
}

fn read_ext(m: &Mem, a: u32) -> f64 {
    f64::from_bits(((m.r32(a + 4) as u64) << 32) | m.r32(a) as u64)
}

fn write_ext(m: &mut Mem, a: u32, v: f64) {
    let bits = v.to_bits();
    m.w32(a, bits as u32);
    m.w32(a + 4, (bits >> 32) as u32);
    m.w32(a + 8, EXT_TAG);
}

/// Imports the runtime cannot provide, and indirect calls. Both see the
/// registers exactly as the callee would: arguments in `r0..r3` and on the
/// stack, the result to be left in `r0`.
pub trait Host {
    fn import(&mut self, name: &str, c: &mut Cpu, m: &mut Mem);
    fn indirect(&mut self, target: u32, c: &mut Cpu, m: &mut Mem);
}

/// A host for code that is not expected to reach any import.
pub struct NoHost;

impl Host for NoHost {
    fn import(&mut self, name: &str, _c: &mut Cpu, _m: &mut Mem) {
        panic!("translated code called the import `{name}`, which has no host");
    }

    fn indirect(&mut self, target: u32, _c: &mut Cpu, _m: &mut Mem) {
        panic!("translated code made an indirect call to 0x{target:08x}, which has no host");
    }
}

/// A barrel shift whose amount is only known at run time.
///
/// `kind` is 0 lsl, 1 lsr, 2 asr, 3 ror. Only the low byte of the amount is
/// used, and the boundary cases are the ones that matter: a shift of 32 is
/// not a Rust shift, and a shift of 0 leaves the carry alone.
#[inline]
pub fn shift_reg(kind: u8, v: u32, amount: u32, cin: bool) -> (u32, bool) {
    let n = amount & 0xFF;
    if n == 0 {
        return (v, cin);
    }
    match kind {
        0 => match n {
            1..=31 => (v << n, (v >> (32 - n)) & 1 != 0),
            32 => (0, v & 1 != 0),
            _ => (0, false),
        },
        1 => match n {
            1..=31 => (v >> n, (v >> (n - 1)) & 1 != 0),
            32 => (0, v >> 31 != 0),
            _ => (0, false),
        },
        2 => {
            if n < 32 {
                (((v as i32) >> n) as u32, (v >> (n - 1)) & 1 != 0)
            } else {
                (((v as i32) >> 31) as u32, v >> 31 != 0)
            }
        }
        _ => {
            let k = n & 31;
            if k == 0 {
                (v, v >> 31 != 0)
            } else {
                (v.rotate_right(k), (v >> (k - 1)) & 1 != 0)
            }
        }
    }
}

/// Guest memory. Two implementations, one API; see [`mem`].
pub use mem::Mem;

pub fn memcpy(c: &mut Cpu, m: &mut Mem) {
    let (d, s, n) = (c.r[0], c.r[1], c.r[2]);
    let tmp = m.read(s, n as usize);
    m.write(d, &tmp);
}

pub fn memmove(c: &mut Cpu, m: &mut Mem) {
    memcpy(c, m);
}

pub fn memset(c: &mut Cpu, m: &mut Mem) {
    let (d, v, n) = (c.r[0], c.r[1] as u8, c.r[2]);
    for k in 0..n {
        m.w8(d + k, v);
    }
}

pub fn strlen(c: &mut Cpu, m: &mut Mem) {
    let mut k = 0;
    while m.r8(c.r[0] + k) != 0 {
        k += 1;
    }
    c.r[0] = k;
}

fn cmp_n(m: &Mem, a: u32, b: u32, n: u32) -> u32 {
    for k in 0..n {
        let (x, y) = (m.r8(a + k), m.r8(b + k));
        if x != y || x == 0 {
            return (x as i32 - y as i32) as u32;
        }
    }
    0
}

pub fn strcmp(c: &mut Cpu, m: &mut Mem) {
    c.r[0] = cmp_n(m, c.r[0], c.r[1], u32::MAX);
}

pub fn strncmp(c: &mut Cpu, m: &mut Mem) {
    c.r[0] = cmp_n(m, c.r[0], c.r[1], c.r[2]);
}

pub fn strstr(c: &mut Cpu, m: &mut Mem) {
    let (hay, needle) = (c.r[0], c.r[1]);
    let len = |a: u32| {
        let mut k = 0;
        while m.r8(a + k) != 0 {
            k += 1;
        }
        k
    };
    let (h, n) = (len(hay), len(needle));
    c.r[0] = 0;
    if n <= h {
        for s in 0..=h - n {
            if (0..n).all(|k| m.r8(hay + s + k) == m.r8(needle + k)) {
                c.r[0] = hay + s;
                return;
            }
        }
    }
}

pub fn strcpy(c: &mut Cpu, m: &mut Mem) {
    let (d, s) = (c.r[0], c.r[1]);
    let mut k = 0;
    loop {
        let b = m.r8(s + k);
        m.w8(d + k, b);
        if b == 0 {
            break;
        }
        k += 1;
    }
}

pub fn strncpy(c: &mut Cpu, m: &mut Mem) {
    let (d, s, n) = (c.r[0], c.r[1], c.r[2]);
    let mut end = false;
    for k in 0..n {
        let b = if end { 0 } else { m.r8(s + k) };
        end |= b == 0;
        m.w8(d + k, b);
    }
}
