//! `xtask sigprobe` — hold `loqng::sig::sequens` to the ARM original.
//!
//! Each function of the `SEQUENS` frame path was translated by
//! `tools/armxlate.py`. Here each is called both ways on the same inputs: the
//! ARM through the oracle, the translation natively, with every buffer
//! mirrored at the same address in `xrt::Mem`, so the two see byte-identical
//! memory. `r0` and every buffer must agree afterwards.
//!
//! ```text
//! cargo run -p xtask --release -- sigprobe [cases]
//! ```
//!
//! Argument lists are read off each prologue (`notes/stage-sig.md` §5.3, §5.6):
//!
//! ```text
//! bsa_border   0x4cc8c (scratch, fmt, src, srcLen, dstLen, swap, crypt) -> m
//! bsa_consumed 0x4cfd0 (scratch, m, srcLen, dstLen) -> k
//! psola_body   0x4fda0 (srcFmt, dstFmt, src, dst, k, gain, swap, crypt)
//! stretch_body 0x4d300 (srcFmt, dstFmt, src, dst, srcLen, dstLen, gain, swap, crypt)
//! win_store    0x4e168 (win, src, dst, N, count, srcFmt, dstFmt, swap, crypt, gain)
//! win_add      0x4ee58 (the same)
//! __udivsi3    0x8bbb8 (a, b)
//! ```

use loqng::eng;
use loqng::sig::sequens as sq;
use loqng::xrt::{Cpu, Mem, NoHost};
use loqng_oracle::{Oracle, OracleConfig};

use crate::Paths;

const MODULE: &str = "LoqTTS6.so";
const SIG_COSENO: u32 = 0x9923e;
const SIG_COSCUBO: u32 = 0x98d56;

const SRC_BYTES: u32 = 0x4000;
const DST_BYTES: u32 = 0x4000;
const SCRATCH_BYTES: u32 = 0x4000;
const STACK_TOP: u32 = 0x7f00_0000;
const STACK_BYTES: u32 = 0x1_0000;

type Native = fn(&mut Cpu, &mut Mem, &mut dyn loqng::xrt::Host);

struct Rig {
    o: Oracle,
    base: u32,
    src: u32,
    dst: u32,
    scratch: u32,
}

/// One argument: a plain value, or an address in the module's own data,
/// which lives at `base + a` in the guest and at `a` in the translation.
#[derive(Clone, Copy)]
enum Arg {
    V(u32),
    Module(u32),
}

impl Rig {
    fn native_mem(&self, src: &[u8], dst: &[u8], scratch: &[u8]) -> Mem {
        self.wrap(Mem::with_image(sq::IMAGE), src, dst, scratch)
    }

    /// The same buffers over the whole-engine image. `eng` is translated at
    /// the oracle's bases, so it takes the oracle's argument values unchanged
    /// — which is what makes this a second, independent check on the same
    /// ARM code through a different translator.
    fn eng_mem(&self, src: &[u8], dst: &[u8], scratch: &[u8]) -> Mem {
        self.wrap(Mem::with_image(&eng::regions()), src, dst, scratch)
    }

    fn wrap(&self, mut m: Mem, src: &[u8], dst: &[u8], scratch: &[u8]) -> Mem {
        m.map(self.src, src.to_vec());
        m.map(self.dst, dst.to_vec());
        m.map(self.scratch, scratch.to_vec());
        m.map(STACK_TOP - STACK_BYTES, vec![0; STACK_BYTES as usize]);
        m
    }

    fn guest_load(&mut self, src: &[u8], dst: &[u8], scratch: &[u8]) -> Result<(), String> {
        self.o.write_bytes(self.src, src)?;
        self.o.write_bytes(self.dst, dst)?;
        self.o.write_bytes(self.scratch, scratch)
    }

    fn guest_call(&mut self, at: u32, args: &[Arg]) -> Result<u32, String> {
        let base = self.base;
        let v: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Arg::V(x) => x,
                Arg::Module(x) => base + x,
            })
            .collect();
        self.o.call_at(MODULE, at, &v)
    }

    fn guest_state(&self) -> Result<[Vec<u8>; 3], String> {
        Ok([
            self.o.read_bytes(self.src, SRC_BYTES as usize)?,
            self.o.read_bytes(self.dst, DST_BYTES as usize)?,
            self.o.read_bytes(self.scratch, SCRATCH_BYTES as usize)?,
        ])
    }

    fn native_state(&self, m: &Mem) -> [Vec<u8>; 3] {
        [
            m.read(self.src, SRC_BYTES as usize),
            m.read(self.dst, DST_BYTES as usize),
            m.read(self.scratch, SCRATCH_BYTES as usize),
        ]
    }
}

fn native_call(f: Native, m: &mut Mem, args: &[Arg]) -> u32 {
    let v: Vec<u32> = args
        .iter()
        .map(|a| match *a {
            Arg::V(x) | Arg::Module(x) => x,
        })
        .collect();
    let mut c = Cpu::default();
    let extra = v.len().saturating_sub(4) as u32;
    let sp = (STACK_TOP - 0x100 - 4 * extra) & !7;
    for (k, &x) in v.iter().skip(4).enumerate() {
        m.w32(sp + 4 * k as u32, x);
    }
    for (k, &x) in v.iter().take(4).enumerate() {
        c.r[k] = x;
    }
    c.r[13] = sp;
    c.r[14] = 0xdead_beef;
    f(&mut c, m, &mut NoHost);
    c.r[0]
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u32 {
        (self.next() % n) as u32
    }

    fn bytes(&mut self, n: u32) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }

    fn fmt(&mut self) -> u32 {
        [b'l', b'l', b'a', b'u'][self.below(4) as usize] as u32
    }

    fn gain(&mut self) -> u32 {
        if self.below(3) == 0 {
            13107
        } else {
            self.below(0x10000)
        }
    }
}

/// Speech-like 16-bit samples: a few sinusoids and noise. Pure noise would
/// leave the epoch search in `bsa_border` nothing to find.
fn voiced(r: &mut Rng, n: u32) -> Vec<u8> {
    let period = 40.0 + r.below(300) as f64;
    let amp = 500.0 + r.below(12000) as f64;
    let mut out = Vec::with_capacity(n as usize);
    for k in 0..n / 2 {
        let t: f64 = k as f64 / period * std::f64::consts::TAU;
        let s = amp * (t.sin() + 0.5 * (2.0 * t).sin() + 0.25 * (3.0 * t).sin())
            + (r.below(400) as f64 - 200.0);
        out.extend_from_slice(&(s.clamp(-32768.0, 32767.0) as i16).to_le_bytes());
    }
    out
}

struct Tally {
    name: &'static str,
    ok: u64,
    bad: u64,
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let n: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(2000);
    println!("opening the ARM engine");
    let cfg = OracleConfig::new(&paths.lib_dir, &paths.data_dir).interpreted();
    let mut o = Oracle::open(&cfg)?;
    let base = o.module_base(MODULE).ok_or("LoqTTS6.so is not loaded")?;
    let src = o.alloc(SRC_BYTES)?;
    let dst = o.alloc(DST_BYTES)?;
    let scratch = o.alloc(SCRATCH_BYTES)?;
    let mut rig = Rig {
        o,
        base,
        src,
        dst,
        scratch,
    };
    println!("{MODULE} at 0x{base:08x}; buffers at 0x{src:08x} 0x{dst:08x} 0x{scratch:08x}\n");

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut tallies: Vec<Tally> = [
        "__udivsi3",
        "bsa_border+bsa_consumed",
        "psola_body",
        "stretch_body",
        "win_store",
        "win_add",
    ]
    .iter()
    .map(|&name| Tally {
        name,
        ok: 0,
        bad: 0,
    })
    .collect();
    let t0 = std::time::Instant::now();

    for case in 0..n {
        for (which, t) in tallies.iter_mut().enumerate() {
            let srcb = if rng.below(4) == 0 {
                rng.bytes(SRC_BYTES)
            } else {
                voiced(&mut rng, SRC_BYTES)
            };
            let dstb = rng.bytes(DST_BYTES);
            let scrb = vec![0u8; SCRATCH_BYTES as usize];
            rig.guest_load(&srcb, &dstb, &scrb)?;
            let mut m = rig.native_mem(&srcb, &dstb, &scrb);
            let mut me = rig.eng_mem(&srcb, &dstb, &scrb);
            // `eng` takes the oracle's own argument values, so the argument
            // list is built once and used by both.
            let eng_run = |f: Native, a: &[Arg], me: &mut Mem| {
                let v: Vec<Arg> = a
                    .iter()
                    .map(|x| match *x {
                        Arg::V(k) => Arg::V(k),
                        Arg::Module(k) => Arg::V(base + k),
                    })
                    .collect();
                native_call(f, me, &v)
            };

            let src_len = 8 + rng.below(400);
            let dst_len = 8 + rng.below(400);
            let (swap, crypt) = (rng.below(2), rng.below(2));
            let (g, s0) = (Arg::V, rig.src);

            #[allow(unused)]
            let mut eng_got = None;
            let (want, got) = match which {
                0 => {
                    let a = [
                        g(rng.next() as u32),
                        g(1 + rng.below(0xffff) * (1 + rng.below(3))),
                    ];
                    eng_got = Some(eng_run(eng::f_0018bbb8, &a, &mut me));
                    (
                        rig.guest_call(0x8bbb8, &a)?,
                        native_call(sq::f_8bbb8, &mut m, &a),
                    )
                }
                1 => {
                    let fmt = rng.fmt();
                    let a = [
                        g(rig.scratch),
                        g(fmt),
                        g(s0),
                        g(src_len),
                        g(dst_len),
                        g(swap),
                        g(crypt),
                    ];
                    let gm = rig.guest_call(0x4cc8c, &a)?;
                    let nm = native_call(sq::f_4cc8c, &mut m, &a);
                    let em = eng_run(eng::f_0014cc8c, &a, &mut me);
                    // Chained as `sub_4bd38` does: only a small `m` goes on.
                    let b = [g(rig.scratch), g(gm), g(src_len), g(dst_len)];
                    let chain = gm == nm && (1..=0x1f).contains(&(gm & 0xffff));
                    let (gk, nk, ek) = if chain {
                        (
                            rig.guest_call(0x4cfd0, &b)?,
                            native_call(sq::f_4cfd0, &mut m, &b),
                            eng_run(eng::f_0014cfd0, &b, &mut me),
                        )
                    } else {
                        (0, 0, 0)
                    };
                    eng_got = Some(em ^ ek.rotate_left(16));
                    (gm ^ gk.rotate_left(16), nm ^ nk.rotate_left(16))
                }
                2 => {
                    let k = rng.below(src_len as u64 + 1);
                    let a = [
                        g(rng.fmt()),
                        g(rng.fmt()),
                        g(s0),
                        g(rig.dst),
                        g(k),
                        g(rng.gain()),
                        g(swap),
                        g(crypt),
                    ];
                    eng_got = Some(eng_run(eng::f_0014fda0, &a, &mut me));
                    (
                        rig.guest_call(0x4fda0, &a)?,
                        native_call(sq::f_4fda0, &mut m, &a),
                    )
                }
                3 => {
                    let a = [
                        g(rng.fmt()),
                        g(rng.fmt()),
                        g(s0),
                        g(rig.dst),
                        g(src_len),
                        g(dst_len),
                        g(rng.gain()),
                        g(swap),
                        g(crypt),
                    ];
                    eng_got = Some(eng_run(eng::f_0014d300, &a, &mut me));
                    (
                        rig.guest_call(0x4d300, &a)?,
                        native_call(sq::f_4d300, &mut m, &a),
                    )
                }
                _ => {
                    let win = if rng.below(2) == 0 {
                        SIG_COSENO
                    } else {
                        SIG_COSCUBO
                    };
                    let big_n = 1 + rng.below(640);
                    let count = rng.below(big_n as u64 + 1);
                    let a = [
                        Arg::Module(win),
                        g(s0),
                        g(rig.dst),
                        g(big_n),
                        g(count),
                        g(rng.fmt()),
                        g(rng.fmt()),
                        g(swap),
                        g(crypt),
                        g(rng.gain()),
                    ];
                    let at = if which == 4 { 0x4e168 } else { 0x4ee58 };
                    let f: Native = if which == 4 { sq::f_4e168 } else { sq::f_4ee58 };
                    let e: Native = if which == 4 {
                        eng::f_0014e168
                    } else {
                        eng::f_0014ee58
                    };
                    eng_got = Some(eng_run(e, &a, &mut me));
                    (rig.guest_call(at, &a)?, native_call(f, &mut m, &a))
                }
            };

            let gs = rig.guest_state()?;
            let ns = rig.native_state(&m);
            let es = rig.native_state(&me);
            let eng_ok = eng_got == Some(want) && es == gs;
            if want == got && gs == ns && eng_ok {
                t.ok += 1;
            } else {
                t.bad += 1;
                if t.bad <= 3 {
                    let which_buf = (0..3).find(|&b| gs[b] != ns[b]);
                    let at = which_buf.and_then(|b| {
                        gs[b]
                            .iter()
                            .zip(ns[b].iter())
                            .position(|(x, y)| x != y)
                            .map(|p| (["src", "dst", "scratch"][b], p))
                    });
                    println!(
                        "  {} case {case}: r0 arm 0x{want:08x} sequens \
                              0x{got:08x} eng {:?}; first differing byte {:?}; \
                              eng buffers {}",
                        t.name,
                        eng_got.map(|v| format!("0x{v:08x}")),
                        at,
                        if es == gs { "agree" } else { "differ" }
                    );
                }
            }
        }
    }

    println!(
        "{n} cases per function, {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    println!(
        "each case checks the ARM against BOTH translations: \
              `sig::sequens` (armxlate) and `eng` (engxlate)\n"
    );
    let mut bad = 0;
    for t in &tallies {
        println!(
            "  {:<26} {:>7}/{:<7} {}",
            t.name,
            t.ok,
            t.ok + t.bad,
            if t.bad == 0 { "BIT-EXACT" } else { "DIFFERS" }
        );
        bad += t.bad;
    }
    #[cfg(feature = "xrt-cover")]
    {
        println!("\nblock coverage (translated side):");
        let (mut all, mut hot) = (0, 0);
        for (f, blocks) in sq::BLOCKS {
            let cold: Vec<String> = blocks
                .iter()
                .filter(|&&b| loqng::xrt::hits(b) == 0)
                .map(|b| format!("{b:05x}"))
                .collect();
            all += blocks.len();
            hot += blocks.len() - cold.len();
            println!(
                "  0x{f:05x}  {:>4}/{:<4}",
                blocks.len() - cold.len(),
                blocks.len()
            );
            if args.iter().any(|a| a == "-v") {
                for c in cold.chunks(12) {
                    println!("      {}", c.join(" "));
                }
            }
        }
        println!("  total    {hot}/{all}");
    }

    if bad > 0 {
        return Err(format!("{bad} case(s) differ"));
    }
    println!("\nBIT-EXACT: the translated SEQUENS functions match the ARM");
    Ok(())
}
