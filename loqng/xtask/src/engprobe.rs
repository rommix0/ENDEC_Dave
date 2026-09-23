//! `xtask engprobe` — hold the whole-module translation to the ARM engine.
//!
//! `tools/engxlate.py` translates every function a module exposes, at the
//! bases the oracle uses, with its memory image relocated at build time. This
//! gates that output the only way that means anything: call a translated
//! function and the ARM one with identical inputs and compare.
//!
//! The first subject is `FonemaLargo2Stretto_US`, because it is already
//! translated a second, independent way (`tools/fonxlate.py`, `fon::exact`)
//! and proved bit-exact over a million windows. Agreement between two
//! separate translators and the ARM original is a strong check on both.
//!
//! ```text
//! python tools/engxlate.py crates/loqng/src/eng LoqEnglish6.9.so
//! cargo run -p xtask --release -- engprobe [cases]
//! ```

use loqng::eng;
use loqng::fon::narrow::Phone;
use loqng::fon::phontab::PHONES;
use loqng::xrt::{Cpu, Mem, NoHost};

use crate::fonprobe::Probe;
use crate::Paths;

/// `LoqEnglish6.9.so` + 0x24680, at the base the oracle reports.
const MAPPER: u32 = 0x001F_4680;

const SCRATCH: u32 = 0x5000_0000;
const STACK_TOP: u32 = 0x7F00_0000;
const STACK_BYTES: u32 = 0x1_0000;

const MAX_PHONES: u32 = 1024;
const MAX_WORDS: u32 = 64;
const STYLE: u32 = 3;

/// Offsets inside the one scratch region, mirroring `fonprobe::Probe::open`.
struct Layout {
    ctx: u32,
    out: u32,
    ph: u32,
    phrase: u32,
    elems: u32,
    cfg: u32,
    p4: u32,
    words: u32,
    size: u32,
}

impl Layout {
    fn new() -> Layout {
        let mut at = SCRATCH;
        let mut take = |n: u32| {
            let a = at;
            at += (n + 15) & !15;
            a
        };
        let ctx = take(0x100);
        let out = take(MAX_PHONES * 16 + 64);
        let ph = take(MAX_PHONES * 8);
        let phrase = take(28 * 4);
        let elems = take(MAX_WORDS * 20);
        let cfg = take(0x400);
        let p4 = take(16);
        let words = take(MAX_WORDS * 64);
        Layout {
            ctx,
            out,
            ph,
            phrase,
            elems,
            cfg,
            p4,
            words,
            size: at - SCRATCH,
        }
    }
}

fn fresh(l: &Layout) -> Mem {
    let mut m = Mem::with_image(&eng::regions());
    m.map(SCRATCH, vec![0; l.size as usize]);
    m.map(STACK_TOP - STACK_BYTES, vec![0; STACK_BYTES as usize]);
    m
}

/// The mapper over a whole phrase, driven as `LoqTTS6` drives it.
fn narrow(l: &Layout, phones: &[Phone], words: &[&str], elem12: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let mut m = fresh(l);
    m.w32(l.ctx + 0x04, l.out);
    m.w32(l.ctx + 0x10, l.ph);
    m.w32(l.ctx + 0x28, l.phrase);
    m.w32(l.ctx + 0x2C, l.elems);
    m.w32(l.ctx + 0x40, l.cfg);
    m.w32(l.cfg + 0x30C, STYLE);
    m.w32(l.phrase + 0x04, words.len() as u32);

    for (k, w) in words.iter().enumerate() {
        let at = l.words + k as u32 * 64;
        for (q, &b) in w.as_bytes().iter().take(63).enumerate() {
            m.w8(at + q as u32, b);
        }
        let e = l.elems + k as u32 * 20;
        m.w32(e + 0x04, at);
        m.w8(e + 0x12, elem12.get(k).copied().unwrap_or(0));
    }
    for (k, p) in phones.iter().enumerate() {
        m.w32(l.ph + k as u32 * 8, p.word);
        m.w8(l.ph + k as u32 * 8 + 4, p.code);
    }

    let (mut i, mut last_j) = (0u32, 0u32);
    let mut owner = Vec::new();
    while (i as usize) < phones.len() {
        let before = i;
        let pos = phones[i as usize].word;
        let mut c = Cpu::default();
        let sp = (STACK_TOP - 64) & !7;
        m.w32(sp, l.ph);
        c.r[0] = l.ctx;
        c.r[1] = pos;
        c.r[2] = i;
        c.r[3] = l.p4;
        c.r[13] = sp;
        c.r[14] = 0xDEAD_BEEF;
        eng::f_001f4680(&mut c, &mut m, &mut NoHost);
        i = c.r[0];
        last_j = m.r32(l.p4);
        while (owner.len() as u32) < last_j {
            owner.push(before);
        }
    }
    (
        (0..last_j).map(|k| m.r8(l.out + k * 16 + 6)).collect(),
        owner,
    )
}

/// The first block where the two translations of this function diverge.
///
/// `fon::exact` is proved bit-exact over a million windows, so where the two
/// block sequences part is where the whole-module translator got a branch
/// wrong. Both record through `xrt::hit`; only the load base differs.
#[cfg(feature = "xrt-cover")]
fn where_they_part(l: &Layout, ph: &[Phone], words: &[&str], elem: &[u8]) {
    use loqng::xrt::{trace_start, trace_take};
    trace_start();
    let _ = loqng::fon::exact::narrow(ph, words, elem);
    let a = trace_take();
    trace_start();
    let _ = narrow(l, ph, words, elem);
    let b: Vec<u32> = trace_take()
        .iter()
        .map(|v| v.wrapping_sub(0x001D_0000))
        .collect();
    println!(
        "\n  block trace: {} from fon::exact, {} from eng",
        a.len(),
        b.len()
    );
    let at = a.iter().zip(b.iter()).position(|(x, y)| x != y);
    match at {
        None => println!("  the traces agree; the difference is in the data"),
        Some(k) => {
            for i in k.saturating_sub(6)..k {
                println!("    both  0x{:05x}", a[i]);
            }
            println!("    fon   0x{:05x}   <- they part here", a[k]);
            println!("    eng   0x{:05x}", b[k]);
        }
    }
}

fn sym(c: u8) -> &'static str {
    PHONES.get(c as usize).map(|p| p.symbol).unwrap_or("?")
}

fn show(v: &[u8]) -> String {
    v.iter().map(|&c| sym(c)).collect::<Vec<_>>().join(" ")
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let n: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let l = Layout::new();
    println!(
        "mapper at 0x{MAPPER:08x}, image {} regions",
        eng::regions().len()
    );
    println!("opening the ARM engine");
    let mut probe = Probe::open(paths)?;

    let trace = paths
        .corpus
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("build")
        .join("fon_trace.txt");
    let phrases = crate::fon::parse(&trace)?;
    let mut bad = 0;
    for p in &phrases {
        let words: Vec<&str> = p.words.iter().map(|s| s.as_str()).collect();
        let want = probe.run_owned(&p.broad, &words, &p.elem12)?;
        let got = narrow(&l, &p.broad, &words, &p.elem12);
        if got != want {
            bad += 1;
            if bad <= 3 {
                println!(
                    "  {} {:?}\n    arm  {}\n    eng  {}",
                    p.capture,
                    p.text,
                    show(&want.0),
                    show(&got.0)
                );
            }
        }
    }
    println!(
        "corpus: {}/{} phrases identical to the ARM function",
        phrases.len() - bad,
        phrases.len()
    );

    let mut vocab: Vec<&str> = phrases
        .iter()
        .flat_map(|p| p.words.iter().map(|s| s.as_str()))
        .filter(|w| w.len() < 60)
        .collect();
    vocab.sort();
    vocab.dedup();
    let lits = loqng::fon::exact::literals();

    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move |k: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % k
    };
    let t0 = std::time::Instant::now();
    let mut diff = 0u64;
    for case in 0..n {
        let nw = 1 + rnd(8) as usize;
        let owned: Vec<String> = (0..nw)
            .map(|_| {
                let v = vocab[rnd(vocab.len() as u64) as usize];
                let s = &lits[rnd(lits.len() as u64) as usize];
                match rnd(4) {
                    0 => s.clone(),
                    1 => format!("{v}{s}"),
                    _ => v.to_string(),
                }
            })
            .map(|s| if s.len() > 60 { s[..60].to_string() } else { s })
            .collect();
        let words: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let elem: Vec<u8> = (0..nw).map(|_| rnd(4) as u8).collect();
        let len = 1 + rnd(30) as usize;
        let mut w = 0u32;
        let ph: Vec<Phone> = (0..len)
            .map(|_| {
                if (w as usize) + 1 < nw && rnd(4) == 0 {
                    w += 1;
                }
                Phone {
                    word: w,
                    code: rnd(PHONES.len() as u64) as u8,
                }
            })
            .collect();

        let want = probe.run_owned(&ph, &words, &elem)?;
        let got = narrow(&l, &ph, &words, &elem);
        if want != got {
            diff += 1;
            if diff <= 3 {
                println!("  case {case}: words {words:?}");
                println!("    arm  {}", show(&want.0));
                println!("    eng  {}", show(&got.0));
            }
            #[cfg(feature = "xrt-cover")]
            if diff == 1 {
                where_they_part(&l, &ph, &words, &elem);
                return Err("traced the first divergence".into());
            }
        }
    }
    println!(
        "random: {}/{n} windows identical ({:.1} s)",
        n - diff,
        t0.elapsed().as_secs_f64()
    );

    if bad > 0 || diff > 0 {
        return Err(format!(
            "{bad} corpus phrase(s) and {diff} window(s) differ"
        ));
    }
    println!("\nBIT-EXACT: the whole-module translation matches the ARM engine");
    Ok(())
}
