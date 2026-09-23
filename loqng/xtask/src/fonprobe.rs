//! `xtask fonprobe` — drive `FonemaLargo2Stretto_US` as a pure function.
//!
//! The static route to this function's rules is an open-ended discovery loop:
//! every pass over the disassembly turns up another compiler idiom that had
//! been silently mis-read — tail-merged stores, inlined `strcmp`, `cmn` bounds
//! checks, ordered code compares, a whole pre-pass before the dispatch. There
//! is no point at which that loop can be declared finished.
//!
//! So stop deriving and start measuring. The mapper is a pure function of a
//! bounded window, and `Oracle` can call a guest address with a constructed
//! context, so it can be asked directly:
//!
//! ```text
//! cargo run -p xtask -- fonprobe           # self-test against the engine
//! ```
//!
//! A probe is **microseconds**, against 1.4 s for a whole synthesis under
//! watchpoints. That is what makes mutation-based guard discovery practical:
//! take a window where a rule fires, perturb one slot, ask again.
//!
//! ## The context the function needs
//!
//! Read off the prologue at `0x24680` (`notes/eng-fonema.md` §2):
//!
//! ```text
//! ctx[0x04]  out[]      narrow array, 16-byte records, code at +6
//! ctx[0x28]  phrases    28-byte records; ctx[0x38] selects one
//! ctx[0x2c]  elems      20-byte LesElem; +0x04 is the spelling, +0x12 the
//!                       byte the `` `OR: `` rule tests
//! ctx[0x40]  cfg        the style lives at +0x30c; Dave is 3
//! phrase[0x04]          entry count
//! phrase[0x18]          index of this phrase's first entry
//! ```
//!
//! The broad array is passed as the fifth argument rather than read from the
//! context, so it is handed over directly.

use loqng::fon::narrow::{narrow, Ctx, Phone};
use loqng::fon::phontab::PHONES;
use loqng_oracle::{Oracle, OracleConfig};

use crate::Paths;

const MODULE: &str = "LoqEnglish6.9.so";
const MAPPER: u32 = 0x24680;
const STYLE: u32 = 3;

const MAX_PHONES: u32 = 1024;
const MAX_WORDS: u32 = 64;

pub struct Probe {
    o: Oracle,
    ctx: u32,
    out: u32,
    ph: u32,
    phrase: u32,
    elems: u32,
    p4: u32,
    words: Vec<u32>,
}

impl Probe {
    pub fn open(paths: &Paths) -> Result<Self, String> {
        let cfg = OracleConfig::new(&paths.lib_dir, &paths.data_dir).interpreted();
        let mut o = Oracle::open(&cfg)?;

        let ctx = o.alloc(0x100)?;
        let out = o.alloc(MAX_PHONES * 16)?;
        let ph = o.alloc(MAX_PHONES * 8)?;
        let phrase = o.alloc(28 * 4)?;
        let elems = o.alloc(MAX_WORDS * 20)?;
        let cfgblk = o.alloc(0x400)?;
        let p4 = o.alloc(4)?;

        o.write_u32(ctx + 0x04, out)?;
        o.write_u32(ctx + 0x10, ph)?;
        o.write_u32(ctx + 0x20, 0)?;
        o.write_u32(ctx + 0x28, phrase)?;
        o.write_u32(ctx + 0x2C, elems)?;
        o.write_u32(ctx + 0x38, 0)?;
        o.write_u32(ctx + 0x40, cfgblk)?;
        o.write_u32(cfgblk + 0x30C, STYLE)?;
        o.write_u32(phrase + 0x18, 0)?;

        // Spellings are allocated once and rewritten in place, so a probe
        // costs no allocation.
        let mut words = Vec::new();
        for _ in 0..MAX_WORDS {
            words.push(o.alloc(64)?);
        }
        Ok(Probe {
            o,
            ctx,
            out,
            ph,
            phrase,
            elems,
            p4,
            words,
        })
    }

    /// Run the mapper over a whole phrase and return the narrow codes, exactly
    /// as `LoqTTS6` drives it: one call per broad phone, `i` from the return
    /// value and `j` from `*p4`.
    pub fn run(
        &mut self,
        phones: &[Phone],
        words: &[&str],
        elem12: &[u8],
    ) -> Result<Vec<u8>, String> {
        Ok(self.run_owned(phones, words, elem12)?.0)
    }

    /// The narrow codes, plus the broad index that produced each one.
    ///
    /// The owner is what makes mutation testing exact. Asking only "is
    /// `out[slot]` still `X`" is wrong twice over: an insertion earlier in the
    /// phrase slides the slot, and a change to any earlier phone alters its
    /// own output. Knowing which broad phone owns the slot removes both.
    pub fn run_owned(
        &mut self,
        phones: &[Phone],
        words: &[&str],
        elem12: &[u8],
    ) -> Result<(Vec<u8>, Vec<u32>), String> {
        // Zero-padded past the end: the function reads ahead without a length
        // check, and a stale tail from the previous probe would otherwise make
        // two identical windows disagree.
        let mut buf = vec![0u8; (phones.len() + 16) * 8];
        for (k, p) in phones.iter().enumerate() {
            buf[k * 8..k * 8 + 4].copy_from_slice(&p.word.to_le_bytes());
            buf[k * 8 + 4] = p.code;
        }
        self.o.write_bytes(self.ph, &buf)?;
        self.o
            .write_bytes(self.out, &vec![0u8; phones.len() * 16 + 64])?;
        self.o
            .write_bytes(self.elems, &vec![0u8; MAX_WORDS as usize * 20])?;

        for (k, w) in words.iter().enumerate() {
            let mut s = w.as_bytes().to_vec();
            s.push(0);
            self.o.write_bytes(self.words[k], &s)?;
            self.o
                .write_u32(self.elems + k as u32 * 20 + 0x04, self.words[k])?;
            self.o.write_u32(self.elems + k as u32 * 20 + 0x08, 0)?;
            self.o.write_u8(
                self.elems + k as u32 * 20 + 0x12,
                elem12.get(k).copied().unwrap_or(0),
            )?;
        }
        self.o.write_u32(self.phrase + 0x04, words.len() as u32)?;
        self.o.write_u32(self.p4, 0)?;

        let (mut i, mut last_j) = (0u32, 0u32);
        let mut owner: Vec<u32> = Vec::new();
        while (i as usize) < phones.len() {
            let pos = phones[i as usize].word;
            let before = i;
            i = self
                .o
                .call_at(MODULE, MAPPER, &[self.ctx, pos, i, self.p4, self.ph])?;
            last_j = self.o.read_u32(self.p4)?;
            while (owner.len() as u32) < last_j {
                owner.push(before);
            }
        }
        let n = last_j as usize;
        let raw = self.o.read_bytes(self.out, n * 16)?;
        Ok(((0..n).map(|k| raw[k * 16 + 6]).collect(), owner))
    }
}

/// Which phones, words and boundaries a given output slot actually depends on.
///
/// Take a window the engine is known to produce `want` at `slot` for, then
/// change one thing and ask again. What flips the answer is the guard; what
/// does not is noise the static analysis would have carried anyway. This is
/// the whole argument for the direct call — it is about 2,500 probes, or a
/// tenth of a second.
#[derive(Default)]
pub struct Measured {
    /// `(offset from i, allowed codes)`.
    pub codes: Vec<(i32, Vec<u8>)>,
    /// `(offset from i, must be the same word as ph[i])`.
    pub bounds: Vec<(i32, bool)>,
    /// A spelling the rule reads. Nothing here says *what* it must be, so a
    /// rule with one is not fully measured and keeps its static guard.
    pub spellings: Vec<usize>,
}

fn minimise(
    p: &mut Probe,
    phones: &[Phone],
    words: &[&str],
    elem: &[u8],
    slot: usize,
    want: u8,
) -> Result<Measured, String> {
    let (base, owner) = p.run_owned(phones, words, elem)?;
    if base.get(slot).copied() != Some(want) {
        return Err(format!(
            "the window does not produce {} at slot {slot}",
            sym(want)
        ));
    }
    let broad = owner[slot];
    // Unchanged is "the same broad phone still produces the same code", which
    // survives insertions elsewhere and ignores what other phones do.
    let same = |g: &(Vec<u8>, Vec<u32>)| {
        g.1.iter()
            .zip(g.0.iter())
            .any(|(&o, &c)| o == broad && c == want)
            && g.1.iter().filter(|&&o| o == broad).count()
                == owner.iter().filter(|&&o| o == broad).count()
    };

    let mut meas = Measured::default();
    println!(
        "  baseline: slot {slot} = {} (from broad phone {broad})",
        sym(want)
    );
    for k in 0..phones.len() {
        let mut keep = Vec::new();
        for c in 0..PHONES.len() as u8 {
            let mut m = phones.to_vec();
            m[k].code = c;
            // A rule may insert or consume, which shifts every later slot.
            // Requiring the length to hold too keeps that from reading as a
            // dependency five phones away, which is what it looked like
            // before: `ph[2]` of "finished" appeared to matter to slot 7.
            // Compare the output PREFIX up to and including the slot. The
            // whole output is wrong — a later phone changing a later rule
            // shifts the length and reads as a dependency — and the slot
            // alone is wrong too, because an insertion before it moves it.
            // Nothing after `slot` can bear on whether this rule fired.
            if same(&p.run_owned(&m, words, elem)?) {
                keep.push(c);
            }
        }
        if keep.len() == PHONES.len() {
            continue; // this phone does not matter
        }
        meas.codes.push((k as i32 - broad as i32, keep.clone()));
        // A guard is usually an inclusion over a few codes or an exclusion of
        // a few; print whichever is the short list, because that is the rule.
        let rel = k as i32 - broad as i32;
        if keep.len() * 2 > PHONES.len() {
            let out: Vec<&str> = (0..PHONES.len() as u8)
                .filter(|c| !keep.contains(c))
                .map(sym)
                .collect();
            println!(
                "    ph[i{rel:+}] (now {:>4}) must NOT be  {}",
                sym(phones[k].code),
                out.join(" ")
            );
        } else {
            let names: Vec<&str> = keep.iter().map(|&c| sym(c)).collect();
            println!(
                "    ph[i{rel:+}] (now {:>4}) must be one of  {}",
                sym(phones[k].code),
                names.join(" ")
            );
        }
    }

    // Word boundaries: pull each one left and right in turn.
    for k in 1..phones.len() {
        if phones[k].word == phones[k - 1].word {
            continue;
        }
        let mut m = phones.to_vec();
        let w = m[k - 1].word;
        for e in m.iter_mut().skip(k) {
            if e.word == w + 1 {
                e.word = w;
            }
        }
        if !same(&p.run_owned(&m, words, elem)?) {
            meas.bounds.push((k as i32 - broad as i32, false));
            println!("    the word boundary before ph[{k}] matters");
        }
    }

    // Spellings: blanking one says whether any string test reads it.
    for k in 0..words.len() {
        let mut w = words.to_vec();
        w[k] = "zzqx";
        if !same(&p.run_owned(phones, &w, elem)?) {
            meas.spellings.push(k);
            println!("    the spelling of word {k} ({:?}) matters", words[k]);
        }
    }
    Ok(meas)
}

fn sym(c: u8) -> &'static str {
    PHONES.get(c as usize).map(|p| p.symbol).unwrap_or("?")
}

/// Cases read out of guest memory during a real synthesis, so the harness can
/// prove it reproduces the engine before anything is derived from it.
/// `(text, words, broad, narrow)` — every field read out of guest memory
/// during a real synthesis. The harness is proved by reproducing `narrow`
/// from a context it builds itself; the port is then a separate question.
const CASES: &[(&str, &[&str], &[(u32, u8)], &[u8])] = &[
    (
        "Testing one two three.",
        &["testing", "one", "two", "three", "."],
        &[
            (0, 0x2F),
            (0, 0x55),
            (0, 0x45),
            (0, 0x2F),
            (0, 0x11),
            (0, 0x52),
            (1, 0x23),
            (1, 0x06),
            (1, 0x50),
            (2, 0x2F),
            (2, 0x14),
            (3, 0x4A),
            (3, 0x27),
            (3, 0x0F),
            (4, 0x03),
        ],
        &[
            0x2F, 0x55, 0x45, 0x2E, 0x11, 0x52, 0x23, 0x06, 0x50, 0x2F, 0x14, 0x4A, 0x27, 0x0F,
            0x03,
        ],
    ),
    // Rule 0x2822c fires at slot 7 here, on the word-final `th` of
    // "finished". `tools/fonwindow.py 2822c` found it in the trace.
    (
        "He finished 3rd out of 21, ahead of the 4th and 5th place runners.",
        &["he", "finished", "third", "out", "of", "twenty", "one", ","],
        &[
            (0, 0x4D),
            (0, 0x0E),
            (1, 0x42),
            (1, 0x10),
            (1, 0x50),
            (1, 0x11),
            (1, 0x48),
            (1, 0x2F),
            (2, 0x4A),
            (2, 0x0C),
            (2, 0x38),
            (3, 0x1B),
            (3, 0x2F),
            (4, 0x18),
            (4, 0x43),
            (5, 0x2F),
            (5, 0x23),
            (5, 0x55),
            (5, 0x50),
            (5, 0x2F),
            (5, 0x0E),
            (6, 0x23),
            (6, 0x06),
            (6, 0x50),
            (7, 0x02),
        ],
        &[
            0x4D, 0x0E, 0x42, 0x10, 0x50, 0x11, 0x48, 0x30, 0x4A, 0x0C, 0x38, 0x1B, 0x34, 0x18,
            0x44, 0x2F, 0x23, 0x55, 0x50, 0x34, 0x0E, 0x23, 0x06, 0x50, 0x02,
        ],
    ),
];

/// `fon::exact` against the corpus trace, then against the ARM function on
/// random windows. Codes and owners must both agree.
fn xlate(probe: &mut Probe, paths: &Paths, args: &[String]) -> Result<(), String> {
    use loqng::fon::exact;

    let trace = paths
        .corpus
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("build")
        .join("fon_trace.txt");
    let phrases = crate::fon::parse(&trace)?;
    // `fontrace.py` dumps 0x400 bytes of `out[]`, which is 64 slots; a longer
    // phrase is compared on the part the trace holds.
    const TRACED: usize = 64;
    let (mut bad, mut clipped) = (0, 0);
    for p in &phrases {
        let words: Vec<&str> = p.words.iter().map(|s| s.as_str()).collect();
        let (mut got, _o) = exact::narrow(&p.broad, &words, &p.elem12);
        if p.narrow.len() == TRACED && got.len() > TRACED {
            got.truncate(TRACED);
            clipped += 1;
        }
        if got != p.narrow {
            bad += 1;
            if bad <= 5 {
                println!(
                    "  corpus {} {:?}\n    want {}\n    got  {}",
                    p.capture,
                    p.text,
                    show(&p.narrow),
                    show(&got)
                );
            }
        }
    }
    println!(
        "corpus: {}/{} phrases exact ({clipped} compared on the {TRACED} slots traced)",
        phrases.len() - bad,
        phrases.len()
    );

    let n: u64 = args
        .iter()
        .position(|a| a == "--xlate")
        .and_then(|k| args.get(k + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let mut vocab: Vec<&str> = phrases
        .iter()
        .flat_map(|p| p.words.iter().map(|s| s.as_str()))
        .filter(|w| w.len() < 60)
        .collect();
    vocab.sort();
    vocab.dedup();
    let lits = exact::literals();
    println!(
        "{} corpus words, {} literals from the module",
        vocab.len(),
        lits.len()
    );

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
        // Literals alone and as suffixes, so the whole-word and the ending
        // tests both get reached, not only what the corpus happens to spell.
        let owned: Vec<String> = (0..nw)
            .map(|_| {
                let v = vocab[rnd(vocab.len() as u64) as usize];
                let l = &lits[rnd(lits.len() as u64) as usize];
                match rnd(4) {
                    0 => l.clone(),
                    1 => format!("{v}{l}"),
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

        let engine = probe.run_owned(&ph, &words, &elem)?;
        let native = exact::narrow(&ph, &words, &elem);
        if engine != native {
            diff += 1;
            if diff <= 5 {
                println!("  case {case}: words {words:?}");
                println!(
                    "    broad  {:?}",
                    ph.iter().map(|p| (p.word, p.code)).collect::<Vec<_>>()
                );
                println!("    engine {}", show(&engine.0));
                println!("    native {}", show(&native.0));
            }
        }
    }
    println!(
        "random: {}/{n} windows identical, codes and owners ({:.1} s)",
        n - diff,
        t0.elapsed().as_secs_f64()
    );

    #[cfg(feature = "fon-cover")]
    {
        let cov = exact::coverage();
        let cold: Vec<u32> = cov.iter().filter(|c| c.1 == 0).map(|c| c.0).collect();
        println!(
            "coverage: {}/{} blocks entered",
            cov.len() - cold.len(),
            cov.len()
        );
        let list = paths
            .corpus
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("build")
            .join("fon_cold.txt");
        let text: String = cold.iter().map(|b| format!("{b:05x}\n")).collect();
        std::fs::write(&list, text).map_err(|e| format!("{}: {e}", list.display()))?;
        println!(
            "  cold blocks written to {} (tools/fonforms.py reads it)",
            list.display()
        );
        // Stage boundaries from `notes/eng-fonema.md`; stage 2 is style 2 only.
        for (name, lo, hi) in [
            ("prologue+stage 1", 0x24680, 0x2cf98),
            ("stage 2 (style 2)", 0x2cf98, 0x2eb30),
            ("stage 3 + epilogue", 0x2eb30, 0x3172c),
        ] {
            let all = cov.iter().filter(|c| c.0 >= lo && c.0 < hi).count();
            let off: Vec<String> = cold
                .iter()
                .filter(|&&b| b >= lo && b < hi)
                .map(|b| format!("{b:05x}"))
                .collect();
            println!("  {name:<20} {}/{all} entered", all - off.len());
            if args.iter().any(|a| a == "-v") {
                for c in off.chunks(12) {
                    println!("      {}", c.join(" "));
                }
            }
        }
    }
    if bad > 0 || diff > 0 {
        return Err(format!(
            "{bad} corpus phrase(s) and {diff} random window(s) differ"
        ));
    }
    println!("\nBIT-EXACT: fon::exact reproduces FonemaLargo2Stretto_US");
    Ok(())
}

fn show(v: &[u8]) -> String {
    v.iter().map(|&c| sym(c)).collect::<Vec<_>>().join(" ")
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    println!("opening the ARM engine");
    let mut probe = Probe::open(paths)?;
    println!("mapper at {MODULE}+0x{MAPPER:05x}\n");

    if args.iter().any(|a| a == "--xlate") {
        return xlate(&mut probe, paths, args);
    }

    // Derive guards for every rule the corpus exercises, straight from the
    // trace: no case has to be transcribed by hand.
    if args.iter().any(|a| a == "--derive") {
        let trace = paths
            .corpus
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("build")
            .join("fon_trace.txt");
        let phrases = crate::fon::parse(&trace)?;
        let only = args
            .iter()
            .position(|a| a == "--derive")
            .and_then(|k| args.get(k + 1))
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());

        // One window per rule: the shortest phrase that fired it, because a
        // short phrase is fewer probes and a cleaner read.
        let mut best: std::collections::BTreeMap<u32, (&crate::fon::Phrase, usize)> =
            std::collections::BTreeMap::new();
        for p in &phrases {
            for &(slot, rule) in &p.rules {
                if rule == loqng::fon::narrow::IDENTITY {
                    continue;
                }
                let e = best.entry(rule).or_insert((p, slot));
                if p.broad.len() < e.0.broad.len() {
                    *e = (p, slot);
                }
            }
        }
        println!("{} rules exercised by the corpus\n", best.len());
        let t0 = std::time::Instant::now();
        let mut done = 0;
        for (rule, (p, slot)) in &best {
            if only.is_some() && only != Some(*rule) {
                continue;
            }
            let want = match p.narrow.get(*slot) {
                Some(&w) => w,
                None => continue,
            };
            println!(
                "rule 0x{rule:05x}  -> {}   ({} phones, {:?})",
                sym(want),
                p.broad.len(),
                p.capture
            );
            let words: Vec<&str> = p.words.iter().map(|s| s.as_str()).collect();
            match minimise(&mut probe, &p.broad, &words, &p.elem12, *slot, want) {
                Ok(_m) => done += 1,
                Err(e) => println!("    skipped: {e}"),
            }
            println!();
        }
        println!(
            "{done} guards derived in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    if let Some(k) = args.iter().position(|a| a == "--minimise") {
        let idx: usize = args.get(k + 1).and_then(|s| s.parse().ok()).unwrap_or(1);
        let slot: usize = args.get(k + 2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let (text, words, phones, want) = CASES[idx];
        let ph: Vec<Phone> = phones
            .iter()
            .map(|(w, c)| Phone { word: *w, code: *c })
            .collect();
        let elem = vec![0u8; words.len()];
        println!("{text:?}\n");
        let t0 = std::time::Instant::now();
        minimise(&mut probe, &ph, words, &elem, slot, want[slot])?;
        println!("\n{:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        return Ok(());
    }

    let (mut bad, mut port_bad) = (0, 0);
    for (text, words, phones, want) in CASES {
        let ph: Vec<Phone> = phones
            .iter()
            .map(|(w, c)| Phone { word: *w, code: *c })
            .collect();
        let elem = vec![0u8; words.len()];
        let engine = probe.run(&ph, words, &elem)?;

        let mut mine = ph.clone();
        let wv: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        let ctx = Ctx {
            words: &wv,
            elem12: &elem,
            style: STYLE,
        };
        let port = narrow(&mut mine, &ctx);

        let show = |v: &[u8]| v.iter().map(|&c| sym(c)).collect::<Vec<_>>().join(" ");
        println!("  {text:?}");
        println!("    recorded  {}", show(want));
        println!(
            "    probe     {}{}",
            show(&engine),
            if engine == *want {
                "   <- the harness is right"
            } else {
                "   MISMATCH"
            }
        );
        println!(
            "    port      {}{}",
            show(&port),
            if port == engine {
                ""
            } else {
                "   (differs; that is `xtask fon`'s job)"
            }
        );
        if engine != *want {
            bad += 1;
        }
        if port != engine {
            port_bad += 1;
        }
    }

    // Timing: this is the whole point, so measure it rather than assert it.
    let (_t, w, p, _n) = CASES[0];
    let ph: Vec<Phone> = p
        .iter()
        .map(|(a, b)| Phone { word: *a, code: *b })
        .collect();
    let elem = vec![0u8; w.len()];
    let t0 = std::time::Instant::now();
    const N: u32 = 200;
    for _ in 0..N {
        probe.run(&ph, w, &elem)?;
    }
    let per = t0.elapsed().as_secs_f64() / N as f64;
    println!(
        "\n{:.2} ms per phrase probe ({} phones)",
        per * 1000.0,
        p.len()
    );
    println!("a whole synthesis under watchpoints is about 1400 ms");

    if bad > 0 {
        return Err(format!(
            "{bad} case(s): the harness does not reproduce the \
                            engine, so nothing derived from it can be trusted"
        ));
    }
    println!(
        "\nthe harness reproduces the engine on {} case(s)",
        CASES.len()
    );
    if port_bad > 0 {
        println!("{port_bad} of them the port still gets wrong — `xtask fon`");
    }
    Ok(())
}
