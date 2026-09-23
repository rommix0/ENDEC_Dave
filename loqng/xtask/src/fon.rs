//! `xtask fon` — diff the ported broad-to-narrow rewrite against the engine.
//!
//! The oracle is `build/fon_trace.txt`, written by `tools/fontrace.py`, which
//! breaks inside `FonemaLargo2Stretto_US` under `loqdave` and reads the broad
//! and narrow arrays straight out of guest memory. `corpus/*/stage-fon.txt`
//! cannot serve here: it is written further downstream and differs from this
//! stage's output (see that tool's header).
//!
//! ```text
//! python tools/fontrace.py            # ~15 s for the 37-capture corpus
//! cargo run -p xtask -- fon
//! ```

use std::path::Path;

use loqng::fon::narrow::{narrow, Ctx, Phone};
use loqng::fon::phontab::PHONES;

use crate::Paths;

pub struct Phrase {
    pub capture: String,
    pub text: String,
    pub words: Vec<String>,
    pub elem12: Vec<u8>,
    pub broad: Vec<Phone>,
    pub narrow: Vec<u8>,
    /// Narrow slot -> the rule the engine actually used, from a `loqdave`
    /// watchpoint on that slot. This is the sharp signal: it says which rule
    /// fired, not merely what came out.
    pub rules: Vec<(usize, u32)>,
}

pub fn parse(path: &Path) -> Result<Vec<Phrase>, String> {
    let body = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "{}: {e}\nrun `python tools/fontrace.py` first",
            path.display()
        )
    })?;
    let mut out: Vec<Phrase> = Vec::new();
    let (mut capture, mut text) = (String::new(), String::new());
    for line in body.lines() {
        let (tag, rest) = line.split_at(line.find(' ').unwrap_or(line.len()));
        let rest = rest.strip_prefix(' ').unwrap_or("");
        match tag {
            "C" => capture = rest.to_string(),
            "T" => text = rest.to_string(),
            "P" => out.push(Phrase {
                capture: capture.clone(),
                text: text.clone(),
                words: Vec::new(),
                elem12: Vec::new(),
                broad: Vec::new(),
                narrow: Vec::new(),
                rules: Vec::new(),
            }),
            "W" | "E" | "B" | "N" | "R" => {
                let p = out.last_mut().ok_or("a record before any `P`")?;
                match tag {
                    "W" => {
                        let (_i, s) = rest.split_at(rest.find(' ').unwrap_or(rest.len()));
                        p.words.push(s.strip_prefix(' ').unwrap_or("").to_string());
                    }
                    "E" => {
                        let v: i32 = rest
                            .rsplit(' ')
                            .next()
                            .unwrap_or("-1")
                            .parse()
                            .unwrap_or(-1);
                        // -1 means the engine never reached the read, so no
                        // rule that tests it can have fired; 0xff never equals
                        // the one value any rule compares against.
                        p.elem12.push(if v < 0 { 0xff } else { v as u8 });
                    }
                    "B" => {
                        for t in rest.split_whitespace() {
                            let (w, c) = t.split_once(',').ok_or("bad B field")?;
                            p.broad.push(Phone {
                                word: w.parse().map_err(|_| "bad word index")?,
                                code: c.parse().map_err(|_| "bad code")?,
                            });
                        }
                    }
                    "N" => {
                        for t in rest.split_whitespace() {
                            p.narrow
                                .push(u8::from_str_radix(t, 16).map_err(|_| "bad narrow byte")?);
                        }
                    }
                    _ => {
                        let f: Vec<&str> = rest.split_whitespace().collect();
                        if f.len() == 3 {
                            p.rules.push((
                                f[0].parse().map_err(|_| "bad slot")?,
                                u32::from_str_radix(f[2], 16).map_err(|_| "bad rule address")?,
                            ));
                        }
                    }
                }
            }
            "" => {}
            other => return Err(format!("unknown record `{other}`")),
        }
    }
    Ok(out)
}

fn sym(c: u8) -> &'static str {
    PHONES.get(c as usize).map(|p| p.symbol).unwrap_or("?")
}

fn show(v: &[u8]) -> String {
    v.iter().map(|&c| sym(c)).collect::<Vec<_>>().join(" ")
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let trace = args
        .iter()
        .position(|a| a == "--trace")
        .and_then(|i| args.get(i + 1))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            paths
                .corpus
                .parent()
                .unwrap_or(Path::new("."))
                .join("build")
                .join("fon_trace.txt")
        });
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let why = args.iter().any(|a| a == "--why");

    let phrases = parse(&trace)?;
    println!("{} phrases from {}", phrases.len(), trace.display());

    let (mut ok, mut bad, mut phones_ok, mut phones_all) = (0usize, 0usize, 0usize, 0usize);
    let mut shown = 0;
    for p in &phrases {
        let mut ph = p.broad.clone();
        let ctx = Ctx {
            words: &p.words,
            elem12: &p.elem12,
            style: 3,
        };
        let got = narrow(&mut ph, &ctx);

        phones_all += p.narrow.len();
        phones_ok += got
            .iter()
            .zip(p.narrow.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if got == p.narrow {
            ok += 1;
            continue;
        }
        bad += 1;
        if shown < 12 || verbose {
            shown += 1;
            let at = got.iter().zip(p.narrow.iter()).position(|(a, b)| a != b);
            println!("\n  {} — {:?}", p.capture, p.text);
            println!("    words  {}", p.words.join(" "));
            println!("    want   {}", show(&p.narrow));
            println!("    got    {}", show(&got));
            match at {
                Some(k) => {
                    println!(
                        "    first difference at {k}: want {} ({}), got {} ({})",
                        sym(p.narrow[k]),
                        p.narrow[k],
                        sym(got[k]),
                        got[k]
                    );
                    if why {
                        // `j` and `i` track together only while no rule has
                        // inserted or consumed, which is true up to the first
                        // difference by construction.
                        let mut fresh = p.broad.clone();
                        let mut scratch = Vec::new();
                        let (mut i, mut j) = (0usize, 0usize);
                        while j < k && i < fresh.len() {
                            let (ni, nj) =
                                loqng::fon::narrow::step(&mut fresh, i, j, &mut scratch, &ctx);
                            i = ni;
                            j = nj;
                        }
                        println!(
                            "    at broad index {i}, fired: {:?}",
                            loqng::fon::narrow::which(&fresh, i, &ctx)
                                .map(|a| format!("0x{a:05x}"))
                        );
                        for (at, blk, stage) in
                            loqng::fon::narrow::blockers(&fresh, i, &ctx, p.narrow[k])
                        {
                            match blk {
                                None => println!("      0x{at:05x} (stage {stage}) would fire"),
                                Some(c) => println!(
                                    "      0x{at:05x} (stage {stage}) arm {} blocked at cond {}: {}", c.0, c.1,
                                    loqng::fon::narrow::describe(at, c.0, c.1)
                                ),
                            }
                        }
                    }
                }
                None => println!(
                    "    lengths differ: want {}, got {}",
                    p.narrow.len(),
                    got.len()
                ),
            }
        }
    }

    // Which rule fired, engine against port. This is where the remaining
    // defect shows itself as a rule rather than as a phrase.
    let mut agree = 0usize;
    let mut total = 0usize;
    let mut wrong: std::collections::BTreeMap<(u32, u32), usize> =
        std::collections::BTreeMap::new();
    for p in &phrases {
        if p.rules.is_empty() {
            continue;
        }
        let want: std::collections::BTreeMap<usize, u32> = p.rules.iter().copied().collect();
        let mut ph = p.broad.clone();
        let mut scratch = Vec::new();
        let ctx = Ctx {
            words: &p.words,
            elem12: &p.elem12,
            style: 3,
        };
        let (mut i, mut j) = (0usize, 0usize);
        while i < ph.len() {
            let mine = loqng::fon::narrow::store_of(&ph, i, &ctx);
            if let Some(&engine) = want.get(&j) {
                total += 1;
                if engine == mine {
                    agree += 1;
                } else {
                    *wrong.entry((engine, mine)).or_default() += 1;
                }
            }
            let (ni, nj) = loqng::fon::narrow::step(&mut ph, i, j, &mut scratch, &ctx);
            i = ni;
            j = nj;
        }
    }

    println!();
    if total > 0 {
        println!(
            "rule agreement: {agree}/{total} phones ({:.2}%)",
            100.0 * agree as f64 / total as f64
        );
        let mut rows: Vec<_> = wrong.into_iter().collect();
        rows.sort_by_key(|(_k, n)| std::cmp::Reverse(*n));
        println!("  {:>5}  {:>9}  {}", "count", "engine", "port");
        for ((engine, mine), n) in rows.iter().take(15) {
            let name = |a: u32| {
                if a == loqng::fon::narrow::IDENTITY {
                    "identity".to_string()
                } else {
                    format!("0x{a:05x}")
                }
            };
            println!("  {n:>5}  {:>9}  {}", name(*engine), name(*mine));
        }
    }

    println!();
    println!("{ok}/{} phrases exact", phrases.len());
    println!(
        "{phones_ok} of {phones_all} narrow phones match up to the first \
              difference ({:.2}%)",
        100.0 * phones_ok as f64 / phones_all.max(1) as f64
    );
    if bad == 0 {
        println!("\nBIT-EXACT: the ported rule table reproduces the engine");
        Ok(())
    } else {
        Err(format!("{bad} phrase(s) differ"))
    }
}
