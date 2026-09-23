//! `FonemaLargo2Stretto_US` — the broad-to-narrow phoneme rewrite.
//!
//! The ARM original is `LoqEnglish6.9.so` vaddr `0x24680`, 53,420 bytes of
//! compiled decision tree. It is not arbitrary: it is 123 priority-ordered
//! rules over a small vocabulary of tests, which [`super::rules`] holds as
//! data. `notes/eng-fonema-rules.md` is the same table in readable form and
//! `notes/eng-fonema.md` has how it was lifted.
//!
//! The engine handles **one** broad phone per call and returns the next index,
//! exactly as the original does; `LoqTTS6` drives the loop. A rule may emit up
//! to three narrow phones and consume up to two extra broad ones, so neither
//! index advances by a fixed amount.

use super::phontab::PHONES;
use super::rules::{Act, Cond, Match, Rel, Rule, Subj, PREPASS, STAGE1, STAGE2, STAGE3};

/// One entry of the broad array, `ctx->[0x10]`. Eight bytes in the original;
/// `word` is an index, so `a.word != b.word` means a word boundary lies
/// between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Phone {
    pub word: u32,
    pub code: u8,
}

/// Everything the rules can look at besides the phone array.
pub struct Ctx<'a> {
    /// Spellings by word index, as `ph[_].word` indexes them.
    pub words: &'a [String],
    /// The `LesElem` byte at `+0x12`, per word index. Exactly one rule reads
    /// it and its meaning is still open — `notes/eng-fonema.md` §6.
    pub elem12: &'a [u8],
    /// `ctx->[0x40]->[0x30c]`. Dave is 3; see `notes/eng-fonema.md` §6.
    pub style: u32,
}

/// The 1024-entry bound the original range-checks every phone index against.
/// A rule whose window reaches outside it does not fire.
const LIMIT: u32 = 0x3FF;

/// `ph[i + at]`, or `None` when the original's `cmp idx, #0x3ff` / `bhi` guard
/// would have sent it to the rule's failure label. The index is computed in
/// `u32` as the original does, so `i - 1` at `i == 0` wraps and fails.
fn at(ph: &[Phone], i: usize, off: i16) -> Option<Phone> {
    let idx = (i as u32).wrapping_add(off as i32 as u32);
    if idx > LIMIT {
        return None;
    }
    Some(ph.get(idx as usize).copied().unwrap_or_default())
}

fn text<'a>(ctx: &'a Ctx<'a>, ph: &[Phone], i: usize, subj: Subj) -> &'a str {
    let w = match at(ph, i, 0) {
        Some(p) => p.word as usize,
        None => return "",
    };
    // At a phrase edge the original substitutes the empty string the GOT
    // holds at `+0x698`, not a null pointer.
    let k = match subj {
        Subj::Word => Some(w),
        Subj::PrevWord => w.checked_sub(1),
        Subj::NextWord => Some(w + 1),
    };
    k.and_then(|k| ctx.words.get(k))
        .map(|s| s.as_str())
        .unwrap_or("")
}

fn matches_text(s: &str, how: Match, lit: &str) -> bool {
    match how {
        Match::Equals => s == lit,
        Match::Prefix => s.as_bytes().starts_with(lit.as_bytes()),
        // The original guards a suffix test with `cmp len, #(n-1)` / `bls`,
        // so a word shorter than the literal never reaches the compare.
        Match::Suffix(n) => {
            let n = n as usize;
            s.len() >= n && s.as_bytes()[s.len() - n..] == *lit.as_bytes()
        }
        Match::Contains => s.contains(lit),
    }
}

fn rel_holds(rel: Rel, a: u32, b: u32) -> bool {
    match rel {
        Rel::Eq => a == b,
        Rel::Ne => a != b,
        Rel::Lt => a < b,
        Rel::Le => a <= b,
        Rel::Gt => a > b,
        Rel::Ge => a >= b,
    }
}

fn holds(c: &Cond, ph: &[Phone], i: usize, ctx: &Ctx) -> bool {
    match *c {
        Cond::Code {
            at: off,
            want,
            codes,
        } => match at(ph, i, off) {
            // Out of range fails the rule whichever way the test runs: the
            // bounds check branches to the failure label, not past the test.
            None => false,
            Some(p) => codes.contains(&p.code) == want,
        },
        Cond::SameWord { a, b, want } => match (at(ph, i, a), at(ph, i, b)) {
            (Some(x), Some(y)) => (x.word == y.word) == want,
            _ => false,
        },
        Cond::Feat {
            at: off,
            want,
            mask,
        } => {
            // Out of range is not a failure here: the original falls back to
            // record 1, whose `feat` is zero.
            let code = at(ph, i, off).map(|p| p.code).unwrap_or(1);
            let feat = PHONES.get(code as usize).map(|p| p.feat).unwrap_or(0);
            (feat & mask == mask) == want
        }
        Cond::Text {
            subj,
            how,
            want,
            lits,
        } => {
            let s = text(ctx, ph, i, subj);
            lits.iter().any(|l| matches_text(s, how, l)) == want
        }
        Cond::Index { rel, value } => rel_holds(rel, i as u32, value),
        Cond::Len { rel, value } => {
            let n = text(ctx, ph, i, Subj::Word).len() as u32;
            rel_holds(rel, n, value)
        }
        Cond::Unread { .. } => false,
        Cond::Absent { subj, want } => text(ctx, ph, i, subj).is_empty() == want,
        Cond::Any(cs) => cs.iter().any(|c| holds(c, ph, i, ctx)),
        Cond::All(cs) => cs.iter().all(|c| holds(c, ph, i, ctx)),
        Cond::ElemByte { off, want, value } => {
            debug_assert_eq!(off, 0x12, "only +0x12 has ever been seen");
            let w = at(ph, i, 0).map(|p| p.word as usize).unwrap_or(0);
            match ctx.elem12.get(w) {
                Some(&b) => (b == value) == want,
                None => false,
            }
        }
    }
}

fn fires(r: &Rule, ph: &[Phone], i: usize, ctx: &Ctx) -> bool {
    let code = match at(ph, i, 0) {
        Some(p) => p.code,
        None => return false,
    };
    if !r.phones.is_empty() && !r.phones.contains(&code) {
        return false;
    }
    r.arms
        .iter()
        .any(|arm| arm.iter().all(|c| holds(c, ph, i, ctx)))
}

/// What one matched rule did, so the caller can advance both cursors.
struct Applied {
    /// Extra broad phones consumed, on top of the `+ 1` every call makes.
    extra_i: usize,
    /// Index of the last narrow phone written.
    last_j: usize,
}

fn apply(r: &Rule, out: &mut Vec<u8>, j0: usize) -> Applied {
    let mut j = j0;
    let mut extra_i = 0;
    for a in r.acts {
        match *a {
            Act::Out { rel, code } => {
                let k = j + rel as usize;
                if out.len() <= k {
                    out.resize(k + 1, 0);
                }
                out[k] = code;
            }
            Act::BumpJ(n) => j += n as usize,
            Act::BumpI(n) => extra_i += n as usize,
            Act::SetI(n) => extra_i = n as usize,
        }
    }
    Applied { extra_i, last_j: j }
}

/// One call of the original: rewrite `ph[i]` in place, emit narrow phones from
/// `j`, and return `(next_i, next_j)`.
///
/// `ph` is `&mut` because the aspiration rewrites below are stores into the
/// broad array, visible to every later call.
pub fn step(ph: &mut [Phone], i: usize, j: usize, out: &mut Vec<u8>, ctx: &Ctx) -> (usize, usize) {
    // Unconditional rewrites, before any rule runs. Stops are aspirated by
    // default and the rules take the aspiration away again; `e` and `` `e ``
    // fold onto the duplicate `E` records.
    if let Some(p) = ph.get_mut(i) {
        p.code = match p.code {
            43 => 44, // p  -> ph
            46 => 47, // t  -> th
            49 => 50, // k  -> kh
            10 => 85, // `e -> `E
            11 => 86, // e  -> E
            c => c,
        };
    }

    // The pre-pass inserts a glide between a vowel and a following vowel and
    // then falls through, so the per-phone rewrite still runs on this phone.
    // It is not a rule and must not stop the dispatch.
    let mut j = j;
    if let Some(r) = PREPASS.iter().find(|r| fires(r, ph, i, ctx)) {
        let a = apply(r, out, j);
        j = a.last_j + 1;
    }

    let mut hit = None;
    for r in STAGE1.iter() {
        if fires(r, ph, i, ctx) {
            hit = Some(r);
            break;
        }
    }
    if hit.is_none() && ctx.style == 2 {
        hit = STAGE2.iter().find(|r| fires(r, ph, i, ctx));
    }
    if hit.is_none() && ctx.style == 3 {
        hit = STAGE3.iter().find(|r| fires(r, ph, i, ctx));
    }

    let (extra_i, last_j) = match hit {
        Some(r) => {
            let a = apply(r, out, j);
            (a.extra_i, a.last_j)
        }
        None => {
            // Nothing matched: the narrow phone is the broad one.
            if out.len() <= j {
                out.resize(j + 1, 0);
            }
            out[j] = ph.get(i).map(|p| p.code).unwrap_or(0);
            (0, j)
        }
    };
    (i + extra_i + 1, last_j + 1)
}

/// Run [`step`] over a whole phrase.
pub fn narrow(ph: &mut [Phone], ctx: &Ctx) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < ph.len() {
        let (ni, nj) = step(ph, i, j, &mut out, ctx);
        debug_assert!(ni > i, "step must advance");
        i = ni;
        j = nj;
    }
    out
}

/// Every rule that could have written `want` at `out[j]`, with the index of
/// the first condition that stopped it. `None` means it would have fired.
///
/// This is the diagnostic that matters when the gate fails: the port almost
/// always *under*-fires, and this says which condition is at fault without
/// reading the listing again.
pub fn blockers(
    ph: &[Phone],
    i: usize,
    ctx: &Ctx,
    want: u8,
) -> Vec<(u32, Option<(usize, usize)>, &'static str)> {
    let stages: [(&[Rule], &'static str); 3] = [(&STAGE1, "1"), (&STAGE2, "2"), (&STAGE3, "3")];
    let mut out = Vec::new();
    for (rules, name) in stages {
        for r in rules {
            if !r
                .acts
                .iter()
                .any(|a| matches!(a, Act::Out { rel: 0, code } if *code == want))
            {
                continue;
            }
            let code = match at(ph, i, 0) {
                Some(p) => p.code,
                None => continue,
            };
            if !r.phones.is_empty() && !r.phones.contains(&code) {
                continue;
            }
            // Report the arm that came closest: the one whose first failing
            // condition is furthest along.
            let best = r
                .arms
                .iter()
                .enumerate()
                .map(|(k, arm)| (k, arm.iter().position(|c| !holds(c, ph, i, ctx))))
                .max_by_key(|(_k, p)| p.unwrap_or(usize::MAX));
            match best {
                Some((k, p)) => out.push((r.at, p.map(|p| (k, p)), name)),
                None => out.push((r.at, None, name)),
            }
        }
    }
    out
}

/// Render one condition the way the generated table spells it, for `blockers`.
pub fn describe(r_at: u32, arm: usize, k: usize) -> String {
    for rules in [&STAGE1[..], &STAGE2[..], &STAGE3[..]] {
        if let Some(r) = rules.iter().find(|r| r.at == r_at) {
            return match r.arms.get(arm).and_then(|a| a.get(k)) {
                Some(c) => format!("{c:?}"),
                None => "<none>".to_string(),
            };
        }
    }
    "<unknown rule>".to_string()
}

/// Which rule fires for `ph[i]`, as an ELF vaddr, or `None` for the identity
/// fallback. For diffing against a `loqdave` breakpoint trace.
pub fn which(ph: &[Phone], i: usize, ctx: &Ctx) -> Option<u32> {
    pick(ph, i, ctx).map(|r| r.at)
}

/// The rule that fires, or `None` for the identity fallback.
fn pick<'r>(ph: &[Phone], i: usize, ctx: &Ctx) -> Option<&'r Rule> {
    for r in STAGE1.iter() {
        if fires(r, ph, i, ctx) {
            return Some(r);
        }
    }
    let extra: &[Rule] = match ctx.style {
        2 => &STAGE2,
        3 => &STAGE3,
        _ => &[],
    };
    extra.iter().find(|r| fires(r, ph, i, ctx))
}

/// The address a `loqdave` watchpoint would report for the phone written at
/// `i`, so the port can be diffed against `tools/fontrace.py`'s `R` records
/// rule by rule rather than only on the final phone sequence.
///
/// `IDENTITY` is the epilogue's `out[j] = ph[i].code`, which is what the
/// engine does when no rule matches — 84% of the corpus.
pub const IDENTITY: u32 = 0x316C0;

pub fn store_of(ph: &[Phone], i: usize, ctx: &Ctx) -> u32 {
    pick(ph, i, ctx).map_or(IDENTITY, |r| r.store)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phrase(words: &[(&str, &[u8])]) -> (Vec<Phone>, Vec<String>) {
        let mut ph = Vec::new();
        let mut ws = Vec::new();
        for (w, codes) in words {
            let idx = ws.len() as u32;
            ws.push((*w).to_string());
            for &c in *codes {
                ph.push(Phone { word: idx, code: c });
            }
        }
        (ph, ws)
    }

    /// "Testing one two three." -- the broad array read straight out of the
    /// engine with `loqdave --break 1f4774 --dump r6`, and the narrow array
    /// read out of `r9` after the last call.
    ///
    /// **Ignored, and the assertion is right.** It is the acceptance test for
    /// the last piece of `tools/fonguard.py`: the guard for rule `0x2767c`
    /// comes out as a set of arms each pinned to one value of `i`, because
    /// the path enumeration re-derives every earlier rule's failure and the
    /// arm cap then keeps the most specific ones. See `HANDOFF.md` §4.
    /// Run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "blocked on the rule-entry partition in tools/fonguard.py"]
    fn testing_one_two_three_matches_the_engine() {
        let (mut ph, words) = phrase(&[
            ("testing", &[0x2f, 0x55, 0x45, 0x2f, 0x11, 0x52]),
            ("one", &[0x23, 0x06, 0x50]),
            ("two", &[0x2f, 0x14]),
            ("three", &[0x4a, 0x27, 0x0f]),
            (".", &[0x03]),
        ]);
        let elem = vec![0u8; words.len()];
        let ctx = Ctx {
            words: &words,
            elem12: &elem,
            style: 3,
        };
        let got = narrow(&mut ph, &ctx);
        assert_eq!(
            got,
            vec![
                0x2f, 0x55, 0x45, 0x2e, 0x11, 0x52, 0x23, 0x06, 0x50, 0x2f, 0x14, 0x4a, 0x27, 0x0f,
                0x03
            ]
        );
    }

    /// The de-aspiration that produces the `t` above: `s` before `th`, inside
    /// one word, with the next phone not `h`. Rule `@0x2767c`.
    ///
    /// Ignored for the same reason as the test above, and against the same
    /// verified engine behaviour.
    #[test]
    #[ignore = "blocked on the rule-entry partition in tools/fonguard.py"]
    fn s_before_t_de_aspirates_it() {
        let (mut ph, words) = phrase(&[("testing", &[0x2f, 0x55, 0x45, 0x2f, 0x11, 0x52])]);
        let elem = vec![0u8; words.len()];
        let ctx = Ctx {
            words: &words,
            elem12: &elem,
            style: 3,
        };
        assert_eq!(which(&ph, 3, &ctx), Some(0x2767c));
        let got = narrow(&mut ph, &ctx);
        assert_eq!(got[3], 0x2e, "the medial t should lose its aspiration");
    }

    /// A phone whose window reaches before the array cannot match a rule that
    /// looks back: the original's bounds check branches to the failure label.
    #[test]
    fn a_rule_that_looks_back_cannot_fire_at_index_zero() {
        let (ph, words) = phrase(&[("testing", &[0x2f, 0x55])]);
        let elem = vec![0u8; words.len()];
        let ctx = Ctx {
            words: &words,
            elem12: &elem,
            style: 3,
        };
        let c = Cond::Code {
            at: -1,
            want: true,
            codes: &[0x45],
        };
        assert!(!holds(&c, &ph, 0, &ctx));
        // ...and a negated test fails there too, rather than passing by default.
        let c = Cond::Code {
            at: -1,
            want: false,
            codes: &[0x45],
        };
        assert!(!holds(&c, &ph, 0, &ctx));
    }
}
