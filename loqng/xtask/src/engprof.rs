//! `xtask engprof` — where the time actually goes.
//!
//! Every gain so far came from reasoning about the shape of the generated
//! code. This measures instead. Build with the profiling counter in:
//!
//! ```text
//! cargo build -p xtask --release --features xrt-prof
//! target\release\xtask.exe engprof [--chars 4000] [--top 30]
//! ```
//!
//! The counter is a direct-mapped array over the guest text, incremented
//! without a lock (see `loqng_xrt::prof`), so what comes out is close to what
//! an uninstrumented run does. Each block's entry count is multiplied by its
//! length in instructions — the gap to the next block start — and the blocks
//! are grouped back into the functions they came from. The result is a
//! ranking of **guest instructions executed**, which for a translation this
//! literal is the thing that costs.

use std::collections::HashMap;

use crate::engbench::{build, Engine};
use crate::{flag, Paths};

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    if !cfg!(feature = "xrt-prof") {
        return Err("built without the counter — rebuild with \
                    `--features xrt-prof`"
            .into());
    }

    let chars: usize = flag(args, "--chars")
        .map(|s| s.parse().unwrap_or(4000))
        .unwrap_or(4000);
    let top: usize = flag(args, "--top")
        .map(|s| s.parse().unwrap_or(30))
        .unwrap_or(30);

    let (mut eng, load) = Engine::open(paths)?;
    println!("voice load {load:.2} s");

    // A warm-up utterance first: the counters are cleared afterwards, so
    // one-off initialisation does not show up as engine work.
    eng.say("Testing one two three.")?;
    take();

    let text = build(chars);
    let (dt, bytes) = eng.say(&text)?;
    let audio = bytes as f64 / 2.0 / 16000.0;
    println!(
        "{chars} characters -> {audio:.2} s of audio in {dt:.2} s \
              ({:.1}x real time, with the counter in)",
        audio / dt
    );

    let counts = take();
    report(&counts, top);
    Ok(())
}

#[cfg(feature = "xrt-prof")]
fn take() -> Vec<(u32, u64)> {
    loqng::xrt::prof::take()
}

#[cfg(not(feature = "xrt-prof"))]
fn take() -> Vec<(u32, u64)> {
    Vec::new()
}

/// A block's length is the gap to the next block start. The last block of a
/// function runs up to the next function, which overstates it by whatever
/// padding the linker left; capping keeps one stray gap from dominating.
const MAX_BLOCK: u32 = 4096;

fn report(counts: &[(u32, u64)], top: usize) {
    let mut blocks: Vec<(u32, usize)> = loqng::eng::blocks().collect();
    blocks.sort_by_key(|b| b.0);

    let mut len: HashMap<u32, u32> = HashMap::new();
    let mut owner: HashMap<u32, usize> = HashMap::new();
    let mut entry: HashMap<usize, u32> = HashMap::new();
    for (k, &(at, who)) in blocks.iter().enumerate() {
        let next = blocks.get(k + 1).map(|b| b.0).unwrap_or(at + 4);
        len.insert(at, (next - at).min(MAX_BLOCK) / 4);
        owner.insert(at, who);
        entry
            .entry(who)
            .and_modify(|e| *e = (*e).min(at))
            .or_insert(at);
    }

    let mut by_fn: HashMap<usize, (u64, u64)> = HashMap::new();
    let (mut total, mut unknown) = (0u64, 0u64);
    for &(at, n) in counts {
        let Some(&who) = owner.get(&at) else {
            unknown += n;
            continue;
        };
        let ins = n * len[&at] as u64;
        total += ins;
        let e = by_fn.entry(who).or_default();
        e.0 += ins;
        e.1 += n;
    }

    let mut rank: Vec<(usize, u64, u64)> = by_fn.into_iter().map(|(f, (i, n))| (f, i, n)).collect();
    rank.sort_by(|a, b| b.1.cmp(&a.1));

    println!(
        "\n{total} guest instructions in {} blocks, {} functions",
        counts.len(),
        rank.len()
    );
    if unknown != 0 {
        println!("{unknown} entries at addresses with no block (should be 0)");
    }

    println!("\n  share  cum   instructions    calls  function");
    let mut cum = 0u64;
    for &(f, ins, n) in rank.iter().take(top) {
        cum += ins;
        let at = entry[&f];
        println!(
            "  {:5.1}% {:5.1}% {:>14} {:>8}  {:08x}  {}",
            100.0 * ins as f64 / total as f64,
            100.0 * cum as f64 / total as f64,
            ins,
            n,
            at,
            place(at)
        );
    }

    // A tail this long is the real finding if the head is flat: there is no
    // single hot loop to attack, only the cost of every instruction.
    let head: u64 = rank.iter().take(20).map(|r| r.1).sum();
    println!(
        "\ntop 20 functions are {:.1}% of all instructions",
        100.0 * head as f64 / total as f64
    );
}

/// `module+offset`, plus the nearest exported symbol at or before it.
fn place(a: u32) -> String {
    let m = loqng::eng::MODULES
        .iter()
        .find(|(_, b, e)| a >= *b && a < *e)
        .map(|(n, b, _)| format!("{n}+0x{:x}", a - b))
        .unwrap_or_else(|| String::from("?"));
    let near = loqng::eng::EXPORTS
        .iter()
        .filter(|(_, at)| *at <= a)
        .max_by_key(|(_, at)| *at);
    match near {
        Some((n, at)) if a - at < 0x4000 => format!("{m}  ({n}+0x{:x})", a - at),
        _ => m,
    }
}
