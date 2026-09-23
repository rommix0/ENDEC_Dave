//! `xtask probe` — find out which trace switches actually produce output.
//!
//! The engine has six instance parameters that ask a stage to serialise its
//! stream. Where each one routes is not documented and is not worth guessing,
//! so each is turned on alone, in every plausible spelling of "on", and both
//! collection points are measured: the `TraceFile` sink and the guest's stdout,
//! which is where an `XxxWrite` dump goes (`FonWrite` compares its `FILE*`
//! against `ELQGetStdout()`).
//!
//! Run this before `capture`. A switch reported silent here will be silent in
//! the corpus too, and that changes which stages Track B can be gated on.

use loqng_oracle::{Oracle, OracleConfig, PROBE_VALUES};

use crate::Paths;

const DEFAULT_TEXT: &str = "The quick brown fox jumped over 24 lazy dogs on March 3rd.";

pub fn run(p: &Paths, text: Option<String>) -> Result<(), String> {
    let text = text.unwrap_or_else(|| DEFAULT_TEXT.to_string());
    let cfg = OracleConfig::new(&p.lib_dir, &p.data_dir);

    println!("probing trace switches with:");
    println!("  {text:?}");
    println!("values tried per switch: {}", PROBE_VALUES.join(" "));
    println!();
    println!(
        "  {:<24} {:<6} {:>9} {:>9}  {}",
        "switch", "stage", "+trace", "+stdout", "verdict"
    );
    println!("  {}", "-".repeat(72));

    let results = Oracle::probe(&cfg, &text)?;
    let mut working = 0;

    for r in &results {
        let stage = r.stage.map(|s| s.to_string()).unwrap_or_else(|| "-".into());
        let verdict = match &r.error {
            Some(e) => format!("error: {e}"),
            None if r.produced_output() => {
                working += 1;
                "produces output".to_string()
            }
            None => "silent in every spelling".to_string(),
        };
        println!(
            "  {:<24} {:<6} {:>9} {:>9}  {verdict}",
            r.key,
            stage,
            r.extra_trace(),
            r.extra_stdout(),
            verdict = verdict
        );
    }

    let baseline_trace = results.first().map(|r| r.baseline_trace).unwrap_or(0);
    let baseline_stdout = results.first().map(|r| r.baseline_stdout).unwrap_or(0);

    println!();
    println!(
        "baseline with no switches: {baseline_trace} bytes of trace, \
         {baseline_stdout} bytes of stdout"
    );
    println!("{working}/{} switches added output", results.len());

    if baseline_trace > 0 {
        println!();
        println!("The baseline trace is not empty, and it is already useful:");
        println!("  TTSEVT_WORDTRANSCRIPTION carries per-word grapheme-to-phoneme");
        println!("  output, which is the gate Fon needs. INPUT carries what Top and");
        println!("  Les received. Both are available without any switch at all.");
    }
    if working == 0 {
        println!();
        println!("No switch changed anything. Remaining things to try, in order:");
        println!("  1. LogLevel — the trace is the engine's event log, so verbosity");
        println!("     may gate these rather than the switches themselves.");
        println!("  2. Session-file placement rather than ttsSetInstanceParam.");
        println!("  3. ttsEnableEvent (LoqTTS6.so+0x347d8) — the dumps ride the event");
        println!("     system, and events may need enabling per instance.");
    }
    Ok(())
}
