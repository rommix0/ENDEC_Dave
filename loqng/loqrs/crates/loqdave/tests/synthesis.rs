//! End-to-end checks. These need a staged engine tree; without one they skip,
//! so `cargo test` still passes on a machine that has no Loquendo install.
//!
//! Stage it with `docker/stage-engine.sh <endec-root>`, which writes
//! `engine/lib` and `engine/data` at the workspace root.

use std::path::PathBuf;

use loqhost::tts::{Request, SAMPLE_RATE};
use loqhost::{Engine, EngineConfig};

fn engine_root() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .parent()?
        .join("engine");
    if root.join("lib/LoqTTS6.so").is_file() && root.join("lib/tts").is_file() {
        Some(root)
    } else {
        None
    }
}

fn build() -> Option<Engine> {
    let root = engine_root()?;
    let mut cfg = EngineConfig::new(root.join("lib"), root.join("data"));
    cfg.license = std::fs::read_to_string(root.join("LicenseCode.txt")).ok();
    Some(Engine::new(&cfg).expect("engine init"))
}

/// 16-bit mono PCM statistics: (samples, peak, rms).
fn stats(pcm: &[u8]) -> (usize, i32, f64) {
    let s: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let peak = s.iter().map(|v| i32::from(v.abs())).max().unwrap_or(0);
    let energy: f64 = s.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    (s.len(), peak, (energy / s.len().max(1) as f64).sqrt())
}

#[test]
fn speaks_a_sentence() {
    let Some(mut engine) = build() else {
        eprintln!("skipping: no engine/ tree staged");
        return;
    };
    let pcm = engine
        .synthesize(&Request::new("This is a test of the Loquendo Dave voice."))
        .expect("synthesis");

    let (samples, peak, rms) = stats(&pcm);
    let seconds = samples as f64 / f64::from(SAMPLE_RATE);

    assert!(
        (2.0..4.0).contains(&seconds),
        "expected roughly 2.7 s of audio, got {seconds:.3} s"
    );
    assert!(peak > 10_000, "audio is too quiet to be speech: peak {peak}");
    assert!(
        (1_000.0..12_000.0).contains(&rms),
        "unexpected loudness: rms {rms:.0}"
    );

    // The ENDEC's writer emits whole 4096-byte blocks; ours keeps the tail.
    assert!(
        pcm.len() >= pcm.len() / 4096 * 4096,
        "output shorter than a whole block"
    );
}

#[test]
fn longer_text_takes_proportionally_longer() {
    let Some(mut engine) = build() else {
        eprintln!("skipping: no engine/ tree staged");
        return;
    };
    let pcm = engine
        .synthesize(&Request::new(
            "The National Weather Service has issued a Tornado Warning \
             for Jefferson County, Alabama, beginning at four fifteen P M \
             and ending at five thirty P M.",
        ))
        .expect("synthesis");

    let (samples, peak, _) = stats(&pcm);
    let seconds = samples as f64 / f64::from(SAMPLE_RATE);
    assert!(
        seconds > 6.0,
        "a long alert should run well past six seconds, got {seconds:.3} s"
    );
    assert!(peak > 10_000, "audio is too quiet to be speech: peak {peak}");
}

#[test]
fn one_session_speaks_many_utterances() {
    let Some(mut engine) = build() else {
        eprintln!("skipping: no engine/ tree staged");
        return;
    };
    engine.open("Dave", "loqmsx", SAMPLE_RATE).expect("open");

    let lines = [
        "Testing one two three.",
        "The National Weather Service has issued a Tornado Warning.",
        "This is an E.A.S. participant test.",
    ];
    let mut durations = Vec::new();
    for line in lines {
        let pcm = engine.speak(line).expect("speak");
        let (samples, peak, _) = stats(&pcm);
        assert!(peak > 10_000, "`{line}` came out too quiet: peak {peak}");
        durations.push(samples as f64 / f64::from(SAMPLE_RATE));
    }
    engine.close().expect("close");

    // Each utterance is its own audio, not a repeat of the first.
    assert!(
        durations[1] > durations[0],
        "the long alert should outlast the short line: {durations:?}"
    );
    assert!(
        durations.iter().all(|d| *d > 1.0),
        "every utterance should produce real audio: {durations:?}"
    );
}

#[test]
fn speak_needs_an_open_session() {
    let Some(mut engine) = build() else {
        eprintln!("skipping: no engine/ tree staged");
        return;
    };
    assert!(
        engine.speak("Testing.").is_err(),
        "speaking without open() should fail rather than misbehave"
    );
}

#[test]
fn rejects_an_unknown_voice() {
    let Some(mut engine) = build() else {
        eprintln!("skipping: no engine/ tree staged");
        return;
    };
    let mut req = Request::new("Testing.");
    req.voice = "NoSuchVoice".to_string();
    assert!(
        engine.synthesize(&req).is_err(),
        "the engine should refuse a voice it cannot find"
    );
}
