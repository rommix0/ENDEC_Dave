//! `Acu` — acustico: the output queue and the audio sink. **Phase 3.**
//!
//! 12,384 bytes, most of it queueing rather than signal work.
//!
//! ```text
//! AcuIniGlob  0x385bc    AcuIniChan   0x3562c
//! AcuSetAudio 0x358c4    AcuRun       0x36690
//! AcuRead     0x46aac    AcuMsec      0x37208
//! AcuFinished 0x371c4    AcuResetSampleCounter 0x37258
//! ```
//!
//! The sink modules are separate shared objects and are tiny: `LoqAudioFile.so`
//! is 2,656 bytes of `.text` with 7 exports, and is replaced by a WAV/raw
//! writer rather than ported.
//!
//! `MixResampler` (`0x97a14`) is reached from here, but only when the output
//! rate differs from the bank's. The Dave bank decodes to 16 kHz and is
//! requested at 16 kHz, so the resampler is not on the path this port must
//! reproduce first.
//!
//! One behaviour worth pinning early, because it is observable in the output
//! and is not obvious: the ENDEC's writer emits whole 4096-byte blocks and
//! drops the trailing partial one. That discards real speech — up to 65 ms in
//! two of the eight regression texts. `loqrs` keeps the tail by default and
//! reproduces the device only under `--block-align 4096`. Decide which one
//! `loqng` matches and make it explicit, not incidental.
