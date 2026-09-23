//! `Sig` — fetch units from the bank, decode, splice, apply prosody.
//! **Phase 3.**
//!
//! 8,364 bytes, the smallest real stage.
//!
//! ```text
//! SigIniGlob   0x6104c    SigIniChan   0x60770
//! SigRun       0x5af60    SigRead      0x46488
//! SigWrite     0x46434    SigVoiceChange 0x615a8
//! UnsetCurrentBin 0x5b67c
//! ```
//!
//! This stage is where the `.sde` and `.bin` formats are actually consumed, so
//! Phase 1's parsers land here. `SigIniChan` is the reader to study for the
//! `.sde` numeric field encoding that `loqng-data::sde` currently leaves as
//! raw strings.
//!
//! The bank reaches the codec through `ELQBinOpen` (`0x93e94`),
//! `ELQBinGetBuffer` (`0x94a58`) and `ELQBinSeek` (`0x94a7c`). The engine
//! `mmap`s the 21 MB bank, so reads are absolute-offset, not stream-position —
//! getting that wrong yields a zero-filled map and a null decoder rather than
//! an error.
//!
//! # What is generated here
//!
//! [`sequens`] is the `SEQUENS` frame path — `sub_4bd38` and everything it
//! calls: epoch and border analysis, the pitch-synchronous and plain-stretch
//! bodies, and the two windowing primitives. Translated by
//! `tools/armxlate.py`; `xtask sigprobe` holds every function to the ARM
//! original.

pub mod sequens;
mod sequens_image;
