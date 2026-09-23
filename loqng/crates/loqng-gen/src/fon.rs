//! `Fon` — fonetizzatore: phonemes, syllables, duration, pitch. **Phase 6.**
//!
//! 51,784 bytes here, plus `ELQ-phonetics` (5,564) and `Psp` (512), plus the
//! language module's `FonemaLargo2StrettoUS`, `FonStretta2Larga_US` (`0x22a8c`),
//! `FonTraslittera_English` (`0x18214`) and `Omografo_English_US` (`0x18994`).
//!
//! ```text
//! FonSillabe       0x503cc    FonEstraiSillaba 0x508e4
//! FonCart          0x46b1c    FonIntrinseca    0x46c9c
//! FonDurata        0x46eb0    FonBaseLine      0x51454
//! FonGuadagno      0x514b4    FonTono          0x516f0
//! FonCalcProsLabStandard      0x49428
//! FonConfigTonoDicStandard    0x509a0
//! FonConfigTonoIntStandard    0x50b0c
//! FonWrite         0x45010    <- already decoded; see `crate::stream`
//! ```
//!
//! **This is the riskiest stage.** `FonTono`, `FonBaseLine` and `FonGuadagno`
//! are the float-heavy parts, and the target is byte-identical output
//! (`PLAN.md` §10.1), so FPA rounding has to match rather than merely be close.
//! Two facts that are settled and must not be re-derived:
//!
//! * ARM OABI doubles are **word-swapped** in memory — most significant word at
//!   the lower address. Any `.rodata` constant table read here decodes that way.
//! * libm is APCS-GNU/FPA: doubles pass in `r0:r1` with `r0` the high word, and
//!   return in `f0`.
//!
//! Phase prologue, before any Rust is written here: count the float
//! instructions in `FonTono`, `FonBaseLine` and `FonGuadagno`. If the total is
//! a few hundred, transcribe the operation sequence and the risk is gone.
//!
//! Gate: `PlainFonOut` matches across the corpus, and Track A + Track B
//! together give byte-identical audio end to end.
//!
//! # What is done
//!
//! The **broad-to-narrow phoneme rewrite** — [`exact::narrow`], bit-exact.
//! `FonemaLargo2Stretto_US` translated instruction for instruction into Rust by
//! `tools/fonxlate.py`; `xtask fonprobe --xlate` holds it to the ARM function.
//!
//! [`narrow`] and [`rules`] are the readable account of the same function: 123
//! rules lifted from the disassembly, agreeing with the engine on 89% of rule
//! choices. They document what the rules *are*; they are not what runs. See
//! `notes/eng-fonema-rules.md`.

pub mod exact;
mod fonema_us;
#[cfg(feature = "fon-cover")]
mod fonema_us_blocks;
mod fonema_us_image;
pub mod narrow;
pub mod phontab;
pub mod rules;
