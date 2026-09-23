//! Generated tables and translations, kept out of `loqng` so that editing
//! the runtime does not recompile three megabytes of them.
//!
//! Nothing here is written by hand:
//!
//! ```text
//! fon::phontab   tools/genphon.py       the 87 US phone records
//! fon::rules     tools/genfonrules.py   the 123-rule table (documentation)
//! fon::narrow    the engine over that table (documentation)
//! fon::exact     tools/fonxlate.py      FonemaLargo2Stretto_US, bit-exact
//! sig::sequens   tools/armxlate.py      the SEQUENS DSP, bit-exact
//! ```
//!
//! `fon::exact` and `sig::sequens` are *second* translations of code that
//! `loqng-eng` also contains. That redundancy is deliberate: two independent
//! translations agreeing with the ARM original is a far stronger check than
//! either alone, and it is what `xtask engprobe` and `xtask sigprobe` use.

pub mod fon;
pub mod sig;
