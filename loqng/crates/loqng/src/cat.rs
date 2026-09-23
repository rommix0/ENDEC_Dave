//! `Cat` — concatenatore: unit selection over the diphone bank. **Phase 4.**
//!
//! 34,016 bytes. The hard part is already done: the context-cost function at
//! `0x39f1c` is ported bit-exact in `loqrs/crates/loqhost/src/cat.rs` and that
//! transcription is the starting point here, not a fresh read.
//!
//! What it established, and what must not be re-derived:
//!
//! * `glob+0x774` is `pos`; `unit = pos >> 1` and `pos & 1` picks which half of
//!   a diphone. Even weights right 10/6/3/1 and left 10/4/2/1; odd weights
//!   right 10/10/6/3/1 and left 6/3/1.
//! * The pairwise class-penalty table is at `module+0x8EB2C`, indexed
//!   `[y * 16 + x]` of `i32`.
//! * `glob+0x91c` -> `{+4 = row stride, +8 = distance table}`; `glob+0x40` ->
//!   pointer to pointer to 0x18-byte phone records with the weight at `+5`.
//! * `glob+0x918` receives the concatenation half of the score as a side
//!   effect.
//! * The open-ended left tail stops differently from the fixed levels and
//!   leaves a different value in `back` for the two diphone halves. Do not
//!   unify them.
//!
//! Still to do:
//!
//! ```text
//! 0x3b014   the candidate loop (calls the context cost)
//! 0x3aa84   ~8.1% of synthesis; calls 0x3a8b4
//! 0x3acb4
//! CatRun    0x3c21c
//! CatWrite  0x46018    the stream serialiser, as FonWrite was for Fon
//! ```
//!
//! `TraceSelection` makes the engine narrate its own decisions; turn it on
//! before reading any of this.
//!
//! Gate: Track A complete — byte-identical audio from a captured phonetic
//! stream, with no ARM code left in the signal path.
