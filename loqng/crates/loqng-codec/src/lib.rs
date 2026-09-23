//! `loqmsx` — the speech-database codec. **Phase 2.**
//!
//! 69,536 bytes of `.text` behind a single exported symbol, and the single
//! best-understood module in the engine: it is **Speex 1.2beta1**, built with
//! `libspeex/fixed_arm4.h`, with three Loquendo customisations — a per-voice
//! LSP codebook, a per-voice innovation codebook, and no output highpass.
//!
//! That makes this a source-guided port rather than a blind one. `fixed_arm4.h`
//! overrides exactly three macros — `MULT16_32_Q14`, `MULT16_32_Q15` and
//! `DIV32_16` — and everything else is stock `fixed_generic.h`, so the
//! arithmetic of any routine can be predicted from upstream source and then
//! confirmed against the disassembly.
//!
//! **Where the C and the disassembly disagree, the disassembly wins.**
//! `signal_mul` is the known case: it looks like it should truncate via
//! `EXTRACT16` and split into two 16x16 multiplies, and the shipped code does
//! neither.
//!
//! # Module ABI
//!
//! `loqmsx(id, 0)` returns a function pointer from a 6-entry `{key, fnptr}`
//! table at `.data 0x1d000`:
//!
//! | id | what | ARM entry |
//! |----|------|-----------|
//! | 1 | version -> `"6.1"` | `0x4880` |
//! | 2 | Open | `0x4990` |
//! | 3 | Close | `0x4bd8` |
//! | 4 | **Decode** | `0x48d0` |
//! | 5 | getcaps | `0x48ac` |
//! | 6 | name -> `"loqmsx"` | `0x485c` |
//!
//! # Already ported bit-exact — and why they were still re-proved
//!
//! Six routines live in `loqrs/crates/loqhost/src/msx.rs` and in `libc.rs`:
//! `fir2x` (`0x2c44`), `iir32` (`0x22c4`), `iir16` (`0x3cd0`), `lsp2lpc`
//! (`0x6500`), `sigmul` (`0x3c98`) and `unpack` (`0x14ac`). They were the
//! starting point here rather than a fresh read.
//!
//! **`loqrs` validates them by whole-utterance audio diff.** That is a strong
//! end-to-end gate — hours of speech, byte-identical — but a blunt one: it
//! says the engine matched, not that every routine is right on every input.
//! Running them through `xtask verify` found a real defect in `lsp2lpc` that
//! the audio gate had never exercised. See [`lpc::saturate_ak`] and
//! [`lpc::mult16_16_p13`].
//!
//! ## The trap: `loqrs` patches these at run time
//!
//! Those six addresses are in `loqrs`'s `NATIVE_PATCHES`, so with patching on,
//! *calling `0x6500` in the guest runs `loqrs`'s Rust, not the ARM*. A
//! differential test against it proves only that one transcription matches
//! another. `xtask verify` therefore opens the oracle with
//! `OracleConfig::interpreted()`, and anything else that means to check
//! against the original must do the same.
//!
//! # Where the time actually goes
//!
//! Re-measured with `loqdave --profile-raw` over three long texts (601,789
//! samples, of which `loqmsx.so` is 67.6% — matching the 66.8% on record, so
//! the measurement is sound). Samples are attributed to the **enclosing
//! function**, using Ghidra's function starts rather than push/pop pairing.
//!
//! | vaddr | % of synthesis | what |
//! |---|---:|---|
//! | `0xc5bc` | 25.0% | [`sb`] — `sb_decode` |
//! | `0x8690` | 10.2% | [`nb`] — `nb_decode` |
//! | `0x153c` | 9.2% | [`innov`] — `split_cb_shape_sign_unquant` |
//! | `0x7414` | 9.1% | [`ltp`] — `pitch_unquant_3tap` |
//! | `0x1d90` | 4.9% | `compute_rms16` |
//! | `0x6a70` | 2.8% | `lsp_interpolate` |
//! | `0x69bc` | 2.5% | `lsp_enforce_margin` |
//! | `0x5718` | 1.3% | [`driver`] — the per-frame core |
//!
//! **This corrects the table in `loqrs/PORTING.md`**, which listed `0x864c` at
//! ~5.5%. `0x864c` is a destructor — seven `free()` calls, 68 bytes — and takes
//! **zero** samples. The hot code is its neighbour `0x8690`, which the old
//! attribution folded into it. That is the exact failure mode `PORTING.md`
//! itself warns about, having already been caught by it once at `0xc518`.
//!
//! Two genuinely hot functions were missing from that table altogether:
//! `0x153c` at 9.2% and `0x1d90` at 4.9%.
//!
//! `0x7414` was listed as unidentified on the grounds that its arg 6 did not
//! match `seed` in the innovation signature. It is the **pitch** unquantiser,
//! where arg 6 is `ltp_params`. The name `split_cb_shape_sign_unquant` that
//! was guessed for it belongs to `0x153c`.
//!
//! # Nothing hot is unidentified any more
//!
//! Every function above 1% of synthesis has a name. What remains is porting
//! work, not reverse engineering.
//!
//! ## Ported and proved bit-exact
//!
//! Each is checked against the ARM original by `xtask verify`, which calls the
//! guest routine and the Rust one over thousands of inputs and compares. A
//! unit test here only pins what I believed; `verify` pins what is true.
//!
//! | routine | offset | % | cases |
//! |---|---|---:|---:|
//! | [`math::spx_sqrt`] | `0x7fb8` | — | 4,025 |
//! | [`lsp::enforce_margin`] | `0x69bc` | 2.5% | 3,000 |
//! | [`math::compute_rms16`] | `0x1d90` | 4.9% | 2,000 |
//! | [`lsp::interpolate`] | `0x6a70` | 2.8% | 2,000 |
//! | [`innov::unquant`] | `0x153c` | 9.2% | 1,500 |
//! | [`ltp::unquant`] | `0x7414` | 9.1% | 1,500 |
//! | [`lsp::unquant`] x4 | `0xbda0` `0xbe50` `0xbf64` `0xc0e0` | <1% | 4,800 |
//! | [`filters::fir2x`] | `0x2c44` | — | 1,500 |
//! | [`filters::iir32`] | `0x22c4` | — | 1,500 |
//! | [`filters::iir16`] | `0x3cd0` | — | 1,500 |
//! | [`filters::signal_mul`] | `0x3c98` | — | 2,000 |
//! | [`lpc::lsp_to_lpc`] | `0x6500` | — | 2,000 |
//!
//! That is every pure leaf on the decode path, plus the three routines that
//! read the bitstream but not a decoder. [`math::div15`] — the module's
//! inlined 15-bit restoring division, which appears three times — is proved
//! through `lsp_interpolate`, which is built on it. [`bits`] is proved through
//! all three unquantisers, whose bit cursors are compared as well as their
//! outputs.
//!
//! **28.5% of synthesis is now ported and proved**, and all three submode
//! callbacks with it.
//!
//! The two big unquantisers take 8 and 15 arguments, of which 1 and 9
//! respectively are never read. `xtask verify` passes junk for every one of
//! those rather than zero, so "unread" is a tested claim and not an
//! assumption. `lsp_unquant`'s spacing global is likewise **driven** through a
//! range rather than left at whatever a cold module holds, because it reads 0
//! there and a sweep stuck at 0 would never exercise the multiply.
//!
//! # The submode table
//!
//! All three callbacks hang off a 0x38-byte submode struct in `.data`, found
//! by searching for the addresses of routines that have **no `bl` caller
//! anywhere** — they are reached only through this table:
//!
//! It is **stock `SpeexSubmode`, member for member** — 14 words, encoder slots
//! NULL in this decoder-only build:
//!
//! ```text
//! +0x00  lbr_pitch           0, or -1 on submodes 3+
//! +0x04  forced_pitch_gain   0
//! +0x08  have_subframe_gain  1 or 3
//! +0x0c  double_codebook     0 or 1
//! +0x10  lsp_quant           0        (encoder, NULL)
//! +0x14  lsp_unquant         0xbda0 / 0xbe50 / 0xbf64 / 0xc0e0
//! +0x18  ltp_quant           0x708c   pitch_search_3tap (encoder)
//! +0x1c  ltp_unquant         0x7414
//! +0x20  ltp_params          .data {gain_cdbk, gain_bits, pitch_bits}
//! +0x24  innovation_quant    0        (encoder, NULL)
//! +0x28  innovation_unquant  0x153c
//! +0x2c  innovation_params   .data split_cb_params
//! +0x30  comb_gain           Q15
//! +0x34  bits_per_frame
//! ```
//!
//! **Eight narrowband submodes at `.data 0x1d120 + k*0x38`**, and three
//! wideband ones at `0x1d384 + k*0x38` (all with the LTP slots NULL, as
//! wideband has no pitch predictor). `nb_decode` is pointed to from `0x1d308`,
//! `sb_decode` from `0x1d500` and `0x1d650`.
//!
//! Dave forces **NB submode 5** (`0x1d238`, `comb_gain` `0x2666` = 0.3 in Q15,
//! matching stock `nb_submode5`) and **SB submode 2** (`0x1d3bc`).
//!
//! Getting the anchor wrong by three words makes every field name slide, so
//! pin it on a slot whose contents are already known — `ltp_unquant` at `+0x1c`
//! was verified independently before the table was read.
//!
//! ## Not ported
//!
//! The two decoders, [`sb`] (`0xc5bc`, 25%) and [`nb`] (`0x8690`, 10.2%).
//! Both are mapped in full and their traps are documented, but they carry deep
//! state and dispatch through submode vtables, so they need a harness that can
//! stand a decoder up rather than build its arguments by hand. That is the
//! next piece of work.
//!
//! Also unported: the `mode->dec` dispatcher `0x107b8` (20 bytes, which is why
//! it never appears in a profile despite being on every decode path).
//!
//! # The three customisations, located
//!
//! The module is known to differ from stock Speex in three ways. Two are now
//! pinned to specific code:
//!
//! * **per-voice innovation codebook** — [`innov`], `0x153c`, now read in full
//!   and proved. It ignores `split_cb_params.shape_cb` entirely and derives
//!   the codebook from decoder state plus a per-voice selector byte:
//!   `cb = state[+0x60] + state[+0x38] * ((1 << shape_bits) * sel)` for the
//!   low band, `+0x64`/`+0x3c` for the high, with `sel` fetched through the
//!   pointer at `state[+0x00]`. The differential harness passes `0xdeadbeef`
//!   as `shape_cb` and still matches, so "unread" is measured.
//! * **no output highpass** — visible as the absence of a highpass call on the
//!   way out of [`sb`].
//! * **per-voice LSP codebook** — [`lsp::unquant`], reached through the
//!   submode struct at `+0x14`. Now read and proved. It keeps stock Speex's
//!   multi-stage structure but sources every stage's codebook from decoder
//!   state, chosen by the same per-voice selector byte the innovation codebook
//!   uses, and takes the ladder's spacing from `.bss` instead of the
//!   compile-time `LSP_LINEAR`. **Four variants ship** — 1, 2 and 3-stage
//!   narrowband plus a 2-stage wideband — and Dave uses the 3-stage `0xbf64`
//!   and the wideband `0xc0e0`.
//!
//! All three are now located, read and differentially verified.
//!
//! # A second, independent source: `loqrs/sapi/`
//!
//! `loqrs` already carries a **complete C reimplementation of this module**,
//! proven bit-exact against the real decoder: `sapi/loqmsx_dll.c` (module ABI
//! and frame cache), `sapi/loq_lsp.c` (the custom unquantisers) and
//! `speex_mod/` (upstream Speex 1.2beta1 plus a one-file `modes.c` patch that
//! repoints `nb_submode5` and `wb_submode2` at them).
//!
//! That changes what the remaining work is. `nb_decode` and `sb_decode` are
//! **stock Speex**, so they can be ported from upstream source rather than
//! read out of ARM — the shim demonstrates that stock plus these three
//! customisations is sufficient.
//!
//! It is also a cross-check, and it has already earned its keep: the shim's
//! three-stage `loq_nb_lsp_unquant` is what showed that a single-stage reading
//! of `lsp_unquant` had to be wrong. Where the two disagree the ARM still
//! wins, but a disagreement is always worth resolving.
//!
//! The shim's signatures are the **stock** 3- and 6-argument ones, because it
//! is called from unmodified Speex; the ARM module's are 5 and 8 arguments,
//! carrying state and a codebook selector that the shim keeps in file-scope
//! globals instead. Do not expect the two argument lists to match.
//!
//! # Stock Speex tables, located byte-exactly
//!
//! A port can read these from the bank's own module image rather than
//! embedding a copy:
//!
//! ```text
//! exc_10_16_table  0x11d28    hexc_10_32_table 0x12eae
//! exc_10_32_table  0x11dc8    hexc_table       0x12fee
//! exc_20_32_table  0x11f08    high_lsp_cdbk    0x133ee
//! exc_5_256_table  0x12188    high_lsp_cdbk2   0x135ee
//! exc_5_64_table   0x12688    cdbk_nb          0x137f4
//! exc_8_128_table  0x127c8    cdbk_nb_low1/2   0x13a74 / 0x13bb4
//! gain_cdbk_nb     0x12c2e    cdbk_nb_high1/2  0x13cf4 / 0x13e34
//! gain_cdbk_lbr    0x12e2e
//! ```
//!
//! The `split_cb_params` structs referencing them start at `.data 0x1d074`,
//! laid out as stock `{subvect_size, nb_subvect, shape_cb, shape_bits,
//! have_sign}`; `0x1d0bc` is `split_cb_nb` = `{5, 8, exc_5_64_table, 6, 0}`.
//!
//! Gate: decodes every unit in `Dave-19200.16000.loqmsx.bin` byte-identically
//! to the ARM module, driven by captured `Sig` inputs.

/// `speex_bits_unpack_unsigned` at `0x14ac`, the bitstream reader.
pub mod bits;

/// The four signal-path filters: `fir2x`, `iir32`, `iir16`, `signal_mul`.
pub mod filters;

/// `lsp_to_lpc` at `0x6500`, with `spx_cos` inlined.
pub mod lpc;

/// The decode driver: module ABI, frame addressing, XOR descrambling and the
/// seek/preroll rules. Everything in `loqmsx.so` that is not Speex.
pub mod driver;

/// `sb_decode`, the wideband sub-band CELP decoder at `0xc5bc`.
pub mod sb;

/// `pitch_unquant_3tap`, the 3-tap long-term predictor at `0x7414`.
pub mod ltp;

/// `split_cb_shape_sign_unquant`, the innovation unquantiser at `0x153c`.
pub mod innov;

/// `nb_decode`, the narrowband CELP decoder at `0x8690`.
pub mod nb;

/// `spx_sqrt`, the fixed-point square root at `0x7fb8`.
pub mod math;

/// `lsp_enforce_margin` at `0x69bc`.
pub mod lsp;

/// Constant tables transcribed from Speex 1.2beta1. Generated; do not edit.
pub mod tables;

/// The frame loop: `nb_decode` and `sb_decode`, built on the modules above.
pub mod decoder;

/// The concatenation window tables. Generated; do not edit.
pub mod window;

/// Random access into a bank: the frame cache and preroll `Decode` implements.
pub mod reader;
