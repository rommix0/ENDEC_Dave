//! `xtask refpcm` — the frame decoder's bit-exactness gate.
//!
//! Pushes identical coded frames through the ARM `sb_decode` and through
//! `loqng_codec::decoder::SbDecoder`, and compares the PCM sample for sample.
//! `xtask decode` shows the Rust decoder produces speech; this is what shows
//! it produces the *same* speech.
//!
//! # Why this does not go through `Decode`
//!
//! The obvious route is the module ABI — `Open` (`0x4990`) then `Decode`
//! (`0x48d0`) — but `Decode` needs a `sig` (an `ELQBin`):
//!
//! * **`ELQBinOpen`'s signature was already worked out** and is in
//!   `notes/stage-sig.md` §6 — this file claimed it was unknown, which was
//!   simply not checking the notes:
//!
//!   ```c
//!   int ELQBinOpen(ELQBin **out, int kind /* 'r' | 'm' | 'd' */,
//!                  void *logCtx, const char *path, void *owner);
//!   ```
//!
//!   The `kind` byte lands at `bin+5`, which is independently confirmed by
//!   the disassembly of `0x84588` below. So this route was open all along;
//!   the one taken here is still cheaper, because it needs no file at all.
//! * Fabricating a `sig` does not work either. `ELQBinGetBuffer` (`0x84a58`)
//!   is a 36-byte thunk onto `0x84588`, which switches on a mode byte at
//!   `sig[5]` — `'m'`, `'d'` or `'r'` — and every arm runs smart-buffer
//!   management (allocate or realloc to `max(len, 0x2800)`, then fill). It is
//!   not a window onto memory, so a hand-built descriptor would have to
//!   reimplement that logic.
//!
//! # The route this takes instead
//!
//! Drive Speex directly, the way the driver's own `init` does, and skip
//! `ELQBin` entirely:
//!
//! ```text
//! Open(&ctx, 16000, NULL)                        // allocates ctx + params
//! ctx[0x11d0] = 1            mode id: wideband   // normally set by init
//! ctx[0x11d2] = 48           coded frame bytes   // normally set by init
//! <fill ctx[0x11ec], the 0x88 voice-parameter block, from the bank>
//! state = speex_decoder_init(mode, params, 48, 1)
//! speex_decoder_ctl(state, 0x18, ctx + 0x30)
//! sb_decode(state, bits, out, frame_index)       // via +0x107b8
//! ```
//!
//! Three facts make it work, all established by disassembly:
//!
//! **1. `Open` needs no bank.** It takes only a sample rate, and it already
//! sets the `SpeexMode` pointer at `ctx[0x24]` (module `+0x1d4d8` for 16 kHz)
//! and allocates the `0x88` parameter block at `ctx[0x11ec]`. Everything
//! `init` adds on top is parsed out of the bank header — which
//! `loqng_data::bank` parses in Rust anyway.
//!
//! **2. `speex_decoder_init` takes four arguments, not one.** This is a third
//! Loquendo structural change on top of the two already recorded, and it is
//! plain at the call site in `init` (`0x57d4`..`0x57f0`):
//!
//! ```text
//! r0 = ctx[0x24]          the SpeexMode pointer
//! r1 = ctx[0x11ec]        the 0x88 voice-parameter block
//! r2 = ctx[0x11d2] (sh)   coded frame size, 48
//! r3 = ctx[0x11d0] (sh)   Speex mode id: 0 nb, 1 wb, 2 uwb
//! ```
//!
//! Passing the coded frame size *into the constructor* is how Loquendo
//! replaced the submode header: `SUBMODE_ENCODING` is 0, so the submode is
//! chosen from the frame size instead of read from the bitstream.
//!
//! **3. `sb_decode` is reached through a 20-byte dispatcher at `+0x107b8`**
//! which passes all four registers through untouched, so the signature is
//! `sb_decode(state, bits, out, frame_index)`. `0x1077c`, `0x10790` and
//! `0x107a4` are its siblings and are *not* interchangeable — they dispatch
//! `mode->[0x18]`, `[0x24]` and `[0x1c]`, and the first two clobber `r1` with
//! the mode pointer. Only `+0x28` is `dec`.

use std::fs;
use std::path::PathBuf;

use loqng_codec::bits::{field, Bits, SPEEX_BITS_LEN};
use loqng_codec::decoder::{SbDecoder, Voice, FULL_FRAME, NB_FRAME};
use loqng_codec::driver::VOICE_PARAMS_LEN;
use loqng_data::bank::{BankHeader, BankParams, BankTables, TableSpan};
use loqng_oracle::{Oracle, OracleConfig};

use crate::{flag, Paths};

const MODULE: &str = "loqmsx.so";

/// `Open(void **ctx_out, int sample_rate, void *log)`.
const OPEN: u32 = 0x4990;
/// `Close(void *ctx)` — from the factory, not guessed.
const CLOSE: u32 = 0x4bd8;
/// The exported factory: `loqmsx(id, 0)` returns an entry point.
const FACTORY: u32 = 0x4800;
/// `speex_decoder_init(mode, params, frame_bytes, mode_id)`.
const DECODER_INIT: u32 = 0x1076c;
/// `speex_decoder_ctl(state, request, ptr)`.
const DECODER_CTL: u32 = 0x10948;
/// `speex_decode_native(state, bits, out, frame_index)` -> `mode->dec`.
const DECODE_NATIVE: u32 = 0x107b8;

/// `SPEEX_SET_SAMPLING_RATE`. This is the request `init` issues right after
/// construction, with `ctx + 0x30` as its payload — whose first word is 16000.
const CTL_SAMPLING_RATE: u32 = 24;
/// `SPEEX_SET_ENH`.
const CTL_ENH: u32 = 0;
/// `SPEEX_SET_LOW_MODE`.
const CTL_LOW_MODE: u32 = 8;
/// `SPEEX_SET_HIGH_MODE`.
const CTL_HIGH_MODE: u32 = 10;
/// `SPEEX_SET_SUBMODE_ENCODING`. Setting this to 0 is what makes the submode
/// forced instead of read from the bitstream, and it is the single most
/// important line in `loqmsx_dll.c::make_decoder`.
const CTL_SUBMODE_ENCODING: u32 = 36;
/// `SPEEX_SET_HIGHPASS`. Loquendo does not highpass its output.
const CTL_HIGHPASS: u32 = 44;
/// `SPEEX_GET_EXC` — copies `frameSize` excitation samples out as `i16`.
///
/// This is the diagnostic that splits the decoder in half: if the excitation
/// matches but the PCM does not, the fault is in the synthesis loop (LSP
/// interpolation, `lsp_to_lpc`, `iir16`, the one-subframe delay); if the
/// excitation already differs, it is in the pitch/innovation/gain path.
const CTL_GET_EXC: u32 = 101;

/// Dave's narrowband submode in STOCK numbering, for `--force-submode`
/// only. This module's own arrays number it 6; see `sb.rs`.
const NB_SUBMODE: u32 = 5;
/// Dave's wideband submode in STOCK numbering. This module numbers it 3.
const SB_SUBMODE: u32 = 2;

/// Narrowband LSP ladder spacing, in `.bss`.
const SPACING_NB_ADDR: u32 = 0x1d8d8;
/// Wideband LSP ladder spacing, two bytes up.
const SPACING_SB_ADDR: u32 = 0x1d8da;

/// Dave's sample rate, which selects the wideband mode.
const RATE: u32 = 16000;
/// `SPEEX_MODEID_WB`.
const MODE_ID_WB: u32 = 1;

const DEFAULT_BANK: &str = "EnglishUs/Dave/Dave-19200.16000.loqmsx.bin";

/// Offsets inside the `0x88` voice-parameter block.
///
/// From `driver::VOICE_PARAMS_LEN`'s field map, plus the innovation fields
/// `innov.rs` recovered independently (`Band::cb_base_offset` and
/// `Band::entry_bytes_offset`).
mod p {
    pub const IDXARR: u32 = 0x00;
    pub const NB_NBITS: [u32; 3] = [0x08, 0x0c, 0x10];
    pub const SB_NBITS: [u32; 2] = [0x1c, 0x20];
    /// Narrowband LSP order. `nb_decoder_init` (`0x83d8`) copies this straight
    /// into `DecState+0x1c`, which is `lpcSize` — it sits between
    /// `nbSubframes` (4) and `min_pitch` (17) in the live struct.
    ///
    /// **The order is per-voice, not a mode constant.** Leaving it zero makes
    /// `lsp_to_lpc` walk off its scratch arrays and fault.
    pub const NB_ORDER: u32 = 0x2c;
    /// Wideband LSP order, copied to `SBDecState+0x18` by `sb_decoder_init`
    /// (`0xc3a0`).
    pub const SB_ORDER: u32 = 0x40;
    pub const NB_SPLIT_LO: u32 = 0x30;
    pub const NB_SPLIT_HI: u32 = 0x34;
    pub const NB_INNOV_SUBVECT: u32 = 0x38;
    pub const SB_INNOV_SUBVECT: u32 = 0x3c;
    pub const SB_SPLIT2: u32 = 0x44;
    pub const NB_LSP_CB: [u32; 3] = [0x54, 0x58, 0x5c];
    pub const NB_INNOV_CB: u32 = 0x60;
    pub const SB_INNOV_CB: u32 = 0x64;
    pub const SB_LSP_CB: [u32; 2] = [0x68, 0x6c];
    pub const NB_INNOV_SHIFT: u32 = 0x84;
    pub const SB_INNOV_SHIFT: u32 = 0x85;
}

/// A `SpeexBits` living in guest memory.
struct GuestBits {
    bits: u32,
    chars: u32,
    cap: usize,
}

impl GuestBits {
    fn new(o: &mut Oracle, cap: usize) -> Result<GuestBits, String> {
        let chars = o.alloc(cap as u32)?;
        let bits = o.alloc(SPEEX_BITS_LEN)?;
        o.write_u32(bits + field::CHARS, chars)?;
        o.write_u32(bits + field::OWNER, 0)?;
        Ok(GuestBits { bits, chars, cap })
    }

    fn load(&self, o: &mut Oracle, payload: &[u8]) -> Result<(), String> {
        if payload.len() > self.cap {
            return Err(format!(
                "payload {} exceeds {} bytes",
                payload.len(),
                self.cap
            ));
        }
        o.write_bytes(self.chars, payload)?;
        o.write_u32(
            self.bits + field::NB_BITS,
            (payload.len() as u32).wrapping_mul(8),
        )?;
        o.write_u32(self.bits + field::CHAR_PTR, 0)?;
        o.write_u32(self.bits + field::BIT_PTR, 0)?;
        o.write_u32(self.bits + field::OVERFLOW, 0)?;
        Ok(())
    }
}

fn dump_words(o: &Oracle, base: u32, at: u32, n: usize, label: &str) {
    print!("{label}");
    for i in 0..n {
        match o.read_u32(base + at + (i as u32) * 4) {
            Ok(v) => print!(" {v:08x}"),
            Err(_) => print!(" --------"),
        }
    }
    println!();
}

fn signed(b: &[u8]) -> Vec<i8> {
    b.iter().map(|&v| v as i8).collect()
}

fn span<'a>(raw: &'a [u8], s: &TableSpan, what: &str) -> Result<&'a [u8], String> {
    s.slice(raw)
        .ok_or_else(|| format!("{what}: span is outside the preamble"))
}

/// Copy a codebook into guest memory and return its address.
fn upload(o: &mut Oracle, bytes: &[u8]) -> Result<u32, String> {
    let a = o.alloc(bytes.len().max(4) as u32)?;
    o.write_bytes(a, bytes)?;
    Ok(a)
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let bank_path = flag(args, "--bank")
        .map(PathBuf::from)
        .unwrap_or_else(|| paths.data_dir.join(DEFAULT_BANK));
    let count: usize = flag(args, "--frames")
        .map(|s| s.parse().unwrap_or(200))
        .unwrap_or(200);

    let name = bank_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bank".to_string());

    // ---- parse the bank in Rust, no engine involved --------------------
    let raw = fs::read(&bank_path).map_err(|e| format!("{}: {e}", bank_path.display()))?;
    let header = BankHeader::parse(&name, &raw).map_err(|e| e.to_string())?;
    let tables = BankTables::parse(&name, &raw, &header).map_err(|e| e.to_string())?;
    let params = BankParams::derive(&name, &raw, &tables).map_err(|e| e.to_string())?;

    let idx_b = span(&raw, &tables.idx, "context array")?;
    let nb1_b = span(&raw, &tables.nb_cb1, "nb lsp 1")?;
    let nb2_b = span(&raw, &tables.nb_cb2, "nb lsp 2")?;
    let nb3_b = span(&raw, &tables.nb_cb3, "nb lsp 3")?;
    let nbi_b = span(&raw, &tables.nb_innov, "nb innovation")?;
    let sb1_b = span(&raw, &tables.sb_cb1, "sb lsp 1")?;
    let sb2_b = span(&raw, &tables.sb_cb2, "sb lsp 2")?;
    let sbi_b = span(&raw, &tables.sb_innov, "sb innovation")?;

    let frame_bytes = header.frame_bytes() as usize;
    let total = header.frames as usize;
    let count = count.min(total);

    println!(
        "{name}: {total} frames of {frame_bytes} coded bytes, key 0x{:02x}",
        tables.xor_key
    );
    println!("comparing the first {count} frames");
    println!();

    // ---- bring up the guest decoder ------------------------------------
    println!("opening the ARM engine (loqmsx is only mapped once a voice is)");
    let cfg = OracleConfig::new(&paths.lib_dir, &paths.data_dir).interpreted();
    let mut o = Oracle::open(&cfg)?;
    let base = o
        .module_base(MODULE)
        .ok_or_else(|| format!("{MODULE} is not loaded"))?;

    // Confirm the entry points against the factory rather than trusting the
    // offsets in this file.
    let want_open = o.call_at(MODULE, FACTORY, &[2, 0])?.wrapping_sub(base);
    let want_close = o.call_at(MODULE, FACTORY, &[3, 0])?.wrapping_sub(base);
    if want_open != OPEN || want_close != CLOSE {
        return Err(format!(
            "factory disagrees: Open +0x{want_open:05x} (expected +0x{OPEN:05x}), \
             Close +0x{want_close:05x} (expected +0x{CLOSE:05x})"
        ));
    }

    let ctx_out = o.alloc(4)?;
    let rc = o.call_at(MODULE, OPEN, &[ctx_out, RATE, 0])?;
    let ctx = o.read_u32(ctx_out)?;
    if rc != 0 || ctx == 0 {
        return Err(format!("Open failed: rc=0x{rc:08x}, ctx=0x{ctx:08x}"));
    }
    let mode = o.read_u32(ctx + 0x24)?;
    let gparams = o.read_u32(ctx + 0x11ec)?;
    if mode == 0 || gparams == 0 {
        return Err(format!(
            "Open left mode=0x{mode:08x} params=0x{gparams:08x}; both must be set"
        ));
    }
    println!(
        "  ctx 0x{ctx:08x}  mode +0x{:05x}  params 0x{gparams:08x}",
        mode.wrapping_sub(base)
    );

    // The two fields `init` would have set from the bank header.
    o.write_bytes(ctx + 0x11d0, &(MODE_ID_WB as u16).to_le_bytes())?;
    o.write_bytes(ctx + 0x11d2, &(frame_bytes as u16).to_le_bytes())?;

    // Upload the per-voice tables and fill the parameter block.
    let g_idx = upload(&mut o, idx_b)?;
    let g_nb = [
        upload(&mut o, nb1_b)?,
        upload(&mut o, nb2_b)?,
        upload(&mut o, nb3_b)?,
    ];
    let g_sb = [upload(&mut o, sb1_b)?, upload(&mut o, sb2_b)?];
    let g_nbi = upload(&mut o, nbi_b)?;
    let g_sbi = upload(&mut o, sbi_b)?;

    o.write_bytes(gparams, &vec![0u8; VOICE_PARAMS_LEN as usize])?;
    o.write_u32(gparams + p::IDXARR, g_idx)?;
    for i in 0..3 {
        o.write_u32(gparams + p::NB_NBITS[i], params.nb.lsp_nbits[i])?;
        o.write_u32(gparams + p::NB_LSP_CB[i], g_nb[i])?;
    }
    for i in 0..2 {
        o.write_u32(gparams + p::SB_NBITS[i], params.sb.lsp_nbits[i])?;
        o.write_u32(gparams + p::SB_LSP_CB[i], g_sb[i])?;
    }
    o.write_u32(gparams + p::NB_ORDER, params.nb.order as u32)?;
    o.write_u32(gparams + p::SB_ORDER, params.sb.order as u32)?;
    o.write_u32(gparams + p::NB_SPLIT_LO, params.nb.lsp_counts[1] as u32)?;
    o.write_u32(gparams + p::NB_SPLIT_HI, params.nb.lsp_counts[2] as u32)?;
    o.write_u32(gparams + p::SB_SPLIT2, params.sb.lsp_counts[1] as u32)?;
    o.write_u32(
        gparams + p::NB_INNOV_SUBVECT,
        params.nb.innov_subvect as u32,
    )?;
    o.write_u32(
        gparams + p::SB_INNOV_SUBVECT,
        params.sb.innov_subvect as u32,
    )?;
    o.write_u32(gparams + p::NB_INNOV_CB, g_nbi)?;
    o.write_u32(gparams + p::SB_INNOV_CB, g_sbi)?;
    o.write_u8(
        gparams + p::NB_INNOV_SHIFT,
        (14 - params.nb.innov_shift) as u8,
    )?;
    o.write_u8(
        gparams + p::SB_INNOV_SHIFT,
        (14 - params.sb.innov_shift) as u8,
    )?;

    // Do NOT set the `.bss` LSP spacings by hand. `nb_decoder_init` reaches
    // `0xbb7c`, which derives both from the two orders and stores them at
    // GOT-relative `[sl, #0x134]` and `[sl, #0x136]` — so anything written
    // here is overwritten. Reading them back afterwards instead gives a free
    // check on `BankParams::derive`.
    let state = o.call_at(
        MODULE,
        DECODER_INIT,
        &[mode, gparams, frame_bytes as u32, MODE_ID_WB],
    )?;
    if state == 0 {
        return Err("speex_decoder_init returned NULL".to_string());
    }
    o.write_u32(ctx + 0x28, state)?;
    let ctl = o.call_at(MODULE, DECODER_CTL, &[state, CTL_SAMPLING_RATE, ctx + 0x30])?;
    println!(
        "  decoder state 0x{state:08x}, sampling-rate ctl -> {}",
        ctl as i32
    );

    // What the module derived from the orders, against what Rust derived from
    // LADDER_SPAN. These must agree or one of the two is wrong.
    let g_nb_sp = o.read_bytes(base + SPACING_NB_ADDR, 2)?;
    let g_sb_sp = o.read_bytes(base + SPACING_SB_ADDR, 2)?;
    let g_nb_sp = u16::from_le_bytes([g_nb_sp[0], g_nb_sp[1]]);
    let g_sb_sp = u16::from_le_bytes([g_sb_sp[0], g_sb_sp[1]]);
    println!(
        "  LSP spacing: guest nb {g_nb_sp} sb {g_sb_sp} / rust nb {} sb {}",
        params.nb.spacing, params.sb.spacing
    );
    if g_nb_sp != params.nb.spacing || g_sb_sp != params.sb.spacing {
        return Err(format!(
            "spacing disagrees: the module derives ({g_nb_sp}, {g_sb_sp}) from orders \
             ({}, {}) but BankParams::derive says ({}, {}). LADDER_SPAN is wrong.",
            params.nb.order, params.sb.order, params.nb.spacing, params.sb.spacing
        ));
    }

    // `sb_decoder_init` (0xc308) lays out SBDecState as:
    //   +0x00 mode   +0x04 st_low   +0x28 coded frame bytes
    //   +0x2c mode id   +0x30 stack (24000 bytes)   +0x7c encode_submode = 1
    let st_low = o.read_u32(state + 0x04)?;
    let sb_stack = o.read_u32(state + 0x30)?;
    let sb_fb = o.read_u32(state + 0x28)?;
    let sb_enc = o.read_u32(state + 0x7c)?;
    println!(
        "  SBDecState: st_low 0x{st_low:08x} stack 0x{sb_stack:08x} \
         frame_bytes {sb_fb} encode_submode {sb_enc}"
    );
    if sb_stack == 0 {
        return Err("SBDecState.stack is NULL; the scratch allocation failed".to_string());
    }
    if st_low == 0 {
        return Err("SBDecState.st_low is NULL; the low band was not created".to_string());
    }
    println!("  DecState (low band) first words:");
    for row in 0..4u32 {
        dump_words(
            &o,
            st_low,
            row * 0x20,
            8,
            &format!("    +0x{:04x}", row * 0x20),
        );
    }
    println!(
        "    +0x01f0 params 0x{:08x}",
        o.read_u32(st_low + 0x1f0).unwrap_or(0)
    );

    // Exactly what `loqmsx_dll.c::make_decoder` does, and in its order.
    // Driving this through `decoder_ctl` rather than poking struct offsets
    // means it does not depend on knowing where `encode_submode` lives.
    let arg = o.alloc(4)?;
    let set = |o: &mut Oracle, who: u32, req: u32, val: u32, what: &str| -> Result<(), String> {
        o.write_u32(arg, val)?;
        let rc = o.call_at(MODULE, DECODER_CTL, &[who, req, arg])?;
        if rc as i32 != 0 {
            return Err(format!("{what}: decoder_ctl returned {}", rc as i32));
        }
        Ok(())
    };
    set(&mut o, state, CTL_ENH, 0, "sb enh")?;
    set(
        &mut o,
        state,
        CTL_SUBMODE_ENCODING,
        0,
        "sb submode encoding",
    )?;
    set(
        &mut o,
        st_low,
        CTL_SUBMODE_ENCODING,
        0,
        "nb submode encoding",
    )?;
    set(&mut o, state, CTL_HIGHPASS, 0, "sb highpass")?;
    set(&mut o, st_low, CTL_HIGHPASS, 0, "nb highpass")?;

    // DO NOT force the submode here. `loqmsx_dll.c` sets LOW_MODE 5 and
    // HIGH_MODE 2, but those are *stock Speex* submode numbers and this
    // module's submode array is numbered differently: `decoder_init` already
    // selected the right pair from the 48-byte coded frame size, which is the
    // whole reason the frame size is a constructor argument. Forcing 5 here
    // selects a submode with `split_cb {8, 5, 7}` and 216 bits a frame --
    // stock `nb_submode4` -- and the bitstream then misaligns by 64 bits.
    if args.iter().any(|a| a == "--force-submode") {
        set(&mut o, st_low, CTL_LOW_MODE, NB_SUBMODE, "nb submode")?;
        set(&mut o, state, CTL_HIGH_MODE, SB_SUBMODE, "sb submode")?;
        println!("  submode FORCED to stock numbering (nb {NB_SUBMODE}, sb {SB_SUBMODE})");
    }
    println!("  configured: submode encoding off, no highpass, submode left as chosen");

    // Read back the submode the guest will actually use, and the field widths
    // it implies. `SpeexSubmode` is 0x38 bytes:
    //   +0x08 have_subframe_gain  +0x14 lsp_unquant  +0x1c ltp_unquant
    //   +0x20 ltp_params          +0x28 innovation_unquant
    //   +0x2c innovation_params   +0x34 bits_per_frame
    // and `split_cb_params` is {subvect_size, nb_subvect, shape_cb,
    // shape_bits, have_sign}.
    // SBDecState mirrors the pattern: submodes at +0x80, submodeID at +0x84
    // (`sb_decoder_init` 0xc3e8/0xc3ec), as DecState has them at +0x74/+0x78.
    let sb_submodes = o.read_u32(state + 0x80)?;
    let sb_id = o.read_u32(state + 0x84)?;
    println!();
    println!("  sb submodes[] at 0x{sb_submodes:08x}, submodeID {sb_id}");

    let submodes = o.read_u32(st_low + 0x74)?;
    let submode_id = o.read_u32(st_low + 0x78)?;
    println!("  nb submodes[] at 0x{submodes:08x}, submodeID {submode_id}");
    let sm = o.read_u32(submodes + submode_id * 4)?;
    if sm != 0 {
        let hsg = o.read_u32(sm + 0x08)?;
        let dbl = o.read_u32(sm + 0x0c)?;
        let lbr = o.read_u32(sm)? as i32;
        let fpg = o.read_u32(sm + 0x04)?;
        let ltpp = o.read_u32(sm + 0x20)?;
        let ip = o.read_u32(sm + 0x2c)?;
        let bpf = o.read_u32(sm + 0x34)?;
        println!(
            "  submode 0x{sm:08x}: lbr_pitch {lbr} forced_gain {fpg} \
             subframe_gain {hsg} double_cb {dbl} bits/frame {bpf}"
        );
        if ltpp != 0 {
            let gb = o.read_u32(ltpp + 4)?;
            let pb = o.read_u32(ltpp + 8)?;
            println!("    ltp_params: gain_bits {gb}, pitch_bits {pb}");
        }
        if ip != 0 {
            let sv = o.read_u32(ip)?;
            let nsv = o.read_u32(ip + 4)?;
            let sb = o.read_u32(ip + 0xc)?;
            let sign = o.read_u32(ip + 0x10)?;
            println!(
                "    split_cb: subvect {sv}, nb_subvect {nsv}, shape_bits {sb}, \
                 have_sign {sign}  -> {} innovation bits",
                nsv * (sb + sign)
            );
        }
    }
    println!();

    // ---- the Rust side -------------------------------------------------
    let nb1 = signed(nb1_b);
    let nb2 = signed(nb2_b);
    let nb3 = signed(nb3_b);
    let nbi = signed(nbi_b);
    let sb1 = signed(sb1_b);
    let sb2 = signed(sb2_b);
    let sbi = signed(sbi_b);
    let voice = Voice {
        ctx: idx_b,
        nb_lsp: [&nb1, &nb2, &nb3],
        nb_innov: &nbi,
        sb_lsp: [&sb1, &sb2],
        sb_innov: &sbi,
        nb_innov_log2: (14 - params.nb.innov_shift) as u8,
        sb_innov_log2: (14 - params.sb.innov_shift) as u8,
    };
    let mut rust = SbDecoder::new();

    // ---- differential loop ---------------------------------------------
    //
    // `--nb` compares only the low band, by calling the dispatcher on
    // `st_low` instead of on the wideband state. That isolates `nb_decode`
    // from the wideband half and the QMF, which is the first thing to do when
    // the full-band comparison diverges.
    let nb_only = args.iter().any(|a| a == "--nb");
    let (who, width) = if nb_only {
        println!("comparing the LOW BAND only ({} samples a frame)", NB_FRAME);
        (st_low, NB_FRAME)
    } else {
        (state, FULL_FRAME)
    };
    let mut rust_nb = loqng_codec::decoder::NbDecoder::new();

    let gbits = GuestBits::new(&mut o, frame_bytes)?;
    let gout = o.alloc((FULL_FRAME * 2) as u32)?;
    let gexc = o.alloc((NB_FRAME * 2) as u32)?;
    let mut coded = vec![0u8; frame_bytes];
    let mut mine = vec![0i16; FULL_FRAME];
    let mut matched = 0usize;

    for f in 0..count {
        let at = header.data_offset as usize + f * frame_bytes;
        let src = raw
            .get(at..at + frame_bytes)
            .ok_or_else(|| format!("frame {f} runs past the end of the file"))?;
        for (d, s) in coded.iter_mut().zip(src) {
            *d = s ^ tables.xor_key;
        }

        // Guest.
        gbits.load(&mut o, &coded)?;
        o.write_bytes(gout, &vec![0u8; FULL_FRAME * 2])?;
        let grc = o.call_at(MODULE, DECODE_NATIVE, &[who, gbits.bits, gout, f as u32])?;
        let theirs = o.read_i16s(gout, width)?;

        // Rust.
        let mut bits = Bits::new(&coded);
        if nb_only {
            rust_nb
                .decode(&voice, f, &mut bits, &mut mine)
                .map_err(|e| format!("frame {f}: rust nb decoder: {e:?}"))?;
        } else {
            rust.decode(&voice, f, &mut bits, &mut mine)
                .map_err(|e| format!("frame {f}: rust decoder: {e:?}"))?;
        }
        let mine = &mine[..width];

        if grc as i32 != 0 {
            return Err(format!(
                "frame {f}: guest sb_decode returned {}",
                grc as i32
            ));
        }

        // Split the decoder in half before reporting a PCM mismatch.
        if nb_only {
            let gexc_raw = o.read_i16s(gexc, NB_FRAME)?;
            o.call_at(MODULE, DECODER_CTL, &[st_low, CTL_GET_EXC, gexc])?;
            let gexc_now = o.read_i16s(gexc, NB_FRAME)?;
            let _ = gexc_raw;
            let rexc = rust_nb.excitation();
            if gexc_now != rexc {
                let at = gexc_now
                    .iter()
                    .zip(rexc.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                let n = gexc_now
                    .iter()
                    .zip(rexc.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                println!("EXCITATION differs at frame {f}: {n} of {NB_FRAME}, first at {at}");
                let lo = at.saturating_sub(2);
                let hi = (at + 8).min(NB_FRAME);
                println!("  guest {:?}", &gexc_now[lo..hi]);
                println!("  rust  {:?}", &rexc[lo..hi]);

                // Did both sides read the same fields? If the bit cursors
                // agree the widths and order are right and the fault is
                // arithmetic; if they differ it is a field-layout error.
                let gcp = o.read_u32(gbits.bits + field::CHAR_PTR)?;
                let gbp = o.read_u32(gbits.bits + field::BIT_PTR)?;
                let gpos = gcp * 8 + gbp;
                println!();
                println!("  bit cursor: guest {gpos}, rust {}", bits.position());
                if gpos != bits.position() as u32 {
                    println!("  -> the two read DIFFERENT field widths. Fix the layout first.");
                } else {
                    println!("  -> same fields consumed, so this is an arithmetic difference.");
                }

                // Magnitude of the disagreement, per subframe.
                println!();
                for sub in 0..4 {
                    let r = sub * 40..(sub + 1) * 40;
                    let d = gexc_now[r.clone()]
                        .iter()
                        .zip(&rexc[r])
                        .map(|(a, b)| (*a as i32 - *b as i32).abs())
                        .max()
                        .unwrap_or(0);
                    println!("  subframe {sub}: largest difference {d}");
                }
                println!();
                println!("So the fault is upstream of synthesis: pitch, innovation or gain.");
                let _ = o.call_at(MODULE, CLOSE, &[ctx]);
                return Err("not bit-exact".to_string());
            } else if theirs != *mine {
                println!("frame {f}: EXCITATION MATCHES but PCM does not.");
                println!("So the fault is in the synthesis loop, not the codebooks.");
            }
        }

        if theirs != *mine {
            let n = theirs
                .iter()
                .zip(mine.iter())
                .filter(|(a, b)| a != b)
                .count();
            let first = theirs
                .iter()
                .zip(mine.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            println!("MISMATCH at frame {f}: {n} of {width} samples differ");
            println!("  first at sample {first}");
            let lo = first.saturating_sub(2);
            let hi = (first + 6).min(width);
            println!("  guest {:?}", &theirs[lo..hi]);
            println!("  rust  {:?}", &mine[lo..hi]);
            println!();
            println!("{matched} frames matched before this one.");
            let _ = o.call_at(MODULE, CLOSE, &[ctx]);
            return Err("not bit-exact".to_string());
        }
        matched += 1;

        if matched % 50 == 0 {
            println!("  {matched} frames bit-exact");
        }
    }

    let _ = o.call_at(MODULE, CLOSE, &[ctx]);
    println!();
    println!(
        "BIT-EXACT: {matched} frames, {} samples, identical to the ARM decoder",
        matched * FULL_FRAME
    );
    Ok(())
}
