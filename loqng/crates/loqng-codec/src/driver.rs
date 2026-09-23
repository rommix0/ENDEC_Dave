//! The decode driver: module ABI, frame addressing, and the decoder state.
//!
//! This is the part of `loqmsx.so` that is **not** Speex. The Speex core can be
//! ported from source (1.2beta1 + `fixed_arm4.h`, three Loquendo tweaks), but
//! everything here is Loquendo's own and has to be read out of the binary.
//!
//! Addresses are `loqmsx.so` module offsets, matching `loqrs`'s
//! `NATIVE_PATCHES`. Add `0x10000` for the Ghidra listing.
//!
//! # Module ABI
//!
//! `loqmsx(id, 0)` walks a 6-entry `{u32 key, u32 fnptr}` table and returns the
//! pointer whose key matches, or 0. The table is at `.data + 0`, and **the
//! entries are not in key order**:
//!
//! | slot | key | what | entry |
//! |---|---|---|---|
//! | 0 | 1 | version -> `"6.1"` | `0x4880` |
//! | 1 | 2 | Open | `0x4990` |
//! | 2 | 3 | Close | `0x4bd8` |
//! | 3 | 5 | getcaps | `0x48ac` |
//! | 4 | 6 | name -> `"loqmsx"` | `0x485c` |
//! | 5 | 4 | **Decode** | `0x48d0` |
//!
//! Reading that table from the file needs care: `.data` has vaddr `0x1d000` but
//! **file offset `0x15000`** — a delta of `0x8000`. `vaddr == file offset` holds
//! only for the first LOAD segment (`.text` and `.rodata`), not for `.data`.
//!
//! # Decode
//!
//! **Five arguments.** A 6-register push plus `sub sp, sp, #0xc` puts the one
//! stack argument at `sp+0x24`, so `bin` is argument 4, not 5:
//!
//! ```text
//! Decode(r0 ctx, r1 start_pos, r2 count, r3 out, sp+0x24 bin) -> int
//!
//!     if !ctx || !out || !bin : return 0xe0060006
//!     if !count : return 0
//!     rc = init(ctx, bin)                          // 0x4c50
//!     if rc : Close(ctx); return rc
//!     start_pos += ctx[0x34]
//!     do { count = frame(ctx, &out, &start_pos, count, bin) } while (count)
//! ```
//!
//! `frame` (`0x5718`) returns how many samples are still wanted, so the loop
//! runs until the request is satisfied. It takes `&out` and `&start_pos` as
//! pointers to `Decode`'s own stack slots and advances both.
//!
//! This matches the ABI `loqrs/sapi/loqmsx_dll.c` documents for its own
//! reimplementation:
//!
//! ```c
//! int Decode(void *ctx, int startPos, unsigned count, short *out, void *sig)
//! ```
//!
//! `start_pos` is a **sample position**, not a pointer; `out` is the PCM
//! destination. An earlier version of this note had the two the other way
//! round and listed a sixth argument that does not exist.
//!
//! # Decoder state
//!
//! Offsets recovered from `frame`'s literal pool:
//!
//! ```text
//! +0x0024  u32   passed as arg 1 to speex_decoder_init
//! +0x0028  ptr   the Speex decoder state
//! +0x0030  ...   payload for decoder_ctl request 0x18
//! +0x0034  u32   OUTPUT DELAY in samples, 170. `Decode` adds it to the
//!                requested start position, and the wrap path subtracts it.
//!                Named "wrap length" here once; it is the codec's own group
//!                delay, and omitting it makes every rendered unit start
//!                170 samples early. See `reader::OUTPUT_DELAY`.
//! +0x0038  u32   PREROLL frames -- see below
//! +0x003c  u32   first sample of the current frame
//! +0x0040  u32   last sample of the current frame
//! +0x0044  u32   first sample of the previous frame
//! +0x0048  u32   last sample of the previous frame
//! +0x004c  u32   bias added to the bank byte offset
//! +0x11d0  u16   passed as arg 4 to speex_decoder_init
//! +0x11d2  u16   CODED FRAME SIZE in bytes (48 for every Dave bank)
//! +0x11d4  u16   samples per frame (320)
//! +0x11d8  ptr   decoded PCM, current frame
//! +0x11dc  ptr   decoded PCM, previous frame
//! +0x11e0  u32   current frame index
//! +0x11e4  u32   previous frame index
//! +0x11e8  u8    XOR key; zero means no descrambling
//! +0x11ec  ptr   bank descriptor, whose +0x78 is the payload offset
//! +0x11f4  u32   total frames in the bank
//! ```
//!
//! # Frame addressing
//!
//! ```text
//! ELQBinGetBuffer(bin, frame_bytes * index + bank[0x78] + state[0x4c],
//!                 frame_bytes, 1, 0)
//! ```
//!
//! A flat array of fixed-size coded frames, which is exactly what the `.bin`
//! header describes: `(file_len - data_offset) / frames` is 48.0000 on all
//! three Dave banks, and 48 x 8 x 50 = 19,200 bps = the bitrate in word 0.
//!
//! # Two things Speex source will not tell you
//!
//! **1. Frames are XOR-scrambled.** If `state[0x11e8]` is non-zero, every byte
//! of the coded frame is XORed with it before the bits are read:
//!
//! ```text
//! for i in 0..frame_bytes: buf[i] ^= key
//! ```
//!
//! A port that skips this decodes noise on any bank that uses it.
//!
//! **2. A random seek pre-rolls.** Speex is stateful, so jumping to an
//! arbitrary frame does not give the right output. `frame` handles three cases:
//!
//! * target == current frame — serve from the current buffer;
//! * target == current + 1 — rotate current into previous, decode one frame;
//! * target == previous frame — serve from the previous buffer;
//! * anything else — **destroy and re-create the decoder**, then decode
//!   forward from `target - state[0x38]` up to and including `target`,
//!   discarding the output, so the filter state converges before the frame
//!   that is actually wanted.
//!
//! `state[0x38]` is the preroll depth and must be reproduced exactly: too few
//! frames and the output differs, too many and it differs as well, because the
//! decoder is re-initialised at a different point.
//!
//! # Speex entry points in this module
//!
//! **CORRECTED.** The first four were written with Ghidra's `+0x10000` bias
//! still applied; `.text` ends at `0x11d28`, so `0x2076c` is not even in the
//! module. All of these are module offsets, and `xtask refpcm` calls
//! `0x1076c` and `0x10948` successfully:
//!
//! | offset | upstream name |
//! |---|---|
//! | `0x1076c` | `speex_decoder_init` — **four args here**, see below |
//! | `0x10790` | `speex_decoder_destroy` |
//! | `0x10920` | `speex_decode_int` |
//! | `0x10948` | `speex_decoder_ctl` |
//! | `0x107b8` | `speex_decode_native`, the 20-byte `mode->dec` dispatcher |
//! | `0x10dd8` | `speex_bits_init` |
//! | `0x10dec` | `speex_bits_reset` |
//! | `0x10e20` | `speex_bits_read_from` |
//!
//! `speex_decoder_init` is **not** the stock one-argument function. It takes
//! `(mode, voice_params, coded_frame_bytes, mode_id)` — the frame size goes
//! to the constructor so the module can pick the submode from it instead of
//! reading a submode header. See `xtask refpcm`.
//!
//! `decoder_ctl` is called twice after init: request `0x18` with `state+0x30`,
//! then request `0` with a zeroed local.

/// Bytes `Open` allocates for the decoder context.
///
/// `Open(r0 = &ctx_out, r1 = sample_rate, r2 = ...)` mallocs this, zeroes it,
/// writes the pointer through `r0`, then mallocs a second block of
/// [`VOICE_PARAMS_LEN`] and stores it at `ctx[0x11ec]`. The sample rate is
/// checked against 8000, 11000, 11025 and others, so it really is a rate.
///
/// 0x11f8 is the last word past the `+0x11f4` field in the state map above,
/// which is how that map is known to be complete.
pub const CTX_LEN: u32 = 0x11f8;

/// How the customisation reaches the unquantisers.
///
/// This is the last structural unknown, and it is settled by the call site
/// inside `nb_decode`:
///
/// ```asm
/// ldr r4, [r6, #0x74]     ; st->submodes
/// ldr r7, [r6, #0x78]     ; st->submodeID
/// ldr r5, [r4, r7, lsl #2]; SUBMODE(...)
/// ldr r3, [r6, #0x1f0]    ; <- the per-voice parameter block
/// ldr r2, [sp, #0x64]     ; bits
/// ldr r0, [sp, #0x34]     ; qlsp
/// ldr r2, [sp, #0x5c]     ; frame index, an argument to nb_decode ITSELF
/// str r2, [sp]
/// mov lr, pc
/// ldr pc, [r5, #0x14]     ; submode->lsp_unquant
/// ```
///
/// So Loquendo made exactly two structural changes to Speex:
///
/// * **`DecState` gains a pointer at `+0x1f0`** holding the per-voice
///   parameter block — the [`VOICE_PARAMS_LEN`] allocation. `init` (`0x4c50`)
///   fills it from the bank preamble.
/// * **`nb_decode` and `sb_decode` take the frame index as an extra
///   argument** and thread it down to all three unquantisers, which use it as
///   `params[+0x00][frame_index]` to pick the context.
///
/// Everything else about the call is stock: the submode struct is indexed the
/// usual way and the callback sits at its usual offset. That is why a port can
/// take upstream `nb_celp.c` almost verbatim and only widen those signatures.
pub const DEC_STATE_PARAMS_PTR: u32 = 0x1f0;

/// Bytes of the second allocation, at `ctx[0x11ec]`.
///
/// **This is the struct the three customised unquantisers are handed**, and
/// every field they read fits inside it:
///
/// ```text
/// +0x00  ptr  idxarr          per-FRAME context selector (bank preamble)
/// +0x08  u32  nb stage 1 index width
/// +0x0c  u32  nb stage 2 index width
/// +0x10  u32  nb stage 3 index width
/// +0x1c  u32  sb stage 1 index width
/// +0x20  u32  sb stage 2 index width
/// +0x30  u32  nb split_lo count
/// +0x34  u32  nb split_hi count
/// +0x44  u32  sb split 2 count
/// +0x54  ptr  nb LSP codebook 1      +0x58  nb 2      +0x5c  nb 3
/// +0x68  ptr  sb LSP codebook 1      +0x6c  sb 2
/// +0x84  u8   nb innovation shift    +0x85  u8   sb innovation shift
/// ```
///
/// Every one of these comes straight out of the `.bin` preamble — see
/// `loqng_data::bank::BankTables`. The two shift bytes are
/// `log2(norm)`, so the codebook scaling is `SIG_SHIFT - byte`, i.e. 7 and 6
/// for Dave.
pub const VOICE_PARAMS_LEN: u32 = 0x88;

/// Size of one coded frame, in bytes, for every Dave bank.
///
/// Not a constant of the format — it follows from the bitrate — but every bank
/// shipped with this voice uses it, and the decoder reads `state[0x11d2]`.
pub const DAVE_FRAME_BYTES: u16 = 48;

/// Samples per frame: 20 ms at 16 kHz, and Speex's wideband frame.
pub const DAVE_FRAME_SAMPLES: u16 = 320;

/// The error `Decode` returns for a null or invalid argument.
pub const ERR_BAD_ARG: u32 = 0xe006_0006;

/// Module ABI request ids, as `loqmsx(id, 0)` takes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    Version = 1,
    Open = 2,
    Close = 3,
    Decode = 4,
    GetCaps = 5,
    Name = 6,
}

impl Request {
    /// Module offset of the entry point this request returns.
    pub fn entry(self) -> u32 {
        match self {
            Request::Version => 0x4880,
            Request::Open => 0x4990,
            Request::Close => 0x4bd8,
            Request::Decode => 0x48d0,
            Request::GetCaps => 0x48ac,
            Request::Name => 0x485c,
        }
    }

    /// The dispatch table's own order, which is not key order.
    pub const TABLE_ORDER: [Request; 6] = [
        Request::Version,
        Request::Open,
        Request::Close,
        Request::GetCaps,
        Request::Name,
        Request::Decode,
    ];
}

/// Undo the per-frame XOR scrambling, in place.
///
/// A zero key means the bank is not scrambled and the buffer is untouched,
/// which is what the original checks before entering its loop.
pub fn descramble(frame: &mut [u8], key: u8) {
    if key == 0 {
        return;
    }
    for b in frame.iter_mut() {
        *b ^= key;
    }
}

/// Byte offset of a coded frame within the bank.
pub fn frame_offset(index: u32, frame_bytes: u32, data_offset: u32, bias: u32) -> u64 {
    frame_bytes as u64 * index as u64 + data_offset as u64 + bias as u64
}

/// What `frame` has to do to serve `target`, given what is already decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Already decoded as the current frame.
    Current,
    /// Already decoded as the previous frame.
    Previous,
    /// One frame forward: rotate and decode a single frame.
    Advance,
    /// Anywhere else: re-init and decode from `from` through `target`.
    Reseek { from: u32 },
}

/// Classify a seek exactly as `frame` does.
///
/// The reseek floor is saturating: the original starts at 0 when the preroll
/// would take it below zero, rather than wrapping.
pub fn plan(target: u32, current: u32, previous: u32, preroll: u32) -> Step {
    if target == current {
        Step::Current
    } else if target == current.wrapping_add(1) {
        Step::Advance
    } else if target == previous {
        Step::Previous
    } else {
        Step::Reseek {
            from: target.saturating_sub(preroll),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_keys_are_not_in_table_order() {
        let keys: Vec<u32> = Request::TABLE_ORDER.iter().map(|r| *r as u32).collect();
        assert_eq!(keys, vec![1, 2, 3, 5, 6, 4]);
        // Decode is last in the table despite being key 4.
        assert_eq!(Request::TABLE_ORDER[5], Request::Decode);
        assert_eq!(Request::Decode.entry(), 0x48d0);
    }

    #[test]
    fn descramble_is_a_no_op_for_a_zero_key() {
        let mut f = [1u8, 2, 3];
        descramble(&mut f, 0);
        assert_eq!(f, [1, 2, 3]);
    }

    #[test]
    fn descramble_round_trips() {
        let mut f = [0x00u8, 0x5a, 0xff];
        descramble(&mut f, 0x5a);
        assert_eq!(f, [0x5a, 0x00, 0xa5]);
        descramble(&mut f, 0x5a);
        assert_eq!(f, [0x00, 0x5a, 0xff]);
    }

    #[test]
    fn frame_offsets_match_the_dave_bank() {
        // Dave: 48 byte frames, payload starts at 657106, no bias.
        assert_eq!(frame_offset(0, 48, 657_106, 0), 657_106);
        assert_eq!(frame_offset(1, 48, 657_106, 0), 657_154);
        // The last frame must end exactly at the file length.
        let last = frame_offset(426_581, 48, 657_106, 0);
        assert_eq!(last + 48, 21_133_042);
    }

    #[test]
    fn seek_planning_matches_the_four_cases() {
        assert_eq!(plan(10, 10, 9, 4), Step::Current);
        assert_eq!(plan(11, 10, 9, 4), Step::Advance);
        assert_eq!(plan(9, 10, 9, 4), Step::Previous);
        assert_eq!(plan(100, 10, 9, 4), Step::Reseek { from: 96 });
    }

    #[test]
    fn a_reseek_near_zero_does_not_wrap() {
        assert_eq!(plan(2, 50, 49, 8), Step::Reseek { from: 0 });
    }

    #[test]
    fn current_wins_over_previous_when_they_coincide() {
        // After a reseek both can hold the same index; the original tests
        // current first, so that branch has to win.
        assert_eq!(plan(7, 7, 7, 4), Step::Current);
    }
}
