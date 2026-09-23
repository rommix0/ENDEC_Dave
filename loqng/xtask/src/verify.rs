//! `xtask verify` — prove a Rust port bit-exact against the ARM original.
//!
//! Runs the guest function and the Rust function over the same inputs and
//! compares. This is the gate every ported routine has to pass; a unit test in
//! `loqng-codec` only pins what I *believed*, which is worth much less.
//!
//! No patching and no change to `loqrs`: `Host::call`, `Host::heap` and
//! `Machine` are already public, so the guest routine can simply be called.
//! `loqmsx.so` is only in memory once a session is open, which is why this
//! pays the voice-bank load first.
//!
//! # Leaves, and what comes after them
//!
//! The first four routines here are pure leaves: one call, scalars and buffers
//! in, a result out. The next two — `split_cb_shape_sign_unquant` and
//! `pitch_unquant_3tap` — are not, because they read a `SpeexBits` and, in the
//! innovation case, a decoder state. They still need no *decoder*: the fields
//! each one actually touches can be built by hand, which is what
//! [`GuestBits`] and `verify_split_cb_shape_sign_unquant` do.
//!
//! Every argument the disassembly says is **unread** is deliberately passed
//! junk. If any of them were live, the comparison would diverge, so the claim
//! is tested rather than asserted.

use loqng_codec::bits::{field, Bits, SPEEX_BITS_LEN};
use loqng_codec::innov::{self, SplitCbParams, VoiceCodebook};
use loqng_codec::lsp::{self, LspStage, BASE_NB, BASE_SB, SPACING_ADDR_NB, SPACING_ADDR_SB};
use loqng_codec::ltp::{self, LtpParams};
use loqng_codec::math;
use loqng_codec::{filters, lpc};
use loqng_oracle::{Oracle, OracleConfig};

use crate::Paths;

const MODULE: &str = "loqmsx.so";

/// `spx_sqrt`, module offset.
const SPX_SQRT: u32 = 0x7fb8;
/// `lsp_enforce_margin`, module offset.
const LSP_ENFORCE_MARGIN: u32 = 0x69bc;
/// `compute_rms16`, module offset.
const COMPUTE_RMS16: u32 = 0x1d90;
/// `lsp_interpolate`, module offset.
const LSP_INTERPOLATE: u32 = 0x6a70;
/// `split_cb_shape_sign_unquant`, module offset.
const SPLIT_CB_UNQUANT: u32 = 0x153c;
/// `pitch_unquant_3tap`, module offset.
const PITCH_UNQUANT_3TAP: u32 = 0x7414;
/// `signal_mul`, module offset.
const SIGNAL_MUL: u32 = 0x3c98;
/// `lsp_to_lpc`, module offset.
const LSP_TO_LPC: u32 = 0x6500;
/// `iir_mem` over 16-bit samples, module offset.
const IIR16: u32 = 0x3cd0;
/// `iir_mem` over 32-bit samples, module offset.
const IIR32: u32 = 0x22c4;
/// `fir2x` polyphase QMF synthesis, module offset.
const FIR2X: u32 = 0x2c44;

/// The four `lsp_unquant` variants, reached only through the submode struct.
///
/// `stages` is literally the number of `bl 0x14ac` calls in each — one
/// `unpack` per codebook stage, which is what tells them apart. Dave forces NB
/// submode 5 and SB submode 2, so `0xbf64` and `0xc0e0` are the live pair.
const LSP_VARIANTS: [LspVariant; 4] = [
    LspVariant {
        name: "lsp_unquant nb x1",
        entry: 0xbda0,
        stages: 1,
        band: NB,
    },
    LspVariant {
        name: "lsp_unquant nb x2",
        entry: 0xbe50,
        stages: 2,
        band: NB,
    },
    LspVariant {
        name: "lsp_unquant nb x3",
        entry: 0xbf64,
        stages: 3,
        band: NB,
    },
    LspVariant {
        name: "lsp_unquant sb x2",
        entry: 0xc0e0,
        stages: 2,
        band: SB,
    },
];

/// Which decoder-state fields a band's variant reads. Index 0 of `counts` is
/// unused: stage 0 always covers `order` coefficients.
struct LspBand {
    spacing: u32,
    base: i32,
    nbits: [u32; 3],
    counts: [u32; 3],
    cbs: [u32; 3],
}

const NB: LspBand = LspBand {
    spacing: SPACING_ADDR_NB,
    base: BASE_NB,
    nbits: [0x08, 0x0c, 0x10],
    counts: [0, 0x30, 0x34],
    cbs: [0x54, 0x58, 0x5c],
};

const SB: LspBand = LspBand {
    spacing: SPACING_ADDR_SB,
    base: BASE_SB,
    nbits: [0x1c, 0x20, 0],
    counts: [0, 0x44, 0],
    cbs: [0x68, 0x6c, 0],
};

struct LspVariant {
    name: &'static str,
    entry: u32,
    stages: usize,
    band: LspBand,
}

/// Deterministic LCG, so a failure is reproducible and the workspace stays
/// dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// A value in `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n.max(1)
    }

    /// One of `choices`.
    fn pick<T: Copy>(&mut self, choices: &[T]) -> T {
        choices[self.below(choices.len() as u32) as usize]
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u32() as u8).collect()
    }
}

struct Report {
    name: String,
    checked: usize,
    failures: Vec<String>,
}

impl Report {
    fn new(name: &str) -> Self {
        Report {
            name: name.to_string(),
            checked: 0,
            failures: Vec::new(),
        }
    }

    fn fail(&mut self, msg: String) {
        // Enough to diagnose, not enough to flood the terminal.
        if self.failures.len() < 8 {
            self.failures.push(msg);
        }
    }

    fn print(&self) -> bool {
        if self.failures.is_empty() {
            println!("  {:<28} {:>7} cases   BIT-EXACT", self.name, self.checked);
            return true;
        }
        println!(
            "  {:<28} {:>7} cases   {} MISMATCHES",
            self.name,
            self.checked,
            self.failures.len()
        );
        for f in &self.failures {
            println!("      {f}");
        }
        false
    }
}

pub fn run(p: &Paths) -> Result<(), String> {
    println!("opening the ARM engine (loqmsx is only mapped once a voice is)");
    // Interpreted, not natively patched: six of the routines below are ones
    // `loqrs` replaces with its own Rust, and comparing Rust to Rust would
    // prove nothing about the ARM.
    let cfg = OracleConfig::new(&p.lib_dir, &p.data_dir).interpreted();
    let started = std::time::Instant::now();
    let mut o = Oracle::open(&cfg)?;
    println!("ready in {:.2} s", started.elapsed().as_secs_f64());

    let base = o
        .module_base(MODULE)
        .ok_or_else(|| format!("{MODULE} is not loaded"))?;
    println!("{MODULE} at 0x{base:08x}");
    println!();

    let mut ok = true;
    ok &= verify_spx_sqrt(&mut o)?.print();
    ok &= verify_lsp_enforce_margin(&mut o)?.print();
    ok &= verify_compute_rms16(&mut o)?.print();
    ok &= verify_lsp_interpolate(&mut o)?.print();
    ok &= verify_split_cb_shape_sign_unquant(&mut o)?.print();
    ok &= verify_pitch_unquant_3tap(&mut o)?.print();
    for v in &LSP_VARIANTS {
        ok &= verify_lsp_unquant(&mut o, base, v)?.print();
    }
    ok &= verify_signal_mul(&mut o)?.print();
    ok &= verify_lsp_to_lpc(&mut o)?.print();
    ok &= verify_iir(&mut o, false)?.print();
    ok &= verify_iir(&mut o, true)?.print();
    ok &= verify_fir2x(&mut o)?.print();

    println!();
    if ok {
        println!("all ports are bit-exact against the ARM original");
        Ok(())
    } else {
        Err("at least one port differs from the ARM original".to_string())
    }
}

// ---- a guest SpeexBits, built by hand -----------------------------------

/// A `SpeexBits` struct and its payload buffer, allocated once and rewound
/// between trials.
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

    /// Rewind to the start of a fresh payload. `nbBits` covers the whole
    /// buffer, which is the invariant the real decoder maintains.
    fn load(&self, o: &mut Oracle, payload: &[u8]) -> Result<(), String> {
        if payload.len() > self.cap {
            return Err(format!(
                "payload of {} exceeds the {} byte buffer",
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

    /// Bits consumed, and whether the reader overflowed.
    fn state(&self, o: &Oracle) -> Result<(i32, bool), String> {
        let char_ptr = o.read_u32(self.bits + field::CHAR_PTR)? as i32;
        let bit_ptr = o.read_u32(self.bits + field::BIT_PTR)? as i32;
        let overflow = o.read_u32(self.bits + field::OVERFLOW)?;
        Ok((
            char_ptr.wrapping_shl(3).wrapping_add(bit_ptr),
            overflow != 0,
        ))
    }
}

fn as_i8(v: &[u8]) -> Vec<i8> {
    v.iter().map(|b| *b as i8).collect()
}

// ---- the pure leaves -----------------------------------------------------

fn verify_spx_sqrt(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("spx_sqrt");
    let mut rng = Rng(0x5eed_1234);

    // Every boundary the normalisation ladder tests, plus their neighbours.
    let mut cases: Vec<i32> = vec![
        i32::MIN,
        -1,
        0,
        1,
        2,
        3,
        0xffe,
        0xfff,
        0x1000,
        0x3ffe,
        0x3fff,
        0x4000,
        0x7fff,
        0x8000,
        0x3fffe,
        0x3ffff,
        0x40000,
        0xffffe,
        0xfffff,
        0x100000,
        0xfffffe,
        0xffffff,
        0x1000000,
        i32::MAX,
        i32::MAX - 1,
    ];
    for _ in 0..4000 {
        cases.push((rng.next_u32() >> (rng.next_u32() % 31)) as i32);
    }

    for x in cases {
        let guest = o.call_at(MODULE, SPX_SQRT, &[x as u32])? as i32;
        let rust = math::spx_sqrt(x);
        r.checked += 1;
        if guest != rust {
            r.fail(format!("spx_sqrt({x}): guest {guest}, rust {rust}"));
        }
    }
    Ok(r)
}

fn verify_compute_rms16(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("compute_rms16");
    let mut rng = Rng(0xc0ff_ee01);

    // The loops step by four with no remainder pass, so only multiples of 4
    // are meaningful inputs.
    const MAX: usize = 64;
    let buf = o.alloc((MAX * 2) as u32)?;

    for trial in 0..2000usize {
        let len = 4 * (1 + (rng.next_u32() as usize % (MAX / 4)));
        // Span both branches: quiet blocks take the shift-up path, loud ones
        // the halving path, and the peak floor of 10 needs silence too.
        let amp: i32 = match trial % 4 {
            0 => 0,
            1 => 0x400,
            2 => 0x3000,
            _ => 0x7fff,
        };
        let x: Vec<i16> = (0..len)
            .map(|_| {
                if amp == 0 {
                    0
                } else {
                    ((rng.next_u32() as i32 % (2 * amp + 1)) - amp) as i16
                }
            })
            .collect();

        o.write_i16s(buf, &x)?;
        let guest = o.call_at(MODULE, COMPUTE_RMS16, &[buf, len as u32])? as i32;
        let rust = math::compute_rms16(&x, len);
        r.checked += 1;
        if guest != rust {
            r.fail(format!("len {len} amp {amp}: guest {guest}, rust {rust}"));
        }
    }
    Ok(r)
}

fn verify_lsp_interpolate(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("lsp_interpolate");
    let mut rng = Rng(0x1a7e_4b01);

    const MAX: usize = 20;
    let old_p = o.alloc((MAX * 2) as u32)?;
    let new_p = o.alloc((MAX * 2) as u32)?;
    let out_p = o.alloc((MAX * 2) as u32)?;

    for _ in 0..2000usize {
        let len = 1 + (rng.next_u32() as usize % MAX);
        let nb = 1 + (rng.next_u32() as i32 % 8);
        let sub = rng.next_u32() as i32 % nb;
        let old: Vec<i16> = (0..len).map(|_| rng.next_u32() as i16).collect();
        let new: Vec<i16> = (0..len).map(|_| rng.next_u32() as i16).collect();

        o.write_i16s(old_p, &old)?;
        o.write_i16s(new_p, &new)?;
        o.write_i16s(out_p, &vec![0i16; len])?;
        o.call_at(
            MODULE,
            LSP_INTERPOLATE,
            &[old_p, new_p, out_p, len as u32, sub as u32, nb as u32],
        )?;
        let guest = o.read_i16s(out_p, len)?;

        let mut rust = vec![0i16; len];
        lsp::interpolate(&old, &new, &mut rust, len, sub, nb);
        r.checked += 1;
        if guest != rust {
            r.fail(format!(
                "len {len} sub {sub}/{nb}: guest {:?} rust {:?}",
                &guest[..len.min(6)],
                &rust[..len.min(6)]
            ));
        }
    }
    Ok(r)
}

fn verify_lsp_enforce_margin(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("lsp_enforce_margin");
    let mut rng = Rng(0xfeed_9876);

    // One buffer, reused; the guest writes in place.
    const MAX: usize = 20;
    let buf = o.alloc((MAX * 2) as u32)?;

    // Both real margins, plus edges.
    let margins: [i16; 5] = [0, 1, 16, 410, 2000];

    for trial in 0..3000usize {
        let len = 1 + (rng.next_u32() as usize % MAX);
        let margin = margins[rng.next_u32() as usize % margins.len()];

        // A mix of sorted-and-spread, crowded, and outright hostile inputs, so
        // both clamps and the wrapping paths get exercised.
        let mut lsp: Vec<i16> = match trial % 3 {
            0 => {
                let mut v = 0i32;
                (0..len)
                    .map(|_| {
                        v += (rng.next_u32() % 3000) as i32;
                        v.min(32767) as i16
                    })
                    .collect()
            }
            1 => (0..len).map(|i| (i as i16).wrapping_mul(3)).collect(),
            _ => (0..len).map(|_| rng.next_u32() as i16).collect(),
        };

        o.write_i16s(buf, &lsp)?;
        o.call_at(
            MODULE,
            LSP_ENFORCE_MARGIN,
            &[buf, len as u32, margin as i32 as u32],
        )?;
        let guest = o.read_i16s(buf, len)?;

        lsp::enforce_margin(&mut lsp, margin);
        r.checked += 1;
        if guest != lsp {
            r.fail(format!(
                "len {len} margin {margin}: guest {:?} rust {:?}",
                &guest[..len.min(8)],
                &lsp[..len.min(8)]
            ));
        }
    }
    Ok(r)
}

// ---- the two bitstream readers ------------------------------------------

/// `split_cb_shape_sign_unquant` at `0x153c`.
///
/// Eight arguments, of which one is a decoder state this builds by hand: the
/// routine reads only `+0x00`, `+0x38`/`+0x3c`, `+0x60`/`+0x64` and the two
/// shift bytes at `+0x84`/`+0x85`, so a real decoder is not needed.
///
/// ```text
/// r0 state, r1 exc, r2 par, r3 band,
/// sp+0x2c (unread), sp+0x30 bits, sp+0x34 stack, sp+0x38 cdbk_offset
/// ```
fn verify_split_cb_shape_sign_unquant(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("split_cb_shape_sign_unquant");
    let mut rng = Rng(0x5c5b_1001);

    const SUBVECT: [usize; 4] = [5, 8, 10, 20];
    const NB_SUBVECT: [usize; 4] = [2, 4, 5, 8];
    const SHAPE_BITS: [u32; 3] = [4, 5, 6];
    const MAX_SELECTORS: usize = 4;
    // subvect_size * (1 << shape_bits) * MAX_SELECTORS, at the maxima.
    const CB_LEN: usize = 20 * 64 * MAX_SELECTORS;
    const EXC_WORDS: usize = 8 * 20;
    const SELECTORS: usize = 64;

    let state = o.alloc(0x100)?;
    let sel_arr = o.alloc(SELECTORS as u32)?;
    let cb_low = o.alloc(CB_LEN as u32)?;
    let cb_high = o.alloc(CB_LEN as u32)?;
    let par = o.alloc(0x14)?;
    let exc = o.alloc((EXC_WORDS * 4) as u32)?;
    let scratch = o.alloc(0x200)?;
    let gb = GuestBits::new(o, 64)?;

    o.write_u32(state + 0x00, sel_arr)?;
    o.write_u32(state + 0x60, cb_low)?;
    o.write_u32(state + 0x64, cb_high)?;

    for _ in 0..1500usize {
        let subvect_size = rng.pick(&SUBVECT);
        let nb_subvect = rng.pick(&NB_SUBVECT);
        let shape_bits = rng.pick(&SHAPE_BITS);
        let have_sign = rng.next_u32() & 1 == 1;
        let high = rng.next_u32() & 1 == 1;
        // Bytes per entry is its own state field; for a real bank it equals
        // subvect_size, and a value that does not is what would expose
        // confusing the two.
        let entry_bytes = subvect_size;
        let shift = rng.below(15) as u8;
        let cdbk_offset = rng.below(SELECTORS as u32);

        let selectors = rng.bytes(SELECTORS);
        let selectors: Vec<u8> = selectors
            .iter()
            .map(|b| (*b as usize % MAX_SELECTORS) as u8)
            .collect();
        let cb = rng.bytes(CB_LEN);
        let payload = rng.bytes(16);

        o.write_bytes(sel_arr, &selectors)?;
        o.write_bytes(if high { cb_high } else { cb_low }, &cb)?;
        o.write_u32(state + if high { 0x3c } else { 0x38 }, entry_bytes as u32)?;
        o.write_u8(state + if high { 0x85 } else { 0x84 }, shift)?;
        o.write_u32s(
            par,
            &[
                subvect_size as u32,
                nb_subvect as u32,
                0xdead_beef, // shape_cb: the customisation is that this is unread
                shape_bits,
                have_sign as u32,
            ],
        )?;
        o.write_i32s(exc, &vec![0i32; EXC_WORDS])?;
        gb.load(o, &payload)?;

        o.call_at(
            MODULE,
            SPLIT_CB_UNQUANT,
            &[
                state,
                exc,
                par,
                high as u32,
                0x1234_5678, // sp+0x2c, claimed unread
                gb.bits,
                scratch,
                cdbk_offset,
            ],
        )?;

        let n = nb_subvect * subvect_size;
        let guest = o.read_i32s(exc, n)?;
        let (guest_pos, guest_overflow) = gb.state(o)?;

        let p = SplitCbParams {
            subvect_size,
            nb_subvect,
            shape_bits,
            have_sign,
        };
        let cbk = VoiceCodebook {
            cb: &as_i8(&cb),
            entry_bytes,
            shift,
        };
        let mut b = Bits::new(&payload);
        let mut rust = vec![0i32; EXC_WORDS];
        innov::unquant(&cbk, selectors[cdbk_offset as usize], &p, &mut b, &mut rust);
        let rust = &rust[..n];

        r.checked += 1;
        if guest != rust {
            r.fail(format!(
                "sv {subvect_size} nb {nb_subvect} bits {shape_bits} sign {have_sign} \
                 band {} shift {shift}: guest {:?} rust {:?}",
                high as u32,
                &guest[..n.min(6)],
                &rust[..n.min(6)]
            ));
        } else if (guest_pos, guest_overflow) != (b.position(), b.overflow != 0) {
            r.fail(format!(
                "sv {subvect_size} nb {nb_subvect} bits {shape_bits}: cursor guest \
                 {guest_pos}/{guest_overflow} rust {}/{}",
                b.position(),
                b.overflow != 0
            ));
        }
    }
    Ok(r)
}

/// `pitch_unquant_3tap` at `0x7414`.
///
/// Fifteen arguments. Nine of them the disassembly says are never read, and
/// every one of those is passed junk here so that the claim is tested.
///
/// The history buffer is allocated with `HIST` samples in front of the pointer
/// the routine is given, because it indexes `exc2` negatively — up to
/// `2 * (start + pitch) + 1` back.
fn verify_pitch_unquant_3tap(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("pitch_unquant_3tap");
    let mut rng = Rng(0x71c4_2002);

    const HIST: usize = 512;
    const MAX_NSF: usize = 40;
    const MAX_START: u32 = 20;
    const MAX_CDBK_OFFSET: u32 = 4;
    // (cdbk_offset << gain_bits) * 4 at the maxima, plus one whole codebook.
    const CDBK_LEN: usize = (MAX_CDBK_OFFSET as usize + 1) * 128 * 4;

    let hist_buf = o.alloc(((HIST + MAX_NSF) * 2) as u32)?;
    let exc2 = hist_buf + (HIST * 2) as u32;
    let exc = o.alloc((MAX_NSF * 4) as u32)?;
    let par = o.alloc(0x0c)?;
    let cdbk = o.alloc(CDBK_LEN as u32)?;
    let pitch_val = o.alloc(4)?;
    let gain_val = o.alloc(8)?;
    let scratch = o.alloc(0x100)?;
    let gb = GuestBits::new(o, 64)?;

    o.write_u32(par + 0x00, cdbk)?;

    for _ in 0..1500usize {
        let gain_bits = rng.pick(&[5u32, 6, 7]);
        let pitch_bits = rng.pick(&[5u32, 6, 7]);
        let nsf = 1 + rng.below(MAX_NSF as u32) as usize;
        let start = rng.below(MAX_START) as i32;
        let cdbk_offset = rng.below(MAX_CDBK_OFFSET);

        let hist: Vec<i16> = (0..HIST).map(|_| rng.next_u32() as i16).collect();
        let cdbk_bytes = rng.bytes(CDBK_LEN);
        let payload = rng.bytes(8);

        o.write_i16s(hist_buf, &hist)?;
        // Poison the current subframe: the predictor must never read forward.
        o.write_i16s(exc2, &vec![0x7ffe_i16; MAX_NSF])?;
        o.write_bytes(cdbk, &cdbk_bytes)?;
        o.write_u32s(par, &[cdbk, gain_bits, pitch_bits])?;
        o.write_i32s(exc, &vec![0x5a5a_5a5a_u32 as i32; MAX_NSF])?;
        o.write_i16s(gain_val, &[0i16; 4])?;
        o.write_u32(pitch_val, 0)?;
        gb.load(o, &payload)?;

        o.call_at(
            MODULE,
            PITCH_UNQUANT_3TAP,
            &[
                exc2,
                exc,
                start as u32,
                0xdead_0003, // end
                0xdead_0004, // pitch_coef
                par,
                nsf as u32,
                pitch_val,
                gain_val,
                gb.bits,
                scratch,
                0xdead_000b, // count_lost
                0xdead_000c, // subframe_offset
                0xdead_000d, // last_pitch_gain
                cdbk_offset,
            ],
        )?;

        let guest_exc = o.read_i32s(exc, nsf)?;
        let guest_pitch = o.read_u32(pitch_val)? as i32;
        let guest_gains = o.read_i16s(gain_val, 3)?;
        let (guest_pos, guest_overflow) = gb.state(o)?;

        let p = LtpParams {
            gain_bits,
            pitch_bits,
        };
        let mut b = Bits::new(&payload);
        let mut rust_exc = vec![0i32; nsf];
        let (rust_pitch, rust_gains) = ltp::unquant(
            &hist,
            &mut rust_exc,
            start,
            &p,
            &as_i8(&cdbk_bytes),
            nsf,
            &mut b,
            cdbk_offset,
        );

        r.checked += 1;
        let label =
            format!("nsf {nsf} start {start} gb {gain_bits} pb {pitch_bits} off {cdbk_offset}");
        if guest_pitch != rust_pitch {
            r.fail(format!(
                "{label}: pitch_val guest {guest_pitch} rust {rust_pitch}"
            ));
        } else if guest_gains != rust_gains {
            r.fail(format!(
                "{label}: gain_val guest {guest_gains:?} rust {rust_gains:?}"
            ));
        } else if guest_exc != rust_exc {
            let at = guest_exc
                .iter()
                .zip(rust_exc.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "{label}: exc[{at}] guest {} rust {}",
                guest_exc[at], rust_exc[at]
            ));
        } else if (guest_pos, guest_overflow) != (b.position(), b.overflow != 0) {
            r.fail(format!(
                "{label}: cursor guest {guest_pos}/{guest_overflow} rust {}/{}",
                b.position(),
                b.overflow != 0
            ));
        }
    }
    Ok(r)
}

/// One `lsp_unquant` variant — the third Loquendo customisation.
///
/// ```text
/// r0 lsp, r1 order, r2 bits, r3 state, sp+N cdbk_offset
/// ```
///
/// All four variants run the same algorithm; they differ only in how many
/// codebook stages they apply and which state fields carry the per-stage
/// index width, coefficient count and codebook base. Stage 0 always covers
/// `order` coefficients and shifts by 5; later stages refine a split and shift
/// by 4, with the narrowband third stage starting at `split_lo`.
///
/// The linear base spacing is a `u16` in `.bss` and reads 0 on a cold module,
/// so it is **driven** through a range rather than observed — a sweep stuck at
/// 0 would never exercise the multiply. The original value is restored after.
fn verify_lsp_unquant(o: &mut Oracle, base: u32, v: &LspVariant) -> Result<Report, String> {
    let mut r = Report::new(v.name);
    let mut rng = Rng(0x15b0_3003 ^ (v.entry as u64));

    const MAX_ORDER: usize = 16;
    const MAX_SELECTORS: usize = 4;
    const NBITS: [u32; 4] = [3, 4, 5, 6];
    // (count * sel) << nbits, plus one whole codebook, at the maxima.
    const CB_LEN: usize = 8192;
    const SELECTORS: usize = 64;

    let live = o.read_i16s(base + v.band.spacing, 1)?[0] as u16;
    let spacings: [u16; 6] = [live, 0, 1, 2048, 4096, 0xffff];

    let state = o.alloc(0x80)?;
    let sel_arr = o.alloc(SELECTORS as u32)?;
    let cbs: Vec<u32> = (0..3)
        .map(|_| o.alloc(CB_LEN as u32))
        .collect::<Result<_, _>>()?;
    let lsp_p = o.alloc((MAX_ORDER * 2) as u32)?;
    let gb = GuestBits::new(o, 64)?;

    o.write_u32(state + 0x00, sel_arr)?;
    for s in 0..v.stages {
        o.write_u32(state + v.band.cbs[s], cbs[s])?;
    }

    for _ in 0..1200usize {
        let order = 1 + rng.below(MAX_ORDER as u32) as usize;
        let spacing = rng.pick(&spacings);
        let cdbk_offset = rng.below(SELECTORS as u32);

        // Stage windows must stay inside `order`: the third narrowband stage
        // writes lsp[split_lo .. split_lo + split_hi].
        let mut counts = [order, 0, 0];
        let mut starts = [0usize, 0, 0];
        if v.stages > 1 {
            counts[1] = 1 + rng.below(order as u32) as usize;
            starts[1] = 0;
        }
        if v.stages > 2 {
            counts[2] = 1 + rng.below((order - counts[1] + 1) as u32) as usize;
            starts[2] = counts[1];
        }

        let nbits: Vec<u32> = (0..v.stages).map(|_| rng.pick(&NBITS)).collect();
        let selectors: Vec<u8> = rng
            .bytes(SELECTORS)
            .iter()
            .map(|b| (*b as usize % MAX_SELECTORS) as u8)
            .collect();
        let cb_bytes: Vec<Vec<u8>> = (0..v.stages).map(|_| rng.bytes(CB_LEN)).collect();
        let payload = rng.bytes(16);

        o.write_bytes(sel_arr, &selectors)?;
        o.write_i16s(base + v.band.spacing, &[spacing as i16])?;
        for s in 0..v.stages {
            o.write_bytes(cbs[s], &cb_bytes[s])?;
            o.write_u32(state + v.band.nbits[s], nbits[s])?;
            if s > 0 {
                o.write_u32(state + v.band.counts[s], counts[s] as u32)?;
            }
        }
        // Poison the output: the routine must write every element itself.
        o.write_i16s(lsp_p, &vec![0x5a5a_u16 as i16; MAX_ORDER])?;
        gb.load(o, &payload)?;

        o.call_at(
            MODULE,
            v.entry,
            &[lsp_p, order as u32, gb.bits, state, cdbk_offset],
        )?;

        let guest = o.read_i16s(lsp_p, order)?;
        let (guest_pos, guest_overflow) = gb.state(o)?;

        let signed: Vec<Vec<i8>> = cb_bytes.iter().map(|c| as_i8(c)).collect();
        let stages: Vec<LspStage> = (0..v.stages)
            .map(|s| {
                if s == 0 {
                    LspStage::first(&signed[0], nbits[0], order)
                } else {
                    LspStage::split(&signed[s], nbits[s], counts[s], starts[s])
                }
            })
            .collect();

        let mut b = Bits::new(&payload);
        let mut rust = vec![0x5a5a_u16 as i16; MAX_ORDER];
        lsp::unquant(
            &mut rust,
            order,
            spacing,
            v.band.base,
            &stages,
            selectors[cdbk_offset as usize],
            &mut b,
        );
        let rust = &rust[..order];

        r.checked += 1;
        if guest != rust {
            let at = guest
                .iter()
                .zip(rust.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "order {order} nbits {nbits:?} counts {:?} spacing {spacing}: \
                 lsp[{at}] guest {} rust {}",
                &counts[..v.stages],
                guest[at],
                rust[at]
            ));
        } else if (guest_pos, guest_overflow) != (b.position(), b.overflow != 0) {
            r.fail(format!(
                "order {order} nbits {nbits:?}: cursor guest {guest_pos}/\
                 {guest_overflow} rust {}/{}",
                b.position(),
                b.overflow != 0
            ));
        }
    }

    o.write_i16s(base + v.band.spacing, &[live as i16])?;
    Ok(r)
}

// ---- the four signal-path filters and lsp_to_lpc ------------------------
//
// All five already ran as bit-exact native patches inside `loqrs`, validated
// by whole-utterance audio diff. That is a strong gate but a blunt one: it
// says an hour of speech matched, not which input would break. These call the
// guest routine directly, so a failure names the case.

/// `signal_mul` at `0x3c98`. `(x, y, scale, len)`.
fn verify_signal_mul(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("signal_mul");
    let mut rng = Rng(0x519a_1101);

    const MAX: usize = 160;
    let x = o.alloc((MAX * 4) as u32)?;
    let y = o.alloc((MAX * 4) as u32)?;

    for trial in 0..2000usize {
        let len = 1 + rng.below(MAX as u32) as usize;
        // Span the range where the ARM4 macro and the generic one diverge:
        // inputs far wider than 16 bits, and gains at both Q14 extremes.
        let scale: i32 = match trial % 4 {
            0 => 0x4000,
            1 => -0x4000,
            2 => rng.next_u32() as i16 as i32,
            _ => rng.next_u32() as i32,
        };
        let xs: Vec<i32> = (0..len)
            .map(|_| match trial % 3 {
                0 => rng.next_u32() as i32,
                1 => (rng.next_u32() as i32) >> 12,
                _ => (rng.next_u32() as i32) & 0x7f,
            })
            .collect();

        o.write_i32s(x, &xs)?;
        o.write_i32s(y, &vec![0x5a5a_5a5a_u32 as i32; len])?;
        o.call_at(MODULE, SIGNAL_MUL, &[x, y, scale as u32, len as u32])?;
        let guest = o.read_i32s(y, len)?;

        let mut rust = vec![0x5a5a_5a5a_u32 as i32; len];
        filters::signal_mul(&xs, &mut rust, scale, len as i32);

        r.checked += 1;
        if guest != rust {
            let at = guest
                .iter()
                .zip(rust.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "len {len} scale {scale:#x}: y[{at}] x {:#x} guest {} rust {}",
                xs[at], guest[at], rust[at]
            ));
        }
    }
    Ok(r)
}

/// `lsp_to_lpc` at `0x6500`. `(freq, ak, lpcrdr, stack)`.
///
/// `lpcrdr` must be even — the cascade runs `lpcrdr >> 1` sections. The guest
/// carves `xp`, `xq` and `freqn` out of `stack`, so that has to be real
/// memory even though nothing left in it is observable.
fn verify_lsp_to_lpc(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("lsp_to_lpc");
    let mut rng = Rng(0x15c0_2202);

    const MAX_ORDER: usize = 20;
    const ORDERS: [usize; 5] = [4, 8, 10, 16, 20];
    let freq = o.alloc((MAX_ORDER * 2) as u32)?;
    let ak = o.alloc((MAX_ORDER * 2) as u32)?;
    let stack = o.alloc(0x4000)?;

    for trial in 0..2000usize {
        let order = rng.pick(&ORDERS);
        // Realistic LSPs are sorted inside 0..LSP_MAX; hostile ones exercise
        // the wrap in the reflected branch of spx_cos and the clamp.
        let lsp: Vec<i16> = match trial % 3 {
            0 => {
                let mut v = 0i32;
                (0..order)
                    .map(|_| {
                        v += (rng.next_u32() % (25736 / order as u32)) as i32;
                        v.min(25736) as i16
                    })
                    .collect()
            }
            1 => (0..order)
                .map(|i| ((i + 1) as i32 * 25736 / (order as i32 + 1)) as i16)
                .collect(),
            _ => (0..order).map(|_| rng.next_u32() as i16).collect(),
        };

        o.write_i16s(freq, &lsp)?;
        o.write_i16s(ak, &vec![0x5a5a_u16 as i16; MAX_ORDER])?;
        o.call_at(MODULE, LSP_TO_LPC, &[freq, ak, order as u32, stack])?;
        let guest = o.read_i16s(ak, order)?;

        let mut rust = vec![0x5a5a_u16 as i16; MAX_ORDER];
        lpc::lsp_to_lpc(&lsp, &mut rust, order as i32);
        let rust = &rust[..order];

        r.checked += 1;
        if guest != rust {
            let at = guest
                .iter()
                .zip(rust.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "order {order}: ak[{at}] guest {} rust {} (lsp {:?})",
                guest[at],
                rust[at],
                &lsp[..order.min(4)]
            ));
        }
    }
    Ok(r)
}

/// The two all-pole synthesis filters: `iir16` at `0x3cd0` and `iir32` at
/// `0x22c4`. Both take `(x, den, out, n, ord, mem)`, differing only in sample
/// width, so one harness covers them.
fn verify_iir(o: &mut Oracle, wide: bool) -> Result<Report, String> {
    let mut r = Report::new(if wide { "iir32" } else { "iir16" });
    let mut rng = Rng(if wide { 0x11f3_3201 } else { 0x11f1_6202 });
    let entry = if wide { IIR32 } else { IIR16 };

    const MAX_N: usize = 160;
    const MAX_ORD: usize = 20;
    let x = o.alloc((MAX_N * 4) as u32)?;
    let den = o.alloc((MAX_ORD * 2) as u32)?;
    let out = o.alloc((MAX_N * 4) as u32)?;
    let mem = o.alloc((MAX_ORD * 4) as u32)?;

    for trial in 0..1500usize {
        let n = 1 + rng.below(MAX_N as u32) as usize;
        let ord = 1 + rng.below(MAX_ORD as u32) as usize;
        let coefs: Vec<i16> = (0..ord).map(|_| rng.next_u32() as i16).collect();
        // A cold filter, a loaded one, and one carrying history near the top
        // of the range, where the 16-bit feedback add wraps rather than
        // saturating.
        let mem0: Vec<i32> = (0..ord)
            .map(|_| match trial % 3 {
                0 => 0,
                1 => (rng.next_u32() as i32) >> 8,
                _ => rng.next_u32() as i32,
            })
            .collect();

        o.write_i16s(den, &coefs)?;
        o.write_i32s(mem, &mem0)?;

        let mut rust_mem = mem0.clone();
        let (guest, rust) = if wide {
            let xs: Vec<i32> = (0..n).map(|_| rng.next_u32() as i32).collect();
            o.write_i32s(x, &xs)?;
            o.write_i32s(out, &vec![0x5a5a_5a5a_u32 as i32; n])?;
            o.call_at(MODULE, entry, &[x, den, out, n as u32, ord as u32, mem])?;
            let g = o.read_i32s(out, n)?;

            let mut ro = vec![0x5a5a_5a5a_u32 as i32; n];
            filters::iir32(&xs, &coefs, n as i32, ord as i32, &mut ro, &mut rust_mem);
            (g, ro)
        } else {
            let xs: Vec<i16> = (0..n).map(|_| rng.next_u32() as i16).collect();
            o.write_i16s(x, &xs)?;
            o.write_i16s(out, &vec![0x5a5a_u16 as i16; n])?;
            o.call_at(MODULE, entry, &[x, den, out, n as u32, ord as u32, mem])?;
            let g: Vec<i32> = o.read_i16s(out, n)?.iter().map(|v| *v as i32).collect();

            let mut ro = vec![0x5a5a_u16 as i16; n];
            filters::iir16(&xs, &coefs, n as i32, ord as i32, &mut ro, &mut rust_mem);
            (g, ro.iter().map(|v| *v as i32).collect())
        };

        // The history is carried between calls, so it is as much an output as
        // the samples are.
        let guest_mem = o.read_i32s(mem, ord)?;

        r.checked += 1;
        if guest != rust {
            let at = guest
                .iter()
                .zip(rust.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "n {n} ord {ord}: out[{at}] guest {} rust {}",
                guest[at], rust[at]
            ));
        } else if guest_mem != rust_mem {
            let at = guest_mem
                .iter()
                .zip(rust_mem.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "n {n} ord {ord}: history[{at}] guest {} rust {}",
                guest_mem[at], rust_mem[at]
            ));
        }
    }
    Ok(r)
}

/// `fir2x` at `0x2c44`: polyphase QMF synthesis. `(x, coef, out, n, ord, mem)`.
///
/// `n` is written four samples at a time, so only multiples of four are
/// meaningful. The history array has an **8-byte stride with the sample at
/// +4**, which is the detail most likely to be lost in a port.
fn verify_fir2x(o: &mut Oracle) -> Result<Report, String> {
    let mut r = Report::new("fir2x");
    let mut rng = Rng(0xf12c_4403);

    const MAX_N: usize = 160;
    const MAX_ORD: usize = 32;
    let x = o.alloc((MAX_N * 4) as u32)?;
    let coef = o.alloc((MAX_ORD * 2 + 8) as u32)?;
    let out = o.alloc((MAX_N * 4) as u32)?;
    let mem = o.alloc((MAX_ORD * 8) as u32)?;

    for trial in 0..1500usize {
        let n = 4 * (1 + rng.below((MAX_N / 4) as u32) as usize);
        let ord = 4 * (1 + rng.below((MAX_ORD / 4) as u32) as usize);
        let half = n / 2;
        let hist = ord / 2;

        let xs: Vec<i32> = (0..half)
            .map(|_| match trial % 3 {
                0 => rng.next_u32() as i32,
                1 => (rng.next_u32() as i32) >> 10,
                _ => 0,
            })
            .collect();
        let taps = ord.div_ceil(4) * 4;
        let cs: Vec<i16> = (0..taps).map(|_| rng.next_u32() as i16).collect();
        // Two words per history entry; only the second is read or written.
        let mem0: Vec<i32> = (0..hist * 2)
            .map(|k| {
                if k % 2 == 1 {
                    rng.next_u32() as i16 as i32
                } else {
                    0x7e7e_7e7e_u32 as i32
                }
            })
            .collect();

        o.write_i32s(x, &xs)?;
        o.write_i16s(coef, &cs)?;
        o.write_i32s(mem, &mem0)?;
        o.write_i32s(out, &vec![0x5a5a_5a5a_u32 as i32; n])?;

        o.call_at(MODULE, FIR2X, &[x, coef, out, n as u32, ord as u32, mem])?;
        let guest = o.read_i32s(out, n)?;
        let guest_mem = o.read_i32s(mem, hist * 2)?;

        let mut rust = vec![0x5a5a_5a5a_u32 as i32; n];
        let mut rust_mem = mem0.clone();
        filters::fir2x_run(&xs, &cs, &mut rust, n as i32, ord as i32, &mut rust_mem);

        r.checked += 1;
        if guest != rust {
            let at = guest
                .iter()
                .zip(rust.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "n {n} ord {ord}: out[{at}] guest {} rust {}",
                guest[at], rust[at]
            ));
        } else if guest_mem != rust_mem {
            let at = guest_mem
                .iter()
                .zip(rust_mem.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            r.fail(format!(
                "n {n} ord {ord}: history[{at}] guest {} rust {}",
                guest_mem[at], rust_mem[at]
            ));
        }
    }
    Ok(r)
}
