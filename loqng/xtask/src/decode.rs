//! `xtask decode` — run the ported frame decoder over a real voice bank.
//!
//! This is the first thing in `loqng` that produces audio. It takes the
//! `init`-free route the handoff notes proposed: `loqng_data::bank` parses the
//! whole preamble in Rust, so nothing here needs the module's own `init`
//! (`0x4c50`), the `0x88` parameter block, or `ELQBinGetBuffer`. The bank is
//! just a file.
//!
//! ```text
//! cargo run -p xtask -- decode --frames 400 --out dave.wav
//! ```
//!
//! # What this does and does not prove
//!
//! It **does** prove the decoder is self-consistent end to end: the bit budget
//! lands, every codebook index stays in range, and the output is speech rather
//! than noise. Those are not nothing — a wrong field width or a wrong ladder
//! base produces obvious garbage immediately, which is how the wideband base
//! error was originally caught.
//!
//! It does **not** prove bit-exactness. Only a differential run against the
//! ARM module does that, and that gate belongs in `xtask verify` once the
//! guest-side `sb_decode` state can be built. Do not let a pleasant-sounding
//! WAV stand in for it.
//!
//! # The parameters are checked, not assumed
//!
//! `loqng_codec::decoder` hardcodes Dave's configuration — order 10/8, the
//! 7/4/4 and 5/3 LSP stages, 8x5 and 4x10 innovation subvectors. The bank
//! states all of that in its own header, and [`check_params`] compares the two
//! and refuses to decode on any mismatch. A bank with a different layout
//! therefore fails loudly instead of decoding to noise.

use std::fs;
use std::path::{Path, PathBuf};

use loqng_codec::bits::Bits;
use loqng_codec::decoder::{
    SbDecoder, Voice, FULL_FRAME, NB_ORDER, NB_SPACING, NB_SPLIT_CB, SB_ORDER, SB_SPACING,
    SB_SPLIT_CB,
};
use loqng_data::bank::{BankHeader, BankParams, BankTables, TableSpan};

use crate::{flag, Paths};

/// The bank every other default in this repo points at.
const DEFAULT_BANK: &str = "EnglishUs/Dave/Dave-19200.16000.loqmsx.bin";

/// Reinterpret a codebook as signed bytes.
///
/// The tables are signed in the format and unsigned in the file; this is the
/// one place that conversion happens.
fn signed(b: &[u8]) -> Vec<i8> {
    b.iter().map(|&v| v as i8).collect()
}

fn span(raw: &[u8], s: &TableSpan, what: &str) -> Result<Vec<i8>, String> {
    s.slice(raw).map(signed).ok_or_else(|| {
        format!(
            "{what}: span {}..{} is outside the preamble",
            s.offset,
            s.offset + s.len
        )
    })
}

/// Refuse to decode a bank whose stated layout differs from the compiled one.
fn check_params(p: &BankParams) -> Result<(), String> {
    let mut bad = Vec::new();
    let mut want = |name: &str, got: usize, exp: usize| {
        if got != exp {
            bad.push(format!(
                "{name}: bank says {got}, decoder compiled for {exp}"
            ));
        }
    };

    want("nb order", p.nb.order, NB_ORDER);
    want("sb order", p.sb.order, SB_ORDER);
    want("nb spacing", p.nb.spacing as usize, NB_SPACING as usize);
    want("sb spacing", p.sb.spacing as usize, SB_SPACING as usize);
    want("nb lsp stages", p.nb.lsp_nbits.len(), 3);
    want("sb lsp stages", p.sb.lsp_nbits.len(), 2);
    want(
        "nb innovation subvector",
        p.nb.innov_subvect,
        NB_SPLIT_CB.subvect_size,
    );
    want(
        "sb innovation subvector",
        p.sb.innov_subvect,
        SB_SPLIT_CB.subvect_size,
    );
    want(
        "nb innovation shape bits",
        p.nb.innov_shape_bits as usize,
        NB_SPLIT_CB.shape_bits as usize,
    );
    want(
        "sb innovation shape bits",
        p.sb.innov_shape_bits as usize,
        SB_SPLIT_CB.shape_bits as usize,
    );

    for (i, exp) in [7u32, 4, 4].iter().enumerate() {
        if let Some(got) = p.nb.lsp_nbits.get(i) {
            want(
                &format!("nb lsp stage {} bits", i + 1),
                *got as usize,
                *exp as usize,
            );
        }
    }
    for (i, exp) in [5u32, 3].iter().enumerate() {
        if let Some(got) = p.sb.lsp_nbits.get(i) {
            want(
                &format!("sb lsp stage {} bits", i + 1),
                *got as usize,
                *exp as usize,
            );
        }
    }

    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "this bank does not match the compiled configuration:\n  {}",
            bad.join("\n  ")
        ))
    }
}

/// Minimal 16-bit mono PCM WAV.
fn write_wav(path: &Path, rate: u32, pcm: &[i16]) -> Result<(), String> {
    let data_len = (pcm.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + pcm.len() * 2);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    v.extend_from_slice(&2u16.to_le_bytes()); // block align
    v.extend_from_slice(&16u16.to_le_bytes()); // bits
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        v.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(path, &v).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(paths: &Paths, args: &[String]) -> Result<(), String> {
    let bank_path = flag(args, "--bank")
        .map(PathBuf::from)
        .unwrap_or_else(|| paths.data_dir.join(DEFAULT_BANK));
    let start: usize = flag(args, "--start")
        .map(|s| s.parse().unwrap_or(0))
        .unwrap_or(0);
    let count: usize = flag(args, "--frames")
        .map(|s| s.parse().unwrap_or(400))
        .unwrap_or(400);
    let out = flag(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("decode.wav"));

    let name = bank_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bank".to_string());

    let raw = fs::read(&bank_path).map_err(|e| format!("{}: {e}", bank_path.display()))?;
    let header = BankHeader::parse(&name, &raw).map_err(|e| e.to_string())?;
    let tables = BankTables::parse(&name, &raw, &header).map_err(|e| e.to_string())?;
    let params = BankParams::derive(&name, &raw, &tables).map_err(|e| e.to_string())?;

    println!("{name}");
    println!(
        "  {} Hz, {} frames of {} samples, {} coded bytes each",
        header.output_rate,
        header.frames,
        header.frame_samples,
        header.frame_bytes()
    );
    println!(
        "  preamble {} bytes, {} codebook contexts, XOR key 0x{:02x}",
        header.data_offset, tables.contexts, tables.xor_key
    );
    println!(
        "  nb: order {} spacing {} lsp {:?} innov {}x{}b shift {}",
        params.nb.order,
        params.nb.spacing,
        params.nb.lsp_nbits,
        params.nb.innov_subvect,
        params.nb.innov_shape_bits,
        params.nb.innov_shift
    );
    println!(
        "  sb: order {} spacing {} lsp {:?} innov {}x{}b shift {}",
        params.sb.order,
        params.sb.spacing,
        params.sb.lsp_nbits,
        params.sb.innov_subvect,
        params.sb.innov_shape_bits,
        params.sb.innov_shift
    );

    check_params(&params)?;

    let idx = tables
        .idx
        .slice(&raw)
        .ok_or_else(|| "context array is outside the preamble".to_string())?;
    let nb1 = span(&raw, &tables.nb_cb1, "nb lsp stage 1")?;
    let nb2 = span(&raw, &tables.nb_cb2, "nb lsp stage 2")?;
    let nb3 = span(&raw, &tables.nb_cb3, "nb lsp stage 3")?;
    let nbi = span(&raw, &tables.nb_innov, "nb innovation")?;
    let sb1 = span(&raw, &tables.sb_cb1, "sb lsp stage 1")?;
    let sb2 = span(&raw, &tables.sb_cb2, "sb lsp stage 2")?;
    let sbi = span(&raw, &tables.sb_innov, "sb innovation")?;

    // The shift stored here is SIG_SHIFT - log2(norm); the decoder wants the
    // log2 itself, because it re-derives the shift from SIG_SHIFT.
    let voice = Voice {
        ctx: idx,
        nb_lsp: [&nb1, &nb2, &nb3],
        nb_innov: &nbi,
        sb_lsp: [&sb1, &sb2],
        sb_innov: &sbi,
        nb_innov_log2: (14 - params.nb.innov_shift) as u8,
        sb_innov_log2: (14 - params.sb.innov_shift) as u8,
    };

    let frame_bytes = header.frame_bytes() as usize;
    let data_offset = header.data_offset as usize;
    let total = header.frames as usize;
    if start >= total {
        return Err(format!(
            "frame {start} is past the end of the bank ({total} frames)"
        ));
    }
    let count = count.min(total - start);

    let mut dec = SbDecoder::new();
    let mut pcm: Vec<i16> = Vec::with_capacity(count * FULL_FRAME);
    let mut frame = vec![0i16; FULL_FRAME];
    let mut coded = vec![0u8; frame_bytes];

    for f in start..start + count {
        let at = data_offset + f * frame_bytes;
        let src = raw
            .get(at..at + frame_bytes)
            .ok_or_else(|| format!("frame {f} runs past the end of the file"))?;
        for (d, s) in coded.iter_mut().zip(src) {
            *d = s ^ tables.xor_key;
        }

        let mut bits = Bits::new(&coded);
        dec.decode(&voice, f, &mut bits, &mut frame)
            .map_err(|e| format!("frame {f}: {e:?}"))?;

        // Every frame must consume exactly the coded bytes it was given.
        let used = bits.position() as usize;
        if used != frame_bytes * 8 {
            return Err(format!(
                "frame {f} consumed {used} bits of {}; the field widths are wrong",
                frame_bytes * 8
            ));
        }
        pcm.extend_from_slice(&frame);
    }

    let peak = pcm.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0);
    let energy: i64 = pcm.iter().map(|&s| (s as i64) * (s as i64)).sum();
    let rms = if pcm.is_empty() {
        0.0
    } else {
        ((energy as f64) / pcm.len() as f64).sqrt()
    };

    println!();
    println!(
        "decoded frames {}..{} -> {} samples ({:.2}s)",
        start,
        start + count,
        pcm.len(),
        pcm.len() as f64 / header.output_rate as f64
    );
    println!("  peak {peak}, rms {rms:.1}");
    if peak == 0 {
        return Err("decoded to pure silence; the codebooks are not being read".to_string());
    }

    write_wav(&out, header.output_rate, &pcm)?;
    println!("  wrote {}", out.display());
    Ok(())
}
