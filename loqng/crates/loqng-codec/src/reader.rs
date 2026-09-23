//! Random access into a voice bank: `Decode(ctx, startPos, count, out, sig)`.
//!
//! [`SbDecoder`] decodes frame *N* given frames *0..N*. The engine instead
//! asks for an arbitrary **sample** range, which is what the Cat stage emits:
//! a list of `CONFINI` boundaries naming segments of the bank to concatenate.
//! This turns one into the other.
//!
//! # Why a seek is not free
//!
//! The decoder is stateful in four ways that all survive a frame boundary —
//! the excitation history, `mem_sp`, `interp_qlpc` and the QMF memories — so
//! frame *N* decoded cold is not frame *N* decoded in sequence. The original
//! handles this by **re-creating the decoder and pre-rolling**, and
//! `driver.rs` records the rule:
//!
//! > anything else — destroy and re-create the decoder, then decode forward
//! > from `target - state[0x38]` up to and including `target`, discarding the
//! > output.
//!
//! `state[0x38]` is [`PREROLL`], and it is 6: read live out of a fresh `Open`
//! by `xtask refpcm`. **It has to be exact.** Too few frames and the filter
//! state has not converged; too many and the decoder was re-initialised at a
//! different point, which is just as wrong.
//!
//! # The cache is part of the specification, not an optimisation
//!
//! Because a seek re-creates the decoder, *when* the original chooses to seek
//! is observable in the output. It keeps two decoded frames and seeks only
//! when the target is neither of them nor the next one:
//!
//! ```text
//! target == current      serve from the current buffer
//! target == current + 1  rotate current into previous, decode one frame
//! target == previous     serve from the previous buffer
//! anything else          re-create + preroll
//! ```
//!
//! Dropping the two-frame cache and seeking every time would still produce
//! *plausible* audio while differing from the original wherever a unit
//! straddles a frame boundary. [`BankReader`] reproduces the cache for that
//! reason.

use crate::bits::Bits;
use crate::decoder::{DecodeError, SbDecoder, Voice, FULL_FRAME};

/// Frames of preroll before a sought frame — `state[0x38]`.
///
/// Read live from a fresh `Open` rather than inferred. Do not tune it.
pub const PREROLL: u32 = 6;

/// Samples added to every requested start position — `state[0x34]`, 170.
///
/// `Decode` does `pos = startPos + dOff` before any frame arithmetic, so a
/// unit's `CONFINI` start is **not** the sample the decoder reads: it is 170
/// samples earlier than the one the caller gets. This is the codec's own group
/// delay — the low band's 40-sample subframe lag plus the 64-tap QMF — and the
/// driver carries it rather than compensating inside the decoder.
///
/// Leaving it out is not subtle: every rendered unit starts 170 samples early,
/// and the whole utterance differs from sample 0 while still being exactly the
/// right length. `driver.rs` called this field "wrap length" for a while,
/// which is how it got missed.
pub const OUTPUT_DELAY: u64 = 170;

/// One decoded frame, with the frame index it came from.
#[derive(Debug, Clone)]
struct Cached {
    frame: Option<u32>,
    pcm: Vec<i16>,
}

impl Cached {
    fn new() -> Self {
        Cached {
            frame: None,
            pcm: vec![0i16; FULL_FRAME],
        }
    }
}

/// Where the coded frames come from.
///
/// The bank file is the obvious source, but keeping this a trait lets a test
/// feed synthetic frames and keeps `loqng-codec` free of file handling.
pub trait CodedFrames {
    /// Total frames available.
    fn frames(&self) -> u32;
    /// The descrambled coded bytes of one frame, or `None` if out of range.
    fn frame(&self, index: u32) -> Option<&[u8]>;
}

/// A flat `&[u8]` of coded frames, already de-XORed.
pub struct FlatFrames<'a> {
    data: &'a [u8],
    frame_bytes: usize,
}

impl<'a> FlatFrames<'a> {
    pub fn new(data: &'a [u8], frame_bytes: usize) -> Self {
        FlatFrames { data, frame_bytes }
    }
}

impl<'a> CodedFrames for FlatFrames<'a> {
    fn frames(&self) -> u32 {
        if self.frame_bytes == 0 {
            0
        } else {
            (self.data.len() / self.frame_bytes) as u32
        }
    }

    fn frame(&self, index: u32) -> Option<&[u8]> {
        let at = (index as usize).checked_mul(self.frame_bytes)?;
        self.data.get(at..at + self.frame_bytes)
    }
}

/// Reads arbitrary sample ranges out of a bank, reproducing the original's
/// frame cache and preroll.
pub struct BankReader<'v> {
    voice: Voice<'v>,
    dec: SbDecoder,
    cur: Cached,
    prev: Cached,
    /// Bytes of the coded frame, for the XOR pass.
    key: u8,
}

impl<'v> BankReader<'v> {
    pub fn new(voice: Voice<'v>, key: u8) -> Self {
        BankReader {
            voice,
            dec: SbDecoder::new(),
            cur: Cached::new(),
            prev: Cached::new(),
            key,
        }
    }

    /// Throw the decoder away, as the original's seek path does.
    fn reseat(&mut self) {
        self.dec = SbDecoder::new();
        self.cur.frame = None;
        self.prev.frame = None;
    }

    /// Decode one frame into `cur`, rotating `cur` into `prev` first.
    fn advance<F: CodedFrames>(&mut self, src: &F, frame: u32) -> Result<(), DecodeError> {
        let coded = src
            .frame(frame)
            .ok_or(DecodeError::NoContext(frame as usize))?;
        let mut buf = vec![0u8; coded.len()];
        for (d, s) in buf.iter_mut().zip(coded) {
            *d = s ^ self.key;
        }

        std::mem::swap(&mut self.cur, &mut self.prev);
        let mut bits = Bits::new(&buf);
        let mut out = vec![0i16; FULL_FRAME];
        self.dec
            .decode(&self.voice, frame as usize, &mut bits, &mut out)?;
        self.cur.frame = Some(frame);
        self.cur.pcm = out;
        Ok(())
    }

    /// Make `frame` available, seeking and pre-rolling if it is not reachable
    /// by one sequential step.
    fn reach<F: CodedFrames>(&mut self, src: &F, frame: u32) -> Result<(), DecodeError> {
        if self.cur.frame == Some(frame) || self.prev.frame == Some(frame) {
            return Ok(());
        }
        if self.cur.frame == Some(frame.wrapping_sub(1)) && frame > 0 {
            return self.advance(src, frame);
        }

        // A real seek: re-create, then converge.
        self.reseat();
        let start = frame.saturating_sub(PREROLL);
        for f in start..=frame {
            self.advance(src, f)?;
        }
        Ok(())
    }

    fn cached(&self, frame: u32) -> Option<&[i16]> {
        if self.cur.frame == Some(frame) {
            Some(&self.cur.pcm)
        } else if self.prev.frame == Some(frame) {
            Some(&self.prev.pcm)
        } else {
            None
        }
    }

    /// Read `count` samples starting at absolute sample `pos`.
    ///
    /// This is `Decode(ctx, startPos, count, out, sig)` with the bank supplied
    /// directly instead of through an `ELQBin`. [`OUTPUT_DELAY`] is applied
    /// here, exactly as `Decode` applies it.
    pub fn read<F: CodedFrames>(
        &mut self,
        src: &F,
        pos: u64,
        count: usize,
        out: &mut Vec<i16>,
    ) -> Result<(), DecodeError> {
        let mut at = pos.wrapping_add(OUTPUT_DELAY);
        let mut left = count;
        while left > 0 {
            let frame = (at / FULL_FRAME as u64) as u32;
            let within = (at % FULL_FRAME as u64) as usize;
            self.reach(src, frame)?;
            let pcm = self
                .cached(frame)
                .ok_or(DecodeError::NoContext(frame as usize))?;
            let take = (FULL_FRAME - within).min(left);
            out.extend_from_slice(&pcm[within..within + take]);
            at += take as u64;
            left -= take;
        }
        Ok(())
    }
}

/// One unit the Cat stage selected: a sample range in the bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unit {
    pub start: u64,
    pub count: usize,
}

impl Unit {
    /// Convert a `CONFINI` pair of bank times, in seconds, at `rate`.
    ///
    /// The engine prints these with six decimals and the round-trip is exact:
    /// summing `count` over a whole utterance reproduces the captured
    /// `audio.raw` length to the sample on every corpus entry.
    pub fn from_confini(t0: f64, t1: f64, rate: u32) -> Unit {
        let a = (t0 * rate as f64).round().max(0.0) as u64;
        let b = (t1 * rate as f64).round().max(0.0) as u64;
        Unit {
            start: a,
            count: b.saturating_sub(a) as usize,
        }
    }
}

/// Render a whole unit list by plain concatenation, with no smoothing.
pub fn render<F: CodedFrames>(
    reader: &mut BankReader,
    src: &F,
    units: &[Unit],
) -> Result<Vec<i16>, DecodeError> {
    let total: usize = units.iter().map(|u| u.count).sum();
    let mut out = Vec::with_capacity(total);
    for u in units {
        reader.read(src, u.start, u.count, &mut out)?;
    }
    Ok(out)
}

/// Scale a sample by a window entry, the way both window primitives do.
///
/// `sample * win` then, from `sub_4e168`:
///
/// ```text
/// r3 += (r3 >> 31) >>> 17     ; += 0x7fff when negative
/// r3 >>= 15
/// ```
///
/// The conditional add makes the arithmetic shift **truncate toward zero**
/// instead of flooring. Using a plain `>> 15` is wrong on every negative
/// sample, which is half of them.
pub fn window_scale(sample: i16, win: i16) -> i32 {
    let p = (sample as i32).wrapping_mul(win as i32);
    if p < 0 {
        (p.wrapping_add(0x7fff)) >> 15
    } else {
        p >> 15
    }
}

/// Window index for the fade-**out** leg: rises 0 -> 627 across `n` samples.
///
/// `sub_4e168` keeps a running `627 * j` and divides by `n`, so the step is
/// `627 / n` and is generally fractional — with `n` of 64 it is about 9.8.
/// That fractional stepping is observable: the fade-out and fade-in weights do
/// **not** sum to exactly 32767, and reproducing the two indices separately is
/// what makes the output match.
pub fn fade_out_index(j: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (crate::window::WINDOW_LAST as usize * j / n).min(crate::window::WINDOW_LEN - 1)
}

/// Window index for the fade-**in** leg: falls 627 -> 0 across `n` samples.
pub fn fade_in_index(j: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (crate::window::WINDOW_LAST as usize * (n - j.min(n)) / n).min(crate::window::WINDOW_LEN - 1)
}

/// Overlap-add two segments across a join.
///
/// `left` fades out and `right` fades in, both `n` samples long, and the
/// result replaces the `n` output samples centred on the boundary. The add
/// saturates to `i16`, as `sub_4ee58` does.
pub fn crossfade(win: &[i16], left: &[i16], right: &[i16], n: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let lo = win.get(fade_out_index(j, n)).copied().unwrap_or(0);
        let hi = win.get(fade_in_index(j, n)).copied().unwrap_or(0);
        let a = window_scale(left.get(j).copied().unwrap_or(0), lo);
        let b = window_scale(right.get(j).copied().unwrap_or(0), hi);
        out.push(a.wrapping_add(b).clamp(-32768, 32767) as i16);
    }
    out
}

/// Samples of overlap either side of a join.
///
/// Empirical: the differences between a hard-cut concatenation and the
/// engine's own audio are runs of exactly 64 samples centred on each
/// discontinuous unit boundary, so the overlap is 64 with the join at its
/// centre. Contiguous boundaries show no difference at all and get no
/// smoothing.
pub const JOIN_HALF: usize = 32;

/// Render a unit list, overlap-adding at the tail of every unit whose
/// `joins[i]` is set.
///
/// `joins[i]` is `ALGO == "CONCATENAZIONE"`, **not** whether the two units are
/// discontiguous in the bank. That distinction is load-bearing and cost a
/// wrong turn: contiguous joins are smoothed too. When the segments are
/// contiguous the two legs read the *same* bank samples, so the crossfade is a
/// near-identity — but only near, because the fade-out and fade-in window
/// entries do not sum to exactly 32767. The result is a handful of samples
/// differing by 1 or 2, scattered across the overlap, which is precisely what
/// the corpus shows at those boundaries. Skipping them leaves runs of 5 to 62
/// samples wrong at every `CONCATENAZIONE` boundary that happens to be
/// contiguous.
///
/// The bank is walked in the order the engine walks it — each unit in turn,
/// reading `JOIN_HALF` samples **past** a unit's end and `JOIN_HALF`
/// **before** the next unit's start — so the decoder's cache and seek pattern
/// match too, not just the arithmetic.
pub fn render_joined<F: CodedFrames>(
    reader: &mut BankReader,
    src: &F,
    units: &[Unit],
    joins: &[bool],
    win: &[i16],
) -> Result<Vec<i16>, DecodeError> {
    let half = JOIN_HALF;
    let n = 2 * half;
    let total: usize = units.iter().map(|u| u.count).sum();
    let mut out: Vec<i16> = Vec::with_capacity(total);
    // Second half of a pending crossfade, to lay over the next unit's head.
    let mut pending: Option<Vec<i16>> = None;

    for (i, u) in units.iter().enumerate() {
        let at = out.len();
        reader.read(src, u.start, u.count, &mut out)?;

        if let Some(p) = pending.take() {
            for (k, v) in p.iter().enumerate() {
                if let Some(slot) = out.get_mut(at + k) {
                    *slot = *v;
                }
            }
        }

        let Some(next) = units.get(i + 1) else {
            continue;
        };
        if !joins.get(i).copied().unwrap_or(false) {
            continue;
        }
        if u.count < half || next.count < half || next.start < half as u64 {
            continue;
        }

        // Left leg: this unit's last `half` samples plus `half` beyond its end.
        let mut left: Vec<i16> = out[out.len() - half..].to_vec();
        reader.read(src, u.start + u.count as u64, half, &mut left)?;

        // Right leg: `half` before the next unit's start, plus its first `half`.
        let mut right: Vec<i16> = Vec::with_capacity(n);
        reader.read(src, next.start - half as u64, n, &mut right)?;

        let blend = crossfade(win, &left, &right, n);
        let tail = out.len() - half;
        out[tail..].copy_from_slice(&blend[..half]);
        pending = Some(blend[half..].to_vec());
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confini_rounds_to_the_nearest_sample() {
        // The first unit of corpus capture 0000.
        let u = Unit::from_confini(2516.225070, 2516.233250, 16000);
        assert_eq!(u.start, 40259601);
        assert_eq!(u.count, 131);
    }

    #[test]
    fn a_zero_length_unit_is_not_negative() {
        let u = Unit::from_confini(5.0, 5.0, 16000);
        assert_eq!(u.count, 0);
        // And a reversed pair saturates rather than wrapping.
        let u = Unit::from_confini(5.0, 4.0, 16000);
        assert_eq!(u.count, 0);
    }

    #[test]
    fn flat_frames_slices_on_the_coded_size() {
        let data: Vec<u8> = (0..96u8).collect();
        let f = FlatFrames::new(&data, 48);
        assert_eq!(f.frames(), 2);
        assert_eq!(f.frame(0).unwrap()[0], 0);
        assert_eq!(f.frame(1).unwrap()[0], 48);
        assert!(f.frame(2).is_none());
    }

    #[test]
    fn preroll_is_six_and_clamps_at_the_start_of_the_bank() {
        // A seek near frame 0 must not underflow.
        assert_eq!(PREROLL, 6);
        assert_eq!(3u32.saturating_sub(PREROLL), 0);
        assert_eq!(9u32.saturating_sub(PREROLL), 3);
    }
}
