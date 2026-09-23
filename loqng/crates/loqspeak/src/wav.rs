//! A RIFF/WAVE header for 16-bit mono PCM.
//!
//! The engine hands back raw little-endian samples; this is the forty-four
//! bytes that make them a file anything will play.

pub fn wav(pcm: &[u8], rate: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let block = channels * bits / 8;
    let byte_rate = rate * block as u32;
    let data = pcm.len() as u32;

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_44_bytes_and_says_what_it_is() {
        let pcm = [0u8; 320];
        let w = wav(&pcm, 16000);
        assert_eq!(w.len(), 44 + pcm.len());
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..12], b"WAVE");
        assert_eq!(&w[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes(w[4..8].try_into().unwrap()),
            36 + pcm.len() as u32
        );
        assert_eq!(
            u32::from_le_bytes(w[40..44].try_into().unwrap()),
            pcm.len() as u32
        );
        // 16 kHz, mono, 16-bit: 32000 bytes a second, 2 to a frame.
        assert_eq!(u32::from_le_bytes(w[24..28].try_into().unwrap()), 16000);
        assert_eq!(u32::from_le_bytes(w[28..32].try_into().unwrap()), 32000);
        assert_eq!(u16::from_le_bytes(w[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(w[34..36].try_into().unwrap()), 16);
    }

    #[test]
    fn samples_are_untouched() {
        let pcm: Vec<u8> = (0..256).map(|k| k as u8).collect();
        assert_eq!(&wav(&pcm, 22050)[44..], &pcm[..]);
    }
}
