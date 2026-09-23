//! Opening the engine and speaking through it.
//!
//! The call sequence is the vendor driver's, unchanged from `xtask engrun`:
//!
//! ```text
//! ttsNewSession(&h, "")            ttsNewVoice(&h, inst, voice, rate, endec)
//! ttsNewInstance(&h, session, "")  ttsSetAudio(inst, "LoqAudioFile", f, "l", 0)
//! ttsSetInstanceParam(...)         ttsSetModularStructure(inst, "top", "acu", buf)
//! ttsRead(inst, text, 1, 0, 1, 2)
//! ```

use std::path::PathBuf;

use loqng::engine::{Runtime, GUEST_OUT};
use loqng::xrt::Mem;

/// Where the voice tree comes from.
pub enum Source {
    /// Compiled into this binary.
    BuiltIn,
    /// Read from disk, as `(lib, data)`.
    Disk(PathBuf, PathBuf),
}

/// Boxed, and boxed *before* the engine runs: a worker thread takes raw
/// pointers to both, and moving either afterwards leaves it pointing at the
/// old address.
pub struct Engine {
    rt: Box<Runtime>,
    m: Box<Mem>,
    inst: u32,
    #[allow(dead_code)]
    pub rate: u32,
}

impl Engine {
    pub fn open(src: &Source, voice: &str, rate: u32) -> Result<Engine, String> {
        let (rt, m) = match src {
            Source::BuiltIn => Runtime::built_in(loqng_voice::FILES),
            Source::Disk(lib, data) => Runtime::new(lib, data),
        };
        let (mut rt, mut m) = (Box::new(rt), Box::new(m));
        rt.init(&mut m);

        let call = |rt: &mut Runtime, m: &mut Mem, name: &str, a: &[u32]| -> Result<(), String> {
            let at = rt
                .lookup(name)
                .ok_or_else(|| format!("{name} is not exported by the engine"))?;
            let rc = rt.call(m, at, a);
            if rc == 0 {
                return Ok(());
            }
            let log = String::from_utf8_lossy(&rt.log).into_owned();
            Err(format!(
                "{name} failed, 0x{rc:08x}{}",
                if log.is_empty() {
                    String::new()
                } else {
                    format!("\n{}", log.trim_end())
                }
            ))
        };

        let out = rt.alloc(16);
        let scratch = rt.alloc(512);
        let empty = rt.cstring(&mut m, "");

        call(&mut rt, &mut m, "ttsNewSession", &[out, empty])?;
        let session = m.r32(out);
        call(
            &mut rt,
            &mut m,
            "ttsNewInstance",
            &[out + 4, session, empty],
        )?;
        let inst = m.r32(out + 4);

        // `InputTextCoding=ansi` is why input is read as Latin-1, not UTF-8.
        for (k, v) in [
            ("MultiSpacePause", "NO"),
            ("MaxParPause", "0"),
            ("InputTextCoding", "ansi"),
        ] {
            let kp = rt.cstring(&mut m, k);
            let vp = rt.cstring(&mut m, v);
            call(&mut rt, &mut m, "ttsSetInstanceParam", &[inst, kp, vp])?;
        }

        let vname = rt.cstring(&mut m, voice);
        let endec = rt.cstring(&mut m, "loqmsx");
        call(
            &mut rt,
            &mut m,
            "ttsNewVoice",
            &[out + 8, inst, vname, rate, endec],
        )?;

        let device = rt.cstring(&mut m, "LoqAudioFile");
        let file = rt.cstring(&mut m, GUEST_OUT);
        let linear = rt.cstring(&mut m, "l");
        call(
            &mut rt,
            &mut m,
            "ttsSetAudio",
            &[inst, device, file, linear, 0],
        )?;

        let top = rt.cstring(&mut m, "top");
        let acu = rt.cstring(&mut m, "acu");
        call(
            &mut rt,
            &mut m,
            "ttsSetModularStructure",
            &[inst, top, acu, scratch],
        )?;

        Ok(Engine { rt, m, inst, rate })
    }

    /// Speak once and take the audio. Raw 16-bit little-endian mono.
    pub fn say(&mut self, text: &str) -> Result<Vec<u8>, String> {
        // The engine emits nothing until it sees a sentence end.
        let mut say = text.to_string();
        if !say.trim_end().ends_with(['.', '!', '?', ';', ':']) {
            say.push('.');
        }
        let tp = self.rt.cstring(&mut self.m, &say);
        let at = self.rt.lookup("ttsRead").ok_or("no ttsRead")?;
        let inst = self.inst;
        let rc = self.rt.call(&mut self.m, at, &[inst, tp, 1, 0, 1, 2]);
        // `ttsRead` queues the utterance; the worker renders it.
        self.rt.pump();
        if rc != 0 {
            return Err(format!("ttsRead failed, 0x{rc:08x}"));
        }
        Ok(self.rt.take_sink(GUEST_OUT))
    }

    /// Anything the engine wrote to its log, and clear it.
    pub fn log(&mut self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut self.rt.log)).into_owned()
    }
}
