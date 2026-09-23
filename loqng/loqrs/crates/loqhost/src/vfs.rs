//! Virtual filesystem. The guest sees the layout the ENDEC had; every path it
//! opens is rewritten onto a host directory, so nothing is hardcoded in the
//! container and no absolute guest path has to exist for real.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub enum Backing {
    /// Whole file slurped at open time; reads are pure memory.
    Read {
        data: Vec<u8>,
        pos: usize,
    },
    Write {
        file: fs::File,
    },
    /// Captured in memory so the caller can take the bytes back.
    Sink {
        data: Vec<u8>,
    },
    Stdout,
    Stderr,
    Stdin,
}

pub struct OpenFile {
    pub path: String,
    pub backing: Backing,
    pub eof: bool,
    pub error: bool,
}

pub struct DirStream {
    pub entries: Vec<String>,
    pub pos: usize,
}

pub struct Vfs {
    /// Longest guest prefix wins.
    mounts: Vec<(String, PathBuf)>,
    /// Guest paths served from memory rather than disk.
    overlay: HashMap<String, Vec<u8>>,
    /// Guest paths that become in-memory sinks instead of host files.
    sink_paths: Vec<String>,
    /// Index of the stdin slot, so it can be refilled from a string.
    stdin_slot: Option<usize>,
    pub cwd: String,
    pub files: Vec<Option<OpenFile>>,
    pub dirs: Vec<Option<DirStream>>,
    pub trace: bool,
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

impl Vfs {
    pub fn new() -> Self {
        Vfs {
            mounts: Vec::new(),
            overlay: HashMap::new(),
            sink_paths: Vec::new(),
            stdin_slot: None,
            cwd: "/loq".to_string(),
            files: Vec::new(),
            dirs: Vec::new(),
            trace: false,
        }
    }

    pub fn mount(&mut self, guest_prefix: &str, host_dir: impl AsRef<Path>) {
        self.mounts
            .push((norm(guest_prefix), host_dir.as_ref().to_path_buf()));
        self.mounts.sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));
    }

    /// Serve `guest_path` from memory instead of disk.
    pub fn add_overlay(&mut self, guest_path: &str, data: Vec<u8>) {
        self.overlay.insert(norm(guest_path), data);
    }

    pub fn absolute(&self, path: &str) -> String {
        if path.starts_with('/') {
            norm(path)
        } else {
            norm(&format!("{}/{}", self.cwd, path))
        }
    }

    pub fn host_path(&self, guest_path: &str) -> Option<PathBuf> {
        let abs = self.absolute(guest_path);
        for (prefix, dir) in &self.mounts {
            if abs == *prefix {
                return Some(dir.clone());
            }
            let with_slash = format!("{prefix}/");
            if let Some(rest) = abs.strip_prefix(&with_slash) {
                return Some(dir.join(rest));
            }
        }
        None
    }

    pub fn read_file(&self, guest_path: &str) -> Option<Vec<u8>> {
        let abs = self.absolute(guest_path);
        if let Some(d) = self.overlay.get(&abs) {
            return Some(d.clone());
        }
        let host = self.host_path(&abs)?;
        fs::read(host).ok()
    }

    pub fn open(&mut self, guest_path: &str, mode: &str) -> Option<usize> {
        let abs = self.absolute(guest_path);
        let writing = mode.contains('w') || mode.contains('a') || mode.contains('+');

        if writing && self.sink_paths.iter().any(|p| *p == abs) {
            if self.trace {
                eprintln!("[vfs] open {abs} ({mode}) -> memory sink");
            }
            // Reopening a sink keeps what is already there, so several
            // utterances accumulate in one buffer.
            if let Some(i) = self.files.iter().position(|s| {
                matches!(s, Some(f) if f.path == abs && matches!(f.backing, Backing::Sink { .. }))
            }) {
                return Some(i);
            }
            return Some(self.install(OpenFile {
                path: abs,
                backing: Backing::Sink { data: Vec::new() },
                eof: false,
                error: false,
            }));
        }

        let backing = if writing {
            let host = self.host_path(&abs)?;
            if let Some(parent) = host.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .append(mode.contains('a'))
                .truncate(mode.contains('w'))
                .open(host)
                .ok()?;
            Backing::Write { file: f }
        } else {
            let data = if let Some(d) = self.overlay.get(&abs) {
                d.clone()
            } else {
                fs::read(self.host_path(&abs)?).ok()?
            };
            Backing::Read { data, pos: 0 }
        };

        if self.trace {
            eprintln!("[vfs] open {abs} ({mode})");
        }
        Some(self.install(OpenFile {
            path: abs,
            backing,
            eof: false,
            error: false,
        }))
    }

    /// An in-memory file the host can read back after the guest closes it.
    pub fn open_sink(&mut self, guest_path: &str) -> usize {
        let abs = self.absolute(guest_path);
        self.install(OpenFile {
            path: abs,
            backing: Backing::Sink { data: Vec::new() },
            eof: false,
            error: false,
        })
    }

    pub fn take_sink(&mut self, guest_path: &str) -> Option<Vec<u8>> {
        let abs = self.absolute(guest_path);
        for slot in self.files.iter_mut() {
            if let Some(f) = slot {
                if f.path == abs {
                    if let Backing::Sink { data } = &mut f.backing {
                        return Some(std::mem::take(data));
                    }
                }
            }
        }
        None
    }

    fn install(&mut self, f: OpenFile) -> usize {
        if let Some(i) = self.files.iter().position(|s| s.is_none()) {
            self.files[i] = Some(f);
            i
        } else {
            self.files.push(Some(f));
            self.files.len() - 1
        }
    }

    pub fn install_std(&mut self) -> (usize, usize, usize) {
        let i = self.install(OpenFile {
            path: "<stdin>".into(),
            backing: Backing::Stdin,
            eof: false,
            error: false,
        });
        self.stdin_slot = Some(i);
        let o = self.install(OpenFile {
            path: "<stdout>".into(),
            backing: Backing::Stdout,
            eof: false,
            error: false,
        });
        let e = self.install(OpenFile {
            path: "<stderr>".into(),
            backing: Backing::Stderr,
            eof: false,
            error: false,
        });
        (i, o, e)
    }

    /// Route writes to this guest path into memory; collect with take_sink.
    pub fn add_sink_path(&mut self, guest_path: &str) {
        let p = self.absolute(guest_path);
        if !self.sink_paths.contains(&p) {
            self.sink_paths.push(p);
        }
    }

    /// Feed the guest a fixed stdin instead of the process's real one.
    pub fn set_stdin(&mut self, data: Vec<u8>) {
        if let Some(i) = self.stdin_slot {
            if let Some(f) = self.get_mut(i) {
                f.backing = Backing::Read { data, pos: 0 };
                f.eof = false;
            }
        }
    }

    /// How many bytes a sink currently holds.
    pub fn sink_len(&mut self, guest_path: &str) -> usize {
        let abs = self.absolute(guest_path);
        for slot in self.files.iter() {
            if let Some(f) = slot {
                if f.path == abs {
                    if let Backing::Sink { data } = &f.backing {
                        return data.len();
                    }
                }
            }
        }
        0
    }

    /// Everything a sink gained since `from`.
    pub fn sink_since(&mut self, guest_path: &str, from: usize) -> Vec<u8> {
        let abs = self.absolute(guest_path);
        for slot in self.files.iter() {
            if let Some(f) = slot {
                if f.path == abs {
                    if let Backing::Sink { data } = &f.backing {
                        return data.get(from..).unwrap_or(&[]).to_vec();
                    }
                }
            }
        }
        Vec::new()
    }

    /// Drop any sink contents so the next run starts clean.
    pub fn clear_sinks(&mut self) {
        for slot in self.files.iter_mut() {
            let drop_it = matches!(
                slot.as_ref().map(|f| &f.backing),
                Some(Backing::Sink { .. })
            );
            if drop_it {
                *slot = None;
            }
        }
    }

    pub fn close(&mut self, idx: usize) -> bool {
        match self.files.get_mut(idx) {
            Some(slot @ Some(_)) => {
                // Sinks stay addressable so the host can collect their bytes.
                let is_sink = matches!(
                    slot.as_ref().map(|f| &f.backing),
                    Some(Backing::Sink { .. })
                );
                if !is_sink {
                    *slot = None;
                }
                true
            }
            _ => false,
        }
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut OpenFile> {
        self.files.get_mut(idx).and_then(|s| s.as_mut())
    }

    pub fn read(&mut self, idx: usize, len: usize) -> Vec<u8> {
        let Some(f) = self.get_mut(idx) else {
            return Vec::new();
        };
        match &mut f.backing {
            Backing::Read { data, pos } => {
                let end = (*pos + len).min(data.len());
                let out = data[*pos..end].to_vec();
                if out.len() < len {
                    f.eof = true;
                }
                *pos = end;
                out
            }
            Backing::Stdin => {
                use std::io::Read;
                let mut buf = vec![0u8; len];
                match std::io::stdin().read(&mut buf) {
                    Ok(n) => {
                        buf.truncate(n);
                        if n < len {
                            f.eof = true;
                        }
                        buf
                    }
                    Err(_) => {
                        f.error = true;
                        Vec::new()
                    }
                }
            }
            _ => {
                f.eof = true;
                Vec::new()
            }
        }
    }

    /// Absolute read that ignores and does not move the stream position.
    /// This is what mmap needs.
    pub fn read_at(&mut self, idx: usize, off: usize, len: usize) -> Vec<u8> {
        match self.get_mut(idx) {
            Some(f) => match &f.backing {
                Backing::Read { data, .. } => {
                    let start = off.min(data.len());
                    let end = (off + len).min(data.len());
                    data[start..end].to_vec()
                }
                _ => Vec::new(),
            },
            None => Vec::new(),
        }
    }

    pub fn write(&mut self, idx: usize, bytes: &[u8]) -> usize {
        let Some(f) = self.get_mut(idx) else {
            return 0;
        };
        match &mut f.backing {
            Backing::Write { file } => file.write(bytes).unwrap_or(0),
            Backing::Sink { data } => {
                data.extend_from_slice(bytes);
                bytes.len()
            }
            Backing::Stdout => {
                let mut out = std::io::stdout();
                let _ = out.write_all(bytes);
                bytes.len()
            }
            Backing::Stderr => {
                let mut out = std::io::stderr();
                let _ = out.write_all(bytes);
                bytes.len()
            }
            _ => 0,
        }
    }

    pub fn seek(&mut self, idx: usize, off: i64, whence: i32) -> i64 {
        let Some(f) = self.get_mut(idx) else {
            return -1;
        };
        match &mut f.backing {
            Backing::Read { data, pos } => {
                let base = match whence {
                    0 => 0,
                    1 => *pos as i64,
                    _ => data.len() as i64,
                };
                let np = (base + off).clamp(0, data.len() as i64);
                *pos = np as usize;
                f.eof = false;
                0
            }
            Backing::Write { file } => {
                use std::io::Seek;
                let sf = match whence {
                    0 => std::io::SeekFrom::Start(off.max(0) as u64),
                    1 => std::io::SeekFrom::Current(off),
                    _ => std::io::SeekFrom::End(off),
                };
                file.seek(sf).map(|_| 0).unwrap_or(-1)
            }
            Backing::Sink { data } => {
                let _ = data;
                0
            }
            _ => -1,
        }
    }

    pub fn tell(&mut self, idx: usize) -> i64 {
        let Some(f) = self.get_mut(idx) else {
            return -1;
        };
        match &mut f.backing {
            Backing::Read { pos, .. } => *pos as i64,
            Backing::Write { file } => {
                use std::io::Seek;
                file.stream_position().map(|p| p as i64).unwrap_or(-1)
            }
            Backing::Sink { data } => data.len() as i64,
            _ => -1,
        }
    }

    pub fn opendir(&mut self, guest_path: &str) -> Option<usize> {
        let abs = self.absolute(guest_path);
        let mut entries = vec![".".to_string(), "..".to_string()];

        if let Some(host) = self.host_path(&abs) {
            if let Ok(rd) = fs::read_dir(&host) {
                for e in rd.flatten() {
                    entries.push(e.file_name().to_string_lossy().into_owned());
                }
            } else if !self
                .overlay
                .keys()
                .any(|k| k.starts_with(&format!("{abs}/")))
            {
                return None;
            }
        }
        // Overlay files show up in their parent directory too.
        let prefix = format!("{abs}/");
        for k in self.overlay.keys() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                if !rest.contains('/') && !entries.iter().any(|e| e == rest) {
                    entries.push(rest.to_string());
                }
            }
        }
        if self.trace {
            eprintln!("[vfs] opendir {abs} -> {} entries", entries.len());
        }

        let stream = DirStream { entries, pos: 0 };
        if let Some(i) = self.dirs.iter().position(|s| s.is_none()) {
            self.dirs[i] = Some(stream);
            Some(i)
        } else {
            self.dirs.push(Some(stream));
            Some(self.dirs.len() - 1)
        }
    }

    pub fn readdir(&mut self, idx: usize) -> Option<String> {
        let d = self.dirs.get_mut(idx)?.as_mut()?;
        let e = d.entries.get(d.pos)?.clone();
        d.pos += 1;
        Some(e)
    }

    pub fn closedir(&mut self, idx: usize) {
        if let Some(slot) = self.dirs.get_mut(idx) {
            *slot = None;
        }
    }

    pub fn remove(&mut self, guest_path: &str) -> bool {
        match self.host_path(&self.absolute(guest_path)) {
            Some(h) => fs::remove_file(h).is_ok(),
            None => false,
        }
    }
}

/// Collapse `.`, `..` and duplicate separators; always returns a leading slash.
fn norm(path: &str) -> String {
    let unified = path.replace('\\', "/");
    let mut parts: Vec<&str> = Vec::new();
    for seg in unified.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    format!("/{}", parts.join("/"))
}

// SNAPSHOT-MARK
use armemu::{Reader, SnapResult, Writer};

impl Vfs {
    /// Save what the guest can observe: the open descriptors and where they
    /// are positioned. Mounts and overlays are rebuilt from the config on the
    /// way back in, so they are not carried here.
    pub fn save_state(&self, w: &mut Writer) {
        w.put(&self.cwd);
        w.put(&self.stdin_slot);
        w.u32(self.files.len() as u32);
        for slot in &self.files {
            let Some(f) = slot else {
                w.u8(0);
                continue;
            };
            w.u8(1);
            w.put(&f.path);
            w.put(&f.eof);
            w.put(&f.error);
            match &f.backing {
                // Only the offset: the bytes are still on disk, and the
                // fingerprint is what catches them having changed.
                Backing::Read { pos, .. } => {
                    w.u8(0);
                    w.put(pos);
                }
                Backing::Write { .. } => w.u8(1),
                Backing::Sink { data } => {
                    w.u8(2);
                    w.bytes(data);
                }
                Backing::Stdout => w.u8(3),
                Backing::Stderr => w.u8(4),
                Backing::Stdin => w.u8(5),
            }
        }
        w.u32(self.dirs.len() as u32);
        for slot in &self.dirs {
            match slot {
                None => w.u8(0),
                Some(d) => {
                    w.u8(1);
                    w.put(&d.entries);
                    w.put(&d.pos);
                }
            }
        }
    }

    /// Restore onto a `Vfs` whose mounts and overlays are already set up.
    /// Slot indices are preserved exactly, because the guest holds `FILE*`
    /// objects that point at them by index.
    pub fn restore_state(&mut self, r: &mut Reader) -> SnapResult<()> {
        self.cwd = r.get()?;
        self.stdin_slot = r.get()?;

        let n = r.count()?;
        let mut files = Vec::with_capacity(n);
        for _ in 0..n {
            if r.u8()? == 0 {
                files.push(None);
                continue;
            }
            let path: String = r.get()?;
            let eof: bool = r.get()?;
            let error: bool = r.get()?;
            let backing = match r.u8()? {
                0 => {
                    let pos: usize = r.get()?;
                    let data = self
                        .read_file(&path)
                        .ok_or_else(|| format!("snapshot has {path} open, but it is gone"))?;
                    if pos > data.len() {
                        return Err(format!("snapshot is positioned past the end of {path}"));
                    }
                    Backing::Read { data, pos }
                }
                1 => {
                    let host = self
                        .host_path(&path)
                        .ok_or_else(|| format!("snapshot writes to {path}, which is unmapped"))?;
                    let file = fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .append(true)
                        .open(host)
                        .map_err(|e| format!("snapshot reopens {path}: {e}"))?;
                    Backing::Write { file }
                }
                2 => Backing::Sink {
                    data: r.bytes()?.to_vec(),
                },
                3 => Backing::Stdout,
                4 => Backing::Stderr,
                5 => Backing::Stdin,
                t => return Err(format!("snapshot has backing tag {t}")),
            };
            files.push(Some(OpenFile {
                path,
                backing,
                eof,
                error,
            }));
        }
        self.files = files;

        let n = r.count()?;
        let mut dirs = Vec::with_capacity(n);
        for _ in 0..n {
            if r.u8()? == 0 {
                dirs.push(None);
                continue;
            }
            dirs.push(Some(DirStream {
                entries: r.get()?,
                pos: r.get()?,
            }));
        }
        self.dirs = dirs;
        Ok(())
    }
}
