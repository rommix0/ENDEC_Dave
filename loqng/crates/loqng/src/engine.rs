//! The runtime the translated engine runs on.
//!
//! `eng` is the whole Loquendo engine as Rust ([`crate::eng`]), but a shared
//! object still expects a C library underneath it: memory, files, the dynamic
//! loader, a few maths calls. That is this module. It is the last thing
//! standing between the translation and audio, and it is the only part of the
//! port that is written rather than generated.
//!
//! Nothing here interprets ARM. [`Runtime`] implements [`crate::xrt::Host`],
//! so an import in the translated code is an ordinary Rust call, and an
//! indirect transfer goes to [`crate::eng::call_addr`].
//!
//! The reference for every one of these is `loqrs`
//! `crates/loqhost/src/libc.rs`, which is proved against the ARM oracle over
//! the whole corpus. Where behaviour is subtle the comment says which fact
//! was paid for there — the `mmap` one especially.

pub mod cfmt;
pub mod sched;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::eng;
use crate::xrt::{Cpu, Host, Mem};

/// Guest heap. The engine peaks around 41 MB, so this is room to spare.
const HEAP: u32 = 0x1000_0000;
const HEAP_SIZE: u32 = 96 << 20;
/// Files mapped with `mmap` get their own regions above the heap.
const MMAP_BASE: u32 = 0x6000_0000;
const STACK_TOP: u32 = 0x7F00_0000;
const STACK_SIZE: u32 = 8 << 20;
/// Worker stacks sit below the main one, one per guest thread.
const WORKER_STACK_TOP: u32 = 0x7E00_0000;
const WORKER_STACK: u32 = 4 << 20;

/// The engine's memory and runtime, reached by a worker thread. Sound only
/// because the scheduler's baton means one thread touches them at a time;
/// see [`sched`].
///
/// **Neither may move once a worker exists.** A `Runtime` built as a local
/// and then returned by value leaves the worker pointing at the old address.
/// Keep them where they are, or box them before the engine runs.
#[derive(Clone, Copy)]
struct Ptrs {
    rt: *mut Runtime,
    mem: *mut Mem,
}

unsafe impl Send for Ptrs {}

const DIRENT: u32 = 0x110;

/// Guest paths, fixed regardless of where the real files live. The engine
/// derives everything else from these, so they are part of the contract with
/// it rather than a convenience.
pub const GUEST_ROOT: &str = "/loq";
pub const GUEST_LIB: &str = "/loq/lib";
pub const GUEST_DATA: &str = "/loq/data";
pub const GUEST_SESSION: &str = "/loq/default.session";
pub const GUEST_OUT: &str = "/loq/out.raw";

struct Open {
    guest: String,
    real: Option<PathBuf>,
    /// Captured in memory instead of written out — the audio sink.
    sink: bool,
    /// A real file rather than the console.
    is_file: bool,
    data: Vec<u8>,
    pos: usize,
    #[allow(dead_code)]
    write: bool,
    dirty: bool,
}

struct Dir {
    names: Vec<String>,
    at: usize,
    ent: u32,
}

/// Bump allocator with a reuse list. No coalescing: the engine's allocation
/// pattern is load-heavy and then steady, so fragmentation never bites.
struct Heap {
    next: u32,
    end: u32,
    live: HashMap<u32, u32>,
    free: Vec<(u32, u32)>,
}

impl Heap {
    fn new() -> Heap {
        Heap {
            next: HEAP,
            end: HEAP + HEAP_SIZE,
            live: HashMap::new(),
            free: Vec::new(),
        }
    }

    fn alloc(&mut self, want: u32) -> u32 {
        let n = (want.max(1) + 7) & !7;
        if let Some(k) = self.free.iter().position(|&(_, s)| s >= n) {
            let (a, s) = self.free.swap_remove(k);
            self.live.insert(a, s);
            return a;
        }
        let a = self.next;
        assert!(
            a + n <= self.end,
            "the guest heap is exhausted at {n} bytes"
        );
        self.next += n;
        self.live.insert(a, n);
        a
    }

    fn free(&mut self, a: u32) {
        if a == 0 {
            return;
        }
        if let Some(n) = self.live.remove(&a) {
            self.free.push((a, n));
        }
    }

    fn size(&self, a: u32) -> u32 {
        self.live.get(&a).copied().unwrap_or(0)
    }
}

pub struct Runtime {
    heap: Heap,
    files: HashMap<u32, Open>,
    dirs: HashMap<u32, Dir>,
    errno: u32,
    dl_err: u32,
    mmap_next: u32,
    /// Guest prefix -> real directory, longest match first.
    mounts: Vec<(String, PathBuf)>,
    /// Files served from memory, such as the generated session file.
    overlays: HashMap<String, Vec<u8>>,
    /// The voice tree compiled into the binary, if there is one: guest path
    /// to bytes, sorted by path. Consulted after `overlays` and before the
    /// real filesystem, so a mounted directory is an override, not a rival.
    /// This is what makes a single-file deployment possible — see
    /// `loqng-voice`.
    built_in: &'static [(&'static str, &'static [u8])],
    /// Paths whose writes are kept rather than written out. The audio lands
    /// here, so the engine needs no real output file at all.
    pub sinks: HashMap<String, Vec<u8>>,
    /// Anything the engine wrote to stdout or stderr, for diagnostics.
    pub log: Vec<u8>,
    seed: u32,
    /// Worker threads the engine asked for: `(start routine, argument)`.
    pub deferred: Vec<(u32, u32)>,
    /// How often it waited on a condition variable. Non-zero means the
    /// engine really does expect another thread to be running.
    pub cond_waits: u32,
    /// Report every file and module the engine looks for. This is the first
    /// thing to turn on when it refuses to load a voice.
    pub trace: bool,
    /// One scheduler per engine; see [`sched`].
    sched: Arc<sched::Sched>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for Runtime {
    /// Leave the workers parked.
    ///
    /// A worker blocks inside `pthread_cond_wait`, which is deep inside
    /// translated engine code. Waking it makes it *resume that code*, and
    /// waking it in order to shut down therefore runs the engine against
    /// memory that is being freed — an access violation, which is exactly
    /// what happened. There is no way to unwind a translated frame, so the
    /// only safe thing is never to wake it: it stays parked on a condition
    /// variable nobody will ever signal, and the process reaps it on exit.
    ///
    /// The cost is one parked thread per engine instance. A service should
    /// keep one [`Runtime`] and speak through it repeatedly, which is the
    /// engine's own model anyway.
    fn drop(&mut self) {
        self.workers.clear();
    }
}

fn cstr(m: &Mem, mut a: u32) -> String {
    let mut out = Vec::new();
    while a != 0 {
        let b = m.r8(a);
        if b == 0 {
            break;
        }
        out.push(b);
        a += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn put_cstr(m: &mut Mem, a: u32, s: &[u8]) {
    for (k, &b) in s.iter().enumerate() {
        m.w8(a + k as u32, b);
    }
    m.w8(a + s.len() as u32, 0);
}

impl Runtime {
    /// Build the memory the engine runs in: its own image, a heap and a
    /// stack, with the host's data symbols filled in.
    pub fn new(lib_dir: impl Into<PathBuf>, data_dir: impl Into<PathBuf>) -> (Runtime, Mem) {
        Runtime::with(lib_dir, data_dir, &[])
    }

    /// Run entirely from a voice tree compiled into the binary.
    ///
    /// `files` maps guest paths (`/loq/data/...`, `/loq/lib/...`) to their
    /// bytes. Nothing is read from disk, so the process needs no voice tree
    /// beside it and no install step.
    pub fn built_in(files: &'static [(&'static str, &'static [u8])]) -> (Runtime, Mem) {
        Runtime::with("", "", files)
    }

    pub fn with(
        lib_dir: impl Into<PathBuf>,
        data_dir: impl Into<PathBuf>,
        files: &'static [(&'static str, &'static [u8])],
    ) -> (Runtime, Mem) {
        let (lib_dir, data_dir) = (lib_dir.into(), data_dir.into());
        let mut m = Mem::with_image(&eng::regions());
        // `zeroed`, not `map`: the heap is 96 MB and the stack 8, and a `Vec`
        // of them would be allocated, zeroed, copied in and dropped for
        // nothing. It matters most on a small board.
        m.zeroed(HEAP, HEAP_SIZE);
        m.zeroed(STACK_TOP - STACK_SIZE, STACK_SIZE);

        let mut overlays = HashMap::new();
        // The engine reads its paths out of a session file rather than being
        // told them, so one is synthesised here. No licence line: the ARM
        // engine needs none, verified in `loqrs`.
        overlays.insert(
            GUEST_SESSION.to_string(),
            format!(
                "\"DataPath\" = \"{GUEST_DATA}\"\n\
             \"LibraryPath\" = \"{GUEST_LIB}\"\n\
             \"LogFile\" = \"stderr\"\n"
            )
            .into_bytes(),
        );

        let mut rt = Runtime {
            heap: Heap::new(),
            files: HashMap::new(),
            dirs: HashMap::new(),
            errno: 0,
            dl_err: 0,
            mmap_next: MMAP_BASE,
            // An empty directory means "do not look at the filesystem at
            // all", which is what `built_in` wants: with a mount of `""`,
            // a path that is not built in would be read relative to the
            // working directory, and a binary that ships its own voice
            // tree must not pick up a file from wherever it was run.
            mounts: if lib_dir.as_os_str().is_empty() {
                Vec::new()
            } else {
                vec![
                    (GUEST_LIB.to_string(), lib_dir.clone()),
                    (GUEST_DATA.to_string(), data_dir),
                    (GUEST_ROOT.to_string(), lib_dir),
                ]
            },
            overlays,
            built_in: files,
            sinks: HashMap::from([(GUEST_OUT.to_string(), Vec::new())]),
            log: Vec::new(),
            seed: 1,
            deferred: Vec::new(),
            cond_waits: 0,
            trace: std::env::var_os("LOQNG_TRACE").is_some(),
            sched: sched::Sched::new(),
            workers: Vec::new(),
        };
        rt.errno = rt.heap.alloc(4);
        rt.dl_err = rt.heap.alloc(64);
        put_cstr(&mut m, rt.dl_err, b"cannot open shared object");

        // `stdin`/`stdout`/`stderr` are pointers the engine dereferences, so
        // each needs a FILE object behind it.
        for (name, at) in eng::HOSTDATA {
            if matches!(*name, "stdin" | "stdout" | "stderr") {
                let f = rt.heap.alloc(8);
                rt.files.insert(
                    f,
                    Open {
                        guest: (*name).to_string(),
                        real: None,
                        sink: false,
                        is_file: false,
                        data: Vec::new(),
                        pos: 0,
                        write: *name != "stdin",
                        dirty: false,
                    },
                );
                m.w32(*at, f);
            }
        }
        (rt, m)
    }

    pub fn sp(&self) -> u32 {
        (STACK_TOP - 64) & !7
    }

    /// Call a translated function by address, ARM calling convention.
    pub fn call(&mut self, m: &mut Mem, at: u32, args: &[u32]) -> u32 {
        let mut c = Cpu::default();
        let extra = args.len().saturating_sub(4) as u32;
        let sp = (self.sp() - 4 * extra) & !7;
        for (k, &v) in args.iter().skip(4).enumerate() {
            m.w32(sp + 4 * k as u32, v);
        }
        for (k, &v) in args.iter().take(4).enumerate() {
            c.r[k] = v;
        }
        c.r[13] = sp;
        c.r[14] = 0xDEAD_BEEF;
        if !eng::call_addr(at, &mut c, m, self) {
            panic!("no translated function at 0x{at:08x}");
        }
        c.r[0]
    }

    /// Lend the baton to the worker threads until they block again.
    ///
    /// `ttsRead` queues an utterance and returns; the rendering happens on
    /// the engine's own worker. This is the caller saying "your turn".
    pub fn pump(&mut self) {
        let me = sched::tid();
        while self.sched.hand_off(me) {}
    }

    /// Run every module constructor, once, before anything else.
    pub fn init(&mut self, m: &mut Mem) {
        for &a in eng::INIT {
            self.call(m, a, &[]);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<u32> {
        eng::EXPORTS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, a)| *a)
    }

    /// Copy a Rust string into fresh guest memory.
    pub fn cstring(&mut self, m: &mut Mem, s: &str) -> u32 {
        let a = self.heap.alloc(s.len() as u32 + 1);
        put_cstr(m, a, s.as_bytes());
        a
    }

    pub fn alloc(&mut self, n: u32) -> u32 {
        self.heap.alloc(n)
    }

    /// A guest path, normalised against the working directory the engine is
    /// given (`/loq`).
    fn norm(&self, p: &str) -> String {
        let p = p.replace('\\', "/");
        let mut out = Vec::new();
        let full = if p.starts_with('/') {
            p.clone()
        } else {
            format!("{GUEST_ROOT}/{p}")
        };
        for part in full.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    out.pop();
                }
                other => out.push(other.to_string()),
            }
        }
        format!("/{}", out.join("/"))
    }

    /// Where a guest path lands on the real filesystem, if anywhere.
    fn real(&self, guest: &str) -> Option<PathBuf> {
        let mut best: Option<(usize, PathBuf)> = None;
        for (prefix, dir) in &self.mounts {
            if guest == prefix || guest.starts_with(&format!("{prefix}/")) {
                let rest = guest[prefix.len()..].trim_start_matches('/');
                let cand = if rest.is_empty() {
                    dir.clone()
                } else {
                    dir.join(rest)
                };
                if best.as_ref().is_none_or(|(n, _)| prefix.len() > *n) {
                    best = Some((prefix.len(), cand));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// The bytes of a built-in file, if the binary carries one for this path.
    fn stored(&self, guest: &str) -> Option<&'static [u8]> {
        self.built_in
            .iter()
            .find(|(p, _)| *p == guest)
            .map(|(_, b)| *b)
    }

    /// What a built-in directory contains: the one path component that
    /// follows `guest` in every built-in file under it, deduplicated. A
    /// directory is not stored as such, so it is recovered from the paths.
    fn stored_dir(&self, guest: &str) -> Vec<String> {
        let prefix = format!("{}/", guest.trim_end_matches('/'));
        let mut names: Vec<String> = Vec::new();
        for (p, _) in self.built_in {
            let Some(rest) = p.strip_prefix(&prefix) else {
                continue;
            };
            let name = rest.split('/').next().unwrap_or(rest).to_string();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        names
    }

    /// Everything the engine wrote to a captured path, whether or not it has
    /// closed the file. It keeps the audio file open across utterances, so
    /// waiting for a flush would report nothing.
    pub fn sink(&self, guest: &str) -> Vec<u8> {
        if let Some(o) = self.files.values().find(|o| o.sink && o.guest == guest) {
            if !o.data.is_empty() {
                return o.data.clone();
            }
        }
        self.sinks.get(guest).cloned().unwrap_or_default()
    }

    /// Take everything written to a captured path and reset it, so the next
    /// utterance through the same engine is measured on its own.
    pub fn take_sink(&mut self, guest: &str) -> Vec<u8> {
        let mut out = self
            .sinks
            .insert(guest.to_string(), Vec::new())
            .unwrap_or_default();
        for o in self.files.values_mut() {
            if o.sink && o.guest == guest {
                out.append(&mut o.data);
                o.pos = 0;
            }
        }
        out
    }

    fn flush(&mut self, f: u32) {
        let Some(o) = self.files.get_mut(&f) else {
            return;
        };
        if !o.dirty {
            return;
        }
        o.dirty = false;
        if o.sink {
            let data = o.data.clone();
            let guest = o.guest.clone();
            self.sinks.insert(guest, data);
        } else if let Some(p) = o.real.clone() {
            let _ = std::fs::write(p, &o.data);
        }
    }
}

/// `arg(i)` in the ARM procedure call standard: r0-r3, then the caller's
/// stack, which is exactly where the translated `bl` left it.
fn arg(c: &Cpu, m: &Mem, i: usize) -> u32 {
    if i < 4 {
        c.r[i]
    } else {
        m.r32(c.r[13] + 4 * (i as u32 - 4))
    }
}

/// A `double` argument. ARM OABI puts the most significant word first, in an
/// even-numbered register pair — a fact `loqrs` paid for and this inherits.
fn argf(c: &Cpu, m: &Mem, i: usize) -> f64 {
    let hi = arg(c, m, i) as u64;
    let lo = arg(c, m, i + 1) as u64;
    f64::from_bits((hi << 32) | lo)
}

impl Host for Runtime {
    fn indirect(&mut self, target: u32, c: &mut Cpu, m: &mut Mem) {
        if !eng::call_addr(target, c, m, self) {
            panic!(
                "indirect call to 0x{target:08x}, which is not a \
                    translated function"
            );
        }
    }

    fn import(&mut self, name: &str, c: &mut Cpu, m: &mut Mem) {
        // Arguments are snapshotted so nothing below holds a borrow of `c`
        // or `m` while the handler needs them mutably. Twenty words covers
        // every variadic call the engine makes.
        let av: [u32; 20] = std::array::from_fn(|i| arg(c, m, i));
        let a = |i: usize| av[i];
        macro_rules! ret {
            ($v:expr) => {{
                c.r[0] = $v;
                return;
            }};
        }

        match name {
            // ---- memory -------------------------------------------------
            "malloc" | "ELQmalloc" => {
                let v = self.heap.alloc(a(0));
                ret!(v);
            }
            "calloc" => {
                let n = a(0).saturating_mul(a(1));
                let p = self.heap.alloc(n);
                for k in 0..n {
                    m.w8(p + k, 0);
                }
                ret!(p);
            }
            "realloc" | "ELQrealloc" => {
                let (old, n) = (a(0), a(1));
                if old == 0 {
                    let v = self.heap.alloc(n);
                    ret!(v);
                }
                let keep = self.heap.size(old).min(n);
                let p = self.heap.alloc(n);
                for k in 0..keep {
                    let b = m.r8(old + k);
                    m.w8(p + k, b);
                }
                self.heap.free(old);
                ret!(p);
            }
            "free" | "ELQfree" => self.heap.free(a(0)),
            "memcpy" | "memmove" => {
                let (d, s, n) = (a(0), a(1), a(2));
                let tmp = m.read(s, n as usize);
                m.write(d, &tmp);
                ret!(d);
            }
            "memset" => {
                let (d, v, n) = (a(0), a(1) as u8, a(2));
                for k in 0..n {
                    m.w8(d + k, v);
                }
                ret!(d);
            }
            "memcmp" => {
                let (x, y, n) = (a(0), a(1), a(2));
                let mut r = 0i32;
                for k in 0..n {
                    let (p, q) = (m.r8(x + k), m.r8(y + k));
                    if p != q {
                        r = p as i32 - q as i32;
                        break;
                    }
                }
                ret!(r as u32);
            }

            // ---- strings ------------------------------------------------
            "strlen" => {
                let mut k = 0;
                while m.r8(a(0) + k) != 0 {
                    k += 1;
                }
                ret!(k);
            }
            "strcpy" => {
                let (d, s) = (a(0), a(1));
                let v = cstr(m, s);
                put_cstr(m, d, v.as_bytes());
                ret!(d);
            }
            "strncpy" => {
                let (d, s, n) = (a(0), a(1), a(2));
                let mut end = false;
                for k in 0..n {
                    let b = if end { 0 } else { m.r8(s + k) };
                    end |= b == 0;
                    m.w8(d + k, b);
                }
                ret!(d);
            }
            "strcat" | "ELQscscat" => {
                let (d, s) = (a(0), a(1));
                let mut k = 0;
                while m.r8(d + k) != 0 {
                    k += 1;
                }
                let v = cstr(m, s);
                put_cstr(m, d + k, v.as_bytes());
                ret!(d);
            }
            "strncat" => {
                let (d, s, n) = (a(0), a(1), a(2));
                let mut k = 0;
                while m.r8(d + k) != 0 {
                    k += 1;
                }
                let mut j = 0;
                while j < n && m.r8(s + j) != 0 {
                    m.w8(d + k + j, m.r8(s + j));
                    j += 1;
                }
                m.w8(d + k + j, 0);
                ret!(d);
            }
            "strcmp" | "ELQstricmp" => {
                let (x, y) = (cstr(m, a(0)), cstr(m, a(1)));
                let (x, y) = if name == "strcmp" {
                    (x, y)
                } else {
                    (x.to_lowercase(), y.to_lowercase())
                };
                let r = match x.as_bytes().cmp(y.as_bytes()) {
                    std::cmp::Ordering::Less => -1i32,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                ret!(r as u32);
            }
            "strncmp" => {
                let (x, y, n) = (a(0), a(1), a(2));
                let mut r = 0i32;
                for k in 0..n {
                    let (p, q) = (m.r8(x + k), m.r8(y + k));
                    if p != q || p == 0 {
                        r = p as i32 - q as i32;
                        break;
                    }
                }
                ret!(r as u32);
            }
            "strchr" | "strrchr" => {
                let (s, ch) = (a(0), a(1) as u8);
                let v = cstr(m, s);
                let at = if name == "strchr" {
                    v.as_bytes().iter().position(|&b| b == ch)
                } else {
                    v.as_bytes().iter().rposition(|&b| b == ch)
                };
                ret!(match at {
                    Some(k) => s + k as u32,
                    None if ch == 0 => s + v.len() as u32,
                    None => 0,
                });
            }
            "strstr" => {
                let (h, nd) = (a(0), a(1));
                let (hs, ns) = (cstr(m, h), cstr(m, nd));
                ret!(match hs.find(&ns) {
                    Some(k) => h + k as u32,
                    None => 0,
                });
            }
            "strpbrk" => {
                let (s, set) = (a(0), a(1));
                let (v, w) = (cstr(m, s), cstr(m, set));
                ret!(match v.bytes().position(|b| w.as_bytes().contains(&b)) {
                    Some(k) => s + k as u32,
                    None => 0,
                });
            }
            "strspn" | "strcspn" => {
                let (v, w) = (cstr(m, a(0)), cstr(m, a(1)));
                let want = name == "strspn";
                let n = v
                    .bytes()
                    .take_while(|b| w.as_bytes().contains(b) == want)
                    .count();
                ret!(n as u32);
            }
            "__strtol_internal" | "__strtoul_internal" => {
                let s = cstr(m, a(0));
                let base = a(2);
                let t = s.trim_start();
                let lead = s.len() - t.len();
                let (t, neg) = match t.strip_prefix('-') {
                    Some(r) => (r, true),
                    None => (t.strip_prefix('+').unwrap_or(t), false),
                };
                let radix = if base == 0 { 10 } else { base };
                let digits = t
                    .bytes()
                    .take_while(|b| (*b as char).is_digit(radix))
                    .count();
                let v = i64::from_str_radix(&t[..digits], radix).unwrap_or(0);
                if a(1) != 0 {
                    let used = if digits == 0 {
                        0
                    } else {
                        lead + digits + neg as usize
                    };
                    m.w32(a(1), a(0) + used as u32);
                }
                ret!(if neg { -v as u32 } else { v as u32 });
            }
            "__strtod_internal" => {
                let s = cstr(m, a(0));
                let t = s.trim_start();
                let n = t
                    .bytes()
                    .take_while(|b| b.is_ascii_digit() || b"+-.eE".contains(b))
                    .count();
                let v: f64 = t[..n].parse().unwrap_or(0.0);
                if a(1) != 0 {
                    m.w32(a(1), a(0) + (s.len() - t.len() + n) as u32);
                }
                c.fp[0] = v;
            }

            // ---- files --------------------------------------------------
            "fopen" | "ELQfopen" => {
                let guest = self.norm(&cstr(m, a(0)));
                let mode = cstr(m, a(1));
                let write = mode.starts_with(['w', 'a']);
                let sink = self.sinks.contains_key(&guest);
                let real = self.real(&guest);
                let mut from = "disk";
                let data = if let Some(o) = self.overlays.get(&guest) {
                    from = "overlay";
                    o.clone()
                } else if sink || (write && mode.starts_with('w')) {
                    from = "sink";
                    Vec::new()
                } else if let Some(b) = self.stored(&guest) {
                    from = "built in";
                    b.to_vec()
                } else {
                    match real.as_ref().map(std::fs::read) {
                        Some(Ok(d)) => d,
                        _ if !write => ret!(0),
                        _ => Vec::new(),
                    }
                };
                let f = self.heap.alloc(8);
                let pos = if mode.starts_with('a') { data.len() } else { 0 };
                if self.trace {
                    eprintln!(
                        "[open] {guest:?} {mode:?} -> {} bytes ({from}){}",
                        data.len(),
                        if real.as_ref().is_some_and(|p| p.exists())
                            || self.overlays.contains_key(&guest)
                            || self.stored(&guest).is_some()
                            || sink
                        {
                            ""
                        } else {
                            "  MISSING"
                        }
                    );
                }
                self.files.insert(
                    f,
                    Open {
                        guest,
                        real,
                        sink,
                        is_file: true,
                        data,
                        pos,
                        write,
                        dirty: write,
                    },
                );
                ret!(f);
            }
            "fclose" | "ELQfclose" => {
                self.flush(a(0));
                self.files.remove(&a(0));
                self.heap.free(a(0));
                ret!(0);
            }
            "fread" | "ELQfread" => {
                let (p, sz, n, f) = (a(0), a(1), a(2), a(3));
                let want = (sz * n) as usize;
                let Some(o) = self.files.get_mut(&f) else {
                    ret!(0)
                };
                let got = want.min(o.data.len().saturating_sub(o.pos));
                let bytes = o.data[o.pos..o.pos + got].to_vec();
                o.pos += got;
                m.write(p, &bytes);
                ret!(if sz == 0 { 0 } else { got as u32 / sz });
            }
            "fwrite" | "ELQfwrite" => {
                let (p, sz, n, f) = (a(0), a(1), a(2), a(3));
                let bytes = m.read(p, (sz * n) as usize);
                match self.files.get_mut(&f) {
                    Some(o) if o.is_file => {
                        if o.pos + bytes.len() > o.data.len() {
                            o.data.resize(o.pos + bytes.len(), 0);
                        }
                        o.data[o.pos..o.pos + bytes.len()].copy_from_slice(&bytes);
                        o.pos += bytes.len();
                        o.dirty = true;
                    }
                    _ => self.log.extend_from_slice(&bytes),
                }
                ret!(n);
            }
            "fgetc" => {
                let Some(o) = self.files.get_mut(&a(0)) else {
                    ret!(!0)
                };
                if o.pos >= o.data.len() {
                    ret!(!0);
                }
                o.pos += 1;
                ret!(o.data[o.pos - 1] as u32);
            }
            "fgets" | "ELQfgets" => {
                let (p, n, f) = (a(0), a(1), a(2));
                let Some(o) = self.files.get_mut(&f) else {
                    ret!(0)
                };
                if o.pos >= o.data.len() {
                    ret!(0);
                }
                let mut line = Vec::new();
                while o.pos < o.data.len() && line.len() + 1 < n as usize {
                    let b = o.data[o.pos];
                    o.pos += 1;
                    line.push(b);
                    if b == b'\n' {
                        break;
                    }
                }
                put_cstr(m, p, &line);
                ret!(p);
            }
            "fputc" | "putchar" => {
                let b = a(0) as u8;
                let f = if name == "putchar" { 0 } else { a(1) };
                match self.files.get_mut(&f) {
                    Some(o) if o.is_file => {
                        o.data.push(b);
                        o.pos += 1;
                        o.dirty = true;
                    }
                    _ => self.log.push(b),
                }
                ret!(a(0));
            }
            "fputs" | "ELQfputs" => {
                let s = cstr(m, a(0));
                let f = a(1);
                match self.files.get_mut(&f) {
                    Some(o) if o.is_file => {
                        o.data.extend_from_slice(s.as_bytes());
                        o.pos = o.data.len();
                        o.dirty = true;
                    }
                    _ => self.log.extend_from_slice(s.as_bytes()),
                }
                ret!(0);
            }
            "fseek" | "ELQfseek" => {
                let (f, off, whence) = (a(0), a(1) as i32, a(2));
                if let Some(o) = self.files.get_mut(&f) {
                    let base = match whence {
                        1 => o.pos as i64,
                        2 => o.data.len() as i64,
                        _ => 0,
                    };
                    o.pos = (base + off as i64).clamp(0, o.data.len() as i64) as usize;
                }
                ret!(0);
            }
            "ftell" | "ELQftell" => {
                ret!(self.files.get(&a(0)).map_or(0, |o| o.pos as u32));
            }
            "rewind" | "ELQrewind" => {
                if let Some(o) = self.files.get_mut(&a(0)) {
                    o.pos = 0;
                }
            }
            "feof" => {
                let v = self.files.get(&a(0)).is_some_and(|o| o.pos >= o.data.len());
                ret!(v as u32);
            }
            // The engine maps files it opened with `fopen`, so the descriptor
            // has to be something `mmap` can find the open file by. The FILE
            // pointer itself is exactly that.
            "fileno" => ret!(a(0)),
            "fflush" => {
                self.flush(a(0));
                ret!(0);
            }
            "remove" | "ELQremove" => {
                let g = self.norm(&cstr(m, a(0)));
                if let Some(p) = self.real(&g) {
                    let _ = std::fs::remove_file(p);
                }
                ret!(0);
            }

            // ---- formatted i/o ------------------------------------------
            "printf" | "fprintf" | "sprintf" | "snprintf" => {
                let (fmt_i, out) = match name {
                    "printf" => (0usize, None),
                    "fprintf" => (1, None),
                    "sprintf" => (1, Some(a(0))),
                    _ => (2, Some(a(0))),
                };
                let text = {
                    let mut i = fmt_i + 1;
                    let mut next = || {
                        let v = av[i.min(av.len() - 1)];
                        i += 1;
                        v
                    };
                    cfmt::format(m, av[fmt_i], &mut next)
                };
                match out {
                    Some(p) => {
                        let cap = if name == "snprintf" {
                            (a(1) as usize).saturating_sub(1)
                        } else {
                            text.len()
                        };
                        put_cstr(m, p, &text[..text.len().min(cap)]);
                    }
                    None => {
                        let f = if name == "fprintf" { a(0) } else { 0 };
                        match self.files.get_mut(&f) {
                            Some(o) if o.is_file => {
                                o.data.extend_from_slice(&text);
                                o.pos = o.data.len();
                                o.dirty = true;
                            }
                            _ => self.log.extend_from_slice(&text),
                        }
                    }
                }
                ret!(text.len() as u32);
            }
            "sscanf" => {
                let input = cstr(m, a(0));
                let fmt = cstr(m, a(1));
                let mut i = 2usize;
                let mut next = || {
                    let v = av[i.min(av.len() - 1)];
                    i += 1;
                    v
                };
                let n = cfmt::scan(m, &input, &fmt, &mut next);
                ret!(n);
            }

            // ---- directories --------------------------------------------
            "opendir" => {
                let g = self.norm(&cstr(m, a(0)));
                let mut names: Vec<String> = vec![".".into(), "..".into()];
                match self.real(&g).map(std::fs::read_dir) {
                    Some(Ok(rd)) => names.extend(
                        rd.filter_map(|e| e.ok())
                            .map(|e| e.file_name().to_string_lossy().into_owned()),
                    ),
                    _ => {
                        let stored = self.stored_dir(&g);
                        if stored.is_empty() {
                            ret!(0);
                        }
                        names.extend(stored);
                    }
                }
                names.sort();
                let h = self.heap.alloc(8);
                let ent = self.heap.alloc(DIRENT);
                self.dirs.insert(h, Dir { names, at: 0, ent });
                ret!(h);
            }
            "readdir" => {
                let Some(d) = self.dirs.get_mut(&a(0)) else {
                    ret!(0)
                };
                if d.at >= d.names.len() {
                    ret!(0);
                }
                let name = d.names[d.at].clone();
                let ent = d.ent;
                d.at += 1;
                // struct dirent: d_ino, d_off, d_reclen, d_type, d_name[].
                for k in 0..DIRENT {
                    m.w8(ent + k, 0);
                }
                put_cstr(m, ent + 11, name.as_bytes());
                ret!(ent);
            }
            "closedir" => {
                self.dirs.remove(&a(0));
                ret!(0);
            }

            // ---- the dynamic loader -------------------------------------
            //
            // Every module is already here, so this is a lookup, not a load.
            "dlopen" => {
                let want = cstr(m, a(0));
                let base = want.rsplit(['/', '\\']).next().unwrap_or(&want);
                let h = match eng::MODULES.iter().position(|(n, _, _)| *n == base) {
                    Some(k) => 0x9000_0000 + k as u32,
                    None => 0,
                };
                if self.trace {
                    eprintln!(
                        "[dlopen] {want:?} -> 0x{h:08x}{}",
                        if h == 0 { "  NOT BUILT IN" } else { "" }
                    );
                }
                ret!(h);
            }
            "dlsym" => {
                let name = cstr(m, a(1));
                let at = self.lookup(&name).unwrap_or(0);
                if self.trace && at == 0 {
                    eprintln!("[dlsym] {name:?} is not exported by any module");
                }
                ret!(at);
            }
            "dlclose" => ret!(0),
            "dlerror" => ret!(self.dl_err),

            // ---- mapped files -------------------------------------------
            //
            // `loqrs` lesson: the engine maps the 21 MB bank and then reads
            // it at absolute offsets. A map that honours the file's current
            // position instead yields zeros, and the failure surfaces far
            // away as a null decoder.
            "mmap" => {
                let (len, f, off) = (a(1), a(4), a(5));
                let bytes = {
                    let Some(o) = self.files.get(&f) else {
                        if self.trace {
                            eprintln!("[mmap] no open file for fd 0x{f:08x}");
                        }
                        ret!(!0)
                    };
                    let start = off as usize;
                    let mut bytes = vec![0u8; len as usize];
                    if start < o.data.len() {
                        let n = (len as usize).min(o.data.len() - start);
                        bytes[..n].copy_from_slice(&o.data[start..start + n]);
                    }
                    bytes
                };
                let at = self.mmap_next;
                self.mmap_next += ((len + 0xFFF) & !0xFFF).max(0x1000);
                if self.trace {
                    eprintln!("[mmap] fd 0x{f:08x} +{off} len {len} -> 0x{at:08x}");
                }
                m.map(at, bytes);
                ret!(at);
            }
            "munmap" => ret!(0),

            // ---- odds and ends ------------------------------------------
            "__errno_location" => ret!(self.errno),
            "time" => {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs() as u32);
                if a(0) != 0 {
                    m.w32(a(0), t);
                }
                ret!(t);
            }
            "gettimeofday" => {
                if a(0) != 0 {
                    let d = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default();
                    m.w32(a(0), d.as_secs() as u32);
                    m.w32(a(0) + 4, d.subsec_micros());
                }
                ret!(0);
            }
            "localtime_r" | "gmtime_r" => ret!(a(1)),
            "nanosleep" => ret!(0),
            "getenv" => ret!(0),
            "getpid" => ret!(1000),
            "gethostname" => {
                put_cstr(m, a(0), b"loqng");
                ret!(0);
            }
            "rand" => {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                ret!((self.seed >> 16) & 0x7FFF);
            }
            "srand" => self.seed = a(0),
            "abort" => panic!("the engine called abort()"),
            "exit" | "_exit" => panic!("the engine called exit({})", a(0)),
            "__divsi3" => {
                let (x, y) = (a(0) as i32, a(1) as i32);
                ret!(if y == 0 { 0 } else { x.wrapping_div(y) as u32 });
            }
            "__udivsi3" => ret!(if a(1) == 0 { 0 } else { a(0) / a(1) }),
            "__modsi3" => {
                let (x, y) = (a(0) as i32, a(1) as i32);
                ret!(if y == 0 { 0 } else { x.wrapping_rem(y) as u32 });
            }
            "__umodsi3" => ret!(if a(1) == 0 { 0 } else { a(0) % a(1) }),
            "__gmon_start__" | "_Jv_RegisterClasses" => ret!(0),

            // System V shared memory and semaphores are only used for a
            // multi-process mode this never enters.
            "shmget" | "semget" => ret!(!0),
            "shmat" => ret!(!0),
            "shmdt" | "shmctl" | "semop" | "semctl" => ret!(0),

            // ---- threads ------------------------------------------------
            //
            // The engine creates one worker ("Sintesi"). Locks and condition
            // variables are bookkeeping only while a single thread runs, so
            // they start here as the trivial implementation; `deferred`
            // records the worker so the scheduler can be added exactly when
            // something proves it is needed.
            "pthread_mutex_init"
            | "pthread_mutex_destroy"
            | "pthread_mutex_lock"
            | "pthread_mutex_trylock"
            | "pthread_mutex_unlock"
            | "pthread_cond_init"
            | "pthread_cond_destroy"
            | "pthread_detach" => ret!(0),
            "pthread_cond_signal" | "pthread_cond_broadcast" => {
                self.sched.cond_signal(a(0));
                ret!(0);
            }
            "pthread_cond_wait" | "pthread_cond_timedwait" => {
                self.cond_waits += 1;
                let me = sched::tid();
                if !self.sched.cond_wait(me, a(0)) && self.trace {
                    eprintln!(
                        "[sched] t{me} waited on 0x{:08x} with nothing \
                               else runnable",
                        a(0)
                    );
                }
                ret!(0);
            }
            "pthread_self" => ret!(1 + sched::tid() as u32),
            "pthread_create" => {
                // r0 = &pthread_t, r2 = start routine, r3 = argument.
                let (entry, arg) = (a(2), a(3));
                let k = self.sched.enrol();
                if a(0) != 0 {
                    m.w32(a(0), 1 + k as u32);
                }
                self.deferred.push((entry, arg));

                // The stack has to live in guest memory, below the main
                // thread's, and the worker reaches the engine through raw
                // pointers; see this module's safety note.
                let top = WORKER_STACK_TOP - (k as u32 - 1) * WORKER_STACK;
                m.zeroed(top - WORKER_STACK, WORKER_STACK);
                let ptrs = Ptrs {
                    rt: self as *mut Runtime,
                    mem: m as *mut Mem,
                };
                if self.trace {
                    eprintln!(
                        "[sched] thread {k} at 0x{entry:08x}, stack \
                               0x{:08x}",
                        top
                    );
                }
                let s = self.sched.clone();
                let h = std::thread::Builder::new()
                    .name(format!("guest-{k}"))
                    .stack_size(16 << 20)
                    .spawn(move || {
                        let p = ptrs;
                        sched::set_tid(k);
                        if !s.wait_turn(k) {
                            return;
                        }
                        let rt = unsafe { &mut *p.rt };
                        let mem = unsafe { &mut *p.mem };
                        let mut c = Cpu::default();
                        c.r[0] = arg;
                        c.r[13] = (top - 64) & !7;
                        c.r[14] = 0xDEAD_BEEF;
                        if !eng::call_addr(entry, &mut c, mem, rt) {
                            panic!(
                                "thread start routine 0x{entry:08x} is \
                                    not translated"
                            );
                        }
                        s.finish(k);
                    })
                    .expect("spawning a guest thread");
                self.workers.push(h);
                ret!(0);
            }
            "pthread_join" => {
                let me = sched::tid();
                let other = a(0).saturating_sub(1) as usize;
                while !self.sched.finished(other) {
                    if !self.sched.hand_off(me) {
                        break;
                    }
                }
                ret!(0);
            }

            // ---- maths --------------------------------------------------
            //
            // APCS-GNU/FPA: doubles arrive in a register pair, high word
            // first, and the result goes back in f0.
            "sqrt" => c.fp[0] = argf(c, m, 0).sqrt(),
            "sin" => c.fp[0] = argf(c, m, 0).sin(),
            "cos" => c.fp[0] = argf(c, m, 0).cos(),
            "tan" => c.fp[0] = argf(c, m, 0).tan(),
            "atan" => c.fp[0] = argf(c, m, 0).atan(),
            "atan2" => c.fp[0] = argf(c, m, 0).atan2(argf(c, m, 2)),
            "exp" => c.fp[0] = argf(c, m, 0).exp(),
            "log" => c.fp[0] = argf(c, m, 0).ln(),
            "log10" => c.fp[0] = argf(c, m, 0).log10(),
            "pow" => c.fp[0] = argf(c, m, 0).powf(argf(c, m, 2)),
            "floor" => c.fp[0] = argf(c, m, 0).floor(),
            "ceil" => c.fp[0] = argf(c, m, 0).ceil(),
            "fabs" => c.fp[0] = argf(c, m, 0).abs(),
            "fmod" => c.fp[0] = argf(c, m, 0) % argf(c, m, 2),

            other => panic!(
                "the engine called `{other}`, which the runtime \
                             does not implement yet"
            ),
        }
    }
}
