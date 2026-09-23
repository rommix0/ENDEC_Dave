//! The guest's C library. Every imported symbol the Loquendo objects reference
//! is answered here; nothing of glibc is loaded into the guest.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use armemu::{Machine, Mem, Stop};

use crate::cfmt::{self, VaList};
use crate::heap::Heap;
use crate::msx;
use crate::vfs::Vfs;

const FILE_MAGIC: u32 = 0x4C_4F_51_46; // "LOQF"
const DIR_MAGIC: u32 = 0x4C_4F_51_44; // "LOQD"
const THREAD_STACK: u32 = 1024 * 1024;
const ETIMEDOUT: u32 = 110;

/// Why a thread is not currently runnable. Switching happens only at these
/// points, so execution stays deterministic.
#[derive(Clone, Debug, PartialEq)]
enum Blocked {
    No,
    Cond { cond: u32, mutex: u32, timed: bool },
    Mutex(u32),
    Join(u32),
    Finished,
}

/// `MULT16_32_Q14` as the shipped module computes it.
///
/// loqmsx was built with `ARM4_ASM`, so this is `fixed_arm4.h`'s version --
/// `smull` then a 64-bit shift -- NOT `fixed_generic.h`'s two-16x16 split.
/// Confirmed instruction-for-instruction against `loqmsx.so+0x3c98`.
#[inline(always)]
fn mult16_32_q14(x: i16, y: i32) -> i32 {
    (((y as i64).wrapping_mul(x as i64)) >> 14) as i32
}

/// `MULT16_16_P13(a,b)` = `(4096 + (i16)a * (i16)b) >> 13`.
#[inline(always)]
fn mult16_16_p13(a: i32, b: i32) -> i32 {
    4096i32.wrapping_add((a as i16 as i32).wrapping_mul(b as i16 as i32)) >> 13
}

/// Speex's `spx_cos` (libspeex/math_approx.c), inlined by the ARM compiler
/// into `lsp_to_lpc` -- which is why that routine makes no calls.
#[inline(always)]
fn spx_cos(x: i16) -> i16 {
    const K1: i32 = 8192;
    const K2: i32 = -4096;
    const K3: i32 = 340;
    const K4: i32 = -10;
    let poly = |x2: i32| {
        mult16_16_p13(
            x2,
            K2.wrapping_add(mult16_16_p13(x2, K3.wrapping_add(mult16_16_p13(K4, x2)))),
        )
    };
    if (x as i32) < 12868 {
        let x2 = mult16_16_p13(x as i32, x as i32) as i16 as i32;
        K1.wrapping_add(poly(x2)) as i16
    } else {
        let xx = (25736i32 - x as i32) as i16 as i32;
        let x2 = mult16_16_p13(xx, xx) as i16 as i32;
        (-K1).wrapping_sub(poly(x2)) as i16
    }
}

pub struct Host {
    pub heap: Heap,
    pub vfs: Vfs,
    pub trace_calls: bool,
    pub trace_branches: bool,
    /// Patch native replacements over hot guest routines as modules load.
    pub native: bool,
    /// Base addresses of the patched modules, recorded when they load so the
    /// handlers can reach module-relative tables without a lookup per call.
    native_bases: HashMap<String, u32>,
    pub dumps: Vec<String>,
    /// Guest addresses whose names should be logged when reached.
    pub api_names: HashMap<u32, String>,
    pub lib_dir: PathBuf,
    pub exited: Option<i32>,

    file_objs: HashMap<u32, usize>,
    dir_objs: HashMap<u32, usize>,
    errno_addr: u32,
    dirent_buf: u32,
    tm_buf: u32,
    dl_error: u32,
    std_vars: (u32, u32, u32),
    strtok_state: u32,
    rng: u32,
    shm: HashMap<i32, (u32, u32)>,
    next_shm: i32,
    sem: HashMap<i32, Vec<i32>>,
    next_sem: i32,
    dl_modules: HashMap<u32, usize>,
    unimplemented: Vec<String>,
    /// Reused buffers for the native codec kernel; see [`crate::msx`].
    msx: msx::Scratch,
    /// Guest word the candidate loop passes by pointer to `LoqTTS6.so+0x2aa84`.
    /// Allocated on first use; see [`Host::cat_candidates`].
    cat_scratch: u32,
    blocked: Vec<Blocked>,
    /// mutex address -> owning thread index
    mutex_owner: HashMap<u32, usize>,
    /// Set by a handler that has already positioned the PC itself.
    pc_set: bool,
    /// Thread index of each in-flight [`Host::call`]. A nested call made
    /// from a host handler runs on whatever thread trapped into it, so
    /// `run` cannot assume a halt belongs to thread 0.
    call_threads: Vec<usize>,
}

impl Host {
    pub fn new(heap_base: u32, heap_limit: u32) -> Self {
        Host {
            heap: Heap::new(heap_base, heap_limit),
            vfs: Vfs::new(),
            trace_calls: false,
            trace_branches: false,
            native: true,
            native_bases: HashMap::new(),
            dumps: Vec::new(),
            api_names: HashMap::new(),
            lib_dir: PathBuf::new(),
            exited: None,
            file_objs: HashMap::new(),
            dir_objs: HashMap::new(),
            errno_addr: 0,
            dirent_buf: 0,
            tm_buf: 0,
            dl_error: 0,
            std_vars: (0, 0, 0),
            strtok_state: 0,
            rng: 1,
            shm: HashMap::new(),
            next_shm: 1,
            sem: HashMap::new(),
            next_sem: 1,
            dl_modules: HashMap::new(),
            unimplemented: Vec::new(),
            msx: msx::Scratch::default(),
            cat_scratch: 0,
            call_threads: Vec::new(),
            blocked: vec![Blocked::No],
            mutex_owner: HashMap::new(),
            pc_set: false,
        }
    }

    /// Allocate the guest-visible objects libc exposes as data.
    pub fn init_data(&mut self, m: &mut Machine) {
        self.errno_addr = m.alloc_hostdata(4);
        self.dirent_buf = m.alloc_hostdata(280);
        self.tm_buf = m.alloc_hostdata(64);
        self.strtok_state = m.alloc_hostdata(4);

        let (i, o, e) = self.vfs.install_std();
        let mk = |m: &mut Machine, host: &mut Host, idx: usize| -> u32 {
            let obj = m.alloc_hostdata(8);
            let _ = m.mem.write_u32(obj, FILE_MAGIC);
            let _ = m.mem.write_u32(obj + 4, idx as u32);
            host.file_objs.insert(obj, idx);
            let var = m.alloc_hostdata(4);
            let _ = m.mem.write_u32(var, obj);
            var
        };
        let vin = mk(m, self, i);
        let vout = mk(m, self, o);
        let verr = mk(m, self, e);
        self.std_vars = (vin, vout, verr);

        let msg = m.alloc_hostdata(64);
        let _ = m.mem.write_cstr(msg, b"cannot open shared object");
        self.dl_error = msg;

        m.set_override("stdin", vin);
        m.set_override("stdout", vout);
        m.set_override("stderr", verr);
    }

    /// Addresses for symbols that are data rather than functions.
    pub fn resolve_data(&mut self, name: &str) -> Option<u32> {
        match name {
            "stdin" => Some(self.std_vars.0),
            "stdout" => Some(self.std_vars.1),
            "stderr" => Some(self.std_vars.2),
            _ => None,
        }
    }

    pub fn unimplemented_calls(&self) -> &[String] {
        &self.unimplemented
    }

    /// Swap a freshly loaded module's hottest routines for native ones.
    ///
    /// This hangs off `dlopen` rather than off session setup so that both the
    /// session API and the vendor driver get it — the coding module is loaded
    /// the same way either way.
    pub(crate) fn patch_native(&mut self, m: &mut Machine, idx: usize) -> Result<(), String> {
        if !self.native {
            return Ok(());
        }
        let name = m.modules[idx].name.clone();
        let base = m.modules[idx].base;
        for (module, off, call) in crate::tts::NATIVE_PATCHES {
            if *module == name {
                m.patch_hostcall(base + off, call)?;
                self.native_bases.insert(name.clone(), base);
                if self.trace_calls {
                    eprintln!("[native] {name}+0x{off:x} -> {call}");
                }
            }
        }
        Ok(())
    }

    fn native_base(&self, module: &str) -> Result<u32, String> {
        self.native_bases
            .get(module)
            .copied()
            .ok_or_else(|| format!("{module} was patched but its base was not recorded"))
    }

    /// Native stand-in for `loqmsx.so+0x2c44`, the 2x polyphase FIR that
    /// profiling put at 38.6% of all synthesis instructions.
    ///
    /// Marshals guest memory into host buffers, runs [`crate::msx::fir2x`],
    /// and writes the outputs and the updated history back.
    #[allow(clippy::too_many_arguments)]
    fn msx_fir2x(
        &mut self,
        m: &mut Machine,
        x: u32,
        coef: u32,
        out: u32,
        n: i32,
        ord: i32,
        mem: u32,
    ) -> Result<(), String> {
        let half = if n > 0 { (n / 2) as usize } else { 0 };
        let hist = msx::history_len(ord);
        let slots = msx::scratch_len(n, ord).max(half + hist);

        let s = &mut self.msx.taps;
        s.clear();
        s.resize(slots, 0);
        // The guest fills its scratch newest-first, rescaling by 2^14 with
        // rounding, then appends the saved history.
        for k in 0..half {
            let v = m
                .mem
                .read_u32(x + ((half - 1 - k) * 4) as u32)
                .map_err(|e| e.to_string())? as i32;
            s[k] = (v.wrapping_add(0x2000) >> 14) as i16;
        }
        for k in 0..hist {
            let v = m
                .mem
                .read_u32(mem + (k as u32) * 8 + 4)
                .map_err(|e| e.to_string())?;
            s[half + k] = v as u16 as i16;
        }

        let c = &mut self.msx.coef;
        c.clear();
        // The last group is read whole even when `ord` is not a multiple of
        // four, exactly as the unrolled guest loop does.
        let taps = (ord.max(0) as usize).div_ceil(4) * 4;
        for j in 0..taps {
            let w = m
                .mem
                .read_u16(coef + (j as u32) * 2)
                .map_err(|e| e.to_string())?;
            c.push(w as i16);
        }

        let o = &mut self.msx.out;
        o.clear();
        o.resize(n.max(0) as usize, 0);
        msx::fir2x(s, c, n, ord, o);

        for (i, v) in o.iter().enumerate() {
            m.mem
                .write_u32(out + (i as u32) * 4, *v as u32)
                .map_err(|e| e.to_string())?;
        }
        // The newest samples become the next call's history.
        for k in 0..hist {
            m.mem
                .write_u32(mem + (k as u32) * 8 + 4, s[k] as i32 as u32)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Native stand-in for the two all-pole synthesis filters,
    /// `loqmsx.so+0x22c4` (32-bit samples) and `+0x3cd0` (16-bit).
    ///
    /// They take the same six arguments and differ only in sample width, so
    /// the marshalling is shared. Copying in and out beats touching guest
    /// memory inside the loop, which runs `n * ord` times.
    #[allow(clippy::too_many_arguments)]
    fn msx_iir(
        &mut self,
        m: &mut Machine,
        x: u32,
        den: u32,
        out: u32,
        n: i32,
        ord: i32,
        mem: u32,
        wide: bool,
    ) -> Result<(), String> {
        let n_u = n.max(0) as usize;
        let ord_u = ord.max(0) as usize;

        let c = &mut self.msx.coef;
        c.clear();
        for j in 0..ord_u {
            let w = m
                .mem
                .read_u16(den + (j as u32) * 2)
                .map_err(|e| e.to_string())?;
            c.push(w as i16);
        }

        let mm = &mut self.msx.mem;
        mm.clear();
        for k in 0..ord_u {
            let v = m
                .mem
                .read_u32(mem + (k as u32) * 4)
                .map_err(|e| e.to_string())?;
            mm.push(v as i32);
        }

        if wide {
            let xs = &mut self.msx.input;
            xs.clear();
            for i in 0..n_u {
                let v = m
                    .mem
                    .read_u32(x + (i as u32) * 4)
                    .map_err(|e| e.to_string())?;
                xs.push(v as i32);
            }
            let o = &mut self.msx.out;
            o.clear();
            o.resize(n_u, 0);
            msx::iir32(xs, c, n, ord, o, mm);
            for (i, v) in o.iter().enumerate() {
                m.mem
                    .write_u32(out + (i as u32) * 4, *v as u32)
                    .map_err(|e| e.to_string())?;
            }
        } else {
            let xs = &mut self.msx.input16;
            xs.clear();
            for i in 0..n_u {
                let v = m
                    .mem
                    .read_u16(x + (i as u32) * 2)
                    .map_err(|e| e.to_string())?;
                xs.push(v as i16);
            }
            let o = &mut self.msx.out16;
            o.clear();
            o.resize(n_u, 0);
            msx::iir16(xs, c, n, ord, o, mm);
            for (i, v) in o.iter().enumerate() {
                m.mem
                    .write_u16(out + (i as u32) * 2, *v as u16)
                    .map_err(|e| e.to_string())?;
            }
        }

        for (k, v) in mm.iter().enumerate() {
            m.mem
                .write_u32(mem + (k as u32) * 4, *v as u32)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Run the guest until it returns to the host or faults.
    pub fn run(&mut self, m: &mut Machine) -> Result<(), String> {
        loop {
            match m.run(50_000_000) {
                Stop::Budget => continue,
                Stop::Halted => {
                    // A nested call set MAGIC_RETURN itself, so its halt is
                    // expected on whichever thread it was made from.
                    if self.call_threads.last() == Some(&m.current) {
                        return Ok(());
                    }
                    if m.current == 0 {
                        return Ok(());
                    }
                    return Err("a worker thread reached the host return address".into());
                }
                Stop::HostCall(index) => {
                    let name = m.hostcall_name(index).to_string();
                    if self.trace_calls {
                        eprintln!(
                            "[call] t{} {name}(0x{:x}, 0x{:x}, 0x{:x}, 0x{:x})",
                            m.current_tid(),
                            m.cpu.r[0],
                            m.cpu.r[1],
                            m.cpu.r[2],
                            m.cpu.r[3]
                        );
                    }
                    self.pc_set = false;
                    self.dispatch(&name, m)?;
                    if let Some(code) = self.exited {
                        return Err(format!("guest called exit({code})"));
                    }
                    if !self.pc_set {
                        m.cpu.r[15] = m.cpu.r[14];
                    }
                }
                Stop::Watch { pc, addr, value } => {
                    eprintln!(
                        "[watch] 0x{addr:08x} = 0x{value:08x} written at {}\n{}",
                        m.describe(pc),
                        m.regs()
                    );
                }
                Stop::Breakpoint(pc) => {
                    if let Some(name) = self.api_names.get(&pc).cloned() {
                        let sp = m.cpu.r[13];
                        let s4 = m.mem.read_u32(sp).unwrap_or(0);
                        let s5 = m.mem.read_u32(sp + 4).unwrap_or(0);
                        let s6 = m.mem.read_u32(sp + 8).unwrap_or(0);
                        eprintln!(
                            "[api] {name}({}, {}, {}, {} | {}, {}, {}) lr={}",
                            m.arg(m.cpu.r[0]),
                            m.arg(m.cpu.r[1]),
                            m.arg(m.cpu.r[2]),
                            m.arg(m.cpu.r[3]),
                            m.arg(s4),
                            m.arg(s5),
                            m.arg(s6),
                            m.describe(m.cpu.r[14])
                        );
                        continue;
                    }
                    eprintln!("[break] {}\n{}", m.describe(pc), m.regs());
                    for d in &self.dumps {
                        eprintln!("{}", m.dump(d));
                    }
                }
                Stop::ThreadExit => {
                    let cur = m.current;
                    m.threads[cur].finished = true;
                    m.threads[cur].retval = m.cpu.r[0];
                    self.blocked[cur] = Blocked::Finished;
                    let tid = m.threads[cur].tid;
                    if self.trace_calls {
                        eprintln!("[thread] t{tid} exited");
                    }
                    for i in 0..self.blocked.len() {
                        if self.blocked[i] == Blocked::Join(tid) {
                            self.blocked[i] = Blocked::No;
                        }
                    }
                    self.schedule(m)?;
                }
                Stop::Undefined { pc, insn } => {
                    return Err(format!(
                        "undefined instruction 0x{insn:08x} at {}\n{}",
                        m.describe(pc),
                        m.backtrace()
                    ))
                }
                Stop::Swi { pc, imm } => {
                    return Err(format!(
                        "unexpected SWI 0x{imm:x} at {}\n{}",
                        m.describe(pc),
                        m.backtrace()
                    ))
                }
                Stop::Fault { pc, fault } => {
                    return Err(format!("{fault} at {}\n{}", m.describe(pc), m.backtrace()))
                }
            }
        }
    }

    /// Let any runnable worker thread run until it blocks or finishes.
    /// Needed after an asynchronous engine call returns to us early.
    pub fn pump(&mut self, m: &mut Machine) -> Result<(), String> {
        loop {
            let next = (0..m.threads.len()).find(|&i| {
                i != m.current && self.blocked[i] == Blocked::No && !m.threads[i].finished
            });
            match next {
                None => return Ok(()),
                Some(i) => {
                    let here = m.current;
                    m.switch_to(i);
                    self.run(m)?;
                    if m.current != here {
                        m.switch_to(here);
                    }
                }
            }
        }
    }

    /// Call a guest function and return r0.
    pub fn call(&mut self, m: &mut Machine, func: u32, args: &[u32]) -> Result<u32, String> {
        let saved_sp = m.cpu.r[13];
        // setup_call overwrites lr with MAGIC_RETURN. When this call is nested
        // inside a host handler, the handler's own veneer still has to return
        // through lr afterwards, so it is saved and put back.
        let saved_lr = m.cpu.r[14];
        m.setup_call(func, args)?;
        self.call_threads.push(m.current);
        let r = self.run(m);
        self.call_threads.pop();
        r?;
        m.cpu.r[13] = saved_sp;
        m.cpu.r[14] = saved_lr;
        Ok(m.cpu.r[0])
    }

    /// Native stand-in for `loqmsx.so+0x6500`, Speex's fixed-point
    /// `lsp_to_lpc` (libspeex/lsp.c) with `spx_cos` inlined.
    ///
    /// 13.2% of remaining guest instructions. The `xp`/`xq`/`freqn` working
    /// set is ALLOC'd from the caller's scratch arena, and `stack` is passed
    /// BY VALUE, so nothing left there is observable -- only `ak` is output.
    /// That lets the whole cascade run in host memory and touch guest memory
    /// only to read `freq` and write `ak`.
    ///
    /// The `a < -32767` clamp really does assign +32767 upstream. That is an
    /// upstream bug, but the shipped module has it, so it is reproduced.
    fn spx_lsp_to_lpc(
        &mut self,
        mach: &mut Machine,
        freq: u32,
        ak: u32,
        lpcrdr: i32,
    ) -> Result<(), String> {
        const QIMP: i32 = 21;

        let n = lpcrdr.max(0) as usize;
        let m = (lpcrdr >> 1).max(0) as usize;
        let row = (lpcrdr + 3).max(0) as usize;
        if n == 0 || m == 0 {
            return Ok(());
        }

        let mut freqn = vec![0i16; n];
        for (i, f) in freqn.iter_mut().enumerate() {
            let v = mach
                .mem
                .read_u16(freq.wrapping_add((i * 2) as u32))
                .map_err(|e| e.to_string())? as i16;
            // ANGLE2X(a) = SHL16(spx_cos(a), 2), stored back into an i16.
            *f = ((spx_cos(v) as i32) << 2) as i16;
        }

        let mut xp = vec![0i32; (m + 1) * row];
        let mut xq = vec![0i32; (m + 1) * row];
        let xin: i32 = 1 << (QIMP - 1);

        for i in 0..=m {
            let b = i * row;
            xp[b + 1] = 0;
            xp[b + 2] = xin;
            xp[b + 2 + 2 * i] = xin;
            xq[b + 1] = 0;
            xq[b + 2] = xin;
            xq[b + 2 + 2 * i] = xin;
        }

        xp[row + 3] = mult16_32_q14(freqn[0], xp[2]).wrapping_neg();
        xq[row + 3] = mult16_32_q14(freqn[1], xq[2]).wrapping_neg();

        let (mut xout1, mut xout2) = (0i32, 0i32);

        for i in 1..m {
            let (c, nx) = (i * row, (i + 1) * row);
            let mut j = 1usize;
            while j < 2 * (i + 1) - 1 {
                let mult = mult16_32_q14(freqn[2 * i], xp[c + j + 1]);
                xp[nx + j + 2] = xp[c + j + 2].wrapping_sub(mult).wrapping_add(xp[c + j]);
                let mult = mult16_32_q14(freqn[2 * i + 1], xq[c + j + 1]);
                xq[nx + j + 2] = xq[c + j + 2].wrapping_sub(mult).wrapping_add(xq[c + j]);
                j += 1;
            }
            // Last column: xp[i][j+2] and xq[i][j+2] are known zero.
            let mult = mult16_32_q14(freqn[2 * i], xp[c + j + 1]);
            xp[nx + j + 2] = xp[c + j].wrapping_sub(mult);
            let mult = mult16_32_q14(freqn[2 * i + 1], xq[c + j + 1]);
            xq[nx + j + 2] = xq[c + j].wrapping_sub(mult);
        }

        let last = m * row;
        let shift = QIMP - 13;
        for j in 1..=n {
            let p = xp[last + j + 2];
            let q = xq[last + j + 2];
            let sum = p.wrapping_add(xout1).wrapping_add(q).wrapping_sub(xout2);
            let mut a = sum.wrapping_add(1 << (shift - 1)) >> shift;
            xout1 = p;
            xout2 = q;
            if a < -32767 {
                a = 32767;
            }
            if a > 32767 {
                a = 32767;
            }
            mach.mem
                .write_u16(ak.wrapping_add(((j - 1) * 2) as u32), a as u16)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Native stand-in for `loqmsx.so+0x3c98`, Speex's `signal_mul`
    /// (libspeex/filters.c) -- argument order `(x, y, scale, len)` matches.
    ///
    /// NOTE this is transcribed from the DISASSEMBLY, not from the C. Stock
    /// fixed-point Speex computes
    /// `SHL32(MULT16_32_Q14(EXTRACT16(SHR32(x[i],7)),scale),7)`, where
    /// `EXTRACT16` truncates to 16 bits and `MULT16_32_Q14` splits the product
    /// into two 16x16 multiplies. The shipped ARM code does neither: there is
    /// no `sxth`, and it uses a full 64-bit `smull` followed by a 64-bit shift.
    /// Since the shipped module is what has to be reproduced bit-for-bit, the
    /// hardware wins and the C is only a guide to intent.
    ///
    /// The original is a do-while, so `len == 0` would wrap; every call site
    /// passes a positive length (the C is a zero-trip `for`), so this uses a
    /// zero-trip loop rather than reproducing that.
    fn spx_signal_mul(
        &mut self,
        m: &mut Machine,
        x: u32,
        y: u32,
        scale: i32,
        len: i32,
    ) -> Result<(), String> {
        for i in 0..len.max(0) as u32 {
            let off = i.wrapping_mul(4);
            let v = m
                .mem
                .read_u32(x.wrapping_add(off))
                .map_err(|e| e.to_string())? as i32;
            let p = (scale as i64).wrapping_mul((v >> 7) as i64);
            let out = ((p >> 14) as i32) << 7;
            m.mem
                .write_u32(y.wrapping_add(off), out as u32)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Native stand-in for `loqmsx.so+0x14ac`, which is stock Speex 1.2beta1's
    /// `speex_bits_unpack_unsigned` from `libspeex/bits.c`.
    ///
    /// loqmsx is Speex 1.2beta1 (fixed-point) with three Loquendo changes: a
    /// per-voice LSP codebook, a per-voice innovation codebook, and no output
    /// highpass. This routine is none of them -- it is untouched upstream, so
    /// this is a transcription of published source rather than of a
    /// disassembly. It profiles at 8.6% of every remaining guest instruction
    /// because the bitstream is read one bit at a time.
    ///
    /// `SpeexBits` (include/speex/speex_bits.h) is
    /// `+0 chars, +4 nbBits, +8 charPtr, +0xC bitPtr, +0x10 owner, +0x14 overflow`,
    /// which the disassembly confirms field for field. The original stores the
    /// two cursors back on every iteration; it has no calls, so nothing can
    /// observe the intermediate values and they are written once at the end.
    fn spx_bits_unpack_unsigned(
        &mut self,
        m: &mut Machine,
        bits: u32,
        nb: i32,
    ) -> Result<u32, String> {
        use crate::cat::w;

        let chars = w(&m.mem, bits)?;
        let nb_bits = w(&m.mem, bits.wrapping_add(4))? as i32;
        let mut char_ptr = w(&m.mem, bits.wrapping_add(8))? as i32;
        let mut bit_ptr = w(&m.mem, bits.wrapping_add(0xC))? as i32;

        if (char_ptr << 3).wrapping_add(bit_ptr).wrapping_add(nb) > nb_bits {
            m.mem
                .write_u32(bits.wrapping_add(0x14), 1)
                .map_err(|e| e.to_string())?;
        }
        if w(&m.mem, bits.wrapping_add(0x14))? != 0 {
            return Ok(0);
        }

        let mut d: u32 = 0;
        for _ in 0..nb.max(0) {
            let byte = m
                .mem
                .read_u8(chars.wrapping_add(char_ptr as u32))
                .map_err(|e| e.to_string())? as u32;
            d = (d << 1) | ((byte >> (7 - bit_ptr)) & 1);
            bit_ptr += 1;
            if bit_ptr == 8 {
                bit_ptr = 0;
                char_ptr += 1;
            }
        }

        m.mem
            .write_u32(bits.wrapping_add(8), char_ptr as u32)
            .map_err(|e| e.to_string())?;
        m.mem
            .write_u32(bits.wrapping_add(0xC), bit_ptr as u32)
            .map_err(|e| e.to_string())?;
        Ok(d)
    }

    /// Native stand-in for `LoqTTS6.so+0x2b014`, the unit-selection candidate
    /// loop.
    ///
    /// With the four leaf kernels out of the way, profiling put 29.9% of every
    /// remaining guest instruction inside this one function -- by far the
    /// largest block left, and exactly the "another ~19% in the candidate loop
    /// that calls it" that [`crate::cat`]'s header predicted.
    ///
    /// Unlike the four routines already replaced this is NOT a leaf. It makes
    /// three calls per surviving candidate:
    ///
    ///   * `LoqTTS6.so+0x29f1c`, the context cost -- already native, so it is
    ///     called straight through as [`crate::cat::context_cost`] rather than
    ///     bouncing back out through the veneer.
    ///   * `LoqTTS6.so+0x2aa84` and `+0x2acb4`, still guest code, reached with
    ///     [`Host::call`]. Both are gated on a flag in `glob` AND on the
    ///     running best score, so the common iteration makes no guest call at
    ///     all -- which is why they profile at 5.2% and under 2% while the loop
    ///     body itself is 29.9%.
    ///
    /// `+0x2aa84` takes a POINTER to a caller stack slot holding `rec`, so it
    /// needs a guest-visible word; `cat_scratch` is that word, allocated once.
    ///
    /// Everything is integer and every table is in guest memory, so this is a
    /// transcription rather than a model of it. The three prunes below are the
    /// original's `ble` tests and are therefore signed.
    #[allow(clippy::too_many_arguments)]
    fn cat_candidates(
        &mut self,
        m: &mut Machine,
        glob: u32,
        sel: u32,
        chan: u32,
        cls: u32,
        best_out: u32,
        score_out: u32,
    ) -> Result<(), String> {
        use crate::cat::{b, h, w, COST_TABLE};

        let base = self.native_base("LoqTTS6.so")?;
        let table = base.wrapping_add(COST_TABLE);

        let cand = w(&m.mem, sel)?;
        let pos = h(&m.mem, glob.wrapping_add(0x774))?;
        let odd = pos & 1 != 0;
        // The class byte the cost function takes is read from the descriptor,
        // not from this function's own `cls` argument, which only gets stored
        // into the winning record.
        let cls2 = b(&m.mem, sel.wrapping_add(4))?;

        let klass = b(
            &m.mem,
            cand.wrapping_add((pos >> 1).wrapping_mul(0x10))
                .wrapping_add(6),
        )?;
        let heads = w(&m.mem, chan.wrapping_add(0x1C))?;
        let koff = klass.wrapping_mul(8);
        let hdr = heads.wrapping_add(koff);
        let pool = w(&m.mem, chan.wrapping_add(8))?;
        let count = w(&m.mem, hdr)?;
        let units = w(&m.mem, chan.wrapping_add(0x18))?;
        let tgts = w(&m.mem, chan.wrapping_add(0x10))?;

        let mut i: u32 = 0;
        while i < count {
            let arr = w(&m.mem, hdr.wrapping_add(4))?;
            let ent = arr.wrapping_add(i.wrapping_mul(4));
            let rec = pool.wrapping_add(h(&m.mem, ent)?.wrapping_mul(8));
            let idx = h(&m.mem, ent.wrapping_add(2))?;
            let head = w(&m.mem, rec)?;

            let mut unit = units.wrapping_add(idx.wrapping_add(head).wrapping_mul(0x10));
            if odd {
                unit = unit.wrapping_add(8);
            }
            // An empty half-unit is no candidate at all.
            if w(&m.mem, unit)? == 0 && w(&m.mem, unit.wrapping_add(4))? == 0 {
                i = i.wrapping_add(1);
                continue;
            }

            let tgt = tgts.wrapping_add(head.wrapping_mul(4));

            // 2 when the odd-position lookahead agrees on class, the pairwise
            // penalty clears the floor, and the following half-unit exists.
            let mut kind: u32 = 1;
            if odd {
                let a = cand.wrapping_add((pos.wrapping_add(1) >> 1).wrapping_mul(0x10));
                let e = tgt.wrapping_add(idx.wrapping_mul(4));
                if b(&m.mem, a.wrapping_add(6))? == b(&m.mem, e.wrapping_add(4))? {
                    let ti = b(&m.mem, a.wrapping_add(7))?
                        .wrapping_add(b(&m.mem, e.wrapping_add(5))?.wrapping_mul(0x10));
                    let pen = w(&m.mem, table.wrapping_add(ti.wrapping_mul(4)))? as i32;
                    if pen > 9
                        && (w(&m.mem, unit.wrapping_add(8))? != 0
                            || w(&m.mem, unit.wrapping_add(0xC))? != 0)
                    {
                        kind = 2;
                    }
                }
            }

            let mut score =
                crate::cat::context_cost(&mut m.mem, base, glob, tgt, cand, cls2, rec, idx)?;

            // The unit that already follows the previous pick costs one less
            // to keep, so continuing a run wins ties.
            if w(&m.mem, glob.wrapping_add(0x77C))? == rec
                && w(&m.mem, glob.wrapping_add(0x778))? == w(&m.mem, unit)?
            {
                score = score.wrapping_add(1);
            }

            let best = w(&m.mem, score_out.wrapping_add(0x10))? as i32;
            if score.wrapping_mul(17).wrapping_add(0x2E7C) <= best {
                i = i.wrapping_add(1);
                continue;
            }

            let bias = w(&m.mem, glob.wrapping_add(0x918))? as i32;
            let adj = score.wrapping_sub(bias);

            let a1 = if w(&m.mem, glob.wrapping_add(0x798))? == 0 {
                10i32
            } else {
                // The original passes `sp+0x40`, and the three words there are
                // contiguous: rec, unit, kind -- the same shape it later writes
                // into `best_out`. So this is a pointer to that record, not to a
                // single word, and all three have to be live before the call.
                if self.cat_scratch == 0 {
                    self.cat_scratch = m.alloc_hostdata(16);
                }
                let slot = self.cat_scratch;
                m.mem.write_u32(slot, rec).map_err(|e| e.to_string())?;
                m.mem
                    .write_u32(slot.wrapping_add(4), unit)
                    .map_err(|e| e.to_string())?;
                m.mem
                    .write_u16(slot.wrapping_add(8), kind as u16)
                    .map_err(|e| e.to_string())?;
                let f = base.wrapping_add(0x2AA84);
                self.call(m, f, &[glob, tgt, cand, slot, rec, idx])? as i32
            };

            let best = w(&m.mem, score_out.wrapping_add(0x10))? as i32;
            if a1.wrapping_mul(score.wrapping_add(0x2BC)) <= best {
                i = i.wrapping_add(1);
                continue;
            }

            let a2 = if w(&m.mem, glob.wrapping_add(0x7A0))? == 0 {
                0i32
            } else {
                let f = base.wrapping_add(0x2ACB4);
                self.call(m, f, &[glob, unit, rec, idx, tgt, cand])? as i32
            };

            let total = a1.wrapping_mul(score.wrapping_add(a2));
            let best = w(&m.mem, score_out.wrapping_add(0x10))? as i32;
            if total <= best {
                i = i.wrapping_add(1);
                continue;
            }

            m.mem
                .write_u32(best_out.wrapping_add(4), unit)
                .map_err(|e| e.to_string())?;
            m.mem
                .write_u16(best_out.wrapping_add(8), kind as u16)
                .map_err(|e| e.to_string())?;
            m.mem.write_u32(best_out, rec).map_err(|e| e.to_string())?;
            m.mem
                .write_u8(best_out.wrapping_add(0xC), cls as u8)
                .map_err(|e| e.to_string())?;
            m.mem
                .write_u32(score_out.wrapping_add(0x10), total as u32)
                .map_err(|e| e.to_string())?;
            m.mem
                .write_u32(score_out, adj as u32)
                .map_err(|e| e.to_string())?;
            m.mem
                .write_u32(score_out.wrapping_add(8), bias as u32)
                .map_err(|e| e.to_string())?;
            m.mem
                .write_u32(score_out.wrapping_add(0xC), a2 as u32)
                .map_err(|e| e.to_string())?;
            m.mem
                .write_u32(score_out.wrapping_add(4), a1 as u32)
                .map_err(|e| e.to_string())?;

            i = i.wrapping_add(1);
        }
        Ok(())
    }

    fn set_errno(&mut self, m: &mut Machine, v: i32) {
        let _ = m.mem.write_u32(self.errno_addr, v as u32);
    }

    fn file_index(&self, ptr: u32) -> Option<usize> {
        self.file_objs.get(&ptr).copied()
    }

    fn new_file_obj(&mut self, m: &mut Machine, idx: usize) -> u32 {
        let obj = m.alloc_hostdata(8);
        let _ = m.mem.write_u32(obj, FILE_MAGIC);
        let _ = m.mem.write_u32(obj + 4, idx as u32);
        self.file_objs.insert(obj, idx);
        obj
    }

    fn dispatch(&mut self, name: &str, m: &mut Machine) -> Result<(), String> {
        let r = m.cpu.r;
        let sp = m.cpu.r[13];

        macro_rules! ret {
            ($v:expr) => {{
                m.cpu.r[0] = $v as u32;
                return Ok(());
            }};
        }
        macro_rules! retf {
            ($v:expr) => {{
                m.cpu.fpa.f[0] = $v;
                return Ok(());
            }};
        }
        macro_rules! arg_f64 {
            ($i:expr) => {
                f64::from_bits(((r[$i] as u64) << 32) | r[$i + 1] as u64)
            };
        }

        match name {
            // ---------- memory ----------
            "malloc" | "ELQmalloc" => {
                let p = self.heap.malloc(&mut m.mem, r[0]);
                ret!(p)
            }
            "calloc" => {
                let p = self.heap.calloc(&mut m.mem, r[0], r[1]);
                ret!(p)
            }
            "realloc" | "ELQrealloc" => {
                let p = self.heap.realloc(&mut m.mem, r[0], r[1]);
                ret!(p)
            }
            "free" | "ELQfree" => {
                self.heap.free(r[0]);
                ret!(0)
            }
            "memcpy" | "memmove" => {
                if r[2] > 0 {
                    let src = m
                        .mem
                        .read_bytes(r[1], r[2] as usize)
                        .map_err(mem_err(name))?;
                    m.mem.write_bytes(r[0], &src).map_err(mem_err(name))?;
                }
                ret!(r[0])
            }
            "memset" => {
                m.mem
                    .fill(r[0], r[2] as usize, r[1] as u8)
                    .map_err(mem_err(name))?;
                ret!(r[0])
            }
            "memcmp" => {
                let a = m
                    .mem
                    .read_bytes(r[0], r[2] as usize)
                    .map_err(mem_err(name))?;
                let b = m
                    .mem
                    .read_bytes(r[1], r[2] as usize)
                    .map_err(mem_err(name))?;
                ret!(match a.cmp(&b) {
                    std::cmp::Ordering::Less => -1i32,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                })
            }

            // ---------- strings ----------
            "strlen" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                ret!(s.len() as u32)
            }
            "strcpy" => {
                let s = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                m.mem.write_cstr(r[0], &s).map_err(mem_err(name))?;
                ret!(r[0])
            }
            "strncpy" => {
                let s = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                let n = r[2] as usize;
                let mut buf = vec![0u8; n];
                let copy = s.len().min(n);
                buf[..copy].copy_from_slice(&s[..copy]);
                m.mem.write_bytes(r[0], &buf).map_err(mem_err(name))?;
                ret!(r[0])
            }
            "strcat" => {
                let dst = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let s = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                m.mem
                    .write_cstr(r[0] + dst.len() as u32, &s)
                    .map_err(mem_err(name))?;
                ret!(r[0])
            }
            "strncat" => {
                let dst = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let mut s = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                s.truncate(r[2] as usize);
                m.mem
                    .write_cstr(r[0] + dst.len() as u32, &s)
                    .map_err(mem_err(name))?;
                ret!(r[0])
            }
            "strcmp" | "ELQstricmp" => {
                let mut a = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let mut b = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                if name == "ELQstricmp" {
                    a.make_ascii_lowercase();
                    b.make_ascii_lowercase();
                }
                ret!(match a.cmp(&b) {
                    std::cmp::Ordering::Less => -1i32,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                })
            }
            "strncmp" => {
                let n = r[2] as usize;
                let mut a = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let mut b = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                a.truncate(n);
                b.truncate(n);
                ret!(match a.cmp(&b) {
                    std::cmp::Ordering::Less => -1i32,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                })
            }
            "strchr" | "strrchr" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let c = r[1] as u8;
                if c == 0 {
                    ret!(r[0] + s.len() as u32)
                }
                let pos = if name == "strchr" {
                    s.iter().position(|b| *b == c)
                } else {
                    s.iter().rposition(|b| *b == c)
                };
                ret!(pos.map_or(0, |p| r[0] + p as u32))
            }
            "strstr" => {
                let h = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let n = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                if n.is_empty() {
                    ret!(r[0])
                }
                let pos = h.windows(n.len()).position(|w| w == n.as_slice());
                ret!(pos.map_or(0, |p| r[0] + p as u32))
            }
            "strpbrk" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let set = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                let pos = s.iter().position(|b| set.contains(b));
                ret!(pos.map_or(0, |p| r[0] + p as u32))
            }
            "strspn" | "strcspn" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let set = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                let want = name == "strspn";
                let n = s.iter().take_while(|b| set.contains(b) == want).count();
                ret!(n as u32)
            }
            "strtok" | "ELQstrtok" => {
                let sep = m.mem.read_cstr(r[1]).map_err(mem_err(name))?;
                let mut cur = if r[0] != 0 {
                    r[0]
                } else {
                    m.mem.read_u32(self.strtok_state).unwrap_or(0)
                };
                if cur == 0 {
                    ret!(0)
                }
                loop {
                    let c = m.mem.read_u8(cur).map_err(mem_err(name))?;
                    if c == 0 {
                        let _ = m.mem.write_u32(self.strtok_state, 0);
                        ret!(0)
                    }
                    if !sep.contains(&c) {
                        break;
                    }
                    cur += 1;
                }
                let start = cur;
                loop {
                    let c = m.mem.read_u8(cur).map_err(mem_err(name))?;
                    if c == 0 {
                        let _ = m.mem.write_u32(self.strtok_state, 0);
                        ret!(start)
                    }
                    if sep.contains(&c) {
                        m.mem.write_u8(cur, 0).map_err(mem_err(name))?;
                        let _ = m.mem.write_u32(self.strtok_state, cur + 1);
                        ret!(start)
                    }
                    cur += 1;
                }
            }
            "ELQstrrev" => {
                let mut s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                s.reverse();
                m.mem.write_cstr(r[0], &s).map_err(mem_err(name))?;
                ret!(r[0])
            }

            // ---------- conversions ----------
            "__strtol_internal" | "__strtoul_internal" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let (val, used) = parse_int(&s, r[2] as i32);
                if r[1] != 0 {
                    let _ = m.mem.write_u32(r[1], r[0] + used as u32);
                }
                ret!(val as u32)
            }
            "__strtod_internal" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let (val, used) = parse_double(&s);
                if r[1] != 0 {
                    let _ = m.mem.write_u32(r[1], r[0] + used as u32);
                }
                retf!(val)
            }
            "ELQisnumber" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let ok = !s.is_empty() && s.iter().all(|c| c.is_ascii_digit());
                ret!(ok as u32)
            }
            "ELQltoa" => {
                let s = (r[0] as i32).to_string().into_bytes();
                m.mem.write_cstr(r[1], &s).map_err(mem_err(name))?;
                ret!(r[1])
            }

            // ---------- stdio ----------
            "fopen" => {
                let path = m.mem.read_cstring_lossy(r[0]).map_err(mem_err(name))?;
                let mode = m.mem.read_cstring_lossy(r[1]).map_err(mem_err(name))?;
                match self.vfs.open(&path, &mode) {
                    Some(idx) => {
                        let obj = self.new_file_obj(m, idx);
                        ret!(obj)
                    }
                    None => {
                        self.set_errno(m, 2);
                        ret!(0)
                    }
                }
            }
            "fclose" => {
                if let Some(idx) = self.file_index(r[0]) {
                    self.vfs.close(idx);
                    self.file_objs.remove(&r[0]);
                }
                ret!(0)
            }
            "fread" => {
                let want = (r[1] as usize).saturating_mul(r[2] as usize);
                let Some(idx) = self.file_index(r[3]) else {
                    ret!(0)
                };
                let data = self.vfs.read(idx, want);
                if !data.is_empty() {
                    m.mem.write_bytes(r[0], &data).map_err(mem_err(name))?;
                }
                let items = if r[1] == 0 {
                    0
                } else {
                    data.len() / r[1] as usize
                };
                ret!(items as u32)
            }
            "fwrite" => {
                let n = (r[1] as usize).saturating_mul(r[2] as usize);
                let Some(idx) = self.file_index(r[3]) else {
                    ret!(0)
                };
                let data = m.mem.read_bytes(r[0], n).map_err(mem_err(name))?;
                let w = self.vfs.write(idx, &data);
                let items = if r[1] == 0 { 0 } else { w / r[1] as usize };
                ret!(items as u32)
            }
            "fgetc" => {
                let Some(idx) = self.file_index(r[0]) else {
                    ret!(-1i32)
                };
                let d = self.vfs.read(idx, 1);
                ret!(if d.is_empty() { -1i32 } else { d[0] as i32 })
            }
            "fgets" => {
                let Some(idx) = self.file_index(r[2]) else {
                    ret!(0)
                };
                let max = r[1] as usize;
                if max == 0 {
                    ret!(0)
                }
                let mut line = Vec::new();
                while line.len() + 1 < max {
                    let d = self.vfs.read(idx, 1);
                    if d.is_empty() {
                        break;
                    }
                    line.push(d[0]);
                    if d[0] == b'\n' {
                        break;
                    }
                }
                if line.is_empty() {
                    ret!(0)
                }
                m.mem.write_cstr(r[0], &line).map_err(mem_err(name))?;
                ret!(r[0])
            }
            "fputc" | "putchar" => {
                let (ch, target) = if name == "putchar" {
                    (r[0] as u8, self.stdout_index(m))
                } else {
                    (r[0] as u8, self.file_index(r[1]))
                };
                if let Some(idx) = target {
                    self.vfs.write(idx, &[ch]);
                }
                ret!(ch as u32)
            }
            "fputs" => {
                let s = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                if let Some(idx) = self.file_index(r[1]) {
                    self.vfs.write(idx, &s);
                }
                ret!(1)
            }
            "printf" => {
                let mut va = VaList::new(1, sp);
                let out = cfmt::format(&m.cpu, &m.mem, r[0], &mut va);
                if let Some(idx) = self.stdout_index(m) {
                    self.vfs.write(idx, &out);
                }
                ret!(out.len() as u32)
            }
            "fprintf" => {
                let mut va = VaList::new(2, sp);
                let out = cfmt::format(&m.cpu, &m.mem, r[1], &mut va);
                if let Some(idx) = self.file_index(r[0]) {
                    self.vfs.write(idx, &out);
                }
                ret!(out.len() as u32)
            }
            "sprintf" => {
                let mut va = VaList::new(2, sp);
                let out = cfmt::format(&m.cpu, &m.mem, r[1], &mut va);
                m.mem.write_cstr(r[0], &out).map_err(mem_err(name))?;
                ret!(out.len() as u32)
            }
            "snprintf" => {
                let mut va = VaList::new(3, sp);
                let mut out = cfmt::format(&m.cpu, &m.mem, r[2], &mut va);
                let total = out.len();
                if r[1] > 0 {
                    out.truncate(r[1] as usize - 1);
                    m.mem.write_cstr(r[0], &out).map_err(mem_err(name))?;
                }
                ret!(total as u32)
            }
            "sscanf" => {
                let input = m.mem.read_cstr(r[0]).map_err(mem_err(name))?;
                let mut va = VaList::new(2, sp);
                let n = cfmt::scan(&m.cpu, &mut m.mem, &input, r[1], &mut va);
                ret!(n)
            }
            "fflush" => ret!(0),
            "fseek" => {
                let Some(idx) = self.file_index(r[0]) else {
                    ret!(-1i32)
                };
                let v = self.vfs.seek(idx, r[1] as i32 as i64, r[2] as i32);
                ret!(v as i32)
            }
            "ftell" => {
                let Some(idx) = self.file_index(r[0]) else {
                    ret!(-1i32)
                };
                let v = self.vfs.tell(idx);
                ret!(v as i32)
            }
            "rewind" => {
                if let Some(idx) = self.file_index(r[0]) {
                    self.vfs.seek(idx, 0, 0);
                }
                ret!(0)
            }
            "fileno" => ret!(self.file_index(r[0]).map_or(-1i32, |i| i as i32)),
            "feof" => {
                let e = self
                    .file_index(r[0])
                    .and_then(|i| self.vfs.get_mut(i).map(|f| f.eof))
                    .unwrap_or(false);
                ret!(e as u32)
            }
            "remove" | "unlink" => {
                let path = m.mem.read_cstring_lossy(r[0]).map_err(mem_err(name))?;
                ret!(if self.vfs.remove(&path) { 0 } else { -1i32 })
            }
            "ELQWriteToLog" => {
                if self.trace_calls {
                    let s = m.mem.read_cstring_lossy(r[0]).unwrap_or_default();
                    eprintln!("[loq] {s}");
                }
                ret!(0)
            }

            // ---------- directories ----------
            "opendir" => {
                let path = m.mem.read_cstring_lossy(r[0]).map_err(mem_err(name))?;
                match self.vfs.opendir(&path) {
                    Some(idx) => {
                        let obj = m.alloc_hostdata(8);
                        let _ = m.mem.write_u32(obj, DIR_MAGIC);
                        let _ = m.mem.write_u32(obj + 4, idx as u32);
                        self.dir_objs.insert(obj, idx);
                        ret!(obj)
                    }
                    None => {
                        self.set_errno(m, 2);
                        ret!(0)
                    }
                }
            }
            "readdir" => {
                let Some(idx) = self.dir_objs.get(&r[0]).copied() else {
                    ret!(0)
                };
                match self.vfs.readdir(idx) {
                    Some(entry) => {
                        let buf = self.dirent_buf;
                        let _ = m.mem.fill(buf, 280, 0);
                        let _ = m.mem.write_u32(buf, 1);
                        let _ = m.mem.write_u32(buf + 4, 0);
                        let _ = m.mem.write_u16(buf + 8, 268);
                        let _ = m.mem.write_u8(buf + 10, 0);
                        let mut bytes = entry.into_bytes();
                        bytes.truncate(255);
                        let _ = m.mem.write_cstr(buf + 11, &bytes);
                        ret!(buf)
                    }
                    None => ret!(0),
                }
            }
            "closedir" => {
                if let Some(idx) = self.dir_objs.remove(&r[0]) {
                    self.vfs.closedir(idx);
                }
                ret!(0)
            }

            // ---------- dynamic loading ----------
            "dlopen" => {
                if r[0] == 0 {
                    ret!(0xFFFF_FFFFu32)
                }
                let path = m.mem.read_cstring_lossy(r[0]).map_err(mem_err(name))?;
                match self.dlopen(m, &path) {
                    Ok(h) => ret!(h),
                    Err(e) => {
                        if self.trace_calls {
                            eprintln!("[dl] {path}: {e}");
                        }
                        ret!(0)
                    }
                }
            }
            "dlsym" => {
                let sym = m.mem.read_cstring_lossy(r[1]).map_err(mem_err(name))?;
                let addr = if r[0] == 0xFFFF_FFFF {
                    m.lookup(&sym)
                } else {
                    self.dl_modules
                        .get(&r[0])
                        .and_then(|i| m.modules[*i].symbol(&sym))
                };
                if self.trace_calls {
                    eprintln!("[dl] dlsym({sym}) -> {addr:?}");
                }
                ret!(addr.unwrap_or(0))
            }
            "dlclose" => ret!(0),
            "dlerror" => ret!(self.dl_error),

            // ---------- time ----------
            "time" => {
                let t = now_secs();
                if r[0] != 0 {
                    let _ = m.mem.write_u32(r[0], t);
                }
                ret!(t)
            }
            "gettimeofday" => {
                if r[0] != 0 {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default();
                    let _ = m.mem.write_u32(r[0], now.as_secs() as u32);
                    let _ = m.mem.write_u32(r[0] + 4, now.subsec_micros());
                }
                ret!(0)
            }
            "localtime_r" | "gmtime_r" => {
                let t = m.mem.read_u32(r[0]).unwrap_or(0);
                write_tm(&mut m.mem, r[1], t);
                ret!(r[1])
            }
            "nanosleep" => ret!(0),

            // ---------- process / environment ----------
            "getenv" => ret!(0),
            "getpid" => ret!(1000),
            "gethostname" => {
                m.mem.write_cstr(r[0], b"endec").map_err(mem_err(name))?;
                ret!(0)
            }
            "abort" => Err("guest called abort()".to_string()),
            "exit" | "_exit" => {
                self.exited = Some(r[0] as i32);
                Ok(())
            }
            "rand" => {
                self.rng = self.rng.wrapping_mul(1103515245).wrapping_add(12345);
                ret!((self.rng >> 16) & 0x7FFF)
            }
            "srand" => {
                self.rng = r[0];
                ret!(0)
            }
            "__errno_location" => ret!(self.errno_addr),

            // ---------- mapping ----------
            "mmap" => {
                let len = r[1];
                let fd = m.mem.read_u32(sp).unwrap_or(0xFFFF_FFFF);
                let off = m.mem.read_u32(sp + 4).unwrap_or(0);
                let p = self.heap.malloc(&mut m.mem, len);
                if p == 0 {
                    ret!(-1i32)
                }
                let _ = m.mem.fill(p, len as usize, 0);
                if (fd as i32) >= 0 {
                    let data = self.vfs.read_at(fd as usize, off as usize, len as usize);
                    if self.vfs.trace {
                        eprintln!(
                            "[vfs] mmap fd={fd} off=0x{off:x} len=0x{len:x} -> 0x{p:08x} ({} bytes)",
                            data.len()
                        );
                    }
                    if !data.is_empty() {
                        m.mem.write_bytes(p, &data).map_err(mem_err(name))?;
                    }
                }
                ret!(p)
            }
            "munmap" => {
                self.heap.free(r[0]);
                ret!(0)
            }

            // ---------- System V IPC (single process, so these are local) ----------
            "shmget" => {
                let id = self.next_shm;
                self.next_shm += 1;
                self.shm.insert(id, (0, r[1]));
                ret!(id)
            }
            "shmat" => {
                let id = r[0] as i32;
                let Some(entry) = self.shm.get(&id).copied() else {
                    ret!(-1i32)
                };
                if entry.0 != 0 {
                    ret!(entry.0)
                }
                let p = self.heap.malloc(&mut m.mem, entry.1.max(4096));
                let _ = m.mem.fill(p, entry.1.max(4096) as usize, 0);
                self.shm.insert(id, (p, entry.1));
                ret!(p)
            }
            "shmdt" => ret!(0),
            "shmctl" => ret!(0),
            "semget" => {
                let id = self.next_sem;
                self.next_sem += 1;
                self.sem.insert(id, vec![0; r[1].max(1) as usize]);
                ret!(id)
            }
            "semop" => ret!(0),
            "semctl" => ret!(0),

            // ---------- threads ----------
            "pthread_mutex_init"
            | "pthread_mutex_destroy"
            | "pthread_cond_init"
            | "pthread_cond_destroy"
            | "pthread_attr_init"
            | "pthread_attr_destroy"
            | "pthread_attr_setdetachstate"
            | "pthread_attr_setstacksize" => ret!(0),
            "pthread_self" => ret!(m.current_tid()),
            "pthread_detach" => {
                if let Some(i) = m.thread_index(r[0]) {
                    m.threads[i].detached = true;
                }
                ret!(0)
            }
            "pthread_create" => {
                let stack = self.heap.malloc(&mut m.mem, THREAD_STACK);
                if stack == 0 {
                    ret!(11)
                }
                let idx = m.spawn_thread(r[2], r[3], stack, THREAD_STACK);
                self.blocked.push(Blocked::No);
                let tid = m.threads[idx].tid;
                if r[0] != 0 {
                    m.mem.write_u32(r[0], tid).map_err(mem_err(name))?;
                }
                if self.trace_calls {
                    eprintln!("[thread] created t{tid} entry=0x{:x}", r[2]);
                }
                ret!(0)
            }
            "pthread_join" => {
                let Some(idx) = m.thread_index(r[0]) else {
                    ret!(3)
                };
                if m.threads[idx].finished {
                    if r[1] != 0 {
                        let v = m.threads[idx].retval;
                        m.mem.write_u32(r[1], v).map_err(mem_err(name))?;
                    }
                    ret!(0)
                }
                self.block(m, Blocked::Join(r[0]), 0)?;
                Ok(())
            }
            "pthread_mutex_lock" | "pthread_mutex_trylock" => {
                match self.mutex_owner.get(&r[0]).copied() {
                    None => {
                        self.mutex_owner.insert(r[0], m.current);
                        ret!(0)
                    }
                    Some(owner) if owner == m.current => ret!(0),
                    Some(_) => {
                        if name.ends_with("trylock") {
                            ret!(16)
                        }
                        self.block(m, Blocked::Mutex(r[0]), 0)?;
                        Ok(())
                    }
                }
            }
            "pthread_mutex_unlock" => {
                if self.mutex_owner.get(&r[0]) == Some(&m.current) {
                    self.mutex_owner.remove(&r[0]);
                    self.wake_mutex_waiter(r[0]);
                }
                ret!(0)
            }
            "pthread_cond_signal" | "pthread_cond_broadcast" => {
                let all = name.ends_with("broadcast");
                for i in 0..self.blocked.len() {
                    if let Blocked::Cond { cond, mutex, .. } = self.blocked[i].clone() {
                        if cond == r[0] {
                            self.blocked[i] = Blocked::Mutex(mutex);
                            if !all {
                                break;
                            }
                        }
                    }
                }
                let mutexes: Vec<u32> = self
                    .blocked
                    .iter()
                    .filter_map(|b| match b {
                        Blocked::Mutex(mx) => Some(*mx),
                        _ => None,
                    })
                    .collect();
                for mx in mutexes {
                    if !self.mutex_owner.contains_key(&mx) {
                        self.wake_mutex_waiter(mx);
                    }
                }
                ret!(0)
            }
            "pthread_cond_wait" | "pthread_cond_timedwait" => {
                let mutex = r[1];
                if self.mutex_owner.get(&mutex) == Some(&m.current) {
                    self.mutex_owner.remove(&mutex);
                    self.wake_mutex_waiter(mutex);
                }
                let timed = name.ends_with("timedwait");
                self.block(
                    m,
                    Blocked::Cond {
                        cond: r[0],
                        mutex,
                        timed,
                    },
                    0,
                )?;
                Ok(())
            }

            // ---------- math ----------
            "sqrt" => retf!(arg_f64!(0).sqrt()),
            "sin" => retf!(arg_f64!(0).sin()),
            "cos" => retf!(arg_f64!(0).cos()),
            "tan" => retf!(arg_f64!(0).tan()),
            "atan" => retf!(arg_f64!(0).atan()),
            "atan2" => retf!(arg_f64!(0).atan2(arg_f64!(2))),
            "exp" => retf!(arg_f64!(0).exp()),
            "log" => retf!(arg_f64!(0).ln()),
            "log10" => retf!(arg_f64!(0).log10()),
            "pow" => retf!(arg_f64!(0).powf(arg_f64!(2))),
            "floor" => retf!(arg_f64!(0).floor()),
            "ceil" => retf!(arg_f64!(0).ceil()),
            "fabs" => retf!(arg_f64!(0).abs()),
            "fmod" => retf!(arg_f64!(0) % arg_f64!(2)),

            // ---------- wide characters (wchar_t is 4 bytes on ARM Linux) ----------
            "wcslen" => {
                let mut n = 0u32;
                while m.mem.read_u32(r[0] + n * 4).map_err(mem_err(name))? != 0 {
                    n += 1;
                }
                ret!(n)
            }
            "wcschr" => {
                let mut p = r[0];
                loop {
                    let c = m.mem.read_u32(p).map_err(mem_err(name))?;
                    if c == r[1] {
                        ret!(p)
                    }
                    if c == 0 {
                        ret!(0)
                    }
                    p += 4;
                }
            }
            "wcscat" => {
                let mut d = r[0];
                while m.mem.read_u32(d).map_err(mem_err(name))? != 0 {
                    d += 4;
                }
                let mut s = r[1];
                loop {
                    let c = m.mem.read_u32(s).map_err(mem_err(name))?;
                    m.mem.write_u32(d, c).map_err(mem_err(name))?;
                    if c == 0 {
                        break;
                    }
                    d += 4;
                    s += 4;
                }
                ret!(r[0])
            }
            "fgetws" => {
                let Some(idx) = self.file_index(r[2]) else {
                    ret!(0)
                };
                let max = r[1] as usize;
                if max == 0 {
                    ret!(0)
                }
                let mut n = 0usize;
                while n + 1 < max {
                    let d = self.vfs.read(idx, 4);
                    if d.len() < 4 {
                        break;
                    }
                    let c = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
                    m.mem
                        .write_u32(r[0] + (n as u32) * 4, c)
                        .map_err(mem_err(name))?;
                    n += 1;
                    if c == 0x0A {
                        break;
                    }
                }
                if n == 0 {
                    ret!(0)
                }
                m.mem
                    .write_u32(r[0] + (n as u32) * 4, 0)
                    .map_err(mem_err(name))?;
                ret!(r[0])
            }
            "fputws" => {
                let Some(idx) = self.file_index(r[1]) else {
                    ret!(-1i32)
                };
                let mut p = r[0];
                let mut out = Vec::new();
                loop {
                    let c = m.mem.read_u32(p).map_err(mem_err(name))?;
                    if c == 0 {
                        break;
                    }
                    out.extend_from_slice(&c.to_le_bytes());
                    p += 4;
                }
                self.vfs.write(idx, &out);
                ret!(1)
            }

            // ---------- C runtime startup ----------
            "__gmon_start__" | "_Jv_RegisterClasses" | "__libc_start_main" => ret!(0),

            // ---------- libgcc ----------
            "__divsi3" => {
                let b = r[1] as i32;
                ret!(if b == 0 {
                    0
                } else {
                    (r[0] as i32).wrapping_div(b)
                })
            }
            "__udivsi3" => ret!(if r[1] == 0 { 0 } else { r[0] / r[1] }),
            "__modsi3" => {
                let b = r[1] as i32;
                ret!(if b == 0 {
                    0
                } else {
                    (r[0] as i32).wrapping_rem(b)
                })
            }
            "__umodsi3" => ret!(if r[1] == 0 { 0 } else { r[0] % r[1] }),

            // Native stand-ins for interpreted guest code, installed by
            // `Engine::patch_native`. These are not libc.
            "loqmsx:fir2x" => {
                let ord = m.mem.read_u32(sp).map_err(|e| e.to_string())? as i32;
                let mem = m.mem.read_u32(sp + 4).map_err(|e| e.to_string())?;
                self.msx_fir2x(m, r[0], r[1], r[2], r[3] as i32, ord, mem)?;
                return Ok(());
            }
            "loqmsx:iir32" => {
                let ord = m.mem.read_u32(sp).map_err(|e| e.to_string())? as i32;
                let mem = m.mem.read_u32(sp + 4).map_err(|e| e.to_string())?;
                self.msx_iir(m, r[0], r[1], r[2], r[3] as i32, ord, mem, true)?;
                return Ok(());
            }
            "loqmsx:iir16" => {
                let ord = m.mem.read_u32(sp).map_err(|e| e.to_string())? as i32;
                let mem = m.mem.read_u32(sp + 4).map_err(|e| e.to_string())?;
                self.msx_iir(m, r[0], r[1], r[2], r[3] as i32, ord, mem, false)?;
                return Ok(());
            }
            "loqmsx:lsp2lpc" => {
                self.spx_lsp_to_lpc(m, r[0], r[1], r[2] as i32)?;
                return Ok(());
            }
            "loqmsx:sigmul" => {
                self.spx_signal_mul(m, r[0], r[1], r[2] as i32, r[3] as i32)?;
                return Ok(());
            }
            "loqmsx:unpack" => {
                let v = self.spx_bits_unpack_unsigned(m, r[0], r[1] as i32)?;
                ret!(v);
            }
            "loqtts:cands" => {
                let best = m.mem.read_u32(sp).map_err(|e| e.to_string())?;
                let score = m.mem.read_u32(sp + 4).map_err(|e| e.to_string())?;
                self.cat_candidates(m, r[0], r[1], r[2], r[3] & 0xFF, best, score)?;
                return Ok(());
            }
            "loqtts:ctxcost" => {
                let base = self.native_base("LoqTTS6.so")?;
                let rec = m.mem.read_u32(sp).map_err(|e| e.to_string())?;
                let idx = m.mem.read_u32(sp + 4).map_err(|e| e.to_string())? & 0xFFFF;
                let v = crate::cat::context_cost(
                    &mut m.mem,
                    base,
                    r[0],
                    r[1],
                    r[2],
                    r[3] & 0xFF,
                    rec,
                    idx,
                )?;
                ret!(v);
            }

            other => Err(format!(
                "unimplemented libc function `{other}` called from {}",
                m.describe(m.cpu.r[14].wrapping_sub(4))
            )),
        }
    }

    /// Park the current thread, publish `retval` as its libc return value, and
    /// hand the CPU to someone else.
    fn block(&mut self, m: &mut Machine, why: Blocked, retval: u32) -> Result<(), String> {
        m.cpu.r[0] = retval;
        m.cpu.r[15] = m.cpu.r[14];
        self.pc_set = true;
        self.blocked[m.current] = why;
        self.schedule(m)
    }

    fn wake_mutex_waiter(&mut self, mutex: u32) {
        for i in 0..self.blocked.len() {
            if self.blocked[i] == Blocked::Mutex(mutex) {
                self.blocked[i] = Blocked::No;
                self.mutex_owner.insert(mutex, i);
                return;
            }
        }
    }

    /// Round-robin over runnable threads. A timed condition wait is the release
    /// valve when nothing else can run.
    fn schedule(&mut self, m: &mut Machine) -> Result<(), String> {
        let n = m.threads.len();
        for step in 1..=n {
            let i = (m.current + step) % n;
            if self.blocked[i] == Blocked::No && !m.threads[i].finished {
                m.switch_to(i);
                return Ok(());
            }
        }
        for i in 0..n {
            if let Blocked::Cond {
                mutex, timed: true, ..
            } = self.blocked[i].clone()
            {
                self.blocked[i] = Blocked::No;
                self.mutex_owner.entry(mutex).or_insert(i);
                m.switch_to(i);
                m.cpu.r[0] = ETIMEDOUT;
                return Ok(());
            }
        }
        Err(format!("all {n} threads are blocked: {:?}", self.blocked))
    }

    fn stdout_index(&self, m: &Machine) -> Option<usize> {
        let ptr = m.mem.read_u32(self.std_vars.1).ok()?;
        self.file_objs.get(&ptr).copied()
    }

    fn dlopen(&mut self, m: &mut Machine, path: &str) -> Result<u32, String> {
        let base = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
        if let Some((h, _)) = self
            .dl_modules
            .iter()
            .find(|(_, i)| m.modules[**i].name == base)
            .map(|(h, i)| (*h, *i))
        {
            return Ok(h);
        }
        if let Some(i) = m.modules.iter().position(|mm| mm.name == base) {
            let h = 0x9000_0000 + i as u32;
            self.dl_modules.insert(h, i);
            return Ok(h);
        }

        let guest = format!("{}/{}", crate::tts::GUEST_LIB, base);
        let data = self
            .vfs
            .read_file(&guest)
            .ok_or_else(|| format!("{guest}: no such engine module"))?;
        let idx = m.load(&base, &data)?;
        let stubs = m.link(|mach, sym| self.resolve_for_link(mach, sym))?;
        if self.trace_calls && !stubs.is_empty() {
            eprintln!("[dl] {base}: stubbed {} symbols", stubs.len());
        }
        self.patch_native(m, idx)?;
        let h = 0x9000_0000 + idx as u32;
        self.dl_modules.insert(h, idx);
        Ok(h)
    }

    /// Used as the linker's resolver: data symbols get real storage, everything
    /// else becomes a host trap.
    pub fn resolve_for_link(&mut self, _m: &mut Machine, sym: &str) -> Option<u32> {
        self.resolve_data(sym)
    }
}

fn mem_err(name: &str) -> impl Fn(armemu::Fault) -> String + '_ {
    move |f| format!("{name}: {f}")
}

fn now_secs() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

fn write_tm(mem: &mut Mem, addr: u32, t: u32) {
    // Civil-time conversion for UTC; the engine only uses this for log stamps.
    let days = (t / 86400) as i64;
    let rem = t % 86400;
    let (y, mo, d) = civil_from_days(days);
    let fields = [
        (rem % 60) as i32,
        ((rem / 60) % 60) as i32,
        (rem / 3600) as i32,
        d,
        mo - 1,
        y - 1900,
        ((days + 4).rem_euclid(7)) as i32,
        0,
        0,
    ];
    for (i, v) in fields.iter().enumerate() {
        let _ = mem.write_u32(addr + (i as u32) * 4, *v as u32);
    }
}

fn civil_from_days(z: i64) -> (i32, i32, i32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    ((y + i64::from(m <= 2)) as i32, m as i32, d as i32)
}

fn parse_int(s: &[u8], base: i32) -> (i64, usize) {
    let mut i = 0;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    let mut neg = false;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut base = base;
    if (base == 0 || base == 16) && i + 1 < s.len() && s[i] == b'0' && (s[i + 1] | 32) == b'x' {
        base = 16;
        i += 2;
    } else if base == 0 {
        base = if i < s.len() && s[i] == b'0' { 8 } else { 10 };
    }
    let digits_start = i;
    let mut val: i64 = 0;
    while i < s.len() {
        let d = (s[i] as char).to_digit(base as u32);
        match d {
            Some(d) => {
                val = val.saturating_mul(base as i64).saturating_add(d as i64);
                i += 1;
            }
            None => break,
        }
    }
    if i == digits_start {
        return (0, 0);
    }
    let _ = start;
    (if neg { -val } else { val }, i)
}

fn parse_double(s: &[u8]) -> (f64, usize) {
    let mut i = 0;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        i += 1;
    }
    while i < s.len() && (s[i].is_ascii_digit() || s[i] == b'.') {
        i += 1;
    }
    if i < s.len() && (s[i] | 32) == b'e' {
        let save = i;
        i += 1;
        if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
            i += 1;
        }
        if i < s.len() && s[i].is_ascii_digit() {
            while i < s.len() && s[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            i = save;
        }
    }
    if i == start {
        return (0.0, 0);
    }
    let text = String::from_utf8_lossy(&s[start..i]);
    (text.parse().unwrap_or(0.0), i)
}

// SNAPSHOT-MARK
use armemu::{Reader, Snap, SnapResult, Writer};

impl Snap for Blocked {
    fn save(&self, w: &mut Writer) {
        match self {
            Blocked::No => w.u8(0),
            Blocked::Cond { cond, mutex, timed } => {
                w.u8(1);
                w.u32(*cond);
                w.u32(*mutex);
                w.put(timed);
            }
            Blocked::Mutex(a) => {
                w.u8(2);
                w.u32(*a);
            }
            Blocked::Join(a) => {
                w.u8(3);
                w.u32(*a);
            }
            Blocked::Finished => w.u8(4),
        }
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(match r.u8()? {
            0 => Blocked::No,
            1 => Blocked::Cond {
                cond: r.u32()?,
                mutex: r.u32()?,
                timed: r.get()?,
            },
            2 => Blocked::Mutex(r.u32()?),
            3 => Blocked::Join(r.u32()?),
            4 => Blocked::Finished,
            t => return Err(format!("snapshot has blocked tag {t}")),
        })
    }
}

impl Host {
    /// Everything the guest can reach back into. The tracing flags are left
    /// out on purpose: they belong to the run, not to the saved state.
    pub fn save_state(&self, w: &mut Writer) {
        w.put(&self.heap);
        self.vfs.save_state(w);
        w.put(&self.dumps);
        w.put(&self.api_names);
        w.put(&self.lib_dir);
        w.put(&self.exited);
        w.put(&self.file_objs);
        w.put(&self.dir_objs);
        w.u32(self.errno_addr);
        w.u32(self.dirent_buf);
        w.u32(self.tm_buf);
        w.u32(self.dl_error);
        w.put(&self.std_vars);
        w.u32(self.strtok_state);
        w.u32(self.rng);
        w.put(&self.shm);
        w.put(&self.next_shm);
        w.put(&self.sem);
        w.put(&self.next_sem);
        w.put(&self.dl_modules);
        w.put(&self.unimplemented);
        w.put(&self.blocked);
        w.put(&self.mutex_owner);
        w.put(&self.native_bases);
        w.put(&self.pc_set);
    }

    pub fn restore_state(&mut self, r: &mut Reader) -> SnapResult<()> {
        self.heap = r.get()?;
        self.vfs.restore_state(r)?;
        self.dumps = r.get()?;
        self.api_names = r.get()?;
        self.lib_dir = r.get()?;
        self.exited = r.get()?;
        self.file_objs = r.get()?;
        self.dir_objs = r.get()?;
        self.errno_addr = r.u32()?;
        self.dirent_buf = r.u32()?;
        self.tm_buf = r.u32()?;
        self.dl_error = r.u32()?;
        self.std_vars = r.get()?;
        self.strtok_state = r.u32()?;
        self.rng = r.u32()?;
        self.shm = r.get()?;
        self.next_shm = r.get()?;
        self.sem = r.get()?;
        self.next_sem = r.get()?;
        self.dl_modules = r.get()?;
        self.unimplemented = r.get()?;
        self.blocked = r.get()?;
        self.mutex_owner = r.get()?;
        self.native_bases = r.get()?;
        self.pc_set = r.get()?;
        Ok(())
    }
}
