//! In-process ARM machine: loads ARM ELF shared objects into a private guest
//! address space and executes them with an ARMv5TE + FPA11 interpreter.
//!
//! There is no guest kernel and no guest libc. Imported symbols are bound to
//! trap addresses that hand control back to the embedding Rust program.

pub mod block;
pub mod cpu;
pub mod elf;
pub mod fpa;
pub mod mem;
pub mod snap;
#[cfg(all(unix, target_arch = "x86_64"))]
pub mod x64;

use std::collections::HashMap;

pub use cpu::{Cpu, Stop, HOSTCALL_BASE, MAGIC_RETURN};
pub use elf::{Loader, Module};
pub use mem::{Fault, Mem};
pub use snap::{Reader, Snap, SnapResult, Writer};

pub const MODULE_BASE: u32 = 0x0010_0000;
pub const HEAP_BASE: u32 = 0x1000_0000;
pub const HEAP_LIMIT: u32 = 0x6000_0000;
pub const HOSTDATA_BASE: u32 = 0xD000_0000;
pub const STACK_TOP: u32 = 0x7F00_0000;
pub const STACK_SIZE: u32 = 8 * 1024 * 1024;

pub struct Machine {
    pub mem: Mem,
    pub cpu: Cpu,
    /// Decoded-instruction cache. Not part of the snapshot: it is derived
    /// state, rebuilt on demand, and flushed whenever guest code changes.
    pub blocks: block::BlockCache,
    /// Execute through the block cache.
    ///
    /// Off by default: with every op still deferring to the interpreter's own
    /// handler, the cache costs about 19% and saves only the (already
    /// page-cached) fetch. It earns its keep once the decoded ops are emitted
    /// as native code; until then this stays opt-in so nothing regresses.
    pub jit: bool,
    pub modules: Vec<Module>,
    loader: Loader,
    hostcalls: Vec<String>,
    hostcall_index: HashMap<String, u32>,
    /// Bump allocator for host-owned guest data (FILE objects, errno, ...).
    hostdata_next: u32,
    /// Relocations are not idempotent, so each module is bound exactly once.
    linked: Vec<bool>,
    /// Symbols the host owns outright; these beat any module's definition.
    overrides: HashMap<String, u32>,
    /// Saved contexts. `cpu` above is whichever of these is running.
    pub threads: Vec<ThreadCtx>,
    pub current: usize,
    next_tid: u32,
    /// Profile samples drained from threads that have been switched away from.
    samples: HashMap<u32, u64>,
}

#[derive(Clone)]
pub struct ThreadCtx {
    pub cpu: Cpu,
    pub tid: u32,
    pub stack_base: u32,
    pub stack_size: u32,
    pub finished: bool,
    pub retval: u32,
    pub detached: bool,
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

impl Machine {
    pub fn new() -> Self {
        let mut m = Machine {
            mem: Mem::new(),
            cpu: Cpu::new(),
            blocks: block::BlockCache::new(),
            // Opt-in while the backend's coverage is still narrow. Read once
            // here, never on a hot path. A proper CLI flag replaces this once
            // it is worth enabling by default.
            jit: std::env::var_os("LOQ_JIT").is_some(),
            modules: Vec::new(),
            loader: Loader::new(MODULE_BASE),
            hostcalls: Vec::new(),
            hostcall_index: HashMap::new(),
            hostdata_next: HOSTDATA_BASE,
            linked: Vec::new(),
            overrides: HashMap::new(),
            threads: Vec::new(),
            current: 0,
            next_tid: 2,
            samples: HashMap::new(),
        };
        m.mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE);
        m.cpu.r[13] = STACK_TOP - 16;
        m.threads.push(ThreadCtx {
            cpu: m.cpu.clone(),
            tid: 1,
            stack_base: STACK_TOP - STACK_SIZE,
            stack_size: STACK_SIZE,
            finished: false,
            retval: 0,
            detached: false,
        });
        m
    }

    /// Create a runnable thread whose start routine is `entry(arg)`.
    pub fn spawn_thread(
        &mut self,
        entry: u32,
        arg: u32,
        stack_base: u32,
        stack_size: u32,
    ) -> usize {
        let mut cpu = Cpu::new();
        cpu.history = self.cpu.history;
        cpu.profile = self.cpu.profile;
        cpu.breakpoints = self.cpu.breakpoints.clone();
        cpu.r[0] = arg;
        cpu.r[13] = (stack_base + stack_size - 16) & !7;
        cpu.r[14] = cpu::MAGIC_THREAD_EXIT;
        cpu.r[15] = entry;
        let tid = self.next_tid;
        self.next_tid += 1;
        self.threads.push(ThreadCtx {
            cpu,
            tid,
            stack_base,
            stack_size,
            finished: false,
            retval: 0,
            detached: false,
        });
        self.threads.len() - 1
    }

    pub fn switch_to(&mut self, idx: usize) {
        if idx == self.current {
            return;
        }
        let icount = self.cpu.icount;
        // Same reasoning as icount: this counts the run, not the thread, so it
        // has to survive the clone swap below or every switch resets it.
        let jitted = self.cpu.jitted;
        let jit_native = self.cpu.jit_native;
        let icount_base = self.cpu.icount_base;
        // Samples belong to the run, not the thread; keep them out of the
        // per-thread clone so switching stays cheap and nothing is lost.
        for (pc, n) in self.cpu.samples.drain() {
            *self.samples.entry(pc).or_insert(0) += n;
        }
        self.threads[self.current].cpu = self.cpu.clone();
        let history = self.cpu.history;
        let bps = self.cpu.breakpoints.clone();
        self.cpu = self.threads[idx].cpu.clone();
        self.cpu.icount = icount;
        self.cpu.jitted = jitted;
        self.cpu.jit_native = jit_native;
        self.cpu.icount_base = icount_base;
        self.cpu.history = history;
        self.cpu.breakpoints = bps;
        self.current = idx;
    }

    pub fn current_tid(&self) -> u32 {
        self.threads[self.current].tid
    }

    pub fn thread_index(&self, tid: u32) -> Option<usize> {
        self.threads.iter().position(|t| t.tid == tid)
    }

    pub fn load(&mut self, name: &str, data: &[u8]) -> Result<usize, String> {
        let m = self.loader.load(&mut self.mem, name, data)?;
        self.modules.push(m);
        self.linked.push(false);
        // New code at addresses a previous module may have occupied.
        self.blocks.flush();
        Ok(self.modules.len() - 1)
    }

    pub fn module(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// Resolve a symbol across all loaded modules.
    pub fn lookup(&self, name: &str) -> Option<u32> {
        self.modules.iter().find_map(|m| m.symbol(name))
    }

    /// Guest address that traps back to the host, allocated per symbol name.
    pub fn hostcall_addr(&mut self, name: &str) -> u32 {
        if let Some(i) = self.hostcall_index.get(name) {
            return HOSTCALL_BASE + i * 4;
        }
        let i = self.hostcalls.len() as u32;
        self.hostcalls.push(name.to_string());
        self.hostcall_index.insert(name.to_string(), i);
        HOSTCALL_BASE + i * 4
    }

    /// Redirect guest calls to `addr` into the host call `name`.
    ///
    /// Writes the veneer a PLT would, over the function's first two words:
    ///
    /// ```text
    ///     ldr pc, [pc, #-4]
    ///     .word <host call address>
    /// ```
    ///
    /// The interpreter already traps any PC inside the host-call page, so this
    /// costs nothing per instruction — unlike a per-step address check. `lr`
    /// is untouched, so the handler returns the ordinary way.
    pub fn patch_hostcall(&mut self, addr: u32, name: &str) -> Result<(), String> {
        let target = self.hostcall_addr(name);
        self.mem
            .write_u32(addr, 0xE51F_F004)
            .map_err(|e| format!("patching {name} at 0x{addr:08x}: {e}"))?;
        self.mem
            .write_u32(addr + 4, target)
            .map_err(|e| format!("patching {name} at 0x{addr:08x}: {e}"))?;
        // The veneer overwrote two instructions; anything cached for them is
        // now stale.
        self.blocks.flush();
        Ok(())
    }

    /// Base address of a loaded module, by file name.
    pub fn module_base(&self, name: &str) -> Option<u32> {
        self.modules.iter().find(|m| m.name == name).map(|m| m.base)
    }

    pub fn hostcall_name(&self, index: u32) -> &str {
        self.hostcalls
            .get(index as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }

    /// Force `name` to resolve to `addr` everywhere, beating module exports.
    pub fn set_override(&mut self, name: &str, addr: u32) {
        self.overrides.insert(name.to_string(), addr);
    }

    /// Reserve `len` bytes of host-owned guest memory.
    pub fn alloc_hostdata(&mut self, len: u32) -> u32 {
        let len = ((len + 7) & !7).max(8);
        let a = self.hostdata_next;
        self.mem.map(a, len);
        self.hostdata_next = a + len;
        a
    }

    /// Bind every loaded module. Symbols no module defines go to `resolve`;
    /// anything it declines becomes a host trap stub. Returns the stubbed names.
    pub fn link<F>(&mut self, mut resolve: F) -> Result<Vec<String>, String>
    where
        F: FnMut(&mut Machine, &str) -> Option<u32>,
    {
        let mut table: HashMap<String, u32> = self.overrides.clone();
        for m in &self.modules {
            for (k, v) in &m.exports {
                table.entry(k.clone()).or_insert(*v);
            }
        }

        let mut wanted: Vec<String> = Vec::new();
        for m in &self.modules {
            for name in m.undefined_symbols() {
                if !table.contains_key(name) && !wanted.iter().any(|w| w == name) {
                    wanted.push(name.to_string());
                }
            }
        }

        let mut stubbed = Vec::new();
        for name in wanted {
            let addr = match resolve(self, &name) {
                Some(a) => a,
                None => {
                    stubbed.push(name.clone());
                    self.hostcall_addr(&name)
                }
            };
            table.insert(name, addr);
        }

        let Machine {
            mem,
            modules,
            linked,
            ..
        } = self;
        for (i, m) in modules.iter().enumerate() {
            if linked[i] {
                continue;
            }
            elf::relocate(mem, m, |name| table.get(name).copied())?;
            linked[i] = true;
        }
        Ok(stubbed)
    }

    /// Set up a guest call: args in r0-r3 then the stack, return to MAGIC_RETURN.
    pub fn setup_call(&mut self, func: u32, args: &[u32]) -> Result<(), String> {
        for (i, a) in args.iter().take(4).enumerate() {
            self.cpu.r[i] = *a;
        }
        if args.len() > 4 {
            let extra = &args[4..];
            let bytes = (extra.len() as u32) * 4;
            let mut sp = (self.cpu.r[13] - bytes) & !7;
            self.cpu.r[13] = sp;
            for a in extra {
                self.mem.write_u32(sp, *a).map_err(|e| e.to_string())?;
                sp += 4;
            }
        }
        self.cpu.r[14] = MAGIC_RETURN;
        self.cpu.r[15] = func;
        Ok(())
    }

    pub fn run(&mut self, budget: u64) -> Stop {
        if self.jit {
            self.cpu.run_cached(&mut self.mem, &mut self.blocks, budget)
        } else {
            self.cpu.run(&mut self.mem, budget)
        }
    }

    /// Drop every cached block. Required after anything that rewrites guest
    /// code or replaces guest memory wholesale (veneer patching, module load,
    /// snapshot restore).
    pub fn flush_blocks(&mut self) {
        self.blocks.flush();
    }

    /// Best-effort rendering of one argument: a short printable string if it
    /// points at one, otherwise the raw word.
    pub fn arg(&self, v: u32) -> String {
        if v > 0x1000 && self.mem.is_mapped(v) {
            if let Ok(bytes) = self.mem.read_cstr(v) {
                let printable = !bytes.is_empty()
                    && bytes.len() < 120
                    && bytes.iter().all(|c| (0x20..0x7f).contains(c));
                if printable {
                    return format!("0x{v:08x} \"{}\"", String::from_utf8_lossy(&bytes));
                }
            }
        }
        format!("0x{v:08x}")
    }

    /// Samples from every thread, including the one running now.
    /// Raw samples as `module+offset count`, one per line, hottest first.
    ///
    /// `profile_report` folds by symbol, which is useless for a stripped
    /// module: everything lands in one bucket. This keeps the offsets so an
    /// external symbol table can attribute them to functions.
    pub fn profile_raw(&self) -> String {
        let mut rows: Vec<(&str, u32, u64)> = Vec::new();
        let merged = self.all_samples();
        for (pc, n) in &merged {
            match self.modules.iter().find(|m| *pc >= m.base && *pc < m.end) {
                Some(m) => rows.push((&m.name, pc - m.base, *n)),
                None => rows.push(("?", *pc, *n)),
            }
        }
        rows.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)).then(a.1.cmp(&b.1)));

        let mut out = String::new();
        for (name, off, n) in rows {
            out.push_str(&format!("{name}+0x{off:x} {n}\n"));
        }
        out
    }

    fn all_samples(&self) -> HashMap<u32, u64> {
        let mut out = self.samples.clone();
        for (pc, n) in &self.cpu.samples {
            *out.entry(*pc).or_insert(0) += n;
        }
        out
    }

    pub fn clear_samples(&mut self) {
        self.samples.clear();
        self.cpu.samples.clear();
        for t in &mut self.threads {
            t.cpu.samples.clear();
        }
    }

    /// Where the guest spent its time, by nearest preceding symbol.
    pub fn profile_report(&self, top: usize) -> String {
        if self.cpu.samples.is_empty() && self.samples.is_empty() {
            return "  (no samples; run with --profile)".to_string();
        }
        // Resolve each sampled PC once, then fold by symbol.
        let mut by_symbol: HashMap<String, u64> = HashMap::new();
        let mut cache: HashMap<u32, String> = HashMap::new();
        let merged = self.all_samples();
        for (pc, n) in &merged {
            let key = cache
                .entry(pc & !0x3F)
                .or_insert_with(|| self.describe_symbol(*pc));
            *by_symbol.entry(key.clone()).or_insert(0) += n;
        }

        let total: u64 = by_symbol.values().sum();
        let mut rows: Vec<(String, u64)> = by_symbol.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));

        let mut out = format!("  {:>6}  {:>13}  {}\n", "share", "instructions", "where");
        for (name, n) in rows.iter().take(top) {
            out.push_str(&format!(
                "  {:>5.1}%  {:>13}  {}\n",
                100.0 * *n as f64 / total as f64,
                n * 256,
                name
            ));
        }
        let shown: u64 = rows.iter().take(top).map(|(_, n)| *n).sum();
        out.push_str(&format!(
            "  {:>5.1}%  {:>13}  ({} other sites)\n",
            100.0 * (total - shown) as f64 / total as f64,
            (total - shown) * 256,
            rows.len().saturating_sub(top)
        ));
        out
    }

    /// Module plus nearest preceding symbol, with no offset.
    fn describe_symbol(&self, addr: u32) -> String {
        for m in &self.modules {
            if addr >= m.base && addr < m.end {
                return match m.describe(addr) {
                    Some((sym, _)) => format!("{}!{}", m.name, sym),
                    None => m.name.clone(),
                };
            }
        }
        format!("0x{addr:08x}")
    }

    /// Just the registers, one line per bank.
    pub fn regs(&self) -> String {
        format!(
            "  r0-r3  {:08x} {:08x} {:08x} {:08x}\n  r4-r7  {:08x} {:08x} {:08x} {:08x}\n  r8-r11 {:08x} {:08x} {:08x} {:08x}\n  ip {:08x} sp {:08x} lr {:08x}",
            self.cpu.r[0], self.cpu.r[1], self.cpu.r[2], self.cpu.r[3],
            self.cpu.r[4], self.cpu.r[5], self.cpu.r[6], self.cpu.r[7],
            self.cpu.r[8], self.cpu.r[9], self.cpu.r[10], self.cpu.r[11],
            self.cpu.r[12], self.cpu.r[13], self.cpu.r[14]
        )
    }

    /// Hex dump for a spec like "r6+0x11d0:0x20" or "0x11d33c38:16".
    pub fn dump(&self, spec: &str) -> String {
        let (loc, len) = match spec.rsplit_once(':') {
            Some((l, n)) => (
                l,
                usize::from_str_radix(
                    n.trim_start_matches("0x"),
                    if n.starts_with("0x") { 16 } else { 10 },
                )
                .unwrap_or(16),
            ),
            None => (spec, 16),
        };
        let (base_txt, off) = match loc.split_once('+') {
            Some((b, o)) => (
                b,
                u32::from_str_radix(o.trim_start_matches("0x"), 16).unwrap_or(0),
            ),
            None => (loc, 0),
        };
        let base = if let Some(n) = base_txt.strip_prefix('r') {
            match n.parse::<usize>() {
                Ok(i) if i < 16 => self.cpu.r[i],
                _ => return format!("  {spec}: bad register"),
            }
        } else {
            u32::from_str_radix(base_txt.trim_start_matches("0x"), 16).unwrap_or(0)
        };
        let addr = base.wrapping_add(off);
        let mut out = format!("  dump {spec} @ 0x{addr:08x}:");
        for i in 0..len {
            if i % 16 == 0 {
                out.push_str(&format!("\n    {:08x} ", addr + i as u32));
            }
            match self.mem.read_u8(addr + i as u32) {
                Ok(v) => out.push_str(&format!("{v:02x} ")),
                Err(_) => out.push_str("?? "),
            }
        }
        out
    }

    /// Registers plus recent control transfers, for post-mortem.
    pub fn backtrace(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "  r0-r3  {:08x} {:08x} {:08x} {:08x}\n  r4-r7  {:08x} {:08x} {:08x} {:08x}\n",
            self.cpu.r[0],
            self.cpu.r[1],
            self.cpu.r[2],
            self.cpu.r[3],
            self.cpu.r[4],
            self.cpu.r[5],
            self.cpu.r[6],
            self.cpu.r[7]
        ));
        out.push_str(&format!(
            "  r8-r11 {:08x} {:08x} {:08x} {:08x}\n  ip {:08x} sp {:08x} lr {:08x} pc {:08x}\n",
            self.cpu.r[8],
            self.cpu.r[9],
            self.cpu.r[10],
            self.cpu.r[11],
            self.cpu.r[12],
            self.cpu.r[13],
            self.cpu.r[14],
            self.cpu.r[15]
        ));
        if self.cpu.branches.is_empty() {
            out.push_str("  (run with --trace-branches for control-transfer history)\n");
            return out;
        }
        out.push_str("  recent control transfers (oldest first):\n");
        let n = self.cpu.branches.len();
        let head = self.cpu.branch_head % n;
        let ordered: Vec<(u32, u32)> = (0..n).map(|i| self.cpu.branches[(head + i) % n]).collect();
        for (from, to) in ordered.iter().rev().take(24).rev() {
            out.push_str(&format!(
                "    {} -> {}\n",
                self.describe(*from),
                self.describe(*to)
            ));
        }
        out
    }

    /// Human-readable location, for error messages.
    pub fn describe(&self, addr: u32) -> String {
        if (HOSTCALL_BASE..cpu::HOSTCALL_END).contains(&addr) {
            return format!(
                "hostcall {}",
                self.hostcall_name((addr - HOSTCALL_BASE) / 4)
            );
        }
        for m in &self.modules {
            if addr >= m.base && addr < m.end {
                return match m.describe(addr) {
                    Some((sym, off)) => format!("{}!{}+0x{:x} [0x{:08x}]", m.name, sym, off, addr),
                    None => format!("{}+0x{:x}", m.name, addr - m.base),
                };
            }
        }
        format!("0x{addr:08x}")
    }
}

impl Snap for ThreadCtx {
    fn save(&self, w: &mut Writer) {
        w.put(&self.cpu);
        w.u32(self.tid);
        w.u32(self.stack_base);
        w.u32(self.stack_size);
        w.put(&self.finished);
        w.u32(self.retval);
        w.put(&self.detached);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        Ok(ThreadCtx {
            cpu: r.get()?,
            tid: r.u32()?,
            stack_base: r.u32()?,
            stack_size: r.u32()?,
            finished: r.get()?,
            retval: r.u32()?,
            detached: r.get()?,
        })
    }
}

impl Snap for Machine {
    fn save(&self, w: &mut Writer) {
        w.put(&self.mem);
        w.put(&self.cpu);
        w.put(&self.modules);
        w.put(&self.loader);
        w.put(&self.hostcalls);
        w.put(&self.hostcall_index);
        w.u32(self.hostdata_next);
        w.put(&self.linked);
        w.put(&self.overrides);
        w.put(&self.threads);
        w.put(&self.current);
        w.u32(self.next_tid);
        w.put(&self.samples);
    }

    fn load(r: &mut Reader) -> SnapResult<Self> {
        let m = Machine {
            mem: r.get()?,
            cpu: r.get()?,
            // Derived state, never snapshotted: a restored machine starts with
            // a cold cache, which is also how restore-time invalidation is
            // handled. The switch is re-read rather than forced on — loqdave
            // restores a snapshot on every run, so hardcoding it here silently
            // enabled the JIT in every measurement.
            blocks: block::BlockCache::new(),
            jit: std::env::var_os("LOQ_JIT").is_some(),
            modules: r.get()?,
            loader: r.get()?,
            hostcalls: r.get()?,
            hostcall_index: r.get()?,
            hostdata_next: r.u32()?,
            linked: r.get()?,
            overrides: r.get()?,
            threads: r.get()?,
            current: r.get()?,
            next_tid: r.u32()?,
            samples: r.get()?,
        };
        if m.current >= m.threads.len() {
            return Err(format!(
                "snapshot runs thread {} of {}",
                m.current,
                m.threads.len()
            ));
        }
        Ok(m)
    }
}
