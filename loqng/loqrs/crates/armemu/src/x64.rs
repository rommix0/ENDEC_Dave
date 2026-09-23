//! x86-64 backend: executable memory and the encoder the block compiler emits
//! through.
//!
//! Only the handful of forms the compiler actually uses are encoded, nearly
//! all of them 32-bit operations on the legacy registers, so there are almost
//! no REX prefixes and no ModRM special cases. That is what lets this stay a
//! few hundred lines instead of a general assembler.
//!
//! Register file access is `[rbx + reg*4]`. A guest register index is at most
//! 15, so the displacement is at most 60 and always fits in the disp8 form.
//! `rbx`, `r14` and `r15` are callee-saved and hold the register file, the
//! `Cpu` and the `Mem` pointers, so a callback into the interpreter can use
//! `rdi`/`rsi` for arguments without disturbing them.

use std::io;

/// A page-aligned, executable code buffer.
///
/// Written while writable, then flipped to read+execute: W^X is kept, so a bug
/// in the compiler cannot scribble over already-published code.
pub struct ExecBuf {
    ptr: *mut u8,
    len: usize,
}

// The buffer is owned outright and never aliased once published.
unsafe impl Send for ExecBuf {}
unsafe impl Sync for ExecBuf {}

impl ExecBuf {
    /// Publish `code` into a fresh executable mapping.
    pub fn new(code: &[u8]) -> io::Result<ExecBuf> {
        if code.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty code"));
        }
        let page = 4096;
        let len = (code.len() + page - 1) & !(page - 1);
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            let ptr = ptr as *mut u8;
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
            if libc::mprotect(
                ptr as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_EXEC,
            ) != 0
            {
                let e = io::Error::last_os_error();
                libc::munmap(ptr as *mut libc::c_void, len);
                return Err(e);
            }
            Ok(ExecBuf { ptr, len })
        }
    }

    /// The entry point, as the compiled-block ABI.
    ///
    /// # Safety
    /// The caller must pass a valid guest register file, `Cpu` and `Mem`, all
    /// live for the duration of the call. Emitted code touches the register
    /// file directly and reaches everything else through the interpreter
    /// callback.
    #[inline(always)]
    pub unsafe fn as_fn(&self) -> unsafe extern "sysv64" fn(*mut u32, *mut u8, *mut u8) -> u64 {
        std::mem::transmute(self.ptr)
    }
}

impl Drop for ExecBuf {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// Byte-level x86-64 encoder for the subset the block compiler emits.
#[derive(Default)]
pub struct Emitter {
    pub code: Vec<u8>,
}

/// Displacement of guest register `r` within the register file.
#[inline]
fn disp(r: u8) -> u8 {
    r * 4
}

impl Emitter {
    pub fn new() -> Self {
        Emitter {
            code: Vec::with_capacity(64),
        }
    }

    fn imm32(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `mov eax, [rdi + r*4]`
    pub fn load_reg(&mut self, r: u8) {
        // 8B /r, ModRM mod=01 reg=eax(0) rm=rdi(7) -> 0x43
        self.code.extend_from_slice(&[0x8B, 0x43, disp(r)]);
    }

    /// `mov [rdi + r*4], eax`
    pub fn store_reg(&mut self, r: u8) {
        // 89 /r, same ModRM
        self.code.extend_from_slice(&[0x89, 0x43, disp(r)]);
    }

    /// `mov dword [rdi + r*4], imm32`
    pub fn store_imm(&mut self, r: u8, v: u32) {
        // C7 /0
        self.code.extend_from_slice(&[0xC7, 0x43, disp(r)]);
        self.imm32(v);
    }

    /// `add eax, imm32` (eax short form)
    pub fn add_eax(&mut self, v: u32) {
        self.code.push(0x05);
        self.imm32(v);
    }

    /// `sub eax, imm32`
    pub fn sub_eax(&mut self, v: u32) {
        self.code.push(0x2D);
        self.imm32(v);
    }

    /// `and eax, imm32`
    pub fn and_eax(&mut self, v: u32) {
        self.code.push(0x25);
        self.imm32(v);
    }

    /// `or eax, imm32`
    pub fn or_eax(&mut self, v: u32) {
        self.code.push(0x0D);
        self.imm32(v);
    }

    /// `xor eax, imm32`
    pub fn xor_eax(&mut self, v: u32) {
        self.code.push(0x35);
        self.imm32(v);
    }

    /// `mov eax, imm32`
    pub fn mov_eax(&mut self, v: u32) {
        self.code.push(0xB8);
        self.imm32(v);
    }

    /// `add eax, [rdi + r*4]`
    pub fn add_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x03, 0x43, disp(r)]);
    }

    /// `sub eax, [rdi + r*4]`
    pub fn sub_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x2B, 0x43, disp(r)]);
    }

    /// `and eax, [rdi + r*4]`
    pub fn and_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x23, 0x43, disp(r)]);
    }

    /// `or eax, [rdi + r*4]`
    pub fn or_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x0B, 0x43, disp(r)]);
    }

    /// `xor eax, [rdi + r*4]`
    pub fn xor_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x33, 0x43, disp(r)]);
    }

    /// `not eax`
    pub fn not_eax(&mut self) {
        self.code.extend_from_slice(&[0xF7, 0xD0]);
    }

    /// `ret`
    pub fn ret(&mut self) {
        self.code.push(0xC3);
    }

    pub fn finish(self) -> Vec<u8> {
        self.code
    }
}

// ------------------------------------------------------------- frame + calls

impl Emitter {
    /// `push rbx / r14 / r15`, then park the three arguments in them.
    ///
    /// Three pushes leave `rsp` 16-byte aligned at the point of a `call`,
    /// which is what the SysV ABI requires. Do not add a fourth.
    pub fn prologue(&mut self) {
        self.code.extend_from_slice(&[
            0x53, // push rbx
            0x41, 0x56, // push r14
            0x41, 0x57, // push r15
            0x48, 0x89, 0xFB, // mov rbx, rdi   (register file)
            0x49, 0x89, 0xF6, // mov r14, rsi   (Cpu)
            0x49, 0x89, 0xD7, // mov r15, rdx   (Mem)
        ]);
    }

    /// Unwind and return `insns` guest instructions executed, `native` of them
    /// without a callback.
    ///
    /// Both are packed into the one return register: the native count is
    /// diagnostic, and every exit knows it statically, so carrying it costs
    /// five bytes of immediate at a block exit and nothing at all per
    /// instruction.
    pub fn epilogue(&mut self, insns: u32, native: u32) {
        self.code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, imm64
        let packed = (u64::from(native) << 32) | u64::from(insns);
        self.code.extend_from_slice(&packed.to_le_bytes());
        self.code.extend_from_slice(&[
            0x41, 0x5F, // pop r15
            0x41, 0x5E, // pop r14
            0x5B, // pop rbx
            0xC3, // ret
        ]);
    }

    /// Call the interpreter for one instruction: `f(cpu, mem, insn, pc)`.
    pub fn call_step(&mut self, f: usize, insn: u32, pc: u32) {
        self.code.extend_from_slice(&[0x4C, 0x89, 0xF7]); // mov rdi, r14
        self.code.extend_from_slice(&[0x4C, 0x89, 0xFE]); // mov rsi, r15
        self.code.push(0xBA); // mov edx, insn
        self.imm32(insn);
        self.code.push(0xB9); // mov ecx, pc
        self.imm32(pc);
        self.code.extend_from_slice(&[0x48, 0xB8]); // movabs rax, f
        self.code.extend_from_slice(&(f as u64).to_le_bytes());
        self.code.extend_from_slice(&[0xFF, 0xD0]); // call rax
    }

    /// `test rax, rax; jnz <patched later>`. Returns the patch site.
    pub fn test_jnz(&mut self) -> usize {
        self.code.extend_from_slice(&[0x48, 0x85, 0xC0]); // test rax, rax
        self.code.extend_from_slice(&[0x0F, 0x85]); // jnz rel32
        let at = self.code.len();
        self.imm32(0);
        at
    }

    /// Point a previously reserved rel32 at the current end of the buffer.
    pub fn patch(&mut self, at: usize) {
        let rel = (self.code.len() - (at + 4)) as u32;
        self.code[at..at + 4].copy_from_slice(&rel.to_le_bytes());
    }
}

// ------------------------------------------------------ guest memory access
//
// The guest address space is a flat table of 65,536 page pointers that is
// allocated once at full size, so a guest access inlines to a shift, an
// indexed load of the page pointer, a null test and the access itself. The
// table's address is baked in as an immediate; see `Mem::ptrs_base`.
//
// Three scratch registers are used, all caller-saved and all dead across the
// fast path: `ecx` holds the guest address (and later the value being
// stored), `esi` the write-back value, and `eax`/`rdx` the resolved host
// pointer. None of them survive a `call_step`, which is why the fast path
// contains no calls.

impl Emitter {
    /// `mov ecx, [rbx + r*4]`
    pub fn load_ecx(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x8B, 0x4B, disp(r)]);
    }

    /// `mov ecx, imm32`
    pub fn mov_ecx(&mut self, v: u32) {
        self.code.push(0xB9);
        self.imm32(v);
    }

    /// `add ecx, imm32`
    pub fn add_ecx(&mut self, v: u32) {
        self.code.extend_from_slice(&[0x81, 0xC1]);
        self.imm32(v);
    }

    /// `sub ecx, imm32`
    pub fn sub_ecx(&mut self, v: u32) {
        self.code.extend_from_slice(&[0x81, 0xE9]);
        self.imm32(v);
    }

    /// `add ecx, [rbx + r*4]`
    pub fn add_ecx_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x03, 0x4B, disp(r)]);
    }

    /// `sub ecx, [rbx + r*4]`
    pub fn sub_ecx_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x2B, 0x4B, disp(r)]);
    }

    /// `mov esi, ecx`
    pub fn mov_esi_ecx(&mut self) {
        self.code.extend_from_slice(&[0x89, 0xCE]);
    }

    /// `add esi, imm32`
    pub fn add_esi(&mut self, v: u32) {
        self.code.extend_from_slice(&[0x81, 0xC6]);
        self.imm32(v);
    }

    /// `sub esi, imm32`
    pub fn sub_esi(&mut self, v: u32) {
        self.code.extend_from_slice(&[0x81, 0xEE]);
        self.imm32(v);
    }

    /// `add esi, [rbx + r*4]`
    pub fn add_esi_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x03, 0x73, disp(r)]);
    }

    /// `sub esi, [rbx + r*4]`
    pub fn sub_esi_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x2B, 0x73, disp(r)]);
    }

    /// `mov [rbx + r*4], esi` — the write-back of an indexed transfer.
    pub fn store_esi(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x89, 0x73, disp(r)]);
    }

    /// Forward `jcc rel32`, where `cc` is the second opcode byte (`0x84` = jz,
    /// `0x85` = jnz). Returns the patch site.
    pub fn jcc(&mut self, cc: u8) -> usize {
        self.code.extend_from_slice(&[0x0F, cc]);
        let at = self.code.len();
        self.imm32(0);
        at
    }

    /// `test cl, 3; jnz <patched later>` — leave the fast path when a word
    /// access is not aligned. Returns the patch site.
    pub fn bail_unaligned(&mut self) -> usize {
        self.code.extend_from_slice(&[0xF6, 0xC1, 0x03]); // test cl, 3
        self.jcc(0x85)
    }

    /// Resolve the guest address in `ecx` to `rdx` = page base, `eax` = page
    /// offset. Returns the patch site of the unmapped-page bail.
    ///
    /// `ptrs` is the address of the page-pointer table.
    pub fn page_lookup(&mut self, ptrs: usize) -> usize {
        self.code.extend_from_slice(&[0x89, 0xC8]); // mov eax, ecx
        self.code.extend_from_slice(&[0xC1, 0xE8, 0x10]); // shr eax, 16
        self.code.extend_from_slice(&[0x48, 0xBA]); // movabs rdx, ptrs
        self.code.extend_from_slice(&(ptrs as u64).to_le_bytes());
        self.code.extend_from_slice(&[0x48, 0x8B, 0x14, 0xC2]); // mov rdx, [rdx+rax*8]
        self.code.extend_from_slice(&[0x48, 0x85, 0xD2]); // test rdx, rdx
        let site = self.jcc(0x84); // jz
        self.code.extend_from_slice(&[0x0F, 0xB7, 0xC1]); // movzx eax, cx
        site
    }

    /// `and eax, ~3` — the page offset of a word access, which the guest
    /// architecture takes from `addr & !3`.
    pub fn align_offset(&mut self) {
        self.code.extend_from_slice(&[0x83, 0xE0, 0xFC]);
    }

    /// `mov eax, [rdx + rax]`
    pub fn mem_load_word(&mut self) {
        self.code.extend_from_slice(&[0x8B, 0x04, 0x02]);
    }

    /// `movzx eax, byte [rdx + rax]`
    pub fn mem_load_byte(&mut self) {
        self.code.extend_from_slice(&[0x0F, 0xB6, 0x04, 0x02]);
    }

    /// `mov [rdx + rax], ecx`
    pub fn mem_store_word(&mut self) {
        self.code.extend_from_slice(&[0x89, 0x0C, 0x02]);
    }

    /// `mov [rdx + rax], cl`
    pub fn mem_store_byte(&mut self) {
        self.code.extend_from_slice(&[0x88, 0x0C, 0x02]);
    }
}

// ----------------------------------------------------- condition evaluation
//
// The guest's flags live as four `bool` fields on `Cpu`, which `r14` points
// at. Whoever last set them — a callback, or the interpreter before the block
// was entered — has already stored them there, so a conditional instruction
// can be emitted without emitting flag *generation* first. `disp32` is used
// throughout rather than the shorter `disp8` because the offsets come from
// `offset_of!` on a type with no layout guarantee.

impl Emitter {
    /// Bytes emitted so far, for [`Emitter::truncate`].
    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Discard everything after `n`, undoing a speculative emit.
    pub fn truncate(&mut self, n: usize) {
        self.code.truncate(n);
    }

    /// `cmp byte [r14 + off], 0`
    pub fn cmp_flag(&mut self, off: u32) {
        self.code.extend_from_slice(&[0x41, 0x80, 0xBE]);
        self.imm32(off);
        self.code.push(0x00);
    }

    /// `mov al, [r14 + off]`
    pub fn mov_al_flag(&mut self, off: u32) {
        self.code.extend_from_slice(&[0x41, 0x8A, 0x86]);
        self.imm32(off);
    }

    /// `cmp al, [r14 + off]`
    pub fn cmp_al_flag(&mut self, off: u32) {
        self.code.extend_from_slice(&[0x41, 0x3A, 0x86]);
        self.imm32(off);
    }
}

// ------------------------------------------------------------ shifted operands
//
// A shifted operand is materialised into a scratch register first and the ALU
// op then reads it from there rather than straight out of the register file.
// `ecx` carries a data-processing operand; `edi` carries a memory offset,
// because `ecx` is already the address there.

impl Emitter {
    /// `shl`/`shr`/`sar`/`ror` by an immediate, on the host register whose
    /// 3-bit encoding is `rm`.
    ///
    /// The ARM special cases are folded in: `lsr #0` and `asr #0` mean 32, and
    /// `lsl #0` is the identity. False for `ror #0`, which is RRX and needs the
    /// carry flag this backend does not maintain.
    fn shift_host(&mut self, rm: u8, typ: u32, amount: u32) -> bool {
        let (digit, amount) = match (typ, amount) {
            (0, 0) => return true, // lsl #0: nothing to do
            (0, n) => (4u8, n),    // lsl
            (1, 0) => {
                // lsr #32: every bit shifted out.
                self.code.extend_from_slice(&[0x31, 0xC0 | (rm << 3) | rm]);
                return true;
            }
            (1, n) => (5, n),       // lsr
            (2, 0) => (7, 31),      // asr #32, which is asr #31 for the value
            (2, n) => (7, n),       // asr
            (3, 0) => return false, // rrx
            (_, n) => (1, n),       // ror
        };
        self.code
            .extend_from_slice(&[0xC1, 0xC0 | (digit << 3) | rm, amount as u8]);
        true
    }

    pub fn shift_ecx(&mut self, typ: u32, amount: u32) -> bool {
        self.shift_host(1, typ, amount)
    }

    pub fn shift_edi(&mut self, typ: u32, amount: u32) -> bool {
        self.shift_host(7, typ, amount)
    }

    /// `mov [rbx + r*4], ecx`
    pub fn store_ecx(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x89, 0x4B, disp(r)]);
    }

    /// `mov eax, ecx`
    pub fn mov_eax_ecx(&mut self) {
        self.code.extend_from_slice(&[0x89, 0xC8]);
    }

    /// `not ecx`
    pub fn not_ecx(&mut self) {
        self.code.extend_from_slice(&[0xF7, 0xD1]);
    }

    /// `mov edi, [rbx + r*4]`
    pub fn load_edi(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x8B, 0x7B, disp(r)]);
    }

    /// `add ecx, edi`
    pub fn add_ecx_edi(&mut self) {
        self.code.extend_from_slice(&[0x01, 0xF9]);
    }

    /// `sub ecx, edi`
    pub fn sub_ecx_edi(&mut self) {
        self.code.extend_from_slice(&[0x29, 0xF9]);
    }

    /// `add esi, edi`
    pub fn add_esi_edi(&mut self) {
        self.code.extend_from_slice(&[0x01, 0xFE]);
    }

    /// `sub esi, edi`
    pub fn sub_esi_edi(&mut self) {
        self.code.extend_from_slice(&[0x29, 0xFE]);
    }
}

/// `<op> eax, ecx` for the five commutative-or-not ALU forms the compiler
/// needs when the second operand has been shifted into `ecx`.
macro_rules! eax_ecx {
    ($($name:ident = $opcode:literal),* $(,)?) => {
        impl Emitter {
            $(
                pub fn $name(&mut self) {
                    self.code.extend_from_slice(&[$opcode, 0xC8]);
                }
            )*
        }
    };
}

eax_ecx! {
    add_eax_ecx = 0x01,
    or_eax_ecx = 0x09,
    and_eax_ecx = 0x21,
    sub_eax_ecx = 0x29,
    xor_eax_ecx = 0x31,
}

// -------------------------------------------------------------- flag writes
//
// ARM's NZCV map onto x86's SF, ZF, CF and OF closely enough that a
// flag-setting instruction is the defining operation followed by four
// `setcc`s straight into the guest's flag bytes. `setcc` does not itself
// touch flags, so the four can follow one another, and the result store can
// follow them.

/// Second opcode byte of `0F 9x`, i.e. the `setcc` condition.
pub const SET_S: u8 = 0x98; // sign
pub const SET_Z: u8 = 0x94; // zero
pub const SET_C: u8 = 0x92; // carry
pub const SET_NC: u8 = 0x93; // no carry — ARM's C for a subtraction
pub const SET_O: u8 = 0x90; // overflow

impl Emitter {
    /// `setcc byte [r14 + off]`
    pub fn setcc_flag(&mut self, cc: u8, off: u32) {
        self.code.extend_from_slice(&[0x41, 0x0F, cc, 0x86]);
        self.imm32(off);
    }

    /// `mov byte [r14 + off], imm8`
    pub fn mov_flag_imm(&mut self, off: u32, v: bool) {
        self.code.extend_from_slice(&[0x41, 0xC6, 0x86]);
        self.imm32(off);
        self.code.push(u8::from(v));
    }

    /// `test eax, eax` — for the ops whose x86 form leaves the flags alone.
    pub fn test_eax(&mut self) {
        self.code.extend_from_slice(&[0x85, 0xC0]);
    }
}

// -------------------------------------------- halfword access and multiply
//
// A halfword is the one access that can straddle a page: a word is aligned
// before use and a byte cannot cross at all, but a halfword at offset 0xFFFF
// has its second byte in the next page. The interpreter handles that a byte at
// a time, so the fast path checks for it and hands those over.

impl Emitter {
    /// `cmp eax, 0xFFFE; ja <patched later>` — leave the fast path when a
    /// halfword would run off the end of its page. Returns the patch site.
    pub fn bail_page_cross(&mut self) -> usize {
        self.code.push(0x3D); // cmp eax, imm32
        self.imm32((crate::mem::PAGE_SIZE - 2) as u32);
        self.jcc(0x87) // ja
    }

    /// `movzx eax, word [rdx + rax]`
    pub fn mem_load_half(&mut self) {
        self.code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x02]);
    }

    /// `movsx eax, word [rdx + rax]`
    pub fn mem_load_shalf(&mut self) {
        self.code.extend_from_slice(&[0x0F, 0xBF, 0x04, 0x02]);
    }

    /// `movsx eax, byte [rdx + rax]`
    pub fn mem_load_sbyte(&mut self) {
        self.code.extend_from_slice(&[0x0F, 0xBE, 0x04, 0x02]);
    }

    /// `mov [rdx + rax], cx`
    pub fn mem_store_half(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x89, 0x0C, 0x02]);
    }

    /// `imul eax, [rbx + r*4]` — the low half, which is the same signed or not.
    pub fn imul_reg(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x0F, 0xAF, 0x43, disp(r)]);
    }

    /// `movsxd rax, dword [rbx + r*4]`
    pub fn movsxd_rax(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x48, 0x63, 0x43, disp(r)]);
    }

    /// `movsxd rcx, dword [rbx + r*4]`
    pub fn movsxd_rcx(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x48, 0x63, 0x4B, disp(r)]);
    }

    /// `mov edx, [rbx + r*4]` — zero-extending into `rdx`.
    pub fn load_edx(&mut self, r: u8) {
        self.code.extend_from_slice(&[0x8B, 0x53, disp(r)]);
    }

    /// `imul rax, rcx` — the full 64-bit product.
    pub fn imul_rax_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x0F, 0xAF, 0xC1]);
    }

    /// `shl rcx, 32`
    pub fn shl_rcx_32(&mut self) {
        self.code.extend_from_slice(&[0x48, 0xC1, 0xE1, 0x20]);
    }

    /// `shr rax, 32`
    pub fn shr_rax_32(&mut self) {
        self.code.extend_from_slice(&[0x48, 0xC1, 0xE8, 0x20]);
    }

    /// `or rcx, rdx`
    pub fn or_rcx_rdx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x09, 0xD1]);
    }

    /// `add rax, rcx`
    pub fn add_rax_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x01, 0xC8]);
    }

    /// `test rax, rax`
    pub fn test_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    }
}
