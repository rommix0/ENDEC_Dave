//! C `printf`/`scanf` formatting against guest memory.
//!
//! Varargs follow APCS: words come from r0-r3 and then the caller's stack, with
//! no 8-byte alignment for 64-bit types, and doubles arrive most-significant
//! word first.

use armemu::{Cpu, Mem};

pub struct VaList {
    next_reg: usize,
    sp: u32,
}

impl VaList {
    /// `first_reg` is the register holding the first variadic argument.
    pub fn new(first_reg: usize, sp: u32) -> Self {
        VaList {
            next_reg: first_reg,
            sp,
        }
    }

    pub fn word(&mut self, cpu: &Cpu, mem: &Mem) -> u32 {
        if self.next_reg < 4 {
            let v = cpu.r[self.next_reg];
            self.next_reg += 1;
            v
        } else {
            let v = mem.read_u32(self.sp).unwrap_or(0);
            self.sp = self.sp.wrapping_add(4);
            v
        }
    }

    pub fn double(&mut self, cpu: &Cpu, mem: &Mem) -> f64 {
        let hi = self.word(cpu, mem);
        let lo = self.word(cpu, mem);
        f64::from_bits(((hi as u64) << 32) | lo as u64)
    }

    pub fn u64(&mut self, cpu: &Cpu, mem: &Mem) -> u64 {
        let lo = self.word(cpu, mem);
        let hi = self.word(cpu, mem);
        ((hi as u64) << 32) | lo as u64
    }
}

#[derive(Default)]
struct Spec {
    minus: bool,
    plus: bool,
    space: bool,
    hash: bool,
    zero: bool,
    width: Option<usize>,
    prec: Option<usize>,
    long_long: bool,
    short_: bool,
    byte: bool,
}

fn pad(out: &mut Vec<u8>, body: &[u8], s: &Spec, numeric_prefix: usize) {
    let width = s.width.unwrap_or(0);
    if body.len() >= width {
        out.extend_from_slice(body);
        return;
    }
    let fill = width - body.len();
    if s.minus {
        out.extend_from_slice(body);
        out.extend(std::iter::repeat(b' ').take(fill));
    } else if s.zero && s.prec.is_none() {
        out.extend_from_slice(&body[..numeric_prefix]);
        out.extend(std::iter::repeat(b'0').take(fill));
        out.extend_from_slice(&body[numeric_prefix..]);
    } else {
        out.extend(std::iter::repeat(b' ').take(fill));
        out.extend_from_slice(body);
    }
}

fn fmt_int(v: i64, s: &Spec) -> (Vec<u8>, usize) {
    let neg = v < 0;
    let mag = v.unsigned_abs();
    let mut digits = mag.to_string().into_bytes();
    if let Some(p) = s.prec {
        while digits.len() < p {
            digits.insert(0, b'0');
        }
        if p == 0 && mag == 0 {
            digits.clear();
        }
    }
    let mut body = Vec::new();
    if neg {
        body.push(b'-');
    } else if s.plus {
        body.push(b'+');
    } else if s.space {
        body.push(b' ');
    }
    let prefix = body.len();
    body.extend_from_slice(&digits);
    (body, prefix)
}

fn fmt_uint(v: u64, base: u32, upper: bool, s: &Spec) -> (Vec<u8>, usize) {
    let mut digits = match base {
        8 => format!("{v:o}"),
        16 if upper => format!("{v:X}"),
        16 => format!("{v:x}"),
        _ => format!("{v}"),
    }
    .into_bytes();
    if let Some(p) = s.prec {
        while digits.len() < p {
            digits.insert(0, b'0');
        }
        if p == 0 && v == 0 {
            digits.clear();
        }
    }
    let mut body = Vec::new();
    if s.hash && v != 0 {
        match base {
            8 => body.push(b'0'),
            16 => body.extend_from_slice(if upper { b"0X" } else { b"0x" }),
            _ => {}
        }
    }
    let prefix = body.len();
    body.extend_from_slice(&digits);
    (body, prefix)
}

fn fmt_exp(x: f64, prec: usize, upper: bool, s: &Spec) -> Vec<u8> {
    let mut mant = x.abs();
    let mut exp = 0i32;
    if mant != 0.0 && mant.is_finite() {
        exp = mant.log10().floor() as i32;
        mant /= 10f64.powi(exp);
        // log10 rounding can leave the mantissa just outside [1, 10).
        if mant >= 10.0 {
            mant /= 10.0;
            exp += 1;
        } else if mant < 1.0 {
            mant *= 10.0;
            exp -= 1;
        }
        // Rounding to `prec` digits can push it to 10.0.
        let scaled = format!("{:.*}", prec, mant);
        if scaled.starts_with("10") {
            mant /= 10.0;
            exp += 1;
        }
    }
    let mut body = Vec::new();
    if x.is_sign_negative() {
        body.push(b'-');
    } else if s.plus {
        body.push(b'+');
    } else if s.space {
        body.push(b' ');
    }
    let mut num = format!("{:.*}", prec, mant);
    if prec == 0 && s.hash {
        num.push('.');
    }
    body.extend_from_slice(num.as_bytes());
    body.push(if upper { b'E' } else { b'e' });
    body.push(if exp < 0 { b'-' } else { b'+' });
    let a = exp.unsigned_abs();
    body.extend_from_slice(format!("{a:02}").as_bytes());
    body
}

fn fmt_float(x: f64, conv: u8, s: &Spec) -> (Vec<u8>, usize) {
    let upper = conv.is_ascii_uppercase();
    if x.is_nan() || x.is_infinite() {
        let mut body = Vec::new();
        if x.is_sign_negative() {
            body.push(b'-');
        } else if s.plus {
            body.push(b'+');
        } else if s.space {
            body.push(b' ');
        }
        let word: &[u8] = if x.is_nan() {
            if upper {
                b"NAN"
            } else {
                b"nan"
            }
        } else if upper {
            b"INF"
        } else {
            b"inf"
        };
        let prefix = body.len();
        body.extend_from_slice(word);
        return (body, prefix);
    }

    match conv.to_ascii_lowercase() {
        b'f' => {
            let prec = s.prec.unwrap_or(6);
            let mut body = Vec::new();
            if x.is_sign_negative() {
                body.push(b'-');
            } else if s.plus {
                body.push(b'+');
            } else if s.space {
                body.push(b' ');
            }
            let prefix = body.len();
            let mut num = format!("{:.*}", prec, x.abs());
            if prec == 0 && s.hash {
                num.push('.');
            }
            body.extend_from_slice(num.as_bytes());
            (body, prefix)
        }
        b'e' => {
            let body = fmt_exp(x, s.prec.unwrap_or(6), upper, s);
            let prefix = usize::from(!body.is_empty() && !body[0].is_ascii_digit());
            (body, prefix)
        }
        _ => {
            // %g: pick %e or %f by exponent, then trim trailing zeros.
            let p = match s.prec {
                Some(0) | None if s.prec == Some(0) => 1,
                Some(v) => v.max(1),
                None => 6,
            };
            let exp = if x == 0.0 {
                0
            } else {
                let mut e = x.abs().log10().floor() as i32;
                let scaled = format!("{:.*}", p - 1, x.abs() / 10f64.powi(e));
                if scaled.starts_with("10") {
                    e += 1;
                }
                e
            };
            let mut body = if exp < -4 || exp >= p as i32 {
                let mut sp = Spec {
                    prec: Some(p - 1),
                    ..Spec::default()
                };
                sp.plus = s.plus;
                sp.space = s.space;
                sp.hash = s.hash;
                fmt_exp(x, p - 1, upper, &sp)
            } else {
                let dec = (p as i32 - 1 - exp).max(0) as usize;
                let mut b = Vec::new();
                if x.is_sign_negative() {
                    b.push(b'-');
                } else if s.plus {
                    b.push(b'+');
                } else if s.space {
                    b.push(b' ');
                }
                b.extend_from_slice(format!("{:.*}", dec, x.abs()).as_bytes());
                b
            };
            if !s.hash {
                trim_g(&mut body);
            }
            let prefix = usize::from(!body.is_empty() && !body[0].is_ascii_digit());
            (body, prefix)
        }
    }
}

fn trim_g(body: &mut Vec<u8>) {
    let epos = body.iter().position(|c| *c == b'e' || *c == b'E');
    let (mant_end, tail) = match epos {
        Some(i) => (i, body[i..].to_vec()),
        None => (body.len(), Vec::new()),
    };
    if !body[..mant_end].contains(&b'.') {
        return;
    }
    let mut end = mant_end;
    while end > 0 && body[end - 1] == b'0' {
        end -= 1;
    }
    if end > 0 && body[end - 1] == b'.' {
        end -= 1;
    }
    body.truncate(end);
    body.extend_from_slice(&tail);
}

/// Render a C format string. Returns the bytes that would be written.
pub fn format(cpu: &Cpu, mem: &Mem, fmt_addr: u32, va: &mut VaList) -> Vec<u8> {
    let fmt = mem.read_cstr(fmt_addr).unwrap_or_default();
    let mut out = Vec::with_capacity(fmt.len() + 32);
    let mut i = 0;

    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= fmt.len() {
            break;
        }
        if fmt[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }

        let mut s = Spec::default();
        loop {
            match fmt.get(i) {
                Some(b'-') => s.minus = true,
                Some(b'+') => s.plus = true,
                Some(b' ') => s.space = true,
                Some(b'#') => s.hash = true,
                Some(b'0') => s.zero = true,
                _ => break,
            }
            i += 1;
        }
        if fmt.get(i) == Some(&b'*') {
            let w = va.word(cpu, mem) as i32;
            if w < 0 {
                s.minus = true;
                s.width = Some((-w) as usize);
            } else {
                s.width = Some(w as usize);
            }
            i += 1;
        } else {
            let start = i;
            while fmt.get(i).is_some_and(|c| c.is_ascii_digit()) {
                i += 1;
            }
            if i > start {
                s.width = std::str::from_utf8(&fmt[start..i])
                    .ok()
                    .and_then(|v| v.parse().ok());
            }
        }
        if fmt.get(i) == Some(&b'.') {
            i += 1;
            if fmt.get(i) == Some(&b'*') {
                let p = va.word(cpu, mem) as i32;
                s.prec = Some(p.max(0) as usize);
                i += 1;
            } else {
                let start = i;
                while fmt.get(i).is_some_and(|c| c.is_ascii_digit()) {
                    i += 1;
                }
                s.prec = Some(
                    std::str::from_utf8(&fmt[start..i])
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                );
            }
        }
        loop {
            match fmt.get(i) {
                Some(b'l') => {
                    if s.long_long {
                        // already seen one 'l'
                    }
                    s.long_long = fmt.get(i + 1) == Some(&b'l');
                }
                Some(b'h') => {
                    if fmt.get(i + 1) == Some(&b'h') {
                        s.byte = true;
                    } else {
                        s.short_ = true;
                    }
                }
                Some(b'L') | Some(b'q') | Some(b'j') | Some(b'z') | Some(b't') => {}
                _ => break,
            }
            i += 1;
        }

        let conv = match fmt.get(i) {
            Some(c) => *c,
            None => break,
        };
        i += 1;

        match conv {
            b'd' | b'i' => {
                let v = if s.long_long {
                    va.u64(cpu, mem) as i64
                } else {
                    let w = va.word(cpu, mem);
                    if s.short_ {
                        w as u16 as i16 as i64
                    } else if s.byte {
                        w as u8 as i8 as i64
                    } else {
                        w as i32 as i64
                    }
                };
                let (body, p) = fmt_int(v, &s);
                pad(&mut out, &body, &s, p);
            }
            b'u' | b'x' | b'X' | b'o' => {
                let v = if s.long_long {
                    va.u64(cpu, mem)
                } else {
                    let w = va.word(cpu, mem);
                    if s.short_ {
                        w as u16 as u64
                    } else if s.byte {
                        w as u8 as u64
                    } else {
                        w as u64
                    }
                };
                let base = match conv {
                    b'o' => 8,
                    b'x' | b'X' => 16,
                    _ => 10,
                };
                let (body, p) = fmt_uint(v, base, conv == b'X', &s);
                pad(&mut out, &body, &s, p);
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                let x = va.double(cpu, mem);
                let (body, p) = fmt_float(x, conv, &s);
                pad(&mut out, &body, &s, p);
            }
            b'c' => {
                let v = va.word(cpu, mem) as u8;
                pad(&mut out, &[v], &s, 0);
            }
            b's' => {
                let ptr = va.word(cpu, mem);
                let mut bytes = if ptr == 0 {
                    b"(null)".to_vec()
                } else {
                    mem.read_cstr(ptr).unwrap_or_default()
                };
                if let Some(p) = s.prec {
                    bytes.truncate(p);
                }
                pad(&mut out, &bytes, &s, 0);
            }
            b'p' => {
                let v = va.word(cpu, mem);
                let body = format!("0x{v:x}").into_bytes();
                pad(&mut out, &body, &s, 0);
            }
            b'n' => {
                let _ = va.word(cpu, mem);
            }
            other => {
                out.push(b'%');
                out.push(other);
            }
        }
    }
    out
}

/// Minimal `sscanf`. Returns the number of successful conversions.
pub fn scan(cpu: &Cpu, mem: &mut Mem, input: &[u8], fmt_addr: u32, va: &mut VaList) -> i32 {
    let fmt = mem.read_cstr(fmt_addr).unwrap_or_default();
    let mut ip = 0usize;
    let mut fp = 0usize;
    let mut count = 0i32;

    while fp < fmt.len() {
        let f = fmt[fp];
        if f.is_ascii_whitespace() {
            while ip < input.len() && input[ip].is_ascii_whitespace() {
                ip += 1;
            }
            fp += 1;
            continue;
        }
        if f != b'%' {
            if ip < input.len() && input[ip] == f {
                ip += 1;
                fp += 1;
                continue;
            }
            return count;
        }
        fp += 1;
        let suppress = fmt.get(fp) == Some(&b'*');
        if suppress {
            fp += 1;
        }
        let start = fp;
        while fmt.get(fp).is_some_and(|c| c.is_ascii_digit()) {
            fp += 1;
        }
        let width: Option<usize> = if fp > start {
            std::str::from_utf8(&fmt[start..fp])
                .ok()
                .and_then(|v| v.parse().ok())
        } else {
            None
        };
        let mut long_ = false;
        while matches!(
            fmt.get(fp),
            Some(b'l') | Some(b'h') | Some(b'L') | Some(b'q')
        ) {
            if fmt[fp] == b'l' || fmt[fp] == b'L' {
                long_ = true;
            }
            fp += 1;
        }
        let conv = match fmt.get(fp) {
            Some(c) => *c,
            None => break,
        };
        fp += 1;

        if conv != b'c' && conv != b'[' {
            while ip < input.len() && input[ip].is_ascii_whitespace() {
                ip += 1;
            }
        }
        if ip >= input.len() && conv != b'n' {
            return if count == 0 { -1 } else { count };
        }

        let limit = width.unwrap_or(usize::MAX);
        match conv {
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' => {
                let base = match conv {
                    b'x' | b'X' => 16,
                    b'o' => 8,
                    _ => 10,
                };
                let s0 = ip;
                if ip < input.len() && (input[ip] == b'-' || input[ip] == b'+') {
                    ip += 1;
                }
                while ip < input.len()
                    && ip - s0 < limit
                    && (input[ip] as char).is_digit(base as u32)
                {
                    ip += 1;
                }
                if ip == s0 {
                    return count;
                }
                let text = String::from_utf8_lossy(&input[s0..ip]).into_owned();
                let val = i64::from_str_radix(text.trim_start_matches('+'), base).unwrap_or(0);
                if !suppress {
                    let ptr = va.word(cpu, mem);
                    let _ = mem.write_u32(ptr, val as u32);
                }
                count += 1;
            }
            b'f' | b'e' | b'g' => {
                let s0 = ip;
                if ip < input.len() && (input[ip] == b'-' || input[ip] == b'+') {
                    ip += 1;
                }
                while ip < input.len()
                    && ip - s0 < limit
                    && (input[ip].is_ascii_digit() || input[ip] == b'.')
                {
                    ip += 1;
                }
                if ip < input.len() && (input[ip] | 32) == b'e' {
                    let save = ip;
                    ip += 1;
                    if ip < input.len() && (input[ip] == b'-' || input[ip] == b'+') {
                        ip += 1;
                    }
                    if ip < input.len() && input[ip].is_ascii_digit() {
                        while ip < input.len() && input[ip].is_ascii_digit() {
                            ip += 1;
                        }
                    } else {
                        ip = save;
                    }
                }
                if ip == s0 {
                    return count;
                }
                let val: f64 = String::from_utf8_lossy(&input[s0..ip])
                    .parse()
                    .unwrap_or(0.0);
                if !suppress {
                    let ptr = va.word(cpu, mem);
                    if long_ {
                        let bits = val.to_bits();
                        let _ = mem.write_u32(ptr, (bits >> 32) as u32);
                        let _ = mem.write_u32(ptr + 4, bits as u32);
                    } else {
                        let _ = mem.write_u32(ptr, (val as f32).to_bits());
                    }
                }
                count += 1;
            }
            b's' => {
                let s0 = ip;
                while ip < input.len() && ip - s0 < limit && !input[ip].is_ascii_whitespace() {
                    ip += 1;
                }
                if ip == s0 {
                    return count;
                }
                if !suppress {
                    let ptr = va.word(cpu, mem);
                    let _ = mem.write_cstr(ptr, &input[s0..ip]);
                }
                count += 1;
            }
            b'c' => {
                let n = width.unwrap_or(1);
                if ip + n > input.len() {
                    return count;
                }
                if !suppress {
                    let ptr = va.word(cpu, mem);
                    let _ = mem.write_bytes(ptr, &input[ip..ip + n]);
                }
                ip += n;
                count += 1;
            }
            b'[' => {
                let negate = fmt.get(fp) == Some(&b'^');
                if negate {
                    fp += 1;
                }
                let mut set = Vec::new();
                if fmt.get(fp) == Some(&b']') {
                    set.push(b']');
                    fp += 1;
                }
                while fp < fmt.len() && fmt[fp] != b']' {
                    set.push(fmt[fp]);
                    fp += 1;
                }
                fp += 1;
                let s0 = ip;
                while ip < input.len() && ip - s0 < limit && (set.contains(&input[ip]) != negate) {
                    ip += 1;
                }
                if ip == s0 {
                    return count;
                }
                if !suppress {
                    let ptr = va.word(cpu, mem);
                    let _ = mem.write_cstr(ptr, &input[s0..ip]);
                }
                count += 1;
            }
            b'n' => {
                if !suppress {
                    let ptr = va.word(cpu, mem);
                    let _ = mem.write_u32(ptr, ip as u32);
                }
            }
            _ => return count,
        }
    }
    count
}
