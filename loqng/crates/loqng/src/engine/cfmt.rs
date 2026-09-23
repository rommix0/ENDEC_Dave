//! `printf` and `sscanf` for the translated engine.
//!
//! The engine formats log lines, builds file names and parses its own text
//! dumps, so a usable subset of the C conversions is needed. Anything not
//! handled panics with the specifier rather than quietly printing something
//! else, because a wrong file name is a very confusing failure later.

use crate::xrt::Mem;

/// One conversion, as parsed from the format string.
struct Spec {
    minus: bool,
    zero: bool,
    plus: bool,
    space: bool,
    hash: bool,
    width: Option<usize>,
    prec: Option<usize>,
    long: bool,
    conv: char,
}

fn cstr(m: &Mem, mut a: u32) -> String {
    let mut out = Vec::new();
    if a == 0 {
        return "(null)".into();
    }
    loop {
        let b = m.r8(a);
        if b == 0 {
            break;
        }
        out.push(b);
        a += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn pad(s: String, sp: &Spec) -> String {
    let w = sp.width.unwrap_or(0);
    if s.len() >= w {
        return s;
    }
    let fill = w - s.len();
    if sp.minus {
        s + &" ".repeat(fill)
    } else if sp.zero && !matches!(sp.conv, 's' | 'c') {
        // Zero padding goes after any sign.
        let (sign, rest) = match s.strip_prefix(['-', '+']) {
            Some(r) => (&s[..1], r),
            None => ("", s.as_str()),
        };
        format!("{sign}{}{rest}", "0".repeat(fill))
    } else {
        " ".repeat(fill) + &s
    }
}

fn signed(v: i64, sp: &Spec) -> String {
    let mut s = v.abs().to_string();
    if let Some(p) = sp.prec {
        if s.len() < p {
            s = "0".repeat(p - s.len()) + &s;
        }
    }
    if v < 0 {
        format!("-{s}")
    } else if sp.plus {
        format!("+{s}")
    } else if sp.space {
        format!(" {s}")
    } else {
        s
    }
}

/// Format `fmt` with arguments drawn from `next`, which yields successive
/// 32-bit words exactly as the ARM procedure call standard laid them out.
pub fn format(m: &Mem, fmt: u32, next: &mut dyn FnMut() -> u32) -> Vec<u8> {
    let f = cstr(m, fmt);
    let mut out = String::new();
    let mut it = f.chars().peekable();
    while let Some(ch) = it.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        let mut sp = Spec {
            minus: false,
            zero: false,
            plus: false,
            space: false,
            hash: false,
            width: None,
            prec: None,
            long: false,
            conv: '%',
        };
        loop {
            match it.peek() {
                Some('-') => sp.minus = true,
                Some('0') => sp.zero = true,
                Some('+') => sp.plus = true,
                Some(' ') => sp.space = true,
                Some('#') => sp.hash = true,
                _ => break,
            }
            it.next();
        }
        let mut num = String::new();
        while matches!(it.peek(), Some(c) if c.is_ascii_digit()) {
            num.push(*it.peek().unwrap());
            it.next();
        }
        if it.peek() == Some(&'*') {
            it.next();
            sp.width = Some(next() as i32 as usize);
        } else if !num.is_empty() {
            sp.width = num.parse().ok();
        }
        if it.peek() == Some(&'.') {
            it.next();
            let mut p = String::new();
            while matches!(it.peek(), Some(c) if c.is_ascii_digit()) {
                p.push(*it.peek().unwrap());
                it.next();
            }
            sp.prec = Some(if it.peek() == Some(&'*') {
                it.next();
                next() as usize
            } else {
                p.parse().unwrap_or(0)
            });
        }
        while matches!(it.peek(), Some('l') | Some('h') | Some('L')) {
            if it.peek() == Some(&'l') {
                sp.long = true;
            }
            it.next();
        }
        sp.conv = match it.next() {
            Some(c) => c,
            None => break,
        };

        let piece = match sp.conv {
            '%' => "%".to_string(),
            'd' | 'i' => signed(next() as i32 as i64, &sp),
            'u' => signed(next() as u64 as i64, &sp),
            'x' => {
                let v = format!("{:x}", next());
                if sp.hash {
                    format!("0x{v}")
                } else {
                    v
                }
            }
            'X' => {
                let v = format!("{:X}", next());
                if sp.hash {
                    format!("0X{v}")
                } else {
                    v
                }
            }
            'o' => format!("{:o}", next()),
            'p' => format!("0x{:08x}", next()),
            'c' => ((next() as u8) as char).to_string(),
            's' => {
                let s = cstr(m, next());
                match sp.prec {
                    Some(p) if p < s.len() => s[..p].to_string(),
                    _ => s,
                }
            }
            'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                // ARM OABI passes a double in a register pair, most
                // significant word first, and the pair is 8-byte aligned.
                let hi = next();
                let lo = next();
                let v = f64::from_bits(((hi as u64) << 32) | lo as u64);
                let p = sp.prec.unwrap_or(6);
                match sp.conv {
                    'e' => format!("{v:.p$e}"),
                    'E' => format!("{v:.p$E}"),
                    'g' | 'G' => {
                        let s = format!("{v}");
                        if s.len() > 17 {
                            format!("{v:.p$}")
                        } else {
                            s
                        }
                    }
                    _ => {
                        let s = format!("{v:.p$}");
                        if sp.plus && v >= 0.0 {
                            format!("+{s}")
                        } else {
                            s
                        }
                    }
                }
            }
            other => panic!("printf: unsupported conversion %{other}"),
        };
        out.push_str(&pad(piece, &sp));
    }
    out.into_bytes()
}

/// `sscanf`. Returns how many conversions were assigned; `store` receives
/// each result with the pointer argument it belongs to.
pub fn scan(m: &mut Mem, input: &str, fmt: &str, next: &mut dyn FnMut() -> u32) -> u32 {
    let mut n = 0u32;
    let src: Vec<char> = input.chars().collect();
    let mut i = 0usize;
    let mut it = fmt.chars().peekable();

    let skip_ws = |i: &mut usize, src: &[char]| {
        while *i < src.len() && src[*i].is_whitespace() {
            *i += 1;
        }
    };

    while let Some(ch) = it.next() {
        if ch.is_whitespace() {
            skip_ws(&mut i, &src);
            continue;
        }
        if ch != '%' {
            if i < src.len() && src[i] == ch {
                i += 1;
                continue;
            }
            break;
        }
        let mut suppress = false;
        if it.peek() == Some(&'*') {
            suppress = true;
            it.next();
        }
        let mut width = String::new();
        while matches!(it.peek(), Some(c) if c.is_ascii_digit()) {
            width.push(*it.peek().unwrap());
            it.next();
        }
        let width: Option<usize> = width.parse().ok();
        let mut long = false;
        while matches!(it.peek(), Some('l') | Some('h') | Some('L')) {
            long |= it.peek() == Some(&'l');
            it.next();
        }
        let conv = match it.next() {
            Some(c) => c,
            None => break,
        };
        skip_ws(&mut i, &src);
        let start = i;
        match conv {
            'd' | 'i' | 'u' | 'x' => {
                if i < src.len() && (src[i] == '-' || src[i] == '+') {
                    i += 1;
                }
                let radix = if conv == 'x' { 16 } else { 10 };
                while i < src.len() && src[i].is_digit(radix) && width.is_none_or(|w| i - start < w)
                {
                    i += 1;
                }
                if i == start {
                    break;
                }
                let t: String = src[start..i].iter().collect();
                let v = i64::from_str_radix(t.trim_start_matches('+'), radix).unwrap_or(0);
                if !suppress {
                    let p = next();
                    m.w32(p, v as u32);
                    n += 1;
                }
            }
            'f' | 'e' | 'g' => {
                while i < src.len() && (src[i].is_ascii_digit() || "+-.eE".contains(src[i])) {
                    i += 1;
                }
                if i == start {
                    break;
                }
                let t: String = src[start..i].iter().collect();
                let v: f64 = t.parse().unwrap_or(0.0);
                if !suppress {
                    let p = next();
                    let bits = v.to_bits();
                    if long {
                        m.w32(p, (bits >> 32) as u32);
                        m.w32(p + 4, bits as u32);
                    } else {
                        m.w32(p, (v as f32).to_bits());
                    }
                    n += 1;
                }
            }
            's' => {
                while i < src.len()
                    && !src[i].is_whitespace()
                    && width.is_none_or(|w| i - start < w)
                {
                    i += 1;
                }
                if i == start {
                    break;
                }
                if !suppress {
                    let p = next();
                    for (k, c) in src[start..i].iter().enumerate() {
                        m.w8(p + k as u32, *c as u8);
                    }
                    m.w8(p + (i - start) as u32, 0);
                    n += 1;
                }
            }
            'c' => {
                if i >= src.len() {
                    break;
                }
                if !suppress {
                    let p = next();
                    m.w8(p, src[i] as u8);
                    n += 1;
                }
                i += 1;
            }
            other => panic!("sscanf: unsupported conversion %{other}"),
        }
    }
    n
}
