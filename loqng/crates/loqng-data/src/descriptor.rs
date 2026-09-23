//! `.session`, `.vde` and `.lde` files.
//!
//! One flat syntax serves all three: `; comment` lines, blank lines, and
//! `"Key" = "Value"` with arbitrary whitespace around the `=`. A key may carry
//! a bracketed index — `"Lexicon[1]"`, `"SignalDescriptor[0]"` — and those
//! repeat, so order is preserved rather than collapsed into a map.
//!
//! `"Include"` names another descriptor to pull in; `Dave.vde` includes
//! `EnglishUs6.9.lde` that way. Resolution is left to the caller, which knows
//! the data root, and `includes()` reports what to load.
//!
//! Lookups are case-insensitive because the engine compares these keys with
//! `ELQstricmp`.

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: String,
    /// The `N` of `Key[N]`, absent for a plain key.
    pub index: Option<u32>,
    pub value: String,
    /// 1-based line in the source file, for diagnostics.
    pub line: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Descriptor {
    pub entries: Vec<Entry>,
}

impl Descriptor {
    /// Parse one descriptor file. `name` is used only in error messages.
    pub fn parse(name: &str, text: &str) -> Result<Descriptor> {
        let mut entries = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = i + 1;
            let s = raw.trim();
            if s.is_empty() || s.starts_with(';') {
                continue;
            }
            let (key, value) = split_pair(name, line, s)?;
            let (key, index) = split_index(&key);
            entries.push(Entry {
                key,
                index,
                value,
                line,
            });
        }
        Ok(Descriptor { entries })
    }

    pub fn parse_bytes(name: &str, raw: &[u8]) -> Result<Descriptor> {
        Descriptor::parse(name, &crate::text::decode_auto(raw))
    }

    /// First value for a key, ignoring any index.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.key.eq_ignore_ascii_case(key))
            .map(|e| e.value.as_str())
    }

    /// Value for an exact `Key[index]`.
    pub fn get_indexed(&self, key: &str, index: u32) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.key.eq_ignore_ascii_case(key) && e.index == Some(index))
            .map(|e| e.value.as_str())
    }

    /// Every value for a key, in file order, indexed or not.
    pub fn all(&self, key: &str) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| e.key.eq_ignore_ascii_case(key))
            .map(|e| e.value.as_str())
            .collect()
    }

    /// Indexed values for a key, sorted by index.
    ///
    /// `Dave.vde` numbers `SignalDescriptor` and `BinFileName` from 0 and they
    /// pair up positionally, so the index matters and file order does not.
    pub fn indexed(&self, key: &str) -> Vec<(u32, &str)> {
        let mut v: Vec<(u32, &str)> = self
            .entries
            .iter()
            .filter(|e| e.key.eq_ignore_ascii_case(key))
            .filter_map(|e| e.index.map(|i| (i, e.value.as_str())))
            .collect();
        v.sort_by_key(|(i, _)| *i);
        v
    }

    /// Descriptors named by `"Include"`, relative to this file's directory.
    pub fn includes(&self) -> Vec<&str> {
        self.all("Include")
    }

    /// Merge an included descriptor. Existing entries win, matching the
    /// engine's behaviour that a voice descriptor overrides the language
    /// descriptor it includes.
    pub fn merge_under(&mut self, other: &Descriptor) {
        for e in &other.entries {
            let present = self
                .entries
                .iter()
                .any(|x| x.key.eq_ignore_ascii_case(&e.key) && x.index == e.index);
            if !present {
                self.entries.push(e.clone());
            }
        }
    }
}

/// `"Key" = "Value"` -> `(Key, Value)`.
///
/// Quotes are required on both sides in every vendor file seen so far, but a
/// bare right-hand side is accepted because it costs nothing and the engine's
/// INI reader tolerates it.
fn split_pair(file: &str, line: usize, s: &str) -> Result<(String, String)> {
    let eq = find_separator(s).ok_or_else(|| Error::new(file, line, format!("no `=` in `{s}`")))?;
    let key = unquote(s[..eq].trim());
    let value = unquote(s[eq + 1..].trim());
    if key.is_empty() {
        return Err(Error::new(file, line, format!("empty key in `{s}`")));
    }
    Ok((key, value))
}

/// The `=` that separates key from value, skipping any inside quotes.
pub(crate) fn find_separator(s: &str) -> Option<usize> {
    let mut quoted = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '=' if !quoted => return Some(i),
            _ => {}
        }
    }
    None
}

fn unquote(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

/// `Lexicon[1]` -> `("Lexicon", Some(1))`.
fn split_index(key: &str) -> (String, Option<u32>) {
    let Some(open) = key.rfind('[') else {
        return (key.to_string(), None);
    };
    if !key.ends_with(']') {
        return (key.to_string(), None);
    }
    match key[open + 1..key.len() - 1].parse::<u32>() {
        Ok(n) => (key[..open].trim().to_string(), Some(n)),
        Err(_) => (key.to_string(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VDE: &str = concat!(
        "; Voice descriptor file for Loquendo TTS\n",
        "\n",
        "\"Include\"\t\t=\t\"EnglishUs6.9.lde\"\n",
        "\"Name\"\t\t\t=\t\"Dave-19200\"\n",
        ";\"Lexicon[1]\"\t\t=\t\"EnglishUs/Dave/Dave.lex\"\n",
        "\"Lexicon[1]\"\t\t=\t\"EnglishUs/Dave/DaveAU.lex\"\n",
        "\"SignalDescriptor[0]\" \t=\t\"EnglishUs/Dave/Dave\"\n",
        "\"SignalDescriptor[2]\"   =\t\"EnglishUs/Dave/DaveAU\"\n",
        "\"SignalDescriptor[1]\" \t=\t\"EnglishUs/Dave/DaveGilded\"\n",
    );

    #[test]
    fn parses_a_voice_descriptor() {
        let d = Descriptor::parse("Dave.vde", VDE).unwrap();
        assert_eq!(d.get("Name"), Some("Dave-19200"));
        assert_eq!(d.get("name"), Some("Dave-19200"));
        assert_eq!(d.includes(), vec!["EnglishUs6.9.lde"]);
    }

    #[test]
    fn comments_are_not_entries() {
        let d = Descriptor::parse("Dave.vde", VDE).unwrap();
        // The commented-out Dave.lex line must not win over DaveAU.lex.
        assert_eq!(
            d.get_indexed("Lexicon", 1),
            Some("EnglishUs/Dave/DaveAU.lex")
        );
        assert_eq!(d.all("Lexicon").len(), 1);
    }

    #[test]
    fn indexed_entries_sort_by_index_not_file_order() {
        let d = Descriptor::parse("Dave.vde", VDE).unwrap();
        let sd = d.indexed("SignalDescriptor");
        assert_eq!(
            sd,
            vec![
                (0, "EnglishUs/Dave/Dave"),
                (1, "EnglishUs/Dave/DaveGilded"),
                (2, "EnglishUs/Dave/DaveAU"),
            ]
        );
    }

    #[test]
    fn merge_keeps_the_more_specific_value() {
        let mut voice = Descriptor::parse("v", "\"Description\" = \"voice\"\n").unwrap();
        let lang = Descriptor::parse(
            "l",
            "\"Description\" = \"lang\"\n\"Library\" = \"LoqEnglish6.9\"\n",
        )
        .unwrap();
        voice.merge_under(&lang);
        assert_eq!(voice.get("Description"), Some("voice"));
        assert_eq!(voice.get("Library"), Some("LoqEnglish6.9"));
    }

    #[test]
    fn a_value_may_contain_an_equals_sign() {
        let d = Descriptor::parse("s", "\"Extra\" = \"Language=English\"\n").unwrap();
        assert_eq!(d.get("Extra"), Some("Language=English"));
    }
}
