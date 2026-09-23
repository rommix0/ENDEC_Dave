//! `.lex` — the abbreviation, acronym and symbol expansions `Les` applies.
//!
//! High-bit-encoded text (see [`crate::text`]). The first line is the lexicon's
//! own name — `EnglishUs6.9`, `Dave` — and the rest is `;` comments and
//! `"word" = "expansion"` pairs, which is the same syntax as a descriptor.
//!
//! The vendor header documents one wrinkle, and the files use it: matching is
//! case-insensitive **unless** the word begins with `\x`, in which case the
//! rest is matched exactly. That is how `\xCC`, `\xCc` and `\xcc` coexist as
//! three different entries in the Italian source file this format came from.

use crate::descriptor::Descriptor;
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexEntry {
    pub word: String,
    pub expansion: String,
    /// Set when the source key carried the `\x` prefix.
    pub case_sensitive: bool,
    pub line: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lexicon {
    /// The first line of the file.
    pub name: String,
    pub entries: Vec<LexEntry>,
}

impl Lexicon {
    pub fn parse(name: &str, text: &str) -> Result<Lexicon> {
        // Line 1 is the lexicon's name in the vendor files, but not in every
        // one: `sage.lex` in the SAGE tree opens straight with its only entry.
        // Treat line 1 as a name only when it is not itself a `"k" = "v"` pair,
        // or that entry is silently lost.
        let first = text.lines().next().unwrap_or("").trim();
        let has_header = !first.is_empty() && !first.starts_with(';') && !looks_like_entry(first);
        let header = if has_header {
            first.to_string()
        } else {
            String::new()
        };

        // Blank the header line rather than dropping it, so the descriptor
        // parser still reports true line numbers.
        let rest: String = text
            .lines()
            .enumerate()
            .map(|(i, l)| if i == 0 && has_header { "" } else { l })
            .collect::<Vec<_>>()
            .join("\n");

        let d = Descriptor::parse(name, &rest)?;
        let entries = d
            .entries
            .into_iter()
            .map(|e| {
                // The index syntax has no meaning here, so a key like `a[1]`
                // must be put back together before it is used as a word.
                let key = match e.index {
                    Some(i) => format!("{}[{}]", e.key, i),
                    None => e.key,
                };
                let (word, case_sensitive) = match key.strip_prefix("\\x") {
                    Some(w) => (w.to_string(), true),
                    None => (key, false),
                };
                LexEntry {
                    word,
                    expansion: e.value,
                    case_sensitive,
                    line: e.line,
                }
            })
            .collect();

        Ok(Lexicon {
            name: header,
            entries,
        })
    }

    pub fn parse_bytes(name: &str, raw: &[u8]) -> Result<Lexicon> {
        Lexicon::parse(name, &crate::text::decode_auto(raw))
    }

    /// Look a word up the way the engine does: exact match first, so a
    /// case-sensitive entry always beats a case-insensitive one.
    pub fn lookup(&self, word: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.case_sensitive && e.word == word)
            .or_else(|| {
                self.entries
                    .iter()
                    .find(|e| !e.case_sensitive && e.word.eq_ignore_ascii_case(word))
            })
            .map(|e| e.expansion.as_str())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Whether a line is a `"key" = "value"` pair rather than a bare name.
fn looks_like_entry(s: &str) -> bool {
    match crate::descriptor::find_separator(s) {
        Some(eq) => !s[..eq].trim().is_empty(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEX: &str = concat!(
        "EnglishUs6.9\n",
        ";;; This file contains abbreviations, acronyms, symbols\n",
        "; The syntax is: \"word\" = \"expansion\"\n",
        "\n",
        "\"s. Martin\" = \"saint Martin\"\n",
        "\"%\" = \"per cent\"\n",
        "\"\\xCC\" = \"Carabinieri\"\n",
        "\"cc\" = \"carbon copy\"\n",
    );

    /// The three-way case the vendor header documents: `\x` entries that
    /// differ only by case are three distinct entries.
    const THREE: &str = concat!(
        "EnglishUs6.9\n",
        "\"\\xCC\" = \"Carabinieri\"\n",
        "\"\\xCc\" = \"carbon copy\"\n",
        "\"\\xcc\" = \"conto corrente\"\n",
    );

    #[test]
    fn reads_name_and_entries() {
        let l = Lexicon::parse("EnglishUs6.9.lex", LEX).unwrap();
        assert_eq!(l.name, "EnglishUs6.9");
        assert_eq!(l.len(), 4);
    }

    #[test]
    fn multi_word_keys_survive() {
        let l = Lexicon::parse("x.lex", LEX).unwrap();
        assert_eq!(l.lookup("s. Martin"), Some("saint Martin"));
        assert_eq!(l.lookup("S. MARTIN"), Some("saint Martin"));
    }

    #[test]
    fn an_exact_entry_outranks_a_loose_one() {
        let l = Lexicon::parse("x.lex", LEX).unwrap();
        // `\xCC` matches "CC" exactly, so it beats the case-insensitive "cc".
        assert_eq!(l.lookup("CC"), Some("Carabinieri"));
        // Any other casing misses it and falls through to "cc".
        assert_eq!(l.lookup("cc"), Some("carbon copy"));
        assert_eq!(l.lookup("Cc"), Some("carbon copy"));

        let exact = l.entries.iter().find(|e| e.case_sensitive).unwrap();
        assert_eq!(exact.word, "CC");
        assert_eq!(exact.expansion, "Carabinieri");
    }

    #[test]
    fn three_casings_of_one_word_stay_three_entries() {
        let l = Lexicon::parse("x.lex", THREE).unwrap();
        assert_eq!(l.len(), 3);
        assert_eq!(l.lookup("CC"), Some("Carabinieri"));
        assert_eq!(l.lookup("Cc"), Some("carbon copy"));
        assert_eq!(l.lookup("cc"), Some("conto corrente"));
        // Nothing here is case-insensitive, so an unseen casing finds nothing.
        assert_eq!(l.lookup("cC"), None);
    }

    #[test]
    fn symbols_are_entries_too() {
        let l = Lexicon::parse("x.lex", LEX).unwrap();
        assert_eq!(l.lookup("%"), Some("per cent"));
    }

    /// `sage.lex` in the SAGE tree has no name line — its only entry is line 1.
    /// Consuming that as a header loses the entry entirely.
    #[test]
    fn a_lexicon_with_no_name_line_keeps_its_first_entry() {
        let l = Lexicon::parse("sage.lex", "\"endec\" = \"\\fE-n-d-`E-kh\"\n\n").unwrap();
        assert_eq!(l.name, "");
        assert_eq!(l.len(), 1);
        assert_eq!(l.lookup("endec"), Some("\\fE-n-d-`E-kh"));
    }

    #[test]
    fn an_empty_lexicon_is_not_an_error() {
        let l = Lexicon::parse("user.lex", "").unwrap();
        assert_eq!(l.name, "");
        assert!(l.is_empty());
    }

    #[test]
    fn a_comment_first_line_is_not_a_name() {
        let l = Lexicon::parse("x.lex", "; just a comment\n\"a\" = \"b\"\n").unwrap();
        assert_eq!(l.name, "");
        assert_eq!(l.lookup("a"), Some("b"));
    }
}
