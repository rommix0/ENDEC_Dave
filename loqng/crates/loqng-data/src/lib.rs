//! Readers for the data files a Loquendo TTS 6 voice ships with.
//!
//! Nothing here executes guest code or depends on the ARM oracle. These are
//! plain file formats, and every one of them is covered by a test that reads
//! the real Dave tree.
//!
//! What each extension is, and how far this crate gets with it:
//!
//! | ext | what | status |
//! |---|---|---|
//! | `.session` `.vde` `.lde` | descriptor files, `"Key" = "Value"` | complete |
//! | `.lex` | abbreviation/acronym expansions | complete |
//! | `.sde` | signal descriptor: recordings and unit halves | complete — base-93 fields, chained unit spans |
//! | `.bin` | the unit bank | header only |
//! | `.phd` | phonetic dictionary | complete — delta decode, records, binary search |
//! | `.atm` | grapheme-to-phoneme automaton | complete — every byte of both vendor files accounted for |
//!
//! See `PLAN.md` §5, Phase 1.

pub mod atm;
pub mod bank;
pub mod descriptor;
pub mod lexicon;
pub mod phd;
pub mod sde;
pub mod text;

pub use bank::BankHeader;
pub use descriptor::Descriptor;
pub use lexicon::Lexicon;
pub use phd::Phd;
pub use sde::{Sde, SdeRecord};

/// Everything in this crate fails the same way: a file that does not look like
/// what its extension claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub file: String,
    /// 1-based, or 0 when the problem is not tied to a line.
    pub line: usize,
    pub what: String,
}

impl Error {
    pub fn new(file: &str, line: usize, what: impl Into<String>) -> Self {
        Error {
            file: file.to_string(),
            line,
            what: what.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(f, "{}: {}", self.file, self.what)
        } else {
            write!(f, "{}:{}: {}", self.file, self.line, self.what)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
