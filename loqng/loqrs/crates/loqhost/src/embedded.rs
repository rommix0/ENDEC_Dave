//! Engine modules baked in at build time; see build.rs.

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// Look up one embedded module by file name.
pub fn get(name: &str) -> Option<&'static [u8]> {
    MODULES.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

/// Total size of everything baked in.
pub fn total_bytes() -> usize {
    MODULES.iter().map(|(_, b)| b.len()).sum()
}
