//! `Top` — text preprocessing. **Phase 7.**
//!
//! 68,876 bytes are attributed to this subsystem, but 82,204 of that bucket is
//! expat and is not ported: SSML is deferred (`PLAN.md` §10.2). What is in
//! scope is about 30 KB —
//!
//! ```text
//! TopTextConvert          0x633b8   charset conversion
//! TopXMLout_utf8toAnsi    0x63cd4
//! TopLoadFindReplaceTable 0x65dc8   the user find/replace table
//! TopRegCompile           0x73354   Loquendo's own regex engine
//! TopRegFind              0x73fa4
//! TopRegSetReplaceExp     0x74718
//! TopRegGetReplaceString  0x74760
//! TopRun                  0x6521c
//! ```
//!
//! Keep the charset/entity layer a separate, swappable stage in front of the
//! rest so an SSML front end can be added later against the same handlers;
//! `XMLstartElement` (`0x72238`) documents the tag set. Do not stub the SSML
//! entry points in a way that silently treats markup as literal text.
