//! Host environment for the Loquendo ARM engine: guest libc, heap, virtual
//! filesystem, and the Loquendo TTS API bindings.

pub mod cat;
pub mod cfmt;
pub mod embedded;
pub mod heap;
pub mod libc;
pub mod msx;
pub mod tts;
pub mod vfs;

pub use libc::Host;
pub use tts::{Engine, EngineConfig};
