//! `xtask modbases` — where `loqrs` puts each engine module, in load order.
//!
//! The whole-engine translation reproduces the oracle's memory layout, so it
//! needs these bases, the order the modules arrive in, and the host-call
//! names in the order they were bound (each one's index fixes its trap
//! address, and those addresses end up in GOTs).

use loqng_oracle::{Oracle, OracleConfig};

use crate::Paths;

pub fn run(paths: &Paths, _args: &[String]) -> Result<(), String> {
    let cfg = OracleConfig::new(&paths.lib_dir, &paths.data_dir).interpreted();
    let mut o = Oracle::open(&cfg)?;
    let e = o.engine_mut();
    println!("modules, in load order:");
    for m in &e.machine.modules {
        println!("  0x{:08x}..0x{:08x}  {}", m.base, m.end, m.name);
    }
    Ok(())
}
