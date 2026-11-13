//! Prints the native kernel conformance digests in the form the wasm32 driver
//! (`conformance-wasm/run.mjs`) prints them, so `scripts/wasm-conformance.sh`
//! can compare the two substrates line for line.

fn main() {
    use hologram_compute::cpu::conformance::{dot_sum_digest, matmul_digest};
    println!("dot_sum {:016x}", dot_sum_digest());
    println!("matmul {:016x}", matmul_digest());
}
