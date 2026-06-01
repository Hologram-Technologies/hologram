#[cfg(feature = "archive")]
#[test]
fn archive_feature_exports_archive_items() {
    assert_eq!(hologram::archive::MAGIC, *b"HOLO");
}

#[cfg(feature = "backend")]
#[test]
fn backend_feature_exports_backend_items() {
    assert_eq!(hologram::backend::MAX_RANK, 8);
}

#[cfg(feature = "bench")]
#[test]
fn bench_feature_exports_bench_module() {
    #[allow(unused_imports)]
    use hologram::bench as _;
}

#[cfg(feature = "cli")]
#[test]
fn cli_feature_exports_cli_module() {
    let _ = core::any::type_name::<hologram::cli::cmd::Cli>();
}

#[cfg(feature = "compiler")]
#[test]
fn compiler_feature_exports_compiler_items() {
    let _ = core::any::type_name::<hologram::compiler::Compiler>();
}

#[cfg(feature = "exec")]
#[test]
fn exec_feature_exports_exec_items() {
    let _ = core::any::type_name::<hologram::exec::BufferArena>();
}

#[cfg(feature = "ffi")]
#[test]
fn ffi_feature_exports_ffi_items() {
    use std::os::raw::{c_int, c_uchar};

    let _compile_empty: unsafe extern "C" fn(*mut c_uchar, usize) -> c_int =
        hologram::ffi::hologram_compile_empty;
}

#[cfg(feature = "graph")]
#[test]
fn graph_feature_exports_graph_items() {
    let _graph = hologram::graph::Graph::new();
}

#[cfg(feature = "host")]
#[test]
fn host_feature_exports_host_items() {
    let _ = core::any::type_name::<hologram::host::HologramHostTypes>();
}

#[cfg(feature = "ops")]
#[test]
fn ops_feature_exports_ops_items() {
    assert!(hologram::ops::HOLOGRAM_INLINE_BYTES > 0);
}

#[cfg(feature = "types")]
#[test]
fn types_feature_exports_types_items() {
    assert!(hologram::types::MemoryTier::CpuL1.is_cpu());
}
