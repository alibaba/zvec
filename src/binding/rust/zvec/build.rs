// Copyright 2025-present the zvec project
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = env::var("ZVEC_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/../../../../build/lib", manifest_dir));
    
    let lib_dir_path = PathBuf::from(&lib_dir);
    let lib_dir_abs = lib_dir_path.canonicalize().unwrap_or(lib_dir_path);
    let lib_dir = lib_dir_abs.to_str().unwrap();

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}/libzvec_c.a", lib_dir);
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}/libzvec_db.a", lib_dir);
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}/libzvec_core.a", lib_dir);
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}/libzvec_ailego.a", lib_dir);
    }

    println!("cargo:rerun-if-changed=build.rs");
}
