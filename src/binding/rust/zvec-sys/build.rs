// Copyright 2025-present the zvec project
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = env::var("ZVEC_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/../../../../build/lib", manifest_dir));
    let ext_lib_dir =
        env::var("ZVEC_EXT_LIB_DIR").unwrap_or_else(|_| format!("{}/../../../../build/external/usr/local/lib", manifest_dir));
    let arrow_ext_lib_dir = 
        format!("{}/../../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/release", manifest_dir);
    let arrow_deps_lib_dir = 
        format!("{}/../../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/re2_ep-prefix/src/re2_ep-build", manifest_dir);
    let arrow_utf8_lib_dir = 
        format!("{}/../../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/utf8proc_ep-prefix/src/utf8proc_ep-build", manifest_dir);

    let lib_dir_path = PathBuf::from(&lib_dir);
    let lib_dir_abs = lib_dir_path.canonicalize().unwrap_or(lib_dir_path);
    let lib_dir = lib_dir_abs.to_str().unwrap();

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-search=native={}", ext_lib_dir);
    println!("cargo:rustc-link-search=native={}", arrow_ext_lib_dir);
    println!("cargo:rustc-link-search=native={}", arrow_deps_lib_dir);
    println!("cargo:rustc-link-search=native={}", arrow_utf8_lib_dir);

    println!("cargo:rustc-link-lib=static=zvec_c");
    println!("cargo:rustc-link-lib=static=zvec_db");
    println!("cargo:rustc-link-lib=static=zvec_core");
    println!("cargo:rustc-link-lib=static=zvec_ailego");
    println!("cargo:rustc-link-lib=static=zvec_proto");

    // Static dependencies
    println!("cargo:rustc-link-lib=static=rocksdb");
    println!("cargo:rustc-link-lib=static=roaring");
    println!("cargo:rustc-link-lib=static=glog");
    println!("cargo:rustc-link-lib=static=protobuf");
    println!("cargo:rustc-link-lib=static=arrow");
    println!("cargo:rustc-link-lib=static=parquet");
    println!("cargo:rustc-link-lib=static=arrow_dataset");
    println!("cargo:rustc-link-lib=static=arrow_acero");
    println!("cargo:rustc-link-lib=static=arrow_compute");
    println!("cargo:rustc-link-lib=static=arrow_bundled_dependencies");
    println!("cargo:rustc-link-lib=static=re2");
    println!("cargo:rustc-link-lib=static=utf8proc");
    println!("cargo:rustc-link-lib=static=lz4");
    println!("cargo:rustc-link-lib=static=antlr4-runtime");
    println!("cargo:rustc-link-lib=static=gflags_nothreads");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=static=z");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=z");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
