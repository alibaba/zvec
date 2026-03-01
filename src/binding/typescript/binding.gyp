{
  "targets": [
    {
      "target_name": "zvec_addon",
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "sources": ["src/addon.cc"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../c/include"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "conditions": [
        ["OS=='mac'", {
          "libraries": [
            "-Wl,-force_load,<(module_root_dir)/../../../build/lib/libzvec_c.a",
            "-Wl,-force_load,<(module_root_dir)/../../../build/lib/libzvec_db.a",
            "-Wl,-force_load,<(module_root_dir)/../../../build/lib/libzvec_core.a",
            "-Wl,-force_load,<(module_root_dir)/../../../build/lib/libzvec_ailego.a",
            "-L<(module_root_dir)/../../../build/lib",
            "-L<(module_root_dir)/../../../build/external/usr/local/lib",
            "-L<(module_root_dir)/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/release",
            "-L<(module_root_dir)/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/re2_ep-prefix/src/re2_ep-build",
            "-L<(module_root_dir)/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/utf8proc_ep-prefix/src/utf8proc_ep-build",
            "-lzvec_proto", "-lrocksdb", "-lroaring", "-lglog", "-lprotobuf",
            "-larrow", "-lparquet", "-larrow_dataset", "-larrow_acero",
            "-larrow_compute", "-larrow_bundled_dependencies",
            "-lre2", "-lutf8proc", "-llz4", "-lantlr4-runtime", "-lgflags_nothreads",
            "-lz", "-lc++",
            "-framework CoreFoundation", "-framework Security"
          ]
        }],
        ["OS=='linux'", {
          "libraries": [
            "-Wl,--whole-archive",
            "-L<(module_root_dir)/../../../build/lib",
            "-lzvec_c", "-lzvec_db", "-lzvec_core", "-lzvec_ailego",
            "-Wl,--no-whole-archive",
            "-L<(module_root_dir)/../../../build/external/usr/local/lib",
            "-L<(module_root_dir)/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/release",
            "-L<(module_root_dir)/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/re2_ep-prefix/src/re2_ep-build",
            "-L<(module_root_dir)/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/utf8proc_ep-prefix/src/utf8proc_ep-build",
            "-lzvec_proto", "-lrocksdb", "-lroaring", "-lglog", "-lprotobuf",
            "-larrow", "-lparquet", "-larrow_dataset", "-larrow_acero",
            "-larrow_compute", "-larrow_bundled_dependencies",
            "-lre2", "-lutf8proc", "-llz4", "-lantlr4-runtime", "-lgflags_nothreads",
            "-lz", "-lstdc++"
          ]
        }]
      ]
    }
  ]
}
