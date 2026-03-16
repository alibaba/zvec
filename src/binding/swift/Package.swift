// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Allow overriding library paths via environment variables for standalone use.
// Defaults to relative paths for in-tree builds.
let env = ProcessInfo.processInfo.environment
let zvecLibDir = env["ZVEC_LIB_DIR"] ?? "../../../build/lib"
let zvecExtLibDir = env["ZVEC_EXT_LIB_DIR"] ?? "../../../build/external/usr/local/lib"
let arrowLibDir = env["ZVEC_ARROW_LIB_DIR"] ?? "../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/release"
let arrowDepsDir = env["ZVEC_ARROW_DEPS_DIR"] ?? "../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/re2_ep-prefix/src/re2_ep-build"
let arrowUtf8Dir = env["ZVEC_ARROW_UTF8_DIR"] ?? "../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/utf8proc_ep-prefix/src/utf8proc_ep-build"

let package = Package(
    name: "ZVec",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "ZVec", targets: ["ZVec"]),
    ],
    targets: [
        .target(
            name: "CZVec",
            path: "Sources/CZVec",
            publicHeadersPath: "include"
        ),
        .target(
            name: "ZVec",
            dependencies: ["CZVec"],
            linkerSettings: [
                .unsafeFlags([
                    "-L\(zvecLibDir)",
                    "-L\(zvecExtLibDir)",
                    "-L\(arrowLibDir)",
                    "-L\(arrowDepsDir)",
                    "-L\(arrowUtf8Dir)",
                    "-Xlinker", "-force_load", "-Xlinker", "\(zvecLibDir)/libzvec_c.a",
                    "-Xlinker", "-force_load", "-Xlinker", "\(zvecLibDir)/libzvec_db.a",
                    "-Xlinker", "-force_load", "-Xlinker", "\(zvecLibDir)/libzvec_core.a",
                    "-Xlinker", "-force_load", "-Xlinker", "\(zvecLibDir)/libzvec_ailego.a",
                    "-lzvec_proto", "-lrocksdb", "-lroaring", "-lglog", "-lprotobuf", "-larrow", "-lparquet", "-larrow_dataset", "-larrow_acero", "-larrow_compute", "-larrow_bundled_dependencies", "-lre2", "-lutf8proc", "-llz4", "-lantlr4-runtime", "-lgflags_nothreads", "-lz", "-lstdc++"
                ])
            ]
        ),
        .executableTarget(
            name: "one_minute",
            dependencies: ["ZVec"],
            path: "Examples"
        ),
        .testTarget(
            name: "ZVecTests",
            dependencies: ["ZVec"],
            path: "Tests/ZVecTests"
        ),
    ]
)
