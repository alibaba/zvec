// swift-tools-version: 5.9
import PackageDescription

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
                    "-L../../../build/lib",
                    "-L../../../build/external/usr/local/lib",
                    "-L../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/release",
                    "-L../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/re2_ep-prefix/src/re2_ep-build",
                    "-L../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/utf8proc_ep-prefix/src/utf8proc_ep-build",
                    "-Xlinker", "-force_load", "-Xlinker", "../../../build/lib/libzvec_c.a",
                    "-Xlinker", "-force_load", "-Xlinker", "../../../build/lib/libzvec_db.a",
                    "-Xlinker", "-force_load", "-Xlinker", "../../../build/lib/libzvec_core.a",
                    "-Xlinker", "-force_load", "-Xlinker", "../../../build/lib/libzvec_ailego.a",
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
