include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_binary(
    NAME txt2vecs
    STRICT PACKED
    SRCS txt2vecs.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags core_framework zvec_ailego
  )

cc_binary(
    NAME hnsw_rabitq_builder
    STRICT PACKED
    SRCS hnsw_rabitq_builder.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags magic_enum core_framework core_metric core_interface core_knn_hnsw core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_ivf core_knn_vamana core_quantizer core_utility
)

cc_binary(
    NAME ivf_rabitq_builder
    STRICT PACKED
    SRCS ivf_rabitq_builder.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags magic_enum core_framework core_metric core_interface core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_ivf core_knn_vamana core_quantizer core_utility
)

cc_binary(
    NAME ivf_rabitq_dedicated_builder
    STRICT PACKED
    SRCS ivf_rabitq_dedicated_builder.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags magic_enum core_framework core_metric core_interface core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_ivf core_knn_vamana core_quantizer core_utility
)

cc_binary(
    NAME local_builder
    STRICT PACKED
    SRCS local_builder.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf core_interface core_knn_diskann
  )

cc_binary(
    NAME recall
    STRICT PACKED
    SRCS recall.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
  )

cc_binary(
    NAME bench
    STRICT PACKED
    SRCS bench.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
)


cc_binary(
    NAME recall_original
    STRICT PACKED
    SRCS recall_original.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
)

cc_binary(
    NAME bench_original
    STRICT PACKED
    SRCS bench_original.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
)

cc_binary(
        NAME local_builder_original
        STRICT PACKED
        SRCS local_builder_original.cc
        INCS ${PROJECT_ROOT_DIR}/src/core/
        LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf core_interface core_knn_diskann
)

# Reads index segments directly to compare list sizes and centroid norms.
cc_binary(
    NAME ivf_rabitq_partition_stat
    STRICT PACKED
    SRCS ivf_rabitq_partition_stat.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags core_framework core_utility
  )

# Offline GT candidate coverage: replicates FP32 centroid selection and maps
# GT ids to inverted lists straight from the index segments.
cc_binary(
    NAME ivf_rabitq_gt_coverage
    STRICT PACKED
    SRCS ivf_rabitq_gt_coverage.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags core_framework core_utility
  )

# Same YAML inputs and public query API as bench/recall; single-query A/B diagnostics.
cc_binary(
    NAME ivf_rabitq_scan_diagnostic
    STRICT PACKED
    SRCS ivf_rabitq_scan_diagnostic.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
)

# Micro-benchmark for the RaBitQ batch estimate hot path: turbo block
# scanner vs rabitqlib split_batch_estdist vs the accumulate_hacc kernel.
cc_binary(
    NAME ivf_rabitq_kernel_bench
    STRICT PACKED
    SRCS ivf_rabitq_kernel_bench.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/ ${PROJECT_ROOT_DIR}/src/turbo/
    LIBS gflags zvec_ailego core_framework core_metric core_quantizer core_knn_cluster zvec_turbo rabitqlib
  )

# Replays recorded real per-query list/block access sequences for both the
# integrated turbo scanner and the dedicated rabitqlib estimator, timing
# original vs sequential order and kernel-only vs with-prepare variants.
cc_binary(
    NAME ivf_rabitq_scan_replay
    STRICT PACKED
    SRCS ivf_rabitq_scan_replay.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/ ${PROJECT_ROOT_DIR}/src/turbo/
    LIBS gflags zvec_ailego core_framework core_metric core_quantizer core_knn_cluster core_utility zvec_turbo rabitqlib
)
