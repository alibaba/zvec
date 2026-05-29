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
    NAME local_builder
    STRICT PACKED
    SRCS local_builder.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_vamana core_knn_cluster core_knn_ivf core_interface
  )

cc_binary(
    NAME recall
    STRICT PACKED
    SRCS recall.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface
  )

cc_binary(
    NAME bench
    STRICT PACKED
    SRCS bench.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface
)


cc_binary(
    NAME recall_original
    STRICT PACKED
    SRCS recall_original.cc flow.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface
)

cc_binary(
    NAME bench_original
    STRICT PACKED
    SRCS bench_original.cc flow.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface
)

cc_binary(
        NAME local_builder_original
        STRICT PACKED
        SRCS local_builder_original.cc
        INCS ${PROJECT_ROOT_DIR}/src/core/
        LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_vamana core_knn_cluster core_knn_ivf core_interface
)
