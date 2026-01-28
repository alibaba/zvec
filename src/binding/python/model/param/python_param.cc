// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "python_param.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <zvec/core/interface/constants.h>
#include <zvec/db/index_params.h>
#include "python_doc.h"

namespace zvec {

static std::string index_type_to_string(const IndexType type) {
  switch (type) {
    case IndexType::INVERT:
      return "INVERT";
    case IndexType::FLAT:
      return "FLAT";
    case IndexType::IVF:
      return "IVF";
    case IndexType::HNSW:
      return "HNSW";
    default:
      return "UNDEFINED";
  }
}

static std::string metric_type_to_string(const MetricType type) {
  switch (type) {
    case MetricType::COSINE:
      return "COSINE";
    case MetricType::IP:
      return "IP";
    case MetricType::L2:
      return "L2";
    default:
      return "UNDEFINED";
  }
}

static std::string quantize_type_to_string(const QuantizeType type) {
  switch (type) {
    case QuantizeType::UNDEFINED:
      return "UNDEFINED";
    case QuantizeType::INT8:
      return "INT8";
    case QuantizeType::INT4:
      return "INT4";
    case QuantizeType::FP16:
      return "FP16";
    default:
      return "UNDEFINED";
  }
}

template <typename T>
T checked_cast(const py::handle &h, const std::string &vector_field,
               const std::string &expected_type) {
  try {
    return py::cast<T>(h);
  } catch (const py::cast_error &e) {
    std::string actual_type = std::string(py::str(py::type::of(h)));
    std::string msg =
        vector_field + ": expected " + expected_type + ", got " + actual_type;
    throw py::type_error(msg);
  }
}

template <typename T>
std::string serialize_vector(const T *data, size_t n) {
  std::string buf;
  buf.resize(n * sizeof(T));
  std::memcpy(buf.data(), data, n * sizeof(T));
  return buf;
}

template <typename ValueType, typename ValueCastFn>
std::pair<std::string, std::string> serialize_sparse_vector(
    const py::dict &sparse_dict, ValueCastFn &&value_caster) {
  const size_t n = sparse_dict.size();
  if (n == 0) return {{}, {}};

  std::string indices_buf;
  indices_buf.resize(n * sizeof(uint32_t));
  auto *indices_ptr = reinterpret_cast<uint32_t *>(indices_buf.data());

  std::string values_buf;
  values_buf.resize(n * sizeof(ValueType));
  auto *values_ptr = reinterpret_cast<ValueType *>(values_buf.data());

  size_t i = 0;
  for (const auto &[py_key, py_val] : sparse_dict) {
    indices_ptr[i] = checked_cast<uint32_t>(py_key, "Sparse indices", "UINT32");
    values_ptr[i] = value_caster(py_val, i);
    ++i;
  }
  return {std::move(indices_buf), std::move(values_buf)};
}

void ZVecPyParams::Initialize(pybind11::module_ &parent) {
  auto m =
      parent.def_submodule("param", "This module contains the params of Zvec");

  // binding index_params [invert/hnsw/flat/ivf]
  bind_index_params(m);

  // bind query_params [hnsw/ivf]
  bind_query_params(m);

  // bind options [collection/index/optimize/column]
  bind_options(m);

  // bind vector query
  bind_vector_query(m);
}

void ZVecPyParams::bind_index_params(pybind11::module_ &m) {
  // binding base index params
  py::class_<IndexParams, std::shared_ptr<IndexParams>> index_params(
      m, "IndexParam", R"pbdoc(
Base class for all index parameter configurations.

This abstract base class defines the common interface for index types.
It should not be instantiated directly; use derived classes instead.

Attributes:
    type (IndexType): The type of the index (e.g., HNSW, FLAT, INVERT).
)pbdoc");
  index_params
      .def_property_readonly(
          "type",
          [](const IndexParams &self) -> IndexType { return self.type(); },
          "IndexType: The type of the index.")
      .def("clone", &IndexParams::clone, py::return_value_policy::copy)
      .def(
          "__eq__",
          [](const IndexParams &self, const py::object &other) {
            if (!py::isinstance<IndexParams>(other)) return false;
            return self == other.cast<const IndexParams &>();
          },
          py::is_operator())
      .def(
          "to_dict",
          [](const IndexParams &self) -> py::dict {
            py::dict dict;
            dict["type"] = index_type_to_string(self.type());
            return dict;
          },
          "Convert to dictionary with all fields")
      .def(py::pickle(
          [](const IndexParams &self) {  // __getstate__
            return py::make_tuple(self.type());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 1)
              throw std::runtime_error("Invalid state for IndexParams");
            return std::shared_ptr<IndexParams>();
          }));

  // binding invert index params
  py::class_<InvertIndexParams, IndexParams, std::shared_ptr<InvertIndexParams>>
      invert_params(m, "InvertIndexParam", R"pbdoc(
Parameters for configuring an invert index.

This class controls whether range query
optimization is enabled for invert index structures.

Attributes:
    type (IndexType): Always `IndexType.INVERTED`.
    enable_range_optimization (bool): Whether range optimization is enabled.
    enable_extended_wildcard (bool): Whether extended wildcard (suffix and infix) search is enabled.

Examples:
    >>> params = InvertIndexParam(enable_range_optimization=True, enable_extended_wildcard=False)
    >>> print(params.enable_range_optimization)
    True
    >>> print(params.enable_extended_wildcard)
    False
    >>> config = params.to_dict()
    >>> print(config)
    {'enable_range_optimization': True, 'enable_extended_wildcard': False}
)pbdoc");
  invert_params
      .def(py::init<bool, bool>(), py::arg("enable_range_optimization") = false,
           py::arg("enable_extended_wildcard") = false,
           R"pbdoc(
Constructs an InvertIndexParam instance.

Args:
    enable_range_optimization (bool, optional): If True, enables range query
        optimization for the invert index. Defaults to False.
    enable_extended_wildcard (bool, optional): If True, enables extended wildcard
        search including suffix and infix patterns. Defaults to False.
)pbdoc")
      .def_property_readonly("enable_range_optimization",
                             &InvertIndexParams::enable_range_optimization,
                             R"pbdoc(
bool: Whether range optimization is enabled for this inverted index.
)pbdoc")
      .def_property_readonly("enable_extended_wildcard",
                             &InvertIndexParams::enable_extended_wildcard,
                             R"pbdoc(
bool: Whether extended wildcard (suffix and infix) search is enabled.
Note: Prefix search is always enabled regardless of this setting.
)pbdoc")
      .def(
          "to_dict",
          [](const InvertIndexParams &self) -> py::dict {
            py::dict dict;
            dict["enable_range_optimization"] =
                self.enable_range_optimization();
            dict["enable_extended_wildcard"] = self.enable_extended_wildcard();
            return dict;
          },
          "Convert to dictionary with all fields")
      .def("__repr__",
           [](const InvertIndexParams &self) -> std::string {
             return "{"
                    "\"enable_range_optimization\":" +
                    std::to_string(self.enable_range_optimization()) +
                    ","
                    "\"enable_extended_wildcard\":" +
                    std::to_string(self.enable_extended_wildcard()) + "}";
           })
      .def(py::pickle(
          [](const InvertIndexParams &self) {  // __getstate__
            return py::make_tuple(self.enable_range_optimization(),
                                  self.enable_extended_wildcard());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 2)
              throw std::runtime_error("Invalid state for InvertIndexParams");
            return std::make_shared<InvertIndexParams>(t[0].cast<bool>(),
                                                       t[1].cast<bool>());
          }));

  // binding base vector index params
  py::class_<VectorIndexParams, IndexParams, std::shared_ptr<VectorIndexParams>>
      vector_params(m, "VectorIndexParam", R"pbdoc(
Base class for vector index parameter configurations.

Encapsulates common settings for all vector index types.

Attributes:
    type (IndexType): The specific vector index type (e.g., HNSW, FLAT).
    metric_type (MetricType): Distance metric used for similarity search.
    quantize_type (QuantizeType): Optional vector quantization type.
)pbdoc");
  vector_params
      .def_property_readonly(
          "metric_type",
          [](const VectorIndexParams &self) -> MetricType {
            return self.metric_type();
          },
          "MetricType: Distance metric (e.g., IP, COSINE, L2).")
      .def_property_readonly(
          "quantize_type",
          [](const VectorIndexParams &self) -> QuantizeType {
            return self.quantize_type();
          },
          "QuantizeType: Vector quantization type (e.g., FP16, INT8).")
      .def(
          "to_dict",
          [](const VectorIndexParams &self) -> py::dict {
            py::dict dict;
            dict["type"] = index_type_to_string(self.type());
            dict["metric_type"] = metric_type_to_string(self.metric_type());
            dict["quantize_type"] =
                quantize_type_to_string(self.quantize_type());
            return dict;
          },
          "Convert to dictionary with all fields")
      .def(py::pickle(
          [](const VectorIndexParams &self) {  // __getstate__
            return py::make_tuple(self.type(), self.metric_type(),
                                  self.quantize_type());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 3)
              throw std::runtime_error("Invalid state for VectorIndexParams");
            // 基类，不能直接实例化，用于子类
            return std::shared_ptr<VectorIndexParams>();
          }));

  // binding hnsw index params
  py::class_<HnswIndexParams, VectorIndexParams,
             std::shared_ptr<HnswIndexParams>>
      hnsw_params(m, "HnswIndexParam", R"pbdoc(
Parameters for configuring an HNSW (Hierarchical Navigable Small World) index.

HNSW is a graph-based approximate nearest neighbor search index. This class
encapsulates its construction hyperparameters.

Attributes:
    metric_type (MetricType): Distance metric used for similarity computation.
        Default is ``MetricType.IP`` (inner product).
    m (int): Number of bi-directional links created for every new element
        during construction. Higher values improve accuracy but increase
        memory usage and construction time. Default is 50.
    ef_construction (int): Size of the dynamic candidate list for nearest
        neighbors during index construction. Larger values yield better
        graph quality at the cost of slower build time. Default is 500.
    quantize_type (QuantizeType): Optional quantization type for vector
        compression (e.g., FP16, INT8). Default is `QuantizeType.UNDEFINED` to
        disable quantization.

Examples:
    >>> from zvec.typing import MetricType, QuantizeType
    >>> params = HnswIndexParam(
    ...     metric_type=MetricType.COSINE,
    ...     m=16,
    ...     ef_construction=200,
    ...     quantize_type=QuantizeType.INT8
    ... )
    >>> print(params)
    {'metric_type': 'IP', 'm': 16, 'ef_construction': 200, 'quantize_type': 'INT8'}
)pbdoc");
  hnsw_params
      .def(py::init<MetricType, int, int, QuantizeType>(),
           py::arg("metric_type") = MetricType::IP,
           py::arg("m") = core_interface::kDefaultHnswNeighborCnt,
           py::arg("ef_construction") =
               core_interface::kDefaultHnswEfConstruction,
           py::arg("quantize_type") = QuantizeType::UNDEFINED)
      .def_property_readonly(
          "m", &HnswIndexParams::m,
          "int: Maximum number of neighbors per node in upper layers.")
      .def_property_readonly(
          "ef_construction", &HnswIndexParams::ef_construction,
          "int: Candidate list size during index construction.")
      .def(
          "to_dict",
          [](const HnswIndexParams &self) -> py::dict {
            py::dict dict;
            dict["type"] = index_type_to_string(self.type());
            dict["metric_type"] = metric_type_to_string(self.metric_type());
            dict["m"] = self.m();
            dict["ef_construction"] = self.ef_construction();
            dict["quantize_type"] =
                quantize_type_to_string(self.quantize_type());
            return dict;
          },
          "Convert to dictionary with all fields")
      .def("__repr__",
           [](const HnswIndexParams &self) -> std::string {
             return "{"
                    "\"metric_type\":" +
                    metric_type_to_string(self.metric_type()) +
                    ", \"m\":" + std::to_string(self.m()) +
                    ", \"ef_construction\":" +
                    std::to_string(self.ef_construction()) +
                    ", \"quantize_type\":" +
                    quantize_type_to_string(self.quantize_type()) + "}";
           })
      .def(py::pickle(
          [](const HnswIndexParams &self) {
            return py::make_tuple(self.metric_type(), self.m(),
                                  self.ef_construction(), self.quantize_type());
          },
          [](py::tuple t) {
            if (t.size() != 4)
              throw std::runtime_error("Invalid state for HnswIndexParams");
            return std::make_shared<HnswIndexParams>(
                t[0].cast<MetricType>(), t[1].cast<int>(), t[2].cast<int>(),
                t[3].cast<QuantizeType>());
          }));

  // FlatIndexParams
  py::class_<FlatIndexParams, VectorIndexParams,
             std::shared_ptr<FlatIndexParams>>
      flat_params(m, "FlatIndexParam", R"pbdoc(
Parameters for configuring a flat (brute-force) index.

A flat index performs exact nearest neighbor search by comparing the query
vector against all vectors in the collection. It is simple, accurate, and
suitable for small to medium datasets or as a baseline.

Attributes:
    metric_type (MetricType): Distance metric used for similarity computation.
        Default is ``MetricType.IP`` (inner product).
    quantize_type (QuantizeType): Optional quantization type for vector
        compression (e.g., FP16, INT8). Use ``QuantizeType.UNDEFINED`` to
        disable quantization. Default is ``QuantizeType.UNDEFINED``.

Examples:
    >>> from zvec.typing import MetricType, QuantizeType
    >>> params = FlatIndexParam(
    ...     metric_type=MetricType.L2,
    ...     quantize_type=QuantizeType.FP16
    ... )
    >>> print(params)
    {'metric_type': 'L2', 'quantize_type': 'FP16'}
)pbdoc");
  flat_params
      .def(py::init<MetricType, QuantizeType>(),
           py::arg("metric_type") = MetricType::IP,
           py::arg("quantize_type") = QuantizeType::UNDEFINED,
           R"pbdoc(
Constructs a FlatIndexParam instance.

Args:
    metric_type (MetricType, optional): Distance metric. Defaults to MetricType.IP.
    quantize_type (QuantizeType, optional): Vector quantization type.
        Defaults to QuantizeType.UNDEFINED (no quantization).
)pbdoc")
      .def(
          "to_dict",
          [](const FlatIndexParams &self) -> py::dict {
            py::dict dict;
            dict["metric_type"] = metric_type_to_string(self.metric_type());
            dict["quantize_type"] =
                quantize_type_to_string(self.quantize_type());
            return dict;
          },
          "Convert to dictionary with all fields")
      .def("__repr__",
           [](const FlatIndexParams &self) -> std::string {
             return "{"
                    "\"metric_type\":" +
                    metric_type_to_string(self.metric_type()) +
                    ", \"quantize_type\":" +
                    quantize_type_to_string(self.quantize_type()) + "}";
           })
      .def(py::pickle(
          [](const FlatIndexParams &self) {
            return py::make_tuple(self.metric_type(), self.quantize_type());
          },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state for FlatIndexParams");
            return std::make_shared<FlatIndexParams>(t[0].cast<MetricType>(),
                                                     t[1].cast<QuantizeType>());
          }));

  // IVFIndexParams
  py::class_<IVFIndexParams, VectorIndexParams, std::shared_ptr<IVFIndexParams>>
      ivf_params(m, "IVFIndexParam", R"pbdoc(
Parameters for configuring an IVF (Inverted File Index) index.

IVF partitions the vector space into clusters (inverted lists). At query time,
only a subset of clusters is searched, providing a trade-off between speed
and accuracy.

Attributes:
    metric_type (MetricType): Distance metric used for similarity computation.
        Default is ``MetricType.IP`` (inner product).
    n_list (int): Number of clusters (inverted lists) to partition the dataset into.
        If set to 0, the system will auto-select a reasonable value based on data size.
        Default is 0 (auto).
    n_iters (int): Number of iterations for k-means clustering during index training.
        Higher values yield more stable centroids. Default is 10.
    use_soar (bool): Whether to enable SOAR (Scalable Optimized Adaptive Routing)
        for improved IVF search performance. Default is False.
    quantize_type (QuantizeType): Optional quantization type for vector
        compression (e.g., FP16, INT8). Default is ``QuantizeType.UNDEFINED``.

Examples:
    >>> from zvec.typing import MetricType, QuantizeType
    >>> params = IVFIndexParam(
    ...     metric_type=MetricType.COSINE,
    ...     n_list=100,
    ...     n_iters=15,
    ...     use_soar=True,
    ...     quantize_type=QuantizeType.INT8
    ... )
    >>> print(params.n_list)
    100
)pbdoc");
  ivf_params
      .def(py::init<MetricType, int, int, bool, QuantizeType>(),
           py::arg("metric_type") = MetricType::IP, py::arg("n_list") = 0,
           py::arg("n_iters") = 10, py::arg("use_soar") = false,
           py::arg("quantize_type") = QuantizeType::UNDEFINED,
           R"pbdoc(
Constructs an IVFIndexParam instance.

Args:
    metric_type (MetricType, optional): Distance metric. Defaults to MetricType.IP.
    n_list (int, optional): Number of inverted lists (clusters). Set to 0 for auto.
        Defaults to 0.
    n_iters (int, optional): Number of k-means iterations during training.
        Defaults to 10.
    use_soar (bool, optional): Enable SOAR optimization. Defaults to False.
    quantize_type (QuantizeType, optional): Vector quantization type.
        Defaults to QuantizeType.UNDEFINED.
)pbdoc")
      .def_property_readonly("n_list", &IVFIndexParams::n_list,
                             "int: Number of inverted lists (0 = auto).")
      .def_property_readonly(
          "n_iters", &IVFIndexParams::n_iters,
          "int: Number of k-means iterations during training.")
      .def_property_readonly("use_soar", &IVFIndexParams::use_soar,
                             "bool: Whether SOAR optimization is enabled.")
      .def(
          "to_dict",
          [](const IVFIndexParams &self) -> py::dict {
            py::dict dict;
            dict["type"] = index_type_to_string(self.type());
            dict["metric_type"] = metric_type_to_string(self.metric_type());
            dict["n_list"] = self.n_list();
            dict["n_iters"] = self.n_iters();
            dict["use_soar"] = self.use_soar();
            dict["quantize_type"] =
                quantize_type_to_string(self.quantize_type());
            return dict;
          },
          "Convert to dictionary with all fields")
      .def("__repr__",
           [](const IVFIndexParams &self) {
             return "{"
                    "\"metric_type\":" +
                    metric_type_to_string(self.metric_type()) +
                    ", \"n_list\":" + std::to_string(self.n_list()) +
                    ", \"n_iters\":" + std::to_string(self.n_iters()) +
                    ", \"use_soar\":" + std::to_string(self.use_soar()) +
                    ", \"quantize_type\":" +
                    quantize_type_to_string(self.quantize_type()) + "}";
           })
      .def(py::pickle(
          [](const IVFIndexParams &self) {
            return py::make_tuple(self.metric_type(), self.n_list(),
                                  self.n_iters(), self.use_soar(),
                                  self.quantize_type());
          },
          [](py::tuple t) {
            if (t.size() != 5)
              throw std::runtime_error("Invalid state for IVFIndexParams");
            return std::make_shared<IVFIndexParams>(
                t[0].cast<MetricType>(), t[1].cast<int>(), t[2].cast<int>(),
                t[3].cast<bool>(), t[4].cast<QuantizeType>());
          }));
}

void ZVecPyParams::bind_query_params(py::module_ &m) {
  // binding base query params
  py::class_<QueryParams, std::shared_ptr<QueryParams>> query_params(
      m, "QueryParam", R"pbdoc(
Base class for all query parameter configurations.

This abstract base class defines common query settings such as search radius
and whether to force linear (brute-force) search. It should not be instantiated
directly; use derived classes like `HnswQueryParam` or `IVFQueryParam`.

Attributes:
    type (IndexType): The index type this query is configured for.
    radius (float): Search radius for range queries. Used in combination with
        top-k to filter results. Default is 0.0 (disabled).
    is_linear (bool): If True, forces brute-force linear search instead of
        using the index. Useful for debugging or small datasets. Default is False.
    is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.
)pbdoc");
  query_params
      .def_property_readonly(
          "type",
          [](const QueryParams &self) -> IndexType { return self.type(); },
          "IndexType: The type of index this query targets.")
      .def_property_readonly(
          "radius",
          [](const QueryParams &self) -> float { return self.radius(); },
          "IndexType: The type of index this query targets.")
      .def_property_readonly(
          "is_linear",
          [](const QueryParams &self) -> bool { return self.is_linear(); },
          "bool: Whether to bypass the index and use brute-force linear "
          "search.")
      .def_property_readonly(
          "is_using_refiner",
          [](const QueryParams &self) -> bool {
            return self.is_using_refiner();
          },
          "bool: Whether to use refiner for the query.")
      .def(py::pickle(
          [](const QueryParams &self) {  // __getstate__
            return py::make_tuple(self.type(), self.radius(), self.is_linear());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 3)
              throw std::runtime_error("Invalid state for QueryParams");
            return std::shared_ptr<QueryParams>();
          }));

  // binding hnsw query params
  py::class_<HnswQueryParams, QueryParams, std::shared_ptr<HnswQueryParams>>
      hnsw_params(m, "HnswQueryParam", R"pbdoc(
Query parameters for HNSW (Hierarchical Navigable Small World) index.

Controls the trade-off between search speed and accuracy via the `ef` parameter.

Attributes:
    type (IndexType): Always ``IndexType.HNSW``.
    ef (int): Size of the dynamic candidate list during search.
        Larger values improve recall but slow down search.
        Default is 300.
    radius (float): Search radius for range queries. Default is 0.0.
    is_linear (bool): Force linear search. Default is False.
    is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.

Examples:
    >>> params = HnswQueryParam(ef=300)
    >>> print(params.ef)
    300
    >>> print(params.to_dict() if hasattr(params, 'to_dict') else params)
    {"type":"HNSW", "ef":300}
)pbdoc");
  hnsw_params
      .def(py::init<int, float, bool, bool>(),
           py::arg("ef") = core_interface::kDefaultHnswEfSearch,
           py::arg("radius") = 0.0f, py::arg("is_linear") = false,
           py::arg("is_using_refiner") = false,
           R"pbdoc(
Constructs an HnswQueryParam instance.

Args:
    ef (int, optional): Search-time candidate list size.
        Higher values improve accuracy. Defaults to 100.
    radius (float, optional): Search radius for range queries. Default is 0.0.
    is_linear (bool, optional): Force linear search. Default is False.
    is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.
)pbdoc")
      .def_property_readonly(
          "ef", [](const HnswQueryParams &self) -> int { return self.ef(); },
          "int: Size of the dynamic candidate list during HNSW search.")
      .def("__repr__",
           [](const HnswQueryParams &self) -> std::string {
             return "{"
                    "\"type\":" +
                    index_type_to_string(self.type()) +
                    ", \"ef\":" + std::to_string(self.ef()) +
                    ", \"radius\":" + std::to_string(self.radius()) +
                    ", \"is_linear\":" + std::to_string(self.is_linear()) +
                    ", \"is_using_refiner\":" +
                    std::to_string(self.is_using_refiner()) + "}";
           })
      .def(py::pickle(
          [](const HnswQueryParams &self) {
            return py::make_tuple(self.ef(), self.radius(), self.is_linear(),
                                  self.is_using_refiner());
          },
          [](py::tuple t) {
            if (t.size() != 4)
              throw std::runtime_error("Invalid state for HnswQueryParams");
            auto obj = std::make_shared<HnswQueryParams>(t[0].cast<int>());
            obj->set_radius(t[1].cast<float>());
            obj->set_is_linear(t[2].cast<bool>());
            obj->set_is_using_refiner(t[3].cast<bool>());
            return obj;
          }));

  // binding ivf query params
  py::class_<IVFQueryParams, QueryParams, std::shared_ptr<IVFQueryParams>>
      ivf_params(m, "IVFQueryParam", R"pbdoc(
Query parameters for IVF (Inverted File Index) index.

Controls how many inverted lists (`nprobe`) to visit during search.

Attributes:
    type (IndexType): Always ``IndexType.IVF``.
    nprobe (int): Number of closest clusters (inverted lists) to search.
        Higher values improve recall but increase latency.
        Default is 10.
    radius (float): Search radius for range queries. Default is 0.0.
    is_linear (bool): Force linear search. Default is False.

Examples:
    >>> params = IVFQueryParam(nprobe=20)
    >>> print(params.nprobe)
    20
)pbdoc");
  ivf_params
      .def(py::init<int>(), py::arg("nprobe") = 10, R"pbdoc(
Constructs an IVFQueryParam instance.

Args:
    nprobe (int, optional): Number of inverted lists to probe during search.
        Higher values improve accuracy. Defaults to 10.
)pbdoc")
      .def_property_readonly(
          "nprobe",
          [](const IVFQueryParams &self) -> int { return self.nprobe(); },
          "int: Number of inverted lists to search during IVF query.")
      .def("__repr__",
           [](const IVFQueryParams &self) -> std::string {
             return "{"
                    "\"type\":" +
                    index_type_to_string(self.type()) +
                    ", \"nprobe\":" + std::to_string(self.nprobe()) + "}";
           })
      .def(py::pickle(
          [](const IVFQueryParams &self) {
            return py::make_tuple(self.nprobe(), self.radius(),
                                  self.is_linear());
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state for IVFQueryParams");
            auto obj = std::make_shared<IVFQueryParams>(t[0].cast<int>());
            obj->set_radius(t[1].cast<float>());
            obj->set_is_linear(t[2].cast<bool>());
            return obj;
          }));
}

void ZVecPyParams::bind_options(py::module_ &m) {
  // binding collection options
  py::class_<CollectionOptions>(m, "CollectionOption", R"pbdoc(
Options for opening or creating a collection.

Attributes:
    read_only (bool): Whether the collection is opened in read-only mode.
        Default is False.
    enable_mmap (bool): Whether to use memory-mapped I/O for data files.
        Default is True.

Examples:
    >>> opt = CollectionOption(read_only=True, enable_mmap=False)
    >>> print(opt.read_only)
    True
)pbdoc")
      .def(py::init<bool, bool>(), py::arg("read_only") = false,
           py::arg("enable_mmap") = true,
           R"pbdoc(
Constructs a CollectionOption instance.

Args:
    read_only (bool, optional): Open collection in read-only mode.
        Defaults to False.
    enable_mmap (bool, optional): Enable memory-mapped I/O.
        Defaults to True.
)pbdoc")
      .def_property_readonly(
          "enable_mmap",
          [](const CollectionOptions &self) { return self.enable_mmap_; })
      .def_property_readonly(
          "read_only",
          [](const CollectionOptions &self) { return self.read_only_; })
      .def("__repr__",
           [](const CollectionOptions &self) -> std::string {
             return "{"
                    "\"enable_mmap\":" +
                    std::to_string(self.enable_mmap_) +
                    ", \"read_only\":" + std::to_string(self.read_only_) + "}";
           })
      .def(py::pickle(
          [](const CollectionOptions &self) {
            return py::make_tuple(self.read_only_, self.enable_mmap_,
                                  self.max_buffer_size_);
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error(
                  "Invalid pickle data for CollectionOptions");
            CollectionOptions obj{};
            obj.read_only_ = t[0].cast<bool>();
            obj.enable_mmap_ = t[1].cast<bool>();
            obj.max_buffer_size_ = t[2].cast<uint32_t>();
            return obj;
          }));

  // SegmentOptions
  py::class_<SegmentOptions>(m, "SegmentOption", R"pbdoc(
Options for segment-level operations.

Currently, this class mirrors CollectionOption and is used internally.
It supports read-only mode, memory mapping, and buffer configuration.

Note:
    This class is primarily for internal use. Most users should use
    CollectionOption instead.

Examples:
    >>> opt = SegmentOption()
    >>> print(opt.enable_mmap)
    True
)pbdoc")
      .def(py::init<>(), "Constructs a SegmentOption with default settings.")
      .def_property_readonly(
          "enable_mmap",
          [](const SegmentOptions &self) { return self.enable_mmap_; },
          "bool: Whether memory-mapped I/O is enabled.")
      .def_property_readonly(
          "read_only",
          [](const SegmentOptions &self) { return self.read_only_; },
          "bool: Whether the segment is read-only.")
      .def_property_readonly(
          "max_buffer_size",
          [](const SegmentOptions &self) { return self.max_buffer_size_; },
          "int: Maximum buffer size in bytes (internal use).")
      .def("__repr__",
           [](const SegmentOptions &self) -> std::string {
             return "{"
                    "\"enable_mmap\":" +
                    std::to_string(self.enable_mmap_) +
                    ", \"read_only\":" + std::to_string(self.read_only_) +
                    ", \"max_buffer_size\":" +
                    std::to_string(self.max_buffer_size_) + "}";
           })
      .def(py::pickle(
          [](const SegmentOptions &self) {
            return py::make_tuple(self.read_only_, self.enable_mmap_,
                                  self.max_buffer_size_);
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error(
                  "Invalid pickle data for SegmentOptions");
            SegmentOptions obj{};
            obj.read_only_ = t[0].cast<bool>();
            obj.enable_mmap_ = t[1].cast<bool>();
            obj.max_buffer_size_ = t[2].cast<uint32_t>();
            return obj;
          }));

  // CreateIndexOptions
  py::class_<CreateIndexOptions>(m, "IndexOption",
                                 R"pbdoc(
Options for creating an index.

Attributes:
    concurrency (int): Number of threads to use during index creation.
        If 0, the system will choose an optimal value automatically.
        Default is 0.

Examples:
    >>> opt = IndexOption(concurrency=4)
    >>> print(opt.concurrency)
    4
)pbdoc")
      .def(py::init<int>(), py::arg("concurrency") = 0,
           R"pbdoc(
Constructs an IndexOption instance.

Args:
    concurrency (int, optional): Number of concurrent threads.
        0 means auto-detect. Defaults to 0.
)pbdoc")
      .def_property_readonly(
          "concurrency",
          [](const CreateIndexOptions &self) { return self.concurrency_; },
          "int: Number of threads used for index creation (0 = auto).")
      .def(py::pickle(
          [](const CreateIndexOptions &self) {
            return py::make_tuple(self.concurrency_);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error(
                  "Invalid pickle data for CreateIndexOptions");
            CreateIndexOptions obj{};
            obj.concurrency_ = t[0].cast<int>();
            return obj;
          }));

  // OptimizeOptions
  py::class_<OptimizeOptions>(m, "OptimizeOption", R"pbdoc(
Options for optimizing a collection (e.g., merging segments).

Attributes:
    concurrency (int): Number of threads to use during optimization.
        If 0, the system will choose an optimal value automatically.
        Default is 0.

Examples:
    >>> opt = OptimizeOption(concurrency=2)
    >>> print(opt.concurrency)
    2
)pbdoc")
      .def(py::init<int>(), py::arg("concurrency") = 0,
           R"pbdoc(
Constructs an OptimizeOption instance.

Args:
    concurrency (int, optional): Number of concurrent threads.
        0 means auto-detect. Defaults to 0.
)pbdoc")
      .def_property_readonly(
          "concurrency",
          [](const OptimizeOptions &self) { return self.concurrency_; },
          "int: Number of threads used for optimization (0 = auto).")
      .def(py::pickle(
          [](const OptimizeOptions &self) {
            return py::make_tuple(self.concurrency_);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error(
                  "Invalid pickle data for OptimizeOptions");
            OptimizeOptions obj{};
            obj.concurrency_ = t[0].cast<int>();
            return obj;
          }));

  // AddColumnOptions
  py::class_<AddColumnOptions>(m, "AddColumnOption",
                               R"pbdoc(
Options for adding a new column to a collection.

Attributes:
    concurrency (int): Number of threads to use when backfilling data
        for the new column. If 0, auto-detect is used. Default is 0.

Examples:
    >>> opt = AddColumnOption(concurrency=1)
    >>> print(opt.concurrency)
    1
)pbdoc")
      .def(py::init<int>(), py::arg("concurrency") = 0,
           R"pbdoc(
Constructs an AddColumnOption instance.

Args:
    concurrency (int, optional): Number of threads for data backfill.
        0 means auto-detect. Defaults to 0.
)pbdoc")
      .def_property_readonly(
          "concurrency",
          [](const AddColumnOptions &self) { return self.concurrency_; },
          "int: Number of threads used when adding a column (0 = auto).")
      .def(py::pickle(
          [](const AddColumnOptions &self) {
            return py::make_tuple(self.concurrency_);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error(
                  "Invalid pickle data for AddColumnOptions");
            AddColumnOptions obj{};
            obj.concurrency_ = t[0].cast<int>();
            return obj;
          }));

  // AlterColumnOptions
  py::class_<AlterColumnOptions>(m, "AlterColumnOption", R"pbdoc(
Options for altering an existing column (e.g., changing index settings).

Attributes:
    concurrency (int): Number of threads to use during the alteration process.
        If 0, the system will choose an optimal value automatically.
        Default is 0.

Examples:
    >>> opt = AlterColumnOption(concurrency=1)
    >>> print(opt.concurrency)
    1
)pbdoc")
      .def(py::init<int>(), py::arg("concurrency") = 0,
           R"pbdoc(
Constructs an AlterColumnOption instance.

Args:
    concurrency (int, optional): Number of threads for column alteration.
        0 means auto-detect. Defaults to 0.
)pbdoc")
      .def_property_readonly(
          "concurrency",
          [](const AlterColumnOptions &self) { return self.concurrency_; },
          "int: Number of threads used when altering a column (0 = auto).")
      .def(py::pickle(
          [](const AlterColumnOptions &self) {
            return py::make_tuple(self.concurrency_);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error(
                  "Invalid pickle data for AlterColumnOptions");
            AlterColumnOptions obj{};
            obj.concurrency_ = t[0].cast<int>();
            return obj;
          }));
}

void ZVecPyParams::bind_vector_query(py::module_ &m) {
  py::class_<VectorQuery>(m, "_VectorQuery")
      .def(py::init<>())
      // properties
      .def_readwrite("topk", &VectorQuery::topk_)
      .def_readwrite("field_name", &VectorQuery::field_name_)
      .def_readwrite("filter", &VectorQuery::filter_)
      .def_readwrite("include_vector", &VectorQuery::include_vector_)
      .def_readwrite("query_params", &VectorQuery::query_params_)
      .def_readwrite("output_fields", &VectorQuery::output_fields_)
      // vector
      .def("set_vector",
           [](VectorQuery &self, const FieldSchema &field_schema,
              const py::object &obj) {
             const DataType data_type = field_schema.data_type();

             // dense vector
             if (FieldSchema::is_dense_vector_field(data_type)) {
               if (!py::isinstance<py::array>(obj)) {
                 throw py::type_error("Dense vector[" + field_schema.name() +
                                      "] expects a ndarray, got " +
                                      std::string(py::str(py::type::of(obj))));
               }
               const auto arr = obj.cast<py::array>();
               if (arr.ndim() != 1) {
                 throw py::type_error("Dense vector expects 1D array, got " +
                                      std::to_string(arr.ndim()) + "D");
               }
               const auto buf = arr.request();
               switch (data_type) {
                 case DataType::VECTOR_FP32: {
                   self.query_vector_ = serialize_vector<float>(
                       static_cast<const float *>(buf.ptr), buf.size);
                   return;
                 }
                 case DataType::VECTOR_FP64: {
                   self.query_vector_ = serialize_vector<double>(
                       static_cast<const double *>(buf.ptr), buf.size);
                   return;
                 }
                 case DataType::VECTOR_INT8: {
                   self.query_vector_ = serialize_vector<int8_t>(
                       static_cast<const int8_t *>(buf.ptr), buf.size);
                   return;
                 }
                 case DataType::VECTOR_FP16: {
                   self.query_vector_ = serialize_vector<uint16_t>(
                       static_cast<const uint16_t *>(buf.ptr), buf.size);
                   return;
                 }
                 default:
                   throw py::type_error(
                       "Unsupported dense vector type for ndarray input: " +
                       std::to_string(static_cast<int>(data_type)));
               }
             }
             // sparse vector
             if (FieldSchema::is_sparse_vector_field(data_type)) {
               if (!py::isinstance<py::dict>(obj)) {
                 throw py::type_error("Sparse vector[" + field_schema.name() +
                                      "] expects a Python dict, got " +
                                      std::string(py::str(py::type::of(obj))));
               }
               const auto sparse = obj.cast<py::dict>();

               switch (data_type) {
                 case DataType::SPARSE_VECTOR_FP16: {
                   auto [indices, values] =
                       serialize_sparse_vector<ailego::Float16>(
                           sparse, [](const py::handle &h, size_t idx) {
                             float f = checked_cast<float>(
                                 h, "Sparse value[" + std::to_string(idx) + "]",
                                 "FLOAT");
                             return ailego::Float16(f);
                           });
                   self.query_sparse_indices_ = std::move(indices);
                   self.query_sparse_values_ = std::move(values);
                   break;
                 }
                 case DataType::SPARSE_VECTOR_FP32: {
                   auto [indices, values] = serialize_sparse_vector<float>(
                       sparse, [](const py::handle &h, size_t idx) {
                         return checked_cast<float>(
                             h, "Sparse value[" + std::to_string(idx) + "]",
                             "FLOAT");
                       });
                   self.query_sparse_indices_ = std::move(indices);
                   self.query_sparse_values_ = std::move(values);
                   break;
                 }
                 default:
                   throw py::type_error(
                       "Unsupported sparse vector type: " +
                       std::to_string(static_cast<int>(data_type)));
               }
               return;
             }

             throw py::type_error("Unsupported vector field type for field: " +
                                  field_schema.name());
           })
      .def(
          "get_vector",
          [](const VectorQuery &self,
             const FieldSchema &field_schema) -> py::object {
            DataType data_type = field_schema.data_type();
            if (FieldSchema::is_dense_vector_field(data_type)) {
              if (self.query_vector_.empty()) {
                throw std::runtime_error("No dense vector has been set");
              }

              size_t byte_size = self.query_vector_.size();
              const void *data = self.query_vector_.data();

              switch (data_type) {
                case DataType::VECTOR_FP32: {
                  if (byte_size % sizeof(float) != 0) {
                    throw std::runtime_error(
                        "Invalid buffer size for VECTOR_FP32");
                  }
                  size_t dim = byte_size / sizeof(float);
                  return py::array_t<float>({dim}, {sizeof(float)},
                                            static_cast<const float *>(data));
                }
                case DataType::VECTOR_FP64: {
                  if (byte_size % sizeof(double) != 0) {
                    throw std::runtime_error(
                        "Invalid buffer size for VECTOR_FP64");
                  }
                  size_t dim = byte_size / sizeof(double);
                  return py::array_t<double>({dim}, {sizeof(double)},
                                             static_cast<const double *>(data));
                }
                case DataType::VECTOR_INT8: {
                  if (byte_size % sizeof(int8_t) != 0) {
                    throw std::runtime_error(
                        "Invalid buffer size for VECTOR_INT8");
                  }
                  size_t dim = byte_size / sizeof(int8_t);
                  return py::array_t<int8_t>({dim}, {sizeof(int8_t)},
                                             static_cast<const int8_t *>(data));
                }
                case DataType::VECTOR_FP16: {
                  if (byte_size % 2 != 0) {
                    throw std::runtime_error(
                        "Invalid buffer size for VECTOR_FP16");
                  }
                  size_t dim = byte_size / 2;
                  return py::array(py::dtype("float16"), {dim}, {2}, data);
                }

                default:
                  throw py::type_error(
                      "Unsupported dense vector type for get_vector: " +
                      std::to_string(static_cast<int>(data_type)));
              }
            }
            if (FieldSchema::is_sparse_vector_field(data_type)) {
              if (self.query_sparse_indices_.empty()) {
                return py::dict();
              }

              // Deserialize indices: stored as uint32_t[]
              size_t indices_byte_size = self.query_sparse_indices_.size();
              if (indices_byte_size % sizeof(uint32_t) != 0) {
                throw std::runtime_error(
                    "Sparse indices buffer size not aligned to uint32_t");
              }
              size_t n = indices_byte_size / sizeof(uint32_t);
              const uint32_t *indices = reinterpret_cast<const uint32_t *>(
                  self.query_sparse_indices_.data());

              // Deserialize values
              switch (data_type) {
                case DataType::SPARSE_VECTOR_FP32: {
                  if (self.query_sparse_values_.size() != n * sizeof(float)) {
                    throw std::runtime_error(
                        "Sparse FP32 values buffer size mismatch");
                  }
                  const float *values = reinterpret_cast<const float *>(
                      self.query_sparse_values_.data());
                  py::dict result;
                  for (size_t i = 0; i < n; ++i) {
                    result[py::int_(indices[i])] = py::float_(values[i]);
                  }
                  return result;
                }
                case DataType::SPARSE_VECTOR_FP16: {
                  if (self.query_sparse_values_.size() !=
                      n * sizeof(uint16_t)) {
                    throw std::runtime_error(
                        "Sparse FP16 values buffer size mismatch");
                  }
                  const uint16_t *raw_bits = reinterpret_cast<const uint16_t *>(
                      self.query_sparse_values_.data());
                  py::dict result;
                  for (size_t i = 0; i < n; ++i) {
                    float f = ailego::FloatHelper::ToFP32(raw_bits[i]);
                    result[py::int_(indices[i])] = py::float_(f);
                  }
                  return result;
                }
                default:
                  throw py::type_error("Unsupported sparse vector type...");
              }
            }


            throw py::type_error("Unsupported vector field type: " +
                                 field_schema.name());
          },
          py::arg("field_schema"))
      .def(py::pickle(
          [](const VectorQuery &self) {
            return py::make_tuple(
                self.topk_, self.field_name_, self.query_vector_,
                self.query_sparse_indices_, self.query_sparse_values_,
                self.filter_, self.include_vector_, self.output_fields_,
                self.query_params_ ? py::cast(self.query_params_) : py::none());
          },
          [](py::tuple t) {
            if (t.size() != 9)
              throw std::runtime_error("Invalid pickle data for VectorQuery");

            VectorQuery obj{};
            obj.topk_ = t[0].cast<int>();
            obj.field_name_ = t[1].cast<std::string>();
            obj.query_vector_ = t[2].cast<std::string>();
            obj.query_sparse_indices_ = t[3].cast<std::string>();
            obj.query_sparse_values_ = t[4].cast<std::string>();
            obj.filter_ = t[5].cast<std::string>();
            obj.include_vector_ = t[6].cast<bool>();
            obj.output_fields_ = t[7].cast<std::vector<std::string>>();

            if (!t[8].is_none()) {
              obj.query_params_ = t[8].cast<QueryParams::Ptr>();
            }
            return obj;
          }));
}
}  // namespace zvec