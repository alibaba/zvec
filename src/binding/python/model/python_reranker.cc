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

#include "python_reranker.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <zvec/db/collection.h>
#include <zvec/db/reranker.h>

namespace zvec {

void ZVecPyReranker::Initialize(py::module_ &m) {
  // Bind RrfParams
  py::class_<reranker::RrfParams>(m, "_RrfParams")
      .def(py::init<int>(), py::arg("rank_constant") = 60)
      .def_readwrite("rank_constant", &reranker::RrfParams::rank_constant);

  // Bind WeightedParams
  py::class_<reranker::WeightedParams>(m, "_WeightedParams")
      .def(py::init<std::vector<double>>(), py::arg("weights"))
      .def_readwrite("weights", &reranker::WeightedParams::weights);

  // Bind CallbackParams
  py::class_<reranker::CallbackParams>(m, "_CallbackParams")
      .def(py::init<reranker::CallbackParams::Callback>(), py::arg("callback"));

  // Standalone rerank execution function
  m.def(
      "_reranker_rerank",
      [](py::object params, const std::vector<DocPtrList> &results,
         const std::vector<FieldSchema::Ptr> &fields, int topn) -> DocPtrList {
        reranker::RerankParams strategy;
        if (py::isinstance<reranker::RrfParams>(params)) {
          strategy = params.cast<reranker::RrfParams>();
        } else if (py::isinstance<reranker::WeightedParams>(params)) {
          strategy = params.cast<reranker::WeightedParams>();
        } else if (py::isinstance<reranker::CallbackParams>(params)) {
          strategy = params.cast<reranker::CallbackParams>();
        } else {
          throw py::type_error(
              "params must be _RrfParams, _WeightedParams, or _CallbackParams");
        }
        auto result = reranker::rerank(strategy, results, fields, topn);
        if (!result.has_value()) {
          throw std::runtime_error(result.error().message());
        }
        return std::move(result).value();
      },
      py::arg("params"), py::arg("results"), py::arg("fields"),
      py::arg("topn"));

  // Bind MultiQuery struct
  py::class_<MultiQuery>(m, "_MultiQuery")
      .def(py::init<>())
      .def_readwrite("queries", &MultiQuery::queries)
      .def_readwrite("topk", &MultiQuery::topk)
      .def_readwrite("filter", &MultiQuery::filter)
      .def_readwrite("include_vector", &MultiQuery::include_vector)
      .def_readwrite("output_fields", &MultiQuery::output_fields)
      .def(
          "set_rerank_rrf",
          [](MultiQuery &q, int rank_constant) {
            q.rerank = reranker::RrfParams{rank_constant};
          },
          py::arg("rank_constant") = 60)
      .def(
          "set_rerank_weighted",
          [](MultiQuery &q, std::vector<double> weights) {
            q.rerank = reranker::WeightedParams{std::move(weights)};
          },
          py::arg("weights"))
      .def(
          "set_rerank_callback",
          [](MultiQuery &q, reranker::CallbackParams::Callback callback) {
            q.rerank = reranker::CallbackParams{std::move(callback)};
          },
          py::arg("callback"));
}

}  // namespace zvec
