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
#include <zvec/db/type.h>

namespace zvec {

void ZVecPyReranker::Initialize(py::module_ &m) {
  // Bind Reranker base class (abstract, cannot be instantiated directly)
  py::class_<Reranker, Reranker::Ptr>(m, "_Reranker");

  // Bind ScoreBasedReranker intermediate class
  py::class_<ScoreBasedReranker, Reranker, std::shared_ptr<ScoreBasedReranker>>(
      m, "_ScoreBasedReranker");

  // Bind RrfReranker
  py::class_<RrfReranker, ScoreBasedReranker, std::shared_ptr<RrfReranker>>(
      m, "_RrfReranker")
      .def(py::init<int>(), py::arg("rank_constant") = 60)
      .def_property_readonly("rank_constant", &RrfReranker::rank_constant);

  // Bind WeightedReranker
  py::class_<WeightedReranker, ScoreBasedReranker,
             std::shared_ptr<WeightedReranker>>(m, "_WeightedReranker")
      .def(py::init<std::map<std::string, double>>(), py::arg("weights"))
      .def_property_readonly("weights", &WeightedReranker::weights);

  // Bind CallbackReranker
  py::class_<CallbackReranker, Reranker, std::shared_ptr<CallbackReranker>>(
      m, "_CallbackReranker")
      .def(py::init<CallbackReranker::Callback>(), py::arg("callback"));

  // Bind MultiQuery struct
  py::class_<MultiQuery>(m, "_MultiQuery")
      .def(py::init<>())
      .def_readwrite("queries", &MultiQuery::queries)
      .def_readwrite("topk", &MultiQuery::topk)
      .def_readwrite("filter", &MultiQuery::filter)
      .def_readwrite("include_vector", &MultiQuery::include_vector)
      .def_readwrite("output_fields", &MultiQuery::output_fields)
      .def_readwrite("reranker", &MultiQuery::reranker);
}

}  // namespace zvec
