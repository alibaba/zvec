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

#include "python_collection.h"
#include <pybind11/stl.h>
#include <zvec/db/collection.h>

namespace zvec {

inline void throw_if_error(const Status &status) {
  switch (status.code()) {
    case StatusCode::OK:
      return;
    case StatusCode::NOT_FOUND:
      throw py::key_error(status.message());
    case StatusCode::INVALID_ARGUMENT:
      throw py::value_error(status.message());
    case StatusCode::INTERNAL_ERROR:
    case StatusCode::ALREADY_EXISTS:
    case StatusCode::NOT_SUPPORTED:
    case StatusCode::PERMISSION_DENIED:
    case StatusCode::FAILED_PRECONDITION:
    case StatusCode::UNKNOWN:
    default:
      throw std::runtime_error(status.message());
  }
}


template <typename T>
T unwrap_expected(const tl::expected<T, Status> &exp) {
  if (exp.has_value()) {
    return exp.value();
  }
  throw_if_error(exp.error());
  return T{};
}

void ZVecPyCollection::Initialize(pybind11::module_ &m) {
  py::class_<Collection, Collection::Ptr> collection(m, "_Collection");
  bind_db_methods(collection);
  bind_ddl_methods(collection);
  bind_dml_methods(collection);
  bind_dql_methods(collection);
  collection.def(py::pickle(
      [](const Collection &c) {
        return py::make_tuple(c.Path(), c.Schema(), c.Options());
      },
      [](py::tuple t) {
        if (t.size() != 3) {
          throw std::runtime_error("Invalid tuple size for Collection pickle");
        }
        std::string path = t[0].cast<std::string>();
        auto schema = t[1].cast<CollectionSchema>();
        CollectionOptions options = t[2].cast<CollectionOptions>();
        auto result = Collection::Open(path, options);
        // auto result = Collection::CreateAndOpen(path, schema, options);
        return unwrap_expected(result);
      }));
}

void ZVecPyCollection::bind_db_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  col.def_static("CreateAndOpen",
                 [](const std::string &path, const CollectionSchema &schema,
                    const CollectionOptions &options) {
                   auto result =
                       Collection::CreateAndOpen(path, schema, options);
                   return unwrap_expected(result);
                 })
      .def_static("Open", [](const std::string &path,
                             const CollectionOptions &options) {
        auto result = Collection::Open(path, options);
        return unwrap_expected(result);
      });
}


void ZVecPyCollection::bind_ddl_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection properties
  col.def("Path",
          [](const Collection &self) {
            auto ret = self.Path();
            return unwrap_expected(ret);
          })
      .def("Options",
           [](const Collection &self) {
             auto ret = self.Options();
             return unwrap_expected(ret);
           })
      .def("Schema",
           [](const Collection &self) {
             auto ret = self.Schema();
             return unwrap_expected(ret);
           })
      .def("Stats", [](const Collection &self) {
        auto ret = self.Stats();
        return unwrap_expected(ret);
      });

  // bind collection ddl methods
  col.def("Destroy",
          [](Collection &self) {
            const auto status = self.Destroy();
            throw_if_error(status);
          })
      .def("Flush", [](Collection &self) {
        auto status = self.Flush();
        throw_if_error(status);
      });

  // binding index ddl methods
  col.def("CreateIndex",
          [](Collection &self, const std::string &column_name,
             const IndexParams::Ptr &index_options,
             const CreateIndexOptions &options) {
            const auto status =
                self.CreateIndex(column_name, index_options, options);
            throw_if_error(status);
          })
      .def("DropIndex",
           [](Collection &self, const std::string &column_name) {
             const auto status = self.DropIndex(column_name);
             throw_if_error(status);
           })
      .def("Optimize", [](Collection &self, const OptimizeOptions &options) {
        const auto status = self.Optimize(options);
        throw_if_error(status);
      });

  // binding column ddl methods
  col.def("AddColumn",
          [](Collection &self, const FieldSchema::Ptr &column_schema,
             const std::string &expression, const AddColumnOptions &options) {
            const auto status =
                self.AddColumn(column_schema, expression, options);
            throw_if_error(status);
          })
      .def("DropColumn",
           [](Collection &self, std::string &column_name) {
             auto status = self.DropColumn(column_name);
             throw_if_error(status);
           })
      .def("AlterColumn", [](Collection &self, std::string &column_name,
                             const std::string &rename,
                             const FieldSchema::Ptr &new_column_schema,
                             const AlterColumnOptions &options) {
        const auto status =
            self.AlterColumn(column_name, rename, new_column_schema, options);
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dml_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection upsert/insert/update/delete methods
  col.def("Insert",
          [](Collection &self, std::vector<Doc> &docs) {
            const auto result = self.Insert(docs);
            return unwrap_expected(result);
          })
      .def("Update",
           [](Collection &self, std::vector<Doc> &docs) {
             const auto result = self.Update(docs);
             return unwrap_expected(result);
           })
      .def("Upsert",
           [](Collection &self, std::vector<Doc> &docs) {
             const auto result = self.Upsert(docs);
             return unwrap_expected(result);
           })
      .def("Delete",
           [](Collection &self, const std::vector<std::string> &pks) {
             const auto result = self.Delete(pks);
             return unwrap_expected(result);
           })
      .def("DeleteByFilter", [](Collection &self, const std::string &filter) {
        const auto status = self.DeleteByFilter(filter);
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dql_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  col.def("Query",
          [](const Collection &self, const VectorQuery &query) {
            const auto result = self.Query(query);
            // return DocPtrList
            return unwrap_expected(result);
          })
      .def("GroupByQuery",
           [](const Collection &self, const GroupByVectorQuery &query) {
             const auto result = self.GroupByQuery(query);
             // return GroupResults
             return unwrap_expected(result);
           })
      .def("Fetch",
           [](const Collection &self, const std::vector<std::string> &pks) {
             const auto result = self.Fetch(pks);
             // return DocPtrMap
             return unwrap_expected(result);
           });
}

}  // namespace zvec