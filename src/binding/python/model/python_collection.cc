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
#include <zvec/db/doc_iterator.h>

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
  py::class_<GroupResult>(m, "_GroupResult")
      .def_readonly("group_by_value", &GroupResult::group_by_value_)
      .def_readonly("docs", &GroupResult::docs_);

  bind_iterator(m);

  py::class_<Collection, Collection::Ptr> collection(m, "_Collection");
  bind_db_methods(collection);
  bind_ddl_methods(collection);
  bind_dml_methods(collection);
  bind_dql_methods(collection);
  collection.def(py::pickle(
      [](const Collection &c) {
        return py::make_tuple(c.path(), c.schema(), c.options());
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
                   Result<Collection::Ptr> result;
                   {
                     py::gil_scoped_release release;
                     result = Collection::CreateAndOpen(path, schema, options);
                   }
                   return unwrap_expected(result);
                 })
      .def_static("Open", [](const std::string &path,
                             const CollectionOptions &options) {
        Result<Collection::Ptr> result;
        {
          py::gil_scoped_release release;
          result = Collection::Open(path, options);
        }
        return unwrap_expected(result);
      });
}


void ZVecPyCollection::bind_ddl_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection properties
  col.def("Path",
          [](const Collection &self) {
            auto ret = self.path();
            return unwrap_expected(ret);
          })
      .def("Options",
           [](const Collection &self) {
             auto ret = self.options();
             return unwrap_expected(ret);
           })
      .def("Schema",
           [](const Collection &self) {
             auto ret = self.schema();
             return unwrap_expected(ret);
           })
      .def("Stats", [](const Collection &self) {
        auto ret = self.stats();
        return unwrap_expected(ret);
      });

  // bind collection ddl methods
  col.def("Close",
          [](Collection &self) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.close();
            }
            throw_if_error(status);
          })
      .def("Destroy",
           [](Collection &self) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.destroy();
             }
             throw_if_error(status);
           })
      .def("Flush", [](Collection &self) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.flush();
        }
        throw_if_error(status);
      });

  // binding index ddl methods
  col.def("CreateIndex",
          [](Collection &self, const std::string &column_name,
             const IndexParams::Ptr &index_options,
             const CreateIndexOptions &options) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.create_index(column_name, index_options, options);
            }
            throw_if_error(status);
          })
      .def("DropIndex",
           [](Collection &self, const std::string &column_name) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.drop_index(column_name);
             }
             throw_if_error(status);
           })
      .def("Optimize", [](Collection &self, const OptimizeOptions &options) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.optimize(options);
        }
        throw_if_error(status);
      });

  // binding column ddl methods
  col.def("AddColumn",
          [](Collection &self, const FieldSchema::Ptr &column_schema,
             const std::string &expression, const AddColumnOptions &options) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.add_column(column_schema, expression, options);
            }
            throw_if_error(status);
          })
      .def("DropColumn",
           [](Collection &self, std::string &column_name) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.drop_column(column_name);
             }
             throw_if_error(status);
           })
      .def("AlterColumn", [](Collection &self, std::string &column_name,
                             const std::string &rename,
                             const FieldSchema::Ptr &new_column_schema,
                             const AlterColumnOptions &options) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.alter_column(column_name, rename, new_column_schema,
                                     options);
        }
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dml_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection upsert/insert/update/delete methods
  col.def("Insert",
          [](Collection &self, std::vector<Doc> &docs) {
            Result<WriteResults> result;
            {
              py::gil_scoped_release release;
              result = self.insert(docs);
            }
            return unwrap_expected(result);
          })
      .def("Update",
           [](Collection &self, std::vector<Doc> &docs) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.update(docs);
             }
             return unwrap_expected(result);
           })
      .def("Upsert",
           [](Collection &self, std::vector<Doc> &docs) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.upsert(docs);
             }
             return unwrap_expected(result);
           })
      .def("Delete",
           [](Collection &self, const std::vector<std::string> &pks) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.delete_(pks);
             }
             return unwrap_expected(result);
           })
      .def("DeleteByFilter", [](Collection &self, const std::string &filter) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.delete_by_filter(filter);
        }
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dql_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  col.def("Query",
          [](const Collection &self, const SearchQuery &query) {
            Result<DocPtrList> result;
            {
              py::gil_scoped_release release;
              result = self.query(query);
            }
            // return DocPtrList
            return unwrap_expected(result);
          })
      // MultiQuery: multi query with reranker
      .def(
          "Query",
          [](const Collection &self, const MultiQuery &query) {
            Result<DocPtrList> result;
            {
              py::gil_scoped_release release;
              result = self.query(query);
            }
            // return DocPtrList
            return unwrap_expected(result);
          },
          py::arg("query"), "Execute a multi query with re-ranking.")
      .def("GroupByQuery",
           [](const Collection &self, const GroupByVectorQuery &query) {
             Result<GroupResults> result;
             {
               py::gil_scoped_release release;
               result = self.group_by_query(query);
             }
             return unwrap_expected(result);
           })
      .def(
          "Fetch",
          [](const Collection &self, const std::vector<std::string> &pks,
             const std::optional<std::vector<std::string>> &output_fields,
             bool include_vector) {
            Result<DocPtrMap> result;
            {
              py::gil_scoped_release release;
              result = self.fetch(pks, output_fields, include_vector);
            }
            // return DocPtrMap
            return unwrap_expected(result);
          },
          py::arg("pks"), py::arg("output_fields") = py::none(),
          py::arg("include_vector") = true)
      .def(
          "CreateIterator",
          [](Collection &self,
             const std::optional<std::vector<std::string>> &output_fields,
             bool include_vector) {
            IteratorOptions options;
            options.output_fields_ = output_fields;
            options.include_vector_ = include_vector;
            Result<DocIterator::Ptr> result;
            {
              py::gil_scoped_release release;
              result = self.create_iterator(options);
            }
            return unwrap_expected(result);
          },
          py::arg("output_fields") = py::none(),
          py::arg("include_vector") = true,
          // The collection must outlive the returned iterator (documented
          // contract); keep it
          // alive automatically in the binding too.
          py::keep_alive<0, 1>(),
          "Create a document iterator to traverse all documents.")
      .def(
          "_debug_hnsw_storage_mode",
          [](const Collection &self, const std::string &column_name) {
            const auto result = self.debug_get_hnsw_storage_mode(column_name);
            return unwrap_expected(result);
          },
          py::arg("column_name"),
          "Debug-only: returns the storage mode of the HNSW entity on the "
          "given vector column. One of 'mmap', 'buffer_pool', 'contiguous'. "
          "Raises KeyError if no HNSW index exists on the column, or "
          "ValueError if the column's index is not an HNSW index. Intended "
          "for introspection and testing only; not part of the stable API.");
}

void ZVecPyCollection::bind_iterator(py::module_ &m) {
  // Document iterator: Python iterator protocol (__iter__ / __next__).
  // Constructed only via Collection.create_iterator (no py::init).
  py::class_<DocIterator, DocIterator::Ptr>(m, "_DocIterator")
      .def("__iter__", [](py::object self) { return self; })
      .def("__next__",
           [](DocIterator &self) {
             Result<Doc::Ptr> result;
             {
               py::gil_scoped_release release;
               result = self.next();
             }
             // !has_value() -> error (raises); value()==nullptr -> EOF
             auto doc = unwrap_expected(result);
             if (doc == nullptr) {
               throw py::stop_iteration();
             }
             return doc;
           })
      .def("close", [](DocIterator &self) {
        py::gil_scoped_release release;
        self.close();
      });
}

}  // namespace zvec
