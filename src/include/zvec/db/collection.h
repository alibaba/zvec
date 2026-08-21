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
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <zvec/db/options.h>
#include <zvec/db/query.h>
#include <zvec/db/stats.h>
#include <zvec/db/status.h>
#include <zvec/export.h>

namespace zvec {

class ZVEC_API Collection {
 public:
  using Ptr = std::shared_ptr<Collection>;

  /**
   * @brief Create and open a collection.
   *
   * @param path The path to the collection.
   * @param schema The schema of the collection.
   * @param option The options of the collection.
   * @return The collection OR an error.
   */
  static Result<Ptr> CreateAndOpen(const std::string &path,
                                   const CollectionSchema &schema,
                                   const CollectionOptions &option);

  /**
   * @brief Open an existing collection.
   *
   * @param path The path to the collection.
   * @param option The options of the collection.
   * @return The collection OR an error.
   */
  static Result<Ptr> Open(const std::string &path,
                          const CollectionOptions &option);

  virtual ~Collection();

 public:
  virtual Status close() = 0;

  virtual Status destroy() = 0;

  virtual Status flush() = 0;

  virtual Result<std::string> path() const = 0;

  virtual Result<CollectionStats> stats() const = 0;

  virtual Result<CollectionSchema> schema() const = 0;

  virtual Result<CollectionOptions> options() const = 0;

 public:
  virtual Status create_index(
      const std::string &column_name, const IndexParams::Ptr &index_params,
      const CreateIndexOptions &options = CreateIndexOptions{0}) = 0;

  virtual Status drop_index(const std::string &column_name) = 0;

  virtual Status optimize(const OptimizeOptions &options = OptimizeOptions{
                              0}) = 0;

  virtual Status add_column(const FieldSchema::Ptr &column_schema,
                            const std::string &expression,
                            const AddColumnOptions &options = AddColumnOptions{
                                0}) = 0;

  virtual Status drop_column(const std::string &column_name) = 0;

  virtual Status alter_column(
      const std::string &column_name, const std::string &rename,
      const FieldSchema::Ptr &new_column_schema = nullptr,
      const AlterColumnOptions &options = AlterColumnOptions{0}) = 0;

  virtual Result<WriteResults> insert(std::vector<Doc> &docs) = 0;

  virtual Result<WriteResults> upsert(std::vector<Doc> &docs) = 0;

  virtual Result<WriteResults> update(std::vector<Doc> &docs) = 0;

  virtual Result<WriteResults> delete_(const std::vector<std::string> &pks) = 0;

  virtual Status delete_by_filter(const std::string &filter) = 0;

  virtual Result<DocPtrList> query(const SearchQuery &query) const = 0;

  virtual Result<DocPtrList> query(const MultiQuery &query) const = 0;

  virtual Result<GroupResults> group_by_query(
      const GroupByVectorQuery &query) const = 0;

  virtual Result<DocPtrMap> fetch(const std::vector<std::string> &pks,
                                  const std::optional<std::vector<std::string>>
                                      &output_fields = std::nullopt,
                                  bool include_vector = true) const = 0;

 public:
  //! Debug-only: retrieve the storage mode string of an HNSW index on the
  //! given vector column. Returns one of {"mmap", "buffer_pool",
  //! "contiguous"}. Returns an error Status when the column does not exist,
  //! has no index, or the index is not an HNSW index. Intended for
  //! introspection and testing; not part of the stable public API.
  virtual Result<std::string> debug_get_hnsw_storage_mode(
      const std::string &column_name) const = 0;
};

}  // namespace zvec
