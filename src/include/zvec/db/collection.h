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
#include <string>
#include <vector>
#include <zvec/db/doc.h>
#include <zvec/db/options.h>
#include <zvec/db/stats.h>
#include <zvec/db/status.h>

namespace zvec {

class Collection {
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
  virtual Status Destroy() = 0;

  virtual Status Flush() = 0;

  virtual Result<std::string> Path() const = 0;

  virtual Result<CollectionStats> Stats() const = 0;

  virtual Result<CollectionSchema> Schema() const = 0;

  virtual Result<CollectionOptions> Options() const = 0;

 public:
  virtual Status CreateIndex(
      const std::string &column_name, const IndexParams::Ptr &index_params,
      const CreateIndexOptions &options = CreateIndexOptions{0}) = 0;

  virtual Status DropIndex(const std::string &column_name) = 0;

  virtual Status Optimize(const OptimizeOptions &options = OptimizeOptions{
                              0}) = 0;

  virtual Status AddColumn(const FieldSchema::Ptr &column_schema,
                           const std::string &expression,
                           const AddColumnOptions &options = AddColumnOptions{
                               0}) = 0;

  virtual Status DropColumn(const std::string &column_name) = 0;

  virtual Status AlterColumn(
      const std::string &column_name, const std::string &rename,
      const FieldSchema::Ptr &new_column_schema = nullptr,
      const AlterColumnOptions &options = AlterColumnOptions{0}) = 0;

  virtual Result<WriteResults> Insert(std::vector<Doc> &docs) = 0;

  virtual Result<WriteResults> Upsert(std::vector<Doc> &docs) = 0;

  virtual Result<WriteResults> Update(std::vector<Doc> &docs) = 0;

  virtual Result<WriteResults> Delete(const std::vector<std::string> &pks) = 0;

  virtual Status DeleteByFilter(const std::string &filter) = 0;

  virtual Result<DocPtrList> Query(const VectorQuery &query) const = 0;

  virtual Result<GroupResults> GroupByQuery(
      const GroupByVectorQuery &query) const = 0;

  virtual Result<DocPtrMap> Fetch(
      const std::vector<std::string> &pks) const = 0;
};

}  // namespace zvec