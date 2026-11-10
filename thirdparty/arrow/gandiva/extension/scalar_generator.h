// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "gandiva/gandiva_aliases.h"
#include "gandiva/scalar_predicate_abi.h"
#include "gandiva/visibility.h"

namespace gandiva {

class Configuration;
class LLVMGenerator;
class LLVMTypes;

// Optional single-row entry, separate from LLVMGenerator's batch API and state.
// Reuses its expression visitor so ordinary operator semantics have one owner.
class GANDIVA_EXPORT ScalarGenerator {
 public:
  static arrow::Result<std::unique_ptr<ScalarGenerator>> Make(
      const std::shared_ptr<Configuration>& configuration);
  ~ScalarGenerator();

  arrow::Result<ScalarEvalFunction> Build(
      const ExpressionPtr& expression,
      const std::vector<std::string>& parameter_field_names = {});
  arrow::Status BindBuffers(const arrow::RecordBatch& batch,
                            const std::vector<ScalarParameterSlot>& parameters,
                            std::vector<const uint8_t*>* buffers,
                            std::vector<int64_t>* offsets) const;

  LLVMTypes* types() const;
  const std::string& ir() const;
  const void* const* holders() const;
  const std::unordered_map<std::string, FieldDescriptorPtr>& input_descriptors() const;

 private:
  explicit ScalarGenerator(std::unique_ptr<LLVMGenerator> generator);
  arrow::Status CodeGen(const ValueValidityPair& expression,
                        const std::unordered_set<int>& parameter_buffers,
                        const std::string& function_name);

  std::unique_ptr<LLVMGenerator> generator_;
  // Keep decomposed expression/function holders alive for every bound query.
  ValueValidityPairPtr expression_;
  std::unordered_map<std::string, size_t> parameter_indices_;
};

}  // namespace gandiva
