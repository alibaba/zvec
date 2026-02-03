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
// limitations under the License

#pragma once

#include <memory>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <zvec/db/type.h>


namespace zvec::sqlengine {

namespace cp = arrow::compute;

class ContainOp {
 public:
  class ContainOptionsType : public arrow::compute::FunctionOptionsType {
   public:
    static const arrow::compute::FunctionOptionsType *GetInstance() {
      static std::unique_ptr<arrow::compute::FunctionOptionsType> instance(
          new ContainOptionsType());
      return instance.get();
    }

    const char *type_name() const override {
      return "Contain";
    }

    std::string Stringify(
        const arrow::compute::FunctionOptions & /*options*/) const override {
      return type_name();
    }

    bool Compare(const arrow::compute::FunctionOptions &options,
                 const arrow::compute::FunctionOptions &other) const override {
      const auto &lop = static_cast<const Options &>(options);
      const auto &rop = static_cast<const Options &>(other);
      if (lop.args.data_type != rop.args.data_type) {
        return false;
      }
      auto *left_value = lop.args.value_set.get();
      auto *right_value = rop.args.value_set.get();
      if (left_value && right_value) {
        return left_value->Equals(*right_value);
      } else if (!left_value && !right_value) {
        return true;
      } else {
        return false;
      }
    }

    std::unique_ptr<arrow::compute::FunctionOptions> Copy(
        const arrow::compute::FunctionOptions &options) const override {
      const auto &opts = static_cast<const Options &>(options);
      return std::make_unique<Options>(opts.args);
    }
  };

  struct Args {
    std::shared_ptr<arrow::Array> value_set;
    DataType data_type;
  };

  class Options : public cp::FunctionOptions {
   public:
    Options() : Options(Args{}) {}

    Options(Args args)
        : cp::FunctionOptions(ContainOptionsType::GetInstance()),
          args(std::move(args)) {}

    static inline constexpr char const kTypeName[] = "ContainFunctionOptions";

    static Options Defaults() {
      return Options();
    }

    Args get_args() const {
      return args;
    }

    Args args;
  };

  struct ContainState : public arrow::compute::KernelState {
    Args args;

    explicit ContainState(const Options *o) {
      if (o) {
        args = o->get_args();
      }
    }
  };


  static inline const cp::FunctionDoc func_doc{
      "check if contain_all/any",
      "returns contain_all/any(x)",
      {"value_set"},
      "Options",
  };

  static arrow::Status register_op();

  static arrow::Result<std::unique_ptr<arrow::compute::KernelState>>
  InitExprValue(arrow::compute::KernelContext *,
                const arrow::compute::KernelInitArgs &args);
};

}  // namespace zvec::sqlengine