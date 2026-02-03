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

#include <arrow/api.h>
#include <zvec/db/status.h>
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/index/segment/segment.h"

namespace zvec::sqlengine {

namespace cp = arrow::compute;

template <typename OptionsType>
struct OptionsWrapper : public cp::KernelState {
  explicit OptionsWrapper(OptionsType options) : options(std::move(options)) {}

  static arrow::Result<std::unique_ptr<KernelState>> Init(
      cp::KernelContext * /*ctx*/, const cp::KernelInitArgs &args) {
    if (auto options = static_cast<const OptionsType *>(args.options)) {
      return std::make_unique<OptionsWrapper>(*options);
    }

    return arrow::Status::Invalid(
        "Attempted to initialize KernelState from null FunctionOptions");
  }

  static const OptionsType &Get(const KernelState &state) {
    return ::arrow::internal::checked_cast<const OptionsWrapper &>(state)
        .options;
  }

  static const OptionsType &Get(cp::KernelContext *ctx) {
    return Get(*ctx->state());
  }

  OptionsType options;
};

class FetchVectorOp {
 public:
  static Result<std::shared_ptr<arrow::Array>> fetch_dense_vector(
      const Segment &segment, const std::string &vector_name,
      const std::vector<int> &rows);

  static Result<std::shared_ptr<arrow::Array>> fetch_sparse_vector(
      const Segment &segment, const std::string &vector_name,
      const std::vector<int> &rows);

  static arrow::Status register_op();

  class FunctionOptionsType : public cp::FunctionOptionsType {
    const char *type_name() const override {
      return "FetchVectorFunctionOptionsType";
    }

    std::string Stringify(const cp::FunctionOptions &) const override {
      return "FetchVectorFunctionOptionsType";
    }

    bool Compare(const cp::FunctionOptions &,
                 const cp::FunctionOptions &) const override {
      return false;
    }

    std::unique_ptr<cp::FunctionOptions> Copy(
        const cp::FunctionOptions &) const override;
    // optional: support for serialization
    // Result<std::shared_ptr<Buffer>> Serialize(const FunctionOptions&) const
    // override; Result<std::unique_ptr<FunctionOptions>> Deserialize(const
    // Buffer&) const override;
  };

  static cp::FunctionOptionsType *GetFetchVectorFunctionOptionsType() {
    static FunctionOptionsType options_type;
    return &options_type;
  }

  class FetchVectorFunctionOptions : public cp::FunctionOptions {
   public:
    FetchVectorFunctionOptions()
        : cp::FunctionOptions(GetFetchVectorFunctionOptionsType()) {}
  };

  class FetchVectorOptionsType : public arrow::compute::FunctionOptionsType {
   public:
    static const arrow::compute::FunctionOptionsType *GetInstance() {
      static std::unique_ptr<arrow::compute::FunctionOptionsType> instance(
          new FetchVectorOptionsType());
      return instance.get();
    }

    const char *type_name() const override {
      return "FetchVector";
    }

    std::string Stringify(
        const arrow::compute::FunctionOptions & /*options*/) const override {
      return type_name();
    }

    bool Compare(const arrow::compute::FunctionOptions &options,
                 const arrow::compute::FunctionOptions &other) const override {
      const auto &lop = static_cast<const Options &>(options);
      const auto &rop = static_cast<const Options &>(other);
      return lop.args.is_dense == rop.args.is_dense &&
             lop.args.indexer == rop.args.indexer;
    }

    std::unique_ptr<arrow::compute::FunctionOptions> Copy(
        const arrow::compute::FunctionOptions &options) const override {
      const auto &opts = static_cast<const Options &>(options);
      return std::make_unique<Options>(opts.args.indexer, opts.args.is_dense);
    }
  };

  struct Args {
    CombinedVectorColumnIndexer::Ptr indexer;
    bool is_dense{true};
  };

  class Options : public cp::FunctionOptions {
   public:
    Options() : Options(nullptr, true) {}

    Options(CombinedVectorColumnIndexer::Ptr indexer, bool is_dense)
        : cp::FunctionOptions(FetchVectorOptionsType::GetInstance()),
          args{indexer, is_dense} {}

    static inline constexpr char const kTypeName[] =
        "FetchVectorFunctionOptions";

    static Options Defaults() {
      return Options();
    }

    Args get_args() const {
      return args;
    }

    Args args;
  };

  struct FetchVectorState : public arrow::compute::KernelState {
    Args args;

    explicit FetchVectorState(const Options *o) {
      if (o) {
        args = o->get_args();
      }
    }
  };


  static arrow::Status FetchVectorFunction(cp::KernelContext *ctx,
                                           const cp::ExecSpan &batch,
                                           cp::ExecResult *out);

  static inline const cp::FunctionDoc func_doc{
      "fetch dense or sparse vector",
      "returns fetch_vector(x)",
      {"segment_row_id"},
      "Options",
  };

  static arrow::Result<std::unique_ptr<arrow::compute::KernelState>>
  InitExprValue(arrow::compute::KernelContext *,
                const arrow::compute::KernelInitArgs &args);
};

}  // namespace zvec::sqlengine