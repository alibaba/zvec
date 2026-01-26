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
#include <zvec/ailego/utility/float_helper.h>
#include "framework/index_factory.h"
#include "record_quantizer.h"

namespace zvec {
namespace core {

/*! Half Float Reformer
 */
class HalfFloatReformer : public IndexReformer {
 public:
  //! Initialize Reformer
  int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform query
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    switch (qmeta.data_type()) {
      case IndexMeta::DataType::DT_FP16:
        out->assign(reinterpret_cast<const char *>(query),
                    qmeta.element_size());
        *ometa = qmeta;
        break;

      case IndexMeta::DataType::DT_FP32:
        if (qmeta.unit_size() != sizeof(float)) {
          return IndexError_Unsupported;
        }
        out->resize(qmeta.dimension() * sizeof(ailego::Float16));
        ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query),
                                    qmeta.dimension(),
                                    reinterpret_cast<uint16_t *>(&(*out)[0]));
        *ometa = qmeta;
        ometa->set_meta(IndexMeta::DataType::DT_FP16, qmeta.dimension());
        break;

      default:
        return IndexError_Unsupported;
    }
    return 0;
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    switch (qmeta.data_type()) {
      case IndexMeta::DataType::DT_FP16:
        out->assign(reinterpret_cast<const char *>(query),
                    qmeta.element_size() * count);
        *ometa = qmeta;
        break;

      case IndexMeta::DataType::DT_FP32:
        if (qmeta.unit_size() != sizeof(float)) {
          return IndexError_Unsupported;
        }
        out->resize(qmeta.dimension() * count * sizeof(ailego::Float16));
        ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query),
                                    qmeta.dimension() * count,
                                    reinterpret_cast<uint16_t *>(&(*out)[0]));
        *ometa = qmeta;
        ometa->set_meta(IndexMeta::DataType::DT_FP16, qmeta.dimension());
        break;

      default:
        return IndexError_Unsupported;
    }
    return 0;
  }

  //! Normalize results
  int normalize(const void *, const IndexQueryMeta &,
                IndexDocumentList &) const override {
    return 0;
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    IndexMeta::DataType type = qmeta.data_type();

    if (type != IndexMeta::DataType::DT_FP16) {
      return IndexError_Unsupported;
    }

    if (type == IndexMeta::DataType::DT_FP16) {
      size_t dimension = qmeta.dimension();

      out->resize(dimension * sizeof(float));
      float *out_buf = reinterpret_cast<float *>(out->data());

      RecordQuantizer::unquantize_record(in, dimension,
                                         IndexMeta::DataType::DT_FP16, out_buf);
    }

    return 0;
  }
};

/*! Half Float Sparse Reformer
 */
class HalfFloatSparseReformer : public IndexReformer {
 public:
  //! Initialize Reformer
  int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform query
  int transform(uint32_t sparse_count, const uint32_t * /*sparse_indices*/,
                const void *sparse_query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    switch (qmeta.data_type()) {
      case IndexMeta::DataType::DT_FP16:
        out->assign(reinterpret_cast<const char *>(sparse_query),
                    qmeta.unit_size() * sparse_count);
        *ometa = qmeta;

        break;

      case IndexMeta::DataType::DT_FP32:
        if (qmeta.unit_size() != sizeof(float)) {
          return IndexError_Unsupported;
        }

        out->resize(sparse_count * sizeof(ailego::Float16));
        ailego::FloatHelper::ToFP16(
            reinterpret_cast<const float *>(sparse_query), sparse_count,
            reinterpret_cast<uint16_t *>(&(*out)[0]));

        *ometa = qmeta;
        ometa->set_data_type(IndexMeta::DataType::DT_FP16);

        break;

      default:
        return IndexError_Unsupported;
    }

    return 0;
  }

  //! Transform queries
  int transform(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                const void *sparse_query, const IndexQueryMeta &qmeta,
                uint32_t count, std::string *out,
                IndexQueryMeta *ometa) const override {
    size_t sparse_count_total = 0;
    for (size_t i = 0; i < count; i++) {
      sparse_count_total += sparse_count[i];
    }

    if (sparse_count_total > std::numeric_limits<uint32_t>::max()) {
      return IndexError_OutOfRange;
    }

    return this->transform((uint32_t)sparse_count_total, sparse_indices,
                           sparse_query, qmeta, out, ometa);
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const uint32_t sparse_count, const uint32_t * /*sparse_indices*/,
             const void *sparse_query, const IndexQueryMeta &qmeta,
             std::string *sparse_query_out) const override {
    IndexMeta::DataType data_type = qmeta.data_type();

    if (data_type != IndexMeta::DataType::DT_FP16) {
      return IndexError_Unsupported;
    }

    if (data_type == IndexMeta::DataType::DT_FP16) {
      sparse_query_out->resize(sparse_count * sizeof(float));

      float *out_buf = reinterpret_cast<float *>(&(*sparse_query_out)[0]);
      RecordQuantizer::unquantize_sparse_record(
          sparse_query, sparse_count, IndexMeta::DataType::DT_FP16, out_buf);
    }

    return 0;
  }
};

INDEX_FACTORY_REGISTER_REFORMER(HalfFloatReformer);
INDEX_FACTORY_REGISTER_REFORMER(HalfFloatSparseReformer);

}  // namespace core
}  // namespace zvec
