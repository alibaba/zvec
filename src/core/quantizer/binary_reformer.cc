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
#include <ailego/algorithm/binary_quantizer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

/*! Binary Reformer
 */
class BinaryReformer : public IndexReformer {
 public:
  //! Initialize Reformer
  int init(const ailego::Params & /*params*/) override {
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
    IndexMeta::DataType ft = qmeta.data_type();

    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    size_t dim =
        ailego::BinaryQuantizer::EncodedSizeInBinary32(qmeta.dimension()) * 32u;
    out->resize(
        IndexMeta::ElementSizeof(IndexMeta::DataType::DT_BINARY32, dim));
    const float *vec = reinterpret_cast<const float *>(query);

    quantizer_.encode(vec, qmeta.dimension(),
                      reinterpret_cast<uint32_t *>(&(*out)[0]));
    *ometa = qmeta;
    ometa->set_meta(IndexMeta::DataType::DT_BINARY32, dim);

    return 0;
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    size_t dim =
        ailego::BinaryQuantizer::EncodedSizeInBinary32(qmeta.dimension()) * 32u;
    out->resize(count * IndexMeta::ElementSizeof(
                            IndexMeta::DataType::DT_BINARY32, dim));
    const float *vec = reinterpret_cast<const float *>(query);

    quantizer_.encode(vec, qmeta.dimension() * count,
                      reinterpret_cast<uint32_t *>(&(*out)[0]));
    *ometa = qmeta;
    ometa->set_meta(IndexMeta::DataType::DT_BINARY32, dim);

    return 0;
  }

  //! Normalize results
  int normalize(const void *, const IndexQueryMeta &,
                IndexDocumentList &) const override {
    return 0;
  }

 private:
  //! Members
  ailego::BinaryQuantizer quantizer_{};
};

INDEX_FACTORY_REGISTER_REFORMER(BinaryReformer);

}  // namespace core
}  // namespace zvec