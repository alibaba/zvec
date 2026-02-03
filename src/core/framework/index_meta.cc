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

#include <zvec/ailego/encoding/json.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

/*! Index Meta Buffer Format
 */
struct IndexMetaFormatHeader {
  uint32_t header_size;
  uint32_t meta_type;
  uint32_t major_order;
  uint32_t data_type;
  uint32_t dimension;
  uint32_t unit_size;
  uint32_t space_id;
  uint32_t attachment_offset;
  uint32_t attachment_size;
  uint8_t reserved_[4092];
};

static_assert(sizeof(IndexMetaFormatHeader) % 32 == 0,
              "IndexMetaBufferFormat must be aligned with 32 bytes");

void IndexMeta::serialize(std::string *out) const {
  ailego::Params attachment;
  IndexMetaFormatHeader format;
  memset(&format, 0, sizeof(format));
  format.header_size = sizeof(format);
  format.meta_type = static_cast<uint32_t>(meta_type_);
  format.major_order = static_cast<uint32_t>(major_order_);
  format.data_type = static_cast<uint32_t>(data_type_);
  format.dimension = dimension_;
  format.unit_size = unit_size_;
  format.space_id = space_id_;

  if (!metric_name_.empty()) {
    ailego::Params item;
    item.set("name", metric_name_);
    item.set("revision", metric_revision_);
    item.set("params", metric_params_);
    attachment.set("metric", std::move(item));
  }

  if (!converter_name_.empty()) {
    ailego::Params item;
    item.set("name", converter_name_);
    item.set("revision", converter_revision_);
    item.set("params", converter_params_);
    attachment.set("converter", std::move(item));
  }
  if (!reformer_name_.empty()) {
    ailego::Params item;
    item.set("name", reformer_name_);
    item.set("revision", reformer_revision_);
    item.set("params", reformer_params_);
    attachment.set("reformer", std::move(item));
  }
  if (!trainer_name_.empty()) {
    ailego::Params item;
    item.set("name", trainer_name_);
    item.set("revision", trainer_revision_);
    item.set("params", trainer_params_);
    attachment.set("trainer", std::move(item));
  }
  if (!builder_name_.empty()) {
    ailego::Params item;
    item.set("name", builder_name_);
    item.set("revision", builder_revision_);
    item.set("params", builder_params_);
    attachment.set("builder", std::move(item));
  }
  if (!reducer_name_.empty()) {
    ailego::Params item;
    item.set("name", reducer_name_);
    item.set("revision", reducer_revision_);
    item.set("params", reducer_params_);
    attachment.set("reducer", std::move(item));
  }
  if (!searcher_name_.empty()) {
    ailego::Params item;
    item.set("name", searcher_name_);
    item.set("revision", searcher_revision_);
    item.set("params", searcher_params_);
    attachment.set("searcher", std::move(item));
  }
  if (!streamer_name_.empty()) {
    ailego::Params item;
    item.set("name", streamer_name_);
    item.set("revision", streamer_revision_);
    item.set("params", streamer_params_);
    attachment.set("streamer", std::move(item));
  }

  if (!attributes_.empty()) {
    attachment.set("attributes", attributes_);
  }

  out->assign(reinterpret_cast<const char *>(&format), sizeof(format));
  size_t offset = static_cast<uint32_t>(out->size());

  if (!attachment.empty()) {
    std::string buf;
    ailego::Params::SerializeToBuffer(attachment, &buf);
    out->append(buf.data(), buf.size());
    IndexMetaFormatHeader *header = (IndexMetaFormatHeader *)out->data();
    header->attachment_offset = static_cast<uint32_t>(offset);
    header->attachment_size = static_cast<uint32_t>(buf.size());
    offset += buf.size();
  }
}

bool IndexMeta::deserialize(const void *data, size_t len) {
  const IndexMetaFormatHeader *format =
      reinterpret_cast<const IndexMetaFormatHeader *>(data);

  this->clear();
  if (sizeof(IndexMetaFormatHeader) > len) {
    return false;
  }
  if (sizeof(IndexMetaFormatHeader) > format->header_size) {
    return false;
  }

  meta_type_ = static_cast<IndexMeta::MetaType>(format->meta_type);
  major_order_ = static_cast<IndexMeta::MajorOrder>(format->major_order);
  data_type_ = static_cast<IndexMeta::DataType>(format->data_type);
  dimension_ = format->dimension;
  unit_size_ = format->unit_size;
  element_size_ = IndexMeta::ElementSizeof(data_type_, unit_size_, dimension_);
  space_id_ = format->space_id;

  // Read attachment
  ailego::Params attachment;
  if (format->attachment_size) {
    if (format->attachment_offset + format->attachment_size > len) {
      return false;
    }
    std::string str(
        reinterpret_cast<const char *>(data) + format->attachment_offset,
        format->attachment_size);
    if (!ailego::Params::ParseFromBuffer(str, &attachment)) {
      return false;
    }
  }

  ailego::Params item;
  if (attachment.get("metric", &item)) {
    item.get("name", &metric_name_);
    item.get("revision", &metric_revision_);
    item.get("params", &metric_params_);
  }
  if (attachment.get("converter", &item)) {
    item.get("name", &converter_name_);
    item.get("revision", &converter_revision_);
    item.get("params", &converter_params_);
  }
  if (attachment.get("reformer", &item)) {
    item.get("name", &reformer_name_);
    item.get("revision", &reformer_revision_);
    item.get("params", &reformer_params_);
  }
  if (attachment.get("trainer", &item)) {
    item.get("name", &trainer_name_);
    item.get("revision", &trainer_revision_);
    item.get("params", &trainer_params_);
  }
  if (attachment.get("builder", &item)) {
    item.get("name", &builder_name_);
    item.get("revision", &builder_revision_);
    item.get("params", &builder_params_);
  }
  if (attachment.get("reducer", &item)) {
    item.get("name", &reducer_name_);
    item.get("revision", &reducer_revision_);
    item.get("params", &reducer_params_);
  }
  if (attachment.get("searcher", &item)) {
    item.get("name", &searcher_name_);
    item.get("revision", &searcher_revision_);
    item.get("params", &searcher_params_);
  }
  if (attachment.get("streamer", &item)) {
    item.get("name", &streamer_name_);
    item.get("revision", &streamer_revision_);
    item.get("params", &streamer_params_);
  }
  attachment.get("attributes", &attributes_);

  return true;
}

}  // namespace core
}  // namespace zvec
