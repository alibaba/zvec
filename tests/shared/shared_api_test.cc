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

#include <cmath>
#include <cstdint>
#include <memory>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>

int main() {
  zvec::float16_t half_value(1.5F);
  if (std::fabs(static_cast<float>(half_value) - 1.5F) > 0.001F) {
    return 1;
  }

  zvec::CollectionSchema schema("shared-api-test");
  auto field = std::make_shared<zvec::FieldSchema>("id", zvec::DataType::INT64);
  if (!schema.add_field(std::move(field)).ok() || !schema.has_field("id")) {
    return 2;
  }

  zvec::Doc doc;
  doc.set_pk("1");
  if (!doc.set<int64_t>("id", 1)) {
    return 3;
  }

  return 0;
}
