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
#include "hnsw_rabitq_builder.h"
#include "hnsw_rabitq_searcher.h"
#include "hnsw_rabitq_streamer.h"
#include "rabitq_converter.h"
#include "rabitq_reformer.h"

namespace zvec::core {

INDEX_FACTORY_REGISTER_STREAMER(HnswRabitqStreamer);
INDEX_FACTORY_REGISTER_REFORMER_ALIAS(RabitqReformer, RabitqReformer);
INDEX_FACTORY_REGISTER_SEARCHER(HnswRabitqSearcher);
INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(RabitqConverter, RabitqConverter);
INDEX_FACTORY_REGISTER_BUILDER(HnswRabitqBuilder);

}  // namespace zvec::core