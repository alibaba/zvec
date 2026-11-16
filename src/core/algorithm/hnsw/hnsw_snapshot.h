// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#include <zvec/core/framework/index_streamer.h>
namespace zvec::core {
// Separate bridge avoids including dense and RaBitQ HNSW internals together.
int TrainHnswSnapshotRouter(IndexStreamer &streamer, size_t anchors,
                            size_t projected, uint64_t seed);
int DumpHnswSnapshot(IndexStreamer &streamer,
                     const IndexDumper::Pointer &dumper);
}  // namespace zvec::core
