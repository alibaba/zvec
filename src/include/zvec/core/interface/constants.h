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

#include <cstdint>

namespace zvec::core_interface {

// Common query parameters.
inline constexpr float kDefaultQueryRadius = 0.0f;
inline constexpr bool kDefaultQueryIsLinear = false;

// DiskANN: build.
inline constexpr uint32_t kDefaultDiskAnnMaxDegree = 100;
inline constexpr uint32_t kDefaultDiskAnnBuildListSize = 50;
inline constexpr uint32_t kDefaultDiskAnnPqNumChunks = 0;

// DiskANN: query.
inline constexpr uint32_t kDefaultDiskAnnSearchListSize = 300;
inline constexpr float kDefaultDiskAnnRefineScaleFactor = 10.0f;

// Flat: query.
inline constexpr float kDefaultFlatRefineScaleFactor = 10.0f;

// HNSW: build.
inline constexpr uint32_t kDefaultHnswNeighborCount = 50;
inline constexpr uint32_t kDefaultHnswEfConstruction = 500;

// HNSW: query.
inline constexpr uint32_t kDefaultHnswEfSearch = 300;
inline constexpr float kDefaultHnswRefineScaleFactor = 0.0f;
inline constexpr uint32_t kDefaultHnswPrefetchOffset = 8;
inline constexpr uint32_t kDefaultHnswPrefetchLines = 0;

// HNSW RaBitQ: build.
inline constexpr uint32_t kDefaultHnswRabitqNumClusters = 16;

// HNSW RaBitQ: query.
inline constexpr float kDefaultHnswRabitqRefineScaleFactor = 0.0f;

// IVF: build.
inline constexpr uint32_t kDefaultIvfNumLists = 1024;
inline constexpr uint32_t kDefaultIvfNumIterations = 10;
inline constexpr bool kDefaultIvfUseSoar = false;

// IVF: query.
inline constexpr uint32_t kDefaultIvfNumProbes = 10;
inline constexpr float kDefaultIvfRefineScaleFactor = 10.0f;

// IVF RaBitQ: build.
inline constexpr uint32_t kDefaultIvfRabitqNumLists = 1024;

// IVF RaBitQ: query.
inline constexpr uint32_t kDefaultIvfRabitqNumProbes = 10;
inline constexpr float kDefaultIvfRabitqRefineScaleFactor = 10.0f;

// Vamana: build.
inline constexpr uint32_t kDefaultVamanaMaxDegree = 64;
inline constexpr uint32_t kDefaultVamanaBuildListSize = 100;
inline constexpr float kDefaultVamanaAlpha = 1.2f;
inline constexpr uint32_t kDefaultVamanaMaxOcclusionSize = 750;
inline constexpr bool kDefaultVamanaSaturateGraph = false;
inline constexpr bool kDefaultVamanaTwoPassBuild = true;

// Vamana: query.
inline constexpr uint32_t kDefaultVamanaEfSearch = 200;
inline constexpr float kDefaultVamanaRefineScaleFactor = 0.0f;
inline constexpr uint32_t kDefaultVamanaPrefetchOffset = 8;
inline constexpr uint32_t kDefaultVamanaPrefetchLines = 0;

// Quantization parameters.
inline constexpr bool kDefaultQuantizerEnableRotate = false;

// PQ / OPQ.
inline constexpr uint32_t kDefaultPqNumChunks = 8;
inline constexpr uint32_t kDefaultPqNumBits = 8;
inline constexpr bool kDefaultPqFastScan = false;
inline constexpr uint32_t kDefaultOpqNumIterations = 5;
inline constexpr uint32_t kDefaultOpqPqNumIterations = 4;

// RaBitQ.
inline constexpr uint32_t kDefaultRabitqTotalBits = 7;
inline constexpr uint32_t kDefaultRabitqSampleCount = 0;

}  // namespace zvec::core_interface
