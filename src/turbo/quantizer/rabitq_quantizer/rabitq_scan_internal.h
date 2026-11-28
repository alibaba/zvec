// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <rabitqlib/fastscan/query.hpp>
#include "quantizer/quantizer.h"

namespace zvec::turbo::rabitq_internal {
// Private full-block epilogue. The caller selects the ISA and accumulates LUT
// sums with rabitqlib's matching runtime backend before entering this kernel.
int PruneBlockAvx512(const void *block, size_t dim, size_t clusters, int bits,
                     bool inner_product, const float *scalars,
                     const rabitqlib::SplitBatchQuery<float> &query,
                     const int32_t *sums, float worst, DistanceEstimate *out,
                     uint32_t *candidates);
}  // namespace zvec::turbo::rabitq_internal
