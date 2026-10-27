// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <zvec/core/framework/index_factory.h>

namespace zvec::core {
// An application can explicitly compose only the modules it needs. This strong
// root satisfies the framework reference without extracting the full SDK root.
void RegisterBuiltinIndexModules() {
  RegisterBuiltinMetrics();
}
}  // namespace zvec::core

int main() {
  using zvec::core::IndexFactory;
  using MetricFactory = zvec::ailego::Factory<zvec::core::IndexMetric>;
  if (!MetricFactory::MakeUnique("SquaredEuclidean")) return 1;
  if (!IndexFactory::CreateMetric("InnerProduct")) return 2;
  if (IndexFactory::HasBuilder("FlatBuilder")) return 3;
  if (IndexFactory::HasStorage("BufferStorage")) return 4;
  return 0;
}
