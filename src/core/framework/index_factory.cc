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

#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

IndexMetric::Pointer IndexFactory::CreateMetric(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexMetric::Pointer obj =
      ailego::Factory<IndexMetric>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasMetric(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexMetric>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllMetrics() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexMetric>::Classes();
}

IndexDumper::Pointer IndexFactory::CreateDumper(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexDumper::Pointer obj =
      ailego::Factory<IndexDumper>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasDumper(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexDumper>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllDumpers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexDumper>::Classes();
}

IndexStorage::Pointer IndexFactory::CreateStorage(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexStorage::Pointer obj =
      ailego::Factory<IndexStorage>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasStorage(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexStorage>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllStorages() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexStorage>::Classes();
}

IndexConverter::Pointer IndexFactory::CreateConverter(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexConverter::Pointer obj =
      ailego::Factory<IndexConverter>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasConverter(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexConverter>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllConverters() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexConverter>::Classes();
}

IndexReformer::Pointer IndexFactory::CreateReformer(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexReformer::Pointer obj =
      ailego::Factory<IndexReformer>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasReformer(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexReformer>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllReformers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexReformer>::Classes();
}

IndexTrainer::Pointer IndexFactory::CreateTrainer(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexTrainer::Pointer obj =
      ailego::Factory<IndexTrainer>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasTrainer(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexTrainer>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllTrainers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexTrainer>::Classes();
}

IndexBuilder::Pointer IndexFactory::CreateBuilder(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexBuilder::Pointer obj =
      ailego::Factory<IndexBuilder>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasBuilder(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexBuilder>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllBuilders() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexBuilder>::Classes();
}

IndexSearcher::Pointer IndexFactory::CreateSearcher(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexSearcher::Pointer obj =
      ailego::Factory<IndexSearcher>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasSearcher(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexSearcher>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllSearchers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexSearcher>::Classes();
}

IndexStreamer::Pointer IndexFactory::CreateStreamer(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexStreamer::Pointer obj =
      ailego::Factory<IndexStreamer>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasStreamer(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexStreamer>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllStreamers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexStreamer>::Classes();
}

IndexReducer::Pointer IndexFactory::CreateReducer(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexReducer::Pointer obj =
      ailego::Factory<IndexReducer>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasReducer(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexReducer>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllReducers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexReducer>::Classes();
}


IndexCluster::Pointer IndexFactory::CreateCluster(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexCluster::Pointer obj =
      ailego::Factory<IndexCluster>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasCluster(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexCluster>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllClusters() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexCluster>::Classes();
}

IndexStreamerReducer::Pointer IndexFactory::CreateStreamerReducer(
    const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexStreamerReducer::Pointer obj =
      ailego::Factory<IndexStreamerReducer>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasStreamerReducer(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexStreamerReducer>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllStreamerReducers() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexStreamerReducer>::Classes();
}

IndexRefiner::Pointer IndexFactory::CreateRefiner(const std::string &name) {
  RegisterBuiltinIndexModules();
  IndexRefiner::Pointer obj =
      ailego::Factory<IndexRefiner>::MakeShared(name.c_str());
  if (obj) {
    obj->set_name(name);
  }
  return obj;
}

bool IndexFactory::HasRefiner(const std::string &name) {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexRefiner>::Has(name.c_str());
}

std::vector<std::string> IndexFactory::AllRefiners() {
  RegisterBuiltinIndexModules();
  return ailego::Factory<IndexRefiner>::Classes();
}

}  // namespace core
}  // namespace zvec
