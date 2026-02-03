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
#include <ailego/utility/dl_helper.h>
#include <zvec/core/framework/index_plugin.h>

namespace zvec {
namespace core {

bool IndexPlugin::load(const std::string &path) {
  if (handle_) {
    return false;
  }
  handle_ = ailego::DLHelper::Load(path, nullptr);
  return (!!handle_);
}

bool IndexPlugin::load(const std::string &path, std::string *err) {
  if (handle_) {
    *err = "plugin loaded";
    return false;
  }
  handle_ = ailego::DLHelper::Load(path, err);
  return !!handle_;
}

void IndexPlugin::unload(void) {
  if (handle_) {
    ailego::DLHelper::Unload(handle_);
    handle_ = nullptr;
  }
}

bool IndexPluginBroker::emplace(IndexPlugin &&plugin) {
  if (!plugin.is_valid()) {
    return false;
  }
  for (auto iter = plugins_.begin(); iter != plugins_.end(); ++iter) {
    if (iter->handle() == plugin.handle()) {
      plugin.unload();
      return true;
    }
  }
  plugins_.push_back(std::move(plugin));
  return true;
}

}  // namespace core
}  // namespace zvec
