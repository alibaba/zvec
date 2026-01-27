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

#include <string>
#include <vector>

namespace zvec {
namespace core {

/*! Index Plugin
 */
class IndexPlugin {
 public:
  //! Constructor
  IndexPlugin(void) : handle_(nullptr) {}

  //! Constructor
  IndexPlugin(IndexPlugin &&plugin) : handle_(plugin.handle_) {
    plugin.handle_ = nullptr;
  }

  //! Constructor
  explicit IndexPlugin(const std::string &path) : handle_(nullptr) {
    this->load(path);
  }

  //! Destructor
  ~IndexPlugin(void) {}

  //! Test if the plugin is valid
  bool is_valid(void) const {
    return (!!handle_);
  }

  //! Retrieve the handle
  void *handle(void) const {
    return handle_;
  }

  //! Load the library path
  bool load(const std::string &path);

  //! Load the library path
  bool load(const std::string &path, std::string *err);

  //! Unload plugin
  void unload(void);

 private:
  //! Disable them
  IndexPlugin(const IndexPlugin &) = delete;
  IndexPlugin &operator=(const IndexPlugin &) = delete;

  //! Members
  void *handle_;
};

/*! Index Plugin Broker
 */
class IndexPluginBroker {
 public:
  //! Constructor
  IndexPluginBroker(void) : plugins_() {}

  //! Constructor
  IndexPluginBroker(IndexPluginBroker &&broker)
      : plugins_(std::move(broker.plugins_)) {}

  //! Destructor
  ~IndexPluginBroker(void) {}

  //! Emplace a plugin
  bool emplace(IndexPlugin &&plugin);

  //! Emplace a plugin via library path
  bool emplace(const std::string &path) {
    return this->emplace(IndexPlugin(path));
  }

  //! Emplace a plugin via library path
  bool emplace(const std::string &path, std::string *err) {
    IndexPlugin plugin;
    if (!plugin.load(path, err)) {
      return false;
    }
    return this->emplace(std::move(plugin));
  }

  //! Retrieve count of plugins in broker
  size_t count(void) const {
    return plugins_.size();
  }

 private:
  //! Disable them
  IndexPluginBroker(const IndexPluginBroker &) = delete;
  IndexPluginBroker &operator=(const IndexPluginBroker &) = delete;

  //! Members
  std::vector<IndexPlugin> plugins_;
};

}  // namespace core
}  // namespace zvec
