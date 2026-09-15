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
  IndexPlugin() : handle_(nullptr) {}

  //! Constructor
  IndexPlugin(IndexPlugin &&plugin) : handle_(plugin.handle_) {
    plugin.handle_ = nullptr;
  }

  //! Constructor
  explicit IndexPlugin(const std::string &path) : handle_(nullptr) {
    this->load(path);
  }

  //! Destructor
  ~IndexPlugin() = default;

  //! Test if the plugin is valid
  bool is_valid() const {
    return (!!handle_);
  }

  //! Retrieve the handle
  void *handle() const {
    return handle_;
  }

  //! Load the library path
  bool load(const std::string &path);

  //! Load the library path
  bool load(const std::string &path, std::string *err);

  //! Unload plugin
  void unload();

  //! Disable them
  IndexPlugin(const IndexPlugin &) = delete;
  IndexPlugin &operator=(const IndexPlugin &) = delete;

 private:
  //! Members
  void *handle_;
};

/*! Index Plugin Broker
 */
class IndexPluginBroker {
 public:
  //! Constructor
  IndexPluginBroker() : plugins_() {}

  //! Constructor
  IndexPluginBroker(IndexPluginBroker &&broker)
      : plugins_(std::move(broker.plugins_)) {}

  //! Destructor
  ~IndexPluginBroker() = default;

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
  size_t count() const {
    return plugins_.size();
  }

  //! Disable them
  IndexPluginBroker(const IndexPluginBroker &) = delete;
  IndexPluginBroker &operator=(const IndexPluginBroker &) = delete;

 private:
  //! Members
  std::vector<IndexPlugin> plugins_;
};

}  // namespace core
}  // namespace zvec
