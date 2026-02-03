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

#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace zvec {
namespace ailego {

/*! Factory
 */
template <typename TBase>
class Factory {
 public:
  /*! Factory Register
   */
  template <typename TImpl, typename = typename std::enable_if<
                                std::is_base_of<TBase, TImpl>::value>::type>
  class Register {
   public:
    //! Constructor
    Register(const char *key) {
      Factory::Instance()->set(key, [] { return Register::Construct(); });
    }

    //! Constructor
    template <typename... TArgs>
    Register(const char *key, TArgs &&...args) {
      std::tuple<TArgs...> tuple(std::forward<TArgs>(args)...);

      Factory::Instance()->set(key, [tuple] {
        return Register::Construct(
            tuple, typename TupleIndexMaker<sizeof...(TArgs)>::Type());
      });
    }

   protected:
    //! Tuple Index Maker
    template <size_t N, size_t... I>
    struct TupleIndexMaker : TupleIndexMaker<N - 1, N - 1, I...> {};

    //! Tuple Index
    template <size_t...>
    struct TupleIndex {};

    //! Tuple Index Maker (special)
    template <size_t... I>
    struct TupleIndexMaker<0, I...> {
      typedef TupleIndex<I...> Type;
    };

    //! Construct a register object
    template <typename... TArgs, size_t... I>
    static TImpl *Construct(const std::tuple<TArgs...> &tuple,
                            TupleIndex<I...>) {
      return new (std::nothrow) TImpl(std::get<I>(tuple)...);
    }

    //! Construct a register object
    static TImpl *Construct(void) {
      return new (std::nothrow) TImpl();
    }
  };

  //! Produce an instance (c_ptr)
  static TBase *Make(const char *key) {
    return Factory::Instance()->produce(key);
  }

  //! Produce an instance (shared_ptr)
  static std::shared_ptr<TBase> MakeShared(const char *key) {
    return std::shared_ptr<TBase>(Factory::Make(key));
  }

  //! Produce an instance (unique_ptr)
  static std::unique_ptr<TBase> MakeUnique(const char *key) {
    return std::unique_ptr<TBase>(Factory::Make(key));
  }

  //! Test if the class is exist
  static bool Has(const char *key) {
    return Factory::Instance()->has(key);
  }

  //! Retrieve classes in factory
  static std::vector<std::string> Classes(void) {
    return Factory::Instance()->classes();
  }

 protected:
  //! Constructor
  Factory(void) : map_() {}

  //! Retrieve the singleton factory
  static Factory *Instance(void) {
    static Factory factory;
    return (&factory);
  }

  //! Inserts a new class into map
  template <typename TFunc>
  void set(const char *key, TFunc &&func) {
    map_[key] = std::forward<TFunc>(func);
  }

  //! Produce an instance
  TBase *produce(const char *key) {
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      return iter->second();
    }
    return nullptr;
  }

  //! Test if the class is exist
  bool has(const char *key) {
    return (map_.find(key) != map_.end());
  }

  //! Retrieve classes in factory
  std::vector<std::string> classes(void) const {
    std::vector<std::string> vec;
    for (const auto &it : map_) {
      vec.push_back(std::string(it.first));
    }
    return vec;
  }

 private:
  //! Disable them
  Factory(const Factory &);
  Factory(Factory &&);
  Factory &operator=(const Factory &);

  /*! Key Comparer
   */
  struct KeyComparer {
    bool operator()(const char *lhs, const char *rhs) const {
      return (std::strcmp(lhs, rhs) < 0);
    }
  };

  //! Don't use variable buffer as key store.
  //! The key must be use a static buffer to store.
  std::map<const char *, std::function<TBase *()>, KeyComparer> map_;
};

//! Factory Register
#define AILEGO_FACTORY_REGISTER(__NAME__, __BASE__, __IMPL__, ...) \
  static ailego::Factory<__BASE__>::Register<__IMPL__>             \
      __ailegoFactoryRegister_##__NAME__(#__NAME__, ##__VA_ARGS__)

}  // namespace ailego
}  // namespace zvec
