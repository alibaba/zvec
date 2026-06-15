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

#include <atomic>
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

  //! Retrieve the singleton factory.
  //!
  //! We deliberately avoid the canonical
  //!
  //!   static Factory factory; return &factory;
  //!
  //! "magic static" pattern here. That pattern requires the compiler to emit
  //! a guard variable (`_ZGVZN...E7factory`) alongside the object, and both
  //! must be unified across DSOs for the singleton to be shared.
  //!
  //! In the Python extension build the _zvec.so version script exports the
  //! storage (`zvec::*` matches its demangled name) while the guard variable,
  //! whose demangled form is `guard variable for zvec::...`, ends up hidden
  //! (compilers emit the guard in a COMDAT group whose visibility is not
  //! upgraded by our version script). When libzvec_diskann_plugin.so is
  //! loaded, _zvec.so and the plugin then share the `factory` storage but
  //! each have their own guard; the plugin's still-zero guard triggers a
  //! second run of the Factory constructor on the shared storage, wiping
  //! all registrations performed during _zvec.so import (e.g. FlatStreamer).
  //!
  //! A constant-initialized static std::atomic<T*> has NO guard variable
  //! (its zero init is compile-time), so we use a leaked heap singleton with
  //! atomic double-checked locking. Only the atomic pointer itself needs to
  //! be unified across DSOs, and it merges cleanly as STT_GNU_UNIQUE.
  static Factory *Instance(void) {
    static std::atomic<Factory *> ptr{nullptr};
    Factory *cur = ptr.load(std::memory_order_acquire);
    if (cur) {
      return cur;
    }
    Factory *created = new Factory();
    if (ptr.compare_exchange_strong(cur, created, std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
      return created;
    }
    delete created;
    return cur;
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

// Force compiler/linker retention of the factory-registration static object.
// Under clang -O3 + --gc-sections + install.strip (CI Release), these
// file-scope static objects have no external references and can be silently
// dropped along with their .init_array entries, causing runtime
// 'Create vector column indexer failed' failures for indexers like
// VamanaStreamer. `used` defeats compiler dead-code elimination and
// `retain` (GCC>=11 / Clang>=13) defeats linker `--gc-sections`. On MSVC
// the equivalent is achieved elsewhere via /WHOLEARCHIVE. Fall back to just
// `used` on older toolchains that don't know `retain`.
#if defined(__has_attribute)
#if __has_attribute(retain) && __has_attribute(used)
#define AILEGO_FACTORY_REGISTER_ATTR __attribute__((used, retain))
#elif __has_attribute(used)
#define AILEGO_FACTORY_REGISTER_ATTR __attribute__((used))
#else
#define AILEGO_FACTORY_REGISTER_ATTR
#endif
#elif defined(__GNUC__)
#define AILEGO_FACTORY_REGISTER_ATTR __attribute__((used))
#else
#define AILEGO_FACTORY_REGISTER_ATTR
#endif

//! Factory Register
#define AILEGO_FACTORY_REGISTER(__NAME__, __BASE__, __IMPL__, ...) \
  static AILEGO_FACTORY_REGISTER_ATTR                              \
      ailego::Factory<__BASE__>::Register<__IMPL__>                \
          __ailegoFactoryRegister_##__NAME__(#__NAME__, ##__VA_ARGS__)

}  // namespace ailego
}  // namespace zvec
