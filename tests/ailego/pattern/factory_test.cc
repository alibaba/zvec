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

#include <algorithm>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/pattern/factory.h>

using namespace zvec;
using namespace zvec::ailego;

struct Base {
  virtual ~Base() = default;
  virtual void do_something() = 0;
};

struct AAA : public Base {
  AAA() = default;

  void do_something() override {
    printf("do something\n");
  }
};

AILEGO_FACTORY_REGISTER(AAA, Base, AAA);

TEST(Factory, General) {
  EXPECT_TRUE(!ailego::Factory<Base>::MakeShared("BBB"));
  EXPECT_TRUE(!ailego::Factory<Base>::Has("BBB"));

  auto aaa = ailego::Factory<Base>::MakeShared("AAA");
  ASSERT_TRUE(!!aaa);
  aaa->do_something();
  EXPECT_TRUE(!!ailego::Factory<Base>::Has("AAA"));

  auto vec = ailego::Factory<Base>::Classes();
  EXPECT_EQ(1u, vec.size());
  EXPECT_EQ("AAA", std::string(vec[0]));
}

namespace {

struct ValueBase {
  virtual ~ValueBase() = default;
  virtual int value() const = 0;
};

struct Value : ValueBase {
  explicit Value(int input = 1) : value_(input) {}
  int value() const override {
    return value_;
  }
  int value_;
};

using ValueFactory = ailego::Factory<ValueBase>;

struct RecursiveValue : ValueBase {
  RecursiveValue() : child_(ValueFactory::MakeShared("RecursiveLeaf")) {}
  int value() const override {
    return child_ ? child_->value() : -1;
  }
  std::shared_ptr<ValueBase> child_;
};

struct ArgumentState {
  bool query_on_copy{false};
  bool query_on_destroy{false};
  size_t copies{0};
  size_t destructions{0};
};

struct ReentrantArgument {
  explicit ReentrantArgument(std::shared_ptr<ArgumentState> input)
      : state(std::move(input)) {}
  ReentrantArgument(const ReentrantArgument &other) : state(other.state) {
    ++state->copies;
    if (state->query_on_copy) {
      ValueFactory::Has("ReentrantLookup");
    }
  }
  ReentrantArgument(ReentrantArgument &&) = default;
  ~ReentrantArgument() {
    if (state) {
      ++state->destructions;
      if (state->query_on_destroy) {
        ValueFactory::Has("ReentrantLookup");
      }
    }
  }
  std::shared_ptr<ArgumentState> state;
};

struct ArgumentValue : ValueBase {
  explicit ArgumentValue(const ReentrantArgument &) {}
  int value() const override {
    return 31;
  }
};

}  // namespace

TEST(Factory, BuiltinDefaultsPreserveExplicitRegistrations) {
  ValueFactory::Register<Value> extension_first("ExtensionFirst", 7);
  ValueFactory::Register<Value> builtin_after(ValueFactory::BuiltinTag{},
                                              "ExtensionFirst", 1);
  auto value = ValueFactory::MakeShared("ExtensionFirst");
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(7, value->value());

  ValueFactory::Register<Value> builtin_first(ValueFactory::BuiltinTag{},
                                              "BuiltinFirst", 1);
  ValueFactory::Register<Value> extension_after("BuiltinFirst", 9);
  ValueFactory::Register<Value> repeated_builtin(ValueFactory::BuiltinTag{},
                                                 "BuiltinFirst", 2);
  value = ValueFactory::MakeShared("BuiltinFirst");
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(9, value->value());
}

TEST(Factory, ConstructorCanCreateFromSameFactory) {
  ValueFactory::Register<Value> leaf(ValueFactory::BuiltinTag{},
                                     "RecursiveLeaf", 11);
  ValueFactory::Register<RecursiveValue> parent(ValueFactory::BuiltinTag{},
                                                "RecursiveParent");
  auto value = ValueFactory::MakeShared("RecursiveParent");
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(11, value->value());
}

TEST(Factory, CapturedArgumentCopyCanQuerySameFactory) {
  auto state = std::make_shared<ArgumentState>();
  state->query_on_copy = true;
  ValueFactory::Register<ArgumentValue> registration("CopyingArgument",
                                                     ReentrantArgument{state});
  EXPECT_GT(state->copies, 0u);

  auto value = ValueFactory::MakeShared("CopyingArgument");
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(31, value->value());
}

TEST(Factory, ReplacedArgumentDestructorCanQuerySameFactory) {
  auto state = std::make_shared<ArgumentState>();
  ValueFactory::Register<ArgumentValue> original("ReplacedArgument",
                                                 ReentrantArgument{state});
  state->query_on_destroy = true;
  auto destructions = state->destructions;
  ValueFactory::Register<Value> replacement("ReplacedArgument", 17);
  EXPECT_GT(state->destructions, destructions);

  auto value = ValueFactory::MakeShared("ReplacedArgument");
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(17, value->value());
}

TEST(Factory, IgnoredBuiltinArgumentDestructorCanQuerySameFactory) {
  ValueFactory::Register<Value> extension("IgnoredBuiltinArgument", 19);
  auto state = std::make_shared<ArgumentState>();
  state->query_on_destroy = true;
  ValueFactory::Register<ArgumentValue> builtin(ValueFactory::BuiltinTag{},
                                                "IgnoredBuiltinArgument",
                                                ReentrantArgument{state});
  EXPECT_GT(state->destructions, 0u);

  auto value = ValueFactory::MakeShared("IgnoredBuiltinArgument");
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(19, value->value());
}

TEST(Factory, ConcurrentRegistrationAndLookup) {
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([worker] {
      for (size_t iteration = 0; iteration < 128; ++iteration) {
        ValueFactory::Register<Value> builtin(ValueFactory::BuiltinTag{},
                                              "ConcurrentBuiltin", 3);
        ValueFactory::Register<Value> extension("ConcurrentExtension",
                                                int{worker});
        EXPECT_TRUE(ValueFactory::Has("ConcurrentBuiltin"));
        auto names = ValueFactory::Classes();
        EXPECT_NE(names.end(),
                  std::find(names.begin(), names.end(), "ConcurrentExtension"));
        auto builtin_value = ValueFactory::MakeShared("ConcurrentBuiltin");
        ASSERT_NE(nullptr, builtin_value);
        EXPECT_EQ(3, builtin_value->value());
        auto extension_value = ValueFactory::MakeShared("ConcurrentExtension");
        ASSERT_NE(nullptr, extension_value);
        EXPECT_GE(extension_value->value(), 0);
        EXPECT_LT(extension_value->value(), 8);
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
}
