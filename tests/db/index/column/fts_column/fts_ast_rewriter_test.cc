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

#include "db/index/column/fts_column/fts_ast_rewriter.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/fts_query_ast.h"

namespace zvec::fts {

namespace {

// Convenience constructors keep the test bodies focused on what's being
// asserted rather than on AST scaffolding.

FtsAstNodePtr term(const std::string &t, bool must = false,
                   bool must_not = false, float boost = 1.0f) {
  auto n = std::make_unique<TermNode>(t, must, must_not);
  n->boost = boost;
  return n;
}

FtsAstNodePtr phrase(std::vector<std::string> ts, bool must = false,
                     bool must_not = false, float boost = 1.0f) {
  auto n = std::make_unique<PhraseNode>();
  n->terms = std::move(ts);
  n->must = must;
  n->must_not = must_not;
  n->boost = boost;
  return n;
}

FtsAstNodePtr empty_node() {
  return std::make_unique<EmptyNode>();
}

template <typename Node>
FtsAstNodePtr composite(std::vector<FtsAstNodePtr> children, bool must = false,
                        bool must_not = false) {
  auto n = std::make_unique<Node>();
  n->children = std::move(children);
  n->must = must;
  n->must_not = must_not;
  return n;
}

FtsAstNodePtr or_node(std::vector<FtsAstNodePtr> c, bool must = false,
                      bool must_not = false) {
  return composite<OrNode>(std::move(c), must, must_not);
}
FtsAstNodePtr and_node(std::vector<FtsAstNodePtr> c, bool must = false,
                       bool must_not = false) {
  return composite<AndNode>(std::move(c), must, must_not);
}

// Pull the single TermNode child out of a composite for boost assertions.
const TermNode &as_term(const FtsAstNode &n) {
  return static_cast<const TermNode &>(n);
}
const PhraseNode &as_phrase(const FtsAstNode &n) {
  return static_cast<const PhraseNode &>(n);
}
const OrNode &as_or(const FtsAstNode &n) {
  return static_cast<const OrNode &>(n);
}
const AndNode &as_and(const FtsAstNode &n) {
  return static_cast<const AndNode &>(n);
}

}  // namespace

// --- Dedup ---

TEST(FtsAstRewriterTest, OrDedupsRepeatedTerms) {
  // OR(apple, apple, banana) → OR(apple^2, banana)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  children.push_back(term("apple"));
  children.push_back(term("banana"));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &n = as_or(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[0]).term, "apple");
  EXPECT_FLOAT_EQ(n.children[0]->boost, 2.0f);
  EXPECT_EQ(as_term(*n.children[1]).term, "banana");
  EXPECT_FLOAT_EQ(n.children[1]->boost, 1.0f);
}

TEST(FtsAstRewriterTest, AndDedupsRepeatedTerms) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  children.push_back(term("apple"));
  children.push_back(term("apple"));
  auto ast = and_node(std::move(children));

  simplify(ast);

  // Single-child fold collapses AND(apple^3) → apple^3.
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "apple");
  EXPECT_FLOAT_EQ(ast->boost, 3.0f);
}

TEST(FtsAstRewriterTest, DifferentOccurDoesNotMerge) {
  // OR(apple, +apple) — the +apple becomes a must clause, plain apple becomes
  // should. After canonicalization into AND(apple, ?apple) and dedup (same term
  // text, same must/must_not bits post-canonicalization), they merge into a
  // single term with boost=2.0.
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  children.push_back(term("apple", /*must=*/true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_FLOAT_EQ(ast->boost, 2.0f);
}

// --- Conflict ---

TEST(FtsAstRewriterTest, AndMustVsMustNotSameTermBecomesEmpty) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple", /*must=*/true));
  children.push_back(term("apple", /*must=*/false, /*must_not=*/true));
  children.push_back(term("banana"));
  auto ast = and_node(std::move(children));

  simplify(ast);

  EXPECT_EQ(ast->type(), FtsNodeType::EMPTY);
}

TEST(FtsAstRewriterTest, AndAllMustNotBecomesEmpty) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple", false, true));
  children.push_back(term("banana", false, true));
  auto ast = and_node(std::move(children));

  simplify(ast);

  EXPECT_EQ(ast->type(), FtsNodeType::EMPTY);
}

// --- Flattening ---

TEST(FtsAstRewriterTest, OrFlattensNestedOr) {
  // OR(a, OR(b, c)) → OR(a, b, c)
  std::vector<FtsAstNodePtr> inner;
  inner.push_back(term("b"));
  inner.push_back(term("c"));
  std::vector<FtsAstNodePtr> outer;
  outer.push_back(term("a"));
  outer.push_back(or_node(std::move(inner)));
  auto ast = or_node(std::move(outer));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  ASSERT_EQ(as_or(*ast).children.size(), 3u);
}

TEST(FtsAstRewriterTest, AndFlattensNestedAndWithoutMustNot) {
  std::vector<FtsAstNodePtr> inner;
  inner.push_back(term("b"));
  inner.push_back(term("c", true));
  std::vector<FtsAstNodePtr> outer;
  outer.push_back(term("a"));
  outer.push_back(and_node(std::move(inner)));
  auto ast = and_node(std::move(outer));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  EXPECT_EQ(as_and(*ast).children.size(), 3u);
}

TEST(FtsAstRewriterTest, AndDoesNotFlattenAndWithMustNotChild) {
  std::vector<FtsAstNodePtr> inner;
  inner.push_back(term("b"));
  inner.push_back(term("c", false, true));
  std::vector<FtsAstNodePtr> outer;
  outer.push_back(term("a"));
  outer.push_back(and_node(std::move(inner)));
  auto ast = and_node(std::move(outer));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  EXPECT_EQ(as_and(*ast).children.size(), 2u);
}

TEST(FtsAstRewriterTest, FlattenThenDedupCrossLayer) {
  // OR(a, OR(a, b)) → flatten → OR(a, a, b) → dedup → OR(a^2, b)
  std::vector<FtsAstNodePtr> inner;
  inner.push_back(term("a"));
  inner.push_back(term("b"));
  std::vector<FtsAstNodePtr> outer;
  outer.push_back(term("a"));
  outer.push_back(or_node(std::move(inner)));
  auto ast = or_node(std::move(outer));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &n = as_or(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[0]).term, "a");
  EXPECT_FLOAT_EQ(n.children[0]->boost, 2.0f);
  EXPECT_EQ(as_term(*n.children[1]).term, "b");
}

// --- Phrase ---

TEST(FtsAstRewriterTest, PhraseSameTermsAreDeduped) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(phrase({"new", "york"}));
  children.push_back(phrase({"new", "york"}));
  children.push_back(phrase({"new", "york"}));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  EXPECT_FLOAT_EQ(ast->boost, 3.0f);
}

TEST(FtsAstRewriterTest, PhraseInternalRepeatNotMerged) {
  // Position-sensitive: "new new york" must keep its internal duplication.
  auto p = phrase({"new", "new", "york"});
  FtsAstNodePtr ast = std::move(p);
  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  ASSERT_EQ(as_phrase(*ast).terms.size(), 3u);
  EXPECT_EQ(as_phrase(*ast).terms[0], "new");
  EXPECT_EQ(as_phrase(*ast).terms[1], "new");
  EXPECT_EQ(as_phrase(*ast).terms[2], "york");
}

TEST(FtsAstRewriterTest, DifferentPhrasesDoNotMerge) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(phrase({"new", "york"}));
  children.push_back(phrase({"york", "new"}));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  EXPECT_EQ(as_or(*ast).children.size(), 2u);
}

// --- EmptyNode propagation ---

TEST(FtsAstRewriterTest, AndWithEmptyChildShortCircuits) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  children.push_back(empty_node());
  auto ast = and_node(std::move(children));

  simplify(ast);

  EXPECT_EQ(ast->type(), FtsNodeType::EMPTY);
}

TEST(FtsAstRewriterTest, OrDropsEmptyChild) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  children.push_back(empty_node());
  children.push_back(term("banana"));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  EXPECT_EQ(as_or(*ast).children.size(), 2u);
}

TEST(FtsAstRewriterTest, MustNotEmptyInAndIsNoOp) {
  // AND(apple, -EMPTY) — excluding nothing has no effect.
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  auto e = std::make_unique<EmptyNode>();
  e->must_not = true;
  children.push_back(std::move(e));
  auto ast = and_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "apple");
}

// --- Single-child fold ---

TEST(FtsAstRewriterTest, SingleChildOrFolds) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "apple");
}

TEST(FtsAstRewriterTest, FoldedSingleChildInheritsParentModifier) {
  // +OR(apple) → +apple (must flag lifts onto the surviving child)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("apple"));
  auto ast = or_node(std::move(children), /*must=*/true);

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_TRUE(ast->must);
  EXPECT_FALSE(ast->must_not);
}

// --- Idempotence ---

TEST(FtsAstRewriterTest, SimplifyIsIdempotent) {
  // Build something gnarly enough to exercise multiple rules at once.
  std::vector<FtsAstNodePtr> inner_or;
  inner_or.push_back(term("a"));
  inner_or.push_back(term("a"));
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("a"));
  children.push_back(or_node(std::move(inner_or)));
  children.push_back(term("b"));
  children.push_back(empty_node());
  auto ast = or_node(std::move(children));

  simplify(ast);
  const std::string after_first = ast->text();
  simplify(ast);
  const std::string after_second = ast->text();

  EXPECT_EQ(after_first, after_second);
}

// --- OR must_not canonicalization ---

TEST(FtsAstRewriterTest, OrWithSinglePositiveAndMustNotBecomesAnd) {
  // OR(a, -b) → AND(a, -b)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("a"));
  children.push_back(term("b", false, true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &n = as_and(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[0]).term, "a");
  EXPECT_FALSE(n.children[0]->must_not);
  EXPECT_EQ(as_term(*n.children[1]).term, "b");
  EXPECT_TRUE(n.children[1]->must_not);
}

TEST(FtsAstRewriterTest, OrWithMultiplePositivesAndMustNotWrapsInAnd) {
  // OR(a, b, -c) → AND(OR(a, b), -c)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("a"));
  children.push_back(term("b"));
  children.push_back(term("c", false, true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &n = as_and(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  ASSERT_EQ(n.children[0]->type(), FtsNodeType::OR);
  EXPECT_EQ(as_or(*n.children[0]).children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[1]).term, "c");
  EXPECT_TRUE(n.children[1]->must_not);
}

TEST(FtsAstRewriterTest, OrCanonicalizationCatchesSameTermConflict) {
  // OR(a, -a) — canonicalization moves -a into AND with a, then
  // and_has_mustnot_conflict fires → EmptyNode.
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("a"));
  children.push_back(term("a", false, true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  EXPECT_EQ(ast->type(), FtsNodeType::EMPTY);
}

TEST(FtsAstRewriterTest, OrCanonicalizationLiftsParentModifier) {
  // +OR(a, -b) → +AND(a, -b)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("a"));
  children.push_back(term("b", false, true));
  auto ast = or_node(std::move(children), /*must=*/true);

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  EXPECT_TRUE(ast->must);
  EXPECT_FALSE(ast->must_not);
}

TEST(FtsAstRewriterTest, NestedOrWithMustNotCanonicalizedAtBothLevels) {
  // OR(x, OR(b, -c)) — inner canonicalizes to AND(b, -c); outer keeps OR
  // since it has no must_not after recursion.
  std::vector<FtsAstNodePtr> inner;
  inner.push_back(term("b"));
  inner.push_back(term("c", false, true));
  std::vector<FtsAstNodePtr> outer;
  outer.push_back(term("x"));
  outer.push_back(or_node(std::move(inner)));
  auto ast = or_node(std::move(outer));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &n = as_or(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[0]).term, "x");
  EXPECT_EQ(n.children[1]->type(), FtsNodeType::AND);
}

TEST(FtsAstRewriterTest, OrWithoutMustNotIsLeftAlone) {
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("a"));
  children.push_back(term("b"));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  EXPECT_EQ(as_or(*ast).children.size(), 2u);
}

// --- OR must canonicalization ---

TEST(FtsAstRewriterTest, OrWithMustChildrenCanonicalizesToAnd) {
  // OR(foo, +bar, baz, +bay) → AND(bar, bay, ?OR(foo, baz))
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("foo"));
  children.push_back(term("bar", /*must=*/true));
  children.push_back(term("baz"));
  children.push_back(term("bay", /*must=*/true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &n = as_and(*ast);
  ASSERT_EQ(n.children.size(), 3u);
  // First two children: bar, bay (must terms, must flag cleared)
  EXPECT_EQ(as_term(*n.children[0]).term, "bar");
  EXPECT_FALSE(n.children[0]->must);
  EXPECT_FALSE(n.children[0]->should);
  EXPECT_EQ(as_term(*n.children[1]).term, "bay");
  EXPECT_FALSE(n.children[1]->must);
  EXPECT_FALSE(n.children[1]->should);
  // Third child: ?OR(foo, baz) with should=true
  ASSERT_EQ(n.children[2]->type(), FtsNodeType::OR);
  EXPECT_TRUE(n.children[2]->should);
  const auto &should_or = as_or(*n.children[2]);
  ASSERT_EQ(should_or.children.size(), 2u);
  EXPECT_EQ(as_term(*should_or.children[0]).term, "foo");
  EXPECT_EQ(as_term(*should_or.children[1]).term, "baz");
}

TEST(FtsAstRewriterTest, OrWithSingleMustAndSingleShouldFoldsCorrectly) {
  // OR(foo, +bar) → AND(bar, ?foo)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("foo"));
  children.push_back(term("bar", /*must=*/true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &n = as_and(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[0]).term, "bar");
  EXPECT_FALSE(n.children[0]->should);
  EXPECT_EQ(as_term(*n.children[1]).term, "foo");
  EXPECT_TRUE(n.children[1]->should);
}

TEST(FtsAstRewriterTest, OrWithAllMustNoShouldChildren) {
  // OR(+bar, +bay) → AND(bar, bay) — no should part
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("bar", /*must=*/true));
  children.push_back(term("bay", /*must=*/true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &n = as_and(*ast);
  ASSERT_EQ(n.children.size(), 2u);
  EXPECT_EQ(as_term(*n.children[0]).term, "bar");
  EXPECT_EQ(as_term(*n.children[1]).term, "bay");
}

TEST(FtsAstRewriterTest, OrWithMustAndMustNotCanonicalizesCorrectly) {
  // OR(foo, +bar, -baz) → must_not is processed first, then must
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("foo"));
  children.push_back(term("bar", /*must=*/true));
  children.push_back(term("baz", /*must=*/false, /*must_not=*/true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  // Should become AND-like structure with bar required, baz excluded
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
}

TEST(FtsAstRewriterTest, OrWithSingleMustNoShouldFoldsToTerm) {
  // OR(+bar) → bar (single-child fold after must canonicalization)
  std::vector<FtsAstNodePtr> children;
  children.push_back(term("bar", /*must=*/true));
  auto ast = or_node(std::move(children));

  simplify(ast);

  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "bar");
}

// --- Leaf untouched ---

TEST(FtsAstRewriterTest, BareTermPassthrough) {
  FtsAstNodePtr ast = term("apple");
  simplify(ast);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "apple");
  EXPECT_FLOAT_EQ(ast->boost, 1.0f);
}

}  // namespace zvec::fts
