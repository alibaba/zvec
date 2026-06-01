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

#include <gtest/gtest.h>
#include "db/index/column/fts_column/fts_query_ast.h"
#include "db/index/column/fts_column/fts_types.h"
#include "db/index/column/fts_column/parser/fts_query_parser.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"

namespace zvec::fts {

// ============================================================
// Test fixture
// ============================================================

class FtsParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Standard tokenizer + lowercase filter: ASCII tests behave the same as
    // the previous whitespace split (alnum runs become tokens, delimiters
    // get dropped) while CJK tests can exercise the per-character tokens
    // standard produces from non-alnum bytes.
    FtsIndexParams params;
    params.tokenizer_name = "standard";
    params.filters = {"lowercase"};
    pipeline_ = TokenizerFactory::create(params);
    ASSERT_NE(pipeline_, nullptr);
  }

  FtsAstNodePtr parse(const std::string &query) {
    return parser_.parse(query, pipeline_);
  }

  // Overload for tests that need to specify the default operator explicitly.
  FtsAstNodePtr parse(const std::string &query, FtsDefaultOperator default_op) {
    return parser_.parse(query, pipeline_, default_op);
  }

  const std::string &err_msg() {
    return parser_.err_msg();
  }

  // Helpers for type-safe downcasting
  static const TermNode &as_term(const FtsAstNode &node) {
    EXPECT_EQ(node.type(), FtsNodeType::TERM);
    return static_cast<const TermNode &>(node);
  }

  static const PhraseNode &as_phrase(const FtsAstNode &node) {
    EXPECT_EQ(node.type(), FtsNodeType::PHRASE);
    return static_cast<const PhraseNode &>(node);
  }

  static const AndNode &as_and(const FtsAstNode &node) {
    EXPECT_EQ(node.type(), FtsNodeType::AND);
    return static_cast<const AndNode &>(node);
  }

  static const OrNode &as_or(const FtsAstNode &node) {
    EXPECT_EQ(node.type(), FtsNodeType::OR);
    return static_cast<const OrNode &>(node);
  }

 private:
  FtsQueryParser parser_;
  TokenizerPipelinePtr pipeline_;
};

// ============================================================
// Single term
// ============================================================

TEST_F(FtsParserTest, SingleTerm) {
  auto ast = parse("vector");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  const auto &term = as_term(*ast);
  EXPECT_EQ(term.term, "vector");
  EXPECT_FALSE(term.must);
  EXPECT_FALSE(term.must_not);
}

TEST_F(FtsParserTest, SingleTermNumeric) {
  auto ast = parse("2024");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "2024");
}

TEST_F(FtsParserTest, SingleTermWithHyphen) {
  // The lexer's REGULAR_ID rule keeps hyphenated text as one token, but the
  // standard tokenizer on the parser side splits non-alphanumerics. With the
  // default OR operator the term decomposes into Or[full, text] so query
  // segmentation matches the index segmentation.
  auto ast = parse("full-text");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "full");
  EXPECT_EQ(as_term(*or_node.children[1]).term, "text");
}

// ============================================================
// Must (+) and must_not (-/NOT) modifiers
// ============================================================

TEST_F(FtsParserTest, MustModifier) {
  auto ast = parse("+vector");
  ASSERT_NE(ast, nullptr);
  const auto &term = as_term(*ast);
  EXPECT_EQ(term.term, "vector");
  EXPECT_TRUE(term.must);
  EXPECT_FALSE(term.must_not);
}

TEST_F(FtsParserTest, MustNotModifierMinus) {
  // "-slow" is lexed as a single REGULAR_ID token (hyphen is part of the id).
  // To express must_not, use a space: "- slow" -> MINUS_SIGN + REGULAR_ID.
  auto ast = parse("- slow");
  ASSERT_NE(ast, nullptr);
  const auto &term = as_term(*ast);
  EXPECT_EQ(term.term, "slow");
  EXPECT_FALSE(term.must);
  EXPECT_TRUE(term.must_not);
}

TEST_F(FtsParserTest, MustNotModifierMinusNoSpace) {
  // "-slow" without space: FtsLexer treats '-' as MINUS_SIGN modifier,
  // so "-slow" is parsed as must_not:slow (same as "- slow").
  auto ast = parse("-slow");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "slow");
  EXPECT_TRUE(as_term(*ast).must_not);
}

TEST_F(FtsParserTest, MustNotModifierNot) {
  // NOT is now a strict binary operator (`a NOT b` <=> `a AND NOT b`).
  // A leading `NOT a` is therefore a syntax error — there is no left-hand
  // operand for NOT to subtract from.
  auto ast = parse("NOT slow");
  EXPECT_EQ(ast, nullptr);
  EXPECT_FALSE(err_msg().empty());
}

// ============================================================
// Phrase query
// ============================================================

TEST_F(FtsParserTest, DoubleQuotedPhrase) {
  auto ast = parse("\"exact phrase\"");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  const auto &phrase = as_phrase(*ast);
  ASSERT_EQ(phrase.terms.size(), 2u);
  EXPECT_EQ(phrase.terms[0], "exact");
  EXPECT_EQ(phrase.terms[1], "phrase");
  EXPECT_FALSE(phrase.must);
  EXPECT_FALSE(phrase.must_not);
}

TEST_F(FtsParserTest, SingleQuotedPhrase) {
  // Single-quoted strings are not supported as phrase queries (no SQUOTA_STRING
  // token).  The lexer's TERM rule absorbs "'hello", "world", and "'" as
  // individual term tokens, so the query parses as an implicit OR of terms.
  auto ast = parse("'hello world'");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
}

TEST_F(FtsParserTest, PhraseWithMustModifier) {
  auto ast = parse("+\"exact phrase\"");
  ASSERT_NE(ast, nullptr);
  const auto &phrase = as_phrase(*ast);
  EXPECT_TRUE(phrase.must);
  EXPECT_FALSE(phrase.must_not);
}

TEST_F(FtsParserTest, PhraseWithMustNotModifier) {
  auto ast = parse("-\"bad phrase\"");
  ASSERT_NE(ast, nullptr);
  const auto &phrase = as_phrase(*ast);
  EXPECT_FALSE(phrase.must);
  EXPECT_TRUE(phrase.must_not);
}

TEST_F(FtsParserTest, PhraseWithThreeWords) {
  auto ast = parse("\"one two three\"");
  ASSERT_NE(ast, nullptr);
  const auto &phrase = as_phrase(*ast);
  ASSERT_EQ(phrase.terms.size(), 3u);
  EXPECT_EQ(phrase.terms[0], "one");
  EXPECT_EQ(phrase.terms[1], "two");
  EXPECT_EQ(phrase.terms[2], "three");
}

// ============================================================
// Explicit OR
// ============================================================

TEST_F(FtsParserTest, ExplicitOr) {
  auto ast = parse("cat OR dog");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "cat");
  EXPECT_EQ(as_term(*or_node.children[1]).term, "dog");
}

TEST_F(FtsParserTest, MultipleOr) {
  auto ast = parse("a OR b OR c");
  ASSERT_NE(ast, nullptr);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 3u);
}

// ============================================================
// Explicit AND
// ============================================================

TEST_F(FtsParserTest, ExplicitAnd) {
  auto ast = parse("cat AND dog");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(as_term(*and_node.children[0]).term, "cat");
  EXPECT_EQ(as_term(*and_node.children[1]).term, "dog");
}

TEST_F(FtsParserTest, MultipleAnd) {
  auto ast = parse("a AND b AND c");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 3u);
}

// ============================================================
// Operator precedence: AND binds tighter than OR
// ============================================================

TEST_F(FtsParserTest, AndBindsTighterThanOr) {
  // "a OR b AND c" should parse as "a OR (b AND c)"
  auto ast = parse("a OR b AND c");
  ASSERT_NE(ast, nullptr);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);

  // Left child: term "a"
  EXPECT_EQ(as_term(*or_node.children[0]).term, "a");

  // Right child: AND(b, c)
  const auto &and_node = as_and(*or_node.children[1]);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(as_term(*and_node.children[0]).term, "b");
  EXPECT_EQ(as_term(*and_node.children[1]).term, "c");
}

// ============================================================
// Implicit adjacency (seqExpr / default operator)
// ============================================================

TEST_F(FtsParserTest, ImplicitAdjacency) {
  // Adjacent terms without explicit operator: "a b" -> seqExpr -> OR(a, b)
  auto ast = parse("a b");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "a");
  EXPECT_EQ(as_term(*or_node.children[1]).term, "b");
}

TEST_F(FtsParserTest, ImplicitAdjacencyThreeTerms) {
  auto ast = parse("a b c");
  ASSERT_NE(ast, nullptr);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 3u);
}

TEST_F(FtsParserTest, ImplicitAdjacencyWithModifiers) {
  // "+a - b" -> seqExpr -> OR(must:a, must_not:b)
  // Note: "-b" (no space) is lexed as a single REGULAR_ID; use "- b" for
  // must_not.
  auto ast = parse("+a - b");
  ASSERT_NE(ast, nullptr);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_TRUE(as_term(*or_node.children[0]).must);
  EXPECT_TRUE(as_term(*or_node.children[1]).must_not);
}

// ============================================================
// Parentheses grouping
// ============================================================

TEST_F(FtsParserTest, Parentheses) {
  // "(a OR b) AND c"
  auto ast = parse("(a OR b) AND c");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);

  // Left: OR(a, b)
  const auto &or_node = as_or(*and_node.children[0]);
  ASSERT_EQ(or_node.children.size(), 2u);

  // Right: term c
  EXPECT_EQ(as_term(*and_node.children[1]).term, "c");
}

TEST_F(FtsParserTest, NestedParentheses) {
  auto ast = parse("((a OR b) AND c) OR d");
  ASSERT_NE(ast, nullptr);
  const auto &outer_or = as_or(*ast);
  ASSERT_EQ(outer_or.children.size(), 2u);
  EXPECT_EQ(as_term(*outer_or.children[1]).term, "d");
}

// ============================================================
// Mixed complex queries
// ============================================================

TEST_F(FtsParserTest, MixedTermAndPhrase) {
  // "+vector - slow \"exact phrase\""
  // Note: use "- slow" (with space) so MINUS_SIGN is a separate token.
  auto ast = parse("+vector - slow \"exact phrase\"");
  ASSERT_NE(ast, nullptr);
  // Four adjacent items -> seqExpr -> OR(must:vector, must_not:slow, phrase)
  // Actually: +vector and - slow and phrase are three unary nodes in seqExpr
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 3u);

  EXPECT_TRUE(as_term(*or_node.children[0]).must);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "vector");

  EXPECT_TRUE(as_term(*or_node.children[1]).must_not);
  EXPECT_EQ(as_term(*or_node.children[1]).term, "slow");

  EXPECT_EQ(or_node.children[2]->type(), FtsNodeType::PHRASE);
}

TEST_F(FtsParserTest, AndWithPhrase) {
  auto ast = parse("\"machine learning\" AND model");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(and_node.children[0]->type(), FtsNodeType::PHRASE);
  EXPECT_EQ(as_term(*and_node.children[1]).term, "model");
}

TEST_F(FtsParserTest, ComplexBooleanQuery) {
  // "a AND b OR c AND d" -> (a AND b) OR (c AND d)
  auto ast = parse("a AND b OR c AND d");
  ASSERT_NE(ast, nullptr);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);

  const auto &left_and = as_and(*or_node.children[0]);
  ASSERT_EQ(left_and.children.size(), 2u);

  const auto &right_and = as_and(*or_node.children[1]);
  ASSERT_EQ(right_and.children.size(), 2u);
}

// ============================================================
// Single-child simplification (no unnecessary wrapping)
// ============================================================

TEST_F(FtsParserTest, SingleChildNotWrapped) {
  // A single term should not be wrapped in an AndNode/OrNode
  auto ast = parse("hello");
  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->type(), FtsNodeType::TERM);
}

TEST_F(FtsParserTest, SinglePhraseNotWrapped) {
  auto ast = parse("\"hello world\"");
  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->type(), FtsNodeType::PHRASE);
}

// ============================================================
// Error cases
// ============================================================

TEST_F(FtsParserTest, EmptyQueryReturnsNull) {
  auto ast = parse("");
  EXPECT_EQ(ast, nullptr);
}

TEST_F(FtsParserTest, OnlyParenthesesReturnsNull) {
  auto ast = parse("()");
  EXPECT_EQ(ast, nullptr);
}

TEST_F(FtsParserTest, UnclosedPhraseParsesAsTerm) {
  // An unclosed double-quote causes the DQUOTA_STRING rule to fail.  The
  // remaining characters are absorbed by the TERM catch-all rule, so the
  // query parses as a single term rather than returning nullptr.
  auto ast = parse("\"unclosed phrase");
  ASSERT_NE(ast, nullptr);
}

TEST_F(FtsParserTest, UnclosedParenReturnsNull) {
  auto ast = parse("(a OR b");
  EXPECT_EQ(ast, nullptr);
}

// ============================================================
// Empty-AST cases: grammar valid, analyzer drops every term → EmptyNode.
// ============================================================

TEST_F(FtsParserTest, PunctuationOnlyReturnsEmpty) {
  auto ast = parse("!!!");
  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->type(), FtsNodeType::EMPTY);
  EXPECT_TRUE(err_msg().empty());
}

TEST_F(FtsParserTest, MultiplePunctuationTermsReturnsEmpty) {
  auto ast = parse("!!! ??? ...");
  ASSERT_NE(ast, nullptr);
  EXPECT_EQ(ast->type(), FtsNodeType::EMPTY);
  EXPECT_TRUE(err_msg().empty());
}

// ============================================================
// NOT as a binary AND-NOT operator
// ============================================================

TEST_F(FtsParserTest, NotAsBinaryAndNot) {
  // `foo NOT bar` <=> `foo AND NOT bar` -> And[foo, bar(must_not)]
  auto ast = parse("foo NOT bar");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);

  EXPECT_EQ(as_term(*and_node.children[0]).term, "foo");
  EXPECT_FALSE(and_node.children[0]->must_not);

  EXPECT_EQ(as_term(*and_node.children[1]).term, "bar");
  EXPECT_TRUE(and_node.children[1]->must_not);
}

TEST_F(FtsParserTest, AndAndNot) {
  // `a AND NOT b` -> And[a, b(must_not)]
  auto ast = parse("a AND NOT b");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(as_term(*and_node.children[0]).term, "a");
  EXPECT_FALSE(and_node.children[0]->must_not);
  EXPECT_EQ(as_term(*and_node.children[1]).term, "b");
  EXPECT_TRUE(and_node.children[1]->must_not);
}

TEST_F(FtsParserTest, OrThenNot) {
  // Precedence check: NOT shares AND's precedence (higher than OR).
  // `a OR b NOT c` -> Or[a, And[b, c(must_not)]]
  auto ast = parse("a OR b NOT c");
  ASSERT_NE(ast, nullptr);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);

  EXPECT_EQ(as_term(*or_node.children[0]).term, "a");

  const auto &right_and = as_and(*or_node.children[1]);
  ASSERT_EQ(right_and.children.size(), 2u);
  EXPECT_EQ(as_term(*right_and.children[0]).term, "b");
  EXPECT_FALSE(right_and.children[0]->must_not);
  EXPECT_EQ(as_term(*right_and.children[1]).term, "c");
  EXPECT_TRUE(right_and.children[1]->must_not);
}

TEST_F(FtsParserTest, NotWithGroup) {
  // `a NOT (b OR c)` -> And[a, Or[b, c](must_not)]
  auto ast = parse("a NOT (b OR c)");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);

  EXPECT_EQ(as_term(*and_node.children[0]).term, "a");
  EXPECT_FALSE(and_node.children[0]->must_not);

  ASSERT_EQ(and_node.children[1]->type(), FtsNodeType::OR);
  EXPECT_TRUE(and_node.children[1]->must_not);
  const auto &grouped_or = as_or(*and_node.children[1]);
  ASSERT_EQ(grouped_or.children.size(), 2u);
  EXPECT_EQ(as_term(*grouped_or.children[0]).term, "b");
  EXPECT_EQ(as_term(*grouped_or.children[1]).term, "c");
}

TEST_F(FtsParserTest, LeadingNotIsError) {
  // Leading NOT has no left-hand operand and must fail to parse.
  auto ast = parse("NOT a");
  EXPECT_EQ(ast, nullptr);
  EXPECT_FALSE(err_msg().empty());
}

TEST_F(FtsParserTest, MultipleNotsAndAnds) {
  // `a AND b NOT c AND d NOT e` -> And[a, b, c(must_not), d, e(must_not)]
  auto ast = parse("a AND b NOT c AND d NOT e");
  ASSERT_NE(ast, nullptr);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 5u);

  EXPECT_EQ(as_term(*and_node.children[0]).term, "a");
  EXPECT_FALSE(and_node.children[0]->must_not);

  EXPECT_EQ(as_term(*and_node.children[1]).term, "b");
  EXPECT_FALSE(and_node.children[1]->must_not);

  EXPECT_EQ(as_term(*and_node.children[2]).term, "c");
  EXPECT_TRUE(and_node.children[2]->must_not);

  EXPECT_EQ(as_term(*and_node.children[3]).term, "d");
  EXPECT_FALSE(and_node.children[3]->must_not);

  EXPECT_EQ(as_term(*and_node.children[4]).term, "e");
  EXPECT_TRUE(and_node.children[4]->must_not);
}

// ============================================================
// +/- modifiers on parenthesised sub-expressions
// ============================================================

TEST_F(FtsParserTest, MustOnGroup) {
  // `+(a OR b)` -> Or[a, b]{must=true}
  auto ast = parse("+(a OR b)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  EXPECT_TRUE(ast->must);
  EXPECT_FALSE(ast->must_not);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "a");
  EXPECT_EQ(as_term(*or_node.children[1]).term, "b");
}

TEST_F(FtsParserTest, MustNotOnGroup) {
  // `-(a AND b)` -> And[a, b]{must_not=true}
  auto ast = parse("-(a AND b)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  EXPECT_FALSE(ast->must);
  EXPECT_TRUE(ast->must_not);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(as_term(*and_node.children[0]).term, "a");
  EXPECT_EQ(as_term(*and_node.children[1]).term, "b");
}

TEST_F(FtsParserTest, MustGroupAndOther) {
  // `+(a OR b) c` -> implicit-OR collapses three siblings into a single
  // OrNode: Or[Or[a, b]{must=true}, c]
  // (the inner OR keeps its must flag; implicit adjacency is still OR.)
  auto ast = parse("+(a OR b) c");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &outer_or = as_or(*ast);
  ASSERT_EQ(outer_or.children.size(), 2u);

  ASSERT_EQ(outer_or.children[0]->type(), FtsNodeType::OR);
  EXPECT_TRUE(outer_or.children[0]->must);
  const auto &inner_or = as_or(*outer_or.children[0]);
  ASSERT_EQ(inner_or.children.size(), 2u);
  EXPECT_EQ(as_term(*inner_or.children[0]).term, "a");
  EXPECT_EQ(as_term(*inner_or.children[1]).term, "b");

  EXPECT_EQ(as_term(*outer_or.children[1]).term, "c");
}

TEST_F(FtsParserTest, NestedGroupModifier) {
  // `+((a AND b) OR c)` -> the must flag attaches to the outermost OrNode.
  auto ast = parse("+((a AND b) OR c)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  EXPECT_TRUE(ast->must);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);

  ASSERT_EQ(or_node.children[0]->type(), FtsNodeType::AND);
  EXPECT_FALSE(or_node.children[0]->must);  // inner AND not affected
  const auto &inner_and = as_and(*or_node.children[0]);
  ASSERT_EQ(inner_and.children.size(), 2u);
  EXPECT_EQ(as_term(*inner_and.children[0]).term, "a");
  EXPECT_EQ(as_term(*inner_and.children[1]).term, "b");

  EXPECT_EQ(as_term(*or_node.children[1]).term, "c");
}

// ============================================================
// Default operator (FtsDefaultOperator::OR / AND)
// Only adjacent bare terms (no explicit operator) are affected; explicit
// AND / OR / + / - usages keep their original semantics.
// ============================================================

TEST_F(FtsParserTest, DefaultOperatorOr_AdjacentBareTerms) {
  // Backward-compat: omitting default_op or passing OR yields the original
  // implicit-OR behaviour for adjacent bare terms.
  auto ast = parse("vector database", FtsDefaultOperator::OR);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "vector");
  EXPECT_EQ(as_term(*or_node.children[1]).term, "database");
}

TEST_F(FtsParserTest, DefaultOperatorAnd_AdjacentBareTerms) {
  // With AND default, two adjacent bare terms collapse into an AndNode.
  auto ast = parse("vector database", FtsDefaultOperator::AND);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(as_term(*and_node.children[0]).term, "vector");
  EXPECT_EQ(as_term(*and_node.children[1]).term, "database");
}

TEST_F(FtsParserTest, DefaultOperatorAnd_SingleTermUnchanged) {
  // A single term should not be wrapped in an AndNode.
  auto ast = parse("vector", FtsDefaultOperator::AND);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "vector");
}

TEST_F(FtsParserTest, DefaultOperatorAnd_PropagatesIntoParens) {
  // Parenthesised sub-expressions inherit the same default operator.
  // `(a b) c` with AND default -> And[And[a, b], c].
  auto ast = parse("(a b) c", FtsDefaultOperator::AND);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &outer_and = as_and(*ast);
  ASSERT_EQ(outer_and.children.size(), 2u);

  ASSERT_EQ(outer_and.children[0]->type(), FtsNodeType::AND);
  const auto &inner_and = as_and(*outer_and.children[0]);
  ASSERT_EQ(inner_and.children.size(), 2u);
  EXPECT_EQ(as_term(*inner_and.children[0]).term, "a");
  EXPECT_EQ(as_term(*inner_and.children[1]).term, "b");

  EXPECT_EQ(as_term(*outer_and.children[1]).term, "c");
}

TEST_F(FtsParserTest, DefaultOperatorAnd_DoesNotOverrideExplicitOr) {
  // Explicit OR has higher-level structure; default_op only changes the
  // implicit adjacency inside each seqExpr.
  // `a OR b c` with AND default -> Or[a, And[b, c]].
  auto ast = parse("a OR b c", FtsDefaultOperator::AND);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);

  EXPECT_EQ(as_term(*or_node.children[0]).term, "a");

  ASSERT_EQ(or_node.children[1]->type(), FtsNodeType::AND);
  const auto &inner_and = as_and(*or_node.children[1]);
  ASSERT_EQ(inner_and.children.size(), 2u);
  EXPECT_EQ(as_term(*inner_and.children[0]).term, "b");
  EXPECT_EQ(as_term(*inner_and.children[1]).term, "c");
}

TEST_F(FtsParserTest, DefaultOperatorOr_DoesNotOverrideExplicitAnd) {
  // Grammar: andExpr = seqExpr ((AND|NOT) seqExpr)*
  // `a AND b c` parses as seqExpr("a") AND seqExpr("b c").
  // With OR default, seqExpr("b c") -> Or[b, c].
  // Result: And[a, Or[b, c]].
  auto ast = parse("a AND b c", FtsDefaultOperator::OR);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);

  EXPECT_EQ(as_term(*and_node.children[0]).term, "a");

  ASSERT_EQ(and_node.children[1]->type(), FtsNodeType::OR);
  const auto &inner_or = as_or(*and_node.children[1]);
  ASSERT_EQ(inner_or.children.size(), 2u);
  EXPECT_EQ(as_term(*inner_or.children[0]).term, "b");
  EXPECT_EQ(as_term(*inner_or.children[1]).term, "c");
}

TEST_F(FtsParserTest, DefaultOperatorAnd_PreservesPlusMinusModifiers) {
  // `+a b -c` with AND default -> And[a{must}, b, c{must_not}].
  // Modifiers on individual terms are independent of default_op.
  auto ast = parse("+a b -c", FtsDefaultOperator::AND);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 3u);

  const auto &t0 = as_term(*and_node.children[0]);
  EXPECT_EQ(t0.term, "a");
  EXPECT_TRUE(t0.must);
  EXPECT_FALSE(t0.must_not);

  const auto &t1 = as_term(*and_node.children[1]);
  EXPECT_EQ(t1.term, "b");
  EXPECT_FALSE(t1.must);
  EXPECT_FALSE(t1.must_not);

  const auto &t2 = as_term(*and_node.children[2]);
  EXPECT_EQ(t2.term, "c");
  EXPECT_FALSE(t2.must);
  EXPECT_TRUE(t2.must_not);
}

// ============================================================
// Pipeline-aware tokenization (phrase / bare term split through pipeline)
// ============================================================

TEST_F(FtsParserTest, MultiTokenBareTermAndDefaultGroupsAsAnd) {
  // `full-text` lexes as one REGULAR_ID, but standard splits it into
  // ["full", "text"]. With AND default operator the two tokens combine into
  // an AndNode rather than the OR returned by the OR-default test above.
  auto ast = parse("full-text", FtsDefaultOperator::AND);
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::AND);
  const auto &and_node = as_and(*ast);
  ASSERT_EQ(and_node.children.size(), 2u);
  EXPECT_EQ(as_term(*and_node.children[0]).term, "full");
  EXPECT_EQ(as_term(*and_node.children[1]).term, "text");
}

TEST_F(FtsParserTest, MultiTokenBareTermPreservesMustModifier) {
  // `+full-text` -> Or[full, text] with must=true on the composite root.
  auto ast = parse("+full-text");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::OR);
  EXPECT_TRUE(ast->must);
  EXPECT_FALSE(ast->must_not);
  const auto &or_node = as_or(*ast);
  ASSERT_EQ(or_node.children.size(), 2u);
  EXPECT_EQ(as_term(*or_node.children[0]).term, "full");
  EXPECT_EQ(as_term(*or_node.children[1]).term, "text");
}

TEST_F(FtsParserTest, PhraseTokensRunThroughPipeline) {
  // The phrase body is tokenized exactly like document text. With the
  // standard tokenizer, mixed delimiters between alnum runs collapse so
  // "machine, learning!" becomes ["machine", "learning"].
  auto ast = parse("\"machine, learning!\"");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  const auto &phrase = as_phrase(*ast);
  ASSERT_EQ(phrase.terms.size(), 2u);
  EXPECT_EQ(phrase.terms[0], "machine");
  EXPECT_EQ(phrase.terms[1], "learning");
}

TEST_F(FtsParserTest, PhraseLowercaseFilterApplies) {
  // The lowercase filter is part of the pipeline so phrase tokens come back
  // lowercased even when the input mixed case.
  auto ast = parse("\"Machine LEARNING\"");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  const auto &phrase = as_phrase(*ast);
  ASSERT_EQ(phrase.terms.size(), 2u);
  EXPECT_EQ(phrase.terms[0], "machine");
  EXPECT_EQ(phrase.terms[1], "learning");
}

TEST_F(FtsParserTest, AllPunctuationPhraseYieldsEmptyTerms) {
  // Pure non-alnum content is filtered out entirely. The phrase node still
  // exists but carries zero terms; the search engine treats this as
  // "match nothing" without crashing.
  auto ast = parse("\"!!! ???\"");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  EXPECT_TRUE(as_phrase(*ast).terms.empty());
}

// ============================================================
// Unescape: backslash removal for TERM and PHRASE paths.
// Uses WhitespaceTokenizer (no filter) so that special characters are
// preserved in tokens — this lets us observe whether unescape() actually
// stripped the backslashes.
// ============================================================

class FtsParserUnescapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FtsIndexParams params;
    params.tokenizer_name = "whitespace";
    params.filters = {};
    pipeline_ = TokenizerFactory::create(params);
    ASSERT_NE(pipeline_, nullptr);
  }

  FtsAstNodePtr parse(const std::string &query) {
    return parser_.parse(query, pipeline_);
  }

  static const TermNode &as_term(const FtsAstNode &node) {
    EXPECT_EQ(node.type(), FtsNodeType::TERM);
    return static_cast<const TermNode &>(node);
  }

  static const PhraseNode &as_phrase(const FtsAstNode &node) {
    EXPECT_EQ(node.type(), FtsNodeType::PHRASE);
    return static_cast<const PhraseNode &>(node);
  }

 private:
  FtsQueryParser parser_;
  TokenizerPipelinePtr pipeline_;
};

TEST_F(FtsParserUnescapeTest, TermEscapedPlusBecomesLiteralPlus) {
  // Lexer token: C\+\+ (with backslashes). After unescape: C++.
  // WhitespaceTokenizer preserves the '+' in the token text.
  auto ast = parse(R"(C\+\+)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "C++");
}

TEST_F(FtsParserUnescapeTest, TermEscapedMinusBecomesLiteralMinus) {
  // "a\-b" after unescape → "a-b" kept intact by whitespace tokenizer.
  auto ast = parse(R"(a\-b)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "a-b");
}

TEST_F(FtsParserUnescapeTest, TermEscapedBackslashBecomesLiteralBackslash) {
  // "path\\dir" — lexer sees ESCAPED_CHAR(\\), unescape yields "path\dir".
  auto ast = parse(R"(path\\dir)");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::TERM);
  EXPECT_EQ(as_term(*ast).term, "path\\dir");
}

TEST_F(FtsParserUnescapeTest, PhraseEscapedQuoteBecomesLiteralQuote) {
  // Phrase: "hello \"world\"" — after strip_quotes + unescape:
  // 'hello "world"' — whitespace tokenizer splits on space to:
  // ["hello", "\"world\""]
  auto ast = parse(R"("hello \"world\"")");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  const auto &phrase = as_phrase(*ast);
  ASSERT_EQ(phrase.terms.size(), 2u);
  EXPECT_EQ(phrase.terms[0], "hello");
  EXPECT_EQ(phrase.terms[1], "\"world\"");
}

TEST_F(FtsParserUnescapeTest, PhraseEscapedBackslashBecomesLiteral) {
  // Phrase: "a\\b" — after strip+unescape: "a\b" (one backslash, no space),
  // whitespace tokenizer keeps it as single token.
  auto ast = parse(R"("a\\b")");
  ASSERT_NE(ast, nullptr);
  ASSERT_EQ(ast->type(), FtsNodeType::PHRASE);
  const auto &phrase = as_phrase(*ast);
  ASSERT_EQ(phrase.terms.size(), 1u);
  EXPECT_EQ(phrase.terms[0], "a\\b");
}

}  // namespace zvec::fts
