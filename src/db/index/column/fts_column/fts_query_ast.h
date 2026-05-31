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

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace zvec::fts {

/*! AST node type enumeration
 */
enum class FtsNodeType {
  TERM,    // Term node, e.g., "vector"
  PHRASE,  // Phrase node, e.g., "\"exact phrase\""
  AND,     // AND combination node (intersection)
  OR,      // OR combination node (union)
  EMPTY,   // Matches zero documents (analogous to Lucene MatchNoDocsQuery).
};

/*! AST node base class
 *  All FTS AST nodes carry must/must_not modifiers so that the +/- prefix
 *  (and AND NOT semantics) can be applied uniformly to terms, phrases and
 *  composite (AND/OR) sub-expressions.
 */
struct FtsAstNode {
  bool must{false};      // Prefix + means must
  bool must_not{false};  // Prefix - / right-hand side of AND NOT means must_not
  bool should{
      false};  // SHOULD semantics: does not affect matching, only scoring
  // Per-node scoring weight. Currently meaningful only on TermNode / PhraseNode
  // (composite nodes inherit boost from their scored leaves). Repeated terms in
  // a sibling list are collapsed by the AST rewriter into a single node whose
  // boost is the linear sum of duplicates, so that the post-rewrite score
  // matches the pre-rewrite "sum of independent scorers" semantics exactly.
  float boost{1.0f};

  virtual ~FtsAstNode() = default;
  virtual FtsNodeType type() const = 0;

  // Return a human-readable text representation for debugging / logging
  virtual std::string text() const = 0;

 protected:
  // Helper: prepend +/-/? modifier prefix
  std::string modifier_prefix() const {
    if (must) {
      return "+";
    }
    if (must_not) {
      return "-";
    }
    if (should) {
      return "?";
    }
    return "";
  }

  // Helper: append ^X boost suffix when boost differs from default 1.0
  std::string boost_suffix() const {
    if (std::fabs(boost - 1.0f) < 1e-6f) {
      return "";
    }
    return "^" + std::to_string(boost);
  }
};

using FtsAstNodePtr = std::unique_ptr<FtsAstNode>;

/*! Term node
 *  Represents a single query term, can have must (+) or must_not (-) modifiers
 *  inherited from FtsAstNode.
 */
struct TermNode : public FtsAstNode {
  std::string term;

  explicit TermNode(std::string term_text, bool is_must = false,
                    bool is_must_not = false)
      : term(std::move(term_text)) {
    must = is_must;
    must_not = is_must_not;
  }

  FtsNodeType type() const override {
    return FtsNodeType::TERM;
  }

  std::string text() const override {
    return modifier_prefix() + term + boost_suffix();
  }
};

/*! Phrase node
 *  Represents an exact phrase query, e.g., "exact phrase"
 *  Requires exact match of word order and adjacent positions
 */
struct PhraseNode : public FtsAstNode {
  std::vector<std::string> terms;  // Individual words in the phrase

  FtsNodeType type() const override {
    return FtsNodeType::PHRASE;
  }

  std::string text() const override {
    std::string result = modifier_prefix() + "\"";
    for (size_t i = 0; i < terms.size(); ++i) {
      if (i > 0) {
        result += " ";
      }
      result += terms[i];
    }
    result += "\"";
    result += boost_suffix();
    return result;
  }
};

/*! Match-nothing node — used when the analyzer drops every term (e.g.
 *  pure punctuation or all stop-words). Composes naturally with AND/OR so
 *  callers don't have to special-case nullptr.
 */
struct EmptyNode : public FtsAstNode {
  FtsNodeType type() const override {
    return FtsNodeType::EMPTY;
  }

  std::string text() const override {
    return modifier_prefix() + "<empty>";
  }
};

/*! AND combination node
 *  All child nodes must match (intersection semantics)
 */
struct AndNode : public FtsAstNode {
  std::vector<FtsAstNodePtr> children;

  FtsNodeType type() const override {
    return FtsNodeType::AND;
  }

  std::string text() const override {
    std::string result = modifier_prefix() + "AND(";
    for (size_t i = 0; i < children.size(); ++i) {
      if (i > 0) {
        result += " ";
      }
      result += children[i]->text();
    }
    result += ")";
    return result;
  }
};

/*! OR combination node
 *  Any child node matches (union semantics)
 */
struct OrNode : public FtsAstNode {
  std::vector<FtsAstNodePtr> children;

  FtsNodeType type() const override {
    return FtsNodeType::OR;
  }

  std::string text() const override {
    std::string result = modifier_prefix() + "OR(";
    for (size_t i = 0; i < children.size(); ++i) {
      if (i > 0) {
        result += " ";
      }
      result += children[i]->text();
    }
    result += ")";
    return result;
  }
};

}  // namespace zvec::fts
