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

#include "fts_ast_rewriter.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace zvec::fts {

namespace {

// Two AST nodes are dedup-equivalent when they are the same leaf kind and
// carry identical modifiers and identical scoring key (term string for
// TermNode, terms vector for PhraseNode). Boost is intentionally NOT part of
// the key — it is what we accumulate during dedup.
bool same_dedup_key(const FtsAstNode &a, const FtsAstNode &b) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.must != b.must || a.must_not != b.must_not) {
    return false;
  }
  if (a.type() == FtsNodeType::TERM) {
    return static_cast<const TermNode &>(a).term ==
           static_cast<const TermNode &>(b).term;
  }
  if (a.type() == FtsNodeType::PHRASE) {
    return static_cast<const PhraseNode &>(a).terms ==
           static_cast<const PhraseNode &>(b).terms;
  }
  return false;
}

// Same scoring key as same_dedup_key but ignores modifiers — used to detect
// `+apple -apple` style conflicts inside an AND node.
bool same_term_or_phrase_text(const FtsAstNode &a, const FtsAstNode &b) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.type() == FtsNodeType::TERM) {
    return static_cast<const TermNode &>(a).term ==
           static_cast<const TermNode &>(b).term;
  }
  if (a.type() == FtsNodeType::PHRASE) {
    return static_cast<const PhraseNode &>(a).terms ==
           static_cast<const PhraseNode &>(b).terms;
  }
  return false;
}

// Collapse adjacent duplicates (TermNode/PhraseNode siblings sharing the same
// dedup key) into a single node whose boost is the linear sum. O(K^2) — K is
// the sibling count, typically small enough that a hash map would cost more in
// allocations than it would save in comparisons.
void merge_duplicate_siblings(std::vector<FtsAstNodePtr> &children) {
  for (size_t i = 0; i < children.size(); ++i) {
    auto &a = children[i];
    if (!a) {
      continue;
    }
    if (a->type() != FtsNodeType::TERM && a->type() != FtsNodeType::PHRASE) {
      continue;
    }
    for (size_t j = i + 1; j < children.size();) {
      auto &b = children[j];
      if (b && same_dedup_key(*a, *b)) {
        a->boost += b->boost;
        children.erase(children.begin() + j);
      } else {
        ++j;
      }
    }
  }
}

// Flatten guard: an inner OrNode can be inlined into a parent OR only when it
// is a pure disjunction — itself unmodified and containing no must/must_not
// children. Otherwise inlining would change semantics (a must_not child would
// silently widen its exclusion scope from the inner OR to the outer OR).
bool can_inline_into_or(const FtsAstNode &child) {
  if (child.type() != FtsNodeType::OR) {
    return false;
  }
  if (child.must || child.must_not) {
    return false;
  }
  const auto &inner = static_cast<const OrNode &>(child);
  for (const auto &c : inner.children) {
    if (c && (c->must || c->must_not)) {
      return false;
    }
  }
  return true;
}

// Flatten guard: an inner AndNode can be inlined into a parent AND only when
// itself unmodified and containing no must_not children. must children inside
// an AND are equivalent to plain children (build_and_iterator treats both as
// MUST), so they are safe to inline. must_not children are NOT safe to lift
// across a must_not parent boundary.
bool can_inline_into_and(const FtsAstNode &child) {
  if (child.type() != FtsNodeType::AND) {
    return false;
  }
  if (child.must || child.must_not) {
    return false;
  }
  const auto &inner = static_cast<const AndNode &>(child);
  for (const auto &c : inner.children) {
    if (c && c->must_not) {
      return false;
    }
  }
  return true;
}

// Splice inlinable OR children's grandchildren in place of the child. Reuses
// each grandchild's unique_ptr — no AST node allocations.
void flatten_or_children(std::vector<FtsAstNodePtr> &children) {
  std::vector<FtsAstNodePtr> out;
  out.reserve(children.size());
  for (auto &child : children) {
    if (child && can_inline_into_or(*child)) {
      auto &inner = static_cast<OrNode &>(*child);
      for (auto &grandchild : inner.children) {
        if (grandchild) {
          out.push_back(std::move(grandchild));
        }
      }
    } else {
      out.push_back(std::move(child));
    }
  }
  children = std::move(out);
}

void flatten_and_children(std::vector<FtsAstNodePtr> &children) {
  std::vector<FtsAstNodePtr> out;
  out.reserve(children.size());
  for (auto &child : children) {
    if (child && can_inline_into_and(*child)) {
      auto &inner = static_cast<AndNode &>(*child);
      for (auto &grandchild : inner.children) {
        if (grandchild) {
          out.push_back(std::move(grandchild));
        }
      }
    } else {
      out.push_back(std::move(child));
    }
  }
  children = std::move(out);
}

// Drop null children left behind by recursive simplify() reporting "this
// subtree contributed nothing" via a moved-out pointer.
void drop_nulls(std::vector<FtsAstNodePtr> &children) {
  children.erase(std::remove_if(children.begin(), children.end(),
                                [](const FtsAstNodePtr &p) { return !p; }),
                 children.end());
}

// Make an EmptyNode carrying the modifier of the node being replaced. This
// preserves +/- semantics so parent nodes interpret the replacement the same
// way they would the original.
FtsAstNodePtr make_empty_like(const FtsAstNode &original) {
  auto e = std::make_unique<EmptyNode>();
  e->must = original.must;
  e->must_not = original.must_not;
  // Boost is meaningless on EmptyNode — it matches nothing — but keep the
  // value for round-trippable debug output.
  e->boost = original.boost;
  return e;
}

// If the AND contains a positive child and a must_not child with the same
// term/phrase key, the conjunction matches nothing.
bool and_has_mustnot_conflict(const AndNode &n) {
  for (size_t i = 0; i < n.children.size(); ++i) {
    const auto &pi = n.children[i];
    if (!pi || pi->must_not) {
      continue;
    }
    if (pi->type() != FtsNodeType::TERM && pi->type() != FtsNodeType::PHRASE) {
      continue;
    }
    for (size_t j = 0; j < n.children.size(); ++j) {
      if (i == j) {
        continue;
      }
      const auto &pj = n.children[j];
      if (!pj || !pj->must_not) {
        continue;
      }
      if (same_term_or_phrase_text(*pi, *pj)) {
        return true;
      }
    }
  }
  return false;
}

void simplify_and(FtsAstNodePtr &node);
void simplify_or(FtsAstNodePtr &node);

void simplify_and(FtsAstNodePtr &node) {
  auto &n = static_cast<AndNode &>(*node);

  // 1. Recurse first so children are already in normal form.
  for (auto &child : n.children) {
    simplify(child);
  }
  drop_nulls(n.children);

  // 2. EmptyNode propagation: a positive EMPTY makes the whole AND empty;
  //    a must_not EMPTY (i.e. "exclude nothing") is a no-op and is dropped.
  for (auto it = n.children.begin(); it != n.children.end();) {
    if ((*it)->type() == FtsNodeType::EMPTY) {
      if ((*it)->must_not) {
        it = n.children.erase(it);
      } else {
        node = make_empty_like(n);
        return;
      }
    } else {
      ++it;
    }
  }

  // 3. Flatten nested AND, then dedup siblings (linear-boost sum).
  flatten_and_children(n.children);
  merge_duplicate_siblings(n.children);

  // 4. `+apple -apple` style conflict → empty doc set.
  if (and_has_mustnot_conflict(n)) {
    node = make_empty_like(n);
    return;
  }

  // 5. AND containing only must_not children has no positive base set to
  //    subtract from — by convention this matches nothing.
  bool any_positive = false;
  for (const auto &c : n.children) {
    if (!c->must_not) {
      any_positive = true;
      break;
    }
  }
  if (!any_positive) {
    node = make_empty_like(n);
    return;
  }

  // 6. Single-child fold. Combine the outer AND's modifier with the surviving
  //    child; if the combination yields must && must_not, replace with EMPTY
  //    (a self-contradictory clause matches nothing).
  if (n.children.size() == 1) {
    FtsAstNodePtr child = std::move(n.children[0]);
    child->must = child->must || n.must;
    child->must_not = child->must_not || n.must_not;
    if (child->must && child->must_not) {
      auto e = std::make_unique<EmptyNode>();
      e->must = n.must;
      e->must_not = n.must_not;
      node = std::move(e);
      return;
    }
    node = std::move(child);
  }
}

void simplify_or(FtsAstNodePtr &node) {
  auto &n = static_cast<OrNode &>(*node);

  for (auto &child : n.children) {
    simplify(child);
  }
  drop_nulls(n.children);

  // EmptyNode in OR: a positive EMPTY contributes no documents → drop it.
  // A must_not EMPTY excludes nothing → also drop. Either way, simply remove.
  n.children.erase(std::remove_if(n.children.begin(), n.children.end(),
                                  [](const FtsAstNodePtr &p) {
                                    return p && p->type() == FtsNodeType::EMPTY;
                                  }),
                   n.children.end());

  flatten_or_children(n.children);
  merge_duplicate_siblings(n.children);

  // Classify children into must (+), must_not (-), and plain buckets.
  size_t mustnot_count = 0;
  size_t must_count = 0;
  for (const auto &c : n.children) {
    if (c->must_not) {
      ++mustnot_count;
    } else if (c->must) {
      ++must_count;
    }
  }

  // OR with only must_not children has no positive base → matches nothing.
  if (mustnot_count == n.children.size()) {
    node = make_empty_like(n);
    return;
  }

  // Canonicalize OR-with-modifiers into AND:
  //   - must_not children  → AND exclusions
  //   - must children      → AND required clauses (must flag cleared)
  //   - plain children     → positive base (if no must) or SHOULD scoring
  // Conflict cases like `+apple -apple` end up inside the new AND where
  // and_has_mustnot_conflict catches them and collapses to EmptyNode.
  if (mustnot_count > 0 || must_count > 0) {
    std::vector<FtsAstNodePtr> must_children;
    std::vector<FtsAstNodePtr> mustnot_children;
    std::vector<FtsAstNodePtr> plain_children;
    for (auto &c : n.children) {
      if (c->must_not) {
        mustnot_children.push_back(std::move(c));
      } else if (c->must) {
        c->must = false;
        must_children.push_back(std::move(c));
      } else {
        plain_children.push_back(std::move(c));
      }
    }

    auto wrap = std::make_unique<AndNode>();
    wrap->children = std::move(must_children);

    if (!plain_children.empty()) {
      FtsAstNodePtr plain_part;
      if (plain_children.size() == 1) {
        plain_part = std::move(plain_children[0]);
      } else {
        auto inner_or = std::make_unique<OrNode>();
        inner_or->children = std::move(plain_children);
        plain_part = std::move(inner_or);
      }
      // When must children exist, plain terms become SHOULD (scoring only);
      // otherwise they are the positive base of the AND.
      if (must_count > 0) {
        plain_part->should = true;
      }
      wrap->children.push_back(std::move(plain_part));
    }

    for (auto &mn : mustnot_children) {
      wrap->children.push_back(std::move(mn));
    }

    wrap->must = n.must;
    wrap->must_not = n.must_not;
    wrap->boost = n.boost;

    FtsAstNodePtr replacement = std::move(wrap);
    simplify_and(replacement);
    node = std::move(replacement);
    return;
  }

  if (n.children.size() == 1) {
    FtsAstNodePtr child = std::move(n.children[0]);
    child->must = child->must || n.must;
    child->must_not = child->must_not || n.must_not;
    if (child->must && child->must_not) {
      auto e = std::make_unique<EmptyNode>();
      e->must = n.must;
      e->must_not = n.must_not;
      node = std::move(e);
      return;
    }
    node = std::move(child);
  }
}

}  // namespace

void simplify(FtsAstNodePtr &node) {
  if (!node) {
    return;
  }
  switch (node->type()) {
    case FtsNodeType::TERM:
    case FtsNodeType::PHRASE:
    case FtsNodeType::EMPTY:
      return;
    case FtsNodeType::AND:
      simplify_and(node);
      return;
    case FtsNodeType::OR:
      simplify_or(node);
      return;
  }
}

}  // namespace zvec::fts
