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

#include "fts_query_parser.h"
#include <zvec/ailego/utility/string_helper.h>
#include "db/index/column/fts_column/gen/FtsLexer.h"
#include "db/index/column/fts_column/gen/FtsParser.h"
#include "antlr4-runtime.h"

using namespace antlr4;

namespace zvec::fts {

// ============================================================
// Error listener that captures the first error message
// ============================================================

class FtsErrorListener : public BaseErrorListener {
 public:
  void syntaxError(Recognizer * /*recognizer*/,
                   antlr4::Token * /*offending_symbol*/, size_t line,
                   size_t char_position_in_line, const std::string &msg,
                   std::exception_ptr /*exception*/) override {
    if (err_msg_.empty()) {
      err_msg_ = ailego::StringHelper::Concat(
          "[", line, " ", char_position_in_line, " ", msg, "]");
    }
  }

  const std::string &err_msg() const {
    return err_msg_;
  }

 private:
  std::string err_msg_;
};

// ============================================================
// AST builder helpers (anonymous namespace)
// ============================================================

namespace {

// Forward declaration
FtsAstNodePtr build_fts_or_expr(FtsParser::Fts_or_exprContext *or_ctx,
                                const TokenizerPipeline &pipeline,
                                FtsDefaultOperator default_op,
                                std::string *err_msg);

// Strip surrounding single or double quotes from a quoted string token.
std::string strip_quotes(const std::string &quoted) {
  if (quoted.size() >= 2 &&
      ((quoted.front() == '\'' && quoted.back() == '\'') ||
       (quoted.front() == '"' && quoted.back() == '"'))) {
    return quoted.substr(1, quoted.size() - 2);
  }
  return quoted;
}

// Remove lexer-level escape backslashes: \X → X.
// The lexer uses backslash sequences to include special characters inside
// terms (ESCAPED_CHAR) and quoted strings (DQUOTA_STRING). After tokenization
// boundaries are determined, the backslashes must be stripped so downstream
// tokenizer pipelines see the intended literal characters.
std::string unescape(std::string text) {
  size_t write = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\\' && i + 1 < text.size()) {
      ++i;
    }
    text[write++] = text[i];
  }
  text.resize(write);
  return text;
}

// Propagate must/must_not modifier to the root of an already-built AST node.
// Now that must/must_not live on the FtsAstNode base class, this works
// uniformly for terms, phrases and composite (AND/OR) sub-expressions.
// OR-merge with any existing flags so a second application on the same
// node never silently clears modifiers set by a prior pass.
void apply_modifier(FtsAstNode *node, bool is_must, bool is_must_not) {
  if (!node || (!is_must && !is_must_not)) {
    return;
  }
  node->must = node->must || is_must;
  node->must_not = node->must_not || is_must_not;
}

// atom: fts_field_prefix? fts_primary fts_boost?
//
// fts_field_prefix (e.g. "title:") and fts_boost (e.g. "^2") are parsed by
// the grammar but not supported at query execution time — return an error.
//
// fts_primary: fts_term | fts_phrase | LP fts_or_expr RP
FtsAstNodePtr build_fts_atom(FtsParser::Fts_atomContext *atom_ctx, bool is_must,
                             bool is_must_not,
                             const TokenizerPipeline &pipeline,
                             FtsDefaultOperator default_op,
                             std::string *err_msg) {
  // Reject field-prefixed queries (e.g. "title:cancer")
  if (atom_ctx->fts_field_prefix() != nullptr) {
    if (err_msg) {
      *err_msg = "field-prefixed queries are not supported";
    }
    return nullptr;
  }

  // Reject boosted queries (e.g. "term^2")
  if (atom_ctx->fts_boost() != nullptr) {
    if (err_msg) {
      *err_msg = "boost queries are not supported";
    }
    return nullptr;
  }

  FtsParser::Fts_primaryContext *primary_ctx = atom_ctx->fts_primary();
  if (primary_ctx == nullptr) {
    return nullptr;
  }

  if (primary_ctx->fts_term() != nullptr) {
    std::string term_text = unescape(primary_ctx->fts_term()->getText());
    auto tokens = pipeline.process(term_text);
    if (tokens.empty()) {
      // Term filtered out (e.g. stop-word, pure punctuation). Returning
      // nullptr here lets the seq/and/or builders skip this child.
      return nullptr;
    }
    if (tokens.size() == 1) {
      return std::make_unique<TermNode>(std::move(tokens[0].text), is_must,
                                        is_must_not);
    }
    // Multi-token bare term: combine via the configured default operator and
    // attach must/must_not on the composite root.
    FtsAstNodePtr composite;
    if (default_op == FtsDefaultOperator::AND) {
      auto and_node = std::make_unique<AndNode>();
      and_node->children.reserve(tokens.size());
      for (auto &t : tokens) {
        and_node->children.push_back(
            std::make_unique<TermNode>(std::move(t.text)));
      }
      composite = std::move(and_node);
    } else {
      auto or_node = std::make_unique<OrNode>();
      or_node->children.reserve(tokens.size());
      for (auto &t : tokens) {
        or_node->children.push_back(
            std::make_unique<TermNode>(std::move(t.text)));
      }
      composite = std::move(or_node);
    }
    apply_modifier(composite.get(), is_must, is_must_not);
    return composite;
  }

  if (primary_ctx->fts_phrase() != nullptr) {
    std::string raw = primary_ctx->fts_phrase()->getText();
    std::string phrase_text = unescape(strip_quotes(raw));
    auto tokens = pipeline.process(phrase_text);
    auto phrase_node = std::make_unique<PhraseNode>();
    phrase_node->must = is_must;
    phrase_node->must_not = is_must_not;
    phrase_node->terms.reserve(tokens.size());
    for (auto &t : tokens) {
      phrase_node->terms.push_back(std::move(t.text));
    }
    return phrase_node;
  }

  if (primary_ctx->fts_or_expr() != nullptr) {
    // Parenthesised sub-expression — propagate default_op so that adjacent
    // bare terms inside the parentheses share the same implicit semantics.
    auto inner = build_fts_or_expr(primary_ctx->fts_or_expr(), pipeline,
                                   default_op, err_msg);
    apply_modifier(inner.get(), is_must, is_must_not);
    return inner;
  }

  return nullptr;
}

// unary: (PLUS_SIGN | MINUS_SIGN)? atom
// NOT is no longer a unary modifier — it is handled as a binary operator in
// build_fts_and_expr. antlr4 generates separate subclasses for each labeled
// alternative.
FtsAstNodePtr build_fts_unary(FtsParser::Fts_unaryContext *unary_ctx,
                              const TokenizerPipeline &pipeline,
                              FtsDefaultOperator default_op,
                              std::string *err_msg) {
  if (auto *must_ctx = dynamic_cast<FtsParser::Must_atomContext *>(unary_ctx)) {
    return build_fts_atom(must_ctx->fts_atom(), /*is_must=*/true,
                          /*is_must_not=*/false, pipeline, default_op, err_msg);
  }
  if (auto *must_not_ctx =
          dynamic_cast<FtsParser::Must_not_atomContext *>(unary_ctx)) {
    return build_fts_atom(must_not_ctx->fts_atom(), /*is_must=*/false,
                          /*is_must_not=*/true, pipeline, default_op, err_msg);
  }
  // Plain_atomContext (no modifier)
  if (auto *plain_ctx =
          dynamic_cast<FtsParser::Plain_atomContext *>(unary_ctx)) {
    return build_fts_atom(plain_ctx->fts_atom(), /*is_must=*/false,
                          /*is_must_not=*/false, pipeline, default_op, err_msg);
  }
  return nullptr;
}

// seqExpr: unary+
// Adjacent terms use the implicit default operator passed in (OR or AND).
// This is the only place where FtsDefaultOperator actually changes the AST
// structure; all other build_* helpers simply propagate the value.
FtsAstNodePtr build_fts_seq_expr(FtsParser::Fts_seq_exprContext *seq_ctx,
                                 const TokenizerPipeline &pipeline,
                                 FtsDefaultOperator default_op,
                                 std::string *err_msg) {
  auto unary_list = seq_ctx->fts_unary();
  if (unary_list.size() == 1) {
    return build_fts_unary(unary_list[0], pipeline, default_op, err_msg);
  }

  // Parse all children first
  std::vector<FtsAstNodePtr> children;
  for (auto *unary_ctx : unary_list) {
    auto child = build_fts_unary(unary_ctx, pipeline, default_op, err_msg);
    if (!child) {
      if (err_msg && !err_msg->empty()) {
        return nullptr;
      }
      continue;
    }
    children.push_back(std::move(child));
  }
  if (children.size() == 1) {
    return std::move(children[0]);
  }

  // Assign children to the appropriate node type
  if (default_op == FtsDefaultOperator::AND) {
    auto and_node = std::make_unique<AndNode>();
    and_node->children = std::move(children);
    return and_node;
  }
  auto or_node = std::make_unique<OrNode>();
  or_node->children = std::move(children);
  return or_node;
}

// andExpr: seqExpr ((AND | NOT) seqExpr)*
//
// NOT shares the same precedence as AND. Each `NOT seqExpr` on the right of
// the operator marks the produced child as must_not, then the whole
// sub-expression collapses into a single AndNode. Example:
//   `a NOT b`        => And[a, b{must_not}]
//   `a AND b NOT c`  => And[a, b, c{must_not}]
FtsAstNodePtr build_fts_and_expr(FtsParser::Fts_and_exprContext *and_ctx,
                                 const TokenizerPipeline &pipeline,
                                 FtsDefaultOperator default_op,
                                 std::string *err_msg) {
  auto and_node = std::make_unique<AndNode>();
  bool next_is_not = false;
  for (auto *raw : and_ctx->children) {
    if (auto *term = dynamic_cast<antlr4::tree::TerminalNode *>(raw)) {
      const auto token_type = term->getSymbol()->getType();
      if (token_type == FtsParser::AND) {
        next_is_not = false;
      } else if (token_type == FtsParser::NOT) {
        next_is_not = true;
      }
      continue;
    }
    auto *seq_ctx = dynamic_cast<FtsParser::Fts_seq_exprContext *>(raw);
    if (seq_ctx == nullptr) {
      continue;
    }
    auto child = build_fts_seq_expr(seq_ctx, pipeline, default_op, err_msg);
    bool is_not_for_this_child = next_is_not;
    next_is_not = false;
    if (!child) {
      if (err_msg && !err_msg->empty()) {
        return nullptr;
      }
      continue;
    }
    if (is_not_for_this_child) {
      apply_modifier(child.get(), /*is_must=*/false, /*is_must_not=*/true);
    }
    and_node->children.push_back(std::move(child));
  }
  if (and_node->children.empty()) {
    return nullptr;
  }
  if (and_node->children.size() == 1) {
    return std::move(and_node->children[0]);
  }
  return and_node;
}

// orExpr: andExpr (OR andExpr)*
FtsAstNodePtr build_fts_or_expr(FtsParser::Fts_or_exprContext *or_ctx,
                                const TokenizerPipeline &pipeline,
                                FtsDefaultOperator default_op,
                                std::string *err_msg) {
  auto and_list = or_ctx->fts_and_expr();
  if (and_list.size() == 1) {
    return build_fts_and_expr(and_list[0], pipeline, default_op, err_msg);
  }
  auto or_node = std::make_unique<OrNode>();
  for (auto *and_ctx : and_list) {
    auto child = build_fts_and_expr(and_ctx, pipeline, default_op, err_msg);
    if (!child) {
      if (err_msg && !err_msg->empty()) {
        return nullptr;
      }
      continue;
    }
    or_node->children.push_back(std::move(child));
  }
  if (or_node->children.size() == 1) {
    return std::move(or_node->children[0]);
  }
  return or_node;
}

}  // anonymous namespace

// ============================================================
// FtsQueryParser::parse()
// ============================================================

FtsAstNodePtr FtsQueryParser::parse(const std::string &query,
                                    const TokenizerPipelinePtr &pipeline,
                                    FtsDefaultOperator default_op) {
  err_msg_.clear();
  if (!pipeline) {
    err_msg_ = "fts parser: pipeline is required";
    return nullptr;
  }

  try {
    ANTLRInputStream input(query);
    FtsLexer lexer(&input);

    FtsErrorListener lexer_error_listener;
    lexer.removeErrorListeners();
    lexer.addErrorListener(&lexer_error_listener);

    CommonTokenStream tokens(&lexer);

    FtsParser parser(&tokens);

    FtsErrorListener parser_error_listener;
    parser.removeErrorListeners();
    parser.addErrorListener(&parser_error_listener);

    // First attempt with SLL prediction mode (fast path)
    parser.getInterpreter<atn::ParserATNSimulator>()->setPredictionMode(
        atn::PredictionMode::SLL);
    FtsParser::Fts_query_unitContext *tree = parser.fts_query_unit();

    // Fall back to full LL mode if SLL produced errors
    if (lexer.getNumberOfSyntaxErrors() > 0 ||
        parser.getNumberOfSyntaxErrors() > 0) {
      tokens.reset();
      parser.reset();
      parser.getInterpreter<atn::ParserATNSimulator>()->setPredictionMode(
          atn::PredictionMode::LL);
      tree = parser.fts_query_unit();
    }

    if (lexer.getNumberOfSyntaxErrors() > 0) {
      err_msg_ = "fts lexer error " + lexer_error_listener.err_msg();
      return nullptr;
    }
    if (parser.getNumberOfSyntaxErrors() > 0) {
      err_msg_ = "fts syntax error " + parser_error_listener.err_msg();
      return nullptr;
    }

    if (tree == nullptr || tree->fts_or_expr() == nullptr) {
      err_msg_ = "fts parse error: empty or invalid query";
      return nullptr;
    }

    auto result = build_fts_or_expr(tree->fts_or_expr(), *pipeline, default_op,
                                    &err_msg_);
    if (!result && !err_msg_.empty()) {
      return nullptr;
    }
    if (!result) {
      // Grammar valid but analyzer dropped every term: return EmptyNode so
      // callers don't have to treat zero-doc queries as parse errors.
      return std::make_unique<EmptyNode>();
    }
    return result;

  } catch (const std::exception &exception) {
    err_msg_ = "fts parse exception: " + std::string(exception.what());
    return nullptr;
  }
}

}  // namespace zvec::fts
