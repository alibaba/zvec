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

parser grammar FtsParser;

options { tokenVocab = FtsLexer; }

// ── Entry point ───────────────────────────────────────────────────────────────
fts_query_unit
    : fts_or_expr EOF
    ;

// ── OR (lowest precedence) ────────────────────────────────────────────────────
fts_or_expr
    : fts_and_expr (OR fts_and_expr)*
    ;

// ── AND / NOT (same precedence) ──────────────────────────────────────────────
// `a NOT b` is the binary `a AND NOT b` operator: documents matching `a`
// excluding those matching `b`. The explicit form `a AND NOT b` is also
// accepted for readability; semantically it is identical to `a NOT b`.
fts_and_expr
    : fts_seq_expr ((AND NOT? | NOT) fts_seq_expr)*
    ;

// ── Implicit adjacency ────────────────────────────────────────────────────────
// Adjacent atoms without an explicit operator are grouped together; the
// builder treats them as an implicit OR (same behaviour as the original SQL
// parser).
fts_seq_expr
    : fts_unary+
    ;

// ── Unary modifier ────────────────────────────────────────────────────────────
// NOT is *not* a unary modifier here — it is consumed by fts_and_expr above
// as a binary operator. Unary modifiers are limited to `+` (must) and `-`
// (must_not).
fts_unary
    : PLUS_SIGN  fts_atom   # must_atom
    | MINUS_SIGN fts_atom   # must_not_atom
    | fts_atom              # plain_atom
    ;

// ── Atom: optional field prefix + primary + optional boost ───────────────────
fts_atom
    : fts_field_prefix? fts_primary fts_boost?
    ;

// ── Field prefix: REGULAR_ID ':' ─────────────────────────────────────────────
fts_field_prefix
    : REGULAR_ID COLON
    ;

// ── Primary: term | phrase | parenthesised sub-expression ────────────────────
fts_primary
    : fts_term
    | fts_phrase
    | LP fts_or_expr RP
    ;

// ── Boost: '^' NUMBER ────────────────────────────────────────────────────────
fts_boost
    : CARET NUMBER
    ;

fts_natural_term
    : DEFAULT+   // One or more default characters forming a natural language term
    ;

// ── Term: identifier, number, or generic token ───────────────────────────────
fts_term
    : TERM
    | REGULAR_ID
    | NUMBER
    | fts_natural_term
    ;

// ── Phrase: double-quoted string ─────────────────────────────────────────────
fts_phrase
    : DQUOTA_STRING
    ;
