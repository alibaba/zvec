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

lexer grammar FtsLexer;

// ── Boolean operators ────────────────────────────────────────────────────────
OR  : [Oo][Rr];
AND : [Aa][Nn][Dd];
NOT : [Nn][Oo][Tt];

// ── Modifier prefixes ────────────────────────────────────────────────────────
PLUS_SIGN:   '+';
MINUS_SIGN:  '-';

COLON:       ':';
CARET:       '^';

// ── Grouping ─────────────────────────────────────────────────────────────────
LP:          '(';
RP:          ')';

// ── Quoted strings (phrase queries) ──────────────────────────────────────────
DQUOTA_STRING
    : '"' (~["\\\r\n] | '\\' .)* '"'
    ;


fragment ASCII_ALNUM : [A-Za-z0-9_];
fragment ESCAPED_CHAR
    : '\\' [-+=&|!(){}[\]^"~*?:\\/]
    ;
fragment UNI_CHAR    : [\u0080-\uFFFF];
fragment TERM_START  : ASCII_ALNUM | UNI_CHAR;
fragment TERM_BODY   : ASCII_ALNUM | UNI_CHAR | [._#/%\-'@] | ESCAPED_CHAR;

// Matches sequences of letters, digits, underscores and hyphens that start
// with a letter or underscore (same as the original SQLLexer REGULAR_ID).
REGULAR_ID: [A-Za-z_] [A-Za-z0-9_\-]*;

NUMBER: [0-9]+ ('.' [0-9]+)?;

// Generic term
TERM: TERM_START TERM_BODY*;

// ── Whitespace (skip) ─────────────────────────────────────────────────────────
SPACES: [ \t\r\n]+ -> skip;

DEFAULT: . ;
