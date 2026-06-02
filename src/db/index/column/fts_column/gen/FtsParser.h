
// Generated from FtsParser.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"


namespace antlr4 {


class FtsParser : public antlr4::Parser {
 public:
  enum {
    OR = 1,
    AND = 2,
    NOT = 3,
    PLUS_SIGN = 4,
    MINUS_SIGN = 5,
    COLON = 6,
    CARET = 7,
    LP = 8,
    RP = 9,
    DQUOTA_STRING = 10,
    REGULAR_ID = 11,
    NUMBER = 12,
    TERM = 13,
    SPACES = 14,
    DEFAULT = 15
  };

  enum {
    RuleFts_query_unit = 0,
    RuleFts_or_expr = 1,
    RuleFts_and_expr = 2,
    RuleFts_seq_expr = 3,
    RuleFts_unary = 4,
    RuleFts_atom = 5,
    RuleFts_field_prefix = 6,
    RuleFts_primary = 7,
    RuleFts_boost = 8,
    RuleFts_natural_term = 9,
    RuleFts_term = 10,
    RuleFts_phrase = 11
  };

  FtsParser(antlr4::TokenStream *input);
  ~FtsParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN &getATN() const override {
    return _atn;
  };
  virtual const std::vector<std::string> &getTokenNames() const override {
    return _tokenNames;
  };  // deprecated: use vocabulary instead.
  virtual const std::vector<std::string> &getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary &getVocabulary() const override;


  class Fts_query_unitContext;
  class Fts_or_exprContext;
  class Fts_and_exprContext;
  class Fts_seq_exprContext;
  class Fts_unaryContext;
  class Fts_atomContext;
  class Fts_field_prefixContext;
  class Fts_primaryContext;
  class Fts_boostContext;
  class Fts_natural_termContext;
  class Fts_termContext;
  class Fts_phraseContext;

  class Fts_query_unitContext : public antlr4::ParserRuleContext {
   public:
    Fts_query_unitContext(antlr4::ParserRuleContext *parent_ctx,
                          size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    Fts_or_exprContext *fts_or_expr();
    antlr4::tree::TerminalNode *EOF();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_query_unitContext *fts_query_unit();

  class Fts_or_exprContext : public antlr4::ParserRuleContext {
   public:
    Fts_or_exprContext(antlr4::ParserRuleContext *parent_ctx,
                       size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    std::vector<Fts_and_exprContext *> fts_and_expr();
    Fts_and_exprContext *fts_and_expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> OR();
    antlr4::tree::TerminalNode *OR(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_or_exprContext *fts_or_expr();

  class Fts_and_exprContext : public antlr4::ParserRuleContext {
   public:
    Fts_and_exprContext(antlr4::ParserRuleContext *parent_ctx,
                        size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    std::vector<Fts_seq_exprContext *> fts_seq_expr();
    Fts_seq_exprContext *fts_seq_expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> AND();
    antlr4::tree::TerminalNode *AND(size_t i);
    std::vector<antlr4::tree::TerminalNode *> NOT();
    antlr4::tree::TerminalNode *NOT(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_and_exprContext *fts_and_expr();

  class Fts_seq_exprContext : public antlr4::ParserRuleContext {
   public:
    Fts_seq_exprContext(antlr4::ParserRuleContext *parent_ctx,
                        size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    std::vector<Fts_unaryContext *> fts_unary();
    Fts_unaryContext *fts_unary(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_seq_exprContext *fts_seq_expr();

  class Fts_unaryContext : public antlr4::ParserRuleContext {
   public:
    Fts_unaryContext(antlr4::ParserRuleContext *parent_ctx,
                     size_t invoking_state);

    Fts_unaryContext() = default;
    void copyFrom(Fts_unaryContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;
  };

  class Must_not_atomContext : public Fts_unaryContext {
   public:
    Must_not_atomContext(Fts_unaryContext *ctx);

    antlr4::tree::TerminalNode *MINUS_SIGN();
    Fts_atomContext *fts_atom();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class Must_atomContext : public Fts_unaryContext {
   public:
    Must_atomContext(Fts_unaryContext *ctx);

    antlr4::tree::TerminalNode *PLUS_SIGN();
    Fts_atomContext *fts_atom();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  class Plain_atomContext : public Fts_unaryContext {
   public:
    Plain_atomContext(Fts_unaryContext *ctx);

    Fts_atomContext *fts_atom();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_unaryContext *fts_unary();

  class Fts_atomContext : public antlr4::ParserRuleContext {
   public:
    Fts_atomContext(antlr4::ParserRuleContext *parent_ctx,
                    size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    Fts_primaryContext *fts_primary();
    Fts_field_prefixContext *fts_field_prefix();
    Fts_boostContext *fts_boost();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_atomContext *fts_atom();

  class Fts_field_prefixContext : public antlr4::ParserRuleContext {
   public:
    Fts_field_prefixContext(antlr4::ParserRuleContext *parent_ctx,
                            size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *REGULAR_ID();
    antlr4::tree::TerminalNode *COLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_field_prefixContext *fts_field_prefix();

  class Fts_primaryContext : public antlr4::ParserRuleContext {
   public:
    Fts_primaryContext(antlr4::ParserRuleContext *parent_ctx,
                       size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    Fts_termContext *fts_term();
    Fts_phraseContext *fts_phrase();
    antlr4::tree::TerminalNode *LP();
    Fts_or_exprContext *fts_or_expr();
    antlr4::tree::TerminalNode *RP();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_primaryContext *fts_primary();

  class Fts_boostContext : public antlr4::ParserRuleContext {
   public:
    Fts_boostContext(antlr4::ParserRuleContext *parent_ctx,
                     size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CARET();
    antlr4::tree::TerminalNode *NUMBER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_boostContext *fts_boost();

  class Fts_natural_termContext : public antlr4::ParserRuleContext {
   public:
    Fts_natural_termContext(antlr4::ParserRuleContext *parent_ctx,
                            size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> DEFAULT();
    antlr4::tree::TerminalNode *DEFAULT(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_natural_termContext *fts_natural_term();

  class Fts_termContext : public antlr4::ParserRuleContext {
   public:
    Fts_termContext(antlr4::ParserRuleContext *parent_ctx,
                    size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *TERM();
    antlr4::tree::TerminalNode *REGULAR_ID();
    antlr4::tree::TerminalNode *NUMBER();
    Fts_natural_termContext *fts_natural_term();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_termContext *fts_term();

  class Fts_phraseContext : public antlr4::ParserRuleContext {
   public:
    Fts_phraseContext(antlr4::ParserRuleContext *parent_ctx,
                      size_t invoking_state);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DQUOTA_STRING();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
  };

  Fts_phraseContext *fts_phrase();


 private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

}  // namespace antlr4
