
// Generated from FtsParser.g4 by ANTLR 4.8

#pragma once


#include "FtsParserListener.h"
#include "antlr4-runtime.h"


namespace antlr4 {

/**
 * This class provides an empty implementation of FtsParserListener,
 * which can be extended to create a listener which only needs to handle a
 * subset of the available methods.
 */
class FtsParserBaseListener : public FtsParserListener {
 public:
  virtual void enterFts_query_unit(
      FtsParser::Fts_query_unitContext * /*ctx*/) override {}
  virtual void exitFts_query_unit(
      FtsParser::Fts_query_unitContext * /*ctx*/) override {}

  virtual void enterFts_or_expr(
      FtsParser::Fts_or_exprContext * /*ctx*/) override {}
  virtual void exitFts_or_expr(
      FtsParser::Fts_or_exprContext * /*ctx*/) override {}

  virtual void enterFts_and_expr(
      FtsParser::Fts_and_exprContext * /*ctx*/) override {}
  virtual void exitFts_and_expr(
      FtsParser::Fts_and_exprContext * /*ctx*/) override {}

  virtual void enterFts_seq_expr(
      FtsParser::Fts_seq_exprContext * /*ctx*/) override {}
  virtual void exitFts_seq_expr(
      FtsParser::Fts_seq_exprContext * /*ctx*/) override {}

  virtual void enterMust_atom(FtsParser::Must_atomContext * /*ctx*/) override {}
  virtual void exitMust_atom(FtsParser::Must_atomContext * /*ctx*/) override {}

  virtual void enterMust_not_atom(
      FtsParser::Must_not_atomContext * /*ctx*/) override {}
  virtual void exitMust_not_atom(
      FtsParser::Must_not_atomContext * /*ctx*/) override {}

  virtual void enterPlain_atom(
      FtsParser::Plain_atomContext * /*ctx*/) override {}
  virtual void exitPlain_atom(FtsParser::Plain_atomContext * /*ctx*/) override {
  }

  virtual void enterFts_atom(FtsParser::Fts_atomContext * /*ctx*/) override {}
  virtual void exitFts_atom(FtsParser::Fts_atomContext * /*ctx*/) override {}

  virtual void enterFts_field_prefix(
      FtsParser::Fts_field_prefixContext * /*ctx*/) override {}
  virtual void exitFts_field_prefix(
      FtsParser::Fts_field_prefixContext * /*ctx*/) override {}

  virtual void enterFts_primary(
      FtsParser::Fts_primaryContext * /*ctx*/) override {}
  virtual void exitFts_primary(
      FtsParser::Fts_primaryContext * /*ctx*/) override {}

  virtual void enterFts_boost(FtsParser::Fts_boostContext * /*ctx*/) override {}
  virtual void exitFts_boost(FtsParser::Fts_boostContext * /*ctx*/) override {}

  virtual void enterFts_natural_term(
      FtsParser::Fts_natural_termContext * /*ctx*/) override {}
  virtual void exitFts_natural_term(
      FtsParser::Fts_natural_termContext * /*ctx*/) override {}

  virtual void enterFts_term(FtsParser::Fts_termContext * /*ctx*/) override {}
  virtual void exitFts_term(FtsParser::Fts_termContext * /*ctx*/) override {}

  virtual void enterFts_phrase(
      FtsParser::Fts_phraseContext * /*ctx*/) override {}
  virtual void exitFts_phrase(FtsParser::Fts_phraseContext * /*ctx*/) override {
  }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override {}
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override {}
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override {}
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override {}
};

}  // namespace antlr4
