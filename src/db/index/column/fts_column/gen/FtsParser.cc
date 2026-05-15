
// Generated from FtsParser.g4 by ANTLR 4.8


#include "FtsParser.h"
#include "FtsParserListener.h"


using namespace antlrcpp;
using namespace antlr4;
using namespace antlr4;

FtsParser::FtsParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA,
                                             _sharedContextCache);
}

FtsParser::~FtsParser() {
  delete _interpreter;
}

std::string FtsParser::getGrammarFileName() const {
  return "FtsParser.g4";
}

const std::vector<std::string> &FtsParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary &FtsParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- Fts_query_unitContext
//------------------------------------------------------------------

FtsParser::Fts_query_unitContext::Fts_query_unitContext(
    ParserRuleContext *parent_ctx, size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

FtsParser::Fts_or_exprContext *FtsParser::Fts_query_unitContext::fts_or_expr() {
  return getRuleContext<FtsParser::Fts_or_exprContext>(0);
}

tree::TerminalNode *FtsParser::Fts_query_unitContext::EOF() {
  return getToken(FtsParser::EOF, 0);
}


size_t FtsParser::Fts_query_unitContext::getRuleIndex() const {
  return FtsParser::RuleFts_query_unit;
}

void FtsParser::Fts_query_unitContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_query_unit(this);
}

void FtsParser::Fts_query_unitContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_query_unit(this);
}

FtsParser::Fts_query_unitContext *FtsParser::fts_query_unit() {
  Fts_query_unitContext *_localctx =
      _tracker.createInstance<Fts_query_unitContext>(_ctx, getState());
  enterRule(_localctx, 0, FtsParser::RuleFts_query_unit);

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(24);
    fts_or_expr();
    setState(25);
    match(FtsParser::EOF);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_or_exprContext
//------------------------------------------------------------------

FtsParser::Fts_or_exprContext::Fts_or_exprContext(ParserRuleContext *parent_ctx,
                                                  size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

std::vector<FtsParser::Fts_and_exprContext *>
FtsParser::Fts_or_exprContext::fts_and_expr() {
  return getRuleContexts<FtsParser::Fts_and_exprContext>();
}

FtsParser::Fts_and_exprContext *FtsParser::Fts_or_exprContext::fts_and_expr(
    size_t i) {
  return getRuleContext<FtsParser::Fts_and_exprContext>(i);
}

std::vector<tree::TerminalNode *> FtsParser::Fts_or_exprContext::OR() {
  return getTokens(FtsParser::OR);
}

tree::TerminalNode *FtsParser::Fts_or_exprContext::OR(size_t i) {
  return getToken(FtsParser::OR, i);
}


size_t FtsParser::Fts_or_exprContext::getRuleIndex() const {
  return FtsParser::RuleFts_or_expr;
}

void FtsParser::Fts_or_exprContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_or_expr(this);
}

void FtsParser::Fts_or_exprContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_or_expr(this);
}

FtsParser::Fts_or_exprContext *FtsParser::fts_or_expr() {
  Fts_or_exprContext *_localctx =
      _tracker.createInstance<Fts_or_exprContext>(_ctx, getState());
  enterRule(_localctx, 2, FtsParser::RuleFts_or_expr);
  size_t _la = 0;

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(27);
    fts_and_expr();
    setState(32);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FtsParser::OR) {
      setState(28);
      match(FtsParser::OR);
      setState(29);
      fts_and_expr();
      setState(34);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_and_exprContext
//------------------------------------------------------------------

FtsParser::Fts_and_exprContext::Fts_and_exprContext(
    ParserRuleContext *parent_ctx, size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

std::vector<FtsParser::Fts_seq_exprContext *>
FtsParser::Fts_and_exprContext::fts_seq_expr() {
  return getRuleContexts<FtsParser::Fts_seq_exprContext>();
}

FtsParser::Fts_seq_exprContext *FtsParser::Fts_and_exprContext::fts_seq_expr(
    size_t i) {
  return getRuleContext<FtsParser::Fts_seq_exprContext>(i);
}

std::vector<tree::TerminalNode *> FtsParser::Fts_and_exprContext::AND() {
  return getTokens(FtsParser::AND);
}

tree::TerminalNode *FtsParser::Fts_and_exprContext::AND(size_t i) {
  return getToken(FtsParser::AND, i);
}

std::vector<tree::TerminalNode *> FtsParser::Fts_and_exprContext::NOT() {
  return getTokens(FtsParser::NOT);
}

tree::TerminalNode *FtsParser::Fts_and_exprContext::NOT(size_t i) {
  return getToken(FtsParser::NOT, i);
}


size_t FtsParser::Fts_and_exprContext::getRuleIndex() const {
  return FtsParser::RuleFts_and_expr;
}

void FtsParser::Fts_and_exprContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_and_expr(this);
}

void FtsParser::Fts_and_exprContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_and_expr(this);
}

FtsParser::Fts_and_exprContext *FtsParser::fts_and_expr() {
  Fts_and_exprContext *_localctx =
      _tracker.createInstance<Fts_and_exprContext>(_ctx, getState());
  enterRule(_localctx, 4, FtsParser::RuleFts_and_expr);
  size_t _la = 0;

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(35);
    fts_seq_expr();
    setState(46);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FtsParser::AND

           || _la == FtsParser::NOT) {
      setState(41);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case FtsParser::AND: {
          setState(36);
          match(FtsParser::AND);
          setState(38);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == FtsParser::NOT) {
            setState(37);
            match(FtsParser::NOT);
          }
          break;
        }

        case FtsParser::NOT: {
          setState(40);
          match(FtsParser::NOT);
          break;
        }

        default:
          throw NoViableAltException(this);
      }
      setState(43);
      fts_seq_expr();
      setState(48);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_seq_exprContext
//------------------------------------------------------------------

FtsParser::Fts_seq_exprContext::Fts_seq_exprContext(
    ParserRuleContext *parent_ctx, size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

std::vector<FtsParser::Fts_unaryContext *>
FtsParser::Fts_seq_exprContext::fts_unary() {
  return getRuleContexts<FtsParser::Fts_unaryContext>();
}

FtsParser::Fts_unaryContext *FtsParser::Fts_seq_exprContext::fts_unary(
    size_t i) {
  return getRuleContext<FtsParser::Fts_unaryContext>(i);
}


size_t FtsParser::Fts_seq_exprContext::getRuleIndex() const {
  return FtsParser::RuleFts_seq_expr;
}

void FtsParser::Fts_seq_exprContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_seq_expr(this);
}

void FtsParser::Fts_seq_exprContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_seq_expr(this);
}

FtsParser::Fts_seq_exprContext *FtsParser::fts_seq_expr() {
  Fts_seq_exprContext *_localctx =
      _tracker.createInstance<Fts_seq_exprContext>(_ctx, getState());
  enterRule(_localctx, 6, FtsParser::RuleFts_seq_expr);
  size_t _la = 0;

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(50);
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(49);
      fts_unary();
      setState(52);
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (
        (((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) &
          ((1ULL << FtsParser::PLUS_SIGN) | (1ULL << FtsParser::MINUS_SIGN) |
           (1ULL << FtsParser::LP) | (1ULL << FtsParser::DQUOTA_STRING) |
           (1ULL << FtsParser::REGULAR_ID) | (1ULL << FtsParser::NUMBER) |
           (1ULL << FtsParser::TERM) | (1ULL << FtsParser::DEFAULT))) != 0));

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_unaryContext
//------------------------------------------------------------------

FtsParser::Fts_unaryContext::Fts_unaryContext(ParserRuleContext *parent_ctx,
                                              size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}


size_t FtsParser::Fts_unaryContext::getRuleIndex() const {
  return FtsParser::RuleFts_unary;
}

void FtsParser::Fts_unaryContext::copyFrom(Fts_unaryContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Must_not_atomContext
//------------------------------------------------------------------

tree::TerminalNode *FtsParser::Must_not_atomContext::MINUS_SIGN() {
  return getToken(FtsParser::MINUS_SIGN, 0);
}

FtsParser::Fts_atomContext *FtsParser::Must_not_atomContext::fts_atom() {
  return getRuleContext<FtsParser::Fts_atomContext>(0);
}

FtsParser::Must_not_atomContext::Must_not_atomContext(Fts_unaryContext *ctx) {
  copyFrom(ctx);
}

void FtsParser::Must_not_atomContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterMust_not_atom(this);
}
void FtsParser::Must_not_atomContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitMust_not_atom(this);
}
//----------------- Must_atomContext
//------------------------------------------------------------------

tree::TerminalNode *FtsParser::Must_atomContext::PLUS_SIGN() {
  return getToken(FtsParser::PLUS_SIGN, 0);
}

FtsParser::Fts_atomContext *FtsParser::Must_atomContext::fts_atom() {
  return getRuleContext<FtsParser::Fts_atomContext>(0);
}

FtsParser::Must_atomContext::Must_atomContext(Fts_unaryContext *ctx) {
  copyFrom(ctx);
}

void FtsParser::Must_atomContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterMust_atom(this);
}
void FtsParser::Must_atomContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitMust_atom(this);
}
//----------------- Plain_atomContext
//------------------------------------------------------------------

FtsParser::Fts_atomContext *FtsParser::Plain_atomContext::fts_atom() {
  return getRuleContext<FtsParser::Fts_atomContext>(0);
}

FtsParser::Plain_atomContext::Plain_atomContext(Fts_unaryContext *ctx) {
  copyFrom(ctx);
}

void FtsParser::Plain_atomContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterPlain_atom(this);
}
void FtsParser::Plain_atomContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitPlain_atom(this);
}
FtsParser::Fts_unaryContext *FtsParser::fts_unary() {
  Fts_unaryContext *_localctx =
      _tracker.createInstance<Fts_unaryContext>(_ctx, getState());
  enterRule(_localctx, 8, FtsParser::RuleFts_unary);

  auto onExit = finally([=] { exitRule(); });
  try {
    setState(59);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FtsParser::PLUS_SIGN: {
        _localctx = dynamic_cast<Fts_unaryContext *>(
            _tracker.createInstance<FtsParser::Must_atomContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(54);
        match(FtsParser::PLUS_SIGN);
        setState(55);
        fts_atom();
        break;
      }

      case FtsParser::MINUS_SIGN: {
        _localctx = dynamic_cast<Fts_unaryContext *>(
            _tracker.createInstance<FtsParser::Must_not_atomContext>(
                _localctx));
        enterOuterAlt(_localctx, 2);
        setState(56);
        match(FtsParser::MINUS_SIGN);
        setState(57);
        fts_atom();
        break;
      }

      case FtsParser::LP:
      case FtsParser::DQUOTA_STRING:
      case FtsParser::REGULAR_ID:
      case FtsParser::NUMBER:
      case FtsParser::TERM:
      case FtsParser::DEFAULT: {
        _localctx = dynamic_cast<Fts_unaryContext *>(
            _tracker.createInstance<FtsParser::Plain_atomContext>(_localctx));
        enterOuterAlt(_localctx, 3);
        setState(58);
        fts_atom();
        break;
      }

      default:
        throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_atomContext
//------------------------------------------------------------------

FtsParser::Fts_atomContext::Fts_atomContext(ParserRuleContext *parent_ctx,
                                            size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

FtsParser::Fts_primaryContext *FtsParser::Fts_atomContext::fts_primary() {
  return getRuleContext<FtsParser::Fts_primaryContext>(0);
}

FtsParser::Fts_field_prefixContext *
FtsParser::Fts_atomContext::fts_field_prefix() {
  return getRuleContext<FtsParser::Fts_field_prefixContext>(0);
}

FtsParser::Fts_boostContext *FtsParser::Fts_atomContext::fts_boost() {
  return getRuleContext<FtsParser::Fts_boostContext>(0);
}


size_t FtsParser::Fts_atomContext::getRuleIndex() const {
  return FtsParser::RuleFts_atom;
}

void FtsParser::Fts_atomContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_atom(this);
}

void FtsParser::Fts_atomContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_atom(this);
}

FtsParser::Fts_atomContext *FtsParser::fts_atom() {
  Fts_atomContext *_localctx =
      _tracker.createInstance<Fts_atomContext>(_ctx, getState());
  enterRule(_localctx, 10, FtsParser::RuleFts_atom);
  size_t _la = 0;

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(62);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 6, _ctx)) {
      case 1: {
        setState(61);
        fts_field_prefix();
        break;
      }
    }
    setState(64);
    fts_primary();
    setState(66);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == FtsParser::CARET) {
      setState(65);
      fts_boost();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_field_prefixContext
//------------------------------------------------------------------

FtsParser::Fts_field_prefixContext::Fts_field_prefixContext(
    ParserRuleContext *parent_ctx, size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

tree::TerminalNode *FtsParser::Fts_field_prefixContext::REGULAR_ID() {
  return getToken(FtsParser::REGULAR_ID, 0);
}

tree::TerminalNode *FtsParser::Fts_field_prefixContext::COLON() {
  return getToken(FtsParser::COLON, 0);
}


size_t FtsParser::Fts_field_prefixContext::getRuleIndex() const {
  return FtsParser::RuleFts_field_prefix;
}

void FtsParser::Fts_field_prefixContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_field_prefix(this);
}

void FtsParser::Fts_field_prefixContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_field_prefix(this);
}

FtsParser::Fts_field_prefixContext *FtsParser::fts_field_prefix() {
  Fts_field_prefixContext *_localctx =
      _tracker.createInstance<Fts_field_prefixContext>(_ctx, getState());
  enterRule(_localctx, 12, FtsParser::RuleFts_field_prefix);

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(68);
    match(FtsParser::REGULAR_ID);
    setState(69);
    match(FtsParser::COLON);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_primaryContext
//------------------------------------------------------------------

FtsParser::Fts_primaryContext::Fts_primaryContext(ParserRuleContext *parent_ctx,
                                                  size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

FtsParser::Fts_termContext *FtsParser::Fts_primaryContext::fts_term() {
  return getRuleContext<FtsParser::Fts_termContext>(0);
}

FtsParser::Fts_phraseContext *FtsParser::Fts_primaryContext::fts_phrase() {
  return getRuleContext<FtsParser::Fts_phraseContext>(0);
}

tree::TerminalNode *FtsParser::Fts_primaryContext::LP() {
  return getToken(FtsParser::LP, 0);
}

FtsParser::Fts_or_exprContext *FtsParser::Fts_primaryContext::fts_or_expr() {
  return getRuleContext<FtsParser::Fts_or_exprContext>(0);
}

tree::TerminalNode *FtsParser::Fts_primaryContext::RP() {
  return getToken(FtsParser::RP, 0);
}


size_t FtsParser::Fts_primaryContext::getRuleIndex() const {
  return FtsParser::RuleFts_primary;
}

void FtsParser::Fts_primaryContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_primary(this);
}

void FtsParser::Fts_primaryContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_primary(this);
}

FtsParser::Fts_primaryContext *FtsParser::fts_primary() {
  Fts_primaryContext *_localctx =
      _tracker.createInstance<Fts_primaryContext>(_ctx, getState());
  enterRule(_localctx, 14, FtsParser::RuleFts_primary);

  auto onExit = finally([=] { exitRule(); });
  try {
    setState(77);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FtsParser::REGULAR_ID:
      case FtsParser::NUMBER:
      case FtsParser::TERM:
      case FtsParser::DEFAULT: {
        enterOuterAlt(_localctx, 1);
        setState(71);
        fts_term();
        break;
      }

      case FtsParser::DQUOTA_STRING: {
        enterOuterAlt(_localctx, 2);
        setState(72);
        fts_phrase();
        break;
      }

      case FtsParser::LP: {
        enterOuterAlt(_localctx, 3);
        setState(73);
        match(FtsParser::LP);
        setState(74);
        fts_or_expr();
        setState(75);
        match(FtsParser::RP);
        break;
      }

      default:
        throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_boostContext
//------------------------------------------------------------------

FtsParser::Fts_boostContext::Fts_boostContext(ParserRuleContext *parent_ctx,
                                              size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

tree::TerminalNode *FtsParser::Fts_boostContext::CARET() {
  return getToken(FtsParser::CARET, 0);
}

tree::TerminalNode *FtsParser::Fts_boostContext::NUMBER() {
  return getToken(FtsParser::NUMBER, 0);
}


size_t FtsParser::Fts_boostContext::getRuleIndex() const {
  return FtsParser::RuleFts_boost;
}

void FtsParser::Fts_boostContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_boost(this);
}

void FtsParser::Fts_boostContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_boost(this);
}

FtsParser::Fts_boostContext *FtsParser::fts_boost() {
  Fts_boostContext *_localctx =
      _tracker.createInstance<Fts_boostContext>(_ctx, getState());
  enterRule(_localctx, 16, FtsParser::RuleFts_boost);

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(79);
    match(FtsParser::CARET);
    setState(80);
    match(FtsParser::NUMBER);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_natural_termContext
//------------------------------------------------------------------

FtsParser::Fts_natural_termContext::Fts_natural_termContext(
    ParserRuleContext *parent_ctx, size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

std::vector<tree::TerminalNode *>
FtsParser::Fts_natural_termContext::DEFAULT() {
  return getTokens(FtsParser::DEFAULT);
}

tree::TerminalNode *FtsParser::Fts_natural_termContext::DEFAULT(size_t i) {
  return getToken(FtsParser::DEFAULT, i);
}


size_t FtsParser::Fts_natural_termContext::getRuleIndex() const {
  return FtsParser::RuleFts_natural_term;
}

void FtsParser::Fts_natural_termContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_natural_term(this);
}

void FtsParser::Fts_natural_termContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_natural_term(this);
}

FtsParser::Fts_natural_termContext *FtsParser::fts_natural_term() {
  Fts_natural_termContext *_localctx =
      _tracker.createInstance<Fts_natural_termContext>(_ctx, getState());
  enterRule(_localctx, 18, FtsParser::RuleFts_natural_term);

  auto onExit = finally([=] { exitRule(); });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(83);
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
          setState(82);
          match(FtsParser::DEFAULT);
          break;
        }

        default:
          throw NoViableAltException(this);
      }
      setState(85);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                       9, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_termContext
//------------------------------------------------------------------

FtsParser::Fts_termContext::Fts_termContext(ParserRuleContext *parent_ctx,
                                            size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

tree::TerminalNode *FtsParser::Fts_termContext::TERM() {
  return getToken(FtsParser::TERM, 0);
}

tree::TerminalNode *FtsParser::Fts_termContext::REGULAR_ID() {
  return getToken(FtsParser::REGULAR_ID, 0);
}

tree::TerminalNode *FtsParser::Fts_termContext::NUMBER() {
  return getToken(FtsParser::NUMBER, 0);
}

FtsParser::Fts_natural_termContext *
FtsParser::Fts_termContext::fts_natural_term() {
  return getRuleContext<FtsParser::Fts_natural_termContext>(0);
}


size_t FtsParser::Fts_termContext::getRuleIndex() const {
  return FtsParser::RuleFts_term;
}

void FtsParser::Fts_termContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_term(this);
}

void FtsParser::Fts_termContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_term(this);
}

FtsParser::Fts_termContext *FtsParser::fts_term() {
  Fts_termContext *_localctx =
      _tracker.createInstance<Fts_termContext>(_ctx, getState());
  enterRule(_localctx, 20, FtsParser::RuleFts_term);

  auto onExit = finally([=] { exitRule(); });
  try {
    setState(91);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FtsParser::TERM: {
        enterOuterAlt(_localctx, 1);
        setState(87);
        match(FtsParser::TERM);
        break;
      }

      case FtsParser::REGULAR_ID: {
        enterOuterAlt(_localctx, 2);
        setState(88);
        match(FtsParser::REGULAR_ID);
        break;
      }

      case FtsParser::NUMBER: {
        enterOuterAlt(_localctx, 3);
        setState(89);
        match(FtsParser::NUMBER);
        break;
      }

      case FtsParser::DEFAULT: {
        enterOuterAlt(_localctx, 4);
        setState(90);
        fts_natural_term();
        break;
      }

      default:
        throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Fts_phraseContext
//------------------------------------------------------------------

FtsParser::Fts_phraseContext::Fts_phraseContext(ParserRuleContext *parent_ctx,
                                                size_t invoking_state)
    : ParserRuleContext(parent_ctx, invoking_state) {}

tree::TerminalNode *FtsParser::Fts_phraseContext::DQUOTA_STRING() {
  return getToken(FtsParser::DQUOTA_STRING, 0);
}


size_t FtsParser::Fts_phraseContext::getRuleIndex() const {
  return FtsParser::RuleFts_phrase;
}

void FtsParser::Fts_phraseContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->enterFts_phrase(this);
}

void FtsParser::Fts_phraseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FtsParserListener *>(listener);
  if (parserListener != nullptr) parserListener->exitFts_phrase(this);
}

FtsParser::Fts_phraseContext *FtsParser::fts_phrase() {
  Fts_phraseContext *_localctx =
      _tracker.createInstance<Fts_phraseContext>(_ctx, getState());
  enterRule(_localctx, 22, FtsParser::RuleFts_phrase);

  auto onExit = finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(93);
    match(FtsParser::DQUOTA_STRING);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> FtsParser::_decisionToDFA;
atn::PredictionContextCache FtsParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN FtsParser::_atn;
std::vector<uint16_t> FtsParser::_serializedATN;

std::vector<std::string> FtsParser::_ruleNames = {
    "fts_query_unit", "fts_or_expr",      "fts_and_expr",     "fts_seq_expr",
    "fts_unary",      "fts_atom",         "fts_field_prefix", "fts_primary",
    "fts_boost",      "fts_natural_term", "fts_term",         "fts_phrase"};

std::vector<std::string> FtsParser::_literalNames = {
    "", "", "", "", "'+'", "'-'", "':'", "'^'", "'('", "')'"};

std::vector<std::string> FtsParser::_symbolicNames = {
    "",       "OR",    "AND",    "NOT",    "PLUS_SIGN",     "MINUS_SIGN",
    "COLON",  "CARET", "LP",     "RP",     "DQUOTA_STRING", "REGULAR_ID",
    "NUMBER", "TERM",  "SPACES", "DEFAULT"};

dfa::Vocabulary FtsParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> FtsParser::_tokenNames;

FtsParser::Initializer::Initializer() {
  for (size_t i = 0; i < _symbolicNames.size(); ++i) {
    std::string name = _vocabulary.getLiteralName(i);
    if (name.empty()) {
      name = _vocabulary.getSymbolicName(i);
    }

    if (name.empty()) {
      _tokenNames.push_back("<INVALID>");
    } else {
      _tokenNames.push_back(name);
    }
  }

  _serializedATN = {
      0x3,  0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964,
      0x3,  0x11,   0x62,   0x4,    0x2,    0x9,    0x2,    0x4,    0x3,
      0x9,  0x3,    0x4,    0x4,    0x9,    0x4,    0x4,    0x5,    0x9,
      0x5,  0x4,    0x6,    0x9,    0x6,    0x4,    0x7,    0x9,    0x7,
      0x4,  0x8,    0x9,    0x8,    0x4,    0x9,    0x9,    0x9,    0x4,
      0xa,  0x9,    0xa,    0x4,    0xb,    0x9,    0xb,    0x4,    0xc,
      0x9,  0xc,    0x4,    0xd,    0x9,    0xd,    0x3,    0x2,    0x3,
      0x2,  0x3,    0x2,    0x3,    0x3,    0x3,    0x3,    0x3,    0x3,
      0x7,  0x3,    0x21,   0xa,    0x3,    0xc,    0x3,    0xe,    0x3,
      0x24, 0xb,    0x3,    0x3,    0x4,    0x3,    0x4,    0x3,    0x4,
      0x5,  0x4,    0x29,   0xa,    0x4,    0x3,    0x4,    0x5,    0x4,
      0x2c, 0xa,    0x4,    0x3,    0x4,    0x7,    0x4,    0x2f,   0xa,
      0x4,  0xc,    0x4,    0xe,    0x4,    0x32,   0xb,    0x4,    0x3,
      0x5,  0x6,    0x5,    0x35,   0xa,    0x5,    0xd,    0x5,    0xe,
      0x5,  0x36,   0x3,    0x6,    0x3,    0x6,    0x3,    0x6,    0x3,
      0x6,  0x3,    0x6,    0x5,    0x6,    0x3e,   0xa,    0x6,    0x3,
      0x7,  0x5,    0x7,    0x41,   0xa,    0x7,    0x3,    0x7,    0x3,
      0x7,  0x5,    0x7,    0x45,   0xa,    0x7,    0x3,    0x8,    0x3,
      0x8,  0x3,    0x8,    0x3,    0x9,    0x3,    0x9,    0x3,    0x9,
      0x3,  0x9,    0x3,    0x9,    0x3,    0x9,    0x5,    0x9,    0x50,
      0xa,  0x9,    0x3,    0xa,    0x3,    0xa,    0x3,    0xa,    0x3,
      0xb,  0x6,    0xb,    0x56,   0xa,    0xb,    0xd,    0xb,    0xe,
      0xb,  0x57,   0x3,    0xc,    0x3,    0xc,    0x3,    0xc,    0x3,
      0xc,  0x5,    0xc,    0x5e,   0xa,    0xc,    0x3,    0xd,    0x3,
      0xd,  0x3,    0xd,    0x2,    0x2,    0xe,    0x2,    0x4,    0x6,
      0x8,  0xa,    0xc,    0xe,    0x10,   0x12,   0x14,   0x16,   0x18,
      0x2,  0x2,    0x2,    0x64,   0x2,    0x1a,   0x3,    0x2,    0x2,
      0x2,  0x4,    0x1d,   0x3,    0x2,    0x2,    0x2,    0x6,    0x25,
      0x3,  0x2,    0x2,    0x2,    0x8,    0x34,   0x3,    0x2,    0x2,
      0x2,  0xa,    0x3d,   0x3,    0x2,    0x2,    0x2,    0xc,    0x40,
      0x3,  0x2,    0x2,    0x2,    0xe,    0x46,   0x3,    0x2,    0x2,
      0x2,  0x10,   0x4f,   0x3,    0x2,    0x2,    0x2,    0x12,   0x51,
      0x3,  0x2,    0x2,    0x2,    0x14,   0x55,   0x3,    0x2,    0x2,
      0x2,  0x16,   0x5d,   0x3,    0x2,    0x2,    0x2,    0x18,   0x5f,
      0x3,  0x2,    0x2,    0x2,    0x1a,   0x1b,   0x5,    0x4,    0x3,
      0x2,  0x1b,   0x1c,   0x7,    0x2,    0x2,    0x3,    0x1c,   0x3,
      0x3,  0x2,    0x2,    0x2,    0x1d,   0x22,   0x5,    0x6,    0x4,
      0x2,  0x1e,   0x1f,   0x7,    0x3,    0x2,    0x2,    0x1f,   0x21,
      0x5,  0x6,    0x4,    0x2,    0x20,   0x1e,   0x3,    0x2,    0x2,
      0x2,  0x21,   0x24,   0x3,    0x2,    0x2,    0x2,    0x22,   0x20,
      0x3,  0x2,    0x2,    0x2,    0x22,   0x23,   0x3,    0x2,    0x2,
      0x2,  0x23,   0x5,    0x3,    0x2,    0x2,    0x2,    0x24,   0x22,
      0x3,  0x2,    0x2,    0x2,    0x25,   0x30,   0x5,    0x8,    0x5,
      0x2,  0x26,   0x28,   0x7,    0x4,    0x2,    0x2,    0x27,   0x29,
      0x7,  0x5,    0x2,    0x2,    0x28,   0x27,   0x3,    0x2,    0x2,
      0x2,  0x28,   0x29,   0x3,    0x2,    0x2,    0x2,    0x29,   0x2c,
      0x3,  0x2,    0x2,    0x2,    0x2a,   0x2c,   0x7,    0x5,    0x2,
      0x2,  0x2b,   0x26,   0x3,    0x2,    0x2,    0x2,    0x2b,   0x2a,
      0x3,  0x2,    0x2,    0x2,    0x2c,   0x2d,   0x3,    0x2,    0x2,
      0x2,  0x2d,   0x2f,   0x5,    0x8,    0x5,    0x2,    0x2e,   0x2b,
      0x3,  0x2,    0x2,    0x2,    0x2f,   0x32,   0x3,    0x2,    0x2,
      0x2,  0x30,   0x2e,   0x3,    0x2,    0x2,    0x2,    0x30,   0x31,
      0x3,  0x2,    0x2,    0x2,    0x31,   0x7,    0x3,    0x2,    0x2,
      0x2,  0x32,   0x30,   0x3,    0x2,    0x2,    0x2,    0x33,   0x35,
      0x5,  0xa,    0x6,    0x2,    0x34,   0x33,   0x3,    0x2,    0x2,
      0x2,  0x35,   0x36,   0x3,    0x2,    0x2,    0x2,    0x36,   0x34,
      0x3,  0x2,    0x2,    0x2,    0x36,   0x37,   0x3,    0x2,    0x2,
      0x2,  0x37,   0x9,    0x3,    0x2,    0x2,    0x2,    0x38,   0x39,
      0x7,  0x6,    0x2,    0x2,    0x39,   0x3e,   0x5,    0xc,    0x7,
      0x2,  0x3a,   0x3b,   0x7,    0x7,    0x2,    0x2,    0x3b,   0x3e,
      0x5,  0xc,    0x7,    0x2,    0x3c,   0x3e,   0x5,    0xc,    0x7,
      0x2,  0x3d,   0x38,   0x3,    0x2,    0x2,    0x2,    0x3d,   0x3a,
      0x3,  0x2,    0x2,    0x2,    0x3d,   0x3c,   0x3,    0x2,    0x2,
      0x2,  0x3e,   0xb,    0x3,    0x2,    0x2,    0x2,    0x3f,   0x41,
      0x5,  0xe,    0x8,    0x2,    0x40,   0x3f,   0x3,    0x2,    0x2,
      0x2,  0x40,   0x41,   0x3,    0x2,    0x2,    0x2,    0x41,   0x42,
      0x3,  0x2,    0x2,    0x2,    0x42,   0x44,   0x5,    0x10,   0x9,
      0x2,  0x43,   0x45,   0x5,    0x12,   0xa,    0x2,    0x44,   0x43,
      0x3,  0x2,    0x2,    0x2,    0x44,   0x45,   0x3,    0x2,    0x2,
      0x2,  0x45,   0xd,    0x3,    0x2,    0x2,    0x2,    0x46,   0x47,
      0x7,  0xd,    0x2,    0x2,    0x47,   0x48,   0x7,    0x8,    0x2,
      0x2,  0x48,   0xf,    0x3,    0x2,    0x2,    0x2,    0x49,   0x50,
      0x5,  0x16,   0xc,    0x2,    0x4a,   0x50,   0x5,    0x18,   0xd,
      0x2,  0x4b,   0x4c,   0x7,    0xa,    0x2,    0x2,    0x4c,   0x4d,
      0x5,  0x4,    0x3,    0x2,    0x4d,   0x4e,   0x7,    0xb,    0x2,
      0x2,  0x4e,   0x50,   0x3,    0x2,    0x2,    0x2,    0x4f,   0x49,
      0x3,  0x2,    0x2,    0x2,    0x4f,   0x4a,   0x3,    0x2,    0x2,
      0x2,  0x4f,   0x4b,   0x3,    0x2,    0x2,    0x2,    0x50,   0x11,
      0x3,  0x2,    0x2,    0x2,    0x51,   0x52,   0x7,    0x9,    0x2,
      0x2,  0x52,   0x53,   0x7,    0xe,    0x2,    0x2,    0x53,   0x13,
      0x3,  0x2,    0x2,    0x2,    0x54,   0x56,   0x7,    0x11,   0x2,
      0x2,  0x55,   0x54,   0x3,    0x2,    0x2,    0x2,    0x56,   0x57,
      0x3,  0x2,    0x2,    0x2,    0x57,   0x55,   0x3,    0x2,    0x2,
      0x2,  0x57,   0x58,   0x3,    0x2,    0x2,    0x2,    0x58,   0x15,
      0x3,  0x2,    0x2,    0x2,    0x59,   0x5e,   0x7,    0xf,    0x2,
      0x2,  0x5a,   0x5e,   0x7,    0xd,    0x2,    0x2,    0x5b,   0x5e,
      0x7,  0xe,    0x2,    0x2,    0x5c,   0x5e,   0x5,    0x14,   0xb,
      0x2,  0x5d,   0x59,   0x3,    0x2,    0x2,    0x2,    0x5d,   0x5a,
      0x3,  0x2,    0x2,    0x2,    0x5d,   0x5b,   0x3,    0x2,    0x2,
      0x2,  0x5d,   0x5c,   0x3,    0x2,    0x2,    0x2,    0x5e,   0x17,
      0x3,  0x2,    0x2,    0x2,    0x5f,   0x60,   0x7,    0xc,    0x2,
      0x2,  0x60,   0x19,   0x3,    0x2,    0x2,    0x2,    0xd,    0x22,
      0x28, 0x2b,   0x30,   0x36,   0x3d,   0x40,   0x44,   0x4f,   0x57,
      0x5d,
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) {
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

FtsParser::Initializer FtsParser::_init;
