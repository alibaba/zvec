
// Generated from FtsLexer.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"


namespace antlr4 {


class FtsLexer : public antlr4::Lexer {
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

  FtsLexer(antlr4::CharStream *input);
  ~FtsLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string> &getRuleNames() const override;

  virtual const std::vector<std::string> &getChannelNames() const override;
  virtual const std::vector<std::string> &getModeNames() const override;
  virtual const std::vector<std::string> &getTokenNames()
      const override;  // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary &getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN &getATN() const override;

 private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

}  // namespace antlr4
