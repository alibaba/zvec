package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Set;
import org.junit.jupiter.api.Test;

class FilterFieldScannerTest {
  @Test
  void extractsBareIdentifiers() {
    Set<String> ids = FilterFieldScanner.referencedFields("body = 'foo'");
    assertEquals(Set.of("body"), ids);
  }

  @Test
  void ignoresStringLiterals() {
    Set<String> ids = FilterFieldScanner.referencedFields("title = 'body'");
    assertEquals(Set.of("title"), ids);
  }

  @Test
  void ignoresDoubleQuotedLiterals() {
    Set<String> ids = FilterFieldScanner.referencedFields("title = \"body\"");
    assertEquals(Set.of("title"), ids);
  }

  @Test
  void distinguishesIdentifierFromSubstring() {
    Set<String> ids = FilterFieldScanner.referencedFields("bodyguard = 'x'");
    assertEquals(Set.of("bodyguard"), ids);
  }

  @Test
  void compoundExpressions() {
    Set<String> ids = FilterFieldScanner.referencedFields("body = 'foo' AND title != 'bar' OR rank > 10");
    assertEquals(Set.of("body", "title", "rank", "AND", "OR"), ids);
  }

  @Test
  void emptyFilter() {
    assertTrue(FilterFieldScanner.referencedFields("").isEmpty());
    assertTrue(FilterFieldScanner.referencedFields(null).isEmpty());
  }

  @Test
  void escapedQuoteDoesNotEndLiteral() {
    Set<String> ids = FilterFieldScanner.referencedFields("title = 'O\\'Brien' AND body LIKE 'x'");
    assertEquals(Set.of("title", "AND", "body", "LIKE"), ids);
  }
}
