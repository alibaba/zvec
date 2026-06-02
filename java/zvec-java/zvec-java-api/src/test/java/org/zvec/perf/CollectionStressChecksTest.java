package org.zvec.perf;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.zvec.Doc;

class CollectionStressChecksTest {
  @Test
  void matchesExpectedHitWhenExpectedDocumentIsPresent() {
    assertTrue(
        CollectionStressMain.hasExpectedHit(
            "doc_7", List.of(Doc.result("doc_7", 1.0), Doc.result("doc_9", 0.8))));
    assertTrue(
        CollectionStressMain.hasExpectedHit(
            "doc_7", List.of(Doc.result("doc_9", 1.0), Doc.result("doc_7", 0.8))));
  }

  @Test
  void rejectsEmptyResultsAndMissingExpectedHit() {
    assertFalse(CollectionStressMain.hasExpectedHit("doc_7", List.of()));
    assertFalse(
        CollectionStressMain.hasExpectedHit(
            "doc_7", List.of(Doc.result("doc_9", 1.0), Doc.result("doc_8", 0.8))));
  }
}
