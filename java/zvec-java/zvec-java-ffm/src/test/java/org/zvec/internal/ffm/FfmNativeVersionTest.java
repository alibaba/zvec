package org.zvec.internal.ffm;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

class FfmNativeVersionTest {
  @Test
  void returnsNonBlankVersionString() {
    assertFalse(FfmNative.version().isBlank());
  }
}
