package org.zvec.internal.jni;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

class JniNativeVersionTest {
  @Test
  void exposesNativeVersion() {
    assertFalse(JniNative.version().isBlank());
  }
}
