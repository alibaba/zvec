package org.zvec.crypto;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EncryptionExceptionTest {
  @Test
  void hierarchyIsCorrect() {
    assertTrue(EncryptionException.class.isAssignableFrom(EncryptionConfigException.class));
    assertTrue(EncryptionException.class.isAssignableFrom(EncryptionRuntimeException.class));
    assertTrue(EncryptionConfigException.class.isAssignableFrom(EncryptedCollectionException.class));
    assertTrue(EncryptionConfigException.class.isAssignableFrom(EncryptionMetadataMismatchException.class));
    assertTrue(EncryptionConfigException.class.isAssignableFrom(EncryptionMetadataIOException.class));
    assertTrue(EncryptionConfigException.class.isAssignableFrom(UnsupportedFieldTypeException.class));
    assertTrue(EncryptionRuntimeException.class.isAssignableFrom(KeyResolutionException.class));
    assertTrue(EncryptionRuntimeException.class.isAssignableFrom(EncryptionFailedException.class));
    assertTrue(EncryptionRuntimeException.class.isAssignableFrom(DecryptionException.class));
    assertTrue(DecryptionException.class.isAssignableFrom(EnvelopeFormatException.class));
    assertTrue(DecryptionException.class.isAssignableFrom(AuthenticationFailedException.class));
    assertTrue(RuntimeException.class.isAssignableFrom(EncryptionException.class));
  }

  @Test
  void messagesAndCausesPropagate() {
    Throwable cause = new IllegalStateException("inner");
    KeyResolutionException e = new KeyResolutionException("provider returned null for keyId='k1'", cause);
    assertEquals("provider returned null for keyId='k1'", e.getMessage());
    assertSame(cause, e.getCause());
  }
}
