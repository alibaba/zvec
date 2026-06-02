package org.zvec.crypto;

/** Thrown when a collection-level encryption constraint is violated. */
public final class EncryptedCollectionException extends EncryptionConfigException {

  public EncryptedCollectionException(String message) {
    super(message);
  }
}
