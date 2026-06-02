package org.zvec.crypto;

/** Thrown when encryption metadata does not match the expected schema. */
public final class EncryptionMetadataMismatchException extends EncryptionConfigException {

  public EncryptionMetadataMismatchException(String message) {
    super(message);
  }
}
