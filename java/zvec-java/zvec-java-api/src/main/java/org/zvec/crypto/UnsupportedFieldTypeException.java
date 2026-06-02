package org.zvec.crypto;

/** Thrown when encryption is requested for a field type that is not supported. */
public final class UnsupportedFieldTypeException extends EncryptionConfigException {

  public UnsupportedFieldTypeException(String message) {
    super(message);
  }
}
