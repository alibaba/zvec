package org.zvec.crypto;

/** Thrown when an encrypted envelope cannot be parsed due to format errors. */
public final class EnvelopeFormatException extends DecryptionException {

  public EnvelopeFormatException(String message) {
    super(message);
  }
}
