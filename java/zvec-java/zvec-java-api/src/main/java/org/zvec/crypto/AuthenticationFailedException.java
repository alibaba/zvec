package org.zvec.crypto;

/** Thrown when AEAD authentication tag verification fails during decryption. */
public final class AuthenticationFailedException extends DecryptionException {

  public AuthenticationFailedException(String message) {
    super(message);
  }

  public AuthenticationFailedException(String message, Throwable cause) {
    super(message, cause);
  }
}
