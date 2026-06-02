package org.zvec.crypto;

import java.util.Objects;

/** Parsed envelope. Field positions correspond to the binary layout in EnvelopeCodec. */
public final class Envelope {
  public static final int VERSION_V1 = 0x01;
  public static final int ALG_AES_256_GCM = 0x01;
  public static final int PAYLOAD_STRING = 0x00;
  public static final int NONCE_LEN = 12;

  private final int version;
  private final int alg;
  private final int payloadType;
  private final String keyId;
  private final byte[] nonce;
  private final byte[] ciphertext;

  public Envelope(
      int version,
      int alg,
      int payloadType,
      String keyId,
      byte[] nonce,
      byte[] ciphertext) {
    this.version = version;
    this.alg = alg;
    this.payloadType = payloadType;
    this.keyId = Objects.requireNonNull(keyId, "keyId");
    this.nonce = Objects.requireNonNull(nonce, "nonce");
    this.ciphertext = Objects.requireNonNull(ciphertext, "ciphertext");
  }

  public int version() {
    return version;
  }

  public int alg() {
    return alg;
  }

  public int payloadType() {
    return payloadType;
  }

  public String keyId() {
    return keyId;
  }

  public byte[] nonce() {
    return nonce;
  }

  public byte[] ciphertext() {
    return ciphertext;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Envelope)) {
      return false;
    }
    Envelope other = (Envelope) obj;
    return version == other.version
        && alg == other.alg
        && payloadType == other.payloadType
        && keyId.equals(other.keyId)
        && Objects.equals(nonce, other.nonce)
        && Objects.equals(ciphertext, other.ciphertext);
  }

  @Override
  public int hashCode() {
    return Objects.hash(version, alg, payloadType, keyId, nonce, ciphertext);
  }

  @Override
  public String toString() {
    return "Envelope[version="
        + version
        + ", alg="
        + alg
        + ", payloadType="
        + payloadType
        + ", keyId="
        + keyId
        + ", nonce="
        + nonce
        + ", ciphertext="
        + ciphertext
        + "]";
  }
}
