package org.zvec.crypto;

import java.time.Instant;
import java.util.Objects;

/** Per-field crypto metadata as persisted in the sidecar. */
public final class EncryptionSpec {
  public static final String ALG_AES_256_GCM = "AES-256-GCM";

  private final String alg;
  private final String activeKeyId;
  private final Instant createdAt;
  private final Instant rotatedAt;

  public EncryptionSpec(String alg, String activeKeyId, Instant createdAt, Instant rotatedAt) {
    this.alg = Objects.requireNonNull(alg, "alg");
    this.activeKeyId = Objects.requireNonNull(activeKeyId, "activeKeyId");
    this.createdAt = Objects.requireNonNull(createdAt, "createdAt");
    this.rotatedAt = rotatedAt;
    if (!ALG_AES_256_GCM.equals(alg)) {
      throw new IllegalArgumentException("v1 only supports alg=" + ALG_AES_256_GCM + ", got " + alg);
    }
    if (activeKeyId.isEmpty()) {
      throw new IllegalArgumentException("activeKeyId must not be empty");
    }
  }

  public String alg() {
    return alg;
  }

  public String activeKeyId() {
    return activeKeyId;
  }

  public Instant createdAt() {
    return createdAt;
  }

  public Instant rotatedAt() {
    return rotatedAt;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof EncryptionSpec)) {
      return false;
    }
    EncryptionSpec other = (EncryptionSpec) obj;
    return alg.equals(other.alg)
        && activeKeyId.equals(other.activeKeyId)
        && createdAt.equals(other.createdAt)
        && Objects.equals(rotatedAt, other.rotatedAt);
  }

  @Override
  public int hashCode() {
    return Objects.hash(alg, activeKeyId, createdAt, rotatedAt);
  }

  @Override
  public String toString() {
    return "EncryptionSpec[alg="
        + alg
        + ", activeKeyId="
        + activeKeyId
        + ", createdAt="
        + createdAt
        + ", rotatedAt="
        + rotatedAt
        + "]";
  }
}
