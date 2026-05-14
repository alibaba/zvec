package org.zvec.crypto;

import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Whole-collection encryption metadata, mirroring the sidecar JSON shape. */
public final class EncryptionMetadata {
  public static final int VERSION_V1 = 1;

  private final int version;
  private final String collectionName;
  private final Map<String, EncryptionSpec> fields;

  public EncryptionMetadata(int version, String collectionName, Map<String, EncryptionSpec> fields) {
    this.version = version;
    this.collectionName = Objects.requireNonNull(collectionName, "collectionName");
    Objects.requireNonNull(fields, "fields");
    if (version != VERSION_V1) {
      throw new IllegalArgumentException("v1 only supports metadata version=" + VERSION_V1 + ", got " + version);
    }
    this.fields = Map.copyOf(fields);
  }

  public int version() {
    return version;
  }

  public String collectionName() {
    return collectionName;
  }

  public Map<String, EncryptionSpec> fields() {
    return fields;
  }

  public static EncryptionMetadata empty(String collectionName) {
    return new EncryptionMetadata(VERSION_V1, collectionName, Map.of());
  }

  public boolean isEncrypted(String fieldName) {
    return fields.containsKey(fieldName);
  }

  public EncryptionSpec spec(String fieldName) {
    return fields.get(fieldName);
  }

  public Set<String> encryptedFieldNames() {
    return fields.keySet();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof EncryptionMetadata)) {
      return false;
    }
    EncryptionMetadata other = (EncryptionMetadata) obj;
    return version == other.version
        && collectionName.equals(other.collectionName)
        && fields.equals(other.fields);
  }

  @Override
  public int hashCode() {
    return Objects.hash(version, collectionName, fields);
  }

  @Override
  public String toString() {
    return "EncryptionMetadata[version="
        + version
        + ", collectionName="
        + collectionName
        + ", fields="
        + fields
        + "]";
  }
}
