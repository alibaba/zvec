package org.zvec.crypto;

import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.zvec.CollectionSchema;
import org.zvec.DataType;
import org.zvec.FieldSchema;

/** Bundle of CollectionSchema + EncryptionMetadata + KeyProvider. Constructed via reconcile. */
public final class EncryptedSchema {

  /** Sentinel: collection has no encrypted fields. */
  public static final EncryptedSchema NONE =
      new EncryptedSchema(null, EncryptionMetadata.empty("__none__"), null);

  private final CollectionSchema schema;
  private final EncryptionMetadata metadata;
  private final KeyProvider keyProvider;

  private EncryptedSchema(CollectionSchema schema, EncryptionMetadata metadata, KeyProvider keyProvider) {
    this.schema = schema;
    this.metadata = metadata;
    this.keyProvider = keyProvider;
  }

  public static EncryptedSchema reconcile(
      CollectionSchema schema, EncryptionMetadata metadata, KeyProvider keyProvider) {
    Objects.requireNonNull(schema, "schema");
    Objects.requireNonNull(metadata, "metadata");

    if (metadata.encryptedFieldNames().isEmpty()) {
      return NONE;
    }
    Objects.requireNonNull(keyProvider, "keyProvider");

    if (!schema.name().equals(metadata.collectionName())) {
      throw new EncryptionMetadataMismatchException(
          "collection name mismatch: schema='" + schema.name()
          + "' sidecar='" + metadata.collectionName() + "'");
    }
    for (Map.Entry<String, EncryptionSpec> e : metadata.fields().entrySet()) {
      String name = e.getKey();
      FieldSchema field = schema.field(name);
      if (field == null) {
        throw new EncryptionMetadataMismatchException(
            "encrypted field '" + name + "' not present in schema");
      }
      if (field.dataType() != DataType.STRING) {
        throw new EncryptionMetadataMismatchException(
            "encrypted field '" + name + "' has type " + field.dataType()
            + " but v1 only supports STRING");
      }
    }
    return new EncryptedSchema(schema, metadata, keyProvider);
  }

  public boolean isEncrypted(String fieldName) {
    return metadata.isEncrypted(fieldName);
  }

  public Set<String> encryptedFieldNames() { return metadata.encryptedFieldNames(); }

  public String activeKeyId(String fieldName) {
    EncryptionSpec spec = metadata.spec(fieldName);
    return spec == null ? null : spec.activeKeyId();
  }

  public CollectionSchema schema() { return schema; }
  public EncryptionMetadata metadata() { return metadata; }
  public KeyProvider keyProvider() { return keyProvider; }
}
