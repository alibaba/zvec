package org.zvec;

import java.util.List;
import java.util.Objects;
import org.zvec.crypto.DecryptingProjector;
import org.zvec.crypto.EncryptedSchema;
import org.zvec.crypto.EncryptingInsertor;
import org.zvec.crypto.EncryptionMetadata;
import org.zvec.crypto.EncryptionSpec;
import org.zvec.crypto.SidecarMetadata;
import org.zvec.internal.NativeBackend;
import org.zvec.internal.NativeHandle;

public final class Collection implements AutoCloseable {
  private final NativeBackend backend;
  private final NativeHandle handle;
  private final CollectionSchema schema;
  private final CollectionSchema querySchema;
  private final String collectionPath;
  private boolean closed;
  private EncryptedSchema encryptedSchema = EncryptedSchema.NONE;

  Collection(
      NativeBackend backend,
      NativeHandle handle,
      CollectionSchema schema,
      CollectionSchema querySchema,
      String collectionPath) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.handle = Objects.requireNonNull(handle, "handle");
    this.schema = Objects.requireNonNull(schema, "schema");
    this.querySchema = Objects.requireNonNull(querySchema, "querySchema");
    this.collectionPath = Objects.requireNonNull(collectionPath, "collectionPath");
  }

  void attachEncryption(EncryptedSchema es) {
    this.encryptedSchema = Objects.requireNonNull(es, "encryptedSchema");
  }

  public EncryptedSchema encryptedSchema() { return encryptedSchema; }

  public CollectionSchema schema() {
    return schema;
  }

  public void flush() {
    requireOpen();
    backend.flush(handle);
  }

  public int insert(List<Doc> docs) {
    requireOpen();
    Objects.requireNonNull(docs, "docs");
    List<Doc> toInsert = docs;
    if (encryptedSchema != EncryptedSchema.NONE) {
      toInsert = EncryptingInsertor.transform(docs, encryptedSchema);
    }
    return backend.insert(handle, schema, toInsert);
  }

  public List<Doc> query(VectorQuery query) {
    requireOpen();
    Objects.requireNonNull(query, "query");
    if (encryptedSchema != EncryptedSchema.NONE) {
      checkFilterDoesNotReferenceEncryptedFields(query);
    }
    List<Doc> raw = backend.query(handle, querySchema, schema, query);
    if (encryptedSchema != EncryptedSchema.NONE) {
      raw = DecryptingProjector.transform(raw, encryptedSchema);
    }
    return raw;
  }

  private void checkFilterDoesNotReferenceEncryptedFields(VectorQuery query) {
    String filter = query.filter();
    if (filter == null) return;
    java.util.Set<String> referenced = org.zvec.crypto.FilterFieldScanner.referencedFields(filter);
    java.util.Set<String> encrypted = encryptedSchema.encryptedFieldNames();
    for (String fieldName : referenced) {
      if (encrypted.contains(fieldName)) {
        throw new IllegalArgumentException(
            "filter cannot reference encrypted field '" + fieldName + "'");
      }
    }
  }

  public void setActiveKeyId(String fieldName, String newKeyId) {
    requireOpen();
    Objects.requireNonNull(fieldName, "fieldName");
    Objects.requireNonNull(newKeyId, "newKeyId");
    if (encryptedSchema == EncryptedSchema.NONE
        || !encryptedSchema.isEncrypted(fieldName)) {
      throw new IllegalArgumentException(
          "field '" + fieldName + "' is not declared as encrypted");
    }
    EncryptionMetadata current = encryptedSchema.metadata();
    java.util.LinkedHashMap<String, EncryptionSpec> updated =
        new java.util.LinkedHashMap<>(current.fields());
    EncryptionSpec old = updated.get(fieldName);
    EncryptionSpec next = new EncryptionSpec(
        old.alg(), newKeyId, old.createdAt(), java.time.Instant.now());
    updated.put(fieldName, next);
    EncryptionMetadata fresh = new EncryptionMetadata(
        current.version(), current.collectionName(), updated);

    java.nio.file.Path dir = java.nio.file.Paths.get(collectionPath);
    SidecarMetadata.write(dir, fresh);
    encryptedSchema = EncryptedSchema.reconcile(schema, fresh, encryptedSchema.keyProvider());
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    backend.close(handle);
    closed = true;
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Collection is already closed");
    }
  }
}
