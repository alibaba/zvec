package org.zvec;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import org.zvec.crypto.EncryptedSchema;
import org.zvec.crypto.EncryptionMetadata;
import org.zvec.crypto.KeyProvider;
import org.zvec.crypto.SidecarMetadata;
import org.zvec.internal.NativeBackend;
import org.zvec.internal.NativeBackends;
import org.zvec.internal.NativeOpenResult;
import org.zvec.internal.SchemaMetadataStore;

public final class Zvec {
  private Zvec() {}

  public static void ensureInitialized() {
    NativeBackends.backend().ensureInitialized();
  }

  public static Collection createAndOpen(String path, CollectionSchema schema) {
    ensureInitialized();
    Objects.requireNonNull(path, "path");
    Objects.requireNonNull(schema, "schema");

    EncryptionMetadata meta = schema.encryption().orElse(null);
    if (meta != null && !meta.encryptedFieldNames().isEmpty()) {
      java.util.Map<String, KeyProvider> embedded =
          schema.embeddedKeyProviders().orElse(java.util.Map.of());
      if (!embedded.keySet().containsAll(meta.encryptedFieldNames())) {
        throw new org.zvec.crypto.EncryptedCollectionException(
            "schema declares encrypted fields without embedded keys; "
            + "use Zvec.createAndOpen(path, schema, provider)");
      }
      KeyProvider composite = compose(embedded);
      return createAndOpen(path, schema, composite);
    }
    return createBackend(path, schema);
  }

  private static KeyProvider compose(java.util.Map<String, KeyProvider> embedded) {
    return keyId -> {
      for (KeyProvider p : embedded.values()) {
        byte[] k = p.resolve(keyId);
        if (k != null) return k;
      }
      return null;
    };
  }

  public static Collection createAndOpen(String path, CollectionSchema schema, KeyProvider provider) {
    ensureInitialized();
    Objects.requireNonNull(path, "path");
    Objects.requireNonNull(schema, "schema");
    Objects.requireNonNull(provider, "provider");

    EncryptionMetadata meta = schema.encryption().orElse(EncryptionMetadata.empty(schema.name()));

    // Reconcile before native open so misconfig fails fast without creating an orphan directory.
    EncryptedSchema es = meta.encryptedFieldNames().isEmpty()
        ? EncryptedSchema.NONE
        : EncryptedSchema.reconcile(schema, meta, provider);

    // Native open creates the collection directory; sidecars are written into it afterward.
    Collection col = createBackend(path, schema);
    try {
      SidecarMetadata.write(Paths.get(path), meta);
      col.attachEncryption(es);
      return col;
    } catch (RuntimeException e) {
      try { col.close(); } catch (RuntimeException ignored) {}
      throw e;
    }
  }

  public static Collection open(String path) {
    ensureInitialized();
    Collection col = openBackend(Objects.requireNonNull(path, "path"));
    try {
      java.util.Optional<EncryptionMetadata> meta =
          SidecarMetadata.read(java.nio.file.Paths.get(path));
      if (meta.isPresent() && !meta.get().encryptedFieldNames().isEmpty()) {
        throw new org.zvec.crypto.EncryptedCollectionException(
            "collection at '" + path + "' has encrypted fields; use Zvec.openWithKeys");
      }
      return col;
    } catch (RuntimeException e) {
      try { col.close(); } catch (RuntimeException ignored) {}
      throw e;
    }
  }

  public static Collection openWithKeys(String path, KeyProvider provider) {
    Objects.requireNonNull(path, "path");
    Objects.requireNonNull(provider, "provider");
    ensureInitialized();
    Collection col = openBackend(path);
    try {
      java.util.Optional<EncryptionMetadata> meta = SidecarMetadata.read(java.nio.file.Paths.get(path));
      if (meta.isPresent() && !meta.get().encryptedFieldNames().isEmpty()) {
        EncryptedSchema es = EncryptedSchema.reconcile(col.schema(), meta.get(), provider);
        col.attachEncryption(es);
      }
      return col;
    } catch (RuntimeException e) {
      try { col.close(); } catch (RuntimeException ignored) {}
      throw e;
    }
  }

  private static Collection openBackend(String path) {
    NativeBackend backend = NativeBackends.backend();
    NativeOpenResult result = backend.open(path);
    try {
      CollectionSchema querySchema = result.querySchema();
      CollectionSchema publicSchema = SchemaMetadataStore.merge(path, querySchema);
      return new Collection(backend, result.handle(), publicSchema, querySchema, path);
    } catch (RuntimeException e) {
      backend.close(result.handle());
      throw e;
    }
  }

  private static Collection createBackend(String path, CollectionSchema schema) {
    NativeBackend backend = NativeBackends.backend();
    NativeOpenResult result = backend.createAndOpen(path, schema);
    try {
      SchemaMetadataStore.write(path, schema);
      return new Collection(backend, result.handle(), schema, result.querySchema(), path);
    } catch (RuntimeException e) {
      backend.close(result.handle());
      throw e;
    }
  }
}
