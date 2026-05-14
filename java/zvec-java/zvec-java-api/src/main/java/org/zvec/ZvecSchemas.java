package org.zvec;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.zvec.crypto.EncryptionMetadata;
import org.zvec.crypto.EncryptionSpec;
import org.zvec.crypto.KeyProvider;
import org.zvec.crypto.SingletonKeyProvider;
import org.zvec.crypto.UnsupportedFieldTypeException;

public final class ZvecSchemas {
  private ZvecSchemas() {}

  public static Builder collection(String name) {
    return new Builder(name);
  }

  public static final class Builder {
    private final String name;
    private final List<FieldSchema> fields = new ArrayList<>();
    private final List<VectorSchema> vectors = new ArrayList<>();
    private int activeVectorIndex = -1;
    private String activeFieldName = null;
    private DataType activeFieldType = null;
    private final LinkedHashMap<String, EncryptionSpec> encryptionByField = new LinkedHashMap<>();
    private final LinkedHashMap<String, SingletonKeyProvider> embeddedProvidersByField = new LinkedHashMap<>();

    private Builder(String name) {
      this.name = Objects.requireNonNull(name, "name");
    }

    public Builder string(String name) {
      fields.add(new FieldSchema(name, DataType.STRING, false));
      activeFieldName = name;
      activeFieldType = DataType.STRING;
      return this;
    }

    public Builder bool(String name) {
      fields.add(new FieldSchema(name, DataType.BOOL, false));
      activeFieldName = name;
      activeFieldType = DataType.BOOL;
      return this;
    }

    public Builder int64(String name) {
      fields.add(new FieldSchema(name, DataType.INT64, false));
      activeFieldName = name;
      activeFieldType = DataType.INT64;
      return this;
    }

    public Builder doubleField(String name) {
      fields.add(new FieldSchema(name, DataType.DOUBLE, false));
      activeFieldName = name;
      activeFieldType = DataType.DOUBLE;
      return this;
    }

    public Builder vector(String name, int dimension) {
      vectors.add(new VectorSchema(name, DataType.VECTOR_FP32, dimension));
      activeVectorIndex = vectors.size() - 1;
      activeFieldName = null;
      activeFieldType = null;
      return this;
    }

    public Builder encrypted(String keyId) {
      Objects.requireNonNull(keyId, "keyId");
      if (activeFieldName == null) {
        throw new IllegalStateException("encrypted(...) must follow string(name)");
      }
      if (activeFieldType != DataType.STRING) {
        throw new UnsupportedFieldTypeException(
            "v1 only supports encrypted STRING fields; '" + activeFieldName
            + "' has type " + activeFieldType);
      }
      if (encryptionByField.containsKey(activeFieldName)) {
        throw new IllegalStateException(
            "field '" + activeFieldName + "' already marked encrypted");
      }
      encryptionByField.put(activeFieldName,
          new EncryptionSpec("AES-256-GCM", keyId, Instant.now(), null));
      return this;
    }

    public Builder encrypted(String keyId, byte[] key) {
      Objects.requireNonNull(key, "key");
      encrypted(keyId);   // reuses all checks + spec creation
      embeddedProvidersByField.put(activeFieldName, new SingletonKeyProvider(keyId, key));
      return this;
    }

    public Builder expectedDocCount(long expectedDocCount) {
      VectorSchema vector =
          requireActiveVector("expectedDocCount(...) must follow vector(name, dimension)");
      vectors.set(activeVectorIndex, applyExpectedDocCount(vector, expectedDocCount));
      return this;
    }

    public Builder fast() {
      applyProfile("fast() must follow vector(name, dimension)", TuningProfile.FAST);
      return this;
    }

    public Builder balanced() {
      applyProfile("balanced() must follow vector(name, dimension)", TuningProfile.BALANCED);
      return this;
    }

    public Builder accurate() {
      applyProfile("accurate() must follow vector(name, dimension)", TuningProfile.ACCURATE);
      return this;
    }

    public CollectionSchema build() {
      if (encryptionByField.isEmpty()) {
        return new CollectionSchema(name, fields, vectors);
      }
      EncryptionMetadata meta = new EncryptionMetadata(
          EncryptionMetadata.VERSION_V1, name, encryptionByField);
      Map<String, KeyProvider> embedded =
          embeddedProvidersByField.isEmpty()
              ? null
              : new LinkedHashMap<>(embeddedProvidersByField);
      return new CollectionSchema(name, fields, vectors, meta, embedded);
    }

    private void applyProfile(String errorMessage, TuningProfile profile) {
      VectorSchema vector = requireActiveVector(errorMessage);
      vectors.set(activeVectorIndex, applyProfile(vector, profile));
    }

    private VectorSchema requireActiveVector(String errorMessage) {
      if (activeVectorIndex < 0) {
        throw new IllegalStateException(errorMessage);
      }
      return vectors.get(activeVectorIndex);
    }

    private static VectorSchema applyProfile(VectorSchema vector, TuningProfile profile) {
      return new VectorSchema(
          vector.name(),
          vector.dataType(),
          vector.dimension(),
          null,
          Objects.requireNonNull(profile, "profile"),
          vector.expectedDocCount());
    }

    private static VectorSchema applyExpectedDocCount(VectorSchema vector, long expectedDocCount) {
      return new VectorSchema(
          vector.name(),
          vector.dataType(),
          vector.dimension(),
          null,
          vector.tuningProfile(),
          expectedDocCount);
    }
  }
}
