package org.zvec;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.zvec.crypto.EncryptionMetadata;
import org.zvec.crypto.KeyProvider;

public final class CollectionSchema {
  private final String name;
  private final List<FieldSchema> fields;
  private final List<VectorSchema> vectors;
  private final Map<String, FieldSchema> fieldByName;
  private final Map<String, VectorSchema> vectorByName;
  private final EncryptionMetadata encryption;
  private final Map<String, KeyProvider> embeddedKeyProviders;

  public CollectionSchema(String name, List<FieldSchema> fields, List<VectorSchema> vectors) {
    this(name, fields, vectors, null);
  }

  public CollectionSchema(
      String name,
      List<FieldSchema> fields,
      List<VectorSchema> vectors,
      EncryptionMetadata encryption) {
    this(name, fields, vectors, encryption, null);
  }

  public CollectionSchema(
      String name,
      List<FieldSchema> fields,
      List<VectorSchema> vectors,
      EncryptionMetadata encryption,
      Map<String, KeyProvider> embeddedKeyProviders) {
    this.name = Objects.requireNonNull(name, "name");
    this.fields = List.copyOf(Objects.requireNonNull(fields, "fields"));
    this.vectors = List.copyOf(Objects.requireNonNull(vectors, "vectors"));
    this.encryption = encryption;
    this.embeddedKeyProviders = embeddedKeyProviders == null ? null : Map.copyOf(embeddedKeyProviders);

    Map<String, FieldSchema> fieldIndex = new HashMap<>();
    for (FieldSchema field : this.fields) {
      FieldSchema previous =
          fieldIndex.put(Objects.requireNonNull(field, "field").name(), field);
      if (previous != null) {
        throw new IllegalArgumentException("Duplicate field name: " + field.name());
      }
    }

    Map<String, VectorSchema> vectorIndex = new HashMap<>();
    for (VectorSchema vector : this.vectors) {
      VectorSchema previous =
          vectorIndex.put(Objects.requireNonNull(vector, "vector").name(), vector);
      if (previous != null) {
        throw new IllegalArgumentException("Duplicate vector name: " + vector.name());
      }
      if (fieldIndex.containsKey(vector.name())) {
        throw new IllegalArgumentException("Duplicate schema field name: " + vector.name());
      }
    }

    this.fieldByName = Map.copyOf(fieldIndex);
    this.vectorByName = Map.copyOf(vectorIndex);
  }

  public String name() {
    return name;
  }

  public List<FieldSchema> fields() {
    return fields;
  }

  public List<VectorSchema> vectors() {
    return vectors;
  }

  public FieldSchema field(String name) {
    return fieldByName.get(name);
  }

  public VectorSchema vector(String name) {
    return vectorByName.get(name);
  }

  public Optional<EncryptionMetadata> encryption() {
    return Optional.ofNullable(encryption);
  }

  public Optional<Map<String, KeyProvider>> embeddedKeyProviders() {
    return Optional.ofNullable(embeddedKeyProviders);
  }
}
