package org.zvec;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class Doc {
  private final String id;
  private Double score;
  private final LinkedHashMap<String, Object> fields = new LinkedHashMap<>();
  private final LinkedHashMap<String, float[]> vectors = new LinkedHashMap<>();
  private final LinkedHashSet<String> nullFields = new LinkedHashSet<>();

  private Doc(String id) {
    this.id = Objects.requireNonNull(id, "id");
  }

  public static Doc of(String id) {
    return new Doc(id);
  }

  public static Doc result(String id, double score) {
    Doc doc = new Doc(id);
    doc.score = score;
    return doc;
  }

  public Doc field(String name, String value) {
    String fieldName = Objects.requireNonNull(name, "name");
    fields.put(fieldName, Objects.requireNonNull(value, "value"));
    vectors.remove(fieldName);
    nullFields.remove(fieldName);
    return this;
  }

  public Doc field(String name, boolean value) {
    String fieldName = Objects.requireNonNull(name, "name");
    fields.put(fieldName, value);
    vectors.remove(fieldName);
    nullFields.remove(fieldName);
    return this;
  }

  public Doc field(String name, long value) {
    String fieldName = Objects.requireNonNull(name, "name");
    fields.put(fieldName, value);
    vectors.remove(fieldName);
    nullFields.remove(fieldName);
    return this;
  }

  public Doc field(String name, double value) {
    String fieldName = Objects.requireNonNull(name, "name");
    fields.put(fieldName, value);
    vectors.remove(fieldName);
    nullFields.remove(fieldName);
    return this;
  }

  public Doc vector(String name, float[] values) {
    String vectorName = Objects.requireNonNull(name, "name");
    Objects.requireNonNull(values, "values");
    fields.remove(vectorName);
    vectors.put(vectorName, values.clone());
    nullFields.remove(vectorName);
    return this;
  }

  public Doc nullField(String name) {
    String fieldName = Objects.requireNonNull(name, "name");
    fields.remove(fieldName);
    vectors.remove(fieldName);
    nullFields.add(fieldName);
    return this;
  }

  public String id() {
    return id;
  }

  public Double score() {
    return score;
  }

  public Map<String, Object> fields() {
    return Collections.unmodifiableMap(fields);
  }

  public Map<String, float[]> vectors() {
    return Collections.unmodifiableMap(vectors);
  }

  public Set<String> nullFields() {
    return Collections.unmodifiableSet(nullFields);
  }
}
