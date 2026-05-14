package org.zvec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class VectorQuery {
  private final String fieldName;
  private final float[] queryVector;
  private int topK = 10;
  private boolean includeVector;
  private String filter;
  private boolean outputFieldsSpecified;
  private List<String> outputFields = List.of();
  private HnswQueryParams hnswQueryParams;
  private TuningProfile tuningProfile;

  private VectorQuery(String fieldName, float[] queryVector) {
    this.fieldName = Objects.requireNonNull(fieldName, "fieldName");
    this.queryVector = Objects.requireNonNull(queryVector, "queryVector").clone();
    if (this.queryVector.length == 0) {
      throw new IllegalArgumentException("queryVector must not be empty");
    }
  }

  public static VectorQuery of(String fieldName, float[] queryVector) {
    return new VectorQuery(fieldName, queryVector);
  }

  public VectorQuery topK(int topK) {
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be positive");
    }
    this.topK = topK;
    return this;
  }

  public VectorQuery outputFields(String... fields) {
    Objects.requireNonNull(fields, "fields");
    List<String> output = new ArrayList<>(fields.length);
    for (String field : fields) {
      output.add(Objects.requireNonNull(field, "field"));
    }
    this.outputFieldsSpecified = true;
    this.outputFields = List.copyOf(output);
    return this;
  }

  public VectorQuery includeVector(boolean includeVector) {
    this.includeVector = includeVector;
    return this;
  }

  public VectorQuery filter(String filter) {
    this.filter = filter;
    return this;
  }

  public VectorQuery hnsw(HnswQueryParams params) {
    this.hnswQueryParams = Objects.requireNonNull(params, "params");
    this.tuningProfile = null;
    return this;
  }

  public VectorQuery withTuningProfile(TuningProfile profile) {
    this.tuningProfile = Objects.requireNonNull(profile, "profile");
    this.hnswQueryParams = null;
    return this;
  }

  public String fieldName() {
    return fieldName;
  }

  public float[] queryVector() {
    return queryVector.clone();
  }

  public int topK() {
    return topK;
  }

  public boolean includeVector() {
    return includeVector;
  }

  public String filter() {
    return filter;
  }

  public boolean outputFieldsSpecified() {
    return outputFieldsSpecified;
  }

  public List<String> outputFields() {
    return outputFields;
  }

  public HnswQueryParams hnswQueryParams() {
    return hnswQueryParams;
  }

  public TuningProfile tuningProfile() {
    return tuningProfile;
  }
}
