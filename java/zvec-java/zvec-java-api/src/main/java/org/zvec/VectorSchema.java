package org.zvec;

import java.util.Objects;

public final class VectorSchema {
  private final String name;
  private final DataType dataType;
  private final int dimension;
  private final HnswIndexParams hnswIndexParams;
  private final TuningProfile tuningProfile;
  private final Long expectedDocCount;

  public VectorSchema(String name, DataType dataType, int dimension) {
    this(name, dataType, dimension, null, null, null);
  }

  public VectorSchema(
      String name,
      DataType dataType,
      int dimension,
      HnswIndexParams hnswIndexParams,
      TuningProfile tuningProfile,
      Long expectedDocCount) {
    this.name = Objects.requireNonNull(name, "name");
    this.dataType = Objects.requireNonNull(dataType, "dataType");
    if (!dataType.isVector()) {
      throw new IllegalArgumentException("VectorSchema requires a vector data type");
    }
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be greater than 0");
    }
    if (expectedDocCount != null && expectedDocCount <= 0L) {
      throw new IllegalArgumentException("expectedDocCount must be > 0");
    }
    this.dimension = dimension;
    this.hnswIndexParams = hnswIndexParams;
    this.tuningProfile = tuningProfile;
    this.expectedDocCount = expectedDocCount;
  }

  public String name() {
    return name;
  }

  public DataType dataType() {
    return dataType;
  }

  public int dimension() {
    return dimension;
  }

  public HnswIndexParams hnswIndexParams() {
    return hnswIndexParams;
  }

  public TuningProfile tuningProfile() {
    return tuningProfile;
  }

  public Long expectedDocCount() {
    return expectedDocCount;
  }

  public VectorSchema withHnswIndex(HnswIndexParams params) {
    return new VectorSchema(
        name, dataType, dimension, Objects.requireNonNull(params, "params"), null, null);
  }

  public VectorSchema withTuningProfile(TuningProfile profile) {
    return new VectorSchema(
        name, dataType, dimension, null, Objects.requireNonNull(profile, "profile"), null);
  }

  public VectorSchema withTuningProfile(TuningProfile profile, long expectedDocCount) {
    return new VectorSchema(
        name,
        dataType,
        dimension,
        null,
        Objects.requireNonNull(profile, "profile"),
        expectedDocCount);
  }

  public VectorSchema withExpectedDocCount(long expectedDocCount) {
    return new VectorSchema(name, dataType, dimension, hnswIndexParams, tuningProfile, expectedDocCount);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof VectorSchema)) {
      return false;
    }
    VectorSchema other = (VectorSchema) obj;
    return dimension == other.dimension
        && name.equals(other.name)
        && dataType == other.dataType
        && Objects.equals(hnswIndexParams, other.hnswIndexParams)
        && tuningProfile == other.tuningProfile
        && Objects.equals(expectedDocCount, other.expectedDocCount);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, dataType, dimension, hnswIndexParams, tuningProfile, expectedDocCount);
  }

  @Override
  public String toString() {
    return "VectorSchema[name="
        + name
        + ", dataType="
        + dataType
        + ", dimension="
        + dimension
        + ", hnswIndexParams="
        + hnswIndexParams
        + ", tuningProfile="
        + tuningProfile
        + ", expectedDocCount="
        + expectedDocCount
        + "]";
  }
}
