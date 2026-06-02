package org.zvec;

import java.util.Objects;

public final class FieldSchema {
  private final String name;
  private final DataType dataType;
  private final boolean nullable;

  public FieldSchema(String name, DataType dataType, boolean nullable) {
    this.name = Objects.requireNonNull(name, "name");
    this.dataType = Objects.requireNonNull(dataType, "dataType");
    this.nullable = nullable;
    if (dataType.isVector()) {
      throw new IllegalArgumentException("FieldSchema requires a scalar data type");
    }
  }

  public String name() {
    return name;
  }

  public DataType dataType() {
    return dataType;
  }

  public boolean nullable() {
    return nullable;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof FieldSchema)) {
      return false;
    }
    FieldSchema other = (FieldSchema) obj;
    return nullable == other.nullable
        && name.equals(other.name)
        && dataType == other.dataType;
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, dataType, nullable);
  }

  @Override
  public String toString() {
    return "FieldSchema[name="
        + name
        + ", dataType="
        + dataType
        + ", nullable="
        + nullable
        + "]";
  }
}
