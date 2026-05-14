package org.zvec;

public enum DataType {
  STRING(2, false),
  BOOL(3, false),
  INT64(5, false),
  DOUBLE(9, false),
  VECTOR_FP32(23, true);

  private final int code;
  private final boolean vector;

  DataType(int code, boolean vector) {
    this.code = code;
    this.vector = vector;
  }

  public int code() {
    return code;
  }

  public boolean isVector() {
    return vector;
  }
}
