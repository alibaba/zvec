package org.zvec;

import java.util.Objects;

public final class HnswIndexParams {
  private final int m;
  private final int efConstruction;

  public HnswIndexParams(int m, int efConstruction) {
    requirePositive(m, "m");
    requirePositive(efConstruction, "efConstruction");
    this.m = m;
    this.efConstruction = efConstruction;
  }

  public int m() {
    return m;
  }

  public int efConstruction() {
    return efConstruction;
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0");
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof HnswIndexParams)) {
      return false;
    }
    HnswIndexParams other = (HnswIndexParams) obj;
    return m == other.m && efConstruction == other.efConstruction;
  }

  @Override
  public int hashCode() {
    return Objects.hash(m, efConstruction);
  }

  @Override
  public String toString() {
    return "HnswIndexParams[m=" + m + ", efConstruction=" + efConstruction + "]";
  }
}
