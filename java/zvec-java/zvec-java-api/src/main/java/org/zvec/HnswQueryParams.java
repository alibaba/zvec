package org.zvec;

import java.util.Objects;

public final class HnswQueryParams {
  private final int ef;
  private final float radius;
  private final boolean linear;
  private final boolean usingRefiner;

  public HnswQueryParams(int ef, float radius, boolean linear, boolean usingRefiner) {
    requirePositive(ef, "ef");
    if (radius < 0.0f) {
      throw new IllegalArgumentException("radius must be >= 0");
    }
    this.ef = ef;
    this.radius = radius;
    this.linear = linear;
    this.usingRefiner = usingRefiner;
  }

  public int ef() {
    return ef;
  }

  public float radius() {
    return radius;
  }

  public boolean linear() {
    return linear;
  }

  public boolean usingRefiner() {
    return usingRefiner;
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
    if (!(obj instanceof HnswQueryParams)) {
      return false;
    }
    HnswQueryParams other = (HnswQueryParams) obj;
    return ef == other.ef
        && Float.compare(radius, other.radius) == 0
        && linear == other.linear
        && usingRefiner == other.usingRefiner;
  }

  @Override
  public int hashCode() {
    return Objects.hash(ef, radius, linear, usingRefiner);
  }

  @Override
  public String toString() {
    return "HnswQueryParams[ef="
        + ef
        + ", radius="
        + radius
        + ", linear="
        + linear
        + ", usingRefiner="
        + usingRefiner
        + "]";
  }
}
