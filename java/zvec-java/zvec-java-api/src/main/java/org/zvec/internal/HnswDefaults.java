package org.zvec.internal;

import java.util.Objects;
import org.zvec.HnswIndexParams;
import org.zvec.HnswQueryParams;
import org.zvec.TuningProfile;
import org.zvec.VectorQuery;
import org.zvec.VectorSchema;

public final class HnswDefaults {
  private HnswDefaults() {}

  public static HnswIndexParams resolveIndexParams(VectorSchema schema) {
    Objects.requireNonNull(schema, "schema");
    HnswIndexParams rawParams = schema.hnswIndexParams();
    if (rawParams != null) {
      return rawParams;
    }

    TuningProfile profile =
        schema.tuningProfile() == null ? TuningProfile.BALANCED : schema.tuningProfile();
    Long expectedDocCount = schema.expectedDocCount();
    if (expectedDocCount == null) {
      return indexParamsFor(SizeBucket.SMALL, profile);
    }

    return indexParamsFor(bucket(expectedDocCount), profile);
  }

  public static HnswQueryParams resolveQueryParams(VectorSchema schema, VectorQuery query) {
    Objects.requireNonNull(schema, "schema");
    Objects.requireNonNull(query, "query");

    HnswQueryParams rawParams = query.hnswQueryParams();
    if (rawParams != null) {
      return rawParams;
    }

    TuningProfile profile =
        query.tuningProfile() != null
            ? query.tuningProfile()
            : schema.tuningProfile() != null ? schema.tuningProfile() : TuningProfile.BALANCED;
    Long expectedDocCount = schema.expectedDocCount();
    int ef = queryEfFor(expectedDocCount == null ? SizeBucket.SMALL : bucket(expectedDocCount), profile);
    return new HnswQueryParams(ef, 0.0f, false, false);
  }

  private static HnswIndexParams indexParamsFor(SizeBucket bucket, TuningProfile profile) {
    switch (bucket) {
      case SMALL:
        switch (profile) {
          case FAST:
            return new HnswIndexParams(12, 120);
          case BALANCED:
            return new HnswIndexParams(16, 200);
          case ACCURATE:
            return new HnswIndexParams(24, 300);
        }
        break;
      case MEDIUM:
        switch (profile) {
          case FAST:
            return new HnswIndexParams(16, 200);
          case BALANCED:
            return new HnswIndexParams(24, 300);
          case ACCURATE:
            return new HnswIndexParams(32, 400);
        }
        break;
      case LARGE:
        switch (profile) {
          case FAST:
            return new HnswIndexParams(16, 240);
          case BALANCED:
            return new HnswIndexParams(32, 400);
          case ACCURATE:
            return new HnswIndexParams(40, 500);
        }
        break;
    }
    throw new IllegalStateException("Unhandled size bucket/profile: " + bucket + "/" + profile);
  }

  private static int queryEfFor(SizeBucket bucket, TuningProfile profile) {
    switch (bucket) {
      case SMALL:
        switch (profile) {
          case FAST:
            return 32;
          case BALANCED:
            return 64;
          case ACCURATE:
            return 96;
        }
        break;
      case MEDIUM:
        switch (profile) {
          case FAST:
            return 48;
          case BALANCED:
            return 96;
          case ACCURATE:
            return 128;
        }
        break;
      case LARGE:
        switch (profile) {
          case FAST:
            return 64;
          case BALANCED:
            return 128;
          case ACCURATE:
            return 192;
        }
        break;
    }
    throw new IllegalStateException("Unhandled size bucket/profile: " + bucket + "/" + profile);
  }

  private static SizeBucket bucket(long expectedDocCount) {
    if (expectedDocCount <= 100_000L) {
      return SizeBucket.SMALL;
    }
    if (expectedDocCount <= 1_000_000L) {
      return SizeBucket.MEDIUM;
    }
    return SizeBucket.LARGE;
  }

  private enum SizeBucket {
    SMALL,
    MEDIUM,
    LARGE
  }
}
