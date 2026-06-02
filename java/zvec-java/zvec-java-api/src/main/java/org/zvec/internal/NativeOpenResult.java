package org.zvec.internal;

import java.util.Objects;
import org.zvec.CollectionSchema;

public final class NativeOpenResult {
  private final NativeHandle handle;
  private final CollectionSchema querySchema;

  public NativeOpenResult(NativeHandle handle, CollectionSchema querySchema) {
    this.handle = Objects.requireNonNull(handle, "handle");
    this.querySchema = Objects.requireNonNull(querySchema, "querySchema");
  }

  public NativeHandle handle() {
    return handle;
  }

  public CollectionSchema querySchema() {
    return querySchema;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof NativeOpenResult)) {
      return false;
    }
    NativeOpenResult other = (NativeOpenResult) obj;
    return handle.equals(other.handle) && querySchema.equals(other.querySchema);
  }

  @Override
  public int hashCode() {
    return Objects.hash(handle, querySchema);
  }

  @Override
  public String toString() {
    return "NativeOpenResult[handle=" + handle + ", querySchema=" + querySchema + "]";
  }
}
