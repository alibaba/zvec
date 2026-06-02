package org.zvec.internal.ffm;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import org.zvec.internal.NativeHandle;

record FfmHandle(MemorySegment segment) implements NativeHandle {
  FfmHandle {
    Objects.requireNonNull(segment, "segment");
  }
}
