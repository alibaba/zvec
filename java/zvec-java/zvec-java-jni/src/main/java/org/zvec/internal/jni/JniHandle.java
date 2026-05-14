package org.zvec.internal.jni;

import java.util.Objects;
import org.zvec.internal.NativeHandle;

final class JniHandle implements NativeHandle {
  private final long address;

  JniHandle(long address) {
    if (address == 0L) {
      throw new IllegalArgumentException("address must not be 0");
    }
    this.address = address;
  }

  long address() {
    return address;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof JniHandle)) {
      return false;
    }
    JniHandle other = (JniHandle) obj;
    return address == other.address;
  }

  @Override
  public int hashCode() {
    return Objects.hash(address);
  }

  @Override
  public String toString() {
    return "JniHandle[address=" + address + "]";
  }
}
