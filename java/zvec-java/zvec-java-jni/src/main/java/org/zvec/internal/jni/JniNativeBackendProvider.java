package org.zvec.internal.jni;

import org.zvec.internal.NativeBackend;
import org.zvec.internal.NativeBackendProvider;

public final class JniNativeBackendProvider implements NativeBackendProvider {
  @Override
  public NativeBackend create() {
    return new JniNativeBackend();
  }
}
