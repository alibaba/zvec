package org.zvec.internal.ffm;

import org.zvec.internal.NativeBackend;
import org.zvec.internal.NativeBackendProvider;

public final class FfmNativeBackendProvider implements NativeBackendProvider {
  @Override
  public NativeBackend create() {
    return new FfmNativeBackend();
  }
}
