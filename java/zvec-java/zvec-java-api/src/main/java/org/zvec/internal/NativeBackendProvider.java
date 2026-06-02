package org.zvec.internal;

@FunctionalInterface
public interface NativeBackendProvider {
  NativeBackend create();
}
