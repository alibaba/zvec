package org.zvec.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.stream.Collectors;

public final class NativeBackends {
  public static final String BACKEND_PROPERTY = "org.zvec.backend";

  private NativeBackends() {}

  public static NativeBackend backend() {
    return Holder.BACKEND;
  }

  static NativeBackend resolve(Iterable<NativeBackendProvider> providers, String requestedId) {
    Objects.requireNonNull(providers, "providers");

    List<NativeBackend> backends = new ArrayList<>();
    for (NativeBackendProvider provider : providers) {
      NativeBackend backend = Objects.requireNonNull(provider.create(), "backend");
      backends.add(backend);
    }

    if (backends.isEmpty()) {
      throw new IllegalStateException(
          "No zvec native backend found. Add exactly one backend dependency: zvec-java-jni or zvec-java-ffm.");
    }

    String requested = normalize(requestedId);
    if (requested != null) {
      for (NativeBackend backend : backends) {
        if (backend.id().equals(requested)) {
          return backend;
        }
      }
      throw new IllegalStateException(
          "Requested zvec native backend '"
              + requested
              + "' was not found. Available backends: "
              + backendIds(backends)
              + ".");
    }

    if (backends.size() == 1) {
      return backends.get(0);
    }

    throw new IllegalStateException(
        "Multiple zvec native backends found: "
            + backendIds(backends)
            + ". Set -Dorg.zvec.backend=jni or -Dorg.zvec.backend=ffm.");
  }

  private static String normalize(String requestedId) {
    if (requestedId == null || requestedId.isBlank()) {
      return null;
    }
    return requestedId.trim();
  }

  private static String backendIds(List<NativeBackend> backends) {
    return backends.stream().map(NativeBackend::id).collect(Collectors.joining(", "));
  }

  private static final class Holder {
    private static final NativeBackend BACKEND =
        resolve(ServiceLoader.load(NativeBackendProvider.class), System.getProperty(BACKEND_PROPERTY));
  }
}
