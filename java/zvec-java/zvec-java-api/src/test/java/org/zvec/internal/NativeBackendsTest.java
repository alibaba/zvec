package org.zvec.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;
import org.junit.jupiter.api.Test;

class NativeBackendsTest {
  @Test
  void rejectsMissingProviders() {
    IllegalStateException ex =
        assertThrows(IllegalStateException.class, () -> NativeBackends.resolve(List.of(), null));

    assertEquals(
        "No zvec native backend found. Add exactly one backend dependency: zvec-java-jni or zvec-java-ffm.",
        ex.getMessage());
  }

  @Test
  void acceptsSingleProvider() {
    NativeBackend backend = backend("jni");

    assertEquals(backend, NativeBackends.resolve(List.of(provider(backend)), null));
  }

  @Test
  void rejectsMultipleProvidersWithoutSelection() {
    IllegalStateException ex =
        assertThrows(
            IllegalStateException.class,
            () -> NativeBackends.resolve(List.of(provider(backend("jni")), provider(backend("ffm"))), null));

    assertEquals(
        "Multiple zvec native backends found: jni, ffm. Set -Dorg.zvec.backend=jni or -Dorg.zvec.backend=ffm.",
        ex.getMessage());
  }

  @Test
  void selectsExplicitProvider() {
    NativeBackend jni = backend("jni");
    NativeBackend ffm = backend("ffm");

    assertEquals(ffm, NativeBackends.resolve(List.of(provider(jni), provider(ffm)), "ffm"));
  }

  @Test
  void rejectsUnknownExplicitProvider() {
    IllegalStateException ex =
        assertThrows(
            IllegalStateException.class,
            () -> NativeBackends.resolve(List.of(provider(backend("jni"))), "ffm"));

    assertEquals(
        "Requested zvec native backend 'ffm' was not found. Available backends: jni.",
        ex.getMessage());
  }

  private static NativeBackendProvider provider(NativeBackend backend) {
    return () -> backend;
  }

  private static NativeBackend backend(String id) {
    return new NativeBackend() {
      @Override
      public String id() {
        return id;
      }

      @Override
      public String version() {
        return "test";
      }

      @Override
      public void ensureInitialized() {}

      @Override
      public NativeOpenResult open(String path) {
        throw new UnsupportedOperationException();
      }

      @Override
      public NativeOpenResult createAndOpen(String path, CollectionSchema schema) {
        throw new UnsupportedOperationException();
      }

      @Override
      public void close(NativeHandle handle) {}

      @Override
      public void flush(NativeHandle handle) {}

      @Override
      public CollectionSchema readSchema(NativeHandle handle) {
        throw new UnsupportedOperationException();
      }

      @Override
      public int insert(NativeHandle handle, CollectionSchema schema, java.util.List<Doc> docs) {
        throw new UnsupportedOperationException();
      }

      @Override
      public java.util.List<Doc> query(
          NativeHandle handle,
          CollectionSchema querySchema,
          CollectionSchema resultSchema,
          VectorQuery query) {
        throw new UnsupportedOperationException();
      }
    };
  }
}
