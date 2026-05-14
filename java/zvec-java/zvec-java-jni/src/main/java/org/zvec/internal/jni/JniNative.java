package org.zvec.internal.jni;

import java.util.List;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;

final class JniNative {
  static {
    JniNativeLoader.load();
  }

  private JniNative() {}

  static native String version();

  static native void ensureInitialized();

  static native long createAndOpen(String path, CollectionSchema schema);

  static native long open(String path);

  static native void close(long handle);

  static native void flush(long handle);

  static native CollectionSchema readSchema(long handle);

  static native int insert(long handle, CollectionSchema schema, List<Doc> docs);

  static native List<Doc> query(
      long handle,
      CollectionSchema querySchema,
      CollectionSchema resultSchema,
      VectorQuery query);
}
