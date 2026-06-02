package org.zvec.internal.jni;

import java.util.List;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;
import org.zvec.internal.NativeBackend;
import org.zvec.internal.NativeHandle;
import org.zvec.internal.NativeOpenResult;

public final class JniNativeBackend implements NativeBackend {
  @Override
  public String id() {
    return "jni";
  }

  @Override
  public String version() {
    return JniNative.version();
  }

  @Override
  public void ensureInitialized() {
    JniNative.ensureInitialized();
  }

  @Override
  public NativeOpenResult open(String path) {
    ensureInitialized();
    long address = JniNative.open(path);
    JniHandle handle = new JniHandle(address);
    try {
      return new NativeOpenResult(handle, JniNative.readSchema(address));
    } catch (RuntimeException e) {
      close(handle);
      throw e;
    }
  }

  @Override
  public NativeOpenResult createAndOpen(String path, CollectionSchema schema) {
    ensureInitialized();
    long address = JniNative.createAndOpen(path, schema);
    JniHandle handle = new JniHandle(address);
    try {
      return new NativeOpenResult(handle, JniNative.readSchema(address));
    } catch (RuntimeException e) {
      close(handle);
      throw e;
    }
  }

  @Override
  public void close(NativeHandle handle) {
    JniNative.close(address(handle));
  }

  @Override
  public void flush(NativeHandle handle) {
    JniNative.flush(address(handle));
  }

  @Override
  public CollectionSchema readSchema(NativeHandle handle) {
    return JniNative.readSchema(address(handle));
  }

  @Override
  public int insert(NativeHandle handle, CollectionSchema schema, List<Doc> docs) {
    return JniNative.insert(address(handle), schema, docs);
  }

  @Override
  public List<Doc> query(
      NativeHandle handle,
      CollectionSchema querySchema,
      CollectionSchema resultSchema,
      VectorQuery query) {
    return JniNative.query(address(handle), querySchema, resultSchema, query);
  }

  private static long address(NativeHandle handle) {
    if (handle instanceof JniHandle) {
      JniHandle jniHandle = (JniHandle) handle;
      return jniHandle.address();
    }
    throw new IllegalArgumentException("Handle is not a JNI handle: " + handle.getClass().getName());
  }
}
