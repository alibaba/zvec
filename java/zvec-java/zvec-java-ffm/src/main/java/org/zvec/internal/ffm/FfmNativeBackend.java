package org.zvec.internal.ffm;

import java.util.List;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;
import org.zvec.internal.NativeBackend;
import org.zvec.internal.NativeHandle;
import org.zvec.internal.NativeOpenResult;

public final class FfmNativeBackend implements NativeBackend {
  @Override
  public String id() {
    return "ffm";
  }

  @Override
  public String version() {
    return FfmNative.version();
  }

  @Override
  public void ensureInitialized() {
    FfmNative.ensureInitialized();
  }

  @Override
  public NativeOpenResult open(String path) {
    ensureInitialized();
    return FfmCollections.open(path);
  }

  @Override
  public NativeOpenResult createAndOpen(String path, CollectionSchema schema) {
    ensureInitialized();
    return FfmCollections.createAndOpen(path, schema);
  }

  @Override
  public void close(NativeHandle handle) {
    FfmCollections.close(handle);
  }

  @Override
  public void flush(NativeHandle handle) {
    FfmCollections.flush(handle);
  }

  @Override
  public CollectionSchema readSchema(NativeHandle handle) {
    return FfmCollections.readSchema(handle);
  }

  @Override
  public int insert(NativeHandle handle, CollectionSchema schema, List<Doc> docs) {
    return FfmCollections.insert(handle, schema, docs);
  }

  @Override
  public List<Doc> query(
      NativeHandle handle,
      CollectionSchema querySchema,
      CollectionSchema resultSchema,
      VectorQuery query) {
    return FfmCollections.query(handle, querySchema, resultSchema, query);
  }
}
