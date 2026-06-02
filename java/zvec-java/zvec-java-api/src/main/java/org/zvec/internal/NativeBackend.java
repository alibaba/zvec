package org.zvec.internal;

import java.util.List;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;

public interface NativeBackend {
  String id();

  String version();

  void ensureInitialized();

  NativeOpenResult open(String path);

  NativeOpenResult createAndOpen(String path, CollectionSchema schema);

  void close(NativeHandle handle);

  void flush(NativeHandle handle);

  CollectionSchema readSchema(NativeHandle handle);

  int insert(NativeHandle handle, CollectionSchema schema, List<Doc> docs);

  List<Doc> query(
      NativeHandle handle,
      CollectionSchema querySchema,
      CollectionSchema resultSchema,
      VectorQuery query);
}
