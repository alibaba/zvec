package org.zvec.internal.ffm;

import org.zvec.internal.ZvecException;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.VectorQuery;
import org.zvec.VectorSchema;
import org.zvec.internal.NativeHandle;
import org.zvec.internal.NativeOpenResult;

public final class FfmCollections {
  private FfmCollections() {}

  public static NativeOpenResult createAndOpen(String path, CollectionSchema schema) {
    Objects.requireNonNull(path, "path");
    Objects.requireNonNull(schema, "schema");

    MemorySegment nativeSchema = FfmSchemas.toNative(schema);
    MemorySegment handle = MemorySegment.NULL;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outCollection = arena.allocate(ADDRESS);
      FfmNative.check(
          (int)
              FfmNative.handleCollectionCreateAndOpen()
                  .invokeExact(arena.allocateFrom(path), nativeSchema, MemorySegment.NULL, outCollection),
          "zvec_collection_create_and_open");
      handle = outCollection.get(ADDRESS, 0);
      CollectionSchema querySchema = readSchema(handle);
      return new NativeOpenResult(new FfmHandle(handle), querySchema);
    } catch (Throwable t) {
      closeQuietly(handle);
      throw propagate("Failed to create collection", t);
    } finally {
      FfmSchemas.destroy(nativeSchema);
    }
  }

  public static NativeOpenResult open(String path) {
    Objects.requireNonNull(path, "path");

    MemorySegment handle = MemorySegment.NULL;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outCollection = arena.allocate(ADDRESS);
      FfmNative.check(
          (int)
              FfmNative.handleCollectionOpen()
                  .invokeExact(arena.allocateFrom(path), MemorySegment.NULL, outCollection),
          "zvec_collection_open");
      handle = outCollection.get(ADDRESS, 0);
      CollectionSchema querySchema = readSchema(handle);
      return new NativeOpenResult(new FfmHandle(handle), querySchema);
    } catch (Throwable t) {
      closeQuietly(handle);
      throw propagate("Failed to open collection", t);
    }
  }

  public static void close(NativeHandle handle) {
    try {
      FfmNative.check(
          (int) FfmNative.handleCollectionClose().invokeExact(segment(handle)), "zvec_collection_close");
    } catch (Throwable t) {
      throw propagate("Failed to close collection", t);
    }
  }

  public static void flush(NativeHandle handle) {
    try {
      FfmNative.check(
          (int) FfmNative.handleCollectionFlush().invokeExact(segment(handle)), "zvec_collection_flush");
    } catch (Throwable t) {
      throw propagate("Failed to flush collection", t);
    }
  }

  public static CollectionSchema readSchema(NativeHandle handle) {
    return readSchema(segment(handle));
  }

  private static CollectionSchema readSchema(MemorySegment handle) {
    MemorySegment schemaHandle = MemorySegment.NULL;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outSchema = arena.allocate(ADDRESS);
      FfmNative.check(
          (int) FfmNative.handleCollectionGetSchema().invokeExact(handle, outSchema),
          "zvec_collection_get_schema");
      schemaHandle = outSchema.get(ADDRESS, 0);
      return FfmSchemas.fromNative(schemaHandle);
    } catch (Throwable t) {
      throw propagate("Failed to read collection schema", t);
    } finally {
      FfmSchemas.destroy(schemaHandle);
    }
  }

  public static int insert(NativeHandle collectionHandle, CollectionSchema schema, List<Doc> docs) {
    MemorySegment collectionSegment = segment(collectionHandle);
    Objects.requireNonNull(schema, "schema");
    Objects.requireNonNull(docs, "docs");
    if (docs.isEmpty()) {
      return 0;
    }

    List<MemorySegment> nativeDocs = FfmDocs.toFfmDocs(docs, schema);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment docArray = arena.allocate(ADDRESS, nativeDocs.size());
      for (int i = 0; i < nativeDocs.size(); i++) {
        docArray.setAtIndex(ADDRESS, i, nativeDocs.get(i));
      }

      MemorySegment successCount = arena.allocate(JAVA_LONG);
      MemorySegment errorCount = arena.allocate(JAVA_LONG);
      FfmNative.check(
          (int)
              FfmNative.handleCollectionInsert()
                  .invokeExact(
                      collectionSegment,
                      docArray,
                      (long) nativeDocs.size(),
                      successCount,
                      errorCount),
          "zvec_collection_insert");

      long errors = errorCount.get(JAVA_LONG, 0);
      if (errors != 0L) {
        throw new ZvecException(
            -1, "zvec_collection_insert reported " + errors + " per-document failures");
      }
      return Math.toIntExact(successCount.get(JAVA_LONG, 0));
    } catch (Throwable t) {
      throw propagate("Failed to insert documents", t);
    } finally {
      FfmDocs.destroyAll(nativeDocs);
    }
  }

  public static List<Doc> query(
      NativeHandle collectionHandle,
      CollectionSchema querySchema,
      CollectionSchema resultSchema,
      VectorQuery query) {
    MemorySegment collectionSegment = segment(collectionHandle);
    Objects.requireNonNull(querySchema, "querySchema");
    Objects.requireNonNull(resultSchema, "resultSchema");
    Objects.requireNonNull(query, "query");

    VectorSchema runtimeVectorSchema = querySchema.vector(query.fieldName());
    if (runtimeVectorSchema == null) {
      throw new IllegalArgumentException("Unknown vector field: " + query.fieldName());
    }
    VectorSchema publicVectorSchema = resultSchema.vector(query.fieldName());
    if (publicVectorSchema == null) {
      throw new IllegalArgumentException("Unknown vector field: " + query.fieldName());
    }
    MemorySegment nativeQuery = FfmQueries.toNative(runtimeVectorSchema, publicVectorSchema, query);
    MemorySegment nativeResults = MemorySegment.NULL;
    long resultCount = 0L;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outResults = arena.allocate(ADDRESS);
      MemorySegment outResultCount = arena.allocate(JAVA_LONG);
      FfmNative.check(
          (int)
              FfmNative.handleCollectionQuery()
                  .invokeExact(collectionSegment, nativeQuery, outResults, outResultCount),
          "zvec_collection_query");

      nativeResults = outResults.get(ADDRESS, 0);
      resultCount = outResultCount.get(JAVA_LONG, 0);
      if (resultCount == 0) {
        return List.of();
      }
      return FfmDocs.fromNativeQueryDocs(nativeResults, resultCount, resultSchema);
    } catch (Throwable t) {
      throw propagate("Failed to query documents", t);
    } finally {
      freeFfmDocsQuietly(nativeResults, resultCount);
      FfmQueries.destroy(nativeQuery);
    }
  }

  private static void closeQuietly(MemorySegment handle) {
    if (handle == null || handle.equals(MemorySegment.NULL)) {
      return;
    }

    try {
      FfmNative.check(
          (int) FfmNative.handleCollectionClose().invokeExact(handle), "zvec_collection_close");
    } catch (Throwable ignored) {
    }
  }

  private static void freeFfmDocsQuietly(MemorySegment docs, long count) {
    if (docs == null || docs.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      FfmNative.handleDocsFree().invokeExact(docs, count);
    } catch (Throwable ignored) {
    }
  }

  private static MemorySegment segment(NativeHandle handle) {
    Objects.requireNonNull(handle, "handle");
    if (handle instanceof FfmHandle ffmHandle) {
      return ffmHandle.segment();
    }
    throw new IllegalArgumentException("Handle is not an FFM handle: " + handle.getClass().getName());
  }

  private static RuntimeException propagate(String message, Throwable cause) {
    if (cause instanceof RuntimeException runtimeException) {
      return runtimeException;
    }
    return new IllegalStateException(message, cause);
  }
}
