package org.zvec.internal.ffm;

import org.zvec.internal.ZvecException;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import org.zvec.HnswQueryParams;
import org.zvec.VectorSchema;
import org.zvec.VectorQuery;
import org.zvec.internal.HnswDefaults;

public final class FfmQueries {
  private FfmQueries() {}

  public static MemorySegment toNative(
      VectorSchema runtimeSchema, VectorSchema publicSchema, VectorQuery query) {
    Objects.requireNonNull(runtimeSchema, "runtimeSchema");
    Objects.requireNonNull(publicSchema, "publicSchema");
    Objects.requireNonNull(query, "query");

    MemorySegment nativeQuery = MemorySegment.NULL;
    try (Arena arena = Arena.ofConfined()) {
      nativeQuery = (MemorySegment) FfmNative.handleVectorQueryCreate().invokeExact();
      if (nativeQuery.equals(MemorySegment.NULL)) {
        throw FfmNative.lastError("zvec_vector_query_create", -1);
      }

      FfmNative.check(
          (int)
              FfmNative.handleVectorQuerySetFieldName()
                  .invokeExact(nativeQuery, arena.allocateFrom(query.fieldName())),
          "zvec_vector_query_set_field_name");

      float[] vector = query.queryVector();
      MemorySegment vectorBuffer = arena.allocateFrom(JAVA_FLOAT, vector);
      FfmNative.check(
          (int)
              FfmNative.handleVectorQuerySetQueryVector()
                  .invokeExact(nativeQuery, vectorBuffer, (long) vector.length * Float.BYTES),
          "zvec_vector_query_set_query_vector");

      FfmNative.check(
          (int) FfmNative.handleVectorQuerySetTopK().invokeExact(nativeQuery, query.topK()),
          "zvec_vector_query_set_topk");

      FfmNative.check(
          (int)
              FfmNative.handleVectorQuerySetIncludeVector()
                  .invokeExact(nativeQuery, query.includeVector()),
          "zvec_vector_query_set_include_vector");

      if (query.filter() != null) {
        FfmNative.check(
            (int)
                FfmNative.handleVectorQuerySetFilter()
                    .invokeExact(nativeQuery, arena.allocateFrom(query.filter())),
            "zvec_vector_query_set_filter");
      }

      if (query.outputFieldsSpecified()) {
        List<String> outputFields = query.outputFields();
        if (outputFields.isEmpty()) {
          throw new UnsupportedOperationException(
              "The current native C API cannot represent an explicit empty output field projection");
        }
        MemorySegment fieldsBuffer = arena.allocate(ADDRESS, outputFields.size());
        for (int i = 0; i < outputFields.size(); i++) {
          fieldsBuffer.setAtIndex(ADDRESS, i, arena.allocateFrom(outputFields.get(i)));
        }
        FfmNative.check(
            (int)
                FfmNative.handleVectorQuerySetOutputFields()
                    .invokeExact(nativeQuery, fieldsBuffer, (long) outputFields.size()),
            "zvec_vector_query_set_output_fields");
      }

      attachHnswParams(nativeQuery, runtimeSchema, publicSchema, query);

      return nativeQuery;
    } catch (Throwable t) {
      destroyQuietly(nativeQuery);
      throw propagate("Failed to convert vector query to native", t);
    }
  }

  public static void destroy(MemorySegment query) {
    if (query == null || query.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      FfmNative.handleVectorQueryDestroy().invokeExact(query);
    } catch (Throwable t) {
      throw propagate("Failed to destroy native query", t);
    }
  }

  private static void destroyQuietly(MemorySegment query) {
    if (query == null || query.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      FfmNative.handleVectorQueryDestroy().invokeExact(query);
    } catch (Throwable ignored) {
    }
  }

  private static RuntimeException propagate(String message, Throwable cause) {
    if (cause instanceof RuntimeException runtimeException) {
      return runtimeException;
    }
    return new IllegalStateException(message, cause);
  }

  private static void attachHnswParams(
      MemorySegment nativeQuery,
      VectorSchema runtimeSchema,
      VectorSchema publicSchema,
      VectorQuery query)
      throws Throwable {
    HnswQueryParams params = resolveAttachedHnswParams(runtimeSchema, publicSchema, query);
    if (params == null) {
      return;
    }

    MemorySegment nativeParams =
        (MemorySegment)
            FfmNative.handleQueryParamsHnswCreate()
                .invokeExact(params.ef(), params.radius(), params.linear(), params.usingRefiner());
    if (nativeParams.equals(MemorySegment.NULL)) {
      throw FfmNative.lastError("zvec_query_params_hnsw_create", -1);
    }

    boolean handedOff = false;
    try {
      FfmNative.check(
          (int) FfmNative.handleVectorQuerySetHnswParams().invokeExact(nativeQuery, nativeParams),
          "zvec_vector_query_set_hnsw_params");
      handedOff = true;
    } finally {
      if (!handedOff) {
        FfmNative.handleQueryParamsHnswDestroy().invokeExact(nativeParams);
      }
    }
  }

  static boolean shouldAttachHnswParams(VectorSchema schema, VectorQuery query) {
    Objects.requireNonNull(schema, "schema");
    Objects.requireNonNull(query, "query");
    return schema.hnswIndexParams() != null
        || schema.tuningProfile() != null
        || schema.expectedDocCount() != null;
  }

  static HnswQueryParams resolveAttachedHnswParams(
      VectorSchema runtimeSchema, VectorSchema publicSchema, VectorQuery query) {
    Objects.requireNonNull(runtimeSchema, "runtimeSchema");
    Objects.requireNonNull(publicSchema, "publicSchema");
    Objects.requireNonNull(query, "query");
    if (!shouldAttachHnswParams(runtimeSchema, query)) {
      return null;
    }
    return HnswDefaults.resolveQueryParams(publicSchema, query);
  }
}
