package org.zvec.internal.ffm;

import org.zvec.internal.ZvecException;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.zvec.HnswIndexParams;
import org.zvec.CollectionSchema;
import org.zvec.DataType;
import org.zvec.FieldSchema;
import org.zvec.internal.HnswDefaults;
import org.zvec.VectorSchema;

public final class FfmSchemas {
  private FfmSchemas() {}

  public static MemorySegment toNative(CollectionSchema schema) {
    Objects.requireNonNull(schema, "schema");

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment schemaHandle =
          (MemorySegment)
              FfmNative.handleCollectionSchemaCreate().invokeExact(arena.allocateFrom(schema.name()));
      if (schemaHandle.equals(MemorySegment.NULL)) {
        throw FfmNative.lastError("zvec_collection_schema_create", -1);
      }

      boolean success = false;
      try {
        for (FieldSchema field : schema.fields()) {
          addField(schemaHandle, fieldHandle(field, arena));
        }
        for (VectorSchema vector : schema.vectors()) {
          addField(schemaHandle, vectorHandle(vector, arena));
        }
        success = true;
        return schemaHandle;
      } finally {
        if (!success) {
          destroy(schemaHandle);
        }
      }
    } catch (Throwable t) {
      throw propagate("Failed to convert collection schema to native", t);
    }
  }

  public static CollectionSchema fromNative(MemorySegment schemaHandle) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment namePtr =
          (MemorySegment) FfmNative.handleCollectionSchemaGetName().invokeExact(schemaHandle);
      String name = FfmNative.readUtf8CString(namePtr);
      return new CollectionSchema(name, readScalarFields(schemaHandle, arena), readVectorFields(schemaHandle, arena));
    } catch (Throwable t) {
      throw propagate("Failed to convert native schema", t);
    }
  }

  public static void destroy(MemorySegment schemaHandle) {
    if (schemaHandle == null || schemaHandle.equals(MemorySegment.NULL)) {
      return;
    }

    try {
      FfmNative.handleCollectionSchemaDestroy().invokeExact(schemaHandle);
    } catch (Throwable t) {
      throw propagate("Failed to destroy native schema", t);
    }
  }

  private static void addField(MemorySegment schemaHandle, MemorySegment fieldHandle) throws Throwable {
    try {
      FfmNative.check(
          (int) FfmNative.handleCollectionSchemaAddField().invokeExact(schemaHandle, fieldHandle),
          "zvec_collection_schema_add_field");
    } finally {
      FfmNative.handleFieldSchemaDestroy().invokeExact(fieldHandle);
    }
  }

  private static MemorySegment fieldHandle(FieldSchema field, Arena arena) throws Throwable {
    return createFieldHandle(field.name(), field.dataType().code(), field.nullable(), 0, arena);
  }

  private static MemorySegment vectorHandle(VectorSchema vector, Arena arena) throws Throwable {
    MemorySegment fieldHandle =
        createFieldHandle(vector.name(), vector.dataType().code(), false, vector.dimension(), arena);
    boolean success = false;
    try {
      applyHnswIndexParams(fieldHandle, HnswDefaults.resolveIndexParams(vector));
      success = true;
      return fieldHandle;
    } finally {
      if (!success) {
        FfmNative.handleFieldSchemaDestroy().invokeExact(fieldHandle);
      }
    }
  }

  private static MemorySegment createFieldHandle(
      String name, int dataTypeCode, boolean nullable, int dimension, Arena arena) throws Throwable {
    MemorySegment fieldHandle =
        (MemorySegment)
            FfmNative.handleFieldSchemaCreate()
                .invokeExact(arena.allocateFrom(name), dataTypeCode, nullable, dimension);
    if (fieldHandle.equals(MemorySegment.NULL)) {
      throw FfmNative.lastError("zvec_field_schema_create", -1);
    }
    return fieldHandle;
  }

  private static void applyHnswIndexParams(MemorySegment fieldHandle, HnswIndexParams params) throws Throwable {
    if (params == null) {
      return;
    }

    MemorySegment indexParams =
        (MemorySegment)
            FfmNative.handleIndexParamsCreate().invokeExact(FfmNative.ZVEC_INDEX_TYPE_HNSW);
    if (indexParams.equals(MemorySegment.NULL)) {
      throw FfmNative.lastError("zvec_index_params_create", -1);
    }

    try {
      FfmNative.check(
          (int)
              FfmNative.handleIndexParamsSetMetricType()
                  .invokeExact(indexParams, FfmNative.ZVEC_METRIC_TYPE_L2),
          "zvec_index_params_set_metric_type");
      FfmNative.check(
          (int)
              FfmNative.handleIndexParamsSetHnswParams()
                  .invokeExact(indexParams, params.m(), params.efConstruction()),
          "zvec_index_params_set_hnsw_params");
      FfmNative.check(
          (int) FfmNative.handleFieldSchemaSetIndexParams().invokeExact(fieldHandle, indexParams),
          "zvec_field_schema_set_index_params");
    } finally {
      FfmNative.handleIndexParamsDestroy().invokeExact(indexParams);
    }
  }

  private static List<FieldSchema> readScalarFields(MemorySegment schemaHandle, Arena arena) throws Throwable {
    MemorySegment outFields = arena.allocate(ADDRESS);
    MemorySegment outCount = arena.allocate(JAVA_LONG);
    FfmNative.check(
        (int)
            FfmNative.handleCollectionSchemaGetForwardFields()
                .invokeExact(schemaHandle, outFields, outCount),
        "zvec_collection_schema_get_forward_fields");
    return readScalarFieldList(outFields.get(ADDRESS, 0), outCount.get(JAVA_LONG, 0));
  }

  private static List<VectorSchema> readVectorFields(MemorySegment schemaHandle, Arena arena) throws Throwable {
    MemorySegment outFields = arena.allocate(ADDRESS);
    MemorySegment outCount = arena.allocate(JAVA_LONG);
    FfmNative.check(
        (int)
            FfmNative.handleCollectionSchemaGetVectorFields()
                .invokeExact(schemaHandle, outFields, outCount),
        "zvec_collection_schema_get_vector_fields");
    return readVectorFieldList(outFields.get(ADDRESS, 0), outCount.get(JAVA_LONG, 0));
  }

  private static List<FieldSchema> readScalarFieldList(MemorySegment arrayPtr, long count) throws Throwable {
    try {
      if (count == 0 || arrayPtr.equals(MemorySegment.NULL)) {
        return List.of();
      }

      MemorySegment array = arrayPtr.reinterpret(count * ADDRESS.byteSize());
      List<FieldSchema> fields = new ArrayList<>(Math.toIntExact(count));
      for (long index = 0; index < count; index++) {
        MemorySegment fieldPtr = array.getAtIndex(ADDRESS, index);
        String name =
            FfmNative.readUtf8CString(
                (MemorySegment) FfmNative.handleFieldSchemaGetName().invokeExact(fieldPtr));
        int dataTypeCode = (int) FfmNative.handleFieldSchemaGetDataType().invokeExact(fieldPtr);
        boolean nullable = (boolean) FfmNative.handleFieldSchemaIsNullable().invokeExact(fieldPtr);
        fields.add(new FieldSchema(name, fromCode(dataTypeCode), nullable));
      }
      return fields;
    } finally {
      FfmNative.free(arrayPtr);
    }
  }

  private static List<VectorSchema> readVectorFieldList(MemorySegment arrayPtr, long count) throws Throwable {
    try {
      if (count == 0 || arrayPtr.equals(MemorySegment.NULL)) {
        return List.of();
      }

      MemorySegment array = arrayPtr.reinterpret(count * ADDRESS.byteSize());
      List<VectorSchema> vectors = new ArrayList<>(Math.toIntExact(count));
      for (long index = 0; index < count; index++) {
        MemorySegment fieldPtr = array.getAtIndex(ADDRESS, index);
        String name =
            FfmNative.readUtf8CString(
                (MemorySegment) FfmNative.handleFieldSchemaGetName().invokeExact(fieldPtr));
        int dataTypeCode = (int) FfmNative.handleFieldSchemaGetDataType().invokeExact(fieldPtr);
        int dimension = (int) FfmNative.handleFieldSchemaGetDimension().invokeExact(fieldPtr);
        vectors.add(
            new VectorSchema(
                name,
                fromCode(dataTypeCode),
                dimension,
                readHnswIndexParams(fieldPtr),
                null,
                null));
      }
      return vectors;
    } finally {
      FfmNative.free(arrayPtr);
    }
  }

  private static HnswIndexParams readHnswIndexParams(MemorySegment fieldPtr) throws Throwable {
    MemorySegment indexParams =
        (MemorySegment) FfmNative.handleFieldSchemaGetIndexParams().invokeExact(fieldPtr);
    if (indexParams.equals(MemorySegment.NULL)) {
      return null;
    }

    int indexType = (int) FfmNative.handleIndexParamsGetType().invokeExact(indexParams);
    if (indexType != FfmNative.ZVEC_INDEX_TYPE_HNSW) {
      return null;
    }

    int m = (int) FfmNative.handleIndexParamsGetHnswM().invokeExact(indexParams);
    int efConstruction =
        (int) FfmNative.handleIndexParamsGetHnswEfConstruction().invokeExact(indexParams);
    return new HnswIndexParams(m, efConstruction);
  }

  private static DataType fromCode(int code) {
    for (DataType value : DataType.values()) {
      if (value.code() == code) {
        return value;
      }
    }
    throw new IllegalArgumentException("Unsupported native data type code: " + code);
  }

  private static RuntimeException propagate(String message, Throwable cause) {
    if (cause instanceof RuntimeException runtimeException) {
      return runtimeException;
    }
    return new IllegalStateException(message, cause);
  }
}
