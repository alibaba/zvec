package org.zvec.internal.ffm;

import org.zvec.internal.ZvecException;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.zvec.CollectionSchema;
import org.zvec.Doc;
import org.zvec.FieldSchema;
import org.zvec.VectorSchema;

public final class FfmDocs {
  private FfmDocs() {}

  public static List<MemorySegment> toFfmDocs(List<Doc> docs, CollectionSchema schema) {
    Objects.requireNonNull(docs, "docs");
    Objects.requireNonNull(schema, "schema");

    List<MemorySegment> nativeDocs = new ArrayList<>(docs.size());
    try {
      for (Doc doc : docs) {
        nativeDocs.add(toNativeDoc(Objects.requireNonNull(doc, "doc"), schema));
      }
      return nativeDocs;
    } catch (RuntimeException e) {
      destroyAll(nativeDocs);
      throw e;
    }
  }

  public static MemorySegment toNativeDoc(Doc doc, CollectionSchema schema) {
    Objects.requireNonNull(doc, "doc");
    Objects.requireNonNull(schema, "schema");

    MemorySegment nativeDoc = MemorySegment.NULL;
    try (Arena arena = Arena.ofConfined()) {
      nativeDoc = (MemorySegment) FfmNative.handleDocCreate().invokeExact();
      if (nativeDoc.equals(MemorySegment.NULL)) {
        throw FfmNative.lastError("zvec_doc_create", -1);
      }

      FfmNative.handleDocSetPk().invokeExact(nativeDoc, arena.allocateFrom(doc.id()));

      for (Map.Entry<String, Object> entry : doc.fields().entrySet()) {
        FieldSchema field = schema.field(entry.getKey());
        if (field == null) {
          throw new IllegalArgumentException("Unknown scalar field: " + entry.getKey());
        }
        writeScalarField(nativeDoc, field, entry.getValue(), arena);
      }

      for (String fieldName : doc.nullFields()) {
        FieldSchema field = schema.field(fieldName);
        if (field == null) {
          throw new IllegalArgumentException("Unknown scalar field: " + fieldName);
        }
        if (!field.nullable()) {
          throw new IllegalArgumentException("Field is not nullable: " + fieldName);
        }
        FfmNative.check(
            (int)
                FfmNative.handleDocSetFieldNull()
                    .invokeExact(nativeDoc, arena.allocateFrom(fieldName)),
            "zvec_doc_set_field_null");
      }

      for (Map.Entry<String, float[]> entry : doc.vectors().entrySet()) {
        String vectorName = entry.getKey();
        VectorSchema vector = schema.vector(vectorName);
        if (vector == null) {
          throw new IllegalArgumentException("Unknown vector field: " + vectorName);
        }

        float[] values = Objects.requireNonNull(entry.getValue(), "vector values");
        if (values.length != vector.dimension()) {
          throw new IllegalArgumentException(
              "Vector dimension mismatch for field "
                  + vectorName
                  + ": expected "
                  + vector.dimension()
                  + ", got "
                  + values.length);
        }

        MemorySegment valueSegment = arena.allocateFrom(JAVA_FLOAT, values);
        FfmNative.check(
            (int)
                FfmNative.handleDocAddFieldByValue()
                    .invokeExact(
                        nativeDoc,
                        arena.allocateFrom(vectorName),
                        vector.dataType().code(),
                        valueSegment,
                        (long) values.length * Float.BYTES),
            "zvec_doc_add_field_by_value");
      }

      return nativeDoc;
    } catch (Throwable t) {
      destroyQuietly(nativeDoc);
      throw propagate("Failed to convert doc to native", t);
    }
  }

  public static List<Doc> fromNativeQueryDocs(
      MemorySegment nativeDocs, long count, CollectionSchema schema) {
    Objects.requireNonNull(nativeDocs, "nativeDocs");
    Objects.requireNonNull(schema, "schema");
    if (count == 0) {
      return List.of();
    }

    MemorySegment docsArray = nativeDocs.reinterpret(count * ADDRESS.byteSize());
    List<Doc> docs = new ArrayList<>(Math.toIntExact(count));
    for (long i = 0; i < count; i++) {
      docs.add(fromNativeQueryDoc(docsArray.getAtIndex(ADDRESS, i), schema));
    }
    return docs;
  }

  public static void destroyAll(List<MemorySegment> docs) {
    if (docs == null) {
      return;
    }
    for (MemorySegment doc : docs) {
      destroy(doc);
    }
  }

  public static void destroy(MemorySegment doc) {
    if (doc == null || doc.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      FfmNative.handleDocDestroy().invokeExact(doc);
    } catch (Throwable t) {
      throw propagate("Failed to destroy native doc", t);
    }
  }

  private static Doc fromNativeQueryDoc(MemorySegment nativeDoc, CollectionSchema schema) {
    MemorySegment pkCopy = MemorySegment.NULL;
    try {
      pkCopy = (MemorySegment) FfmNative.handleDocGetPkCopy().invokeExact(nativeDoc);
      String id = pkCopy.equals(MemorySegment.NULL) ? "" : FfmNative.readUtf8CString(pkCopy);
      double score = (float) FfmNative.handleDocGetScore().invokeExact(nativeDoc);

      Doc doc = Doc.result(id, score);
      for (String fieldName : readFieldNames(nativeDoc)) {
        FieldSchema scalarField = schema.field(fieldName);
        if (scalarField != null) {
          readScalarField(nativeDoc, doc, scalarField);
          continue;
        }

        VectorSchema vectorField = schema.vector(fieldName);
        if (vectorField != null) {
          readVectorField(nativeDoc, doc, vectorField);
        }
      }
      return doc;
    } catch (Throwable t) {
      throw propagate("Failed to convert native query doc", t);
    } finally {
      FfmNative.free(pkCopy);
    }
  }

  private static List<String> readFieldNames(MemorySegment nativeDoc) {
    MemorySegment namesArray = MemorySegment.NULL;
    long count = 0L;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outNames = arena.allocate(ADDRESS);
      MemorySegment outCount = arena.allocate(JAVA_LONG);
      FfmNative.check(
          (int) FfmNative.handleDocGetFieldNames().invokeExact(nativeDoc, outNames, outCount),
          "zvec_doc_get_field_names");
      namesArray = outNames.get(ADDRESS, 0);
      count = outCount.get(JAVA_LONG, 0);

      if (count == 0) {
        return List.of();
      }

      MemorySegment names = namesArray.reinterpret(count * ADDRESS.byteSize());
      List<String> fieldNames = new ArrayList<>(Math.toIntExact(count));
      for (long i = 0; i < count; i++) {
        fieldNames.add(FfmNative.readUtf8CString(names.getAtIndex(ADDRESS, i)));
      }
      return fieldNames;
    } catch (Throwable t) {
      throw propagate("Failed to read native doc field names", t);
    } finally {
      freeStrArrayQuietly(namesArray, count);
    }
  }

  private static void writeScalarField(
      MemorySegment nativeDoc, FieldSchema field, Object value, Arena arena) throws Throwable {
    MemorySegment fieldName = arena.allocateFrom(field.name());
    switch (field.dataType()) {
      case STRING -> {
        if (!(value instanceof String stringValue)) {
          throw new IllegalArgumentException("Field " + field.name() + " expects STRING");
        }
        byte[] utf8 = stringValue.getBytes(StandardCharsets.UTF_8);
        MemorySegment valueSegment = arena.allocateFrom(JAVA_BYTE, utf8);
        FfmNative.check(
            (int)
                FfmNative.handleDocAddFieldByValue()
                    .invokeExact(
                        nativeDoc,
                        fieldName,
                        field.dataType().code(),
                        valueSegment,
                        (long) utf8.length),
            "zvec_doc_add_field_by_value");
      }
      case BOOL -> {
        if (!(value instanceof Boolean boolValue)) {
          throw new IllegalArgumentException("Field " + field.name() + " expects BOOL");
        }
        MemorySegment valueSegment = arena.allocate(JAVA_BOOLEAN);
        valueSegment.set(JAVA_BOOLEAN, 0, boolValue);
        FfmNative.check(
            (int)
                FfmNative.handleDocAddFieldByValue()
                    .invokeExact(
                        nativeDoc,
                        fieldName,
                        field.dataType().code(),
                        valueSegment,
                        JAVA_BOOLEAN.byteSize()),
            "zvec_doc_add_field_by_value");
      }
      case INT64 -> {
        if (!(value instanceof Long longValue)) {
          throw new IllegalArgumentException("Field " + field.name() + " expects INT64");
        }
        MemorySegment valueSegment = arena.allocate(JAVA_LONG);
        valueSegment.set(JAVA_LONG, 0, longValue);
        FfmNative.check(
            (int)
                FfmNative.handleDocAddFieldByValue()
                    .invokeExact(
                        nativeDoc,
                        fieldName,
                        field.dataType().code(),
                        valueSegment,
                        JAVA_LONG.byteSize()),
            "zvec_doc_add_field_by_value");
      }
      case DOUBLE -> {
        if (!(value instanceof Double doubleValue)) {
          throw new IllegalArgumentException("Field " + field.name() + " expects DOUBLE");
        }
        MemorySegment valueSegment = arena.allocate(JAVA_DOUBLE);
        valueSegment.set(JAVA_DOUBLE, 0, doubleValue);
        FfmNative.check(
            (int)
                FfmNative.handleDocAddFieldByValue()
                    .invokeExact(
                        nativeDoc,
                        fieldName,
                        field.dataType().code(),
                        valueSegment,
                        JAVA_DOUBLE.byteSize()),
            "zvec_doc_add_field_by_value");
      }
      default -> throw new IllegalArgumentException("Unsupported scalar type: " + field.dataType());
    }
  }

  private static void readScalarField(MemorySegment nativeDoc, Doc target, FieldSchema field) {
    try (Arena arena = Arena.ofConfined()) {
      if (isFieldNull(nativeDoc, field.name(), arena)) {
        target.nullField(field.name());
        return;
      }

      ValueCopy valueCopy =
          readFieldValueCopy(nativeDoc, field.name(), field.dataType().code(), arena);
      try {
        switch (field.dataType()) {
          case STRING -> target.field(field.name(), readString(valueCopy));
          case BOOL -> target.field(field.name(), readBoolean(valueCopy));
          case INT64 -> target.field(field.name(), readInt64(valueCopy));
          case DOUBLE -> target.field(field.name(), readDouble(valueCopy));
          default -> throw new IllegalArgumentException("Unsupported scalar type: " + field.dataType());
        }
      } finally {
        FfmNative.free(valueCopy.valuePtr);
      }
    } catch (Throwable t) {
      throw propagate("Failed to read scalar query field: " + field.name(), t);
    }
  }

  private static boolean isFieldNull(MemorySegment nativeDoc, String fieldName, Arena arena)
      throws Throwable {
    return (boolean)
        FfmNative.handleDocIsFieldNull().invokeExact(nativeDoc, arena.allocateFrom(fieldName));
  }

  private static void readVectorField(MemorySegment nativeDoc, Doc target, VectorSchema vectorField) {
    if (vectorField.dataType() != org.zvec.DataType.VECTOR_FP32) {
      throw new IllegalArgumentException("Unsupported vector type: " + vectorField.dataType());
    }

    try (Arena arena = Arena.ofConfined()) {
      ValueCopy valueCopy =
          readFieldValueCopy(nativeDoc, vectorField.name(), vectorField.dataType().code(), arena);
      try {
        target.vector(vectorField.name(), readFp32Vector(valueCopy));
      } finally {
        FfmNative.free(valueCopy.valuePtr);
      }
    } catch (Throwable t) {
      throw propagate("Failed to read vector query field: " + vectorField.name(), t);
    }
  }

  private static ValueCopy readFieldValueCopy(
      MemorySegment nativeDoc, String fieldName, int dataTypeCode, Arena arena) throws Throwable {
    MemorySegment outValue = arena.allocate(ADDRESS);
    MemorySegment outSize = arena.allocate(JAVA_LONG);
    FfmNative.check(
        (int)
            FfmNative.handleDocGetFieldValueCopy()
                .invokeExact(nativeDoc, arena.allocateFrom(fieldName), dataTypeCode, outValue, outSize),
        "zvec_doc_get_field_value_copy");
    return new ValueCopy(outValue.get(ADDRESS, 0), outSize.get(JAVA_LONG, 0));
  }

  private static String readString(ValueCopy valueCopy) {
    if (valueCopy.valueSize == 0) {
      return "";
    }
    if (valueCopy.valuePtr.equals(MemorySegment.NULL)) {
      throw new IllegalStateException("String value pointer is null");
    }
    byte[] bytes = valueCopy.valuePtr.reinterpret(valueCopy.valueSize).toArray(JAVA_BYTE);
    return new String(bytes, StandardCharsets.UTF_8);
  }

  private static boolean readBoolean(ValueCopy valueCopy) {
    if (valueCopy.valueSize < JAVA_BOOLEAN.byteSize() || valueCopy.valuePtr.equals(MemorySegment.NULL)) {
      throw new IllegalStateException("Invalid BOOL value");
    }
    return valueCopy.valuePtr.get(JAVA_BOOLEAN, 0);
  }

  private static long readInt64(ValueCopy valueCopy) {
    if (valueCopy.valueSize < JAVA_LONG.byteSize() || valueCopy.valuePtr.equals(MemorySegment.NULL)) {
      throw new IllegalStateException("Invalid INT64 value");
    }
    return valueCopy.valuePtr.get(JAVA_LONG, 0);
  }

  private static double readDouble(ValueCopy valueCopy) {
    if (valueCopy.valueSize < JAVA_DOUBLE.byteSize() || valueCopy.valuePtr.equals(MemorySegment.NULL)) {
      throw new IllegalStateException("Invalid DOUBLE value");
    }
    return valueCopy.valuePtr.get(JAVA_DOUBLE, 0);
  }

  private static float[] readFp32Vector(ValueCopy valueCopy) {
    if (valueCopy.valueSize == 0) {
      return new float[0];
    }
    if (valueCopy.valuePtr.equals(MemorySegment.NULL)) {
      throw new IllegalStateException("Vector value pointer is null");
    }
    if (valueCopy.valueSize % Float.BYTES != 0) {
      throw new IllegalStateException("Invalid VECTOR_FP32 byte size: " + valueCopy.valueSize);
    }
    return valueCopy.valuePtr.reinterpret(valueCopy.valueSize).toArray(JAVA_FLOAT);
  }

  private static void destroyQuietly(MemorySegment nativeDoc) {
    if (nativeDoc == null || nativeDoc.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      FfmNative.handleDocDestroy().invokeExact(nativeDoc);
    } catch (Throwable ignored) {
    }
  }

  private static void freeStrArrayQuietly(MemorySegment namesArray, long count) {
    if (namesArray == null || namesArray.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      FfmNative.handleFreeStrArray().invokeExact(namesArray, count);
    } catch (Throwable ignored) {
    }
  }

  private static RuntimeException propagate(String message, Throwable cause) {
    if (cause instanceof RuntimeException runtimeException) {
      return runtimeException;
    }
    return new IllegalStateException(message, cause);
  }

  private static final class ValueCopy {
    private final MemorySegment valuePtr;
    private final long valueSize;

    private ValueCopy(MemorySegment valuePtr, long valueSize) {
      this.valuePtr = valuePtr;
      this.valueSize = valueSize;
    }
  }
}
