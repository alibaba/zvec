package org.zvec.internal.ffm;

import org.zvec.internal.ZvecException;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.atomic.AtomicBoolean;

public final class FfmNative {
  public static final int ZVEC_OK = 0;
  static final int ZVEC_INDEX_TYPE_HNSW = 1;
  static final int ZVEC_METRIC_TYPE_L2 = 1;
  private static final long VERSION_STRING_MAX_BYTES = 256;
  private static final Linker LINKER = Linker.nativeLinker();
  private static final SymbolLookup LOOKUP;
  private static final AtomicBoolean SHUTDOWN_HOOK_REGISTERED = new AtomicBoolean(false);
  private static final Object INIT_LOCK = new Object();

  private static final MethodHandle ZVEC_GET_VERSION;
  private static final MethodHandle ZVEC_IS_INITIALIZED;
  private static final MethodHandle ZVEC_INITIALIZE;
  private static final MethodHandle ZVEC_SHUTDOWN;
  private static final MethodHandle ZVEC_GET_LAST_ERROR;
  private static final MethodHandle ZVEC_FREE;
  private static final MethodHandle ZVEC_COLLECTION_SCHEMA_CREATE;
  private static final MethodHandle ZVEC_COLLECTION_SCHEMA_DESTROY;
  private static final MethodHandle ZVEC_COLLECTION_SCHEMA_ADD_FIELD;
  private static final MethodHandle ZVEC_COLLECTION_SCHEMA_GET_NAME;
  private static final MethodHandle ZVEC_COLLECTION_SCHEMA_GET_FORWARD_FIELDS;
  private static final MethodHandle ZVEC_COLLECTION_SCHEMA_GET_VECTOR_FIELDS;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_CREATE;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_DESTROY;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_GET_NAME;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_GET_DATA_TYPE;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_IS_NULLABLE;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_GET_DIMENSION;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_GET_INDEX_PARAMS;
  private static final MethodHandle ZVEC_FIELD_SCHEMA_SET_INDEX_PARAMS;
  private static final MethodHandle ZVEC_INDEX_PARAMS_CREATE;
  private static final MethodHandle ZVEC_INDEX_PARAMS_DESTROY;
  private static final MethodHandle ZVEC_INDEX_PARAMS_GET_TYPE;
  private static final MethodHandle ZVEC_INDEX_PARAMS_SET_METRIC_TYPE;
  private static final MethodHandle ZVEC_INDEX_PARAMS_SET_HNSW_PARAMS;
  private static final MethodHandle ZVEC_INDEX_PARAMS_GET_HNSW_M;
  private static final MethodHandle ZVEC_INDEX_PARAMS_GET_HNSW_EF_CONSTRUCTION;
  private static final MethodHandle ZVEC_COLLECTION_CREATE_AND_OPEN;
  private static final MethodHandle ZVEC_COLLECTION_OPEN;
  private static final MethodHandle ZVEC_COLLECTION_CLOSE;
  private static final MethodHandle ZVEC_COLLECTION_FLUSH;
  private static final MethodHandle ZVEC_COLLECTION_GET_SCHEMA;
  private static final MethodHandle ZVEC_COLLECTION_INSERT;
  private static final MethodHandle ZVEC_COLLECTION_QUERY;
  private static final MethodHandle ZVEC_DOC_CREATE;
  private static final MethodHandle ZVEC_DOC_DESTROY;
  private static final MethodHandle ZVEC_DOCS_FREE;
  private static final MethodHandle ZVEC_DOC_SET_PK;
  private static final MethodHandle ZVEC_DOC_SET_FIELD_NULL;
  private static final MethodHandle ZVEC_DOC_ADD_FIELD_BY_VALUE;
  private static final MethodHandle ZVEC_DOC_GET_PK_COPY;
  private static final MethodHandle ZVEC_DOC_GET_SCORE;
  private static final MethodHandle ZVEC_DOC_GET_FIELD_NAMES;
  private static final MethodHandle ZVEC_DOC_IS_FIELD_NULL;
  private static final MethodHandle ZVEC_DOC_GET_FIELD_VALUE_COPY;
  private static final MethodHandle ZVEC_FREE_STR_ARRAY;
  private static final MethodHandle ZVEC_VECTOR_QUERY_CREATE;
  private static final MethodHandle ZVEC_VECTOR_QUERY_DESTROY;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_TOPK;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_FIELD_NAME;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_QUERY_VECTOR;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_FILTER;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_INCLUDE_VECTOR;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_OUTPUT_FIELDS;
  private static final MethodHandle ZVEC_QUERY_PARAMS_HNSW_CREATE;
  private static final MethodHandle ZVEC_QUERY_PARAMS_HNSW_DESTROY;
  private static final MethodHandle ZVEC_VECTOR_QUERY_SET_HNSW_PARAMS;

  static {
    FfmNativeLoader.load();
    LOOKUP = SymbolLookup.loaderLookup();
    ZVEC_GET_VERSION = downcall("zvec_get_version", FunctionDescriptor.of(ADDRESS));
    ZVEC_IS_INITIALIZED = downcall("zvec_is_initialized", FunctionDescriptor.of(JAVA_BOOLEAN));
    ZVEC_INITIALIZE = downcall("zvec_initialize", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_SHUTDOWN = downcall("zvec_shutdown", FunctionDescriptor.of(JAVA_INT));
    ZVEC_GET_LAST_ERROR = downcall("zvec_get_last_error", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_FREE = downcall("zvec_free", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_COLLECTION_SCHEMA_CREATE =
        downcall("zvec_collection_schema_create", FunctionDescriptor.of(ADDRESS, ADDRESS));
    ZVEC_COLLECTION_SCHEMA_DESTROY =
        downcall("zvec_collection_schema_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_COLLECTION_SCHEMA_ADD_FIELD =
        downcall("zvec_collection_schema_add_field", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_COLLECTION_SCHEMA_GET_NAME =
        downcall("zvec_collection_schema_get_name", FunctionDescriptor.of(ADDRESS, ADDRESS));
    ZVEC_COLLECTION_SCHEMA_GET_FORWARD_FIELDS =
        downcall(
            "zvec_collection_schema_get_forward_fields",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    ZVEC_COLLECTION_SCHEMA_GET_VECTOR_FIELDS =
        downcall(
            "zvec_collection_schema_get_vector_fields",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    ZVEC_FIELD_SCHEMA_CREATE =
        downcall(
            "zvec_field_schema_create",
            FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_BOOLEAN, JAVA_INT));
    ZVEC_FIELD_SCHEMA_DESTROY = downcall("zvec_field_schema_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_FIELD_SCHEMA_GET_NAME =
        downcall("zvec_field_schema_get_name", FunctionDescriptor.of(ADDRESS, ADDRESS));
    ZVEC_FIELD_SCHEMA_GET_DATA_TYPE =
        downcall("zvec_field_schema_get_data_type", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_FIELD_SCHEMA_IS_NULLABLE =
        downcall("zvec_field_schema_is_nullable", FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS));
    ZVEC_FIELD_SCHEMA_GET_DIMENSION =
        downcall("zvec_field_schema_get_dimension", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_FIELD_SCHEMA_GET_INDEX_PARAMS =
        downcall("zvec_field_schema_get_index_params", FunctionDescriptor.of(ADDRESS, ADDRESS));
    ZVEC_FIELD_SCHEMA_SET_INDEX_PARAMS =
        downcall("zvec_field_schema_set_index_params", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_INDEX_PARAMS_CREATE =
        downcall("zvec_index_params_create", FunctionDescriptor.of(ADDRESS, JAVA_INT));
    ZVEC_INDEX_PARAMS_DESTROY = downcall("zvec_index_params_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_INDEX_PARAMS_GET_TYPE =
        downcall("zvec_index_params_get_type", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_INDEX_PARAMS_SET_METRIC_TYPE =
        downcall("zvec_index_params_set_metric_type", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    ZVEC_INDEX_PARAMS_SET_HNSW_PARAMS =
        downcall("zvec_index_params_set_hnsw_params", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
    ZVEC_INDEX_PARAMS_GET_HNSW_M =
        downcall("zvec_index_params_get_hnsw_m", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_INDEX_PARAMS_GET_HNSW_EF_CONSTRUCTION =
        downcall("zvec_index_params_get_hnsw_ef_construction", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_COLLECTION_CREATE_AND_OPEN =
        downcall(
            "zvec_collection_create_and_open",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    ZVEC_COLLECTION_OPEN =
        downcall("zvec_collection_open", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    ZVEC_COLLECTION_CLOSE = downcall("zvec_collection_close", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_COLLECTION_FLUSH = downcall("zvec_collection_flush", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    ZVEC_COLLECTION_GET_SCHEMA =
        downcall("zvec_collection_get_schema", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_COLLECTION_INSERT =
        downcall(
            "zvec_collection_insert",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
    ZVEC_COLLECTION_QUERY =
        downcall(
            "zvec_collection_query",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    ZVEC_DOC_CREATE = downcall("zvec_doc_create", FunctionDescriptor.of(ADDRESS));
    ZVEC_DOC_DESTROY = downcall("zvec_doc_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_DOCS_FREE = downcall("zvec_docs_free", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
    ZVEC_DOC_SET_PK = downcall("zvec_doc_set_pk", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
    ZVEC_DOC_SET_FIELD_NULL =
        downcall("zvec_doc_set_field_null", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_DOC_ADD_FIELD_BY_VALUE =
        downcall(
            "zvec_doc_add_field_by_value",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG));
    ZVEC_DOC_GET_PK_COPY = downcall("zvec_doc_get_pk_copy", FunctionDescriptor.of(ADDRESS, ADDRESS));
    ZVEC_DOC_GET_SCORE = downcall("zvec_doc_get_score", FunctionDescriptor.of(JAVA_FLOAT, ADDRESS));
    ZVEC_DOC_GET_FIELD_NAMES =
        downcall("zvec_doc_get_field_names", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    ZVEC_DOC_IS_FIELD_NULL =
        downcall("zvec_doc_is_field_null", FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, ADDRESS));
    ZVEC_DOC_GET_FIELD_VALUE_COPY =
        downcall(
            "zvec_doc_get_field_value_copy",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_FREE_STR_ARRAY = downcall("zvec_free_str_array", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
    ZVEC_VECTOR_QUERY_CREATE = downcall("zvec_vector_query_create", FunctionDescriptor.of(ADDRESS));
    ZVEC_VECTOR_QUERY_DESTROY =
        downcall("zvec_vector_query_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_VECTOR_QUERY_SET_TOPK =
        downcall("zvec_vector_query_set_topk", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    ZVEC_VECTOR_QUERY_SET_FIELD_NAME =
        downcall("zvec_vector_query_set_field_name", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_VECTOR_QUERY_SET_QUERY_VECTOR =
        downcall(
            "zvec_vector_query_set_query_vector",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
    ZVEC_VECTOR_QUERY_SET_FILTER =
        downcall("zvec_vector_query_set_filter", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    ZVEC_VECTOR_QUERY_SET_INCLUDE_VECTOR =
        downcall(
            "zvec_vector_query_set_include_vector",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_BOOLEAN));
    ZVEC_VECTOR_QUERY_SET_OUTPUT_FIELDS =
        downcall(
            "zvec_vector_query_set_output_fields",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
    ZVEC_QUERY_PARAMS_HNSW_CREATE =
        downcall(
            "zvec_query_params_hnsw_create",
            FunctionDescriptor.of(ADDRESS, JAVA_INT, JAVA_FLOAT, JAVA_BOOLEAN, JAVA_BOOLEAN));
    ZVEC_QUERY_PARAMS_HNSW_DESTROY =
        downcall("zvec_query_params_hnsw_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    ZVEC_VECTOR_QUERY_SET_HNSW_PARAMS =
        downcall("zvec_vector_query_set_hnsw_params", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  }

  private FfmNative() {}

  public static String version() {
    try {
      MemorySegment versionPtr = (MemorySegment) ZVEC_GET_VERSION.invokeExact();
      return readUtf8CString(versionPtr, VERSION_STRING_MAX_BYTES);
    } catch (Throwable t) {
      throw new IllegalStateException("Failed to read zvec version", t);
    }
  }

  public static void ensureInitialized() {
    try {
      if ((boolean) ZVEC_IS_INITIALIZED.invokeExact()) {
        return;
      }

      synchronized (INIT_LOCK) {
        if ((boolean) ZVEC_IS_INITIALIZED.invokeExact()) {
          return;
        }

        check((int) ZVEC_INITIALIZE.invokeExact(MemorySegment.NULL), "zvec_initialize");
        if (SHUTDOWN_HOOK_REGISTERED.compareAndSet(false, true)) {
          Runtime.getRuntime().addShutdownHook(new Thread(FfmNative::shutdownQuietly));
        }
      }
    } catch (Throwable t) {
      if (t instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
      throw new IllegalStateException("Failed to initialize zvec", t);
    }
  }

  static void check(int code, String operation) {
    if (code == ZVEC_OK) {
      return;
    }
    throw lastError(operation, code);
  }

  static ZvecException lastError(String operation, int fallbackCode) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outMessage = arena.allocate(ADDRESS);
      int status = (int) ZVEC_GET_LAST_ERROR.invokeExact(outMessage);
      MemorySegment messagePtr = outMessage.get(ADDRESS, 0);
      try {
        if (status != ZVEC_OK || messagePtr.equals(MemorySegment.NULL)) {
          return new ZvecException(fallbackCode, operation + " failed");
        }
        return new ZvecException(fallbackCode, operation + ": " + messagePtr.getString(0));
      } finally {
        free(messagePtr);
      }
    } catch (Throwable t) {
      return new ZvecException(fallbackCode, operation + " failed");
    }
  }

  static MethodHandle handleCollectionCreateAndOpen() {
    return ZVEC_COLLECTION_CREATE_AND_OPEN;
  }

  static MethodHandle handleCollectionOpen() {
    return ZVEC_COLLECTION_OPEN;
  }

  static MethodHandle handleCollectionClose() {
    return ZVEC_COLLECTION_CLOSE;
  }

  static MethodHandle handleCollectionFlush() {
    return ZVEC_COLLECTION_FLUSH;
  }

  static MethodHandle handleCollectionGetSchema() {
    return ZVEC_COLLECTION_GET_SCHEMA;
  }

  static MethodHandle handleCollectionInsert() {
    return ZVEC_COLLECTION_INSERT;
  }

  static MethodHandle handleCollectionQuery() {
    return ZVEC_COLLECTION_QUERY;
  }

  static MethodHandle handleDocCreate() {
    return ZVEC_DOC_CREATE;
  }

  static MethodHandle handleDocDestroy() {
    return ZVEC_DOC_DESTROY;
  }

  static MethodHandle handleDocsFree() {
    return ZVEC_DOCS_FREE;
  }

  static MethodHandle handleDocSetPk() {
    return ZVEC_DOC_SET_PK;
  }

  static MethodHandle handleDocSetFieldNull() {
    return ZVEC_DOC_SET_FIELD_NULL;
  }

  static MethodHandle handleDocAddFieldByValue() {
    return ZVEC_DOC_ADD_FIELD_BY_VALUE;
  }

  static MethodHandle handleDocGetPkCopy() {
    return ZVEC_DOC_GET_PK_COPY;
  }

  static MethodHandle handleDocGetScore() {
    return ZVEC_DOC_GET_SCORE;
  }

  static MethodHandle handleDocGetFieldNames() {
    return ZVEC_DOC_GET_FIELD_NAMES;
  }

  static MethodHandle handleDocIsFieldNull() {
    return ZVEC_DOC_IS_FIELD_NULL;
  }

  static MethodHandle handleDocGetFieldValueCopy() {
    return ZVEC_DOC_GET_FIELD_VALUE_COPY;
  }

  static MethodHandle handleFreeStrArray() {
    return ZVEC_FREE_STR_ARRAY;
  }

  static MethodHandle handleVectorQueryCreate() {
    return ZVEC_VECTOR_QUERY_CREATE;
  }

  static MethodHandle handleVectorQueryDestroy() {
    return ZVEC_VECTOR_QUERY_DESTROY;
  }

  static MethodHandle handleVectorQuerySetTopK() {
    return ZVEC_VECTOR_QUERY_SET_TOPK;
  }

  static MethodHandle handleVectorQuerySetFieldName() {
    return ZVEC_VECTOR_QUERY_SET_FIELD_NAME;
  }

  static MethodHandle handleVectorQuerySetQueryVector() {
    return ZVEC_VECTOR_QUERY_SET_QUERY_VECTOR;
  }

  static MethodHandle handleVectorQuerySetFilter() {
    return ZVEC_VECTOR_QUERY_SET_FILTER;
  }

  static MethodHandle handleVectorQuerySetIncludeVector() {
    return ZVEC_VECTOR_QUERY_SET_INCLUDE_VECTOR;
  }

  static MethodHandle handleVectorQuerySetOutputFields() {
    return ZVEC_VECTOR_QUERY_SET_OUTPUT_FIELDS;
  }

  static MethodHandle handleQueryParamsHnswCreate() {
    return ZVEC_QUERY_PARAMS_HNSW_CREATE;
  }

  static MethodHandle handleQueryParamsHnswDestroy() {
    return ZVEC_QUERY_PARAMS_HNSW_DESTROY;
  }

  static MethodHandle handleVectorQuerySetHnswParams() {
    return ZVEC_VECTOR_QUERY_SET_HNSW_PARAMS;
  }

  static MethodHandle handleCollectionSchemaCreate() {
    return ZVEC_COLLECTION_SCHEMA_CREATE;
  }

  static MethodHandle handleCollectionSchemaDestroy() {
    return ZVEC_COLLECTION_SCHEMA_DESTROY;
  }

  static MethodHandle handleCollectionSchemaAddField() {
    return ZVEC_COLLECTION_SCHEMA_ADD_FIELD;
  }

  static MethodHandle handleCollectionSchemaGetName() {
    return ZVEC_COLLECTION_SCHEMA_GET_NAME;
  }

  static MethodHandle handleCollectionSchemaGetForwardFields() {
    return ZVEC_COLLECTION_SCHEMA_GET_FORWARD_FIELDS;
  }

  static MethodHandle handleCollectionSchemaGetVectorFields() {
    return ZVEC_COLLECTION_SCHEMA_GET_VECTOR_FIELDS;
  }

  static MethodHandle handleFieldSchemaCreate() {
    return ZVEC_FIELD_SCHEMA_CREATE;
  }

  static MethodHandle handleFieldSchemaDestroy() {
    return ZVEC_FIELD_SCHEMA_DESTROY;
  }

  static MethodHandle handleFieldSchemaGetName() {
    return ZVEC_FIELD_SCHEMA_GET_NAME;
  }

  static MethodHandle handleFieldSchemaGetDataType() {
    return ZVEC_FIELD_SCHEMA_GET_DATA_TYPE;
  }

  static MethodHandle handleFieldSchemaIsNullable() {
    return ZVEC_FIELD_SCHEMA_IS_NULLABLE;
  }

  static MethodHandle handleFieldSchemaGetDimension() {
    return ZVEC_FIELD_SCHEMA_GET_DIMENSION;
  }

  static MethodHandle handleFieldSchemaGetIndexParams() {
    return ZVEC_FIELD_SCHEMA_GET_INDEX_PARAMS;
  }

  static MethodHandle handleFieldSchemaSetIndexParams() {
    return ZVEC_FIELD_SCHEMA_SET_INDEX_PARAMS;
  }

  static MethodHandle handleIndexParamsCreate() {
    return ZVEC_INDEX_PARAMS_CREATE;
  }

  static MethodHandle handleIndexParamsDestroy() {
    return ZVEC_INDEX_PARAMS_DESTROY;
  }

  static MethodHandle handleIndexParamsGetType() {
    return ZVEC_INDEX_PARAMS_GET_TYPE;
  }

  static MethodHandle handleIndexParamsSetMetricType() {
    return ZVEC_INDEX_PARAMS_SET_METRIC_TYPE;
  }

  static MethodHandle handleIndexParamsSetHnswParams() {
    return ZVEC_INDEX_PARAMS_SET_HNSW_PARAMS;
  }

  static MethodHandle handleIndexParamsGetHnswM() {
    return ZVEC_INDEX_PARAMS_GET_HNSW_M;
  }

  static MethodHandle handleIndexParamsGetHnswEfConstruction() {
    return ZVEC_INDEX_PARAMS_GET_HNSW_EF_CONSTRUCTION;
  }

  static void free(MemorySegment segment) {
    if (segment == null || segment.equals(MemorySegment.NULL)) {
      return;
    }

    try {
      ZVEC_FREE.invokeExact(segment);
    } catch (Throwable t) {
      if (t instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
      throw new IllegalStateException("Failed to free native memory", t);
    }
  }

  static String readUtf8CString(MemorySegment cString) {
    return readUtf8CString(cString, Long.MAX_VALUE);
  }

  private static String readUtf8CString(MemorySegment cString, long maxBytes) {
    return cString.reinterpret(maxBytes).getString(0);
  }

  private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(LOOKUP.findOrThrow(name), descriptor);
  }

  private static void shutdownQuietly() {
    try {
      if ((boolean) ZVEC_IS_INITIALIZED.invokeExact()) {
        ZVEC_SHUTDOWN.invokeExact();
      }
    } catch (Throwable ignored) {
    }
  }
}
