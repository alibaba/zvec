#include <jni.h>
#include <zvec/c_api.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kIndexTypeHnsw = 1;
constexpr int kMetricTypeL2 = 1;
constexpr int kTypeString = 2;
constexpr int kTypeBool = 3;
constexpr int kTypeInt64 = 5;
constexpr int kTypeDouble = 9;
constexpr int kTypeVectorFp32 = 23;

struct SchemaDeleter {
  void operator()(zvec_collection_schema_t *p) const {
    zvec_collection_schema_destroy(p);
  }
};
struct FieldDeleter {
  void operator()(zvec_field_schema_t *p) const { zvec_field_schema_destroy(p); }
};
struct IndexDeleter {
  void operator()(zvec_index_params_t *p) const { zvec_index_params_destroy(p); }
};
struct DocDeleter {
  void operator()(zvec_doc_t *p) const { zvec_doc_destroy(p); }
};
struct QueryDeleter {
  void operator()(zvec_vector_query_t *p) const { zvec_vector_query_destroy(p); }
};

using SchemaPtr = std::unique_ptr<zvec_collection_schema_t, SchemaDeleter>;
using FieldPtr = std::unique_ptr<zvec_field_schema_t, FieldDeleter>;
using IndexPtr = std::unique_ptr<zvec_index_params_t, IndexDeleter>;
using DocPtr = std::unique_ptr<zvec_doc_t, DocDeleter>;
using QueryPtr = std::unique_ptr<zvec_vector_query_t, QueryDeleter>;

std::string to_string(JNIEnv *env, jstring value) {
  if (value == nullptr) {
    return "";
  }
  const char *chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) {
    return "";
  }
  std::string out(chars);
  env->ReleaseStringUTFChars(value, chars);
  return out;
}

jstring to_jstring(JNIEnv *env, const std::string &value) {
  return env->NewStringUTF(value.c_str());
}

jclass find_class(JNIEnv *env, const char *name) {
  jclass cls = env->FindClass(name);
  if (cls == nullptr) {
    throw std::runtime_error(std::string("Missing Java class: ") + name);
  }
  return cls;
}

jmethodID method(JNIEnv *env, jclass cls, const char *name, const char *sig) {
  jmethodID mid = env->GetMethodID(cls, name, sig);
  if (mid == nullptr) {
    throw std::runtime_error(std::string("Missing Java method: ") + name + sig);
  }
  return mid;
}

jmethodID static_method(JNIEnv *env, jclass cls, const char *name,
                        const char *sig) {
  jmethodID mid = env->GetStaticMethodID(cls, name, sig);
  if (mid == nullptr) {
    throw std::runtime_error(std::string("Missing Java static method: ") + name +
                             sig);
  }
  return mid;
}

jfieldID static_field(JNIEnv *env, jclass cls, const char *name,
                      const char *sig) {
  jfieldID fid = env->GetStaticFieldID(cls, name, sig);
  if (fid == nullptr) {
    throw std::runtime_error(std::string("Missing Java static field: ") + name);
  }
  return fid;
}

void throw_exception(JNIEnv *env, const char *class_name,
                     const std::string &message) {
  if (env->ExceptionCheck()) {
    return;
  }
  jclass cls = env->FindClass(class_name);
  if (cls != nullptr) {
    env->ThrowNew(cls, message.c_str());
  }
}

void throw_zvec(JNIEnv *env, int code, const std::string &message) {
  if (env->ExceptionCheck()) {
    return;
  }
  jclass cls = env->FindClass("org/zvec/internal/ZvecException");
  if (cls == nullptr) {
    throw_exception(env, "java/lang/IllegalStateException", message);
    return;
  }
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
  if (ctor == nullptr) {
    throw_exception(env, "java/lang/IllegalStateException", message);
    return;
  }
  jstring msg = env->NewStringUTF(message.c_str());
  jobject ex = env->NewObject(cls, ctor, static_cast<jint>(code), msg);
  if (ex != nullptr) {
    env->Throw(static_cast<jthrowable>(ex));
  }
}

std::string last_error_message(const char *operation) {
  char *message = nullptr;
  zvec_get_last_error(&message);
  std::string out(operation);
  if (message != nullptr) {
    out += ": ";
    out += message;
    zvec_free(message);
  } else {
    out += " failed";
  }
  return out;
}

bool check(JNIEnv *env, zvec_error_code_t code, const char *operation) {
  if (code == ZVEC_OK) {
    return true;
  }
  throw_zvec(env, static_cast<int>(code), last_error_message(operation));
  return false;
}

int data_type_code(JNIEnv *env, jobject data_type) {
  jclass cls = find_class(env, "org/zvec/DataType");
  return env->CallIntMethod(data_type, method(env, cls, "code", "()I"));
}

jobject data_type_for_code(JNIEnv *env, int code) {
  jclass cls = find_class(env, "org/zvec/DataType");
  const char *name = nullptr;
  switch (code) {
    case kTypeString:
      name = "STRING";
      break;
    case kTypeBool:
      name = "BOOL";
      break;
    case kTypeInt64:
      name = "INT64";
      break;
    case kTypeDouble:
      name = "DOUBLE";
      break;
    case kTypeVectorFp32:
      name = "VECTOR_FP32";
      break;
    default:
      throw std::runtime_error("Unsupported native data type code: " +
                               std::to_string(code));
  }
  return env->GetStaticObjectField(
      cls, static_field(env, cls, name, "Lorg/zvec/DataType;"));
}

int list_size(JNIEnv *env, jobject list) {
  jclass cls = find_class(env, "java/util/List");
  return env->CallIntMethod(list, method(env, cls, "size", "()I"));
}

jobject list_get(JNIEnv *env, jobject list, int index) {
  jclass cls = find_class(env, "java/util/List");
  return env->CallObjectMethod(list,
                               method(env, cls, "get", "(I)Ljava/lang/Object;"),
                               static_cast<jint>(index));
}

void list_add(JNIEnv *env, jobject list, jobject value) {
  jclass cls = find_class(env, "java/util/List");
  env->CallBooleanMethod(list,
                         method(env, cls, "add", "(Ljava/lang/Object;)Z"),
                         value);
}

jobject new_array_list(JNIEnv *env) {
  jclass cls = find_class(env, "java/util/ArrayList");
  return env->NewObject(cls, method(env, cls, "<init>", "()V"));
}

class Iterator {
 public:
  Iterator(JNIEnv *env, jobject iterable) : env_(env) {
    jobject iterator_obj = env_->CallObjectMethod(
        iterable, method(env_, find_class(env_, "java/lang/Iterable"),
                         "iterator", "()Ljava/util/Iterator;"));
    iterator_ = iterator_obj;
    iterator_class_ = find_class(env_, "java/util/Iterator");
  }

  bool has_next() {
    return env_->CallBooleanMethod(
        iterator_, method(env_, iterator_class_, "hasNext", "()Z"));
  }

  jobject next() {
    return env_->CallObjectMethod(
        iterator_, method(env_, iterator_class_, "next", "()Ljava/lang/Object;"));
  }

 private:
  JNIEnv *env_;
  jobject iterator_;
  jclass iterator_class_;
};

jobject map_entry_set(JNIEnv *env, jobject map) {
  return env->CallObjectMethod(
      map, method(env, find_class(env, "java/util/Map"), "entrySet",
                  "()Ljava/util/Set;"));
}

jobject entry_key(JNIEnv *env, jobject entry) {
  return env->CallObjectMethod(
      entry, method(env, find_class(env, "java/util/Map$Entry"), "getKey",
                    "()Ljava/lang/Object;"));
}

jobject entry_value(JNIEnv *env, jobject entry) {
  return env->CallObjectMethod(
      entry, method(env, find_class(env, "java/util/Map$Entry"), "getValue",
                    "()Ljava/lang/Object;"));
}

void apply_hnsw_index(JNIEnv *env, zvec_field_schema_t *field,
                      jobject vector_schema) {
  jclass defaults_cls = find_class(env, "org/zvec/internal/HnswDefaults");
  jobject params = env->CallStaticObjectMethod(
      defaults_cls,
      static_method(env, defaults_cls, "resolveIndexParams",
                    "(Lorg/zvec/VectorSchema;)Lorg/zvec/HnswIndexParams;"),
      vector_schema);
  if (env->ExceptionCheck() || params == nullptr) {
    return;
  }

  jclass params_cls = find_class(env, "org/zvec/HnswIndexParams");
  int m = env->CallIntMethod(params, method(env, params_cls, "m", "()I"));
  int ef = env->CallIntMethod(
      params, method(env, params_cls, "efConstruction", "()I"));

  IndexPtr index(zvec_index_params_create(kIndexTypeHnsw));
  if (!index) {
    throw std::runtime_error(last_error_message("zvec_index_params_create"));
  }
  if (!check(env,
             zvec_index_params_set_metric_type(index.get(), kMetricTypeL2),
             "zvec_index_params_set_metric_type")) {
    return;
  }
  if (!check(env, zvec_index_params_set_hnsw_params(index.get(), m, ef),
             "zvec_index_params_set_hnsw_params")) {
    return;
  }
  check(env, zvec_field_schema_set_index_params(field, index.get()),
        "zvec_field_schema_set_index_params");
}

SchemaPtr schema_to_native(JNIEnv *env, jobject schema) {
  jclass schema_cls = find_class(env, "org/zvec/CollectionSchema");
  std::string name =
      to_string(env, static_cast<jstring>(env->CallObjectMethod(
                         schema, method(env, schema_cls, "name",
                                        "()Ljava/lang/String;"))));
  SchemaPtr native_schema(zvec_collection_schema_create(name.c_str()));
  if (!native_schema) {
    throw std::runtime_error(last_error_message("zvec_collection_schema_create"));
  }

  jobject fields = env->CallObjectMethod(
      schema, method(env, schema_cls, "fields", "()Ljava/util/List;"));
  jclass field_cls = find_class(env, "org/zvec/FieldSchema");
  for (int i = 0; i < list_size(env, fields); i++) {
    jobject field = list_get(env, fields, i);
    std::string field_name =
        to_string(env, static_cast<jstring>(env->CallObjectMethod(
                           field, method(env, field_cls, "name",
                                         "()Ljava/lang/String;"))));
    jobject data_type = env->CallObjectMethod(
        field, method(env, field_cls, "dataType", "()Lorg/zvec/DataType;"));
    jboolean nullable =
        env->CallBooleanMethod(field, method(env, field_cls, "nullable", "()Z"));
    FieldPtr native_field(zvec_field_schema_create(
        field_name.c_str(), data_type_code(env, data_type), nullable, 0));
    if (!native_field) {
      throw std::runtime_error(last_error_message("zvec_field_schema_create"));
    }
    if (!check(env,
               zvec_collection_schema_add_field(native_schema.get(),
                                                native_field.get()),
               "zvec_collection_schema_add_field")) {
      return nullptr;
    }
  }

  jobject vectors = env->CallObjectMethod(
      schema, method(env, schema_cls, "vectors", "()Ljava/util/List;"));
  jclass vector_cls = find_class(env, "org/zvec/VectorSchema");
  for (int i = 0; i < list_size(env, vectors); i++) {
    jobject vector = list_get(env, vectors, i);
    std::string vector_name =
        to_string(env, static_cast<jstring>(env->CallObjectMethod(
                           vector, method(env, vector_cls, "name",
                                          "()Ljava/lang/String;"))));
    jobject data_type = env->CallObjectMethod(
        vector,
        method(env, vector_cls, "dataType", "()Lorg/zvec/DataType;"));
    jint dimension =
        env->CallIntMethod(vector, method(env, vector_cls, "dimension", "()I"));
    FieldPtr native_field(zvec_field_schema_create(
        vector_name.c_str(), data_type_code(env, data_type), false,
        static_cast<uint32_t>(dimension)));
    if (!native_field) {
      throw std::runtime_error(last_error_message("zvec_field_schema_create"));
    }
    apply_hnsw_index(env, native_field.get(), vector);
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    if (!check(env,
               zvec_collection_schema_add_field(native_schema.get(),
                                                native_field.get()),
               "zvec_collection_schema_add_field")) {
      return nullptr;
    }
  }

  return native_schema;
}

jobject hnsw_index_params_to_java(JNIEnv *env, const zvec_field_schema_t *field) {
  const zvec_index_params_t *params = zvec_field_schema_get_index_params(field);
  if (params == nullptr || zvec_index_params_get_type(params) != kIndexTypeHnsw) {
    return nullptr;
  }
  jclass cls = find_class(env, "org/zvec/HnswIndexParams");
  return env->NewObject(cls, method(env, cls, "<init>", "(II)V"),
                        zvec_index_params_get_hnsw_m(params),
                        zvec_index_params_get_hnsw_ef_construction(params));
}

jobject schema_from_native(JNIEnv *env, zvec_collection_schema_t *schema) {
  jclass field_cls = find_class(env, "org/zvec/FieldSchema");
  jclass vector_cls = find_class(env, "org/zvec/VectorSchema");
  jclass schema_cls = find_class(env, "org/zvec/CollectionSchema");

  jobject fields = new_array_list(env);
  zvec_field_schema_t **field_array = nullptr;
  size_t field_count = 0;
  if (!check(env,
             zvec_collection_schema_get_forward_fields(schema, &field_array,
                                                       &field_count),
             "zvec_collection_schema_get_forward_fields")) {
    return nullptr;
  }
  for (size_t i = 0; i < field_count; i++) {
    zvec_field_schema_t *field = field_array[i];
    jobject data_type =
        data_type_for_code(env, zvec_field_schema_get_data_type(field));
    jobject java_field = env->NewObject(
        field_cls,
        method(env, field_cls, "<init>",
               "(Ljava/lang/String;Lorg/zvec/DataType;Z)V"),
        to_jstring(env, zvec_field_schema_get_name(field)), data_type,
        zvec_field_schema_is_nullable(field));
    list_add(env, fields, java_field);
  }
  zvec_free(field_array);

  jobject vectors = new_array_list(env);
  zvec_field_schema_t **vector_array = nullptr;
  size_t vector_count = 0;
  if (!check(env,
             zvec_collection_schema_get_vector_fields(schema, &vector_array,
                                                      &vector_count),
             "zvec_collection_schema_get_vector_fields")) {
    return nullptr;
  }
  for (size_t i = 0; i < vector_count; i++) {
    zvec_field_schema_t *vector = vector_array[i];
    jobject data_type =
        data_type_for_code(env, zvec_field_schema_get_data_type(vector));
    jobject hnsw = hnsw_index_params_to_java(env, vector);
    jobject java_vector = env->NewObject(
        vector_cls,
        method(env, vector_cls, "<init>",
               "(Ljava/lang/String;Lorg/zvec/DataType;ILorg/zvec/"
               "HnswIndexParams;Lorg/zvec/TuningProfile;Ljava/lang/Long;)V"),
        to_jstring(env, zvec_field_schema_get_name(vector)), data_type,
        static_cast<jint>(zvec_field_schema_get_dimension(vector)), hnsw,
        nullptr, nullptr);
    list_add(env, vectors, java_vector);
  }
  zvec_free(vector_array);

  return env->NewObject(
      schema_cls,
      method(env, schema_cls, "<init>",
             "(Ljava/lang/String;Ljava/util/List;Ljava/util/List;)V"),
      to_jstring(env, zvec_collection_schema_get_name(schema)), fields, vectors);
}

zvec_collection_t *handle(jlong address) {
  if (address == 0) {
    throw std::invalid_argument("Native collection handle is 0");
  }
  return reinterpret_cast<zvec_collection_t *>(address);
}

int field_data_type(JNIEnv *env, jobject schema, const std::string &name) {
  jclass schema_cls = find_class(env, "org/zvec/CollectionSchema");
  jobject field = env->CallObjectMethod(
      schema,
      method(env, schema_cls, "field",
             "(Ljava/lang/String;)Lorg/zvec/FieldSchema;"),
      to_jstring(env, name));
  if (field != nullptr) {
    return data_type_code(env, env->CallObjectMethod(
                                   field,
                                   method(env, find_class(env, "org/zvec/FieldSchema"),
                                          "dataType", "()Lorg/zvec/DataType;")));
  }
  jobject vector = env->CallObjectMethod(
      schema,
      method(env, schema_cls, "vector",
             "(Ljava/lang/String;)Lorg/zvec/VectorSchema;"),
      to_jstring(env, name));
  if (vector != nullptr) {
    return data_type_code(env, env->CallObjectMethod(
                                   vector,
                                   method(env, find_class(env, "org/zvec/VectorSchema"),
                                          "dataType", "()Lorg/zvec/DataType;")));
  }
  throw std::invalid_argument("Unknown field: " + name);
}

void add_doc_field(JNIEnv *env, zvec_doc_t *doc, const std::string &name,
                   int type, jobject value) {
  if (type == kTypeString) {
    std::string string_value = to_string(env, static_cast<jstring>(value));
    check(env, zvec_doc_add_field_by_value(doc, name.c_str(), type,
                                           string_value.data(),
                                           string_value.size()),
          "zvec_doc_add_field_by_value");
    return;
  }
  if (type == kTypeBool) {
    bool bool_value = env->CallBooleanMethod(
        value, method(env, find_class(env, "java/lang/Boolean"),
                      "booleanValue", "()Z"));
    check(env, zvec_doc_add_field_by_value(doc, name.c_str(), type, &bool_value,
                                           sizeof(bool_value)),
          "zvec_doc_add_field_by_value");
    return;
  }
  if (type == kTypeInt64) {
    int64_t long_value = env->CallLongMethod(
        value,
        method(env, find_class(env, "java/lang/Long"), "longValue", "()J"));
    check(env, zvec_doc_add_field_by_value(doc, name.c_str(), type, &long_value,
                                           sizeof(long_value)),
          "zvec_doc_add_field_by_value");
    return;
  }
  if (type == kTypeDouble) {
    double double_value = env->CallDoubleMethod(
        value, method(env, find_class(env, "java/lang/Double"), "doubleValue",
                      "()D"));
    check(env, zvec_doc_add_field_by_value(doc, name.c_str(), type,
                                           &double_value, sizeof(double_value)),
          "zvec_doc_add_field_by_value");
    return;
  }
  throw std::invalid_argument("Unsupported scalar field type: " +
                              std::to_string(type));
}

jobject vector_schema(JNIEnv *env, jobject schema, const std::string &name);
int vector_dimension(JNIEnv *env, jobject schema, const std::string &name);

DocPtr doc_to_native(JNIEnv *env, jobject doc, jobject schema) {
  jclass doc_cls = find_class(env, "org/zvec/Doc");
  DocPtr native_doc(zvec_doc_create());
  if (!native_doc) {
    throw std::runtime_error(last_error_message("zvec_doc_create"));
  }

  std::string id =
      to_string(env, static_cast<jstring>(env->CallObjectMethod(
                         doc, method(env, doc_cls, "id", "()Ljava/lang/String;"))));
  zvec_doc_set_pk(native_doc.get(), id.c_str());

  jobject fields = env->CallObjectMethod(
      doc, method(env, doc_cls, "fields", "()Ljava/util/Map;"));
  for (Iterator it(env, map_entry_set(env, fields)); it.has_next();) {
    jobject entry = it.next();
    std::string name = to_string(env, static_cast<jstring>(entry_key(env, entry)));
    add_doc_field(env, native_doc.get(), name, field_data_type(env, schema, name),
                  entry_value(env, entry));
    if (env->ExceptionCheck()) {
      return nullptr;
    }
  }

  jobject null_fields = env->CallObjectMethod(
      doc, method(env, doc_cls, "nullFields", "()Ljava/util/Set;"));
  for (Iterator it(env, null_fields); it.has_next();) {
    std::string name = to_string(env, static_cast<jstring>(it.next()));
    if (!check(env, zvec_doc_set_field_null(native_doc.get(), name.c_str()),
               "zvec_doc_set_field_null")) {
      return nullptr;
    }
  }

  jobject vectors = env->CallObjectMethod(
      doc, method(env, doc_cls, "vectors", "()Ljava/util/Map;"));
  for (Iterator it(env, map_entry_set(env, vectors)); it.has_next();) {
    jobject entry = it.next();
    std::string name = to_string(env, static_cast<jstring>(entry_key(env, entry)));
    jfloatArray vector_array = static_cast<jfloatArray>(entry_value(env, entry));
    jsize length = env->GetArrayLength(vector_array);
    int expected_dimension = vector_dimension(env, schema, name);
    if (length != expected_dimension) {
      throw std::invalid_argument(
          "Vector dimension mismatch for field " + name + ": expected " +
          std::to_string(expected_dimension) + ", got " + std::to_string(length));
    }
    std::vector<float> values(length);
    env->GetFloatArrayRegion(vector_array, 0, length, values.data());
    if (!check(env,
               zvec_doc_add_field_by_value(native_doc.get(), name.c_str(),
                                           kTypeVectorFp32, values.data(),
                                           values.size() * sizeof(float)),
               "zvec_doc_add_field_by_value")) {
      return nullptr;
    }
  }

  return native_doc;
}

std::vector<DocPtr> docs_to_native(JNIEnv *env, jobject docs, jobject schema) {
  std::vector<DocPtr> out;
  int count = list_size(env, docs);
  out.reserve(count);
  for (int i = 0; i < count; i++) {
    out.push_back(doc_to_native(env, list_get(env, docs, i), schema));
    if (env->ExceptionCheck()) {
      break;
    }
  }
  return out;
}

jobject vector_schema(JNIEnv *env, jobject schema, const std::string &name) {
  return env->CallObjectMethod(
      schema,
      method(env, find_class(env, "org/zvec/CollectionSchema"), "vector",
             "(Ljava/lang/String;)Lorg/zvec/VectorSchema;"),
      to_jstring(env, name));
}

int vector_dimension(JNIEnv *env, jobject schema, const std::string &name) {
  jobject vector = vector_schema(env, schema, name);
  if (vector == nullptr) {
    throw std::invalid_argument("Unknown vector field: " + name);
  }
  return env->CallIntMethod(
      vector, method(env, find_class(env, "org/zvec/VectorSchema"),
                     "dimension", "()I"));
}

void attach_hnsw_query(JNIEnv *env, zvec_vector_query_t *native_query,
                       jobject public_vector_schema, jobject query) {
  jclass defaults_cls = find_class(env, "org/zvec/internal/HnswDefaults");
  jobject params = env->CallStaticObjectMethod(
      defaults_cls,
      static_method(env, defaults_cls, "resolveQueryParams",
                    "(Lorg/zvec/VectorSchema;Lorg/zvec/VectorQuery;)"
                    "Lorg/zvec/HnswQueryParams;"),
      public_vector_schema, query);
  if (env->ExceptionCheck() || params == nullptr) {
    return;
  }
  jclass params_cls = find_class(env, "org/zvec/HnswQueryParams");
  int ef = env->CallIntMethod(params, method(env, params_cls, "ef", "()I"));
  float radius =
      env->CallFloatMethod(params, method(env, params_cls, "radius", "()F"));
  jboolean linear =
      env->CallBooleanMethod(params, method(env, params_cls, "linear", "()Z"));
  jboolean refiner = env->CallBooleanMethod(
      params, method(env, params_cls, "usingRefiner", "()Z"));

  zvec_hnsw_query_params_t *raw =
      zvec_query_params_hnsw_create(ef, radius, linear, refiner);
  if (raw == nullptr) {
    throw std::runtime_error(last_error_message("zvec_query_params_hnsw_create"));
  }
  if (check(env, zvec_vector_query_set_hnsw_params(native_query, raw),
            "zvec_vector_query_set_hnsw_params")) {
    raw = nullptr;
  }
  if (raw != nullptr) {
    zvec_query_params_hnsw_destroy(raw);
  }
}

QueryPtr query_to_native(JNIEnv *env, jobject query, jobject query_schema,
                         jobject result_schema) {
  jclass query_cls = find_class(env, "org/zvec/VectorQuery");
  QueryPtr native_query(zvec_vector_query_create());
  if (!native_query) {
    throw std::runtime_error(last_error_message("zvec_vector_query_create"));
  }

  std::string field_name =
      to_string(env, static_cast<jstring>(env->CallObjectMethod(
                         query, method(env, query_cls, "fieldName",
                                       "()Ljava/lang/String;"))));
  if (!check(env,
             zvec_vector_query_set_field_name(native_query.get(),
                                              field_name.c_str()),
             "zvec_vector_query_set_field_name")) {
    return nullptr;
  }

  jfloatArray query_vector = static_cast<jfloatArray>(env->CallObjectMethod(
      query, method(env, query_cls, "queryVector", "()[F")));
  jsize length = env->GetArrayLength(query_vector);
  std::vector<float> values(length);
  env->GetFloatArrayRegion(query_vector, 0, length, values.data());
  if (!check(env,
             zvec_vector_query_set_query_vector(native_query.get(),
                                                values.data(),
                                                values.size() * sizeof(float)),
             "zvec_vector_query_set_query_vector")) {
    return nullptr;
  }

  if (!check(env,
             zvec_vector_query_set_topk(
                 native_query.get(),
                 env->CallIntMethod(query, method(env, query_cls, "topK", "()I"))),
             "zvec_vector_query_set_topk")) {
    return nullptr;
  }

  if (!check(env,
             zvec_vector_query_set_include_vector(
                 native_query.get(),
                 env->CallBooleanMethod(query,
                                        method(env, query_cls, "includeVector",
                                               "()Z"))),
             "zvec_vector_query_set_include_vector")) {
    return nullptr;
  }

  jstring filter = static_cast<jstring>(env->CallObjectMethod(
      query, method(env, query_cls, "filter", "()Ljava/lang/String;")));
  std::string filter_value;
  if (filter != nullptr) {
    filter_value = to_string(env, filter);
    if (!check(env,
               zvec_vector_query_set_filter(native_query.get(),
                                            filter_value.c_str()),
               "zvec_vector_query_set_filter")) {
      return nullptr;
    }
  }

  if (env->CallBooleanMethod(
          query, method(env, query_cls, "outputFieldsSpecified", "()Z"))) {
    jobject output_fields = env->CallObjectMethod(
        query, method(env, query_cls, "outputFields", "()Ljava/util/List;"));
    int count = list_size(env, output_fields);
    if (count == 0) {
      throw std::runtime_error(
          "The current native C API cannot represent an explicit empty output field projection");
    }
    std::vector<std::string> names;
    std::vector<const char *> ptrs;
    names.reserve(count);
    ptrs.reserve(count);
    for (int i = 0; i < count; i++) {
      names.push_back(to_string(env, static_cast<jstring>(list_get(env, output_fields, i))));
    }
    for (const std::string &name : names) {
      ptrs.push_back(name.c_str());
    }
    if (!check(env,
               zvec_vector_query_set_output_fields(native_query.get(),
                                                   ptrs.data(), ptrs.size()),
               "zvec_vector_query_set_output_fields")) {
      return nullptr;
    }
  }

  jobject runtime_vector = vector_schema(env, query_schema, field_name);
  jobject public_vector = vector_schema(env, result_schema, field_name);
  if (runtime_vector == nullptr || public_vector == nullptr) {
    throw std::invalid_argument("Unknown vector field: " + field_name);
  }
  attach_hnsw_query(env, native_query.get(), public_vector, query);
  return native_query;
}

jobject doc_from_native(JNIEnv *env, zvec_doc_t *doc, jobject schema) {
  jclass doc_cls = find_class(env, "org/zvec/Doc");
  char *pk = const_cast<char *>(zvec_doc_get_pk_copy(doc));
  std::string id = pk == nullptr ? "" : pk;
  zvec_free(pk);

  jobject result = env->CallStaticObjectMethod(
      doc_cls,
      static_method(env, doc_cls, "result",
                    "(Ljava/lang/String;D)Lorg/zvec/Doc;"),
      to_jstring(env, id), static_cast<jdouble>(zvec_doc_get_score(doc)));

  char **names = nullptr;
  size_t count = 0;
  if (!check(env, zvec_doc_get_field_names(doc, &names, &count),
             "zvec_doc_get_field_names")) {
    return nullptr;
  }

  for (size_t i = 0; i < count; i++) {
    std::string name = names[i];
    int type = field_data_type(env, schema, name);
    if (type != kTypeVectorFp32 && zvec_doc_is_field_null(doc, name.c_str())) {
      env->CallObjectMethod(
          result, method(env, doc_cls, "nullField",
                         "(Ljava/lang/String;)Lorg/zvec/Doc;"),
          to_jstring(env, name));
      continue;
    }

    void *value = nullptr;
    size_t size = 0;
    if (!check(env,
               zvec_doc_get_field_value_copy(doc, name.c_str(), type, &value,
                                             &size),
               "zvec_doc_get_field_value_copy")) {
      zvec_free_str_array(names, count);
      return nullptr;
    }
    if (type == kTypeString) {
      std::string text(static_cast<char *>(value), size);
      env->CallObjectMethod(
          result,
          method(env, doc_cls, "field",
                 "(Ljava/lang/String;Ljava/lang/String;)Lorg/zvec/Doc;"),
          to_jstring(env, name), to_jstring(env, text));
    } else if (type == kTypeBool) {
      env->CallObjectMethod(
          result, method(env, doc_cls, "field",
                         "(Ljava/lang/String;Z)Lorg/zvec/Doc;"),
          to_jstring(env, name), *static_cast<bool *>(value));
    } else if (type == kTypeInt64) {
      env->CallObjectMethod(
          result, method(env, doc_cls, "field",
                         "(Ljava/lang/String;J)Lorg/zvec/Doc;"),
          to_jstring(env, name), *static_cast<int64_t *>(value));
    } else if (type == kTypeDouble) {
      env->CallObjectMethod(
          result, method(env, doc_cls, "field",
                         "(Ljava/lang/String;D)Lorg/zvec/Doc;"),
          to_jstring(env, name), *static_cast<double *>(value));
    } else if (type == kTypeVectorFp32) {
      jsize length = static_cast<jsize>(size / sizeof(float));
      jfloatArray vector = env->NewFloatArray(length);
      env->SetFloatArrayRegion(vector, 0, length,
                               static_cast<const jfloat *>(value));
      env->CallObjectMethod(
          result,
          method(env, doc_cls, "vector",
                 "(Ljava/lang/String;[F)Lorg/zvec/Doc;"),
          to_jstring(env, name), vector);
    }
    zvec_free(value);
  }
  zvec_free_str_array(names, count);
  return result;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_zvec_internal_jni_JniNative_version(JNIEnv *env, jclass) {
  return to_jstring(env, zvec_get_version());
}

extern "C" JNIEXPORT void JNICALL
Java_org_zvec_internal_jni_JniNative_ensureInitialized(JNIEnv *env, jclass) {
  if (zvec_is_initialized()) {
    return;
  }
  check(env, zvec_initialize(nullptr), "zvec_initialize");
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_zvec_internal_jni_JniNative_createAndOpen(JNIEnv *env, jclass,
                                                   jstring path,
                                                   jobject schema) {
  try {
    SchemaPtr native_schema = schema_to_native(env, schema);
    if (env->ExceptionCheck()) {
      return 0;
    }
    zvec_collection_t *collection = nullptr;
    std::string path_string = to_string(env, path);
    if (!check(env,
               zvec_collection_create_and_open(path_string.c_str(),
                                               native_schema.get(), nullptr,
                                               &collection),
               "zvec_collection_create_and_open")) {
      return 0;
    }
    return reinterpret_cast<jlong>(collection);
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
    return 0;
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_zvec_internal_jni_JniNative_open(JNIEnv *env, jclass, jstring path) {
  try {
    zvec_collection_t *collection = nullptr;
    std::string path_string = to_string(env, path);
    if (!check(env, zvec_collection_open(path_string.c_str(), nullptr,
                                         &collection),
               "zvec_collection_open")) {
      return 0;
    }
    return reinterpret_cast<jlong>(collection);
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
    return 0;
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_zvec_internal_jni_JniNative_close(JNIEnv *env, jclass, jlong address) {
  try {
    check(env, zvec_collection_close(handle(address)), "zvec_collection_close");
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_zvec_internal_jni_JniNative_flush(JNIEnv *env, jclass, jlong address) {
  try {
    check(env, zvec_collection_flush(handle(address)), "zvec_collection_flush");
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_zvec_internal_jni_JniNative_readSchema(JNIEnv *env, jclass,
                                                jlong address) {
  try {
    zvec_collection_schema_t *schema = nullptr;
    if (!check(env, zvec_collection_get_schema(handle(address), &schema),
               "zvec_collection_get_schema")) {
      return nullptr;
    }
    SchemaPtr owned(schema);
    return schema_from_native(env, schema);
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
    return nullptr;
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_org_zvec_internal_jni_JniNative_insert(JNIEnv *env, jclass, jlong address,
                                            jobject schema, jobject docs) {
  try {
    int count = list_size(env, docs);
    if (count == 0) {
      return 0;
    }
    std::vector<DocPtr> native_docs = docs_to_native(env, docs, schema);
    if (env->ExceptionCheck()) {
      return 0;
    }
    std::vector<const zvec_doc_t *> doc_ptrs;
    doc_ptrs.reserve(native_docs.size());
    for (const auto &doc : native_docs) {
      doc_ptrs.push_back(doc.get());
    }
    size_t success_count = 0;
    size_t error_count = 0;
    if (!check(env,
               zvec_collection_insert(handle(address), doc_ptrs.data(),
                                      doc_ptrs.size(), &success_count,
                                      &error_count),
               "zvec_collection_insert")) {
      return 0;
    }
    if (error_count != 0) {
      throw_zvec(env, -1,
                 "zvec_collection_insert reported " +
                     std::to_string(error_count) + " per-document failures");
      return 0;
    }
    return static_cast<jint>(success_count);
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
    return 0;
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_zvec_internal_jni_JniNative_query(JNIEnv *env, jclass, jlong address,
                                           jobject query_schema,
                                           jobject result_schema,
                                           jobject query) {
  try {
    QueryPtr native_query = query_to_native(env, query, query_schema, result_schema);
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    zvec_doc_t **results = nullptr;
    size_t result_count = 0;
    if (!check(env,
               zvec_collection_query(handle(address), native_query.get(),
                                     &results, &result_count),
               "zvec_collection_query")) {
      return nullptr;
    }
    jobject out = new_array_list(env);
    for (size_t i = 0; i < result_count; i++) {
      list_add(env, out, doc_from_native(env, results[i], result_schema));
      if (env->ExceptionCheck()) {
        break;
      }
    }
    zvec_docs_free(results, result_count);
    return out;
  } catch (const std::invalid_argument &e) {
    throw_exception(env, "java/lang/IllegalArgumentException", e.what());
    return nullptr;
  } catch (const std::exception &e) {
    throw_exception(env, "java/lang/IllegalStateException", e.what());
    return nullptr;
  }
}
