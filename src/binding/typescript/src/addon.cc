// Copyright 2025-present the zvec project
#include <napi.h>
#include "zvec/zvec.h"

// --- Error helper ---
static void ThrowIfError(Napi::Env env, zvec_status_t *status) {
  if (status == nullptr) return;
  int code = zvec_status_code(status);
  const char *msg = zvec_status_message(status);
  std::string err_msg =
      std::string("zvec error ") + std::to_string(code) + ": " + msg;
  zvec_status_destroy(status);
  NAPI_THROW_VOID(Napi::Error::New(env, err_msg));
}

static Napi::Value ThrowIfErrorVal(Napi::Env env, zvec_status_t *status) {
  if (status == nullptr) return env.Undefined();
  int code = zvec_status_code(status);
  const char *msg = zvec_status_message(status);
  std::string err_msg =
      std::string("zvec error ") + std::to_string(code) + ": " + msg;
  zvec_status_destroy(status);
  NAPI_THROW(Napi::Error::New(env, err_msg), env.Undefined());
}

// --- Schema ---
class SchemaWrap : public Napi::ObjectWrap<SchemaWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func =
        DefineClass(env, "Schema",
                    {
                        InstanceMethod("addField", &SchemaWrap::AddField),
                    });
    auto *constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);
    exports.Set("Schema", func);
    return exports;
  }

  SchemaWrap(const Napi::CallbackInfo &info)
      : Napi::ObjectWrap<SchemaWrap>(info) {
    std::string name = info[0].As<Napi::String>().Utf8Value();
    ptr_ = zvec_schema_create(name.c_str());
  }

  ~SchemaWrap() {
    if (ptr_) zvec_schema_destroy(ptr_);
  }

  zvec_schema_t *ptr() {
    return ptr_;
  }

 private:
  Napi::Value AddField(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    std::string name = info[0].As<Napi::String>().Utf8Value();
    uint32_t type = info[1].As<Napi::Number>().Uint32Value();
    uint32_t dim =
        info.Length() > 2 ? info[2].As<Napi::Number>().Uint32Value() : 0;
    zvec_status_t *s =
        zvec_schema_add_field(ptr_, name.c_str(), (zvec_data_type_t)type, dim);
    return ThrowIfErrorVal(env, s);
  }

  zvec_schema_t *ptr_ = nullptr;
};

// --- Doc ---
class DocWrap : public Napi::ObjectWrap<DocWrap> {
 public:
  static Napi::FunctionReference constructor;

  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func =
        DefineClass(env, "Doc",
                    {
                        InstanceMethod("setPK", &DocWrap::SetPK),
                        InstanceMethod("pk", &DocWrap::PK),
                        InstanceMethod("setString", &DocWrap::SetString),
                        InstanceMethod("setInt32", &DocWrap::SetInt32),
                        InstanceMethod("setVector", &DocWrap::SetVector),
                        InstanceMethod("score", &DocWrap::Score),
                    });
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("Doc", func);
    return exports;
  }

  DocWrap(const Napi::CallbackInfo &info) : Napi::ObjectWrap<DocWrap>(info) {
    ptr_ = zvec_doc_create();
    owns_ = true;
  }

  ~DocWrap() {
    if (owns_ && ptr_) zvec_doc_destroy(ptr_);
  }

  zvec_doc_t *ptr() {
    return ptr_;
  }

  static Napi::Object NewFromPtr(Napi::Env env, zvec_doc_t *ptr) {
    Napi::Object obj = constructor.New({});
    DocWrap *wrap = Napi::ObjectWrap<DocWrap>::Unwrap(obj);
    if (wrap->ptr_) zvec_doc_destroy(wrap->ptr_);
    wrap->ptr_ = ptr;
    wrap->owns_ = true;
    return obj;
  }

 private:
  Napi::Value SetPK(const Napi::CallbackInfo &info) {
    std::string pk = info[0].As<Napi::String>().Utf8Value();
    zvec_doc_set_pk(ptr_, pk.c_str());
    return info.Env().Undefined();
  }

  Napi::Value PK(const Napi::CallbackInfo &info) {
    const char *pk = zvec_doc_pk(ptr_);
    return Napi::String::New(info.Env(), pk ? pk : "");
  }

  Napi::Value SetString(const Napi::CallbackInfo &info) {
    std::string field = info[0].As<Napi::String>().Utf8Value();
    std::string value = info[1].As<Napi::String>().Utf8Value();
    return ThrowIfErrorVal(
        info.Env(), zvec_doc_set_string(ptr_, field.c_str(), value.c_str()));
  }

  Napi::Value SetInt32(const Napi::CallbackInfo &info) {
    std::string field = info[0].As<Napi::String>().Utf8Value();
    int32_t value = info[1].As<Napi::Number>().Int32Value();
    return ThrowIfErrorVal(info.Env(),
                           zvec_doc_set_int32(ptr_, field.c_str(), value));
  }

  Napi::Value SetVector(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    std::string field = info[0].As<Napi::String>().Utf8Value();
    Napi::Float32Array arr = info[1].As<Napi::Float32Array>();
    return ThrowIfErrorVal(
        env, zvec_doc_set_float_vector(ptr_, field.c_str(), arr.Data(),
                                       (uint32_t)arr.ElementLength()));
  }

  Napi::Value Score(const Napi::CallbackInfo &info) {
    return Napi::Number::New(info.Env(), zvec_doc_score(ptr_));
  }

  zvec_doc_t *ptr_ = nullptr;
  bool owns_ = false;
};

Napi::FunctionReference DocWrap::constructor;

// --- Collection ---
class CollectionWrap : public Napi::ObjectWrap<CollectionWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env, "Collection",
        {
            InstanceMethod("upsert", &CollectionWrap::Upsert),
            InstanceMethod("fetch", &CollectionWrap::Fetch),
            InstanceMethod("flush", &CollectionWrap::Flush),
            InstanceMethod("stats", &CollectionWrap::Stats),
            InstanceMethod("query", &CollectionWrap::Query),
            InstanceMethod("destroy", &CollectionWrap::DestroyPhysical),
        });
    auto *ctor = new Napi::FunctionReference();
    *ctor = Napi::Persistent(func);
    exports.Set("Collection", func);
    return exports;
  }

  CollectionWrap(const Napi::CallbackInfo &info)
      : Napi::ObjectWrap<CollectionWrap>(info) {
    Napi::Env env = info.Env();
    std::string path = info[0].As<Napi::String>().Utf8Value();

    if (info.Length() > 1 && info[1].IsObject()) {
      // Create and open with schema
      SchemaWrap *schema =
          Napi::ObjectWrap<SchemaWrap>::Unwrap(info[1].As<Napi::Object>());
      zvec_status_t *s =
          zvec_collection_create_and_open(path.c_str(), schema->ptr(), &ptr_);
      ThrowIfError(env, s);
    } else {
      // Open existing
      zvec_status_t *s = zvec_collection_open(path.c_str(), &ptr_);
      ThrowIfError(env, s);
    }
  }

  ~CollectionWrap() {
    if (ptr_) zvec_collection_destroy(ptr_);
  }

 private:
  Napi::Value Upsert(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::Array arr = info[0].As<Napi::Array>();
    std::vector<zvec_doc_t *> ptrs(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
      DocWrap *d =
          Napi::ObjectWrap<DocWrap>::Unwrap(arr.Get(i).As<Napi::Object>());
      ptrs[i] = d->ptr();
    }
    return ThrowIfErrorVal(
        env, zvec_collection_upsert(ptr_, ptrs.data(), ptrs.size()));
  }

  Napi::Value Fetch(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::Array pks = info[0].As<Napi::Array>();
    std::vector<std::string> pk_strs(pks.Length());
    std::vector<const char *> pk_ptrs(pks.Length());
    for (uint32_t i = 0; i < pks.Length(); i++) {
      pk_strs[i] = pks.Get(i).As<Napi::String>().Utf8Value();
      pk_ptrs[i] = pk_strs[i].c_str();
    }

    zvec_doc_list_t *list = nullptr;
    zvec_status_t *s = zvec_collection_fetch(
        ptr_, (const char **)pk_ptrs.data(), pk_ptrs.size(), &list);
    if (s) return ThrowIfErrorVal(env, s);

    size_t size = zvec_doc_list_size(list);
    Napi::Array result = Napi::Array::New(env, size);
    for (size_t i = 0; i < size; i++) {
      zvec_doc_t *docPtr = zvec_doc_list_get(list, i);
      result.Set(i, DocWrap::NewFromPtr(env, docPtr));
    }
    zvec_doc_list_destroy(list);
    return result;
  }

  Napi::Value Query(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    std::string field = info[0].As<Napi::String>().Utf8Value();
    Napi::Float32Array vec = info[1].As<Napi::Float32Array>();
    int topk = info[2].As<Napi::Number>().Int32Value();

    zvec_doc_list_t *list = nullptr;
    zvec_status_t *s =
        zvec_collection_query(ptr_, field.c_str(), vec.Data(),
                              (uint32_t)vec.ElementLength(), topk, &list);
    if (s) return ThrowIfErrorVal(env, s);

    size_t size = zvec_doc_list_size(list);
    Napi::Array result = Napi::Array::New(env, size);
    for (size_t i = 0; i < size; i++) {
      zvec_doc_t *docPtr = zvec_doc_list_get(list, i);
      result.Set(i, DocWrap::NewFromPtr(env, docPtr));
    }
    zvec_doc_list_destroy(list);
    return result;
  }

  Napi::Value Flush(const Napi::CallbackInfo &info) {
    return ThrowIfErrorVal(info.Env(), zvec_collection_flush(ptr_));
  }

  Napi::Value Stats(const Napi::CallbackInfo &info) {
    zvec_stats_t *stats = nullptr;
    zvec_status_t *s = zvec_collection_get_stats(ptr_, &stats);
    if (s) return ThrowIfErrorVal(info.Env(), s);
    uint64_t count = zvec_stats_total_docs(stats);
    zvec_stats_destroy(stats);
    return Napi::Number::New(info.Env(), (double)count);
  }

  Napi::Value DestroyPhysical(const Napi::CallbackInfo &info) {
    return ThrowIfErrorVal(info.Env(), zvec_collection_destroy_physical(ptr_));
  }

  zvec_collection_t *ptr_ = nullptr;
};

// --- DataType constants ---
static Napi::Object InitDataType(Napi::Env env) {
  Napi::Object dt = Napi::Object::New(env);
  dt.Set("Undefined", Napi::Number::New(env, 0));
  dt.Set("Binary", Napi::Number::New(env, 1));
  dt.Set("String", Napi::Number::New(env, 2));
  dt.Set("Bool", Napi::Number::New(env, 3));
  dt.Set("Int32", Napi::Number::New(env, 4));
  dt.Set("Int64", Napi::Number::New(env, 5));
  dt.Set("Uint32", Napi::Number::New(env, 6));
  dt.Set("Uint64", Napi::Number::New(env, 7));
  dt.Set("Float", Napi::Number::New(env, 8));
  dt.Set("Double", Napi::Number::New(env, 9));
  dt.Set("VectorBinary32", Napi::Number::New(env, 20));
  dt.Set("VectorFP16", Napi::Number::New(env, 22));
  dt.Set("VectorFP32", Napi::Number::New(env, 23));
  dt.Set("VectorFP64", Napi::Number::New(env, 24));
  dt.Set("VectorInt8", Napi::Number::New(env, 26));
  dt.Set("SparseVectorFP16", Napi::Number::New(env, 30));
  dt.Set("SparseVectorFP32", Napi::Number::New(env, 31));
  dt.Set("ArrayString", Napi::Number::New(env, 41));
  dt.Set("ArrayInt32", Napi::Number::New(env, 43));
  dt.Set("ArrayFloat", Napi::Number::New(env, 47));
  return dt;
}

// --- Module Init ---
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  SchemaWrap::Init(env, exports);
  DocWrap::Init(env, exports);
  CollectionWrap::Init(env, exports);
  exports.Set("DataType", InitDataType(env));
  return exports;
}

NODE_API_MODULE(zvec_addon, Init)
