// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/encoding/json.h>
#include <zvec/ailego/logger/logger.h>

//! Global environ variable
extern char **environ;

namespace zvec {
namespace ailego {

static void ParseFromJsonObject(const ailego::JsonObject &obj, Params *params) {
  for (ailego::JsonObject::const_iterator it = obj.begin(); it != obj.end();
       ++it) {
    const ailego::JsonValue &val = it->value();

    if (val.is_boolean()) {
      params->set(it->key().as_stl_string(), val.as_bool());
    } else if (val.is_integer()) {
      params->set(it->key().as_stl_string(),
                  static_cast<int64_t>(val.as_integer()));
    } else if (val.is_float()) {
      params->set(it->key().as_stl_string(), val.as_float());
    } else if (val.is_string()) {
      params->set(it->key().as_stl_string(),
                  val.as_string().decode().as_stl_string());
    } else if (val.is_object()) {
      Params subparams;
      ParseFromJsonObject(val.as_object(), &subparams);
      params->set(it->key().as_stl_string(), std::move(subparams));
    }
  }
}

bool Params::ParseFromBuffer(const std::string &buf, Params *params) {
  ailego::JsonValue val;
  ailego::JsonParser parser;

  parser.set_comment(true);
  parser.set_simple(true);
  parser.set_squote(true);
  parser.set_unstrict(false);
  if (!parser.parse(buf.c_str(), &val)) {
    return false;
  }

  if (!val.is_object()) {
    return false;
  }
  ParseFromJsonObject(val.as_object(), params);
  return true;
}

void Params::ParseFromEnvironment(Params *params) {
  // Dump all environ string
  for (size_t i = 0; environ[i]; ++i) {
    const char *env = environ[i];
    const char *p = std::strchr(env, '=');
    if (p) {
      params->set(std::string(env, p - env), std::string(p + 1));
    }
  }
}

static void SerializeToJsonObject(const Params &params,
                                  ailego::JsonObject *obj) {
  for (const auto &it : params.hypercube().cubes()) {
    const ailego::Cube &cube = it.second;
    const char *key = it.first.c_str();

    if (cube.compatible<std::string>()) {
      const auto &val = cube.unsafe_cast<std::string>();
      ailego::JsonString str(val.data(), val.size());
      obj->set(key, ailego::JsonValue(str.encode()));
    } else if (cube.compatible<unsigned long long int>()) {
      obj->set(key,
               ailego::JsonValue(cube.unsafe_cast<unsigned long long int>()));
    } else if (cube.compatible<long long int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<long long int>()));
    } else if (cube.compatible<unsigned long int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<unsigned long int>()));
    } else if (cube.compatible<long int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<long int>()));
    } else if (cube.compatible<unsigned int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<unsigned int>()));
    } else if (cube.compatible<int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<int>()));
    } else if (cube.compatible<unsigned short int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<unsigned short int>()));
    } else if (cube.compatible<short int>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<short int>()));
    } else if (cube.compatible<unsigned char>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<unsigned char>()));
    } else if (cube.compatible<char>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<char>()));
    } else if (cube.compatible<signed char>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<signed char>()));
    } else if (cube.compatible<bool>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<bool>()));
    } else if (cube.compatible<float>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<float>()));
    } else if (cube.compatible<double>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<double>()));
    } else if (cube.compatible<long double>()) {
      obj->set(key, ailego::JsonValue(cube.unsafe_cast<long double>()));
    } else if (cube.compatible<Params>()) {
      ailego::JsonObject subobj;
      SerializeToJsonObject(cube.unsafe_cast<Params>(), &subobj);
      obj->set(key, ailego::JsonValue(subobj));
    } else {
      LOG_WARN("Unsupported serializing \'%s\' <%s>.", key, cube.type().name());
    }
  }
}

void Params::SerializeToBuffer(const Params &params, std::string *buf) {
  if (buf != nullptr) {
    ailego::JsonObject obj;
    SerializeToJsonObject(params, &obj);
    buf->assign(ailego::JsonValue(obj).as_json_string().as_stl_string());
  }
}

}  // namespace ailego
}  // namespace zvec
