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

#include <gtest/gtest.h>
#include <zvec/ailego/encoding/json.h>

using namespace zvec::ailego;

TEST(Json, JsonParser) {
  {
    JsonValue val;
    JsonParser parser;

    JsonString str =
        "{first: {int: 123, float: 1.0, "
        "true:[true, true, true, true], false:[false],  zero:[0,0,0]}, "
        "true:true, false:[false, false, false, false], zero:[0,0]}";

    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val.refer() == 1);

    const JsonValue &val2 = val;
    const JsonObject &obj2 = val2.as_object();

    EXPECT_TRUE(val2.as_object().refer() == 1);
    JsonValue val_result;
    EXPECT_TRUE(obj2.get("first", &val_result));
    EXPECT_TRUE(val_result.refer() == 2);
    EXPECT_TRUE(val_result.as_object().refer() == 2);

    EXPECT_TRUE(obj2.get("true", &val_result));
    EXPECT_TRUE(obj2.get("false", &val_result));
    EXPECT_TRUE(obj2.get("zero", &val_result));

    const JsonValue val3 = val;
    EXPECT_TRUE(val3.refer() == 2);
    EXPECT_TRUE(val3.as_object().refer() == 1);

    JsonValue val4 = val;
    EXPECT_TRUE(val4.refer() == 3);
    EXPECT_TRUE(val3.refer() == 3);
    EXPECT_TRUE(val2.refer() == 3);

    JsonObject &obj4 = val4.as_object();
    EXPECT_TRUE(obj4.refer() == 2);
    EXPECT_TRUE(val4.refer() == 0);
    EXPECT_TRUE(val3.refer() == 2);
    EXPECT_TRUE(val3.as_object().refer() == 2);
    EXPECT_TRUE(val2.refer() == 2);
  }

  {
    JsonString str =
        "[ true,,\'\\u9701abcd \\u38981515\\u89454845\\uabcd\\uef12\'";
    JsonParser parser;
    JsonValue val = JsonValue();

    JsonValue tmp;
    EXPECT_FALSE(tmp.parse(str));

    parser.set_squote();
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonParser parser;
    JsonString str = "{ 0:0, 1: 1, 2:2, 3:3, 4: 4, 5:5}";
    JsonValue val;

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val.refer() == 1);

    JsonValue val2 = val;
    EXPECT_TRUE(val2.refer() == 2);

    JsonObject &obj2 = val2.as_object();
    EXPECT_TRUE(val2.refer() == 0);
    EXPECT_TRUE(obj2.refer() == 2);
    EXPECT_TRUE(obj2["0"].refer() == 2);
    EXPECT_TRUE(obj2["1"].refer() == 2);
    EXPECT_TRUE(obj2["2"].refer() == 2);
    EXPECT_TRUE(obj2["3"].refer() == 2);
    EXPECT_TRUE(obj2["4"].refer() == 2);
    EXPECT_TRUE(obj2["5"].refer() == 2);
    EXPECT_TRUE(obj2.refer() == 0);

    JsonValue val3 = val;
    EXPECT_TRUE(val3.refer() == 2);

    JsonObject::const_iterator iter = obj2.begin();
    EXPECT_TRUE(iter->key().refer() == 2);
    EXPECT_TRUE(iter->value().refer() == 2);
  }

  {
    JsonParser parser;
    JsonString str = "[0, 1, 2, 3, 4, 5]";
    JsonValue val;

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val.refer() == 1);

    JsonValue val2 = val;
    EXPECT_TRUE(val2.refer() == 2);

    JsonArray &arr2 = val2.as_array();
    EXPECT_TRUE(val2.refer() == 0);
    EXPECT_TRUE(arr2.refer() == 2);
    EXPECT_TRUE(arr2[0].refer() == 2);
    EXPECT_TRUE(arr2[1].refer() == 2);
    EXPECT_TRUE(arr2[2].refer() == 2);
    EXPECT_TRUE(arr2[3].refer() == 2);
    EXPECT_TRUE(arr2[4].refer() == 2);
    EXPECT_TRUE(arr2[5].refer() == 2);
    EXPECT_TRUE(arr2.refer() == 0);

    JsonValue val3 = val;
    EXPECT_TRUE(val3.refer() == 2);

    JsonArray::const_iterator iter = arr2.begin();
    EXPECT_TRUE(iter->refer() == 2);
  }

  {
    JsonString str =
        "[ 15, true, null,\'\\u9701abcd "
        "\\u38981515\\u89454845\\uabcd\\uef12\',]";
    JsonParser parser;
    JsonValue val(true);

    parser.set_squote();
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val[1].as_bool());
    val[1] = val[2];
    EXPECT_FALSE(val[1].as_bool());
  }

  {
    JsonParser parser;
    JsonValue val1, val2, val3, val4;

    EXPECT_TRUE(parser.parse(
        "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, "
        "6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,"
        "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"
        "\"32\",\"33\",\"34\",\"35\",\"36\","
        "{\"5\":5,\"4\":4,\"3\":3,\"2\":2,\"1\":1,\"0\":0,\"-1\":-1}]",
        &val1));
    EXPECT_TRUE(
        parser.parse("[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\","
                     "\"9\",\"10\",\"11\",\"12\",\"13\",\"14\",\"15\",\"16\","
                     "\"17\",\"18\",\"19\",\"20\",\"21\",\"22\",\"23\",\"24\","
                     "\"25\",\"26\",\"27\",\"28\",\"29\",\"30\",\"31\","
                     "\"32\",\"33\",\"34\",\"35\",\"36\","
                     "{\"-2\":\"-2\",\"-1\":\"-1\",\"1\":\"1\",\"2\":\"2\","
                     "\"3\":\"3\",\"4\":\"4\",\"5\":\"5\",\"6\":\"6\"},"
                     "[],null,true,false,0.0,1.0,9.999,-1]",
                     &val2));
    EXPECT_TRUE(
        parser.parse("[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\","
                     "\"9\",\"10\",\"11\",\"12\",\"13\",\"14\",\"15\",\"16\","
                     "\"17\",\"18\",\"19\",\"20\",\"21\",\"22\",\"23\",\"24\","
                     "\"25\",\"26\",\"27\",\"28\",\"29\",\"30\",\"31\","
                     "\"32\",\"33\",\"34\",\"35\",\"36\","
                     "{\"5\":\"5\",\"4\":\"4\",\"3\":\"3\","
                     "\"2\":\"2\",\"1\":\"1\",\"0\":0,"
                     "\"-1\":\"-1\",\"-2\":\"-2\",\"6\":\"6\"},"
                     "[],null,true,false,0.0,1.0,9.999,-1]",
                     &val3));
    EXPECT_TRUE(
        parser.parse("[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, "
                     "6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,"
                     "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"
                     "\"32\",\"33\",\"34\",\"35\",\"36\","
                     "{\"-2\":\"-2\",\"-1\":-1,\"1\":1,\"2\":2,"
                     "\"3\":3,\"4\":4,\"5\":5,\"6\":\"6\",\"0\":0},"
                     "[],null,true,false,0.0,1.0,9.999,-1]",
                     &val4));

    JsonValue tmp1 = val1;
    tmp1.merge(val2);

    JsonValue tmp2 = val2;
    tmp2.merge(val1);

    JsonDumper dumper;
    EXPECT_TRUE(dumper.dump(val1));
    EXPECT_TRUE(dumper.dump(val2));
    EXPECT_TRUE(dumper.dump(val3));
    EXPECT_TRUE(dumper.dump(val4));
    EXPECT_TRUE(dumper.dump(tmp1));

    EXPECT_TRUE(tmp1.as_json_string() == val3.as_json_string());
    EXPECT_TRUE(tmp2.as_json_string() == val4.as_json_string());
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str =
        "{\"req\": {\"aid\": \"\", \"friend\": "
        "\"1234567890\", \"uintype\": "
        "0}}";
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_FALSE(parser.parse("", &val));

    JsonValue tmp;
    EXPECT_TRUE(tmp.parse(str));
    EXPECT_TRUE(tmp == val);
    EXPECT_FALSE(tmp != val);

    const JsonValue &req = val["req"];
    EXPECT_TRUE(req.is_object());
    EXPECT_TRUE(req["show"].as_integer() == 0);
    EXPECT_TRUE(req["friend"].as_integer() == 1234567890);
    EXPECT_TRUE(req[1].is_null());
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "[true, false, 0, 1, 2, \"3\"]";
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val[0u].as_bool());
    EXPECT_FALSE(val[1].as_bool());
    EXPECT_TRUE(val[2].as_integer() == 0);
    EXPECT_TRUE(val[3].as_integer() == 1);
    EXPECT_TRUE(val[4].as_integer() == 2);
    EXPECT_TRUE(val[5].as_integer() == 3);

    JsonValue tmp;
    EXPECT_TRUE(tmp.parse(str));
    EXPECT_TRUE(tmp == val);
    EXPECT_FALSE(tmp != val);

    const JsonValue val2 = val;
    EXPECT_TRUE(val2[0u].as_bool());
    EXPECT_FALSE(val2[1].as_bool());
    EXPECT_TRUE(val2[2].as_integer() == 0);
    EXPECT_TRUE(val2[3].as_integer() == 1);
    EXPECT_TRUE(val2[4].as_integer() == 2);
    EXPECT_TRUE(val2[5].as_integer() == 3);
    EXPECT_TRUE(val2[6].is_null());
    EXPECT_TRUE(val2[(JsonValue::size_type)-1].as_integer() == 0);
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{abcd:\"1234\"}";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == std::string("1234"));

    parser.set_unstrict(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_integer() == 1234);
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "/*comments*/ { abcd\t  :  /* //comments */\"1234\" }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    parser.set_comment(false);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    parser.set_comment(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == std::string("1234"));

    parser.set_unstrict(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_integer() == 1234);
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ abcd/*  fff*/  :  /* //comments */\"1234\" }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    parser.set_comment(false);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    parser.set_comment(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == std::string("1234"));

    parser.set_unstrict(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_integer() == 1234);
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str =
        "{ \"abcd\\\"/*  fff*/  :  /* //comments */\"1234\" , {, [,  ]}}";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_unstrict(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ abcd///comments */\"1234\", [] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ abcd/*//*/ : \t  \"1234\" }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == NULL);
    EXPECT_TRUE(val["abcd/*//*/"].as_c_string() == std::string("1234"));

    parser.set_comment(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == std::string("1234"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ ,{}, \"abcd/*//*/ : \t  \"1234\", }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ ccdd: [], abcd\" /*//*/ \n: \t  \"1234\" }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_comment(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == NULL);
    EXPECT_TRUE(val["abcd\""].as_c_string() == std::string("1234"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{,, \"\" \n: \t  \"1234\" }";
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val[""].as_c_string() == std::string("1234"));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val[""].as_c_string() == std::string("1234"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ ,  \n: \t  \"1234\" }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val[""].as_c_string() == NULL);

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ \'ccdd\': [], \'abcd\' /*//*/ \n: \t  \"1234\" }";

    parser.set_comment(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["\'abcd\'"].as_c_string() == std::string("1234"));

    parser.set_squote(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["abcd"].as_c_string() == std::string("1234"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ 1234 : \'abcd\', \'5678\' : [5, \'5678\'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["1234"].as_c_string() == std::string("abcd"));
    EXPECT_TRUE(val["5678"].as_array().at(1).as_c_string() ==
                std::string("5678"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ 1234 : \'ab\"cd\', \'5678\' : [\"5\", \'5678\'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["1234"].as_c_string() == std::string("ab\"cd"));
    EXPECT_TRUE(val["5678"].as_array().at(1).as_c_string() ==
                std::string("5678"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ 1234 : \'ab\\\'cd\', \'5678\' : [\"5\", \'5678\'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_unstrict(true);
    EXPECT_TRUE(parser.parse(str.c_str(), &val));
    EXPECT_TRUE(val["1234"].as_c_string() == std::string("ab\\\'cd"));
    EXPECT_TRUE(val["5678"].as_array().at(1).as_c_string() ==
                std::string("5678"));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str = "{ \'1234\'\' : \'abcd\', \'5678\' : [\"5\", \'5678\'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_unstrict(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str =
        "{ \'1234\' : \'abcd\' \", \'5678\' : [\"5\", \'5678\'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_unstrict(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str =
        "{ \'1234\' : \'abcd\' , \'5678\' : [\"5\" \", \'5678\'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_unstrict(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val;
    JsonParser parser;

    std::string str =
        "{ \'1234\' : \'abcd\' , \'5678\' : [\"5\" , \'5678\' \'] }";
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_squote(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_simple(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));

    parser.set_unstrict(true);
    EXPECT_FALSE(parser.parse(str.c_str(), &val));
  }

  {
    JsonValue val1;
    JsonValue val2;
    JsonString str =
        "{\"a\":1, \"b\":2, \"c\":3, \"string\":  \"string\", "
        "\"array\": [null, true, false, "
        ", 0], \"object\": {\"a\":1.0, \"b\":2.0, \"c\":3.0}, "
        "\"true\": true, \"null\": null}";

    EXPECT_TRUE(val1.parse(str));
    EXPECT_TRUE(val2.parse(str));
    EXPECT_TRUE(val2 == val1);
    EXPECT_FALSE((val2 != val1));
    EXPECT_TRUE(val2.as_object() == val1);
    EXPECT_TRUE(val2 == val1.as_object());
    EXPECT_TRUE(val2.as_object() == val1.as_object());
    EXPECT_TRUE(val1["string"] == val2["string"]);
  }
}

TEST(Json, JsonObject) {
  {
    JsonObject jobj;

    for (int i = 0; i < 1000; ++i) {
      JsonValue key(i);
      EXPECT_TRUE(jobj.set(key.as_json_string().c_str(), JsonValue((float)i)));

      JsonValue::integer_type val;
      EXPECT_TRUE(jobj.get(key.as_json_string(), &val));
      EXPECT_EQ(val, i);
    }

    for (int i = 0; i < 1000; ++i) {
      JsonValue key(i);
      jobj.unset(key.as_json_string().c_str());

      JsonValue::integer_type val = 0;
      EXPECT_FALSE(jobj.get(key.as_stl_string(), &val));
      EXPECT_EQ(val, 0);
    }
  }

  {
    JsonObject obj;

    obj.set("0", JsonValue(0));
    obj.set("1", JsonValue(1));
    obj.set("2", JsonValue(2));
    obj.set("3", JsonValue(3));
    obj.set("4", JsonValue(4));
    obj.set("5", JsonValue(5));
    obj.set("6", JsonValue(6));
    obj.set("7", JsonValue(7));
    obj.set("8", JsonValue(8));
    obj.set("9", JsonValue(9));

    EXPECT_EQ(obj.size(), 10u);
    EXPECT_EQ(obj["0"].as_integer(), 0);
    EXPECT_EQ(obj["1"].as_integer(), 1);
    EXPECT_EQ(obj["2"].as_integer(), 2);
    EXPECT_EQ(obj["3"].as_integer(), 3);
    EXPECT_EQ(obj["4"].as_integer(), 4);
    EXPECT_EQ(obj["5"].as_integer(), 5);
    EXPECT_EQ(obj["6"].as_integer(), 6);
    EXPECT_EQ(obj["7"].as_integer(), 7);
    EXPECT_EQ(obj["8"].as_integer(), 8);
    EXPECT_EQ(obj["9"].as_integer(), 9);
    EXPECT_EQ(obj.size(), 10u);

    int index_id = 0;
    for (JsonObject::const_iterator it = obj.cbegin(); it != obj.cend();
         ++it, ++index_id) {
      EXPECT_EQ(it->value().as_integer(), index_id);
    }

    int index_id_r = 9;
    for (JsonObject::const_reverse_iterator it = obj.crbegin();
         it != obj.crend(); ++it, --index_id_r) {
      EXPECT_EQ(it->value().as_integer(), index_id_r);
    }

    obj.unset("1");
    EXPECT_EQ(obj.size(), 9u);
    obj.unset("3");
    EXPECT_EQ(obj.size(), 8u);
    obj.unset("5");
    EXPECT_EQ(obj.size(), 7u);
    obj.unset("7");
    EXPECT_EQ(obj.size(), 6u);
    obj.unset("9");
    EXPECT_EQ(obj.size(), 5u);

    obj.clear();
    EXPECT_EQ(obj.size(), 0u);
  }

  {
    JsonObject obj;

    // 0
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_FALSE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_FALSE(obj.has("44444"));
    EXPECT_FALSE(obj.has("55555"));
    EXPECT_FALSE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_FALSE(obj.has("88888"));
    EXPECT_FALSE(obj.has("99999"));

    // 1
    EXPECT_TRUE(obj.set("55555", "55555"));
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_FALSE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_FALSE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_FALSE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_FALSE(obj.has("88888"));
    EXPECT_FALSE(obj.has("99999"));

    // 2
    EXPECT_TRUE(obj.set("88888", "88888"));
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_FALSE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_FALSE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_FALSE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_FALSE(obj.has("99999"));

    // 2
    EXPECT_TRUE(obj.set("66666", "66666"));
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_FALSE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_FALSE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_FALSE(obj.has("99999"));

    // 3
    EXPECT_TRUE(obj.set("44444", "44444"));
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_FALSE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_FALSE(obj.has("99999"));

    // 4
    EXPECT_TRUE(obj.set("99999", "99999"));
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_FALSE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_TRUE(obj.has("99999"));

    // 5
    EXPECT_TRUE(obj.set("22222", "22222"));
    EXPECT_FALSE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_TRUE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_TRUE(obj.has("99999"));

    // 6
    EXPECT_TRUE(obj.set("00000", "00000"));
    EXPECT_TRUE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_TRUE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_FALSE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_TRUE(obj.has("99999"));

    // 7
    EXPECT_TRUE(obj.set("77777", "77777"));
    EXPECT_TRUE(obj.has("00000"));
    EXPECT_FALSE(obj.has("11111"));
    EXPECT_TRUE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_TRUE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_TRUE(obj.has("99999"));

    // 8
    EXPECT_TRUE(obj.set("11111", "11111"));
    EXPECT_TRUE(obj.has("00000"));
    EXPECT_TRUE(obj.has("11111"));
    EXPECT_TRUE(obj.has("22222"));
    EXPECT_FALSE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_TRUE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_TRUE(obj.has("99999"));

    // 9
    EXPECT_TRUE(obj.set("33333", "33333"));
    EXPECT_TRUE(obj.has("00000"));
    EXPECT_TRUE(obj.has("11111"));
    EXPECT_TRUE(obj.has("22222"));
    EXPECT_TRUE(obj.has("33333"));
    EXPECT_TRUE(obj.has("44444"));
    EXPECT_TRUE(obj.has("55555"));
    EXPECT_TRUE(obj.has("66666"));
    EXPECT_TRUE(obj.has("77777"));
    EXPECT_TRUE(obj.has("88888"));
    EXPECT_TRUE(obj.has("99999"));

    EXPECT_EQ(10u, obj.size());

    int index_ids[] = {0,     11111, 22222, 33333, 44444,
                       55555, 66666, 77777, 88888, 99999};
    int i = 0;
    for (JsonObject::iterator it = obj.begin(); it != obj.end(); ++it, ++i) {
      EXPECT_EQ(it->value().as_integer(), index_ids[i]);
    }
    EXPECT_EQ(i, (int)obj.size());
  }

  {
    JsonObject obj;

    EXPECT_TRUE(obj.set("key0", "key0"));
    EXPECT_TRUE(obj.set("key1", "key1"));
    EXPECT_TRUE(obj.set("key2", "key2"));
    EXPECT_TRUE(obj.set("key3", "key3"));
    EXPECT_TRUE(obj.set("key4", "key4"));
    EXPECT_TRUE(obj.set("key5", "key5"));
    EXPECT_FALSE(obj.set("key0", "000000"));
    EXPECT_FALSE(obj.set("key1", "000000"));
    EXPECT_FALSE(obj.set("key5", "000000"));
    EXPECT_EQ(obj.size(), 6u);

    JsonString str;
    EXPECT_TRUE(obj.get("key0", &str));
    EXPECT_TRUE(str == JsonString("key0"));
    EXPECT_TRUE(obj.get("key3", &str));
    EXPECT_TRUE(str == JsonString("key3"));
    EXPECT_TRUE(obj.get("key5", &str));
    EXPECT_TRUE(str == JsonString("key5"));
    EXPECT_EQ(obj.size(), 6u);

    obj["key0"] = 0;
    obj["key1"] = 1;
    obj["key2"] = 2;
    obj["key3"] = 3;
    obj["key4"] = 4;
    obj["key5"] = 5;

    EXPECT_EQ(obj.size(), 6u);

    int index_id = 0;
    for (JsonObject::iterator it = obj.begin(); it != obj.end();
         ++it, ++index_id) {
      EXPECT_EQ(it->value().as_integer(), index_id);
    }
    EXPECT_EQ(index_id, 6);

    int index_id_r = 5;
    for (JsonObject::reverse_iterator it = obj.rbegin(); it != obj.rend();
         ++it, --index_id_r) {
      EXPECT_EQ(it->value().as_integer(), index_id_r);
    }
    EXPECT_EQ(index_id_r, -1);
  }

  {
    JsonObject::reverse_iterator it1 = JsonObject::iterator();
    JsonObject::reverse_iterator it2 = JsonObject::reverse_iterator();
    EXPECT_TRUE(it1 == it2);

    JsonObject::iterator it3 = JsonObject::reverse_iterator();
    JsonObject::iterator it4 = JsonObject::iterator();
    EXPECT_TRUE(it3 == it4);

    JsonObject::const_iterator it5 = JsonObject::const_iterator();
    JsonObject::const_iterator it6 = JsonObject::iterator();
    EXPECT_TRUE(it5 == it6);

    JsonObject::const_iterator it7 = JsonObject::reverse_iterator();
    JsonObject::const_iterator it8 = JsonObject::const_reverse_iterator();
    EXPECT_TRUE(it7 == it8);

    JsonObject::const_reverse_iterator it9 = JsonObject::const_iterator();
    JsonObject::const_reverse_iterator it10 = JsonObject::iterator();
    EXPECT_TRUE(it9 == it10);

    JsonObject::const_reverse_iterator it11 = JsonObject::reverse_iterator();
    JsonObject::const_reverse_iterator it12 =
        JsonObject::const_reverse_iterator();
    EXPECT_TRUE(it11 == it12);
  }

  {
    JsonObject obj1;
    JsonObject obj2;
    JsonObject obj3;
    JsonObject::iterator iter1;

    EXPECT_TRUE(obj1.set("aaa", "123456"));
    obj2 = obj1;
    iter1 = obj1.begin();
    obj3 = obj1;
    iter1->value() = "abcdefg";
    EXPECT_TRUE(obj1["aaa"].as_string() == "abcdefg");
    EXPECT_TRUE(obj2["aaa"].as_string() == "123456");
    EXPECT_TRUE(obj3["aaa"].as_string() == "123456");
  }

  {
    JsonObject obj1;

    obj1.set("FTitle", "123456789");
    obj1.set("FDesc", "abcdef");

    const JsonObject &obj2 = obj1;
    EXPECT_TRUE(obj1["FTitle"].as_stl_string() == "123456789");
    EXPECT_TRUE(obj1["FDesc"].as_stl_string() == "abcdef");
    EXPECT_TRUE(obj2["FTitle"].as_stl_string() == "123456789");
    EXPECT_TRUE(obj2["FDesc"].as_stl_string() == "abcdef");
  }
}

TEST(Json, JsonArray) {
  {
    JsonArray arr1;
    arr1.push(JsonValue(0.0));
    arr1.push(JsonValue(2));
    arr1.push("2");
    arr1.push(JsonValue(true));
    arr1.push(JsonArray());
    arr1.push(JsonObject());
    arr1.push(JsonValue());
    arr1.push(JsonString());

    JsonArray arr2 = arr1;
    EXPECT_TRUE(arr2 == arr1);

    JsonArray arr3;
    arr3.push(JsonValue(0.0));
    arr3.push(JsonValue(2));
    arr3.push("2");
    arr3.push(JsonValue(true));
    arr3.push(JsonArray());
    arr3.push(JsonObject());
    arr3.push(JsonValue());
    arr3.push(JsonString());
    EXPECT_TRUE(arr2 == arr3);
    EXPECT_TRUE(arr1 == arr3);

    arr2.push(JsonObject());
    EXPECT_TRUE(arr2 != arr3);
    EXPECT_TRUE(arr2 != arr1);
    EXPECT_TRUE(arr1 == arr3);
  }

  {
    JsonArray jarr;

    EXPECT_TRUE(jarr.capacity() == 0);
    EXPECT_TRUE(jarr.size() == 0);
    jarr.reserve(21);
    EXPECT_TRUE(jarr.capacity() == 32);
    EXPECT_TRUE(jarr.size() == 0);
    jarr.reserve(2);
    EXPECT_TRUE(jarr.capacity() == 32);
    EXPECT_TRUE(jarr.size() == 0);
    jarr.reserve(33);
    EXPECT_TRUE(jarr.capacity() == 64);
    EXPECT_TRUE(jarr.size() == 0);
  }

  {
    JsonArray arr1;
    JsonArray arr2;
    JsonArray arr3;
    JsonArray::iterator iter1;

    arr1.push("123456");
    arr2 = arr1;
    iter1 = arr1.begin();
    arr3 = arr1;
    *iter1 = "abcdefg";
    EXPECT_TRUE(arr1[0].as_string() == "abcdefg");
    EXPECT_TRUE(arr2[0].as_string() == "123456");
    EXPECT_TRUE(arr3[0].as_string() == "123456");
  }

  {
    JsonArray arr1;
    JsonArray arr2;
    JsonArray arr3;

    arr1.push("123456");
    arr2 = arr1;

    JsonValue &val1 = arr1.front();
    arr3 = arr1;
    val1 = "abcdefg";
    EXPECT_TRUE(arr1[0].as_string() == "abcdefg");
    EXPECT_TRUE(arr2[0].as_string() == "123456");
    EXPECT_TRUE(arr3[0].as_string() == "123456");
  }

  {
    JsonArray arr;
    JsonValue val(666);

    arr.push("0");
    arr.push(JsonValue(1));
    arr.push(JsonValue(2));
    arr.push("3");
    arr.push("4");
    arr.push("5");
    arr.push("6");
    arr.push(JsonValue(7.0));
    EXPECT_TRUE(arr.size() == 8);
    EXPECT_TRUE(arr.capacity() == 32);
    EXPECT_TRUE(arr[0].as_string() == "0");
    EXPECT_TRUE(arr[1].as_integer() == 1);
    EXPECT_TRUE(arr[2].as_integer() == 2);
    EXPECT_TRUE(arr[3].as_integer() == 3);
    EXPECT_TRUE(arr[4].as_integer() == 4);
    EXPECT_TRUE(arr[5].as_integer() == 5);
    EXPECT_TRUE(arr[6].as_integer() == 6);
    EXPECT_TRUE(arr[7].as_integer() == 7);
    arr.resize(20, val);
    EXPECT_TRUE(arr.size() == 20);
    arr.resize(5, val);
    EXPECT_TRUE(arr.size() == 5);
    EXPECT_TRUE(arr[0].as_string() == "0");
    EXPECT_TRUE(arr[1].as_integer() == 1);
    EXPECT_TRUE(arr[2].as_integer() == 2);
    EXPECT_TRUE(arr[3].as_string() == "3");
    EXPECT_TRUE(arr[4].as_string() == "4");
    EXPECT_TRUE(val.as_integer() == 666);

    arr.reverse();
    EXPECT_TRUE(arr.size() == 5);
    EXPECT_TRUE(arr[4].as_string() == "0");
    EXPECT_TRUE(arr[3].as_integer() == 1);
    EXPECT_TRUE(arr[2].as_integer() == 2);
    EXPECT_TRUE(arr[1].as_string() == "3");
    EXPECT_TRUE(arr[0].as_string() == "4");

    arr.shift();
    arr.reverse();
    EXPECT_TRUE(arr.size() == 4);
    EXPECT_TRUE(arr[0].as_string() == "0");
    EXPECT_TRUE(arr[1].as_integer() == 1);
    EXPECT_TRUE(arr[2].as_integer() == 2);
    EXPECT_TRUE(arr[3].as_string() == "3");
  }

  {
    JsonArray::reverse_iterator it1 = JsonArray::iterator();
    JsonArray::reverse_iterator it2 = JsonArray::reverse_iterator();
    EXPECT_TRUE(it1 == it2);

    JsonArray::iterator it3 = JsonArray::reverse_iterator();
    JsonArray::iterator it4 = JsonArray::iterator();
    EXPECT_TRUE(it3 == it4);

    JsonArray::const_iterator it5 = JsonArray::const_iterator();
    JsonArray::const_iterator it6 = JsonArray::iterator();
    EXPECT_TRUE(it5 == it6);

    JsonArray::const_iterator it7 = JsonArray::reverse_iterator();
    JsonArray::const_iterator it8 = JsonArray::const_reverse_iterator();
    EXPECT_TRUE(it7 == it8);

    JsonArray::const_reverse_iterator it9 = JsonArray::const_iterator();
    JsonArray::const_reverse_iterator it10 = JsonArray::iterator();
    EXPECT_TRUE(it9 == it10);

    JsonArray::const_reverse_iterator it11 = JsonArray::reverse_iterator();
    JsonArray::const_reverse_iterator it12 =
        JsonArray::const_reverse_iterator();
    EXPECT_TRUE(it11 == it12);
  }

  {
    JsonArray arr;
    arr.resize(1023);
    EXPECT_TRUE(arr.size() == 1023);
    EXPECT_TRUE(arr.capacity() == 1024);
    EXPECT_TRUE(arr[0].is_null());
    EXPECT_TRUE(arr[1022].is_null());
  }

  {
    JsonArray arr;
    EXPECT_TRUE(arr.capacity() == 0);
    arr.resize(0);
    EXPECT_TRUE(arr.capacity() == 32);
    arr.push(0);
    EXPECT_TRUE(arr.capacity() == 32);
    EXPECT_TRUE(arr.size() == 1);
    arr.resize(0);
    EXPECT_TRUE(arr.size() == 0);
    arr.resize(1);
    EXPECT_TRUE(arr.capacity() == 32);
    EXPECT_TRUE(arr.size() == 1);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[0, 1]"));
    EXPECT_TRUE(val.as_array().front() == 0);
    EXPECT_TRUE(val.as_array().front().as_integer() == 0);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[-1]"));
    EXPECT_TRUE(val.as_array().front() == -1);
    EXPECT_TRUE(val.as_array().front().as_integer() == -1);
    EXPECT_TRUE(val.as_array().front().as_integer() == -1);
    EXPECT_TRUE(val.as_array().front().as_integer() == -1);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[4294967295]"));
    EXPECT_TRUE(val.parse("[+4294967295]"));
    EXPECT_TRUE(val.as_array().front() == 4294967295);
    EXPECT_TRUE((int32_t)val.as_array().front().as_integer() == -1);
    EXPECT_TRUE(val.as_array().front().as_integer() == 4294967295);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[ 2147483647 ]"));
    EXPECT_TRUE(val.parse("[ +2147483647 ]"));
    EXPECT_TRUE(val.as_array().front() == 2147483647);
    EXPECT_TRUE(val.as_array().front().as_integer() == 2147483647);
    EXPECT_TRUE(val.as_array().front().as_integer() == 2147483647);
    EXPECT_TRUE(val.as_array().front().as_integer() == 2147483647);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[ -2147483647 ]"));
    EXPECT_TRUE(val.as_array().front() == -2147483647);
    EXPECT_TRUE(val.as_array().front().as_integer() == -2147483647);
    EXPECT_TRUE(val.as_array().front().as_integer() == -2147483647);
    EXPECT_TRUE(val.as_array().front().as_integer() == -2147483647);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[9223372036854775807]"));
    EXPECT_TRUE(val.parse("[+9223372036854775807]"));
    EXPECT_TRUE(val.as_array().front() == 9223372036854775807uLL);
    EXPECT_TRUE((int32_t)val.as_array().front().as_integer() == -1);
    EXPECT_TRUE(val.as_array().front().as_integer() == 9223372036854775807uLL);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[-9223372036854775807]"));
    EXPECT_TRUE(val.as_array().front() == -9223372036854775807LL);
    EXPECT_TRUE(val.as_array().front().as_integer() == -9223372036854775807LL);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[ 1844674407370955161 ]"));
    EXPECT_TRUE(val.parse("[ +1844674407370955161 ]"));
    EXPECT_TRUE(val.as_array().front() == 1844674407370955161uLL);
    EXPECT_TRUE(val.as_array().front().as_integer() == 1844674407370955161uLL);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[ 18446744073709551615 ]"));
    EXPECT_TRUE(val.parse("[ +18446744073709551615 ]"));
    EXPECT_TRUE(val.as_array().front() == 18446744073709551615uLL);
    EXPECT_TRUE(val.as_array().front().as_integer() == -1);
    EXPECT_TRUE(val.as_array().front().as_integer() ==
                JsonValue::integer_type(18446744073709551615uLL));
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[ 18446744073709551616 ]"));
    EXPECT_FALSE(val.as_array().front().is_integer());
    EXPECT_TRUE(val.as_array().front() == 18446744073709551616.0);
    EXPECT_TRUE(val.as_array().front().as_float() == 18446744073709551616.0);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[ 1e+30, 1.3e12 ]"));
    EXPECT_TRUE(val.as_array().front() == 1e+30);
    EXPECT_TRUE(val.as_array().back() == 1.3e12);
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[0,[0]]"));
    EXPECT_TRUE(val.refer() == 1);
    EXPECT_TRUE(val.as_array().refer() == 1);
    EXPECT_TRUE(val.refer() == 0);

    val.as_array().push(val);
    EXPECT_TRUE(val.as_json_string() == "[0,[0],[0,[0]]]");

    val.as_array().pop();
    val.as_array().push(val);
    EXPECT_TRUE(val.as_json_string() == "[0,[0],[0,[0]]]");

    val.as_array().pop();
    val.as_array().pop();
    val.as_array().pop();
    val.as_array().push(val);
    EXPECT_TRUE(val.as_json_string() == "[[]]");
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("[0,[0]]"));
    EXPECT_TRUE(val.refer() == 1);
    EXPECT_TRUE(val.as_array().refer() == 1);
    EXPECT_TRUE(val.refer() == 0);

    val.as_array()[0] = val;
    EXPECT_TRUE(val.as_json_string() == "[[0,[0]],[0]]");
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("{\"0\":[0]}"));
    EXPECT_TRUE(val.refer() == 1);
    EXPECT_TRUE(val.as_object().refer() == 1);
    EXPECT_TRUE(val.refer() == 0);

    JsonObject obj = val.as_object();
    val.as_object()["1"].assign(obj);
    EXPECT_TRUE(val.as_json_string() == "{\"0\":[0],\"1\":{\"0\":[0]}}");
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("{\"0\":[0]}"));
    EXPECT_TRUE(val.refer() == 1);
    EXPECT_TRUE(val.as_object().refer() == 1);
    EXPECT_TRUE(val.refer() == 0);

    EXPECT_TRUE(val.as_object().set("1", val));
    EXPECT_TRUE(val.as_json_string() == "{\"0\":[0],\"1\":{\"0\":[0]}}");
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("{\"0\":[0]}"));
    EXPECT_TRUE(val.refer() == 1);
    EXPECT_TRUE(val.as_object().refer() == 1);
    EXPECT_TRUE(val.refer() == 0);

    JsonValue val2 = val;
    val.as_object()["1"] = val2;
    EXPECT_TRUE(val.as_json_string() == "{\"0\":[0],\"1\":{\"0\":[0]}}");
  }

  {
    JsonValue val;
    EXPECT_TRUE(val.parse("{\"0\":[0]}"));
    EXPECT_TRUE(val.refer() == 1);
    EXPECT_TRUE(val.as_object().refer() == 1);
    EXPECT_TRUE(val.refer() == 0);

    JsonObject obj = val.as_object();
    val.as_object()["1"] = obj;
    EXPECT_TRUE(val.as_json_string() == "{\"0\":[0],\"1\":{\"0\":[0]}}");
  }
}

TEST(Json, JsonString) {
  {
    JsonString str1("1234567890abcdefghijklmn");
    EXPECT_TRUE(str1 == str1.decode());
  }

  {
    JsonString str1("\\\"1234\\\\567890abcdefghijklmn\\t");
    JsonString str2 = "\"1234\\567890abcdefghijklmn\t";
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1(" \\/ \\\\ \\\" \\b \\f \\n \\r \\t ");
    JsonString str2 = " / \\ \" \b \f \n \r \t ";
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1("\\n\\r \\u8096 \\u5141 \\u950B \\u000a \\u000d");
    JsonString str2("\n\r \xE8\x82\x96 \xE5\x85\x81 \xE9\x94\x8B \n \r");
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1("\\u007f");
    JsonString str2("\x7F");
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1("\\u0080");
    JsonString str2("\xC2\x80");
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1("\\u07FF");
    JsonString str2("\xDF\xBF");
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1("\\u0800");
    JsonString str2("\xE0\xA0\x80");
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString str1("\\uFFFF");
    JsonString str2("\xEF\xBF\xBF");
    EXPECT_TRUE(str2 == str1.decode());
  }

  {
    JsonString a("abcdefg");
    JsonString b("abcdefl");
    JsonString c("abcdefg");
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(b != a);

    EXPECT_TRUE(a.compare(c) == 0);
    EXPECT_TRUE(b.compare(c) != 0);
    EXPECT_TRUE(b.compare(c) != 0);
  }

  {
    JsonString a("abcdefg\"");
    JsonString b("abcd");
    JsonString c("abcdefg");
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(b != a);
  }

  {
    JsonString a("abcd\0efg");
    JsonString b("abcd");
    JsonString c("abcdefg\0");
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(b == a);
    EXPECT_TRUE(a.compare(b) == 0);
  }

  {
    JsonString a("abcd\0efg", 8);
    JsonString b("abcd");
    JsonString c("abcdefg\0");
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(b != a);

    EXPECT_TRUE(a.compare("abcd") == 0);
    EXPECT_TRUE(b.compare("abcd\0") == 0);
    EXPECT_TRUE(c.compare("abcdefg") == 0);
  }

  {
    JsonString a("abcd\0efg", 8);
    JsonString b("abcd");
    JsonString c("abcd\0efg", 8);
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(b != a);
  }

  {
    JsonString a;
    JsonString b("\0");
    JsonString c(NULL);
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(b == c);
    EXPECT_TRUE(b == a);
  }

  {
    JsonString a;
    JsonString b("\0", 1);
    JsonString c(NULL);
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(b != a);
  }

  {
    JsonString str1(
        "author:\\u8096\\u5141\\u950b;\\r\\ntest:\\u007f \\u0080 \\u07ff "
        "\\u0800 \\uffff");
    JsonString str2(
        "author:\xE8\x82\x96\xE5\x85\x81\xE9\x94\x8B;\r\ntest:"
        "\x7F \xC2\x80 "
        "\xDF\xBF \xE0\xA0\x80 \xEF\xBF\xBF");
    JsonString str3(
        "author:\xE8\x82\x96\xE5\x85\x81\xE9\x94\x8B;"
        "\\r\\ntest:\x7F \xC2\x80 "
        "\xDF\xBF \xE0\xA0\x80 \xEF\xBF\xBF");
    EXPECT_TRUE(str2 == str1.decode());
    EXPECT_TRUE(str2.encode() == str3);
  }

  {
    JsonString str1("\\007f \\0080 \\u07ff \\u0800 \\uffff");
    JsonString str2("\\u008\\u07ff \\u0800 \\uffff");
    EXPECT_FALSE(str1.decode().is_valid());
    EXPECT_FALSE(str2.decode().is_valid());
  }

  {
    JsonString str1(" \x1f \x0e \x01 \x1e / \\ AAA\" AAA\b \f \n \r \t ");
    JsonString str2(
        " \\u001f \\u000e \\u0001 \\u001e / \\\\ AAA\\\" "
        "AAA\\b \\f \\n \\r "
        "\\t ");
    EXPECT_TRUE(str1.encode() == str2);
    EXPECT_TRUE(str1 == str2.decode());
  }

  {
    JsonString jstr;

    EXPECT_TRUE(jstr.capacity() == 0);
    EXPECT_TRUE(jstr.size() == 0);
    jstr.reserve(21);
    EXPECT_TRUE(jstr.capacity() == 32 - 1);
    EXPECT_TRUE(jstr.size() == 0);
    jstr.reserve(2);
    EXPECT_TRUE(jstr.capacity() == 32 - 1);
    EXPECT_TRUE(jstr.size() == 0);
    jstr.reserve(32);
    EXPECT_TRUE(jstr.capacity() == 64 - 1);
    EXPECT_TRUE(jstr.size() == 0);

    char buf[1000];
    buf[0] = '\0';
    jstr.assign(buf, sizeof(buf));
    EXPECT_TRUE(jstr.capacity() == 1024 - 1);
    EXPECT_TRUE(jstr.length() == 1000);
    EXPECT_TRUE(JsonString(jstr.c_str()) == "");

    memcpy(buf, "abcdef", 7);
    jstr.assign(buf, 200);
    EXPECT_TRUE(jstr.capacity() == 1024 - 1);
    EXPECT_TRUE(jstr.length() == 200);
    EXPECT_TRUE(JsonString(jstr.c_str()) == "abcdef");
  }
}

TEST(Json, JsonValue) {
  {
    EXPECT_TRUE(JsonValue(true) == JsonValue(true));
    EXPECT_TRUE(JsonValue(false) == JsonValue(false));
    EXPECT_TRUE(JsonValue((char)'\r') == JsonValue(0xd));
    EXPECT_TRUE(JsonValue((char)'\r') == JsonValue('\r'));
    EXPECT_TRUE(JsonValue(10000) == JsonValue(10000));
    EXPECT_TRUE(JsonValue(0xffff) == JsonValue(0xffff));
    EXPECT_TRUE(JsonValue(0x10000) == JsonValue(0x10000));
    EXPECT_TRUE(JsonValue(0xffffffff) == JsonValue(0xffffffff));
    EXPECT_TRUE(JsonValue(0x100000000) == JsonValue(0x100000000));
    EXPECT_TRUE(JsonValue(0xffffffffffffffff) == JsonValue(0xffffffffffffffff));
    EXPECT_TRUE(JsonValue(0.999999) == JsonValue(0.999999));
    EXPECT_TRUE(JsonValue(false) != JsonValue(0.0));
    EXPECT_TRUE(JsonValue(0.0) != JsonValue(0));
    EXPECT_TRUE(JsonValue("0.0") != JsonValue(0));
    EXPECT_TRUE(JsonValue("0.0") == JsonValue("0.0"));
    EXPECT_TRUE(JsonValue(std::string("0.0001")) == JsonValue("0.0001"));
  }

  {
    EXPECT_EQ(JsonValue(0).as_json_string().as_stl_string(), "0");
    EXPECT_EQ(JsonValue(1).as_json_string().as_stl_string(), "1");
    EXPECT_EQ(JsonValue(-1).as_json_string().as_stl_string(), "-1");
    EXPECT_EQ(JsonValue(99).as_json_string().as_stl_string(), "99");
    EXPECT_EQ(JsonValue(-99).as_json_string().as_stl_string(), "-99");
    EXPECT_EQ(JsonValue(188).as_json_string().as_stl_string(), "188");
    EXPECT_EQ(JsonValue(-188).as_json_string().as_stl_string(), "-188");
    EXPECT_EQ(JsonValue(1520).as_json_string().as_stl_string(), "1520");
    EXPECT_EQ(JsonValue(-1520).as_json_string().as_stl_string(), "-1520");

    EXPECT_EQ(JsonValue(12345).as_json_string().as_stl_string(), "12345");
    EXPECT_EQ(JsonValue(-12345).as_json_string().as_stl_string(), "-12345");

    EXPECT_EQ(JsonValue(65535).as_json_string().as_stl_string(), "65535");
    EXPECT_EQ(JsonValue(-65535).as_json_string().as_stl_string(), "-65535");

    EXPECT_EQ(JsonValue(65536).as_json_string().as_stl_string(), "65536");
    EXPECT_EQ(JsonValue(-65536).as_json_string().as_stl_string(), "-65536");

    EXPECT_EQ(JsonValue(234567).as_json_string().as_stl_string(), "234567");
    EXPECT_EQ(JsonValue(-234567).as_json_string().as_stl_string(), "-234567");

    EXPECT_EQ(JsonValue(1234567890).as_json_string().as_stl_string(),
              "1234567890");
    EXPECT_EQ(JsonValue(-1234567890).as_json_string().as_stl_string(),
              "-1234567890");

    EXPECT_EQ(JsonValue(9999999999).as_json_string().as_stl_string(),
              "9999999999");
    EXPECT_EQ(JsonValue(-9999999999).as_json_string().as_stl_string(),
              "-9999999999");

    EXPECT_EQ(JsonValue(4294967295).as_json_string().as_stl_string(),
              "4294967295");
    // EXPECT_EQ(JsonValue(-4294967295).as_json_string().as_stl_string(),
    //           "-4294967295LL");

    EXPECT_EQ(JsonValue(4294967296).as_json_string().as_stl_string(),
              "4294967296");
    EXPECT_EQ(JsonValue(-4294967296).as_json_string().as_stl_string(),
              "-4294967296");

    EXPECT_EQ(JsonValue(281474976710655).as_json_string().as_stl_string(),
              "281474976710655");
    EXPECT_EQ(JsonValue(-281474976710655).as_json_string().as_stl_string(),
              "-281474976710655");

    EXPECT_EQ(JsonValue(281474976710656).as_json_string().as_stl_string(),
              "281474976710656");
    EXPECT_EQ(JsonValue(-281474976710656).as_json_string().as_stl_string(),
              "-281474976710656");

    EXPECT_EQ(JsonValue(9223372036854775807ll).as_json_string().as_stl_string(),
              "9223372036854775807");
    EXPECT_EQ(
        JsonValue(-9223372036854775807ll).as_json_string().as_stl_string(),
        "-9223372036854775807");
  }

  {
    JsonValue jval;

    jval.assign("aaaaaaaaaaaa");
    jval.assign("122326263", 10);
    jval.assign(200);
    jval.assign(0xffffffffffff);
  }

  {
    JsonValue val1;
    JsonValue val2;
    JsonValue val3;
    JsonValue val4;

    val1 = "abcdef";
    val2 = val1;
    val3 = val1;

    EXPECT_TRUE(val1.refer() == 3);
    EXPECT_TRUE(val2.refer() == 3);
    EXPECT_TRUE(val3.refer() == 3);
    EXPECT_TRUE(val3.as_stl_string() == "abcdef");

    JsonString &str = val1.as_string();
    EXPECT_TRUE(str.refer() == 2);
    val4 = val1;
    str = "123456";

    EXPECT_TRUE(val1.refer() == 0);
    EXPECT_TRUE(val2.refer() == 2);
    EXPECT_TRUE(val3.refer() == 2);
    EXPECT_TRUE(val4.refer() == 1);
    EXPECT_TRUE(val1.as_stl_string() == "123456");
    EXPECT_TRUE(val2.as_stl_string() == "abcdef");
    EXPECT_TRUE(val3.as_stl_string() == "abcdef");
    EXPECT_TRUE(val4.as_stl_string() == "abcdef");
  }

  {
    JsonValue val1;
    JsonValue val2;
    JsonValue val3;

    val1["abcd"] = "1234";
    val2 = val1.as_object();
    val3 = val2;

    EXPECT_TRUE(val1.refer() == 0);
    EXPECT_TRUE(val2.refer() == 2);
    EXPECT_TRUE(val3.refer() == 2);
    EXPECT_TRUE(val1.as_object().refer() == 0);
    EXPECT_TRUE(val2.as_object().refer() == 2);
    EXPECT_TRUE(val3.as_object().refer() == 2);
  }
}

TEST(Json, General) {
  {
    JsonObject obj;
    JsonArray arr;
    JsonValue val;
    JsonString str;

    EXPECT_TRUE(obj.refer() == -1);
    EXPECT_TRUE(arr.refer() == -1);
    EXPECT_TRUE(val.refer() == -1);
    EXPECT_TRUE(str.refer() == -1);

    val = str;
    EXPECT_TRUE(val.refer() == 1);

    val = obj;
    EXPECT_TRUE(val.refer() == 1);

    val = arr;
    EXPECT_TRUE(val.refer() == 1);

    arr.push("acdef");
    EXPECT_TRUE(arr.refer() == 1);
    arr.begin();
    EXPECT_TRUE(arr.refer() == 0);
    arr.end();
    EXPECT_TRUE(arr.refer() == 0);

    JsonArray arr1 = arr;
    JsonArray arr2 = arr1;
    JsonArray arr3 = arr;
    EXPECT_TRUE(arr1.refer() == 2);
    EXPECT_TRUE(arr2.refer() == 2);
    EXPECT_TRUE(arr3.refer() == 1);
    EXPECT_TRUE(arr.refer() == 0);

    obj.set("1111", "null");
    EXPECT_TRUE(obj.refer() == 1);
    obj.rbegin();
    EXPECT_TRUE(obj.refer() == 0);
    obj.rend();
    EXPECT_TRUE(obj.refer() == 0);

    JsonObject obj1 = obj;
    JsonObject obj2 = obj1;
    JsonObject obj3 = obj;
    EXPECT_TRUE(obj1.refer() == 2);
    EXPECT_TRUE(obj2.refer() == 2);
    EXPECT_TRUE(obj3.refer() == 1);
    EXPECT_TRUE(obj.refer() == 0);
  }

  {
    short a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    unsigned short a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    int a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    unsigned int a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    long a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    unsigned long a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    long long a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    unsigned long long a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    float a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    double a = 1;
    enum { A = 1, B = 2 };
    EXPECT_TRUE(a == A);
    EXPECT_TRUE(A == a);
    EXPECT_TRUE(a != B);
    EXPECT_TRUE(B != a);
  }

  {
    EXPECT_TRUE(JsonString() == JsonString());
    EXPECT_TRUE(JsonValue() == JsonValue());
    EXPECT_TRUE(JsonObject() == JsonObject());
    EXPECT_TRUE(JsonArray() == JsonArray());

    EXPECT_FALSE((JsonString() != JsonString()));
    EXPECT_FALSE((JsonValue() != JsonValue()));
    EXPECT_FALSE((JsonObject() != JsonObject()));
    EXPECT_FALSE((JsonArray() != JsonArray()));

    EXPECT_TRUE(JsonString() != JsonValue());
    EXPECT_TRUE(JsonObject() != JsonValue());
    EXPECT_TRUE(JsonArray() != JsonValue());
    EXPECT_TRUE(JsonValue() != JsonString());
    EXPECT_TRUE(JsonValue() != JsonObject());
    EXPECT_TRUE(JsonValue() != JsonArray());

    EXPECT_FALSE((JsonString() == JsonValue()));
    EXPECT_FALSE((JsonObject() == JsonValue()));
    EXPECT_FALSE((JsonArray() == JsonValue()));
    EXPECT_FALSE((JsonValue() == JsonString()));
    EXPECT_FALSE((JsonValue() == JsonObject()));
    EXPECT_FALSE((JsonValue() == JsonArray()));

    EXPECT_TRUE(JsonString() == std::string());
    EXPECT_TRUE(std::string() == JsonString());
    EXPECT_FALSE((JsonString() != std::string()));
    EXPECT_FALSE((std::string() != JsonString()));

    EXPECT_TRUE(JsonString() == std::string(""));
    EXPECT_TRUE(std::string("") == JsonString());
    EXPECT_FALSE((JsonString() != std::string("")));
    EXPECT_FALSE((std::string("") != JsonString()));

    EXPECT_TRUE(JsonString("") == std::string());
    EXPECT_TRUE(std::string() == JsonString(""));
    EXPECT_FALSE((JsonString("") != std::string()));
    EXPECT_FALSE((std::string() != JsonString("")));
  }
}
