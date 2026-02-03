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

#include "sqlengine/analyzer/simple_rewriter.h"
#include <gtest/gtest.h>
#include "db/sqlengine/analyzer/query_info.h"
#include "db/sqlengine/sqlengine_impl.h"
#include "zvec/db/doc.h"
#include "zvec/db/schema.h"

namespace zvec::sqlengine {

class SimpleRewriterTest : public testing::Test {
 public:
  // Sets up the test fixture.
  static void SetUpTestSuite() {
    schema = std::make_shared<CollectionSchema>();
    auto &param = *schema;
    param.set_name("1collection");

    auto column1 = std::make_shared<FieldSchema>();
    auto vector_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    column1->set_name("face_feature");
    column1->set_index_params(vector_params);
    column1->set_dimension(4);
    column1->set_data_type(DataType::VECTOR_FP32);
    param.add_field(column1);

    auto column2 = std::make_shared<FieldSchema>();
    column2->set_name("age");
    column2->set_data_type(DataType::UINT32);
    param.add_field(column2);

    auto column_gender = std::make_shared<FieldSchema>();
    column_gender->set_name("gender");
    column_gender->set_data_type(DataType::UINT32);
    param.add_field(column_gender);

    auto column3 = std::make_shared<FieldSchema>();
    column3->set_name("category");
    column3->set_data_type(DataType::STRING);
    param.add_field(column3);

    auto column4 = std::make_shared<FieldSchema>();
    column4->set_name("face_feature");
    column4->set_dimension(4);
    column4->set_data_type(DataType::VECTOR_FP32);
    param.add_field(column4);

    auto column5 = std::make_shared<FieldSchema>();
    column5->set_name("filename");
    column5->set_dimension(5);
    column5->set_data_type(DataType::STRING);
    param.add_field(column5);

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("loc");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("fid");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("agent_id");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("state");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("categoryId");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("passed_days");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("category_in");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("category_out");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("intAttr");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("intAttr");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("strAttr");
      column->set_data_type(DataType::STRING);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("partitionName");
      column->set_data_type(DataType::STRING);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("doc_id");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("a");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("is_type1");
      column->set_data_type(DataType::BOOL);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("is_type2");
      column->set_data_type(DataType::BOOL);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("category_array");
      column->set_data_type(DataType::ARRAY_STRING);
      param.add_field(column);
    }
  }

  // Tears down the test fixture.
  static void TearDownTestSuite() {}

  QueryInfo::Ptr parse(const std::string &filter) {
    VectorQuery query;
    query.output_fields_ = {"*"};
    query.topk_ = 11;
    query.include_vector_ = false;
    query.filter_ = filter;

    auto engine = std::make_shared<SQLEngineImpl>(profiler_);
    auto ret = engine->parse_request(schema, query, nullptr);

    // ASSERT_TRUE(ret.has_value());
    QueryInfo::Ptr new_query_info = ret.value();
    return new_query_info;
  }


 protected:
  Profiler::Ptr profiler_{new Profiler};
  inline static CollectionSchema::Ptr schema;
};

class EqOrRewriteTest : public SimpleRewriterTest {};

TEST_F(EqOrRewriteTest, SimpleEqOr) {
  auto info = parse("age = 10 or age = 20 ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in (10, 20)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleManyEqOr) {
  auto info = parse(
      "age = 1 or age = 2 or age = 3 or age = 4 "
      "or age = 5 or age = 6 or age = 7 or age = 8 or age = 9 or age = 10 or "
      "age = 11 or age = 12 or age = 13 or age = 14 or age = 15 or age = 16 or "
      "age = 17 or age = 18 or age = 19 or age = 20 or age = 21 or age = 22 or "
      "age = 23 or age = 24 or age = 25 or age = 26 or age = 27 or age = 28 or "
      "age = 29 or age = 30 or age = 31 or age = 32 or age = 33 or age = 34 or "
      "age = 35 or age = 36 or age = 37 or age = 38 or age = 39 or age = 40 or "
      "age = 41 or age = 42 or age = 43 or age = 44 or age = 45 or age = 46 or "
      "age = 47 or age = 48 or age = 49 or age = 50 or age = 51 or age = 52 or "
      "age = 53 or age = 54 or age = 55 or age = 56 or age = 57 or age = 58 or "
      "age = 59 or age = 60 or age = 61 or age = 62 or age = 63 or age = 64 or "
      "age = 65 or age = 66 or age = 67 or age = 68 or age = 69 or age = 70 or "
      "age = 71 or age = 72 or age = 73 or age = 74 or age = 75 or age = 76 or "
      "age = 77 or age = 78 or age = 79 or age = 80 or age = 81 or age = 82 or "
      "age = 83 or age = 84 or age = 85 or age = 86 or age = 87 or age = 88 or "
      "age = 89 or age = 90 or age = 91 or age = 92 or age = 93 or age = 94 or "
      "age = 95 or age = 96 or age = 97 or age = 98 or age = 99 or age = 100");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "age in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, "
      "19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, "
      "37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, "
      "55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, "
      "73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, "
      "91, 92, 93, 94, 95, 96, 97, 98, 99, 100)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleManyEqOrParas) {
  auto info = parse(
      "age = 1 or age = 2 or age = 3 or age = 4 "
      "or age = 5 or age = 6 or (age = 7 or age = 8 or age = 9 or age = 10 or "
      "age = 11 or age = 12 or age = 13) or age = 14 or age = 15 or age = 16 "
      "or "
      "age = 17 or age = 18 or age = 19 or age = 20 or age = 21 or age = 22 or "
      "age = 23 or age = 24 or age = 25 or age = 26 or age = 27 or age = 28 or "
      "age = 29 or age = 30 or age = 31 or age = 32 or age = 33 or age = 34 or "
      "age = 35 or age = 36 or age = 37 or (age = 38 or age = 39 or age = 40 "
      "or "
      "age = 41 or age = 42 or age = 43 or age = 44 or age = 45 or age = 46 or "
      "age = 47 or age = 48 or age = 49 or age = 50 or age = 51 or age = 52 or "
      "age = 53 or age = 54 or age = 55 or age = 56 or age = 57 or age = 58 or "
      "age = 59 or age = 60 or age = 61 or age = 62 or age = 63 or age = 64 or "
      "age = 65 or age = 66 or age = 67 or age = 68 or age = 69 or age = 70 or "
      "age = 71 or age = 72 or age = 73 or age = 74 or age = 75 or age = 76 or "
      "age = 77 or age = 78 or age = 79 or age = 80 or age = 81 or age = 82 or "
      "age = 83 or age = 84 or age = 85) or age = 86 or age = 87 or age = 88 "
      "or "
      "age = 89 or age = 90 or age = 91 or age = 92 or age = 93 or age = 94 or "
      "age = 95 or age = 96 or age = 97 or (age = 98 or age = 99) or age = "
      "100");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "age in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, "
      "19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, "
      "37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, "
      "55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, "
      "73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, "
      "91, 92, 93, 94, 95, 96, 97, 98, 99, 100)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleNeOr) {
  auto info = parse("age != 10 or age != 20 ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in NOT (10, 20)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleManyNeOr) {
  auto info = parse(
      "age != 1 or age != 2 or age != 3 or age "
      "!= 4 or age != 5 or age != 6 or age != 7 or age != 8 or age != 9 or age "
      "!= 10 or age != 11 or age != 12 or age != 13 or age != 14 or age != 15 "
      "or age != 16 or age != 17 or age != 18 or age != 19 or age != 20");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "age in NOT (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
            "16, 17, 18, "
            "19, 20)(FORWARD)");
}

TEST_F(EqOrRewriteTest, EqAndNe) {
  auto info = parse(
      "age != 10 or age != 20 or age = 30 or "
      "age = 40");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(age in NOT (10, 20)(FORWARD)(OR_A)) or (age in (30, "
            "40)(FORWARD)(OR_A))");
}

TEST_F(EqOrRewriteTest, PreEqOr) {
  {
    auto info = parse(
        "gender =1 or age = 10 or age = 20 or "
        "age = 30 or age = 40");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "(gender=1(FORWARD)(OR_A)) or (age in (10, 20, 30, "
              "40)(FORWARD)(OR_A))");
  }
  {
    auto info = parse(
        "gender =1 and age = 10 or age = 20 or "
        "age = 30 or age = 40");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "((gender=1(FORWARD)(OR_A)) and (age=10(FORWARD)(OR_A))) or (age "
              "in (20, 30, 40)(FORWARD)(OR_A))");
  }
}

TEST_F(EqOrRewriteTest, PostEqOr) {
  {
    auto info = parse(
        "age = 10 or age = 20 or "
        "age = 30 or age = 40 or gender = 1");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "(age in (10, 20, 30, 40)(FORWARD)(OR_A)) or "
              "(gender=1(FORWARD)(OR_A))");
  }
  {
    auto info = parse(
        "age = 10 or age = 20 or "
        "age = 30 or age = 40 and gender = 1");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "(age in (10, 20, 30)(FORWARD)(OR_A)) or "
              "((age=40(FORWARD)(OR_A)) and (gender=1(FORWARD)(OR_A)))");
  }
}

TEST_F(EqOrRewriteTest, PreEqAnd) {
  auto info = parse(
      "gender =1 and (age = 10 or age = 20 or "
      "age = 30 or age = 40)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(gender=1(FORWARD)) and (age in (10, 20, 30, 40)(FORWARD))");
}

TEST_F(EqOrRewriteTest, PostEqAnd) {
  auto info = parse(
      "(age = 10 or age = 20 or "
      "age = 30 or age = 40) and gender=1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(age in (10, 20, 30, 40)(FORWARD)) and (gender=1(FORWARD))");
}

TEST_F(EqOrRewriteTest, PrePostEqAnd) {
  auto info = parse(
      "gender =1 and (age = 10 or age = 20 or "
      "age = 30 or age = 40) and loc != 3");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "((gender=1(FORWARD)) and (age in (10, 20, 30, 40)(FORWARD))) and "
            "(loc!=3(FORWARD))");
}

TEST_F(EqOrRewriteTest, UserCases1) {
  auto info = parse(
      "(agent_id=20) and state=1 and (fid=107 "
      "or fid=174 or fid=593 or fid=602 or fid=592 or fid=134 or fid=135 or "
      "fid=136 or fid=137 or fid=138 or fid=139 or fid=141 or fid=267 or "
      "fid=271 or fid=176 or fid=177 or fid=178 or fid=179 or fid=180 or "
      "fid=182 or fid=183 or fid=184 or fid=270 or fid=479 or fid=488 or "
      "fid=502 or fid=508 or fid=522 or fid=553 or fid=554 or fid=557 or "
      "fid=561 or fid=567 or fid=570 or fid=588 or fid=594 or fid=595 or "
      "fid=596 or fid=597 or fid=598 or fid=603 or fid=604 or fid=605 or "
      "fid=606 or fid=426 or fid=427 or fid=428 or fid=429 or fid=430 or "
      "fid=431 or fid=432 or fid=433 or fid=434 or fid=435 or fid=436 or "
      "fid=437 or fid=438 or fid=439 or fid=440 or fid=441 or fid=442 or "
      "fid=443 or fid=444 or fid=445 or fid=446 or fid=447 or fid=448 or "
      "fid=215 or fid=216 or fid=217 or fid=469 or fid=473 or fid=475 or "
      "fid=476 or fid=477 or fid=478 or fid=524 or fid=528 or fid=529 or "
      "fid=532 or fid=533 or fid=534 or fid=542 or fid=543 or fid=560 or "
      "fid=243 or fid=244 or fid=245 or fid=246 or fid=247 or fid=496 or "
      "fid=497 or fid=506 or fid=248 or fid=249 or fid=250 or fid=251 or "
      "fid=252 or fid=494 or fid=495 or fid=507 or fid=535 or fid=536 or "
      "fid=586 or fid=589 or fid=259 or fid=260 or fid=261 or fid=262 or "
      "fid=263 or fid=264 or fid=265 or fid=491 or fid=492 or fid=493 or "
      "fid=530 or fid=531 or fid=227 or fid=228 or fid=229 or fid=230 or "
      "fid=231 or fid=232 or fid=233 or fid=235 or fid=472 or fid=487 or "
      "fid=537 or fid=559 or fid=236 or fid=237 or fid=238 or fid=239 or "
      "fid=240 or fid=241 or fid=242 or fid=273 or fid=546 or fid=587 or "
      "fid=454 or fid=455 or fid=456 or fid=457 or fid=458 or fid=459 or "
      "fid=460 or fid=461 or fid=449 or fid=450 or fid=451 or fid=452 or "
      "fid=453 or fid=480 or fid=481 or fid=482 or fid=483 or fid=484 or "
      "fid=489 or fid=490 or fid=538 or fid=539 or fid=540 or fid=545 or "
      "fid=503 or fid=504 or fid=547 or fid=548 or fid=549 or fid=550 or "
      "fid=509 or fid=510 or fid=511 or fid=512 or fid=513 or fid=523 or "
      "fid=558 or fid=555 or fid=556 or fid=600 or fid=601 or fid=562 or "
      "fid=563 or fid=564 or fid=565 or fid=566 or fid=591 or fid=568 or "
      "fid=569 or fid=590 or fid=571 or fid=572 or fid=573 or fid=574 or "
      "fid=575 or fid=701 or fid=711 or fid=713 or fid=616 or fid=617 or "
      "fid=618 or fid=619 or fid=620 or fid=621 or fid=622 or fid=623 or "
      "fid=624 or fid=625 or fid=626 or fid=629 or fid=672 or fid=607 or "
      "fid=700 or fid=635 or fid=612 or fid=613 or fid=614 or fid=615 or "
      "fid=679 or fid=670 or fid=680 or fid=681 or fid=702 or fid=706 or "
      "fid=714 or fid=675 or fid=676 or fid=640 or fid=643 or fid=649 or "
      "fid=653 or fid=655 or fid=657 or fid=662 or fid=703 or fid=704 or "
      "fid=705 or fid=707 or fid=641 or fid=642 or fid=644 or fid=645 or "
      "fid=646 or fid=647 or fid=648 or fid=709 or fid=650 or fid=651 or "
      "fid=652 or fid=710 or fid=654 or fid=656 or fid=658 or fid=659 or "
      "fid=660 or fid=661 or fid=663 or fid=664 or fid=665 or fid=666 or "
      "fid=667 or fid=668 or fid=669 or fid=678)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "((agent_id=20(FORWARD)) and (state=1(FORWARD))) and (fid in (107, 174, "
      "593, 602, 592, 134, 135, 136, 137, 138, 139, 141, 267, 271, 176, 177, "
      "178, 179, 180, 182, 183, 184, 270, 479, 488, 502, 508, 522, 553, 554, "
      "557, 561, 567, 570, 588, 594, 595, 596, 597, 598, 603, 604, 605, 606, "
      "426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, "
      "440, 441, 442, 443, 444, 445, 446, 447, 448, 215, 216, 217, 469, 473, "
      "475, 476, 477, 478, 524, 528, 529, 532, 533, 534, 542, 543, 560, 243, "
      "244, 245, 246, 247, 496, 497, 506, 248, 249, 250, 251, 252, 494, 495, "
      "507, 535, 536, 586, 589, 259, 260, 261, 262, 263, 264, 265, 491, 492, "
      "493, 530, 531, 227, 228, 229, 230, 231, 232, 233, 235, 472, 487, 537, "
      "559, 236, 237, 238, 239, 240, 241, 242, 273, 546, 587, 454, 455, 456, "
      "457, 458, 459, 460, 461, 449, 450, 451, 452, 453, 480, 481, 482, 483, "
      "484, 489, 490, 538, 539, 540, 545, 503, 504, 547, 548, 549, 550, 509, "
      "510, 511, 512, 513, 523, 558, 555, 556, 600, 601, 562, 563, 564, 565, "
      "566, 591, 568, 569, 590, 571, 572, 573, 574, 575, 701, 711, 713, 616, "
      "617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 629, 672, 607, 700, "
      "635, 612, 613, 614, 615, 679, 670, 680, 681, 702, 706, 714, 675, 676, "
      "640, 643, 649, 653, 655, 657, 662, 703, 704, 705, 707, 641, 642, 644, "
      "645, 646, 647, 648, 709, 650, 651, 652, 710, 654, 656, 658, 659, 660, "
      "661, 663, 664, 665, 666, 667, 668, 669, 678)(FORWARD))");
}

TEST_F(EqOrRewriteTest, UserCases2) {
  auto info = parse(
      "partitionName = '114634' or "
      "partitionName = '114632' or partitionName = '114635' or partitionName = "
      "'114629' or partitionName = '114630' or partitionName = '114633' or "
      "partitionName = '114636' or partitionName = '114637' or partitionName = "
      "'114631'");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "partitionName in (114634, 114632, 114635, 114629, 114630, 114633, "
            "114636, 114637, 114631)(FORWARD)");
}

TEST_F(EqOrRewriteTest, UserCases3) {
  auto info = parse(
      "(doc_id=1319620650600837120 or "
      "doc_id=1319621497753739264 or doc_id=1319629144649367552 or "
      "doc_id=1319630319721377793 or doc_id=1319667286769324032 or "
      "doc_id=1319671157117808640 or doc_id=1319671403998793728 or "
      "doc_id=2319684930499055617 or doc_id=1319685259995140096)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "doc_id in (1319620650600837120, 1319621497753739264, "
            "1319629144649367552, 1319630319721377793, 1319667286769324032, "
            "1319671157117808640, 1319671403998793728, 2319684930499055617, "
            "1319685259995140096)(FORWARD)");
}

TEST_F(EqOrRewriteTest, UserCases4) {
  auto info = parse(
      "(strAttr ='' or strAttr = 'prd') and "
      "categoryId = 4");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(strAttr in (, prd)(FORWARD)) and (categoryId=4(FORWARD))");
}

TEST_F(EqOrRewriteTest, UserCases5) {
  auto info = parse(
      "intAttr = 1  OR intAttr = 5  OR intAttr "
      "= 6  OR intAttr = 9  and categoryId = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(intAttr in (1, 5, 6)(FORWARD)(OR_A)) or "
            "((intAttr=9(FORWARD)(OR_A)) and (categoryId=1(FORWARD)(OR_A)))");
}

TEST_F(EqOrRewriteTest, UserCases6) {
  auto info = parse(
      ""
      "filename='OhbVrpoi.pdf' or "
      "filename='wRyoG4dB.pdf' or "
      "filename='dJ3fawFf.pdf' or "
      "filename='ZJS9dk3Q.pdf' or "
      "filename='fY2JD8dL.pdf' or "
      "filename='HnJpdoxC.pdf' or "
      "filename='Hbxm1zvi.pdf' or "
      "filename='r5Q8cxHu.pdf' or "
      "filename='dwF9cZtI.pdf'");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "filename in (OhbVrpoi.pdf, "
            "wRyoG4dB.pdf, "
            "dJ3fawFf.pdf, "
            "ZJS9dk3Q.pdf, "
            "fY2JD8dL.pdf, "
            "HnJpdoxC.pdf, "
            "Hbxm1zvi.pdf, "
            "r5Q8cxHu.pdf, "
            "dwF9cZtI.pdf)(FORWARD)");
}

TEST_F(EqOrRewriteTest, NotChanged1) {
  auto info = parse(
      "passed_days>3 and (loc >= "
      "500 or age > 10)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(passed_days>3(FORWARD)) and ((loc>=500(FORWARD)(OR_A)) "
            "or (age>10(FORWARD)(OR_A)))");
}

TEST_F(EqOrRewriteTest, NotChanged2) {
  auto info = parse(
      "strAttr=\"online_252\" AND (intAttr > "
      "103775813 OR intAttr < 103775813) and categoryId = 88888888");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "((strAttr=online_252(FORWARD)) and ((intAttr>103775813(FORWARD)(OR_A)) "
      "or (intAttr<103775813(FORWARD)(OR_A)))) and "
      "(categoryId=88888888(FORWARD))");
}

TEST_F(EqOrRewriteTest, NotChanged3) {
  auto info = parse(
      "(is_type1 = true or is_type2 = "
      "true)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(is_type1=true(FORWARD)(OR_A)) or (is_type2=true(FORWARD)(OR_A))");
}

TEST_F(EqOrRewriteTest, NotChanged4) {
  auto info = parse("(a = 1 or a != 2)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(a=1(FORWARD)(OR_A)) or (a!=2(FORWARD)(OR_A))");
}

class ContainRewriteTest : public SimpleRewriterTest {};

TEST_F(ContainRewriteTest, ContainAllEmptySet) {
  auto info = parse("category_array contain_all ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "category_array IS_NOT_NULL (FORWARD)");
}

TEST_F(ContainRewriteTest, NotContainAllEmptySet) {
  auto info = parse("category_array not contain_all ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, NotContainAnyEmptySet) {
  auto info = parse("category_array not contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "category_array IS_NOT_NULL (FORWARD)");
}

TEST_F(ContainRewriteTest, ContainAnyEmptySet) {
  auto info = parse("category_array contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionAnd) {
  auto info = parse("category_array not contain_all () and a = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionMultiAnd) {
  auto info = parse(
      "category_array not contain_all () and a > 1 and a > 2 and a > 3 and a > "
      "4");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionOr) {
  auto info = parse("category_array not contain_all () or a = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "a=1(FORWARD)");
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionMultiOr) {
  auto info =
      parse("category_array not contain_all () or a > 1 or a > 2 or a > 3");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "((a>1(FORWARD)(OR_A)) or (a>2(FORWARD)(OR_A))) or (a>3(FORWARD)(OR_A))");
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionAndComplex) {
  auto info = parse("(a > 1 or a < 0) and category_array contain_any () ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionOrComplex) {
  auto info = parse("(a > 1 or a < 0) or category_array contain_any () ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), false);
  EXPECT_EQ(info->filter_cond()->text(),
            "(a>1(FORWARD)(OR_A)) or (a<0(FORWARD)(OR_A))");
}

TEST_F(SimpleRewriterTest, MiscOr) {
  auto info = parse("a = 1 or a = 2 or a = 3 or category_array contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), false);
  EXPECT_EQ(info->filter_cond()->text(), "a in (1, 2, 3)(FORWARD)");
}

TEST_F(SimpleRewriterTest, MiscAnd) {
  auto info =
      parse("(a = 1 or a = 2 or a = 3) and category_array contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}


}  // namespace zvec::sqlengine
