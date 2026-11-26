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

#define MATRIX_VAR_INIT_1X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  _VAR_TYPE _VAR_NAME##_0_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT_1X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_1X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_1 = (_VAR_INIT);

#define MATRIX_VAR_INIT_1X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_1X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_3 = (_VAR_INIT);

#define MATRIX_VAR_INIT_1X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_1X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_7 = (_VAR_INIT);

#define MATRIX_VAR_INIT_1X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_1X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT)        \
  _VAR_TYPE _VAR_NAME##_0_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_15 = (_VAR_INIT);

#define MATRIX_VAR_INIT_2X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_1X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_1_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT_2X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_2X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_1 = (_VAR_INIT);

#define MATRIX_VAR_INIT_2X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_2X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_3 = (_VAR_INIT);

#define MATRIX_VAR_INIT_2X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_2X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_7 = (_VAR_INIT);

#define MATRIX_VAR_INIT_2X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_2X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT)        \
  _VAR_TYPE _VAR_NAME##_0_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_15 = (_VAR_INIT);

#define MATRIX_VAR_INIT_2X32(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_2X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_31 = (_VAR_INIT);

#define MATRIX_VAR_INIT_4X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_2X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_2_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT_4X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_4X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_1 = (_VAR_INIT);

#define MATRIX_VAR_INIT_4X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_4X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_3 = (_VAR_INIT);

#define MATRIX_VAR_INIT_4X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_4X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_7 = (_VAR_INIT);

#define MATRIX_VAR_INIT_4X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_4X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT)        \
  _VAR_TYPE _VAR_NAME##_0_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_15 = (_VAR_INIT);

#define MATRIX_VAR_INIT_4X32(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_4X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_31 = (_VAR_INIT);

#define MATRIX_VAR_INIT_8X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_4X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_4_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT_8X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_8X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_1 = (_VAR_INIT);

#define MATRIX_VAR_INIT_8X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_8X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_3 = (_VAR_INIT);

#define MATRIX_VAR_INIT_8X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_8X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_4 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_5 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_6 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_7 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_7 = (_VAR_INIT);

#define MATRIX_VAR_INIT_8X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_8X8(_VAR_TYPE, _VAR_NAME, _VAR_INIT)        \
  _VAR_TYPE _VAR_NAME##_0_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_4_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_5_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_6_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_7_8 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_4_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_5_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_6_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_7_9 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_0_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_10 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_11 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_12 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_13 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_14 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_15 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_15 = (_VAR_INIT);

#define MATRIX_VAR_INIT_8X32(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_8X16(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_16 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_17 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_18 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_19 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_20 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_21 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_22 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_23 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_24 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_25 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_26 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_27 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_28 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_29 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_30 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_1_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_2_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_3_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_4_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_5_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_6_31 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_7_31 = (_VAR_INIT);

#define MATRIX_VAR_INIT_16X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_8X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)        \
  _VAR_TYPE _VAR_NAME##_8_0 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_9_0 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_10_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_11_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_12_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_13_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_14_0 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_15_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT_16X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_16X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_4_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_5_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_6_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_7_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_8_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_9_1 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_10_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_11_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_12_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_13_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_14_1 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_15_1 = (_VAR_INIT);

#define MATRIX_VAR_INIT_16X4(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_16X2(_VAR_TYPE, _VAR_NAME, _VAR_INIT)       \
  _VAR_TYPE _VAR_NAME##_0_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_4_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_5_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_6_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_7_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_8_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_9_2 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_10_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_11_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_12_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_13_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_14_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_15_2 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_0_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_1_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_2_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_3_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_4_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_5_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_6_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_7_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_8_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_9_3 = (_VAR_INIT);                    \
  _VAR_TYPE _VAR_NAME##_10_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_11_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_12_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_13_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_14_3 = (_VAR_INIT);                   \
  _VAR_TYPE _VAR_NAME##_15_3 = (_VAR_INIT);

#define MATRIX_VAR_STORE_1X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...) \
  _STORE((_ARRAY) + (_STEP) * (0), _NORM((_VAR##_0_0), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_1X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_1X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (1), _NORM((_VAR##_0_1), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_1X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_1X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (2), _NORM((_VAR##_0_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (3), _NORM((_VAR##_0_3), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_1X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_1X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (4), _NORM((_VAR##_0_4), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (5), _NORM((_VAR##_0_5), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (6), _NORM((_VAR##_0_6), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (7), _NORM((_VAR##_0_7), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_1X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)    \
  MATRIX_VAR_STORE_1X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (8), _NORM((_VAR##_0_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (9), _NORM((_VAR##_0_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (10), _NORM((_VAR##_0_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (11), _NORM((_VAR##_0_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (12), _NORM((_VAR##_0_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (13), _NORM((_VAR##_0_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (14), _NORM((_VAR##_0_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (15), _NORM((_VAR##_0_15), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_2X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_1X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (1), _NORM((_VAR##_1_0), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_2X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_2X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (2), _NORM((_VAR##_0_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (3), _NORM((_VAR##_1_1), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_2X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_2X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (4), _NORM((_VAR##_0_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (5), _NORM((_VAR##_1_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (6), _NORM((_VAR##_0_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (7), _NORM((_VAR##_1_3), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_2X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_2X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (8), _NORM((_VAR##_0_4), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (9), _NORM((_VAR##_1_4), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (10), _NORM((_VAR##_0_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (11), _NORM((_VAR##_1_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (12), _NORM((_VAR##_0_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (13), _NORM((_VAR##_1_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (14), _NORM((_VAR##_0_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (15), _NORM((_VAR##_1_7), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_2X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)    \
  MATRIX_VAR_STORE_2X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (16), _NORM((_VAR##_0_8), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (17), _NORM((_VAR##_1_8), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (18), _NORM((_VAR##_0_9), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (19), _NORM((_VAR##_1_9), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (20), _NORM((_VAR##_0_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (21), _NORM((_VAR##_1_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (22), _NORM((_VAR##_0_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (23), _NORM((_VAR##_1_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (24), _NORM((_VAR##_0_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (25), _NORM((_VAR##_1_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (26), _NORM((_VAR##_0_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (27), _NORM((_VAR##_1_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (28), _NORM((_VAR##_0_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (29), _NORM((_VAR##_1_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (30), _NORM((_VAR##_0_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (31), _NORM((_VAR##_1_15), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_2X32(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_2X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (32), _NORM((_VAR##_0_16), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (33), _NORM((_VAR##_1_16), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (34), _NORM((_VAR##_0_17), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (35), _NORM((_VAR##_1_17), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (36), _NORM((_VAR##_0_18), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (37), _NORM((_VAR##_1_18), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (38), _NORM((_VAR##_0_19), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (39), _NORM((_VAR##_1_19), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (40), _NORM((_VAR##_0_20), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (41), _NORM((_VAR##_1_20), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (42), _NORM((_VAR##_0_21), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (43), _NORM((_VAR##_1_21), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (44), _NORM((_VAR##_0_22), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (45), _NORM((_VAR##_1_22), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (46), _NORM((_VAR##_0_23), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (47), _NORM((_VAR##_1_23), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (48), _NORM((_VAR##_0_24), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (49), _NORM((_VAR##_1_24), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (50), _NORM((_VAR##_0_25), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (51), _NORM((_VAR##_1_25), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (52), _NORM((_VAR##_0_26), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (53), _NORM((_VAR##_1_26), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (54), _NORM((_VAR##_0_27), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (55), _NORM((_VAR##_1_27), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (56), _NORM((_VAR##_0_28), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (57), _NORM((_VAR##_1_28), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (58), _NORM((_VAR##_0_29), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (59), _NORM((_VAR##_1_29), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (60), _NORM((_VAR##_0_30), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (61), _NORM((_VAR##_1_30), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (62), _NORM((_VAR##_0_31), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (63), _NORM((_VAR##_1_31), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_4X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_2X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (2), _NORM((_VAR##_2_0), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (3), _NORM((_VAR##_3_0), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_4X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_4X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (4), _NORM((_VAR##_0_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (5), _NORM((_VAR##_1_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (6), _NORM((_VAR##_2_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (7), _NORM((_VAR##_3_1), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_4X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_4X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (8), _NORM((_VAR##_0_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (9), _NORM((_VAR##_1_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (10), _NORM((_VAR##_2_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (11), _NORM((_VAR##_3_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (12), _NORM((_VAR##_0_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (13), _NORM((_VAR##_1_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (14), _NORM((_VAR##_2_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (15), _NORM((_VAR##_3_3), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_4X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_4X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (16), _NORM((_VAR##_0_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (17), _NORM((_VAR##_1_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (18), _NORM((_VAR##_2_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (19), _NORM((_VAR##_3_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (20), _NORM((_VAR##_0_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (21), _NORM((_VAR##_1_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (22), _NORM((_VAR##_2_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (23), _NORM((_VAR##_3_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (24), _NORM((_VAR##_0_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (25), _NORM((_VAR##_1_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (26), _NORM((_VAR##_2_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (27), _NORM((_VAR##_3_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (28), _NORM((_VAR##_0_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (29), _NORM((_VAR##_1_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (30), _NORM((_VAR##_2_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (31), _NORM((_VAR##_3_7), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_4X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)    \
  MATRIX_VAR_STORE_4X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (32), _NORM((_VAR##_0_8), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (33), _NORM((_VAR##_1_8), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (34), _NORM((_VAR##_2_8), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (35), _NORM((_VAR##_3_8), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (36), _NORM((_VAR##_0_9), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (37), _NORM((_VAR##_1_9), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (38), _NORM((_VAR##_2_9), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (39), _NORM((_VAR##_3_9), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (40), _NORM((_VAR##_0_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (41), _NORM((_VAR##_1_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (42), _NORM((_VAR##_2_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (43), _NORM((_VAR##_3_10), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (44), _NORM((_VAR##_0_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (45), _NORM((_VAR##_1_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (46), _NORM((_VAR##_2_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (47), _NORM((_VAR##_3_11), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (48), _NORM((_VAR##_0_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (49), _NORM((_VAR##_1_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (50), _NORM((_VAR##_2_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (51), _NORM((_VAR##_3_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (52), _NORM((_VAR##_0_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (53), _NORM((_VAR##_1_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (54), _NORM((_VAR##_2_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (55), _NORM((_VAR##_3_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (56), _NORM((_VAR##_0_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (57), _NORM((_VAR##_1_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (58), _NORM((_VAR##_2_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (59), _NORM((_VAR##_3_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (60), _NORM((_VAR##_0_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (61), _NORM((_VAR##_1_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (62), _NORM((_VAR##_2_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (63), _NORM((_VAR##_3_15), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_4X32(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_4X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (64), _NORM((_VAR##_0_16), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (65), _NORM((_VAR##_1_16), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (66), _NORM((_VAR##_2_16), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (67), _NORM((_VAR##_3_16), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (68), _NORM((_VAR##_0_17), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (69), _NORM((_VAR##_1_17), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (70), _NORM((_VAR##_2_17), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (71), _NORM((_VAR##_3_17), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (72), _NORM((_VAR##_0_18), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (73), _NORM((_VAR##_1_18), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (74), _NORM((_VAR##_2_18), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (75), _NORM((_VAR##_3_18), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (76), _NORM((_VAR##_0_19), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (77), _NORM((_VAR##_1_19), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (78), _NORM((_VAR##_2_19), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (79), _NORM((_VAR##_3_19), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (80), _NORM((_VAR##_0_20), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (81), _NORM((_VAR##_1_20), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (82), _NORM((_VAR##_2_20), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (83), _NORM((_VAR##_3_20), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (84), _NORM((_VAR##_0_21), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (85), _NORM((_VAR##_1_21), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (86), _NORM((_VAR##_2_21), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (87), _NORM((_VAR##_3_21), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (88), _NORM((_VAR##_0_22), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (89), _NORM((_VAR##_1_22), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (90), _NORM((_VAR##_2_22), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (91), _NORM((_VAR##_3_22), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (92), _NORM((_VAR##_0_23), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (93), _NORM((_VAR##_1_23), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (94), _NORM((_VAR##_2_23), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (95), _NORM((_VAR##_3_23), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (96), _NORM((_VAR##_0_24), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (97), _NORM((_VAR##_1_24), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (98), _NORM((_VAR##_2_24), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (99), _NORM((_VAR##_3_24), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (100), _NORM((_VAR##_0_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (101), _NORM((_VAR##_1_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (102), _NORM((_VAR##_2_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (103), _NORM((_VAR##_3_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (104), _NORM((_VAR##_0_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (105), _NORM((_VAR##_1_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (106), _NORM((_VAR##_2_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (107), _NORM((_VAR##_3_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (108), _NORM((_VAR##_0_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (109), _NORM((_VAR##_1_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (110), _NORM((_VAR##_2_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (111), _NORM((_VAR##_3_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (112), _NORM((_VAR##_0_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (113), _NORM((_VAR##_1_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (114), _NORM((_VAR##_2_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (115), _NORM((_VAR##_3_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (116), _NORM((_VAR##_0_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (117), _NORM((_VAR##_1_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (118), _NORM((_VAR##_2_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (119), _NORM((_VAR##_3_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (120), _NORM((_VAR##_0_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (121), _NORM((_VAR##_1_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (122), _NORM((_VAR##_2_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (123), _NORM((_VAR##_3_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (124), _NORM((_VAR##_0_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (125), _NORM((_VAR##_1_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (126), _NORM((_VAR##_2_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (127), _NORM((_VAR##_3_31), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_8X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_4X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (4), _NORM((_VAR##_4_0), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (5), _NORM((_VAR##_5_0), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (6), _NORM((_VAR##_6_0), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (7), _NORM((_VAR##_7_0), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_8X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_8X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (8), _NORM((_VAR##_0_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (9), _NORM((_VAR##_1_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (10), _NORM((_VAR##_2_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (11), _NORM((_VAR##_3_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (12), _NORM((_VAR##_4_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (13), _NORM((_VAR##_5_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (14), _NORM((_VAR##_6_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (15), _NORM((_VAR##_7_1), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_8X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_8X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (16), _NORM((_VAR##_0_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (17), _NORM((_VAR##_1_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (18), _NORM((_VAR##_2_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (19), _NORM((_VAR##_3_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (20), _NORM((_VAR##_4_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (21), _NORM((_VAR##_5_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (22), _NORM((_VAR##_6_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (23), _NORM((_VAR##_7_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (24), _NORM((_VAR##_0_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (25), _NORM((_VAR##_1_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (26), _NORM((_VAR##_2_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (27), _NORM((_VAR##_3_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (28), _NORM((_VAR##_4_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (29), _NORM((_VAR##_5_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (30), _NORM((_VAR##_6_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (31), _NORM((_VAR##_7_3), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_8X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_8X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (32), _NORM((_VAR##_0_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (33), _NORM((_VAR##_1_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (34), _NORM((_VAR##_2_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (35), _NORM((_VAR##_3_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (36), _NORM((_VAR##_4_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (37), _NORM((_VAR##_5_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (38), _NORM((_VAR##_6_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (39), _NORM((_VAR##_7_4), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (40), _NORM((_VAR##_0_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (41), _NORM((_VAR##_1_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (42), _NORM((_VAR##_2_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (43), _NORM((_VAR##_3_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (44), _NORM((_VAR##_4_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (45), _NORM((_VAR##_5_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (46), _NORM((_VAR##_6_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (47), _NORM((_VAR##_7_5), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (48), _NORM((_VAR##_0_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (49), _NORM((_VAR##_1_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (50), _NORM((_VAR##_2_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (51), _NORM((_VAR##_3_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (52), _NORM((_VAR##_4_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (53), _NORM((_VAR##_5_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (54), _NORM((_VAR##_6_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (55), _NORM((_VAR##_7_6), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (56), _NORM((_VAR##_0_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (57), _NORM((_VAR##_1_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (58), _NORM((_VAR##_2_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (59), _NORM((_VAR##_3_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (60), _NORM((_VAR##_4_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (61), _NORM((_VAR##_5_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (62), _NORM((_VAR##_6_7), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (63), _NORM((_VAR##_7_7), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_8X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_8X8(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__)  \
  _STORE((_ARRAY) + (_STEP) * (64), _NORM((_VAR##_0_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (65), _NORM((_VAR##_1_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (66), _NORM((_VAR##_2_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (67), _NORM((_VAR##_3_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (68), _NORM((_VAR##_4_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (69), _NORM((_VAR##_5_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (70), _NORM((_VAR##_6_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (71), _NORM((_VAR##_7_8), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (72), _NORM((_VAR##_0_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (73), _NORM((_VAR##_1_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (74), _NORM((_VAR##_2_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (75), _NORM((_VAR##_3_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (76), _NORM((_VAR##_4_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (77), _NORM((_VAR##_5_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (78), _NORM((_VAR##_6_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (79), _NORM((_VAR##_7_9), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (80), _NORM((_VAR##_0_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (81), _NORM((_VAR##_1_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (82), _NORM((_VAR##_2_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (83), _NORM((_VAR##_3_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (84), _NORM((_VAR##_4_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (85), _NORM((_VAR##_5_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (86), _NORM((_VAR##_6_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (87), _NORM((_VAR##_7_10), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (88), _NORM((_VAR##_0_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (89), _NORM((_VAR##_1_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (90), _NORM((_VAR##_2_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (91), _NORM((_VAR##_3_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (92), _NORM((_VAR##_4_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (93), _NORM((_VAR##_5_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (94), _NORM((_VAR##_6_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (95), _NORM((_VAR##_7_11), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (96), _NORM((_VAR##_0_12), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (97), _NORM((_VAR##_1_12), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (98), _NORM((_VAR##_2_12), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (99), _NORM((_VAR##_3_12), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (100), _NORM((_VAR##_4_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (101), _NORM((_VAR##_5_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (102), _NORM((_VAR##_6_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (103), _NORM((_VAR##_7_12), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (104), _NORM((_VAR##_0_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (105), _NORM((_VAR##_1_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (106), _NORM((_VAR##_2_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (107), _NORM((_VAR##_3_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (108), _NORM((_VAR##_4_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (109), _NORM((_VAR##_5_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (110), _NORM((_VAR##_6_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (111), _NORM((_VAR##_7_13), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (112), _NORM((_VAR##_0_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (113), _NORM((_VAR##_1_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (114), _NORM((_VAR##_2_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (115), _NORM((_VAR##_3_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (116), _NORM((_VAR##_4_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (117), _NORM((_VAR##_5_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (118), _NORM((_VAR##_6_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (119), _NORM((_VAR##_7_14), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (120), _NORM((_VAR##_0_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (121), _NORM((_VAR##_1_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (122), _NORM((_VAR##_2_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (123), _NORM((_VAR##_3_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (124), _NORM((_VAR##_4_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (125), _NORM((_VAR##_5_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (126), _NORM((_VAR##_6_15), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (127), _NORM((_VAR##_7_15), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_8X32(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_8X16(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (128), _NORM((_VAR##_0_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (129), _NORM((_VAR##_1_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (130), _NORM((_VAR##_2_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (131), _NORM((_VAR##_3_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (132), _NORM((_VAR##_4_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (133), _NORM((_VAR##_5_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (134), _NORM((_VAR##_6_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (135), _NORM((_VAR##_7_16), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (136), _NORM((_VAR##_0_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (137), _NORM((_VAR##_1_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (138), _NORM((_VAR##_2_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (139), _NORM((_VAR##_3_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (140), _NORM((_VAR##_4_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (141), _NORM((_VAR##_5_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (142), _NORM((_VAR##_6_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (143), _NORM((_VAR##_7_17), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (144), _NORM((_VAR##_0_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (145), _NORM((_VAR##_1_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (146), _NORM((_VAR##_2_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (147), _NORM((_VAR##_3_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (148), _NORM((_VAR##_4_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (149), _NORM((_VAR##_5_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (150), _NORM((_VAR##_6_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (151), _NORM((_VAR##_7_18), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (152), _NORM((_VAR##_0_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (153), _NORM((_VAR##_1_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (154), _NORM((_VAR##_2_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (155), _NORM((_VAR##_3_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (156), _NORM((_VAR##_4_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (157), _NORM((_VAR##_5_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (158), _NORM((_VAR##_6_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (159), _NORM((_VAR##_7_19), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (160), _NORM((_VAR##_0_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (161), _NORM((_VAR##_1_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (162), _NORM((_VAR##_2_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (163), _NORM((_VAR##_3_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (164), _NORM((_VAR##_4_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (165), _NORM((_VAR##_5_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (166), _NORM((_VAR##_6_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (167), _NORM((_VAR##_7_20), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (168), _NORM((_VAR##_0_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (169), _NORM((_VAR##_1_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (170), _NORM((_VAR##_2_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (171), _NORM((_VAR##_3_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (172), _NORM((_VAR##_4_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (173), _NORM((_VAR##_5_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (174), _NORM((_VAR##_6_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (175), _NORM((_VAR##_7_21), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (176), _NORM((_VAR##_0_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (177), _NORM((_VAR##_1_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (178), _NORM((_VAR##_2_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (179), _NORM((_VAR##_3_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (180), _NORM((_VAR##_4_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (181), _NORM((_VAR##_5_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (182), _NORM((_VAR##_6_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (183), _NORM((_VAR##_7_22), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (184), _NORM((_VAR##_0_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (185), _NORM((_VAR##_1_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (186), _NORM((_VAR##_2_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (187), _NORM((_VAR##_3_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (188), _NORM((_VAR##_4_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (189), _NORM((_VAR##_5_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (190), _NORM((_VAR##_6_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (191), _NORM((_VAR##_7_23), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (192), _NORM((_VAR##_0_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (193), _NORM((_VAR##_1_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (194), _NORM((_VAR##_2_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (195), _NORM((_VAR##_3_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (196), _NORM((_VAR##_4_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (197), _NORM((_VAR##_5_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (198), _NORM((_VAR##_6_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (199), _NORM((_VAR##_7_24), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (200), _NORM((_VAR##_0_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (201), _NORM((_VAR##_1_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (202), _NORM((_VAR##_2_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (203), _NORM((_VAR##_3_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (204), _NORM((_VAR##_4_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (205), _NORM((_VAR##_5_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (206), _NORM((_VAR##_6_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (207), _NORM((_VAR##_7_25), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (208), _NORM((_VAR##_0_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (209), _NORM((_VAR##_1_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (210), _NORM((_VAR##_2_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (211), _NORM((_VAR##_3_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (212), _NORM((_VAR##_4_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (213), _NORM((_VAR##_5_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (214), _NORM((_VAR##_6_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (215), _NORM((_VAR##_7_26), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (216), _NORM((_VAR##_0_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (217), _NORM((_VAR##_1_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (218), _NORM((_VAR##_2_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (219), _NORM((_VAR##_3_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (220), _NORM((_VAR##_4_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (221), _NORM((_VAR##_5_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (222), _NORM((_VAR##_6_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (223), _NORM((_VAR##_7_27), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (224), _NORM((_VAR##_0_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (225), _NORM((_VAR##_1_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (226), _NORM((_VAR##_2_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (227), _NORM((_VAR##_3_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (228), _NORM((_VAR##_4_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (229), _NORM((_VAR##_5_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (230), _NORM((_VAR##_6_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (231), _NORM((_VAR##_7_28), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (232), _NORM((_VAR##_0_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (233), _NORM((_VAR##_1_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (234), _NORM((_VAR##_2_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (235), _NORM((_VAR##_3_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (236), _NORM((_VAR##_4_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (237), _NORM((_VAR##_5_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (238), _NORM((_VAR##_6_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (239), _NORM((_VAR##_7_29), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (240), _NORM((_VAR##_0_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (241), _NORM((_VAR##_1_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (242), _NORM((_VAR##_2_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (243), _NORM((_VAR##_3_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (244), _NORM((_VAR##_4_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (245), _NORM((_VAR##_5_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (246), _NORM((_VAR##_6_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (247), _NORM((_VAR##_7_30), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (248), _NORM((_VAR##_0_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (249), _NORM((_VAR##_1_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (250), _NORM((_VAR##_2_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (251), _NORM((_VAR##_3_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (252), _NORM((_VAR##_4_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (253), _NORM((_VAR##_5_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (254), _NORM((_VAR##_6_31), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (255), _NORM((_VAR##_7_31), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_16X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)    \
  MATRIX_VAR_STORE_8X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (8), _NORM((_VAR##_8_0), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (9), _NORM((_VAR##_9_0), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (10), _NORM((_VAR##_10_0), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (11), _NORM((_VAR##_11_0), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (12), _NORM((_VAR##_12_0), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (13), _NORM((_VAR##_13_0), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (14), _NORM((_VAR##_14_0), ##__VA_ARGS__)); \
  _STORE((_ARRAY) + (_STEP) * (15), _NORM((_VAR##_15_0), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_16X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_16X1(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (16), _NORM((_VAR##_0_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (17), _NORM((_VAR##_1_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (18), _NORM((_VAR##_2_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (19), _NORM((_VAR##_3_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (20), _NORM((_VAR##_4_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (21), _NORM((_VAR##_5_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (22), _NORM((_VAR##_6_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (23), _NORM((_VAR##_7_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (24), _NORM((_VAR##_8_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (25), _NORM((_VAR##_9_1), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (26), _NORM((_VAR##_10_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (27), _NORM((_VAR##_11_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (28), _NORM((_VAR##_12_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (29), _NORM((_VAR##_13_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (30), _NORM((_VAR##_14_1), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (31), _NORM((_VAR##_15_1), ##__VA_ARGS__));

#define MATRIX_VAR_STORE_16X4(_STEP, _VAR, _ARRAY, _STORE, _NORM, ...)     \
  MATRIX_VAR_STORE_16X2(_STEP, _VAR, _ARRAY, _STORE, _NORM, ##__VA_ARGS__) \
  _STORE((_ARRAY) + (_STEP) * (32), _NORM((_VAR##_0_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (33), _NORM((_VAR##_1_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (34), _NORM((_VAR##_2_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (35), _NORM((_VAR##_3_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (36), _NORM((_VAR##_4_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (37), _NORM((_VAR##_5_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (38), _NORM((_VAR##_6_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (39), _NORM((_VAR##_7_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (40), _NORM((_VAR##_8_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (41), _NORM((_VAR##_9_2), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (42), _NORM((_VAR##_10_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (43), _NORM((_VAR##_11_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (44), _NORM((_VAR##_12_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (45), _NORM((_VAR##_13_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (46), _NORM((_VAR##_14_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (47), _NORM((_VAR##_15_2), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (48), _NORM((_VAR##_0_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (49), _NORM((_VAR##_1_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (50), _NORM((_VAR##_2_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (51), _NORM((_VAR##_3_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (52), _NORM((_VAR##_4_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (53), _NORM((_VAR##_5_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (54), _NORM((_VAR##_6_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (55), _NORM((_VAR##_7_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (56), _NORM((_VAR##_8_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (57), _NORM((_VAR##_9_3), ##__VA_ARGS__));   \
  _STORE((_ARRAY) + (_STEP) * (58), _NORM((_VAR##_10_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (59), _NORM((_VAR##_11_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (60), _NORM((_VAR##_12_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (61), _NORM((_VAR##_13_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (62), _NORM((_VAR##_14_3), ##__VA_ARGS__));  \
  _STORE((_ARRAY) + (_STEP) * (63), _NORM((_VAR##_15_3), ##__VA_ARGS__));

#define MATRIX_VAR_PERMUTE_1X1(_VAR, _PERMUTE, ...) \
  (_VAR##_0_0) = _PERMUTE((_VAR##_0_0), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_1X2(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_1X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_1) = _PERMUTE((_VAR##_0_1), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_1X4(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_1X2(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_2) = _PERMUTE((_VAR##_0_2), ##__VA_ARGS__); \
  (_VAR##_0_3) = _PERMUTE((_VAR##_0_3), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_1X8(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_1X4(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_4) = _PERMUTE((_VAR##_0_4), ##__VA_ARGS__); \
  (_VAR##_0_5) = _PERMUTE((_VAR##_0_5), ##__VA_ARGS__); \
  (_VAR##_0_6) = _PERMUTE((_VAR##_0_6), ##__VA_ARGS__); \
  (_VAR##_0_7) = _PERMUTE((_VAR##_0_7), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_1X16(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_1X8(_VAR, _PERMUTE, ##__VA_ARGS__)   \
  (_VAR##_0_8) = _PERMUTE((_VAR##_0_8), ##__VA_ARGS__);   \
  (_VAR##_0_9) = _PERMUTE((_VAR##_0_9), ##__VA_ARGS__);   \
  (_VAR##_0_10) = _PERMUTE((_VAR##_0_10), ##__VA_ARGS__); \
  (_VAR##_0_11) = _PERMUTE((_VAR##_0_11), ##__VA_ARGS__); \
  (_VAR##_0_12) = _PERMUTE((_VAR##_0_12), ##__VA_ARGS__); \
  (_VAR##_0_13) = _PERMUTE((_VAR##_0_13), ##__VA_ARGS__); \
  (_VAR##_0_14) = _PERMUTE((_VAR##_0_14), ##__VA_ARGS__); \
  (_VAR##_0_15) = _PERMUTE((_VAR##_0_15), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_2X1(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_1X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_1_0) = _PERMUTE((_VAR##_1_0), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_2X2(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_2X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_1) = _PERMUTE((_VAR##_0_1), ##__VA_ARGS__); \
  (_VAR##_1_1) = _PERMUTE((_VAR##_1_1), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_2X4(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_2X2(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_2) = _PERMUTE((_VAR##_0_2), ##__VA_ARGS__); \
  (_VAR##_1_2) = _PERMUTE((_VAR##_1_2), ##__VA_ARGS__); \
  (_VAR##_0_3) = _PERMUTE((_VAR##_0_3), ##__VA_ARGS__); \
  (_VAR##_1_3) = _PERMUTE((_VAR##_1_3), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_2X8(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_2X4(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_4) = _PERMUTE((_VAR##_0_4), ##__VA_ARGS__); \
  (_VAR##_1_4) = _PERMUTE((_VAR##_1_4), ##__VA_ARGS__); \
  (_VAR##_0_5) = _PERMUTE((_VAR##_0_5), ##__VA_ARGS__); \
  (_VAR##_1_5) = _PERMUTE((_VAR##_1_5), ##__VA_ARGS__); \
  (_VAR##_0_6) = _PERMUTE((_VAR##_0_6), ##__VA_ARGS__); \
  (_VAR##_1_6) = _PERMUTE((_VAR##_1_6), ##__VA_ARGS__); \
  (_VAR##_0_7) = _PERMUTE((_VAR##_0_7), ##__VA_ARGS__); \
  (_VAR##_1_7) = _PERMUTE((_VAR##_1_7), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_2X16(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_2X8(_VAR, _PERMUTE, ##__VA_ARGS__)   \
  (_VAR##_0_8) = _PERMUTE((_VAR##_0_8), ##__VA_ARGS__);   \
  (_VAR##_1_8) = _PERMUTE((_VAR##_1_8), ##__VA_ARGS__);   \
  (_VAR##_0_9) = _PERMUTE((_VAR##_0_9), ##__VA_ARGS__);   \
  (_VAR##_1_9) = _PERMUTE((_VAR##_1_9), ##__VA_ARGS__);   \
  (_VAR##_0_10) = _PERMUTE((_VAR##_0_10), ##__VA_ARGS__); \
  (_VAR##_1_10) = _PERMUTE((_VAR##_1_10), ##__VA_ARGS__); \
  (_VAR##_0_11) = _PERMUTE((_VAR##_0_11), ##__VA_ARGS__); \
  (_VAR##_1_11) = _PERMUTE((_VAR##_1_11), ##__VA_ARGS__); \
  (_VAR##_0_12) = _PERMUTE((_VAR##_0_12), ##__VA_ARGS__); \
  (_VAR##_1_12) = _PERMUTE((_VAR##_1_12), ##__VA_ARGS__); \
  (_VAR##_0_13) = _PERMUTE((_VAR##_0_13), ##__VA_ARGS__); \
  (_VAR##_1_13) = _PERMUTE((_VAR##_1_13), ##__VA_ARGS__); \
  (_VAR##_0_14) = _PERMUTE((_VAR##_0_14), ##__VA_ARGS__); \
  (_VAR##_1_14) = _PERMUTE((_VAR##_1_14), ##__VA_ARGS__); \
  (_VAR##_0_15) = _PERMUTE((_VAR##_0_15), ##__VA_ARGS__); \
  (_VAR##_1_15) = _PERMUTE((_VAR##_1_15), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_2X32(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_2X16(_VAR, _PERMUTE, ##__VA_ARGS__)  \
  (_VAR##_0_16) = _PERMUTE((_VAR##_0_16), ##__VA_ARGS__); \
  (_VAR##_1_16) = _PERMUTE((_VAR##_1_16), ##__VA_ARGS__); \
  (_VAR##_0_17) = _PERMUTE((_VAR##_0_17), ##__VA_ARGS__); \
  (_VAR##_1_17) = _PERMUTE((_VAR##_1_17), ##__VA_ARGS__); \
  (_VAR##_0_18) = _PERMUTE((_VAR##_0_18), ##__VA_ARGS__); \
  (_VAR##_1_18) = _PERMUTE((_VAR##_1_18), ##__VA_ARGS__); \
  (_VAR##_0_19) = _PERMUTE((_VAR##_0_19), ##__VA_ARGS__); \
  (_VAR##_1_19) = _PERMUTE((_VAR##_1_19), ##__VA_ARGS__); \
  (_VAR##_0_20) = _PERMUTE((_VAR##_0_20), ##__VA_ARGS__); \
  (_VAR##_1_20) = _PERMUTE((_VAR##_1_20), ##__VA_ARGS__); \
  (_VAR##_0_21) = _PERMUTE((_VAR##_0_21), ##__VA_ARGS__); \
  (_VAR##_1_21) = _PERMUTE((_VAR##_1_21), ##__VA_ARGS__); \
  (_VAR##_0_22) = _PERMUTE((_VAR##_0_22), ##__VA_ARGS__); \
  (_VAR##_1_22) = _PERMUTE((_VAR##_1_22), ##__VA_ARGS__); \
  (_VAR##_0_23) = _PERMUTE((_VAR##_0_23), ##__VA_ARGS__); \
  (_VAR##_1_23) = _PERMUTE((_VAR##_1_23), ##__VA_ARGS__); \
  (_VAR##_0_24) = _PERMUTE((_VAR##_0_24), ##__VA_ARGS__); \
  (_VAR##_1_24) = _PERMUTE((_VAR##_1_24), ##__VA_ARGS__); \
  (_VAR##_0_25) = _PERMUTE((_VAR##_0_25), ##__VA_ARGS__); \
  (_VAR##_1_25) = _PERMUTE((_VAR##_1_25), ##__VA_ARGS__); \
  (_VAR##_0_26) = _PERMUTE((_VAR##_0_26), ##__VA_ARGS__); \
  (_VAR##_1_26) = _PERMUTE((_VAR##_1_26), ##__VA_ARGS__); \
  (_VAR##_0_27) = _PERMUTE((_VAR##_0_27), ##__VA_ARGS__); \
  (_VAR##_1_27) = _PERMUTE((_VAR##_1_27), ##__VA_ARGS__); \
  (_VAR##_0_28) = _PERMUTE((_VAR##_0_28), ##__VA_ARGS__); \
  (_VAR##_1_28) = _PERMUTE((_VAR##_1_28), ##__VA_ARGS__); \
  (_VAR##_0_29) = _PERMUTE((_VAR##_0_29), ##__VA_ARGS__); \
  (_VAR##_1_29) = _PERMUTE((_VAR##_1_29), ##__VA_ARGS__); \
  (_VAR##_0_30) = _PERMUTE((_VAR##_0_30), ##__VA_ARGS__); \
  (_VAR##_1_30) = _PERMUTE((_VAR##_1_30), ##__VA_ARGS__); \
  (_VAR##_0_31) = _PERMUTE((_VAR##_0_31), ##__VA_ARGS__); \
  (_VAR##_1_31) = _PERMUTE((_VAR##_1_31), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_4X1(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_2X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_2_0) = _PERMUTE((_VAR##_2_0), ##__VA_ARGS__); \
  (_VAR##_3_0) = _PERMUTE((_VAR##_3_0), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_4X2(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_4X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_1) = _PERMUTE((_VAR##_0_1), ##__VA_ARGS__); \
  (_VAR##_1_1) = _PERMUTE((_VAR##_1_1), ##__VA_ARGS__); \
  (_VAR##_2_1) = _PERMUTE((_VAR##_2_1), ##__VA_ARGS__); \
  (_VAR##_3_1) = _PERMUTE((_VAR##_3_1), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_4X4(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_4X2(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_2) = _PERMUTE((_VAR##_0_2), ##__VA_ARGS__); \
  (_VAR##_1_2) = _PERMUTE((_VAR##_1_2), ##__VA_ARGS__); \
  (_VAR##_2_2) = _PERMUTE((_VAR##_2_2), ##__VA_ARGS__); \
  (_VAR##_3_2) = _PERMUTE((_VAR##_3_2), ##__VA_ARGS__); \
  (_VAR##_0_3) = _PERMUTE((_VAR##_0_3), ##__VA_ARGS__); \
  (_VAR##_1_3) = _PERMUTE((_VAR##_1_3), ##__VA_ARGS__); \
  (_VAR##_2_3) = _PERMUTE((_VAR##_2_3), ##__VA_ARGS__); \
  (_VAR##_3_3) = _PERMUTE((_VAR##_3_3), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_4X8(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_4X4(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_4) = _PERMUTE((_VAR##_0_4), ##__VA_ARGS__); \
  (_VAR##_1_4) = _PERMUTE((_VAR##_1_4), ##__VA_ARGS__); \
  (_VAR##_2_4) = _PERMUTE((_VAR##_2_4), ##__VA_ARGS__); \
  (_VAR##_3_4) = _PERMUTE((_VAR##_3_4), ##__VA_ARGS__); \
  (_VAR##_0_5) = _PERMUTE((_VAR##_0_5), ##__VA_ARGS__); \
  (_VAR##_1_5) = _PERMUTE((_VAR##_1_5), ##__VA_ARGS__); \
  (_VAR##_2_5) = _PERMUTE((_VAR##_2_5), ##__VA_ARGS__); \
  (_VAR##_3_5) = _PERMUTE((_VAR##_3_5), ##__VA_ARGS__); \
  (_VAR##_0_6) = _PERMUTE((_VAR##_0_6), ##__VA_ARGS__); \
  (_VAR##_1_6) = _PERMUTE((_VAR##_1_6), ##__VA_ARGS__); \
  (_VAR##_2_6) = _PERMUTE((_VAR##_2_6), ##__VA_ARGS__); \
  (_VAR##_3_6) = _PERMUTE((_VAR##_3_6), ##__VA_ARGS__); \
  (_VAR##_0_7) = _PERMUTE((_VAR##_0_7), ##__VA_ARGS__); \
  (_VAR##_1_7) = _PERMUTE((_VAR##_1_7), ##__VA_ARGS__); \
  (_VAR##_2_7) = _PERMUTE((_VAR##_2_7), ##__VA_ARGS__); \
  (_VAR##_3_7) = _PERMUTE((_VAR##_3_7), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_4X16(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_4X8(_VAR, _PERMUTE, ##__VA_ARGS__)   \
  (_VAR##_0_8) = _PERMUTE((_VAR##_0_8), ##__VA_ARGS__);   \
  (_VAR##_1_8) = _PERMUTE((_VAR##_1_8), ##__VA_ARGS__);   \
  (_VAR##_2_8) = _PERMUTE((_VAR##_2_8), ##__VA_ARGS__);   \
  (_VAR##_3_8) = _PERMUTE((_VAR##_3_8), ##__VA_ARGS__);   \
  (_VAR##_0_9) = _PERMUTE((_VAR##_0_9), ##__VA_ARGS__);   \
  (_VAR##_1_9) = _PERMUTE((_VAR##_1_9), ##__VA_ARGS__);   \
  (_VAR##_2_9) = _PERMUTE((_VAR##_2_9), ##__VA_ARGS__);   \
  (_VAR##_3_9) = _PERMUTE((_VAR##_3_9), ##__VA_ARGS__);   \
  (_VAR##_0_10) = _PERMUTE((_VAR##_0_10), ##__VA_ARGS__); \
  (_VAR##_1_10) = _PERMUTE((_VAR##_1_10), ##__VA_ARGS__); \
  (_VAR##_2_10) = _PERMUTE((_VAR##_2_10), ##__VA_ARGS__); \
  (_VAR##_3_10) = _PERMUTE((_VAR##_3_10), ##__VA_ARGS__); \
  (_VAR##_0_11) = _PERMUTE((_VAR##_0_11), ##__VA_ARGS__); \
  (_VAR##_1_11) = _PERMUTE((_VAR##_1_11), ##__VA_ARGS__); \
  (_VAR##_2_11) = _PERMUTE((_VAR##_2_11), ##__VA_ARGS__); \
  (_VAR##_3_11) = _PERMUTE((_VAR##_3_11), ##__VA_ARGS__); \
  (_VAR##_0_12) = _PERMUTE((_VAR##_0_12), ##__VA_ARGS__); \
  (_VAR##_1_12) = _PERMUTE((_VAR##_1_12), ##__VA_ARGS__); \
  (_VAR##_2_12) = _PERMUTE((_VAR##_2_12), ##__VA_ARGS__); \
  (_VAR##_3_12) = _PERMUTE((_VAR##_3_12), ##__VA_ARGS__); \
  (_VAR##_0_13) = _PERMUTE((_VAR##_0_13), ##__VA_ARGS__); \
  (_VAR##_1_13) = _PERMUTE((_VAR##_1_13), ##__VA_ARGS__); \
  (_VAR##_2_13) = _PERMUTE((_VAR##_2_13), ##__VA_ARGS__); \
  (_VAR##_3_13) = _PERMUTE((_VAR##_3_13), ##__VA_ARGS__); \
  (_VAR##_0_14) = _PERMUTE((_VAR##_0_14), ##__VA_ARGS__); \
  (_VAR##_1_14) = _PERMUTE((_VAR##_1_14), ##__VA_ARGS__); \
  (_VAR##_2_14) = _PERMUTE((_VAR##_2_14), ##__VA_ARGS__); \
  (_VAR##_3_14) = _PERMUTE((_VAR##_3_14), ##__VA_ARGS__); \
  (_VAR##_0_15) = _PERMUTE((_VAR##_0_15), ##__VA_ARGS__); \
  (_VAR##_1_15) = _PERMUTE((_VAR##_1_15), ##__VA_ARGS__); \
  (_VAR##_2_15) = _PERMUTE((_VAR##_2_15), ##__VA_ARGS__); \
  (_VAR##_3_15) = _PERMUTE((_VAR##_3_15), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_4X32(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_4X16(_VAR, _PERMUTE, ##__VA_ARGS__)  \
  (_VAR##_0_16) = _PERMUTE((_VAR##_0_16), ##__VA_ARGS__); \
  (_VAR##_1_16) = _PERMUTE((_VAR##_1_16), ##__VA_ARGS__); \
  (_VAR##_2_16) = _PERMUTE((_VAR##_2_16), ##__VA_ARGS__); \
  (_VAR##_3_16) = _PERMUTE((_VAR##_3_16), ##__VA_ARGS__); \
  (_VAR##_0_17) = _PERMUTE((_VAR##_0_17), ##__VA_ARGS__); \
  (_VAR##_1_17) = _PERMUTE((_VAR##_1_17), ##__VA_ARGS__); \
  (_VAR##_2_17) = _PERMUTE((_VAR##_2_17), ##__VA_ARGS__); \
  (_VAR##_3_17) = _PERMUTE((_VAR##_3_17), ##__VA_ARGS__); \
  (_VAR##_0_18) = _PERMUTE((_VAR##_0_18), ##__VA_ARGS__); \
  (_VAR##_1_18) = _PERMUTE((_VAR##_1_18), ##__VA_ARGS__); \
  (_VAR##_2_18) = _PERMUTE((_VAR##_2_18), ##__VA_ARGS__); \
  (_VAR##_3_18) = _PERMUTE((_VAR##_3_18), ##__VA_ARGS__); \
  (_VAR##_0_19) = _PERMUTE((_VAR##_0_19), ##__VA_ARGS__); \
  (_VAR##_1_19) = _PERMUTE((_VAR##_1_19), ##__VA_ARGS__); \
  (_VAR##_2_19) = _PERMUTE((_VAR##_2_19), ##__VA_ARGS__); \
  (_VAR##_3_19) = _PERMUTE((_VAR##_3_19), ##__VA_ARGS__); \
  (_VAR##_0_20) = _PERMUTE((_VAR##_0_20), ##__VA_ARGS__); \
  (_VAR##_1_20) = _PERMUTE((_VAR##_1_20), ##__VA_ARGS__); \
  (_VAR##_2_20) = _PERMUTE((_VAR##_2_20), ##__VA_ARGS__); \
  (_VAR##_3_20) = _PERMUTE((_VAR##_3_20), ##__VA_ARGS__); \
  (_VAR##_0_21) = _PERMUTE((_VAR##_0_21), ##__VA_ARGS__); \
  (_VAR##_1_21) = _PERMUTE((_VAR##_1_21), ##__VA_ARGS__); \
  (_VAR##_2_21) = _PERMUTE((_VAR##_2_21), ##__VA_ARGS__); \
  (_VAR##_3_21) = _PERMUTE((_VAR##_3_21), ##__VA_ARGS__); \
  (_VAR##_0_22) = _PERMUTE((_VAR##_0_22), ##__VA_ARGS__); \
  (_VAR##_1_22) = _PERMUTE((_VAR##_1_22), ##__VA_ARGS__); \
  (_VAR##_2_22) = _PERMUTE((_VAR##_2_22), ##__VA_ARGS__); \
  (_VAR##_3_22) = _PERMUTE((_VAR##_3_22), ##__VA_ARGS__); \
  (_VAR##_0_23) = _PERMUTE((_VAR##_0_23), ##__VA_ARGS__); \
  (_VAR##_1_23) = _PERMUTE((_VAR##_1_23), ##__VA_ARGS__); \
  (_VAR##_2_23) = _PERMUTE((_VAR##_2_23), ##__VA_ARGS__); \
  (_VAR##_3_23) = _PERMUTE((_VAR##_3_23), ##__VA_ARGS__); \
  (_VAR##_0_24) = _PERMUTE((_VAR##_0_24), ##__VA_ARGS__); \
  (_VAR##_1_24) = _PERMUTE((_VAR##_1_24), ##__VA_ARGS__); \
  (_VAR##_2_24) = _PERMUTE((_VAR##_2_24), ##__VA_ARGS__); \
  (_VAR##_3_24) = _PERMUTE((_VAR##_3_24), ##__VA_ARGS__); \
  (_VAR##_0_25) = _PERMUTE((_VAR##_0_25), ##__VA_ARGS__); \
  (_VAR##_1_25) = _PERMUTE((_VAR##_1_25), ##__VA_ARGS__); \
  (_VAR##_2_25) = _PERMUTE((_VAR##_2_25), ##__VA_ARGS__); \
  (_VAR##_3_25) = _PERMUTE((_VAR##_3_25), ##__VA_ARGS__); \
  (_VAR##_0_26) = _PERMUTE((_VAR##_0_26), ##__VA_ARGS__); \
  (_VAR##_1_26) = _PERMUTE((_VAR##_1_26), ##__VA_ARGS__); \
  (_VAR##_2_26) = _PERMUTE((_VAR##_2_26), ##__VA_ARGS__); \
  (_VAR##_3_26) = _PERMUTE((_VAR##_3_26), ##__VA_ARGS__); \
  (_VAR##_0_27) = _PERMUTE((_VAR##_0_27), ##__VA_ARGS__); \
  (_VAR##_1_27) = _PERMUTE((_VAR##_1_27), ##__VA_ARGS__); \
  (_VAR##_2_27) = _PERMUTE((_VAR##_2_27), ##__VA_ARGS__); \
  (_VAR##_3_27) = _PERMUTE((_VAR##_3_27), ##__VA_ARGS__); \
  (_VAR##_0_28) = _PERMUTE((_VAR##_0_28), ##__VA_ARGS__); \
  (_VAR##_1_28) = _PERMUTE((_VAR##_1_28), ##__VA_ARGS__); \
  (_VAR##_2_28) = _PERMUTE((_VAR##_2_28), ##__VA_ARGS__); \
  (_VAR##_3_28) = _PERMUTE((_VAR##_3_28), ##__VA_ARGS__); \
  (_VAR##_0_29) = _PERMUTE((_VAR##_0_29), ##__VA_ARGS__); \
  (_VAR##_1_29) = _PERMUTE((_VAR##_1_29), ##__VA_ARGS__); \
  (_VAR##_2_29) = _PERMUTE((_VAR##_2_29), ##__VA_ARGS__); \
  (_VAR##_3_29) = _PERMUTE((_VAR##_3_29), ##__VA_ARGS__); \
  (_VAR##_0_30) = _PERMUTE((_VAR##_0_30), ##__VA_ARGS__); \
  (_VAR##_1_30) = _PERMUTE((_VAR##_1_30), ##__VA_ARGS__); \
  (_VAR##_2_30) = _PERMUTE((_VAR##_2_30), ##__VA_ARGS__); \
  (_VAR##_3_30) = _PERMUTE((_VAR##_3_30), ##__VA_ARGS__); \
  (_VAR##_0_31) = _PERMUTE((_VAR##_0_31), ##__VA_ARGS__); \
  (_VAR##_1_31) = _PERMUTE((_VAR##_1_31), ##__VA_ARGS__); \
  (_VAR##_2_31) = _PERMUTE((_VAR##_2_31), ##__VA_ARGS__); \
  (_VAR##_3_31) = _PERMUTE((_VAR##_3_31), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_8X1(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_4X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_4_0) = _PERMUTE((_VAR##_4_0), ##__VA_ARGS__); \
  (_VAR##_5_0) = _PERMUTE((_VAR##_5_0), ##__VA_ARGS__); \
  (_VAR##_6_0) = _PERMUTE((_VAR##_6_0), ##__VA_ARGS__); \
  (_VAR##_7_0) = _PERMUTE((_VAR##_7_0), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_8X2(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_8X1(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_1) = _PERMUTE((_VAR##_0_1), ##__VA_ARGS__); \
  (_VAR##_1_1) = _PERMUTE((_VAR##_1_1), ##__VA_ARGS__); \
  (_VAR##_2_1) = _PERMUTE((_VAR##_2_1), ##__VA_ARGS__); \
  (_VAR##_3_1) = _PERMUTE((_VAR##_3_1), ##__VA_ARGS__); \
  (_VAR##_4_1) = _PERMUTE((_VAR##_4_1), ##__VA_ARGS__); \
  (_VAR##_5_1) = _PERMUTE((_VAR##_5_1), ##__VA_ARGS__); \
  (_VAR##_6_1) = _PERMUTE((_VAR##_6_1), ##__VA_ARGS__); \
  (_VAR##_7_1) = _PERMUTE((_VAR##_7_1), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_8X4(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_8X2(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_2) = _PERMUTE((_VAR##_0_2), ##__VA_ARGS__); \
  (_VAR##_1_2) = _PERMUTE((_VAR##_1_2), ##__VA_ARGS__); \
  (_VAR##_2_2) = _PERMUTE((_VAR##_2_2), ##__VA_ARGS__); \
  (_VAR##_3_2) = _PERMUTE((_VAR##_3_2), ##__VA_ARGS__); \
  (_VAR##_4_2) = _PERMUTE((_VAR##_4_2), ##__VA_ARGS__); \
  (_VAR##_5_2) = _PERMUTE((_VAR##_5_2), ##__VA_ARGS__); \
  (_VAR##_6_2) = _PERMUTE((_VAR##_6_2), ##__VA_ARGS__); \
  (_VAR##_7_2) = _PERMUTE((_VAR##_7_2), ##__VA_ARGS__); \
  (_VAR##_0_3) = _PERMUTE((_VAR##_0_3), ##__VA_ARGS__); \
  (_VAR##_1_3) = _PERMUTE((_VAR##_1_3), ##__VA_ARGS__); \
  (_VAR##_2_3) = _PERMUTE((_VAR##_2_3), ##__VA_ARGS__); \
  (_VAR##_3_3) = _PERMUTE((_VAR##_3_3), ##__VA_ARGS__); \
  (_VAR##_4_3) = _PERMUTE((_VAR##_4_3), ##__VA_ARGS__); \
  (_VAR##_5_3) = _PERMUTE((_VAR##_5_3), ##__VA_ARGS__); \
  (_VAR##_6_3) = _PERMUTE((_VAR##_6_3), ##__VA_ARGS__); \
  (_VAR##_7_3) = _PERMUTE((_VAR##_7_3), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_8X8(_VAR, _PERMUTE, ...)     \
  MATRIX_VAR_PERMUTE_8X4(_VAR, _PERMUTE, ##__VA_ARGS__) \
  (_VAR##_0_4) = _PERMUTE((_VAR##_0_4), ##__VA_ARGS__); \
  (_VAR##_1_4) = _PERMUTE((_VAR##_1_4), ##__VA_ARGS__); \
  (_VAR##_2_4) = _PERMUTE((_VAR##_2_4), ##__VA_ARGS__); \
  (_VAR##_3_4) = _PERMUTE((_VAR##_3_4), ##__VA_ARGS__); \
  (_VAR##_4_4) = _PERMUTE((_VAR##_4_4), ##__VA_ARGS__); \
  (_VAR##_5_4) = _PERMUTE((_VAR##_5_4), ##__VA_ARGS__); \
  (_VAR##_6_4) = _PERMUTE((_VAR##_6_4), ##__VA_ARGS__); \
  (_VAR##_7_4) = _PERMUTE((_VAR##_7_4), ##__VA_ARGS__); \
  (_VAR##_0_5) = _PERMUTE((_VAR##_0_5), ##__VA_ARGS__); \
  (_VAR##_1_5) = _PERMUTE((_VAR##_1_5), ##__VA_ARGS__); \
  (_VAR##_2_5) = _PERMUTE((_VAR##_2_5), ##__VA_ARGS__); \
  (_VAR##_3_5) = _PERMUTE((_VAR##_3_5), ##__VA_ARGS__); \
  (_VAR##_4_5) = _PERMUTE((_VAR##_4_5), ##__VA_ARGS__); \
  (_VAR##_5_5) = _PERMUTE((_VAR##_5_5), ##__VA_ARGS__); \
  (_VAR##_6_5) = _PERMUTE((_VAR##_6_5), ##__VA_ARGS__); \
  (_VAR##_7_5) = _PERMUTE((_VAR##_7_5), ##__VA_ARGS__); \
  (_VAR##_0_6) = _PERMUTE((_VAR##_0_6), ##__VA_ARGS__); \
  (_VAR##_1_6) = _PERMUTE((_VAR##_1_6), ##__VA_ARGS__); \
  (_VAR##_2_6) = _PERMUTE((_VAR##_2_6), ##__VA_ARGS__); \
  (_VAR##_3_6) = _PERMUTE((_VAR##_3_6), ##__VA_ARGS__); \
  (_VAR##_4_6) = _PERMUTE((_VAR##_4_6), ##__VA_ARGS__); \
  (_VAR##_5_6) = _PERMUTE((_VAR##_5_6), ##__VA_ARGS__); \
  (_VAR##_6_6) = _PERMUTE((_VAR##_6_6), ##__VA_ARGS__); \
  (_VAR##_7_6) = _PERMUTE((_VAR##_7_6), ##__VA_ARGS__); \
  (_VAR##_0_7) = _PERMUTE((_VAR##_0_7), ##__VA_ARGS__); \
  (_VAR##_1_7) = _PERMUTE((_VAR##_1_7), ##__VA_ARGS__); \
  (_VAR##_2_7) = _PERMUTE((_VAR##_2_7), ##__VA_ARGS__); \
  (_VAR##_3_7) = _PERMUTE((_VAR##_3_7), ##__VA_ARGS__); \
  (_VAR##_4_7) = _PERMUTE((_VAR##_4_7), ##__VA_ARGS__); \
  (_VAR##_5_7) = _PERMUTE((_VAR##_5_7), ##__VA_ARGS__); \
  (_VAR##_6_7) = _PERMUTE((_VAR##_6_7), ##__VA_ARGS__); \
  (_VAR##_7_7) = _PERMUTE((_VAR##_7_7), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_8X16(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_8X8(_VAR, _PERMUTE, ##__VA_ARGS__)   \
  (_VAR##_0_8) = _PERMUTE((_VAR##_0_8), ##__VA_ARGS__);   \
  (_VAR##_1_8) = _PERMUTE((_VAR##_1_8), ##__VA_ARGS__);   \
  (_VAR##_2_8) = _PERMUTE((_VAR##_2_8), ##__VA_ARGS__);   \
  (_VAR##_3_8) = _PERMUTE((_VAR##_3_8), ##__VA_ARGS__);   \
  (_VAR##_4_8) = _PERMUTE((_VAR##_4_8), ##__VA_ARGS__);   \
  (_VAR##_5_8) = _PERMUTE((_VAR##_5_8), ##__VA_ARGS__);   \
  (_VAR##_6_8) = _PERMUTE((_VAR##_6_8), ##__VA_ARGS__);   \
  (_VAR##_7_8) = _PERMUTE((_VAR##_7_8), ##__VA_ARGS__);   \
  (_VAR##_0_9) = _PERMUTE((_VAR##_0_9), ##__VA_ARGS__);   \
  (_VAR##_1_9) = _PERMUTE((_VAR##_1_9), ##__VA_ARGS__);   \
  (_VAR##_2_9) = _PERMUTE((_VAR##_2_9), ##__VA_ARGS__);   \
  (_VAR##_3_9) = _PERMUTE((_VAR##_3_9), ##__VA_ARGS__);   \
  (_VAR##_4_9) = _PERMUTE((_VAR##_4_9), ##__VA_ARGS__);   \
  (_VAR##_5_9) = _PERMUTE((_VAR##_5_9), ##__VA_ARGS__);   \
  (_VAR##_6_9) = _PERMUTE((_VAR##_6_9), ##__VA_ARGS__);   \
  (_VAR##_7_9) = _PERMUTE((_VAR##_7_9), ##__VA_ARGS__);   \
  (_VAR##_0_10) = _PERMUTE((_VAR##_0_10), ##__VA_ARGS__); \
  (_VAR##_1_10) = _PERMUTE((_VAR##_1_10), ##__VA_ARGS__); \
  (_VAR##_2_10) = _PERMUTE((_VAR##_2_10), ##__VA_ARGS__); \
  (_VAR##_3_10) = _PERMUTE((_VAR##_3_10), ##__VA_ARGS__); \
  (_VAR##_4_10) = _PERMUTE((_VAR##_4_10), ##__VA_ARGS__); \
  (_VAR##_5_10) = _PERMUTE((_VAR##_5_10), ##__VA_ARGS__); \
  (_VAR##_6_10) = _PERMUTE((_VAR##_6_10), ##__VA_ARGS__); \
  (_VAR##_7_10) = _PERMUTE((_VAR##_7_10), ##__VA_ARGS__); \
  (_VAR##_0_11) = _PERMUTE((_VAR##_0_11), ##__VA_ARGS__); \
  (_VAR##_1_11) = _PERMUTE((_VAR##_1_11), ##__VA_ARGS__); \
  (_VAR##_2_11) = _PERMUTE((_VAR##_2_11), ##__VA_ARGS__); \
  (_VAR##_3_11) = _PERMUTE((_VAR##_3_11), ##__VA_ARGS__); \
  (_VAR##_4_11) = _PERMUTE((_VAR##_4_11), ##__VA_ARGS__); \
  (_VAR##_5_11) = _PERMUTE((_VAR##_5_11), ##__VA_ARGS__); \
  (_VAR##_6_11) = _PERMUTE((_VAR##_6_11), ##__VA_ARGS__); \
  (_VAR##_7_11) = _PERMUTE((_VAR##_7_11), ##__VA_ARGS__); \
  (_VAR##_0_12) = _PERMUTE((_VAR##_0_12), ##__VA_ARGS__); \
  (_VAR##_1_12) = _PERMUTE((_VAR##_1_12), ##__VA_ARGS__); \
  (_VAR##_2_12) = _PERMUTE((_VAR##_2_12), ##__VA_ARGS__); \
  (_VAR##_3_12) = _PERMUTE((_VAR##_3_12), ##__VA_ARGS__); \
  (_VAR##_4_12) = _PERMUTE((_VAR##_4_12), ##__VA_ARGS__); \
  (_VAR##_5_12) = _PERMUTE((_VAR##_5_12), ##__VA_ARGS__); \
  (_VAR##_6_12) = _PERMUTE((_VAR##_6_12), ##__VA_ARGS__); \
  (_VAR##_7_12) = _PERMUTE((_VAR##_7_12), ##__VA_ARGS__); \
  (_VAR##_0_13) = _PERMUTE((_VAR##_0_13), ##__VA_ARGS__); \
  (_VAR##_1_13) = _PERMUTE((_VAR##_1_13), ##__VA_ARGS__); \
  (_VAR##_2_13) = _PERMUTE((_VAR##_2_13), ##__VA_ARGS__); \
  (_VAR##_3_13) = _PERMUTE((_VAR##_3_13), ##__VA_ARGS__); \
  (_VAR##_4_13) = _PERMUTE((_VAR##_4_13), ##__VA_ARGS__); \
  (_VAR##_5_13) = _PERMUTE((_VAR##_5_13), ##__VA_ARGS__); \
  (_VAR##_6_13) = _PERMUTE((_VAR##_6_13), ##__VA_ARGS__); \
  (_VAR##_7_13) = _PERMUTE((_VAR##_7_13), ##__VA_ARGS__); \
  (_VAR##_0_14) = _PERMUTE((_VAR##_0_14), ##__VA_ARGS__); \
  (_VAR##_1_14) = _PERMUTE((_VAR##_1_14), ##__VA_ARGS__); \
  (_VAR##_2_14) = _PERMUTE((_VAR##_2_14), ##__VA_ARGS__); \
  (_VAR##_3_14) = _PERMUTE((_VAR##_3_14), ##__VA_ARGS__); \
  (_VAR##_4_14) = _PERMUTE((_VAR##_4_14), ##__VA_ARGS__); \
  (_VAR##_5_14) = _PERMUTE((_VAR##_5_14), ##__VA_ARGS__); \
  (_VAR##_6_14) = _PERMUTE((_VAR##_6_14), ##__VA_ARGS__); \
  (_VAR##_7_14) = _PERMUTE((_VAR##_7_14), ##__VA_ARGS__); \
  (_VAR##_0_15) = _PERMUTE((_VAR##_0_15), ##__VA_ARGS__); \
  (_VAR##_1_15) = _PERMUTE((_VAR##_1_15), ##__VA_ARGS__); \
  (_VAR##_2_15) = _PERMUTE((_VAR##_2_15), ##__VA_ARGS__); \
  (_VAR##_3_15) = _PERMUTE((_VAR##_3_15), ##__VA_ARGS__); \
  (_VAR##_4_15) = _PERMUTE((_VAR##_4_15), ##__VA_ARGS__); \
  (_VAR##_5_15) = _PERMUTE((_VAR##_5_15), ##__VA_ARGS__); \
  (_VAR##_6_15) = _PERMUTE((_VAR##_6_15), ##__VA_ARGS__); \
  (_VAR##_7_15) = _PERMUTE((_VAR##_7_15), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_8X32(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_8X16(_VAR, _PERMUTE, ##__VA_ARGS__)  \
  (_VAR##_0_16) = _PERMUTE((_VAR##_0_16), ##__VA_ARGS__); \
  (_VAR##_1_16) = _PERMUTE((_VAR##_1_16), ##__VA_ARGS__); \
  (_VAR##_2_16) = _PERMUTE((_VAR##_2_16), ##__VA_ARGS__); \
  (_VAR##_3_16) = _PERMUTE((_VAR##_3_16), ##__VA_ARGS__); \
  (_VAR##_4_16) = _PERMUTE((_VAR##_4_16), ##__VA_ARGS__); \
  (_VAR##_5_16) = _PERMUTE((_VAR##_5_16), ##__VA_ARGS__); \
  (_VAR##_6_16) = _PERMUTE((_VAR##_6_16), ##__VA_ARGS__); \
  (_VAR##_7_16) = _PERMUTE((_VAR##_7_16), ##__VA_ARGS__); \
  (_VAR##_0_17) = _PERMUTE((_VAR##_0_17), ##__VA_ARGS__); \
  (_VAR##_1_17) = _PERMUTE((_VAR##_1_17), ##__VA_ARGS__); \
  (_VAR##_2_17) = _PERMUTE((_VAR##_2_17), ##__VA_ARGS__); \
  (_VAR##_3_17) = _PERMUTE((_VAR##_3_17), ##__VA_ARGS__); \
  (_VAR##_4_17) = _PERMUTE((_VAR##_4_17), ##__VA_ARGS__); \
  (_VAR##_5_17) = _PERMUTE((_VAR##_5_17), ##__VA_ARGS__); \
  (_VAR##_6_17) = _PERMUTE((_VAR##_6_17), ##__VA_ARGS__); \
  (_VAR##_7_17) = _PERMUTE((_VAR##_7_17), ##__VA_ARGS__); \
  (_VAR##_0_18) = _PERMUTE((_VAR##_0_18), ##__VA_ARGS__); \
  (_VAR##_1_18) = _PERMUTE((_VAR##_1_18), ##__VA_ARGS__); \
  (_VAR##_2_18) = _PERMUTE((_VAR##_2_18), ##__VA_ARGS__); \
  (_VAR##_3_18) = _PERMUTE((_VAR##_3_18), ##__VA_ARGS__); \
  (_VAR##_4_18) = _PERMUTE((_VAR##_4_18), ##__VA_ARGS__); \
  (_VAR##_5_18) = _PERMUTE((_VAR##_5_18), ##__VA_ARGS__); \
  (_VAR##_6_18) = _PERMUTE((_VAR##_6_18), ##__VA_ARGS__); \
  (_VAR##_7_18) = _PERMUTE((_VAR##_7_18), ##__VA_ARGS__); \
  (_VAR##_0_19) = _PERMUTE((_VAR##_0_19), ##__VA_ARGS__); \
  (_VAR##_1_19) = _PERMUTE((_VAR##_1_19), ##__VA_ARGS__); \
  (_VAR##_2_19) = _PERMUTE((_VAR##_2_19), ##__VA_ARGS__); \
  (_VAR##_3_19) = _PERMUTE((_VAR##_3_19), ##__VA_ARGS__); \
  (_VAR##_4_19) = _PERMUTE((_VAR##_4_19), ##__VA_ARGS__); \
  (_VAR##_5_19) = _PERMUTE((_VAR##_5_19), ##__VA_ARGS__); \
  (_VAR##_6_19) = _PERMUTE((_VAR##_6_19), ##__VA_ARGS__); \
  (_VAR##_7_19) = _PERMUTE((_VAR##_7_19), ##__VA_ARGS__); \
  (_VAR##_0_20) = _PERMUTE((_VAR##_0_20), ##__VA_ARGS__); \
  (_VAR##_1_20) = _PERMUTE((_VAR##_1_20), ##__VA_ARGS__); \
  (_VAR##_2_20) = _PERMUTE((_VAR##_2_20), ##__VA_ARGS__); \
  (_VAR##_3_20) = _PERMUTE((_VAR##_3_20), ##__VA_ARGS__); \
  (_VAR##_4_20) = _PERMUTE((_VAR##_4_20), ##__VA_ARGS__); \
  (_VAR##_5_20) = _PERMUTE((_VAR##_5_20), ##__VA_ARGS__); \
  (_VAR##_6_20) = _PERMUTE((_VAR##_6_20), ##__VA_ARGS__); \
  (_VAR##_7_20) = _PERMUTE((_VAR##_7_20), ##__VA_ARGS__); \
  (_VAR##_0_21) = _PERMUTE((_VAR##_0_21), ##__VA_ARGS__); \
  (_VAR##_1_21) = _PERMUTE((_VAR##_1_21), ##__VA_ARGS__); \
  (_VAR##_2_21) = _PERMUTE((_VAR##_2_21), ##__VA_ARGS__); \
  (_VAR##_3_21) = _PERMUTE((_VAR##_3_21), ##__VA_ARGS__); \
  (_VAR##_4_21) = _PERMUTE((_VAR##_4_21), ##__VA_ARGS__); \
  (_VAR##_5_21) = _PERMUTE((_VAR##_5_21), ##__VA_ARGS__); \
  (_VAR##_6_21) = _PERMUTE((_VAR##_6_21), ##__VA_ARGS__); \
  (_VAR##_7_21) = _PERMUTE((_VAR##_7_21), ##__VA_ARGS__); \
  (_VAR##_0_22) = _PERMUTE((_VAR##_0_22), ##__VA_ARGS__); \
  (_VAR##_1_22) = _PERMUTE((_VAR##_1_22), ##__VA_ARGS__); \
  (_VAR##_2_22) = _PERMUTE((_VAR##_2_22), ##__VA_ARGS__); \
  (_VAR##_3_22) = _PERMUTE((_VAR##_3_22), ##__VA_ARGS__); \
  (_VAR##_4_22) = _PERMUTE((_VAR##_4_22), ##__VA_ARGS__); \
  (_VAR##_5_22) = _PERMUTE((_VAR##_5_22), ##__VA_ARGS__); \
  (_VAR##_6_22) = _PERMUTE((_VAR##_6_22), ##__VA_ARGS__); \
  (_VAR##_7_22) = _PERMUTE((_VAR##_7_22), ##__VA_ARGS__); \
  (_VAR##_0_23) = _PERMUTE((_VAR##_0_23), ##__VA_ARGS__); \
  (_VAR##_1_23) = _PERMUTE((_VAR##_1_23), ##__VA_ARGS__); \
  (_VAR##_2_23) = _PERMUTE((_VAR##_2_23), ##__VA_ARGS__); \
  (_VAR##_3_23) = _PERMUTE((_VAR##_3_23), ##__VA_ARGS__); \
  (_VAR##_4_23) = _PERMUTE((_VAR##_4_23), ##__VA_ARGS__); \
  (_VAR##_5_23) = _PERMUTE((_VAR##_5_23), ##__VA_ARGS__); \
  (_VAR##_6_23) = _PERMUTE((_VAR##_6_23), ##__VA_ARGS__); \
  (_VAR##_7_23) = _PERMUTE((_VAR##_7_23), ##__VA_ARGS__); \
  (_VAR##_0_24) = _PERMUTE((_VAR##_0_24), ##__VA_ARGS__); \
  (_VAR##_1_24) = _PERMUTE((_VAR##_1_24), ##__VA_ARGS__); \
  (_VAR##_2_24) = _PERMUTE((_VAR##_2_24), ##__VA_ARGS__); \
  (_VAR##_3_24) = _PERMUTE((_VAR##_3_24), ##__VA_ARGS__); \
  (_VAR##_4_24) = _PERMUTE((_VAR##_4_24), ##__VA_ARGS__); \
  (_VAR##_5_24) = _PERMUTE((_VAR##_5_24), ##__VA_ARGS__); \
  (_VAR##_6_24) = _PERMUTE((_VAR##_6_24), ##__VA_ARGS__); \
  (_VAR##_7_24) = _PERMUTE((_VAR##_7_24), ##__VA_ARGS__); \
  (_VAR##_0_25) = _PERMUTE((_VAR##_0_25), ##__VA_ARGS__); \
  (_VAR##_1_25) = _PERMUTE((_VAR##_1_25), ##__VA_ARGS__); \
  (_VAR##_2_25) = _PERMUTE((_VAR##_2_25), ##__VA_ARGS__); \
  (_VAR##_3_25) = _PERMUTE((_VAR##_3_25), ##__VA_ARGS__); \
  (_VAR##_4_25) = _PERMUTE((_VAR##_4_25), ##__VA_ARGS__); \
  (_VAR##_5_25) = _PERMUTE((_VAR##_5_25), ##__VA_ARGS__); \
  (_VAR##_6_25) = _PERMUTE((_VAR##_6_25), ##__VA_ARGS__); \
  (_VAR##_7_25) = _PERMUTE((_VAR##_7_25), ##__VA_ARGS__); \
  (_VAR##_0_26) = _PERMUTE((_VAR##_0_26), ##__VA_ARGS__); \
  (_VAR##_1_26) = _PERMUTE((_VAR##_1_26), ##__VA_ARGS__); \
  (_VAR##_2_26) = _PERMUTE((_VAR##_2_26), ##__VA_ARGS__); \
  (_VAR##_3_26) = _PERMUTE((_VAR##_3_26), ##__VA_ARGS__); \
  (_VAR##_4_26) = _PERMUTE((_VAR##_4_26), ##__VA_ARGS__); \
  (_VAR##_5_26) = _PERMUTE((_VAR##_5_26), ##__VA_ARGS__); \
  (_VAR##_6_26) = _PERMUTE((_VAR##_6_26), ##__VA_ARGS__); \
  (_VAR##_7_26) = _PERMUTE((_VAR##_7_26), ##__VA_ARGS__); \
  (_VAR##_0_27) = _PERMUTE((_VAR##_0_27), ##__VA_ARGS__); \
  (_VAR##_1_27) = _PERMUTE((_VAR##_1_27), ##__VA_ARGS__); \
  (_VAR##_2_27) = _PERMUTE((_VAR##_2_27), ##__VA_ARGS__); \
  (_VAR##_3_27) = _PERMUTE((_VAR##_3_27), ##__VA_ARGS__); \
  (_VAR##_4_27) = _PERMUTE((_VAR##_4_27), ##__VA_ARGS__); \
  (_VAR##_5_27) = _PERMUTE((_VAR##_5_27), ##__VA_ARGS__); \
  (_VAR##_6_27) = _PERMUTE((_VAR##_6_27), ##__VA_ARGS__); \
  (_VAR##_7_27) = _PERMUTE((_VAR##_7_27), ##__VA_ARGS__); \
  (_VAR##_0_28) = _PERMUTE((_VAR##_0_28), ##__VA_ARGS__); \
  (_VAR##_1_28) = _PERMUTE((_VAR##_1_28), ##__VA_ARGS__); \
  (_VAR##_2_28) = _PERMUTE((_VAR##_2_28), ##__VA_ARGS__); \
  (_VAR##_3_28) = _PERMUTE((_VAR##_3_28), ##__VA_ARGS__); \
  (_VAR##_4_28) = _PERMUTE((_VAR##_4_28), ##__VA_ARGS__); \
  (_VAR##_5_28) = _PERMUTE((_VAR##_5_28), ##__VA_ARGS__); \
  (_VAR##_6_28) = _PERMUTE((_VAR##_6_28), ##__VA_ARGS__); \
  (_VAR##_7_28) = _PERMUTE((_VAR##_7_28), ##__VA_ARGS__); \
  (_VAR##_0_29) = _PERMUTE((_VAR##_0_29), ##__VA_ARGS__); \
  (_VAR##_1_29) = _PERMUTE((_VAR##_1_29), ##__VA_ARGS__); \
  (_VAR##_2_29) = _PERMUTE((_VAR##_2_29), ##__VA_ARGS__); \
  (_VAR##_3_29) = _PERMUTE((_VAR##_3_29), ##__VA_ARGS__); \
  (_VAR##_4_29) = _PERMUTE((_VAR##_4_29), ##__VA_ARGS__); \
  (_VAR##_5_29) = _PERMUTE((_VAR##_5_29), ##__VA_ARGS__); \
  (_VAR##_6_29) = _PERMUTE((_VAR##_6_29), ##__VA_ARGS__); \
  (_VAR##_7_29) = _PERMUTE((_VAR##_7_29), ##__VA_ARGS__); \
  (_VAR##_0_30) = _PERMUTE((_VAR##_0_30), ##__VA_ARGS__); \
  (_VAR##_1_30) = _PERMUTE((_VAR##_1_30), ##__VA_ARGS__); \
  (_VAR##_2_30) = _PERMUTE((_VAR##_2_30), ##__VA_ARGS__); \
  (_VAR##_3_30) = _PERMUTE((_VAR##_3_30), ##__VA_ARGS__); \
  (_VAR##_4_30) = _PERMUTE((_VAR##_4_30), ##__VA_ARGS__); \
  (_VAR##_5_30) = _PERMUTE((_VAR##_5_30), ##__VA_ARGS__); \
  (_VAR##_6_30) = _PERMUTE((_VAR##_6_30), ##__VA_ARGS__); \
  (_VAR##_7_30) = _PERMUTE((_VAR##_7_30), ##__VA_ARGS__); \
  (_VAR##_0_31) = _PERMUTE((_VAR##_0_31), ##__VA_ARGS__); \
  (_VAR##_1_31) = _PERMUTE((_VAR##_1_31), ##__VA_ARGS__); \
  (_VAR##_2_31) = _PERMUTE((_VAR##_2_31), ##__VA_ARGS__); \
  (_VAR##_3_31) = _PERMUTE((_VAR##_3_31), ##__VA_ARGS__); \
  (_VAR##_4_31) = _PERMUTE((_VAR##_4_31), ##__VA_ARGS__); \
  (_VAR##_5_31) = _PERMUTE((_VAR##_5_31), ##__VA_ARGS__); \
  (_VAR##_6_31) = _PERMUTE((_VAR##_6_31), ##__VA_ARGS__); \
  (_VAR##_7_31) = _PERMUTE((_VAR##_7_31), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_16X1(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_8X1(_VAR, _PERMUTE, ##__VA_ARGS__)   \
  (_VAR##_8_0) = _PERMUTE((_VAR##_8_0), ##__VA_ARGS__);   \
  (_VAR##_9_0) = _PERMUTE((_VAR##_9_0), ##__VA_ARGS__);   \
  (_VAR##_10_0) = _PERMUTE((_VAR##_10_0), ##__VA_ARGS__); \
  (_VAR##_11_0) = _PERMUTE((_VAR##_11_0), ##__VA_ARGS__); \
  (_VAR##_12_0) = _PERMUTE((_VAR##_12_0), ##__VA_ARGS__); \
  (_VAR##_13_0) = _PERMUTE((_VAR##_13_0), ##__VA_ARGS__); \
  (_VAR##_14_0) = _PERMUTE((_VAR##_14_0), ##__VA_ARGS__); \
  (_VAR##_15_0) = _PERMUTE((_VAR##_15_0), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_16X2(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_16X1(_VAR, _PERMUTE, ##__VA_ARGS__)  \
  (_VAR##_0_1) = _PERMUTE((_VAR##_0_1), ##__VA_ARGS__);   \
  (_VAR##_1_1) = _PERMUTE((_VAR##_1_1), ##__VA_ARGS__);   \
  (_VAR##_2_1) = _PERMUTE((_VAR##_2_1), ##__VA_ARGS__);   \
  (_VAR##_3_1) = _PERMUTE((_VAR##_3_1), ##__VA_ARGS__);   \
  (_VAR##_4_1) = _PERMUTE((_VAR##_4_1), ##__VA_ARGS__);   \
  (_VAR##_5_1) = _PERMUTE((_VAR##_5_1), ##__VA_ARGS__);   \
  (_VAR##_6_1) = _PERMUTE((_VAR##_6_1), ##__VA_ARGS__);   \
  (_VAR##_7_1) = _PERMUTE((_VAR##_7_1), ##__VA_ARGS__);   \
  (_VAR##_8_1) = _PERMUTE((_VAR##_8_1), ##__VA_ARGS__);   \
  (_VAR##_9_1) = _PERMUTE((_VAR##_9_1), ##__VA_ARGS__);   \
  (_VAR##_10_1) = _PERMUTE((_VAR##_10_1), ##__VA_ARGS__); \
  (_VAR##_11_1) = _PERMUTE((_VAR##_11_1), ##__VA_ARGS__); \
  (_VAR##_12_1) = _PERMUTE((_VAR##_12_1), ##__VA_ARGS__); \
  (_VAR##_13_1) = _PERMUTE((_VAR##_13_1), ##__VA_ARGS__); \
  (_VAR##_14_1) = _PERMUTE((_VAR##_14_1), ##__VA_ARGS__); \
  (_VAR##_15_1) = _PERMUTE((_VAR##_15_1), ##__VA_ARGS__);

#define MATRIX_VAR_PERMUTE_16X4(_VAR, _PERMUTE, ...)      \
  MATRIX_VAR_PERMUTE_16X2(_VAR, _PERMUTE, ##__VA_ARGS__)  \
  (_VAR##_0_2) = _PERMUTE((_VAR##_0_2), ##__VA_ARGS__);   \
  (_VAR##_1_2) = _PERMUTE((_VAR##_1_2), ##__VA_ARGS__);   \
  (_VAR##_2_2) = _PERMUTE((_VAR##_2_2), ##__VA_ARGS__);   \
  (_VAR##_3_2) = _PERMUTE((_VAR##_3_2), ##__VA_ARGS__);   \
  (_VAR##_4_2) = _PERMUTE((_VAR##_4_2), ##__VA_ARGS__);   \
  (_VAR##_5_2) = _PERMUTE((_VAR##_5_2), ##__VA_ARGS__);   \
  (_VAR##_6_2) = _PERMUTE((_VAR##_6_2), ##__VA_ARGS__);   \
  (_VAR##_7_2) = _PERMUTE((_VAR##_7_2), ##__VA_ARGS__);   \
  (_VAR##_8_2) = _PERMUTE((_VAR##_8_2), ##__VA_ARGS__);   \
  (_VAR##_9_2) = _PERMUTE((_VAR##_9_2), ##__VA_ARGS__);   \
  (_VAR##_10_2) = _PERMUTE((_VAR##_10_2), ##__VA_ARGS__); \
  (_VAR##_11_2) = _PERMUTE((_VAR##_11_2), ##__VA_ARGS__); \
  (_VAR##_12_2) = _PERMUTE((_VAR##_12_2), ##__VA_ARGS__); \
  (_VAR##_13_2) = _PERMUTE((_VAR##_13_2), ##__VA_ARGS__); \
  (_VAR##_14_2) = _PERMUTE((_VAR##_14_2), ##__VA_ARGS__); \
  (_VAR##_15_2) = _PERMUTE((_VAR##_15_2), ##__VA_ARGS__); \
  (_VAR##_0_3) = _PERMUTE((_VAR##_0_3), ##__VA_ARGS__);   \
  (_VAR##_1_3) = _PERMUTE((_VAR##_1_3), ##__VA_ARGS__);   \
  (_VAR##_2_3) = _PERMUTE((_VAR##_2_3), ##__VA_ARGS__);   \
  (_VAR##_3_3) = _PERMUTE((_VAR##_3_3), ##__VA_ARGS__);   \
  (_VAR##_4_3) = _PERMUTE((_VAR##_4_3), ##__VA_ARGS__);   \
  (_VAR##_5_3) = _PERMUTE((_VAR##_5_3), ##__VA_ARGS__);   \
  (_VAR##_6_3) = _PERMUTE((_VAR##_6_3), ##__VA_ARGS__);   \
  (_VAR##_7_3) = _PERMUTE((_VAR##_7_3), ##__VA_ARGS__);   \
  (_VAR##_8_3) = _PERMUTE((_VAR##_8_3), ##__VA_ARGS__);   \
  (_VAR##_9_3) = _PERMUTE((_VAR##_9_3), ##__VA_ARGS__);   \
  (_VAR##_10_3) = _PERMUTE((_VAR##_10_3), ##__VA_ARGS__); \
  (_VAR##_11_3) = _PERMUTE((_VAR##_11_3), ##__VA_ARGS__); \
  (_VAR##_12_3) = _PERMUTE((_VAR##_12_3), ##__VA_ARGS__); \
  (_VAR##_13_3) = _PERMUTE((_VAR##_13_3), ##__VA_ARGS__); \
  (_VAR##_14_3) = _PERMUTE((_VAR##_14_3), ##__VA_ARGS__); \
  (_VAR##_15_3) = _PERMUTE((_VAR##_15_3), ##__VA_ARGS__);

#define MATRIX_VAR_PROC_2X1(_K, _LHS, _RHS, _RES, _PROCESS) \
  _PROCESS((_LHS##_0), (_RHS), (_RES##_0_##_K))             \
  _PROCESS((_LHS##_1), (_RHS), (_RES##_1_##_K))

#define MATRIX_VAR_PROC_4X1(_K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_2X1(_K, _LHS, _RHS, _RES, _PROCESS)       \
  _PROCESS((_LHS##_2), (_RHS), (_RES##_2_##_K))             \
  _PROCESS((_LHS##_3), (_RHS), (_RES##_3_##_K))

#define MATRIX_VAR_PROC_8X1(_K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_4X1(_K, _LHS, _RHS, _RES, _PROCESS)       \
  _PROCESS((_LHS##_4), (_RHS), (_RES##_4_##_K))             \
  _PROCESS((_LHS##_5), (_RHS), (_RES##_5_##_K))             \
  _PROCESS((_LHS##_6), (_RHS), (_RES##_6_##_K))             \
  _PROCESS((_LHS##_7), (_RHS), (_RES##_7_##_K))

#define MATRIX_VAR_PROC_16X1(_K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_8X1(_K, _LHS, _RHS, _RES, _PROCESS)        \
  _PROCESS((_LHS##_8), (_RHS), (_RES##_8_##_K))              \
  _PROCESS((_LHS##_9), (_RHS), (_RES##_9_##_K))              \
  _PROCESS((_LHS##_10), (_RHS), (_RES##_10_##_K))            \
  _PROCESS((_LHS##_11), (_RHS), (_RES##_11_##_K))            \
  _PROCESS((_LHS##_12), (_RHS), (_RES##_12_##_K))            \
  _PROCESS((_LHS##_13), (_RHS), (_RES##_13_##_K))            \
  _PROCESS((_LHS##_14), (_RHS), (_RES##_14_##_K))            \
  _PROCESS((_LHS##_15), (_RHS), (_RES##_15_##_K))

#define MATRIX_VAR_PROC_1X2(_K, _LHS, _RHS, _RES, _PROCESS) \
  _PROCESS((_LHS), (_RHS##_0), (_RES##_##_K##_0))           \
  _PROCESS((_LHS), (_RHS##_1), (_RES##_##_K##_1))

#define MATRIX_VAR_PROC_1X4(_K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_1X2(_K, _LHS, _RHS, _RES, _PROCESS)       \
  _PROCESS((_LHS), (_RHS##_2), (_RES##_##_K##_2))           \
  _PROCESS((_LHS), (_RHS##_3), (_RES##_##_K##_3))

#define MATRIX_VAR_PROC_1X8(_K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_1X4(_K, _LHS, _RHS, _RES, _PROCESS)       \
  _PROCESS((_LHS), (_RHS##_4), (_RES##_##_K##_4))           \
  _PROCESS((_LHS), (_RHS##_5), (_RES##_##_K##_5))           \
  _PROCESS((_LHS), (_RHS##_6), (_RES##_##_K##_6))           \
  _PROCESS((_LHS), (_RHS##_7), (_RES##_##_K##_7))

#define MATRIX_VAR_PROC_1X16(_K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_1X8(_K, _LHS, _RHS, _RES, _PROCESS)        \
  _PROCESS((_LHS), (_RHS##_8), (_RES##_##_K##_8))            \
  _PROCESS((_LHS), (_RHS##_9), (_RES##_##_K##_9))            \
  _PROCESS((_LHS), (_RHS##_10), (_RES##_##_K##_10))          \
  _PROCESS((_LHS), (_RHS##_11), (_RES##_##_K##_11))          \
  _PROCESS((_LHS), (_RHS##_12), (_RES##_##_K##_12))          \
  _PROCESS((_LHS), (_RHS##_13), (_RES##_##_K##_13))          \
  _PROCESS((_LHS), (_RHS##_14), (_RES##_##_K##_14))          \
  _PROCESS((_LHS), (_RHS##_15), (_RES##_##_K##_15))

#define MATRIX_VAR_INIT(_M, _N, _VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_##_M##X##_N(_VAR_TYPE, _VAR_NAME, _VAR_INIT)

#define MATRIX_VAR_STORE(_M, _N, _STEP, _VAR, _ARRAY, _STORE, _NORM, ...) \
  MATRIX_VAR_STORE_##_M##X##_N(_STEP, _VAR, _ARRAY, _STORE, _NORM,        \
                               ##__VA_ARGS__)

#define MATRIX_VAR_PERMUTE(_M, _N, _VAR, _PERMUTE, ...) \
  MATRIX_VAR_PERMUTE_##_M##X##_N(_VAR, _PERMUTE, ##__VA_ARGS__)

#define MATRIX_VAR_PROC(_M, _N, _K, _LHS, _RHS, _RES, _PROCESS) \
  MATRIX_VAR_PROC_##_M##X##_N(_K, _LHS, _RHS, _RES, _PROCESS)
