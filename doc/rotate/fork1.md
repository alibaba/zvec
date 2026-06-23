### 1. src/include/zvec/core/interface/index_param.h
1. 接口上用Quantizer是不是更合适一些？考虑到enable_rotate也只对int8适用。
2. 这里的接口设计建议先讨论清楚 cc 
3. rotate更偏向于 预处理，那这里和量化的边界和关系是什么
```
提问：
1. 当前turbo中，Rotator 注入 Quantizer了中，是耦合实现；而index_param.h中PreprocessorParam preprocess_param; QuantizerParam quantizer_param;是并列的两层，turbo中是还没决定好Preprocessor实现方式是作为量化器的一部分还是作为一个独立模块吗
2. 把Preprocessor作为Quantizer的一部分，那么传递参数是，是把它作为一个QuantizerParam 的一部分进行指定吗？话说未来实现PQ，独立出rabitq之后，也是通过QuantizerParam统一传参吗
- 嗯，参数还是独立的，一个想法是，预处理可以独立构造，或者通过量化器构造，这样参数也是独立的，构造好，统一保存在量化器中，然后通过量化器的接口来进行统一调用。以后如果有独立预处理器，可以通过通用量化器来调用。
- 参数还是独立的，一个想法是，预处理可以独立构造，或者通过量化器构造，这样参数也是独立的，构造好，统一保存在量化器中，然后通过量化器的接口来进行统一调用。以后如果有独立预处理器，可以通过通用量化器来调用。
```
### 2. src/core/quantizer/record_rotator.cc
1. 看到依赖了rabitq的rotator，跨平台支持怎么样？

### 3. 长度不一致
#### src/core/quantizer/integer_quantizer_converter.cc
1. 长度不一致可能会有问题？
```cpp
memcpy((void *)normalize_buffer_.data(), vec,
pdim * sizeof(float));

normalize_buffer_(owner->front_->element_size(), 0),
```
-> `normalize_buffer_(owner->padded_dim() * sizeof(float), 0)`
#### src/core/quantizer/integer_quantizer_reformer.cc
1. rotator初始化时padding dim，buffer长度是dimension()，可能也会有不一致的问题
```cpp
impl_->rotator.reset(rabitqlib::choose_rotator(
dimension, Impl::to_rabitq(rotator_type), padded_dim));
```
-> `rotate_buffer.reset(new float[rotator_->padded_dim()]);`
### 4. 支持c api接口
新增以下C API函数（声明在 `src/include/zvec/c_api.h`，实现在 `src/binding/c/c_api.cc`）：
- `zvec_index_params_set_quantizer_enable_rotate(params, enable_rotate)` — 设置量化器的随机旋转预处理开关
- `zvec_index_params_get_quantizer_enable_rotate(params)` — 获取当前 enable_rotate 状态

单元测试：`tests/c/c_api_test.c` 中新增 `test_quantizer_enable_rotate()` 函数，覆盖 HNSW/FLAT/INVERT/NULL 等场景。

端到端示例（对标 `examples/python/hnsw`，位于 `examples/c_api/hnsw/`）：
- `int8_build.c` / `int8_query.c` — INT8 量化（无旋转）
- `int8_rotate_build.c` / `int8_rotate_query.c` — INT8 量化 + 随机旋转预处理
- `examples/c/index_example.c` 中也追加了 INT8 + enable_rotate 的简要示例

### 5. 补充一些c++层面的单测

#### DB 层 QuantizerParam 单测
**文件**: `tests/db/index/common/index_params_test.cc`

- `TEST(IndexParamsTest, QuantizerParamBasic)` — 默认构造、`QuantizerParam(true/false)`、`set_enable_rotate`、`operator==`/`!=`
- `TEST(IndexParamsTest, QuantizerParamWithVectorIndex)` — HNSW/Flat/IVF 的 `set_quantizer_param`、clone 保留 `quantizer_param`、不同 `enable_rotate` 的相等性比较

#### Proto 转换 enable_rotate 单测
**文件**: `tests/db/index/common/db_proto_converter_test.cc`

- `TEST(ConverterTest, HnswIndexParamsWithEnableRotate)` — `ToPb` → `FromPb` 往返一致性，覆盖 `enable_rotate=true` 和 `false`
- `TEST(ConverterTest, FlatIndexParamsWithEnableRotate)` — 同上
- `TEST(ConverterTest, IVFIndexParamsWithEnableRotate)` — 同上

#### Core 接口层 enable_rotate 序列化单测
**文件**: `tests/core/interface/index_interface_test.cc`

- `TEST(IndexInterface, QuantizerParamEnableRotateSerialization)` — Builder `.WithEnableRotate(true)` 构建 HNSW/Flat IndexParam，`SerializeToJson` 包含 `"enable_rotate":true`，`DeserializeIndexParamFromJson` 反序列化往返一致性，`omit_empty_value=true` 时 `enable_rotate=false` 被省略
