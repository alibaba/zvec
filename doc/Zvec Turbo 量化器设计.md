# 背景
<font style="color:rgb(0, 0, 0);">当前，距离度量沿用了 Proxima 的风格，采用模板方式进行定义。系统通过模块化设计实现了度量、转换器及指令集等组件的解耦，检索过程由Metric（距离度量）、Quantizer（量化器）、Algorithm（索引算法）等多个模块协作完成。然而，一些现有的组件存在职责边界模糊、实现方式耦合等问题，计划通过改为turbo的形式，统一切面通过输入多个维度信息组合以获得对应的函数集合，从而使得不同逻辑判断统一。</font>

# 设计现状
当前的系统架构设计存在着一些问题，下面将进行简要说明。

## Converter和Reformer功能耦合
当前的量化器由Converter和Reformer两个组件构成，其中，Converter主要负责离线构建，而Reformer主要负责在线查询，不过由于当前设计未区分流式插入与批量合并，Converter和Reformer在离线阶段存在一定的功能耦合：

![画板](https://intranetproxy.alipay.com/skylark/lark/0/2026/jpeg/226557004/1782716498906-f0957db9-24f8-466b-bfb2-351e65d1b5c5.jpeg)

这个实现最初的想法是将离线的一次性操作`build`、`train`等放到Converter中，而可以反复进行的操作`add`和`search`等放到Reformer，但是，这在本身也引入了一些额外的限制，例如，如果量化器需要训练，但索引是流式插入的，例如hnsw+<font style="color:rgb(0, 0, 0);">kUniformInt8</font>，就会存在问题。以下对当前的使用接口进行说明：

| **Converter接口** | **功能** | **Reformer接口** | **功能** |
| --- | --- | --- | --- |
| init() | 声明量化后的 meta 信息 | init() | 从 Converter 声明的 reformer_params 中加载运行时参数 |
| train() | 在全量数据上执行训练，收集统计信息 | load() | 从持久化存储中加载/卸载 Reformer 运行时状态 |
| transform() | 将 全量向量从原始格式转换为量化格式 | transform() | 将查询向量转换为与索引一致的量化格式 |
| dump() | 将 Converter 内部状态（如量化参数）持久化到 dumper | normalize() | 将量化空间计算的距离/分数映射回原始距离尺度 |
| meta() | 返回 Converter 修改后的 IndexMeta | revert() | 将量化向量反量化回 FP32 |


## <font style="color:rgb(0, 0, 0);">Quantizer</font>与<font style="color:rgb(0, 0, 0);">Metric</font>存在耦合
当前设计存在不同功能分布在同一文件/同一功能分布在不同文件的问题，例如：

| 注册文件 | 工厂注册名 | Metric | 备注 |
| --- | --- | --- | --- |
| **integer_quantizer_converter.cc** | Int8StreamingConverter | L2 / IP / MIPS | 基于单条数据的INT8 |
| | Int4StreamingConverter | L2 / IP / MIPS | 基于单条数据的INT4 |
| | Int8QuantizerConverter | - | 全局数据的INT8 |
| | Int4QuantizerConverter | - | 全局数据的INT4 |
| **cosine_converter.cc** | CosineInt8Converter | Cosine | 基于单条数据的INT8 |
| | CosineInt4Converter | Cosine | 基于单条数据的INT4 |
| | CosineFp16Converter | Cosine | FP16 |
| | ... | ... | ... |


由于这样的设计，一些量化器内部需要判断度量类型、数据类型、架构类型，以合理分配量化类型。

## 其他问题
1. Builder、Searcher、Streamer<font style="color:rgb(0, 0, 0);">职责边界模糊：</font>
    1. <font style="color:rgb(0, 0, 0);">Builder：全量数据的构建，负责操作 </font>`<font style="color:rgb(0, 0, 0);">build</font>`
    2. <font style="color:rgb(0, 0, 0);">Searcher：只被 IndexFlow（CLI 工具）和单元测试使用</font>
    3. Streamer：主路径组件负责读写，负责操作 `add` 和`search` 
2. 部分索引实现与量化器耦合：
    1. 与Quantizer完全解耦：HNSW / Flat / Vamana
    2. 硬编码与Quantizer耦合：IVF
    3. 当前自成一体：diskann/hnsw rabitq

# Turbo设计
Turbo 旨在通过统一接口收敛度量（Metric）与量化（Quantization）逻辑，提高代码内聚性，从而降低维护成本，提升代码的可读性与简洁性。

## 使用场景
![画板](https://intranetproxy.alipay.com/skylark/lark/0/2026/jpeg/95883/1782788752754-8c017988-c7ed-469d-8025-b1ab5ce94281.jpeg)

## 模块
Turbo系统包含以下几个模块：

+ 量化器
+ 预处理器
+ 距离计算

都以量化器为核心进行调度：

![画板](https://intranetproxy.alipay.com/skylark/lark/0/2026/jpeg/95883/1782802278889-9e04f816-ec15-42f6-9ff6-3e53ceb7c83e.jpeg)

## 设计目标
设计的目标主要包含以下六个维度的：

| **维度** | **说明** |
| :---: | :---: |
| 源数据类型 | 量化前的数据类型 |
| 目标数据类型 | 量化后的数据类型 |
| 距离类型 | 支持的距离度量方式 |
| 量化器类型 | 具体量化算法及参数管理方式 |
| 距离计算模式 | 单查询单点、单查询多点连续、单查询多点离散、点对点 |
| 指令架构 | 该量化器支持的指令架构 |
| 量化器使用场景 | 训练、流式插入、批量插入/合并、序列化、反序列化 |


## 调用场景
以下为构建、查询的调用场景：

### 构建
```plain
  源向量
    │
    ▼
┌───────────────────────┐
│  Quantizer            │
│  .quantize_datapoint  │
└───────────────────────┘
    │
    ▼
[量化向量 + Extra Meta(如有)] → 写入 Streamer
```

要求：

+ Uniform：无需训练即可编码（全局参数已确定）。
+ Record：逐向量独立计算 scale/bias，无需全局训练。
+ PQ/RabitQ：不支持纯流式插入（需先训练）。

### 查询
```plain
  源向量
    │
    ▼
┌─────────────────────┐
│  Quantizer          │
│  .quantize_query()  │
└─────────────────────┘
    │
    ▼
[量化向量 + Extra Meta(如有)] → 检索 Streamer
```

### 合并
```plain
多个源 Segment (FP32)
    │
    ▼
┌─────────────┐
│ write_holder│  ← 聚合数据到Holder
└─────────────┘
    │
    ▼
┌─────────────┐
│  Quantizer  │  ← 训练全局参数
│  .train     │
└─────────────┘
    │
    ▼
┌───────────────────────┐
│  Quantizer            │  ← 用训练后的Quantizer编码所有向量
│  .quantize_datapoint  |
└───────────────────────┘
    │
    ▼
[量化向量 + Extra Meta(如有)] → 写入 Streamer
```

# 支持范围
## 源数据类型 → 目标数据类型
考虑到 db 层有不保留 Flat FP32 的需求

+ 源数据类型需要支持 4 种通用类型：
    - FP32，P0
    - FP16，P0
    - INT8，P1
    - INT4，P2
+ 目标数据类型支持范围：

| **目标类型** | **说明** | **优先级** |
| :---: | :---: | :---: |
| FP32 | 原始数据不做量化，仅做度量转换（如 Cosine） | P0 |
| FP16 | 半精度浮点 | P0 |
| INT8 | 8-bit 整数量化 | P0 |
| INT4 | 4-bit 整数量化 | P0 |
| X-Bits | 自定义 bit 宽度的量化（如 5-bit、6-bit） | P1 |


## 量化器类型
量化器类型是核心分类维度，决定了端到端的数据布局、训练方式和距离计算方式。

| **量化器类型** | **特征** | **是否带Extra Meta** | **距离计算方式** | **优先级** |
| :---: | :---: | :---: | :---: | :---: |
| **Record** | 逐向量保存 scale/bias，按维度量化 | 是<br/>（Extra Meta 存 scale/bias） | 非对称距离，需反量化或查表 | P0 |
| **Uniform** | 全局统一scale & bias，按维度量化 | 否 | 对称距离，可 SIMD 优化 | P0 |
| **PQ**<br/>** (Product Quantization)** | 分段聚类码本量化 | 是<br/>（码本参数） | 非对称距离，ADC/SDC | P1 |
| **OPQ** | 优化的 PQ，含正交变换 | 是<br/>（变换矩阵 + 码本） | 非对称距离，ADC | P2 |
| **Raw** | 不做数值量化，仅改变类型（FP32→FP16）或做度量预处理 | 否 | 原始距离计算 | P0 |
| ~~**RabitQ**~~ | ~~旋转 + 二进制量化~~ | ~~是（旋转矩阵等）~~ | ~~汉明距离近似~~ | ~~P1~~ |


_**注意**__：量化器类型与目标数据类型是正交但有关联的概念。例如 Record 和 Uniform 都可以输出到 INT8 或 INT4；Raw 可以输出到 FP32、FP16、BF16。_

## 距离类型支持
| 距离类型 | 支持说明 | 优先级 |
| --- | --- | --- |
| **L2 (Squared Euclidean)** | 所有量化方案的基础支持 | P0 |
| **Cosine** | 所有量化方案支持，可通过预处理归一化转化为 IP 或 L2 | P0 |
| **Inner Product (IP)** | 所有量化方案支持 | P0 |
| **MIPS** | 基于 IP 的变形，需做转换 | P1 |


# 量化器
## 接口语义
每个具体的量化器实现以下统一接口，但内部行为因方案而异：

```cpp
class Quantizer {
 public:
  typedef std::shared_ptr<Quantizer> Pointer;

  Quantizer() {}
  virtual ~Quantizer() {}

  //! Initialize quantizer with index metadata and parameters
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Get the output metadata after initialization
  virtual const IndexMeta &meta() const = 0;

  //! Input data type accepted by the quantizer
  virtual DataType input_data_type() const = 0;

  //! Data type
  virtual QuantizeType type() const {
    return type_;
  }

  //! Dimensionality of the input vectors
  virtual int dim() const = 0;

  //! Train the quantizer with a contiguous batch of data
  virtual int train(const void * /*data*/, size_t /*num*/, size_t /*stride*/) {
    return IndexError_NotImplemented;
  }

  //! Whether the quantizer requires training before use
  virtual bool require_train() const = 0;

  //! Train the quantizer with data from an IndexHolder
  virtual int train(IndexHolder::Pointer /*holder*/) {
    return IndexError_NotImplemented;
  }

  //! Byte length of a quantized datapoint vector
  virtual size_t quantized_datapoint_vector_length() const = 0;

  //! Byte length of a quantized query vector
  virtual size_t quantized_query_vector_length() const = 0;

  //! Quantize a datapoint vector
  virtual void quantize_data(const void *input, void *output) const = 0;

  //! Quantize a query vector
  virtual void quantize_query(const void *input, void *output) const = 0;

  //! Distance between a quantized datapoint and a quantized query
  virtual float calc_distance_dp_query(const void *dp,
                                       const void *query) const = 0;

  //! Batched distance between quantized datapoints and a quantized query
  virtual void calc_distance_dp_query_batch(const void *const *dp_list,
                                            int dp_num, const void *query,
                                            float *dist_list) const = 0;

  //! Distance between a quantized datapoint and an unquantized query
  virtual float calc_distance_dp_query_unquantized(const void *dp,
                                                   const void *query) const = 0;

  //! Batched distance between quantized datapoints and an unquantized query
  virtual void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const = 0;

  //! Distance between two quantized datapoints
  virtual float calc_distance_dp_dp(const void *dp1, const void *dp2) const = 0;

  //! Quantize a query vector for search
  virtual int quantize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                       std::string * /*out*/,
                       IndexQueryMeta * /*ometa*/) const {
    return IndexError_NotImplemented;
  }

  //! Dequantize a result vector back to original format
  virtual int dequantize(const void * /*in*/, const IndexQueryMeta & /*qmeta*/,
                         std::string * /*out*/) const {
    return IndexError_NotImplemented;
  }

  virtual DistanceImpl distance(const void * /*query*/,
                                const IndexQueryMeta & /*qmeta*/) const {
    return DistanceImpl{};
  }

  //! Serialize quantizer parameters
  virtual int serialize(std::string * /*out*/) const {
    return IndexError_NotImplemented;
  }

  //! Deserialize quantizer parameters
  virtual int deserialize(std::string & /*in*/) {
    return IndexError_NotImplemented;
  }

  //! Deserialize quantizer parameters from a raw, possibly mmap-backed buffer
  //! (zero-copy entry point for large payloads such as codebooks/matrices).
  virtual int deserialize(const void * /*data*/, size_t /*len*/) {
    return IndexError_NotImplemented;
  }

 protected:
  //! Map a metric name (e.g. "SquaredEuclidean", "Cosine",
  //! "InnerProduct", "MipsSquaredEuclidean") to its MetricType.
  static MetricType metric_from_name(const std::string &name) {
    if (name == "SquaredEuclidean") {
      return MetricType::kSquaredEuclidean;
    }
    if (name == "Cosine") {
      return MetricType::kCosine;
    }
    if (name == "InnerProduct") {
      return MetricType::kInnerProduct;
    }
    if (name == "MipsSquaredEuclidean") {
      return MetricType::kMipsSquaredEuclidean;
    }
    return MetricType::kUnknown;
  }

  QuantizeType type_{QuantizeType::kDefault};
  uint32_t extra_meta_size_{0};
};
```

dp * dp batch for IVF -> P1

## 量化方案
#### Raw 方案（类型转换/度量预处理）
+ **Extra Meta**：对于Cosine向量增加归一化值。
+ **编码流程**：
    - FP32 → FP16：数值截断/舍入。
    - FP32 → FP32 + Cosine 预处理：向量归一化。
+ **距离计算**：使用对应精度的原始距离计算（如 FP16 L2、Cosine 转 IP）。
+ **使用场景**：无需训练，实时转换。不需要训练，支持流式

#### Record 方案（逐向量 scale/bias）
+ **Extra Meta**：每个向量后附加 scale、bias相关的数组（或其他压缩形式）。
+ **编码流程**：输入 FP32 → 计算向量的scale/bias参数 → 量化到 INT8/INT4 → 输出 [codes + scale + bias]。
+ **距离计算**：非对称，查询向量需经过相同 scale/bias 反量化或预处理到同一域。
+ **使用场景**：流式插入友好（每条向量独立计算 meta），但 Extra Meta 增加存储。

#### Uniform 方案（全局 scale/bias）
+ **Extra Meta**：无 per-vector meta，全局参数存于 Quantizer 内部（通过 serialize/deserialize 持久化）。
+ **编码流程**：输入 FP32 → 使用全局 scale/bias → 量化到 INT8/INT4 → 纯码本输出。
+ **距离计算**：对称距离，查询向量和库向量使用同一组全局参数，可高度 SIMD 优化。
+ **使用场景**：批量训练确定全局参数后，流式插入和检索效率最高。

#### PQ / OPQ 方案
+ **Extra Meta**：无 per-vector meta，全局码本参数（centroids）存于 Quantizer 内部（通过 serialize/deserialize 持久化）。
+ **预处理：**OPQ有随机旋转预处理
+ **编码流程**：输入 FP32 → （OPQ 先做正交变换）→ 分段聚类 → 输出码本索引。
+ **距离计算**：ADC（Asymmetric Distance Computation）~~或 SDC~~。
+ **使用场景**：高维向量压缩，适合批量构建，流式插入需支持码本增量更新（P2）。
+ **实现计划**：[PQ in Zvec Turbo](https://yuque.alibaba-inc.com/proxima/aby36h/dt2x3o15c1mbqgxz)

#### TurboQuant方案
+ **Extra Meta：**无per-vector meta，
+ **预处理：**需要加随机旋转
+ **距离计算：**SIMD， INT计算
+ **使用场景：**极致压缩，保持一定召回。不需要训练，支持流式

#### ~~RabitQ 方案~~
+ **Extra Meta**：旋转矩阵等参数。
+ **编码流程**：输入 FP32 → 旋转 → 二进制量化。
+ **距离计算**：基于汉明距离的近似计算。
+ **使用场景**：极高压缩比，精度要求可容忍的场景。



## 训练
量化器训练过程会在合并阶段被调用。量化器根据各自实现，来确定是否需要训练。量化器的训练过程，通过传入一组连续内存的数据，并用步长来指定采样方式，进行训练。

# 距离计算器
为了对距离方式进行封装，设计了距离计算器DistanceImpl类：

![画板](https://intranetproxy.alipay.com/skylark/lark/0/2026/jpeg/95883/1782875740300-e562d52a-5bfc-485d-b6cf-61e523e3edaf.jpeg)

# 距离表
基于上述分类，Turbo 的函数分发不再通过分散的模板特化，而是通过**量化方案 + 目标数据类型 + 距离类型 + SIMD 架构**四个维度组合查询函数表：

类型定义：

+ 度量类型MetricType定义：
    -  kSquaredEuclidean
    -  kCosine
    -  kInnerProduct
    -  kMipsSquaredEuclidean
    -  kUnknown
+ 数据类型DataType定义
    -  kInt4
    -  kInt8
    -  kFp16
    -  kFp32
    -  kUnknown
+ 量化类型QuantizeType定义
    -  kDefault
    -  kUniform
    -  kRecord
    -  kFp16
    -  kFp32
    -  kPQ
    -  kRabit
+ 架构类型CpuArchType定义：
    - kAuto
    - kScalar
    - kSSE
    - kAVX
    - kAVX2
    - kAVX512
    - kAVX512VNNI
    - kAVX512FP16

相应的，获取函数的接口也需要枚举对应的参数类型：

+ DistanceFunc **get_distance_func**(MetricType metric_type, DataType data_type, QuantizeType quantize_type, CpuArchType cpu_arch_type = CpuArchType::kAuto);
+ BatchDistanceFunc **get_batch_distance_func**(MetricType metric_type, DataType data_type, QuantizeType quantize_type, CpuArchType cpu_arch_type = CpuArchType::kAuto);
+ QueryPreprocessFunc **get_query_preprocess_func**(MetricType metric_type, DataType data_type, QuantizeType quantize_type, CpuArchType cpu_arch_type = CpuArchType::kAuto);

**说明**：

+ 不存在的组合在编译期报错（static_assert），避免运行时意外。
+ 运行时通过 Quantizer 返回的 DistanceCalculator 绑定已确定的 scheme/dst_type/metric，仅需March进行动态分发。

# 原值构建支持
由于使用量化值进行图索引构建，会出现不够精确的情况，进一步支持使用原值而非量化值来进行。streamer 和 builder 在实现时严格区分 rdp * rdp （fp32 raw quantizer）; query*dp ，并提供设置原值索引 provider 的接口（之后可考虑和 external_storage 需求中设计的VectorSource 合并）

术语：query / dp / r(eference)dp

# 预处理器
区分概念：预处理操作和预处理器操作，预处理操作包括：旋转、normalize，数据类型偏移等，但并不是所有的预处理操作都需要预处理器来完成，只有那些有状态依赖的操作（例如：旋转）需要使用旋转完成，而normalize，数据类型偏移等操作使用函数指针来完成更加便捷。

预处理器是一种量化器的内部可选组件，主要负责在进行数据检索前对数据进行预处理，主要包括旋转功能，可以对量化前的数据进行预处理，以帮助量化器发挥更好的效果。其中，在需要进行预处理时，量化器**主动**调用预处理器，在进行量化前对向量数据进行处理；需要保存时，预处理器需要的数据同量化器中的参数一起进行保存。该组件需要实现的功能包括：

+ 随机旋转（optional）+ Record INT8/INT4
+ OPQ旋转（optional）+ PQ
+ 通过某种方式，对数据进行降维（optional）

职责边界：哪些功能属于预处理范畴，与索引/量化的边界在何处？预处理器是一种**可选**的量化方法的优化组件，缺少也不影响量化方法的正常运行，由量化方案创建的组件属于量化器。

预处理器中，处理数据同样是一个费时的操作，尤其是在在线处理的情况下，需要使用**距离表**中的方案进行加速，在获取相应的预处理函数的接口时，也需要枚举对应的参数类型，比如：

+ FwhtFunc get_fwht_func(CpuArchType cpu_arch_type = CpuArchType::kAuto);
+ MatVecFunc get_matvec_func(CpuArchType cpu_arch_type = CpuArchType::kAuto);

通过分发的方式可以快速获得对应的预处理函数。具体的实现如下：

![画板](https://intranetproxy.alipay.com/skylark/lark/0/2026/jpeg/226557004/1782812444145-e4b10b66-8bd1-4f6d-aa94-c205ca57235d.jpeg)

由于当前FHT可以适用于任意维度的向量，不必再将FHT和Matrix作为rotator下的子类，去除rotator，将FHT和Matrix作为两个独立的类。对于INT8/INT4使用FHT，而对于OPQ使用Matrix，未来还可能有降维等功能。

# RabitQ量化接入挑战
针对当前HnswRabitq索引接入RabitQ量化器，主要有两个方面的挑战：

1. 距离计算是渐进进行，先计算粗略距离，再进行精确距离计算，这个涉及到在索引遍历时的判断；
2. 返回结果时距离组合，对于流程中的距离返回需要进行改造，涉及到TopK队列的结构体定义；

结论：暂时在第一期不做接入。

# IndexMeta调整
为支持上述设计，IndexMeta 需增加以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| quantize_type | QuantizeType | 量化方案（Record/Uniform/PQ...） |
| extra_meta_size | uint32_t | 每条向量的 Extra Meta 大小（0 表示无） |


兼容性处理：

+ 存量数据无 quantize_type 字段时，按旧逻辑推断（如根据数据类型和是否有 Extra Meta 推断为 Record 或 Raw）。
+ 新增字段以扩展字段方式存入 IndexMeta，保证向后兼容。

# 代码说明
## 距离计算器定义
```cpp
class DistanceImpl {
 public:
  DistanceImpl() = default;

  DistanceImpl(DistanceFunc func, std::string quantized_query, size_t dim)
      : func_(std::move(func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  DistanceImpl(DistanceFunc func, BatchDistanceFunc batch_func,
               std::string quantized_query, size_t dim)
      : func_(std::move(func)),
        batch_func_(std::move(batch_func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  //! Whether the handle is ready to compute distances.
  bool valid() const {
    return static_cast<bool>(func_);
  }

  //! Whether a batch distance function is available.
  bool batch_valid() const {
    return static_cast<bool>(batch_func_);
  }

  //! Compute the distance between the stored query and `candidate`.
  float operator()(const void *candidate) const {
    float d = 0.0f;
    func_(candidate, query_storage_.data(), dim_, &d);
    return d;
  }

  //! Compute distances for a batch of `num` candidates against the
  //! stored query. Falls back to the scalar path when no batch function
  //! is bound.
  void batch(const void **candidates, size_t num, float *out) const {
    if (batch_func_) {
      batch_func_(candidates, query_storage_.data(), num, dim_, out);
      return;
    }
    for (size_t i = 0; i < num; ++i) {
      out[i] = 0.0f;
      func_(candidates[i], query_storage_.data(), dim_, out + i);
    }
  }

  //! Access the quantized query bytes (for pairwise helpers).
  const std::string &query_storage() const {
    return query_storage_;
  }

  size_t dim() const {
    return dim_;
  }

  //! Raw scalar distance function (operates on already-quantized
  //! candidates). Useful for pairwise node-vs-node distance where no
  //! stored query is involved.
  const DistanceFunc &func() const {
    return func_;
  }

  //! Raw batch distance function.
  const BatchDistanceFunc &batch_func() const {
    return batch_func_;
  }

 private:
  DistanceFunc func_{};
  BatchDistanceFunc batch_func_{};
  std::string query_storage_{};
  size_t dim_{0};
};
```

## 索引对接示例
在索引中，通过接入量化器来实现距离对象的接入：

```cpp
class HnswDistCalculator {
 ... ..
 private:
  zvec::turbo::DistanceImpl dist_impl_{};
};
```

初始化：

```cpp
inline void reset_query(const void *query) {
    if (quantizer_) {
      dist_impl_ = quantizer_->distance(query, qmeta_);
    }
    ... ...
}
```

  	距离计算调用：

```cpp
const auto &func = dist_impl_.func();
if (func) {
  func(vec_lhs, vec_rhs, dist_impl_.dim(), &score);
  return score;
}
```

## 量化器实现示例
对于有状态的距离计算，可以在量化器中，通过lambda来捕获对应的状态数据。

```cpp

DistanceImpl MyQuantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
  // Create a lambda-based DistanceFunc that captures this quantizer
  auto dim = padded_dim_;
  auto metric = metric_type_;
  auto cb_val = cb();

  DistanceFunc func = [dim, metric, cb_val](const void *m, const void *q,
                                            size_t /*dim_arg*/, float *out) {
    const auto *dp_bytes = reinterpret_cast<const char *>(m);
    const auto *q_floats = reinterpret_cast<const float *>(q);

    const auto *code = reinterpret_cast<const uint8_t *>(dp_bytes);
    const float *factors = reinterpret_cast<const float *>(dp_bytes + dim);
    float f_add = factors[0];
    float f_rescale = factors[1];

    const float *rotated_q = q_floats;
    float sum_q = q_floats[dim];
    float norm_sq_q = q_floats[dim + 1];

    float ip = ip_float_code(rotated_q, code, dim);
    float est_dist = f_add + f_rescale * (ip + cb_val * sum_q);
    if (metric == MetricType::kSquaredEuclidean) {
      est_dist += norm_sq_q;
    }
    *out = est_dist;
  };

  return DistanceImpl(std::move(func), std::move(qbuf), padded_dim_);
}
```

## 预处理示例
增加预处理来支持向量的前置处理过程，其中旋转也是预处理的一部分。定义如下：

```cpp
class Preprocessor {
 public:
  using Pointer = std::shared_ptr<Preprocessor>;

  virtual ~Preprocessor() = default;

  //! Input dimensionality accepted by apply().
  virtual int in_dim() const = 0;

  //! Output dimensionality produced by apply(). May differ from in_dim()
  //! (e.g. FHT pads to the next power of two).
  virtual int out_dim() const = 0;

  //! Forward transform: map an input vector to the preprocessed space.
  //! \p out must hold at least out_dim() elements.
  virtual void apply(const float *in, float *out) const = 0;

  //! Inverse transform: recover the original-space vector from a preprocessed
  //! one. \p out must hold at least in_dim() elements.
  virtual void apply_inverse(const float *in, float *out) const = 0;

  //! Fit the preprocessor from a contiguous batch of training data.
  //! \p data  pointer to the first element of the batch.
  //! \p num   number of vectors in the batch.
  //! \p stride byte offset between consecutive vectors (0 ⇒ packed).
  virtual void train(const void *data, size_t num, size_t stride) = 0;

  //! Serialize the preprocessor into a self-contained blob (header + payload).
  virtual int serialize(std::string *out) const = 0;

  //! Deserialize the preprocessor from a raw, possibly mmap-backed buffer.
  virtual int deserialize(const void *data, size_t len) = 0;
};
```

对于旋转定义：

```cpp
class Rotator : public Preprocessor {
 public:
  using Pointer = std::shared_ptr<Rotator>;

  //! Kind of rotator.
  virtual RotatorType type() const = 0;

  // in_dim(), out_dim(), apply(), apply_inverse(), train(), serialize(),
  // and deserialize() are inherited from Preprocessor.
};

//! Create an untrained rotator of the given type for in_dim dimensions.
Rotator::Pointer CreateRotator(RotatorType type, int in_dim);

//! Create and restore a rotator from a serialized blob (reads the type from
//! the embedded RotatorSerHeader). Returns nullptr on malformed input.
Rotator::Pointer CreateRotatorFromBlob(const void *data, size_t len);

}  // namespace turbo
```

预处理器作为量化器的成员，进行接入。

# 实现优先级
## 一期（P0）
+ **量化方案**：
    - Raw（FP32→FP16, FP32→FP32 Cosine）
    - Uniform（FP32→INT8, FP32→INT4）
    - Record（FP32→INT8, FP32→INT4）
+ **距离类型**：L2, Cosine, IP
+ **微架构类型：**SSE，AVX，AVX512，AVX512FP16
+ **距离计算模式**：所有单点查询、部分批量、
+ **索引支持：**HNSW，IVF，Vamana，DiskAnn
+ **使用场景**：流式插入、Merge 训练、流式检索、序列化/反序列化

## 二期（P1）
+ **量化方案**：
    - PQ（FP32→PQ）
+ **距离类型**：L2, Cosine, IP
+ **微架构类型：**SSE，AVX，AVX512，AVX512FP16
+ **距离计算模式**：所有批量查询

# 模块分工
在基础PR的基础上，其它模块可以<font style="background-color:#FBDE28;">分PR</font>进行提交、验证和合并。

| **模块** | **内容** | **人员** | **开始日期** | **结束日期** | **人日** | **优先级** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **量化器** | 基础实现 | 弃疾 | 6月22日 | 6月25日 | 3 | P0 |
| | 预处理 | 莊霖 |  |  | 5 | P0 |
| | turbo分发 | 弃疾 | 6月26日 | 6月29日 | 2 | P0 |
| | FP32量化器 | 荐宁 |  |  | 3 | P0 |
| | FP16量化器 | 荐宁 |  |  | 3 | P0 |
| | Int8量化器 | 荐宁 |  |  | 3 | P0 |
| | Int4量化器 | 荐宁 |  |  | 3 | P0 |
| | PQ量化器 | 莊霖 |  |  | 10 | P1 |
| **索引对接** | HNSW | 荐宁 |  |  | 2 | P0 |
| | IVF | 荐宁 |  |  | 2 | P0 |
| | Vamana | 荐宁 |  |  | 2 | P0 |
| | DiskAnn | 弃疾 | 7月6日 | 7月8日 | 2 | P1 |
| **单元测试** | 单元测试 | ALL |  |  | 5 | P0 |
| **集成测试** | 集成测试 | ALL |  |  | 5 | P0 |


# 结语
+ 本设计以**量化方案**为核心分类维度，将原本分散在量化器、度量、数据类型中的判断逻辑统一到 Turbo 量化器中。
+ 确保一期聚焦高频使用场景的 Uniform/Record + INT8/INT4/FP16 + L2/Cosine/IP 组合，再逐步扩展 PQ量化器等复杂方案。



# 参考
+ [距离计算能力](https://yuque.alibaba-inc.com/proxima/aby36h/lr69q2egk5ob18cw)
+ [CI&CD 支持集合](https://yuque.alibaba-inc.com/proxima/aby36h/pyvb301f3fs8s3ua)
+ [TurboQuant-近似最优失真率的在线量化方案](https://yuque.alibaba-inc.com/proxima/aby36h/lgllif6v3hht61be)
