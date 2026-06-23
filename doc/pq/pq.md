# PQ（Product Quantization）实现方式总结

本文档总结当前 zvec 系统（DiskANN 中的 PQ）与 proxima2 系统中 PQ 的实现方式，为后续在 zvec 中添加独立 PQ 量化提供参考。

---

## 一、zvec DiskANN 中的 PQ 实现

### 1.1 定位与作用

PQ 在 zvec DiskANN 中**不是独立索引类型**，而是 DiskANN 图索引的辅助组件，用于在 Beam Search 过程中提供**压缩向量的快速近似距离估计**，以指导图搜索的剪枝与候选排序。

### 1.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `PQTable` | `diskann_pq_table.h/cc` | 存储码本、转置码本矩阵、计算距离查找表、PQ距离聚合 |
| `DiskAnnPqTrainer` | `diskann_pq_trainer.h/cc` | 采样、训练码本、量化全部向量 |
| `MultiChunkCluster` | `cluster/multi_chunk_cluster.h/cc` | 分chunk聚类算法（K-means + KMC2初始化） |
| `DiskAnnPqMeta` | `diskann_entity.h` | PQ元数据定义（full_pivot_data_size、centroid_data_size、chunk_offsets、chunk_num） |

### 1.3 训练流程

```
1. 随机采样（最多 200,000 个向量，固定种子 456321）
2. 可选 zero-mean 中心化（当前代码默认关闭 use_zero_mean=false）
3. 维度切分：将 D 维空间均分为 chunk_num 个 chunk
4. 每个 chunk 独立聚类：
   - 使用 MultiChunkCluster → KMC2 初始化 + NumericalKmeans
   - 聚类数固定为 256（kPQCentroidNum = 2^8）
   - K-means 最大迭代 12 次
5. 将各 chunk 的聚类中心拼接为完整码本（full_pivot_data）
```

### 1.4 编码（量化）流程

```
1. 对每个向量减去全局 centroid（残差编码）
2. 对每个 chunk 维度子空间，找到最近的聚类中心 ID（uint8）
3. 最终编码为 chunk_num 个 uint8 值的拼接
```

### 1.5 距离计算方式（仅 ADC）

zvec DiskANN PQ **仅使用 ADC（Asymmetric Distance Computation）**：

```
查询时:
1. query 减去全局 centroid
2. 计算距离查找表 dist_table[chunk][centroid_id]：
   - L2: dist = Σ (q[d] - c[d])^2，对 chunk 内每个维度 d
   - IP: dist = Σ -q[d] * c[d]
3. 图搜索时，对候选节点：
   - 读取 PQ 编码（chunk_num 个 uint8）
   - 通过查表 + 累加得到近似距离：score = Σ dist_table[chunk][code[chunk]]
```

**码本存储优化**：码本做了转置存储（`transposed_tables_`），按维度优先排列，提升缓存命中率。

### 1.6 固定参数

| 参数 | 值 | 说明 |
|------|----|------|
| `kPQBitNum` | 8 | 每个 chunk 编码位数 |
| `kPQCentroidNum` | 256 | 每个 chunk 的聚类中心数 |
| `kMeanIterNum` | 12 | K-means 最大迭代次数 |
| `kMaxTrainSampleCount` | 200,000 | 最大训练样本数 |
| 数据类型 | FP32 / FP16 | 支持两种浮点类型 |

### 1.7 存储格式

PQ 数据存储为磁盘段（segment），通过 `DiskAnnPqMeta` 记录：
- `full_pivot_data`：完整码本向量（256 × D × sizeof(T)）
- `centroid`：全局中心向量（D × sizeof(T)）
- `chunk_offsets`：每个 chunk 的维度起始偏移
- `pq_data`：所有向量的 PQ 编码（N × chunk_num bytes）

---

## 二、proxima2 中的 PQ 实现

### 2.1 定位与作用

PQ 在 proxima2 中是一个**独立的索引类型**，具备完整的 Builder/Searcher/Dumper 生命周期，采用**倒排索引 + PQ 编码**的结构。

### 2.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `PqBuilder` | `pq_builder.h/cc` | 索引构建全流程：训练 → label → PQ编码 → dump |
| `PqSearcher` | `pq_searcher.h/cc` | 搜索（支持 SDC/ADC 两种模式，暴力搜索 / 倒排搜索） |
| `PqCodeBook` | `pq_code_book.h/cc` | 码本训练、编码、SDC/ADC距离计算 |
| `PqEntity` | `pq_entity.h/cc` | 倒排索引存储管理、搜索执行 |
| `PqCentroidIndex` | `pq_centroid_index.h/cc` | 倒排索引的一级聚类中心索引 |
| `PqDumper` | `pq_dumper.h/cc` | 索引序列化，支持 block 转储（FastScan） |
| `PqDistanceCalculator` | `pq_distance_calculator.h` | 距离计算调度（普通 / FastScan） |
| `PqAdcDistTable` | `pq_adc_dist_table.h/cc` | ADC 距离查找表，支持 float→uint8 量化 |
| `PqUtility` | `pq_utility.h` | FastScan 转置、对齐等工具函数 |

### 2.3 训练流程

```
1. 一级聚类（Coarse Quantizer）：
   - 使用 StratifiedClusterTrainer 训练
   - 支持 1~2 层级聚类（如 1024 或 1024*8）
   - 聚类算法可选 KmeansCluster / OptKmeansCluster
   - 构建 PqCentroidIndex（用 ClusteringBuilder / HcBuilder）

2. PQ 码本训练（Fine Quantizer）：
   - 将 D 维空间均分为 fragment_count 个 fragment
   - 每个 fragment 独立用 StratifiedClusterTrainer 训练
   - 每个 fragment 聚类数 = 2^fragment_bits（如 4bit → 16 个中心）
   - 使用 KmeansCluster 算法
   - 为每个 fragment 构建 LinearSearcher 用于编码时快速查找最近中心
   - 预计算 SDC 距离表（fragment_cent_num × fragment_cent_num per fragment）
```

### 2.4 编码（量化）流程

```
1. 向量先经 converter 变换（如 MipsConverter for IP metric）
2. 一级聚类：找到最近的粗聚类中心 ID → 分配到对应倒排列表
3. PQ 编码：对每个 fragment 子空间，通过 LinearSearcher 找到最近码本中心 ID
4. 使用 BitStringWriter 将 fragment codes 打包为紧凑 bit string
   - element_size = (fragment_count * fragment_bits + 7) / 8
5. 可选：在倒排列表上再叠加 INT8/INT4 量化器
```

### 2.5 存储格式

采用多段（Segment）存储：

| Segment | 内容 |
|---------|------|
| `pq.centroid` | 一级聚类中心索引 |
| `pq.inverted_body` | PQ 编码后的向量数据 |
| `pq.inverted_meta` | 各倒排列表元数据（offset, block_count, vector_count） |
| `pq.keys` | 向量主键 |
| `pq.code_book{N}` | 各 fragment 的码本搜索器 |
| `pq.code_book_cent{N}` | 各 fragment 的码本质心数据 |
| `pq.dist_table` | SDC 预计算距离表 |
| `pq.features` | 可选的原始向量（用于精排） |
| `pq.int8_quantized_params` | 可选的 INT8 量化参数 |

**Block 存储**：倒排列表中的 PQ 向量按 block 分组（默认 32 个），支持列主序转置（FastScan 加速）。

### 2.6 搜索模式

proxima2 PQ 支持**两种距离计算模式**：

#### SDC（Symmetric Distance Computation）
```
1. query 也被 PQ 编码
2. 距离 = Σ dist_table[fragment][query_code][data_code]
3. 利用预计算的 centroid-centroid 距离表
4. 优点：对称，可用于 query-query 距离
5. 缺点：精度低于 ADC
```

#### ADC（Asymmetric Distance Computation）
```
1. query 保持原始浮点表示
2. 计算 ADC 距离表：dist_table[fragment][centroid_id] = distance(query_fragment, centroid_fragment)
3. 距离 = Σ dist_table[fragment][data_code]
4. 距离表可量化为 uint8（用于 FastScan + AVX 加速）
5. 优点：精度更高
6. 缺点：每个 query 需重新计算距离表
```

### 2.7 搜索流程

```
1. 如果向量数 ≤ brute_force_threshold（默认 1000）：
   → 直接暴力搜索所有向量（SDC 或 ADC）
2. 否则（倒排搜索）：
   → 在 CentroidIndex 中搜索最近的若干粗聚类中心
   → 按 scan_ratio 控制扫描的倒排列表数量
   → 在每个倒排列表内用 PQ 距离搜索
   → 结果合并到 heap，排序返回
```

### 2.8 FastScan 加速（4-bit PQ）

当 `fragment_bits=4` 且 `fast_scan=true` 时启用：
- block 内 32 个向量的 4-bit 编码按列主序转置（`TransposeFastScan`）
- ADC 距离表量化为 uint8 并按 AVX 双 lane 对齐
- 使用 `ailego::FastScanLut::Compute` 和 `ailego::FastScanUnpack::Unpack` 进行 SIMD 加速
- 仅对完整 block（32 个向量）使用 FastScan，尾部数据退回普通 ADC

### 2.9 可配置参数

| 参数 | 说明 |
|------|------|
| `fragment_count` | PQ 子空间数量 |
| `fragment_bits` | 每个 fragment 编码位数（4 或 8） |
| `centroid_count` | 一级聚类数（支持 1~2 级） |
| `cluster_class` | 聚类算法选择 |
| `converter_class` | 向量变换器（如 MipsConverter） |
| `quantizer_class` | 倒排列表上的额外量化器（INT8/INT4） |
| `block_vector_count` | FastScan block 大小（1/2/4/8/16/32） |
| `fast_scan` | 是否启用 FastScan |
| `scan_ratio` | 搜索时扫描倒排列表的比例 |
| `adc` | 是否使用 ADC（否则使用 SDC） |

---

## 三、两者对比

| 维度 | zvec DiskANN PQ | proxima2 PQ |
|------|----------------|-------------|
| **定位** | 图索引的辅助距离近似 | 独立索引类型（IVF + PQ） |
| **索引结构** | 无独立索引，依附于 Vamana 图 | 倒排索引 + PQ 编码 |
| **粗聚类** | 无（全局单一 centroid） | 有（1~2 级 CentroidIndex） |
| **PQ 子空间数** | chunk_num（外部配置） | fragment_count（可配置） |
| **编码位数** | 固定 8-bit | 可选 4-bit 或 8-bit |
| **距离计算** | 仅 ADC | SDC + ADC 双模式 |
| **FastScan** | 不支持 | 支持（4-bit + AVX） |
| **码本查找** | 暴力遍历 256 个中心 | 每个 fragment 用 LinearSearcher |
| **码本距离表** | 运行时计算 query-centroid 表 | SDC 预计算 centroid-centroid 表 + ADC 运行时表 |
| **额外量化** | 无 | 可在倒排列表上叠加 INT8/INT4 |
| **残差编码** | 可选（减去全局 centroid） | 通过 converter/reformer 实现 |
| **搜索方式** | 图搜索（Beam Search）中用 PQ 距离 | 倒排列表扫描 + heap 合并 |
| **序列化** | 4 个 segment（pq_meta + pq_data + ...） | 多个 segment（10+） |
| **metric 处理** | Cosine/IP 转 L2 + 维度调整 | MipsConverter + 多种 converter |

---

## 四、在 zvec 中添加独立 PQ 索引的建议

基于以上分析，在 zvec 中实现独立 PQ 索引需要关注以下关键模块：

### 4.1 必须实现的组件

1. **PqBuilder**：训练粗聚类 → 训练码本 → label → PQ 编码 → dump
2. **PqSearcher**：加载索引 → 计算距离表 → 倒排搜索 / 暴力搜索
3. **PqCodeBook**：码本训练、编码、距离计算（SDC/ADC）
4. **PqEntity**：管理倒排列表存储、block 读取
5. **PqDumper**：序列化索引到多个 segment

### 4.2 可复用的 zvec 现有组件

| 组件 | 复用方式 |
|------|----------|
| `MultiChunkCluster` | 用于 PQ 码本的 fragment 级聚类训练 |
| `IndexFramework` | Builder/Searcher/Dumper/Entity 等基类接口 |
| `CompactIndexFeatures` | 训练数据的特征存储 |
| `IndexMetric` | 距离度量函数 |
| `IndexConverter/Reformer` | 向量预处理（MIPS/IP 变换等） |

### 4.3 需要新增的能力

1. **倒排索引结构**：一级粗聚类 + 倒排列表管理
2. **BitString 编解码**：紧凑 bit 打包（支持非 8-bit 对齐）
3. **SDC 预计算距离表**（可选）
4. **ADC 距离表 + uint8 量化**（用于 FastScan）
5. **FastScan 加速**（可选，4-bit PQ + AVX/SIMD）
6. **多 Segment 序列化格式**
