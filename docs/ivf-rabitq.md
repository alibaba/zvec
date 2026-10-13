# IVF 的 RaBitQ 量化模式

通过现有 IVF 类型选择 RaBitQ，无需创建独立的 `IVF_RABITQ` 索引。

```python
import zvec

index_param = zvec.IVFIndexParam(
    metric_type=zvec.MetricType.L2,
    n_list=1024,
    n_iters=20,
    quantize_type=zvec.QuantizeType.RABITQ,
    total_bits=7,
    sample_count=0,
)
query_param = zvec.IVFQueryParam(nprobe=16)
```

将 `index_param` 传给 `VectorSchema`，将 `query_param` 传给 `Query`。
索引类型和查询类型均保持 `IVF`，参数随 collection manifest 保存。

C++ core 接口：

```cpp
zvec::core_interface::RabitqQuantizerParam quantizer(7);
quantizer.num_clusters = 16;  // RaBitQ 模型中心，与 IVF 列表数独立
quantizer.sample_count = 0;
quantizer.niters = 20;
auto param = zvec::core_interface::IVFIndexParamBuilder()
    .with_data_type(zvec::core_interface::DataType::DT_FP32)
    .with_dimension(128)
    .with_metric_type(zvec::core_interface::MetricType::kL2sq)
    .with_n_list(1024)
    .with_n_iters(20)
    .with_quantizer_param(quantizer)
    .build();
```

- `total_bits` 为每维量化位数，范围 1–9，默认 7。
- `sample_count` 为训练采样数量，0 使用全部训练向量。
- `n_list` 和 `n_iters` 必须为正数；`nprobe` 使用现有 IVF 查询参数，RaBitQ 模式下必须大于 0；超过 `n_list` 时由后端裁剪。
- 支持 FP32 及 L2、IP、Cosine；core 为 2–4095 维，collection 保留 64–4095 维限制。
- 沿用现有 RaBitQ 平台限制：Linux x86_64，AVX2/FMA 或 AVX512F/BW/DQ。
- 支持 group-by、过滤、普通查询与分组查询交替，以及 collection 层 refiner。
- 与独立索引一样，不支持 group-by 与 refiner 同时使用。
- RaBitQ 自带旋转，core 的 `enable_rotate` 不增加额外旋转；collection 仍拒绝该选项。不支持 SOAR。
- core 层 RaBitQ postings 不保存原始向量，不支持 `fetch` / `fetch_vector`。
  collection 层继续通过原始 Flat 存储提供向量和重建输入。

标准 HNSW 和 IVF 均使用 Turbo `RabitqQuantizer`：训练、旋转、编码、
完整距离估计和模型序列化共享一份实现。IVF 使用 `IVFBuilder` / `IVFStreamer`
构建和扫描标准 IVF postings。HNSW 使用单记录的 1-bit 下界筛选图邻居；
IVF 使用 32 路 FastScan 粗筛，仅对下界通过当前结果阈值的候选计算完整位数距离。
IVF 的分组查询在扫描时保留每组 top-k，支持过滤，不需要保留全部候选。

C++ 推荐通过 `RabitqQuantizerParam` 配置模型，显式类型参数优先。
旧的 `QuantizerParam(kRabitq)` 加 IVF `total_bits` / `sample_count` 仍可用，
此时 `nlist` / `niters` 同时用作模型聚类设置。Python/C 接口继续使用该兼容映射。
IVF 列表聚类和量化器模型训练分别执行；模型中心不强制等于列表中心。

新文件保存标准 IVF header、`RabitqQuantizer` descriptor 和
`ivf.turbo_quantizer` 模型段，以及可选的 `ivf.turbo_scan.v1` 粗筛段。
粗筛段按 posting block 顺序保存 32 个模型中心 ID 和 RaBitQ `BatchDataMap`，
不足 32 个候选的尾块补零；每个候选仍使用自己的模型中心修正距离。
加载以磁盘 descriptor 为准，默认量化参数也能重开。
没有粗筛段的已有 Turbo IVF 文件继续使用原有完整距离扫描；重建后才获得 FastScan。
旧的独立 IVF-RaBitQ 文件通过 header 自动选择兼容加载器；独立 API 保留原格式。
独立 API 不支持读取新的 Turbo IVF 文件，需要通过标准 IVF API 打开。

验证入口：`ivf_rabitq_index_test`、`ivf_turbo_index_test`、
`manifest_codec_golden_test`、`schema_test`，以及 Python 的
`test_params.py` 和 `test_collection_ivf_quantized_rabitq.py`。
Linux 专属测试覆盖 L2/IP/Cosine、1/7/9 bit、重开、普通 IVF 与 RaBitQ
交替查询，以及从未量化索引合并构建。

迁移构建配置时保留所需的 `n_list`、`total_bits`、`sample_count` 和 metric。
共享量化器使用自己的模型中心与旋转，因此新旧构建的编码、分数和召回率不保证相同。
加载同一份旧文件的兼容入口应返回与独立入口相同的查询结果。

C API 复用原有 RaBitQ 参数访问器，无需新增函数或改变索引类型：

```c
zvec_index_params_t *p = zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
zvec_index_params_set_quantize_type(p, ZVEC_QUANTIZE_TYPE_RABITQ);
zvec_index_params_set_ivf_params(p, 1024, 20, false);
zvec_index_params_set_ivf_rabitq_params(p, 1024, 7, 0);
// zvec_index_params_get_ivf_rabitq_params also accepts this IVF object.
zvec_index_params_destroy(p);
```

collection manifest 中的索引类型仍分别是 `IVF` 和 `IVF_RABITQ`，
已有 collection 重开后继续使用对应的查询参数类型。
标准 IVF 的 BufferPool 分页现在同样适用于新 RaBitQ postings。
旧格式兼容加载仍使用原后端的内存布局。

新建索引使用 RaBitQ 库的高精度 FastScan 内核，查询 LUT 每次查询构建一次。
超过 1024 维时分段累加，避免 SIMD 累加器溢出。普通搜索使用当前 top-k 的
最差距离作为筛选阈值；分组搜索使用候选所属组的阈值，过滤先于精算执行。
全零查询回退完整距离扫描，避免 LUT 的零范围除法。

这版保留原有行记录，在额外数据段保存粗筛码，以兼容向量读取和合并流程。
额外空间约为每个完整块 `32 × (padded_dim / 8 + 16)` 字节；768 维约
112 字节/向量，100 万向量约 107 MiB，另有尾块填充。构建时暂存该粗筛段，
还需相应内存和容器扩容余量。后续可再将两类编码合并为独立的存储布局。
多位精算沿用现有完整距离计算；1-bit 模式直接使用高精度 LUT 粗估距离，
与旧单记录估计器可能存在小幅数值差异。下界筛选仍需评估召回率。

`ivf_block_scan_test` 在通用平台验证筛选、过滤、分组、尾块、三种存储、
并发查询、旧文件回退和损坏数据段。`turbo_rabitq_quantizer_test` 在支持
RaBitQ 的平台对照独立批量编码器和估计器验证布局与数值，涵盖 1/7/9 bit、
L2/IP/Cosine 和大于 1024 维的情况。

```sh
cmake --build build --target ivf_block_scan_test turbo_rabitq_quantizer_test ivf_rabitq_index_test
ctest --test-dir build --output-on-failure -R '^(ivf_block_scan_test|turbo_rabitq_quantizer_test|ivf_rabitq_index_test)$'
```

复测报告中的 cohere1M 时，需重建 Turbo IVF 索引并保持数据、线程数、
`nlist`、`nprobe`、位数和计时方式一致，同时报告 QPS、recall、索引体积和内存。
运行时、召回率及性能需要在支持 RaBitQ 的 Linux x86_64 机器上验证，
macOS 编译和通用 IVF 测试不能替代 SIMD 路径验证。

后续扫描优化不改变量化编码、距离计算或文件格式，可直接复用已有
`ivf.turbo_scan.v1` 索引，无需重新构建。mmap 存储直接访问映射中的
keys、粗筛码和行记录；分页及文件读取使用可复用的窗口缓冲区，每窗口
最多 32 个块且粗筛数据不超过 64 KiB（单块超过上限时单独读取）。
完整行仍仅在块中存在通过下界筛选的候选时读取；整窗口被过滤时不读粗筛码。
普通查询还省去逐候选的过滤/分组分支，并缓存随 top-k 更新的阈值。

内部诊断接口 `IVFSearcherContext::entity()->packed_scan_stats()` 提供
`coarse_count`、`refined_count`、`key_reads`、`coarse_reads`、`row_reads`
和 `coarse_copy_bytes`。统计随 `bind_query()` 重置，累计该查询扫描的列表；
批量查询结束后仅保留最后一个查询，评测需逐查询汇总。
读取次数是扫描器发起的逻辑读取调用数，不代表系统调用、缺页或物理磁盘 I/O；
mmap 直接访问的这几项读取计数为零。`refined_count` 是精算接口调用数，
1-bit 模式该接口直接返回粗估分数。

同索引召回率对照可在查询前调用
`IVFSearcherContext::entity()->set_packed_scan_pruning(false)`，关闭下界剪枝，
保留相同的编码、查询估计器和扫描顺序，对所有未过滤的有效候选调用精算。
该开关默认开启，查询绑定不改变设置，clone 继承设置；对照完成后设回 `true`。
公开入口现在也支持下述查询参数。比较同一索引剪枝开关的 top-k 及 recall，
再比较优化前后的 QPS，可以分别检查召回率损失和读取优化收益。


### 点积复用和服务器 A/B 验证

本轮仍复用 v2/v3 索引，不改变文件格式或模型中心。新增的查询参数可直接放进
现有 `bench` / `recall` YAML 的 `QueryConfig.QueryParam` JSON 中：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `rabitq_pruning` | `true` | 允许下界剪枝；关闭时精算所有未过滤的有效候选 |
| `rabitq_reuse_coarse_dot` | `false` | 精算复用 FastScan 的高精度 LUT 点积，省去二进制码浮点点积重算 |
| `rabitq_collect_stats` | `false` | 在 `SearchResult::ivf_scan_stats_` 返回该次查询计数 |

例如（开启点积复用，保留剪枝）：

```yaml
QueryConfig:
  QueryParam: '{"index_type":"kIVF","topk":50,"nprobe":256,"rabitq_pruning":true,"rabitq_reuse_coarse_dot":true}'
```

默认值保留 v3 的距离估计方式。复用模式只改变多位精算使用的点积，仍按每行自己的
量化中心修正距离；1-bit 模式不受该开关影响。LUT 近似值与原浮点点积不完全相同，
必须单独验证分数、排名和召回率。普通 IVF、旧专用格式、refiner 和 `bf_pks`
组合不能启用这些非默认选项，以免静默忽略。零查询或没有粗筛段的旧 Turbo 文件
仍回退原路径，统计的 `active=false`，不能据此宣称点积复用生效。
每次公开查询都会重新应用全部选项，结果统计也会清空，避免线程局部 context 和
重复使用 `SearchResult` 时遗留上一查询的模式或计数。

新工具 `ivf_rabitq_scan_diagnostic` 接受现有 YAML（FP32 文本查询、单个 TopK、
无过滤/分组/refiner），在同一个已加载索引上逐查询运行四种模式：

| 输出模式 | 剪枝 | 点积复用 |
|---|---|---|
| `baseline` | 开 | 关 |
| `unpruned` | 关 | 关 |
| `reuse` | 开 | 开 |
| `reuse_unpruned` | 关 | 开 |

在服务器仓库目录编译并运行测试：

```sh
cmake -S . -B build_bench -DBUILD_TOOLS=ON
cmake --build build_bench --target bench recall ivf_rabitq_scan_diagnostic \
  ivf_block_scan_test ivf_rabitq_index_test turbo_rabitq_quantizer_test -j8
ctest --test-dir build_bench --output-on-failure \
  -R '^(ivf_block_scan_test|ivf_rabitq_index_test|turbo_rabitq_quantizer_test)$'
export RABITQ_BIN="$PWD/build_bench/bin"
```

然后进入原先运行评测的目录。下面的 `CONFIG` 替换成现有 np256 YAML 的绝对路径；
索引、查询文件和 ground truth 均使用原文件。工具保留 YAML 内的相对路径，
所以生成配置与后续 bench/recall 必须在同一个工作目录运行。

```sh
CONFIG=/absolute/path/to/existing_np256.yaml
"$RABITQ_BIN/ivf_rabitq_scan_diagnostic" "$CONFIG" ./v4_np256 > v4_np256.diagnostic.log 2>&1

for mode in baseline unpruned reuse reuse_unpruned; do
  mkdir -p "v4_np256.$mode.recall"
  "$RABITQ_BIN/recall" "v4_np256.$mode.yaml" > "v4_np256.$mode.recall.log" 2>&1
  "$RABITQ_BIN/bench" "v4_np256.$mode.yaml" > "v4_np256.$mode.bench.log" 2>&1
done
```

只有诊断工具退出码为 0 时继续使用生成的配置；非零退出时 CSV 可能不完整。
输出前缀需唯一，工具会覆盖同名前缀的文件。

- `v4_np256.queries.csv`：每个输入查询、每种模式的计数，是否启用 packed scan、
  是否三段数据均为稳定映射、相对 baseline 的结果集合交集数、ID 顺序是否一致，
  以及公共 ID 的最大分数差。查询编号是从 0 开始的有效输入行序号。
- `v4_np256.results.csv`：四种模式完整的结果 ID、rank（从 0 开始）、公开接口分数。
- `v4_np256.{mode}.yaml`：可直接用于 bench/recall，关闭统计导出，保留原评测
  参数与 ground truth 配置，并将 recall 输出目录按模式隔离。
- 标准输出汇总每个模式的 coarse/refined 总数及精算比例。读取次数仍是逻辑调用数，
  mmap 的读取计数为零不代表没有缺页或内存访问。

该工具不输出 QPS，避免 CSV 写入、查询交替和结果对比污染计时；结果交集也不是
真实 recall，真实 recall 必须使用原 ground truth。建议先测 np16/64/256，
正式 bench 保持相同线程数、30 秒时长和独占环境，预热后交替顺序至少重复三次。

先比较 `baseline` 与 `unpruned`，隔离原精算方式下剪枝的影响；比较 `unpruned`
与 `reuse_unpruned`，隔离点积近似本身的影响；最后比较 `baseline` 与 `reuse`
的 QPS、recall，并用各自的精算比例判断提速来自每次精算成本还是候选数变化。
本地 macOS 只能验证通用路径和 x86 语法，RaBitQ SIMD 的运行测试与性能以服务器结果为准。

### 专用版与集成版的扫描工作量对照

`ivf_rabitq_scan_diagnostic` 现在也接受 `IndexConfig.index_type=kIVFRabitq`
和 `QueryParam.index_type=kIVFRabitq` 的专用索引配置。直接使用原有专用 YAML，
无需重建索引。专用版只输出 `dedicated` 模式：维持原有下界剪枝和 LUT 点积复用，
仅开启统计，不实现集成版的三个算法切换模式。

`IVFRabitqQueryParam.rabitq_collect_stats` 默认 `false`，支持 JSON、复制和 clone。
开启时普通列表扫描通过 `SearchResult::ivf_scan_stats_` 返回计数；
refiner、group-by、`bf_pks` 组合会报参数错误，避免返回未覆盖路径的统计。
关闭统计时，专用扫描循环使用无计数的模板实例。

计数定义：

- `coarse_count` 是实际批估计的向量数，不计 32-lane 块的补齐空位，不是理论均衡值。
- 多位编码的 `refined_count` 是实际调用 extra-bit 精算的次数，不是最终入选数量。
  1-bit 没有 extra-bit 精算，该字段计通过候选判断后送入堆的次数；专用版和集成版
  对等分数的边界判断略有不同，不能将其解释为两侧等价的精算耗时。
- 统计每次查询重新清零，不在索引共享实体上累计。`mapped` 表示所需数据段具有
  稳定映射；专用版查询直接访问打开索引时取得的视图，因此查询期间读调用数为零。
  该值不代表没有访存或缺页。
- 汇总日志新增粗筛数、精算数的 mean/std/min/p50/p95/max；std 为总体标准差，
  分位数使用 nearest-rank。它们描述查询工作量分布，不是全体列表大小分布。
- 专用 CSV 中 overlap 等字段与该索引自身的首个模式对照，不能当作跨索引一致性
  或 GT recall。结果 CSV 保留每个查询的 ID 和分数，可独立核对结果集合。

在仓库目录编译与测试（Linux x86_64）：

```sh
cmake -S . -B build_bench -DBUILD_TOOLS=ON
cmake --build build_bench --target ivf_rabitq_scan_diagnostic bench recall \
  ivf_rabitq_index_test ivf_rabitq_streamer_test -j8
ctest --test-dir build_bench --output-on-failure \
  -R '^(ivf_rabitq_index_test|ivf_rabitq_streamer_test)$'
export RABITQ_BIN="$PWD/build_bench/bin"
```

进入原先的评测工作目录，分别替换两份原有 YAML 的路径，并确保它们使用完全相同的
FP32 查询文件及顺序、topk、GT、距离类型。先测 np16/64/256；下面以 np16 为例。
输出前缀需唯一；工具会覆盖同名前缀文件，退出码非零时不要使用该次产物。

```sh
"$RABITQ_BIN/ivf_rabitq_scan_diagnostic" /absolute/path/integrated_np16.yaml \
  ./v5_integrated_np16 > v5_integrated_np16.diagnostic.log 2>&1
"$RABITQ_BIN/ivf_rabitq_scan_diagnostic" /absolute/path/dedicated_np16.yaml \
  ./v5_dedicated_np16 > v5_dedicated_np16.diagnostic.log 2>&1
```

确认两次退出码均为 0 后，分别用生成的 `v5_integrated_np16.baseline.yaml`、
`v5_integrated_np16.reuse.yaml`、`v5_dedicated_np16.dedicated.yaml` 运行 `recall`
和 `bench`。例如专用版：

```sh
mkdir -p v5_dedicated_np16.dedicated.recall
"$RABITQ_BIN/recall" v5_dedicated_np16.dedicated.yaml \
  > v5_dedicated_np16.dedicated.recall.log 2>&1
"$RABITQ_BIN/bench" v5_dedicated_np16.dedicated.yaml \
  > v5_dedicated_np16.dedicated.bench.log 2>&1
```

生成配置已关闭统计。保持相同线程数、预热和运行时长，交替顺序多轮测试。
先对照相同 nprobe 的实测工作量，再按 GT recall 增补 nprobe 点，找到相近召回后
比较 QPS 与实际候选数。诊断工具自身不计时；数量占比不能代替耗时占比。
这一步可判断候选数是否解释主要差距，但不保证两份索引采用相同分桶；共享中心和
分配的受控实验、全量列表大小分布和分项 profiling 仍是后续独立验证项。

### v9 服务端执行清单：先校正诊断，再定位 99% 召回差距

本轮修复针对 v8 的计时口径，不更换索引、不修改量化算法或默认搜索参数。
`rabitq_segment_timing` 默认关闭；开启时也会返回扫描统计，无需另外打开
`rabitq_collect_stats`。本提交尚未实现真实选桶导出或专用侧禁用剪枝，见步骤 4。

#### 1. 构建和回归

在 Linux x86_64 仓库目录执行。所有临时文件放在当前目录或上一级。

```sh
mkdir -p ../local_validation/v9/tmp
export TMPDIR="$(cd ../local_validation/v9/tmp && pwd)"
cmake -S . -B build_bench -DBUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_bench --target bench recall ivf_rabitq_scan_diagnostic \
  ivf_rabitq_gt_coverage ivf_block_scan_test ivf_rabitq_index_test \
  ivf_rabitq_streamer_test turbo_rabitq_quantizer_test -j8
ctest --test-dir build_bench --output-on-failure \
  -R '^(ivf_block_scan_test|ivf_rabitq_index_test|ivf_rabitq_streamer_test|turbo_rabitq_quantizer_test)$'
export RABITQ_BIN="$PWD/build_bench/bin"
git rev-parse HEAD
```

记录提交、编译配置、CPU/指令集和线程/绑核设置。需要确认修复前后的默认路径
吞吐时，保留父提交 `fa7a94e1` 的独立构建及配套动态库，避免混用构建产物。

#### 2. 分开运行无计时与有计时的诊断

回到 v8 使用的评测工作目录，保持 YAML 中相对路径的解析目录一致。
复用 norm1、dedicated 索引和原查询/GT；`CONFIG` 替换为现有集成 np293 YAML。

```sh
CONFIG=/absolute/path/to/v8_base_np293.yaml
RABITQ_RUN="$PWD/v9/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RABITQ_RUN/tmp"
export TMPDIR="$RABITQ_RUN/tmp"
"$RABITQ_BIN/ivf_rabitq_scan_diagnostic" "$CONFIG" "$RABITQ_RUN/plain" \
  > "$RABITQ_RUN/plain.log" 2>&1
"$RABITQ_BIN/ivf_rabitq_scan_diagnostic" "$CONFIG" "$RABITQ_RUN/timed" \
  --segment-timing > "$RABITQ_RUN/timed.log" 2>&1
```

分别确认退出码为 0 后再使用产物。工具默认不再计时；显式
`--segment-timing` 只支持集成侧，专用侧传入会报错。生成的所有 bench/recall
YAML 始终关闭统计和计时，模式的剪枝/reuse 配置仍按既有四模式生成。

核对：两次 `results.csv` 的 ID、排序和分数字节一致；逐查询 coarse/refined
计数一致；普通诊断的所有时间列为 0。计时诊断中各段非负，分段和不超过
`query_ns`，差额记录为未分配时间，不强制把分段和归一成 100%。

时间列改为：

| 字段 | 实际边界 |
|---|---|
| `prep_ns` | 中心检索/变换 + bind，一次计入；批量查询均摊共享准备时间 |
| `cand_ns` | block 的 key 掩码与粗码准备/读取，不含后续剪枝 |
| `coarse_ns` | `estimate()` 与该 block 的扫描计数更新 |
| `post_scan_ns` | 候选遍历、剪枝、行读取、精算及堆操作，包含全部剪枝的 block |
| `finalize_ns` | 堆排序、key 映射及结果归一化 |
| `query_ns` | 搜索内部查询总时间，含未分配间隙和结果生成；不是公共 API 全程耗时 |

旧 `refine_ns` / stdout `refine` 改名为 `post_scan_ns` / `post_scan`，分析脚本
需按列名更新，不能解释为纯精算耗时。计时使用 `steady_clock` 显式纳秒，去掉
TSC 首次校准等待；与 v8 的 TSC 数值不能直接拼表。按 block 读钟仍有开销，
分别量化计时开/关的吞吐变化，不假定开销均匀或分段占比不受影响。

#### 3. 重测性能与诊断开销

- 配置：baseline np293、reuse np293、dedicated np249；同查询/GT、TopK=50、
  Cosine、7bit。若声称 reuse 的最快达标点，补查 np292 及邻近点。
- 先复现 recall 与候选计数，再以关闭统计/计时的配置测 1/4 线程。
- 固定运行条件，预热至无明显爬升；每轮 30 秒，轮换先后顺序，至少 5 轮。
  保留所有轮次，输出每轮 QPS 和每轮 dedicated/baseline、reuse/baseline 比值，
  再给中位数与范围。若仍明显漂移，单列不稳定结果并补跑，不事后只剔除低值轮。
- 默认路径父提交/本提交 A/B，以及本提交计时开/关 A/B，分开报告。
  不从诊断工具 CSV 写入过程推导 QPS，也不把 QPS 下降百分数当成延迟增长百分数。
- v8 四线程 reuse 五轮有三轮低于 baseline，因此高召回 reuse 的小幅收益仍需
  验证；低召回已观察到的收益不自动外推。

#### 4. 准确性归因：需要服务端另补诊断代码

按两个独立小改动实现并单独验证，不混入性能优化：

1. 默认关闭的真实选桶导出：在实际搜索路径记录查询 ID、实际扫描列表 ID、
   每列表扫描量及是否受 max_scan 限制。两侧分别在 np293/249 和 np256 导出，
   与 GT->list 映射计算覆盖率，核对每查询计数。现有离线工具使用 double 点积，
   不能当作 FP32 搜索的逐位复刻；扫描向量总数相同也不证明列表集合相同。
2. 专用侧默认关闭的禁用剪枝诊断：固定量化码、列表及距离公式，只跳过下界
   剪枝；验证全候选精算计数和默认路径结果不变。再分离两侧的候选覆盖不足、
   unpruned 量化排序损失、pruning 净影响。关闭剪枝不等于原始向量精确排序。

统一输出 `query, covered_gt, baseline_gt, unpruned_gt, reuse_gt`，GT@50 分母
应为 1000*50；缺失、重复 GT 或维度不一致应报错。明确变化查询数与净命中数，
解释 v8 “26 个命中、25 个 query 各 +1”的矛盾。覆盖率接近只能支持本数据集和
所测选点的局部判断，不能直接推出分区普遍等价。

#### 5. 速度归因：正确识别现有内核

`RabitqBlockScanner::estimate()` 已调用
`rabitqlib::fastscan::accumulate_hacc()`，其运行时派发为 AVX2/AVX512。
不能把整个 estimate 称为标量核，再把替换 SIMD 作为预设方案。

先确认服务器实际运行的派发分支；在相同向量维数、block 数和缓存条件下，
用重复批量调用、循环外计时分别测累加核与整个 estimate，保留校验和防止消除
计算。两者差异才用于调查逐候选模型中心读取、距离/下界修正、输出写入等成本。
专用侧用同口径对照；单侧占比不能独立解释跨实现差距。计时仍不能区分的项目
保留为未知，不用粗扫量比乘精算量比推算耗时。

#### 6. 报告交付

追加 v9：提交号及测试结果、诊断开/关一致性、默认路径回归、所有轮次 QPS、
真实选桶覆盖率、桶内损失分解、修正后的计时边界与开销、下一项最小优化。
保留 YAML、CSV、原始日志和分析脚本；更新脚本读取 `post_scan_ns/query_ns`。
将结论区分为实测、推断、待验证，并注明尚未完成的步骤。
