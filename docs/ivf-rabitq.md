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
