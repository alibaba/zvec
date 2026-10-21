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


### 查询开关与诊断

所有开关的名称和默认值保持兼容，可在 `bench` / `recall` YAML 的
`QueryConfig.QueryParam` JSON 中配置：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `rabitq_pruning` | `true` | 下界剪枝；关闭时精算全部有效候选 |
| `rabitq_reuse_coarse_dot` | `false` | 多位精算复用 LUT 点积；仅 Turbo IVF 支持 |
| `rabitq_collect_stats` | `false` | 返回扫描和精算计数 |
| `rabitq_segment_timing` | `false` | 返回扫描及查询准备的分段耗时 |
| `rabitq_segment_timing_stride` | `1` | 每 N 个块采样计时，段耗时仅覆盖采样块 |
| `rabitq_export_scan_lists` | `false` | 导出扫描列表及候选数 |
| `rabitq_export_scan_samples` | `false` | 导出每块首、中、尾候选估计值 |
| `refine_stage_timing` | `false` | 返回基础检索与 kFlat 精排阶段耗时 |

点积复用与原浮点点积不完全相同，启用时需要独立验证召回。它可以与
kFlat refiner 组合；统计、分段计时、样本导出及关闭剪枝不能与 refiner
组合。普通 IVF、旧专用格式和 `bf_pks` 不支持 Turbo IVF 的非默认扫描
选项。查询会重新应用全部选项并清空统计，避免复用 context 时泄漏状态。

32 行整块的 AVX-512 融合路径在距离估计时直接生成候选掩码，仅写出
入选候选，按原始行顺序精算并重新检查收紧的堆阈值。它要求映射行存储、
无过滤或分组、开启剪枝且关闭样本导出。尾块和其他 ISA 使用通用回退。
**导出样本会关闭融合路径**，不能用样本导出时的表现代替生产性能验证。

保留 `ivf_rabitq_scan_diagnostic` 作为诊断入口，接受现有 YAML；Turbo
IVF 输出 `baseline`、`unpruned`、`reuse`、`reuse_unpruned` 四种模式，
专用 IVF-RaBitQ 输出 `dedicated` 模式。工具返回结果 ID/分数、扫描计数，
可选导出列表、候选样本或分段计时；输出前缀必须唯一，避免覆盖既有结果。
带 refiner 的配置使用 `--refine-stages`，不导出扫描诊断。

```sh
export TMPDIR="$PWD/test_tmp"
mkdir -p "$TMPDIR"
cmake --build build --target bench recall ivf_rabitq_scan_diagnostic \
  ivf_block_scan_test ivf_rabitq_index_test turbo_rabitq_quantizer_test
build/bin/ivf_rabitq_scan_diagnostic /absolute/path/to/config.yaml ./diagnostic
```

正式性能比较使用 `bench` 与 `recall`，关闭诊断，固定索引、查询、
TopK、nprobe、线程数、reuse 和 refiner 配置。充分预热后轮换版本运行，
保留逐轮数据，并同时报告召回率与 QPS；同 nprobe 不代表同召回。
正确性检查必须同时覆盖关闭样本导出的融合路径及通用回退。

### 开发与历史实验

`refactor/rabitq-quantizer` 从 `b7815384` 开始精简。量化器、融合扫描、
剪枝、文件格式及默认参数保留；查询 JSON 的动态分派与显式类型解析
共享实现，诊断字段和返回类型不变。

历次 `v10_scripts/` 至 `v14_scripts/`、离线回放、微基准、列表统计和
GT 覆盖分析工具属于性能调查材料，已从交付树移出。完整源文件仍可从
`b7815384` 或原分支 `refactor/unified-rabitq` 恢复，例如：

```sh
git show b7815384:v14_scripts/analyze_v17_bench.py > analyze_v17_bench.py
```

本次清理还在仓库上一级的
`local_validation/rabitq-quantizer-cleanup-b7815384/` 保存了完整副本和
SHA256 清单。需要重新开展历史实验时，应在独立目录恢复对应版本的工具
及构建配置；这些脚本含历史服务器路径，不能作为通用测试入口。
