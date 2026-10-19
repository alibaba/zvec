# Linux ↔ macOS 索引兼容性变更说明

更新日期：2026-10-19。面向研发、测试与代码评审人员。

**本轮已撤回数值精度策略调整，保留文件映射、持久化布局、IVF 查询上下文和 locale 无关分词修复。** 数学层、Turbo 和相应数值测试的 15 个文件与本分支 main 基线 `d7b478df` 逐字节一致。已有平台原来使用 FP16、FP32 或对应 SIMD 内核的策略继续沿用。

**验证结果需要分两层理解：迁移后的数据操作和同平台重开稳定性通过；与目标平台独立重新构建的索引严格一致，尚不能全部成立。** 后者发现了 INT8 构建编码差异和近似索引构建波动，详情见第 6 节。没有为消除这些差异再次改动共享数学层，也没有放宽本轮判定阈值。

Linux x86_64 端通过 Rosetta 在 ARM64 Linux 虚拟机上执行，macOS ARM64 端原生。**物理 Linux x86_64 主机尚未验证。**

| 变更 | 当前处理 | 代码来源 |
| --- | --- | --- |
| 页大小、文件映射、图块布局 | 保留 | [732ce336](https://github.com/iaojnh/zvec/commit/732ce3369b97ec4b0abe56ae7e33933d23c6f1dd)，经 `784efc7b` 合入 |
| locale 无关空白分词 | 保留 | [40b9654c](https://github.com/iaojnh/zvec/commit/40b9654c13e5a7bae2d69ef29b9da549b85b801b) |
| IVF 重绑查询上下文后保留 nprobe | 保留 | [4870bbc2](https://github.com/iaojnh/zvec/commit/4870bbc2941a5f2b5a36adf53c92bbb642d241d9) 中的 IVF 部分 |
| FP16 统一 FP32 运算、AILEGO AVX512FP16 分派、稀疏 FP32 改 double 累加 | 本轮撤回 | 恢复为 main 基线的既有实现及原有测试 |
| 目标平台独立构建对照 | 新增验证脚本和结果 | [verify_vector_index_portability.py](scripts/verify_vector_index_portability.py) |

下列保留修复的源码链接固定到 `4870bbc2`；当前数值策略的后续撤回以本次提交差异和本轮源码哈希记录为准。

**1. 文件映射：区分索引偏移与操作系统页边界。**

本次环境中，Linux 的系统页是 4 KB，macOS ARM64 是 16 KB。原实现直接把索引中的偏移传给 `mmap`。文件偏移 4096 满足 Linux 的映射要求，却不满足这台 Mac 的映射要求。

修复后的处理方式如下，其中 `off` 是文件中的逻辑偏移，`page_size` 是当前机器的系统页大小：

```text
excess        = off % page_size
mapping_start = off - excess
mapping_size  = requested_size + excess
returned_ptr  = mapping_base + excess
```

例如从偏移 4096 读取数据时，macOS 可以从偏移 0 建立映射，再向上层返回基址加 4096 的地址。上层仍按文件中的原始位置访问数据。

对应的刷新、解除映射、Linux/NetBSD 重新映射也还原到真实映射基址，避免只修复打开、后续刷新或释放仍失败。入口增加了负偏移、零长度和长度溢出检查。大页映射仍有独立约束，不能据此认为跨平台 hugetlbfs 已获支持。

共享元数据映射层还修复了两个后续问题：

- 允许在旧文件末尾追加元数据，即使该位置仅满足原平台的页对齐要求。
- 遍历元数据链时，读取每一段自己的长度，允许不同平台追加出的元数据段长度不同；同时检查文件边界和异常回链。

刷新失败会返回错误，元数据脏标记在刷新成功后清除。这样可以暴露写入问题，避免将失败当作成功。

主要代码：[File::MemoryMap 及配套操作](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/ailego/io/file.cc)、[IndexMapping](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/framework/index_mapping.cc)。

**2. 图索引布局：读取和扩容沿用文件创建时的对齐规则。**

HNSW、稀疏 HNSW 和 Vamana 的块大小、节点定位或邻居定位会用到页对齐。文件可以映射成功，并不代表可以用目标机器的页大小重新计算这些布局。

本次将布局对齐信息保存在原有元数据保留字段中：一个字段记录布局版本标记，另一个字段记录创建时的对齐大小。元数据结构大小、已有向量记录和图记录保持不变。

重开时先恢复文件的布局参数，再计算块大小和相关寻址参数。旧文件没有标记时，利用旧实现将 128 字节元数据分配为一个完整写入端页的规则，从持久化容量恢复原始对齐大小；无法识别或不合法的布局会被拒绝。

增量写入也继续使用文件的逻辑块大小。尤其是 HNSW 和稀疏 HNSW，邻居块是否写满改为依据逻辑容量判断，避免把目标机器对齐后增加的物理 padding 当作可用邻居空间。稀疏 HNSW 在打开文件后还会重新计算相关布局，并核对持久化块大小。

此外，“元数据段不存在”和“元数据段存在但读取失败”现在分别处理。读取失败时不会将已有索引当成空索引重建，也不会在错误清理过程中刷新不完整状态。

主要代码：[streamer_layout.h](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/algorithm/streamer_layout.h)、[HNSW ChunkBroker](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/algorithm/hnsw/hnsw_chunk.cc)、[稀疏 HNSW 打开流程](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/algorithm/hnsw_sparse/hnsw_sparse_streamer_entity.cc)、[Vamana 块布局](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/algorithm/vamana/vamana_streamer_entity.h)。

**3. IVF：重新绑定查询上下文后保留本次查询的 nprobe。**

`nprobe` 控制 IVF 查询探测的簇数。本轮发现，增删改后关闭重开，有些查询的结果会发生变化；同平台测试也能复现。

原流程是：当前查询先设置 `nprobe`，随后缓存上下文发现索引实例已经变化，重新绑定到新实例；重新绑定时加载的索引默认参数覆盖了当前查询参数，部分查询因此扫描更少的簇。

修复后将本次查询参数单独保存。上下文重新绑定时，先恢复新索引的默认值，再应用本次查询参数，并按新索引重新计算扫描上限。`nprobe=0` 会明确恢复 `scan_ratio` 模式，避免沿用上一条查询的值；参数更新失败也会返回调用方。

本轮完整迁移验证保持 `nprobe=16`、`nlist=32`，没有通过增大扫描范围掩盖问题。这项修复也适用于同平台的上下文复用和重开场景。

主要代码：[IVFSearcherContext](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/algorithm/ivf/ivf_searcher_context.h)、[IVFIndex 查询准备](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/core/interface/indexes/ivf_index.cc)。

**4. 全文索引：空白分词使用固定规则。**

原 whitespace tokenizer 使用受系统 locale 影响的 `std::isspace` 对单字节分类。同一段 UTF-8 文本可能在不同环境中得到不同的 token 边界。

现在明确将 ASCII 空格和 `\t` 到 `\r` 作为分隔符，其他字节保留在 token 中，并保留字节偏移语义。这个规则不等于按全部 Unicode 空白字符分词。若历史索引在不同 locale 规则下已形成不同 token，本次代码修复本身不会改写其中已有的 posting。

主要代码：[WhitespaceTokenizer](https://github.com/iaojnh/zvec/blob/4870bbc2941a5f2b5a36adf53c92bbb642d241d9/src/db/index/column/fts_column/tokenizer/whitespace_tokenizer.cc)。全文分词属于本分支的附加修复，不计入本轮通用向量索引本地对照矩阵。

**5. 为什么撤回数值策略调整。**

此前为了严格比较 Linux 与 macOS 的分数和排序，将部分 ARM FP16 运算改为 FP32，并调整共享数学层与稀疏内积累加精度。这些是全局计算行为变更，即使索引完全不迁移，也可能影响同平台的查询分数、构建结果和吞吐。

本轮采用的兼容性目标是：文件被正确解释、目标平台按其既有计算策略运行、修改后重开与回迁保持数据和查询状态稳定。跨 CPU 的正常浮点差异需要结合目标平台基准解释，不能仅凭两端分数不同就改动全局数值策略。

因此恢复了 AILEGO 的原生 FP16 分支、原 AVX512FP16 分派、Turbo ARM FP16 cosine，以及稀疏 FP32 的原累加实现。原有数值行为测试也恢复；IVF 的查询参数恢复回归测试继续保留。这不是新增一种跨平台计算模式，也没有增加调用方必须设置的参数。

Flat、DiskANN 仍通过共享文件映射路径受益；HNSW、稀疏 HNSW、Vamana 另有自身块布局修复；IVF 另有独立查询上下文修复。mmap 和 buffer 都参加验证，但不能把 buffer 的通过归因于 mmap 实现。

**6. 调整后的验证结果。**

两端使用相同输入文件、算法参数、插入顺序和修改操作，各自独立构建 38 个配置。每个配置覆盖 Linux 构建、Mac 只读打开、默认打开、增删改后重开和 Linux 回迁；另使用 Mac 本地构建与 Linux 本地修改结果作为对照。

所有输入为 FP32。Flat/HNSW/IVF/Vamana 覆盖无量化、FP16、INT8、INT4；另包含 DiskANN FP32、稀疏 Flat、稀疏 HNSW。每种采用 mmap 和 buffer，共 38 个配置。稠密使用 COSINE，稀疏使用 IP。

| 索引 | 配置数 | Mac 迁移操作完成 | 满足首次本地对照全部严格门槛 |
| --- | ---: | ---: | ---: |
| diskann | 2 | 2/2 | 2/2 |
| flat | 8 | 8/8 | 6/8 |
| flat_sparse | 2 | 2/2 | 2/2 |
| hnsw | 8 | 8/8 | 5/8 |
| hnsw_sparse | 2 | 2/2 | 2/2 |
| ivf | 8 | 8/8 | 5/8 |
| vamana | 8 | 8/8 | 6/8 |

| 阶段 | 完成的操作场景 | 满足严格查询判定 |
| --- | ---: | ---: |
| linux_build | 38 | 38/38 |
| linux_return | 38 | 38/38 |
| macos_readonly | 38 | 28/38 |
| macos_default | 38 | 28/38 |
| macos_mutated | 38 | 29/38 |

这里的“严格”有明确含义：同索引同平台重开、Flat 以及可用的全扫描要求 ID/排序一致（完全等分可换序）、逐 ID 分数差不超过 `2e-5`；独立 ANN 构建的共同命中也检查分数，并设置平均 Recall@10 下降不超过 2 个百分点的诊断门槛。后一个门槛是测试前设置的排查标准，不是产品 SLA。

操作与稳定性结果：

- 152/152 个独立进程场景完成，包括两端本地构建与修改、Mac 迁入验证及 Linux 回迁；两端原始数据、更新值、删除结果和文档数检查通过。
- 同平台默认/只读/重复重开：410/410 通过。IVF 的迁入副本额外 48 次重开通过，另有两端本地对照的重复重开检查。
- FP16 的 8 个配置、32 个目标平台对照均通过，最大同平台分数差约 `3.28e-06`。无需本次已撤回的全局精度调整即可通过这些对照。
- 114 份本地构建源文件或 Mac 修改后文件的哈希核对通过；源码、运行库及原数值实现核对通过。
- 相关单元测试：macOS 428 通过、4 个架构用例跳过；Linux 267 通过、6 个架构用例跳过。另有 16/8 个基准测试默认禁用，均无失败。这不是整个仓库的全量测试结果。

**本地重新构建对照中的差异，没有删除或改成通过：**

| 差异 | 观测与诊断 | 能得出的结论 |
| --- | --- | --- |
| 8 个 INT8 配置 | 同一向量 2173 的第 108 维，Linux 构建编码为 66，Mac 构建为 67；全部对应文件均找到该编码。示例查询分数差约 `2.43e-4` | 两端构建产生的量化表示本就不同，迁移不会重新编码它 |
| INT8 分数复算 | 将两份文件中的编码交给同一个 Mac 原生距离内核，分别得到 `0.727654218674` 和 `0.727896809578`，复现各自集合查询分数 | 样本差异由持久化编码解释；目标平台正确计算不等于重新构建后的分数完全相同 |
| HNSW FP32 buffer | 第一次本地构建 Recall@10 为 66.09%，迁入为 63.91%；三次本机构建范围为 63.48%–68.26%，全扫描对照通过 | 迁入结果处于观测到的本地构建波动范围 |
| IVF INT4 buffer | 第一次本地构建 Recall@10 为 72.17%，迁入为 70.00%；三次本机构建范围为 69.13%–73.04%，全列表对照通过 | 近似查询差异与构建波动相符，未发现全列表计算差异 |

三次本机构建只是诊断样本，不能替代统计检验。两个 ANN 配置修改后的召回也在本地修改对照的观测范围内，但原先超过 2 个百分点门槛的记录继续保留。所有量化和近似搜索误差都需要结合产品要求判断，本文不将严格本地对照描述为全通过。

**7. 验证方式与证据。**

本轮复用了相同的数据规模：HNSW FP32 为 20,000×1024，稀疏 HNSW 为 20,000×768，其余稠密为 4096×128、稀疏 Flat 为 4096×768。HNSW 和稀疏 HNSW 新增 1500 条，其余新增 64 条，各更新 1 条、删除 1 条。修改后不执行 optimize，直接重开与回迁。

每份快照使用 23 个 Top-10 查询，并以原始向量 FP64 距离生成召回参考。HNSW/Vamana 增加线性扫描对照，IVF 普通查询保持 `nprobe=16`，额外用 `nprobe=32` 检查全列表。DiskANN 没有暴露相同全扫描入口，使用逐命中 FP64 距离与本地构建结果对照，未伪造全扫描检查。

仓库验证脚本的 `local` 阶段在两端分别创建索引和修改后基线；`cross` 在 Mac 读取 Linux 文件，与 Mac 本地结果比较并产生回迁副本；`return` 在 Linux 读取回迁副本，与 Linux 本地修改结果比较。脚本拒绝覆盖现有结果，失败保留；任一严格检查失败均返回非零退出码，JSON 的 `execution_success` 单独记录数据操作是否完成。需要两端各自构建的运行库，以及同一份 `datasets` 输入文件。

仓库中的验证脚本与本文随本次调整提交。大型索引、运行库、输入样本及原始诊断产物保留在本地目录：`/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019`，不随代码推送。下列证据链接指向本次执行机器的本地文件，团队获取源码后不会自动获得这些产物。

- [完整验证报告](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/report.txt)、[机器可读汇总](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/summary.json)。
- [原数值实现逐文件核对](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/numerical_baseline_manifest.json)、[运行库哈希](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/runtime_manifest.json)。
- [INT8 编码差异](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/diagnostics/quantization_comparison.json)、[同一 Mac 内核复算](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/diagnostics/native_score_reproduction.json)。
- [同平台重复构建对照](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/diagnostics/local_build_variability.json)。
- [实际运行的冻结脚本](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/verify_vector_index_portability.py)、[调度脚本](/Users/ali.yzf/zvec/cross_platform_indexes_20261009/x86_vector_local_reference_validation_20261019/run_matrix.py)。仓库版进行了格式化、显式长度校验，并在严格失败时返回非零退出码；冻结脚本仅按执行失败退出，以便继续收集所有诊断。152 项本地对照的比较逻辑与保存结果均核对等价，CLI 冒烟也确认严格失败返回非零，记录保存在诊断目录。

此前 `x86_vector_fixed_validation_20261019` 的 190/190 是启用数值策略调整时的历史结果，不能作为当前版本的结论。原报告与产物保留，没有覆盖。

**8. 尚未覆盖的范围。**

物理 Linux x86_64 仍需可访问的原生主机完成同一流程；当前 Rosetta 验证不能替代。AVX512 硬件、Windows、macOS Intel、大端、跨平台 hugetlbfs、混版本部署以及所有维度/指标/输入类型均未获得完整验证。RaBitQ 未列入 macOS 通用向量索引矩阵。

本轮回迁使用两端均具备保留修复的软件版本，不代表可以回滚到旧软件。集合层增删改通过，也不等于 IVF/DiskANN 底层结构新增了原地更新能力。没有进行专项性能基准；原数值内核虽已恢复，文件映射和其他保留修复的总体性能仍需独立测量。
