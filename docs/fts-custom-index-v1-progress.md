# FTS 自定义索引 v1：实施记录

日期：2026-11-18。主规范仍为 [fts-custom-index-v1-implementation.md](fts-custom-index-v1-implementation.md)。本文件记录代码、验证证据和剩余工作，不替代设计规范。P1 独立读写和 P2 查询路径已完成本机验收；P3/P4 的直接转换、持久发布、合并迁移及 DDL 已接入内部门禁，本机关键故障闭环通过。两个指定数据集的 fts-bench 对照已完成，生产 writer 仍默认关闭。后续日期相同的续作记录覆盖下文较早的“尚待”状态。

### 2026-11-18 mmap 热路径修复与双机复测

将 cursor 窗口命中检查及视图访问内联，复杂 refill 保持独立；mmap 不再创建无用 section 句柄，热状态位于新增对齐状态之前。8 MiB buffer 的物理页对齐与计费优化保留，缓存预算、RocksDB、持久格式、评分和生产门禁不变。超长命中仍报错且保留原缓存，边界、预算及 I/O 错误传播继续有效。

191 三版本交错 3 进程 × 3 遍：原始 / 上一版 / 修复版 mmap 为 **1063.24 / 1039.65 / 1060.22 QPS**，修复版距原始版 −0.28%；buffer 8 MiB 为 **317.24 / 406.67 / 410.90 QPS**，原有收益保留（相对原始版 +29.52%）。固定 CPU 50 另测 mmap **1054.32 / 1034.36 / 1058.53 QPS**，各轮均保留，包括修复版的两个慢轮次。Quora 子集 mmap +1.58%，buffer −1.66%，不声称全部指标提升。

Mac 主矩阵受明显后台负载影响，buffer 修复版相对上一版 −5.37%，性能检查失败如实保留。随后六组相邻、交替顺序的补测：mmap 原始/修复中位数 **2235.94 / 2406.25 QPS**，buffer 上一版/修复 **2158.97 / 2204.44 QPS**；配对比值中位数为 **1.0648 / 1.0134**。补测未显示稳定回退，但仍有较大波动，不能作为 Mac 精确提速比例。

两机相同 1396 文件，各先编译目标再执行绝对路径，均 **85 项单测通过**；Mac `_zvec` 重建。**342000 次正式 + 44000 次预热输出**逐位匹配各自机器的原有参考；修改范围格式及 diff 检查通过，临时插桩已移除，未创建 commit。完整原因、异常路径及数据见 [mmap 修复报告](fts-custom-index-v1-du-analysis.md)。最终证据目录为桌面 `zvec-fts-v1/mmap-fix-final-20261118-01/`，其中 `final/` 为当前 G；诊断目录 `mmap-fix-20261118-01/final/` 为中间 F，不能混用。

### 2026-11-18 固定 8 MiB：物理页窗口优化与双机验证

定位到 FTS section 相对的 4 KiB 窗口没有对齐原生物理页，191 上经常跨两个页，导致额外读取、复制和淘汰。最终 buffer 路径改为物理页对齐、section 边界裁剪，并按实际跨页计费；保留单 cursor 有界窗口、零缓存 bypass、范围预算和校验。mmap 保留原有窗口与计费。未扩充 pool、未关闭直接 I/O、未改 RocksDB 或持久格式，没有语料专用分支。

du 完整 2000 查询，交错 3 进程 × 3 遍：固定 8 MiB，Mac QPS **2149.83 → 2315.60（+7.71%）**，191 **317.34 → 413.84（+30.41%）**；p95 分别 **0.994 → 0.945 ms / 5.859 → 4.499 ms**。Mac 每次测量前统一 mmap 预热文件页，不能混用未主动预热时约 671 QPS 的诊断基线。191 单独采样物理读取 **48999 → 32394（−33.89%）**；其 8 MiB pool 仍存在明显容量成本，未追平 mmap/64 MiB。

Quora 固定 1000 查询的 buffer 8 MiB 三遍中位 QPS：Mac **297.23 → 320.91（+7.97%）**、191 **76.92 → 93.13（+21.08%）**。du mmap 控制组仍有 **−0.92% / −2.39%** 的 QPS 变化，明确保留此代价，不声称全面无回退。详细控制组、RSS、方法边界和后续候选见 [固定预算优化报告](fts-custom-index-v1-du-analysis.md)。

两机相同的 1396 个一方源码文件；各先编译具体测试目标，再运行绝对路径，**85 项通过**，Mac `_zvec` 同步重建。**18 万次正式查询 + 1.4 万次预热查询**的有序 DocID / FP32 bits 与同机参考逐位一致，新增窗口复用、尾页、bypass 和紧预算回归通过；本次修改范围格式及 diff 检查通过。最终证据为桌面 `zvec-fts-v1/buffer8-opt-20261118-01/summary.json` 及 `candidate-v4` / `v4-*`；较早中间变体仅作诊断。未创建 commit，生产 writer 门禁未变。

### 2026-11-18 当前优化版 du：Mac / 191 检索续测

按当前相同源码在两机完成 RocksDB、custom mmap、buffer 8/64 MiB 检索对照。191 重新编译并通过 84 项相关测试；Mac 复用上一轮源码指纹完全一致的冻结程序。191 原有 du 文件字节与 Mac 不同，本次使用复制到独立实验目录的 Mac 输入，核验 SHA-256 一致后再新建索引。

每模式 3 个独立进程、每进程完整 2000 查询 3 遍，单线程 OR Top100，以下取 9 遍中位数。Mac QPS（RocksDB / mmap / buffer 8 / 64 MiB）：**2075.55 / 2488.34 / 2161.21 / 2249.51**；191：**810.12 / 1059.10 / 316.81 / 908.75**。mmap 相对同机 RocksDB 提升 19.9% / 30.7%；191 的 8 MiB buffer 仍为明显短板，64 MiB 为其 2.87× QPS，但存在首遍成本，不能仅用后续遍描述启动表现。

**144000 次输出**均与各自机器 RocksDB 的有序 DocID / FP32 bits 一致，跨机文档映射和 Recall@100（0.7857886501265938）相同。跨平台共同命中文档分数最多相差 5 ULP；36/2000 条查询的 Top100 集合不同，替换项均在各自平台末位同分边界，不声称跨机逐位一致。未清 OS cache、未绑核，首遍不是冷盘结果。本轮未修改生产代码或新建 commit。完整延迟、RSS、首遍/后续遍与证据见 [du 双机检索对照](fts-custom-index-v1-du-analysis.md)，桌面证据目录为 `zvec-fts-v1/du-search-20261118-01/`。

### 2026-11-18 du-retrieval 通用优化落地

根据下节采样，保留三项局部优化：短位置列表不构造无用 header/checkpoint，定宽整数编码一次追加，批量 posting 按 128 行块读取 doclen。缓存仍为 8 B 临时行 / 4 MiB，RocksDB 链路、最终格式、评分和原有校验不变；未引入语料专用分支、全量 doclen 缓存或新增配置。

本机交错三轮构建中位数：du seal **6.698 → 5.122 s（−23.53%）**，总构建 **27.209 → 25.415 s（−6.59%）**；Quora seal **2.975 → 2.216 s（−25.50%）**，总构建 **11.937 → 11.060 s（−7.35%）**。最终索引大小完全不变。构建版与最终查询版源码快照分开保存，不将查询改动归因于 seal 收益。

du 全量 2000 查询三轮中位 QPS，mmap / buffer 8 MiB / buffer 64 MiB 分别 **2255.71 → 2435.64 / 1970.98 → 2124.73 / 2072.01 → 2213.19**；Quora 固定 1000 查询三轮分别 **298.33 → 328.61 / 269.60 → 293.64 / 273.24 → 296.94**。Quora 全量 10000 查询单轮 mmap **302.39 → 331.15 QPS**。本轮仅本机，未重测 191/RocksDB，不与历史绝对值拼接成新倍数。

**74000 次查询输出**的有序 DocID / FP32 bits 与既有参考完全一致，12 次构建的文档映射一致。先编译具体测试目标再直接运行绝对路径，format / field / reader / container / mutable snapshot 共 **84 项通过**，`fts_bench` 和 `_zvec` 已重建。新增缺失行校验用例确认批读取不绕过非匹配行的整块校验且错误不会变成 EOF。详细数据见 [du 分析与优化结果](fts-custom-index-v1-du-analysis.md)，证据在桌面 `zvec-fts-v1/du-opt-20261118-01/`。词典/小 term raw 缓冲尚未修改，生产门禁仍关闭，未创建新 commit。

### 2026-11-18 du-retrieval 热点诊断

使用当前 8 B / 4 MiB 二进制完成本机两次构建、三种查询路径的独立采样，生产代码未修改。du 100001 行的临时表仅 800008 bytes、196 页，writer 缓存加载只占 write_field 样本 0.18%；词表 448882 项，为 Quora 的 4.78 倍。导入 insert 样本中 RocksDB Write 63.25%、分词 25.59%；seal 中定宽整数编码路径 17.54%，位置 header/checkpoint 定宽字段构造占其分母 11.12%，raw posting 追加/回读路径合计约 21.9%。这些为条件化栈样本，不是预期提速，包含子调用的比例不能相加。

发现短位置列表也无条件构造不会写出的 header/checkpoint，以及 AppendU32/U64 的逐字节 push_back，作为首批局部优化候选。查询分词不是热点：custom ReadLength 占 search 样本 33.27% / 36.37%（mmap / buffer），词典 seek 约 18%，二分 fence 比较会完整物化只需首词的块。建议验证批量 doclen、词典首词视图；再基于 DF/工作量统计决定小 term raw 缓冲、稀疏窗口或 batch/WAND 通用选择，不按数据集调参。

五份有效采样、两次构建映射及 **150000 次诊断查询输出**已核验一致；采样计时不混入正式基准。首次仅停留在 dyld 的后台启动失败单独隔离，不计入分析。详细热点、代码位置、候选风险和验收顺序见 [du 优化诊断](fts-custom-index-v1-du-analysis.md)，证据位于桌面 `zvec-fts-v1/du-analysis-20261118-01/`。尚未测得候选实现的提速，未创建 commit。

### 2026-11-18 最终缓存取舍及当前 RocksDB/custom 对照

根据四组实验，最终采用 **8 B 临时行 + 4 MiB 默认写回缓存**，覆盖 524288 篇文档，与原先 16 B / 8 MiB 相同，同时缓存数据预算减半。更新缓存常量和说明，保持按需分配、4 KiB 页、校验与磁盘格式不变。没有数据集专用分支；这是默认内存预算的取舍，超过缓存容量的大表性能仍未单独量化。下节的“保留 8 MiB”是实验完成时的历史状态，由本节最终配置覆盖。

使用同机同版本二进制补齐 RocksDB/custom 新对照，两机 Quora 各 6 次构建、本机 du 6 次构建。三轮总构建中位数：Quora 本机 RocksDB **9.743 s** / custom **13.215 s**，191 **22.299 s / 26.140 s**；du 本机 **25.368 s / 30.489 s**。custom 索引分别缩小 **47.89% / 48.37% / 41.31%**；构建仍慢 35.64% / 17.23% / 20.19%，本机 ingest 有波动，分阶段及逐轮指标完整保留。

Quora 全量单轮 QPS（RocksDB / mmap / buffer 8 MiB / buffer 64 MiB）：本机 **161.44 / 287.73 / 254.85 / 264.09**，191 **66.78 / 133.12 / 72.24 / 110.82**。另有固定 1000 查询三轮；本机 64 MiB pool 波动明显，不据此宣称更大读取缓存稳定更快。du 完整 2000 查询三轮中位 QPS 为 **1936.93 / 2023.27 / 1911.68 / 2033.83**，差异较小，部分范围重叠。writer 的 4 MiB 缓存与 reader pool 是独立预算。

共 **128000 次查询输出**的有序 DocID / 原始 FP32 bits 与同机既有参考完全一致，召回未变；新产物的文档映射跨格式、轮次一致，Quora 跨机一致。本机重编译 fts_bench、field、mutable 测试及 `_zvec`，直接执行绝对路径测试通过 **27 + 4** 项；相同 4 MiB 行为上一轮已通过本机 116 / 191 118 项。191 复用该冻结二进制，1383 文件逐一核验只有缓存说明注释差异。修改范围 clang-format、`git diff --check` 通过，未创建 commit。

完整构建分项、大小、RSS、QPS、p50/p95/p99、范围及口径见 [当前对比报告](fts-custom-index-v1-benchmark.md)，[早期 v6 报告](fts-custom-index-v1-benchmark-v6.md) 已单独保留。证据入口：本机桌面 `zvec-fts-v1/current-vs-rocksdb-20261118-01/summary.json`；全部指标为新测，旧输出只用于正确性参照。未清 OS cache、未绑核，不宣称冷盘或独占机器表现，生产 writer 门禁仍关闭。

### 2026-11-18 临时行记录缩小到 8 字节：双机四组对照

只调整 field writer 的私有构建临时表：`uint32 length + uint32 present + uint64 accumulated_tf` 改为 `uint32 length + uint32 accumulated_tf`。缺失为 `(0, UINT32_MAX)`，存在但零 token 为 `(0, 0)`；完整的 `(UINT32_MAX, UINT32_MAX)` 仍表示存在且累计完成。累加前继续检查范围，结束时继续逐文档校验 TF 总和，全局 token 统计保持 uint64。RocksDB 键值编码、普通 mutable insert、最终 custom 文件编码和查询评分路径不变。

这是通用的状态压缩，不增加数据集分支或自动调参。4 KiB 页可容纳的记录从 256 增至 512，临时行表大小减半，同一缓存预算的文档覆盖数翻倍。生产实现保留既有 **8 MiB 上限**，仍按实际页数缩小、按需分配；4 MiB 仅为独立编译的实验配置，不因为 Quora 刚好能装入而调整默认值。缓存预算不等于整个 writer 或进程的内存上限。

四组均包含上一轮的 DOC_LEN 合并写入和关闭修复，只比较临时行大小与 writer 缓存上限。两机使用同一份 1383 文件源码清单，每组交错构建三轮；Quora 为 522931 文档、6032630 tokens，standard + lowercase、单线程，reader cache 固定 8 MiB。以下为各分项独立取中位数，单位秒：

| 机器 | 临时记录 | writer 缓存 | ingest/flush | seal/sync | 总构建 |
|---|---:|---:|---:|---:|---:|
| 本机 Apple M6 / macOS | 16 B | 4 MiB | 10.721 | 7.077 | 17.809 |
| 本机 | 8 B | 4 MiB | 9.415 | **3.182** | **12.589** |
| 本机 | 16 B | 8 MiB | 10.372 | 3.791 | 14.124 |
| 本机 | 8 B | 8 MiB | 9.568 | **3.121** | **12.698** |
| 191 Xeon Platinum 8369B / Linux | 16 B | 4 MiB | 21.059 | 7.805 | 28.943 |
| 191 | 8 B | 4 MiB | 20.991 | **5.236** | **26.243** |
| 191 | 16 B | 8 MiB | 21.132 | 5.434 | 26.589 |
| 191 | 8 B | 8 MiB | 21.095 | **5.292** | **26.420** |

- 固定 4 MiB，16 B → 8 B 的 seal 中位数本机减少 **55.04%**、191 减少 **32.92%**；总时间分别减少 29.31% / 9.33%。Quora 临时表从 7.98 MiB 降至 3.99 MiB，跨过缓存容量门槛，是这组较大收益的重要条件。
- 固定 8 MiB，两种记录均可完整驻留，seal 本机减少 **17.69%**、191 减少 **2.61%**，总时间分别减少 10.10% / 0.63%。本机未修改的 ingest 中位数也下降 7.75%–12.19%，不能把总时间差全部归因于本次改动；191 ingest 差异不足 0.4%。8 B 在 4 MiB / 8 MiB 下的差异很小，不能据此推导其他规模应选 4 MiB。
- 正式计时没有与同机本任务编译、测试或采样重叠；未绑核、未清 OS cache，不宣称独占机器、冷盘或统计显著性。全部逐轮指标与范围保留在证据中。

额外使用本机 **du-retrieval 长文档语料**作对照：100001 文档、21303021 tokens，jieba + lowercase，两种临时表均小于 4 MiB。固定 writer 缓存 4 MiB，每版交错构建三轮：

| 临时记录 | ingest/flush 秒 | seal/sync 秒 | 总构建秒 |
|---|---:|---:|---:|
| 16 B | 23.312 | 7.190 | 30.590 |
| 8 B | 23.226 | 6.876 | 30.044 |

此对照观察到 seal 减少 **4.35%**、总时间减少 **1.78%**，收益较小，不能将 Quora 的大幅提速推广到所有数据。临时表空间减半和覆盖文档数翻倍是确定的结构收益；吞吐收益依赖工作集、访问分布和机器。超过缓存容量的性能收益尚未单独量化，下面的大表用例验证的是正确性。

- 两机共 24 次 Quora 构建均为 **64500320 bytes**，文档 ID 映射跨配置、轮次和机器一致；6 次 du 构建均为 **175364320 bytes**，文档映射一致。Quora 输入指纹跨机一致。
- Quora 每机每配置运行 mmap 全量 10000 查询、buffer pool 固定 1000 查询，共 **88000** 行有序 DocID / 原始 FP32 bits 与各自机器的既有参考完全一致。全量 Recall@100 保持本机 0.9480075398652996、191 0.9480504565319663。du 两版分别运行 mmap / buffer pool 全量 2000 查询，共 **8000** 行逐位一致，Recall@100 同为 0.7857886501265927；全部全量查询均有结果。查询沿用持久化的分词配置。本轮查询用于一致性验收，未重做多轮查询吞吐比较。
- 两种缓存配置分别通过本机 **116** 项常规测试、191 **118** 项测试，共 232 / 236 次执行：field 27、mutable snapshot 4、container 27、generation 14、publication 本机 44 / Linux 46。Linux 每配置包含 2 项进程退出恢复测试；本机沿用此前的 re-exec 限制，未将排除的 death tests 计为通过。`_zvec` 重建成功。均先用 Ninja 编译目标，再直接执行绝对路径测试二进制。
- 新增边界覆盖缺失/零 token、完整 UINT32_MAX 长度、累加溢出前拒绝、超过 uint32 的全局 token 总数及 source 失败。缓存回归扩大至 **2097153 行**，8 B 表超过 16 MiB，明确覆盖 4 MiB / 8 MiB 两种配置的冲突淘汰、重读、尾页和 finish 校验。Standards / Spec 增量复核均为 0 项问题；修改范围 clang-format 和 `git diff --check` 通过。

证据入口为本机桌面 `zvec-fts-v1/build-row8-20261118-01/summary.json`，`analyze.py` 可重新校验、汇总；`remote191/results/` 保存 191 正式输出。16 B / 8 MiB 复用上一轮冻结二进制，其实际源码与本轮基线只有 `collection.cc` 的已核验纯格式差异，其余三版分别编译并记录指纹。191 初次源码切换保留旧时间戳造成一组未重编译，该尝试已中止并隔离至远端 `aborted-mtime-check/`，不计入结果；正式轮次使用更新时间戳的切换流程重新编译并核验四个不同二进制。未创建 commit。

后续通用优化优先看占比更高的 mutable ingest，例如有界批内合并 bitmap / MAX_TF、减少 memtable 记录；需要独立验证批次可见性、WAL/checkpoint 和异常语义。此次不引入该写入路径变更，也不继续按 Quora 大小扩大或缩小默认缓存。

### 2026-11-18 继续构建优化与关闭、向量候选异常修复

在已验收的 4 MiB 版本上继续优化，两机使用同一份 1383 文件冻结源码。磁盘格式、BM25、普通 RocksDB insert 和生产 writer 门禁不变。

- 文档统计写回缓存从最多 1024 页增加到 **2048 个 4 KiB 页（8 MiB 数据上限）**，槽数仍按实际文档页数缩小、页面按需分配。Quora 的 7.98 MiB 临时文档表可以完整容纳；另有槽元数据及单页加载空间，不能将此预算解释为整个 writer 或进程的 RSS 上限。
- DOC_LEN 的 native-endian 行号仍由完整 snapshot 读取，但改为每 **65536** 条排序后合并相邻的五字节记录写入。最多 512 KiB 记录及不足 64 KiB 写缓冲，在创建 field writer 前释放；保留 presence、精确 uint32 长度、行域与迭代/I/O 错误检查，不建立随行数增长的数组。跨多个批次、间隙与尾批的双模式回归通过。
- 新增关闭回归先复现无外部 reader 时 **8 次 CF/default flush 变成 16 次**、底层 close 失败却返回成功。`FtsIndexer::close()` 释放自身列引用后，无外部 owner 时显式关闭并返回错误，保留失败 context 允许重试；有外部 reader/snapshot 时只 flush 并延迟关闭。后者仍必须在最终释放时 flush，因为保留的 mutable column 可能继续写入。已关闭/未成功打开的 context 不再被 deleter 重复关闭。
- 新增双模式用例先复现同 ID 向量重建的明确提交失败遗留新文件。create/optimize 任务现在持有新 raw/quant 文件候选 owner，预打开 reader 共享 pin；明确失败在最后 owner 释放后清理，成功或不确定提交后 preserve。单个 merge 在 open/merge/flush 中途失败时也清理私有输出。planner 同时覆盖首次 custom 提交。回归验证失败重试、旧文件存活、候选 reader 延迟清理及不确定提交恢复。

Quora 522931 文档、6032630 tokens，standard + lowercase、单线程、reader cache 8 MiB。每机交替执行冻结 4 MiB 版和本轮优化版各三次，每轮新建索引；下表分项分别取中位数。

| 机器 | ingest/flush 秒（旧 → 新） | seal/sync 秒（旧 → 新） | 总构建秒（旧 → 新） | 总耗时减少 |
|---|---:|---:|---:|---:|
| 本机 Apple M6 / macOS | 9.165 → 9.037 | **6.897 → 3.064** | **16.065 → 12.161** | **24.30%** |
| 191 Xeon Platinum 8369B / Linux | 20.828 → 20.858 | **8.367 → 5.401** | **29.215 → 26.283** | **10.04%** |

seal/sync 分别减少 **55.57% / 35.45%**。本机总时间三轮为旧 16.065 / 16.250 / 15.923 秒、新 12.187 / 12.161 / 12.066 秒；191 为旧 29.215 / 29.267 / 29.194 秒、新 26.316 / 26.283 / 26.070 秒。这是扩大缓存、doclen 合并写入与关闭修复的组合对照，未单独量化各项贡献；ingest 未修改，其波动不能归因于这些优化。

- 12 次正式构建最终索引均为 **64500320 bytes**，所有文档 ID 映射跨轮次、跨机一致；输入与源码指纹跨机一致。正式计时没有与同机本任务编译、测试或采样重叠；未绑核、未清 OS cache，不宣称冷盘或独占机器结果。
- 两机各运行新索引 mmap 全量 10000 查询、8 MiB buffer 固定 1000 查询，共 **22000** 行完整有序 DocID / 原始 FP32 bits 与各自 4 MiB 参考完全一致。全量 Recall@100 仍为本机 0.9480075398652996、191 0.9480504565319663。没有重新进行读取模式的多轮吞吐对照。
- 本机 **442 项常规测试**通过：field 25、mutable snapshot 4、container 27、generation 14、column indexer 96、segment publication 44、segment 108、helper 17、collection 107；最终 planner 分支补齐后再跑相邻 vector/mixed 8 项通过，`_zvec` 重建成功。191 **216 项**通过，包含 column indexer 100、publication 46 中的 **6 项实际退出恢复测试**。本机此前的 death-test re-exec 限制未重试，未计为通过。Ninja 只编译，测试均直接执行绝对路径二进制。
- Standards / Spec 两轴增量复核均为 0 项新增问题；改动范围 clang-format 与 `git diff --check` 通过。测试完成后仅对 `collection.cc` 一处表达式做格式调整，证据中保留实际编译的源码快照并验证格式化后与工作区一致；未因此重跑性能实验。
- 范围边界：关闭修复恢复的是**显式 FtsIndexer::close 返回语义**，不是 Collection/Segment 析构链的端到端错误传播；此次向量候选清理覆盖 create/optimize，既有 drop_index 新 Flat 候选失败残留仍未覆盖。WAL 恢复的 DOC_LEN 统计重建保留，用于防止重放重复计数。

完整证据保存在双机独立桌面 `zvec-fts-v1/build-next-20261118-01/`。本机 `summary.json` 为统一入口，`remote191/` 保存回传证据；本机 `baseline-1` / `candidate-1` 为早期探索轮次，不计入正式三轮。未创建 commit。接下来构建成本主要仍在 mutable ingest；批内合并 bitmap/MAX_TF 或 FTS 专用 memtable 配置仍需独立实验，尚无提速承诺。

### 2026-11-18 4 MiB 写回缓存实验及后续方向

按用户要求测试扩大构建写回缓存。容量和写回粒度分开：从一个 4 KiB 块改为最多 **1024 个 4 KiB 页**，有效缓存数据上限 **4 MiB**，另有有界槽元数据和单页加载临时空间；这不是整个 writer 或进程的 RSS 上限。页按需分配，小字段按实际页数减少槽数，直接映射定位，冲突时先写回脏页，finish 写回全部剩余脏页后再校验累计 TF。没有把每次读写粒度放大到 4 MiB。

在本机和 191 冻结、核对同一份 1383 文件源码；以 `build-opt-20261118-01/` 的最终 4 KiB 版本为基线。Quora 522931 文档、6032630 tokens、standard + lowercase、单线程，reader 的 `index_cache_mb=8` 保持一致，与此次 writer 写回缓存是不同预算。每机交替执行两版本各三轮，每轮重新建索引；下表各列分别取中位数。

| 机器 | 构建总秒（4 KiB → 4 MiB） | 总耗时减少 | seal/sync 秒（4 KiB → 4 MiB） | seal 耗时减少 |
|---|---:|---:|---:|---:|
| 本机 | **18.382 → 16.318** | **11.23%** | **9.106 → 7.021** | **22.90%** |
| 191 | **30.220 → 28.929** | **4.27%** | **9.105 → 8.322** | **8.60%** |

- 本机总时间三轮为 4 KiB：18.382 / 19.345 / 18.348 秒，4 MiB：16.318 / 18.244 / 16.262 秒；191 为 4 KiB：29.689 / 30.505 / 30.220 秒，4 MiB：29.435 / 28.929 / 28.928 秒。
- ingest/flush 中位数本机 9.266 → 9.301 秒、191 21.092 → 20.626 秒。该路径没有修改，因此总时间差不能全部归因于缓存，尤其 191 的收益有限。RocksDB 中间 SST 体积和后台工作仍有轮次差异；正式构建没有与本任务编译、测试或采样重叠，未绑核、未清 OS cache，不宣称冷盘或独占机器结果。
- 12 次正式构建最终体积均为 **64500320 bytes**，全部文档 ID 映射跨轮次、跨机一致，语料/查询/qrels 哈希跨机一致。两机分别对第三轮新索引运行 mmap 全量 10000 查询、8 MiB buffer pool 固定 1000 查询；共 **22000** 行有序 DocID / 原始 FP32 bits 与同机 4 KiB 参考一致。全量 Recall@100 本机 **0.9480075398652996**、191 **0.9480504565319663**，均未改变。
- 两机各 **111** 项常规测试通过，本机 `_zvec` 重建成功。字段回归扩展到 524289 行，覆盖多个页竞争同一槽、淘汰后重读、单行尾页、未淘汰脏页在 finish 时落回，以及累计不足/超额；零行与最大行号边界经静态检查。Ninja 仅编译，测试均以绝对路径执行。修改范围 clang-format、`git diff --check` 和 Standards / Spec 两轴复核通过。本轮未重跑此前已记录的 Mac death-test re-exec 问题。
- 保留 4 MiB 实现，未创建 commit。双机桌面独立目录为 `zvec-fts-v1/build-cache-4m-20261118-01/`；本机 `summary.json` 汇总正式指标、源码/输入身份、测试和逐位差分，原始输出及采样一并保留。失败的诊断启动不计入正式轮次。

后续候选按实施成本和收益来源区分，以下均**尚未实现或测得提速**：

1. **8 MiB 容量对照，验证成本最低。** Quora 当前临时文档统计表为 `522931 × 16 = 8366896 bytes`，约 7.98 MiB；8 MiB 的页缓存能容纳这份表，4 MiB 仍会发生容量不足和直接映射冲突。只对该数据规模有此推导，不能承诺其他数据集同样全命中。
2. **doclen 暂存的分页聚合。** `ReadDocuments` 因 legacy native-endian 行号顺序仍逐行随机覆盖 5 字节暂存记录，采样能看到对应 fseek/flush 开销。该阶段可应用有界页缓存或分批重排；Quora 这份暂存仅约 2.49 MiB。需要保留 presence、完整 uint32 长度与源错误检查。
3. **缩小临时行状态。** 当前 16 字节包含 uint64 累计 TF，但每次更新已校验累计量不超过 uint32 doclen；可验证将累计量改为 uint32、把记录缩到 12 字节，降低暂存 I/O 和缓存占用。必须同步调整行/页边界并保留溢出检查；这属于私有临时格式优化。
4. **批量导入减少 mutable 写入记录，潜在收益来源更大。** 当前每文档已只有一次跨 CF WriteBatch；采样中 `DBImpl::WriteImpl` 约占已采集 `FtsColumnIndexer::insert` 栈的 80%，主要落在 `WriteBatchInternal::InsertInto` 和 memtable 插入/比较。分词约占该采样 insert 栈的 3–4%，不是当前主要热点。应测试在有界批内按词合并 bitmap 和 maxTF，减少 memtable 项数，以及 FTS 专用 memtable 配置对照；单纯拼接更大的 WriteBatch 仍保留逐项插入成本。批量路径必须保持完成可见性、统计与 checkpoint 契约，不能直接删掉普通 mutable 查询需要的辅助 CF。

采样与正式计时分开。4 MiB 诊断中 `load_document_block` 仍占所采集 `write_field` 栈约一半，说明容量实验仍有依据；这些采样占比用于定位，不作为各候选方案的预期提速百分比。

### 2026-11-18 索引构建优化：双机三轮验收

在现有 hardening 工作区上继续优化封存阶段，保留生产 writer 默认关闭、磁盘格式、BM25 评分和普通打开/查询的有限校验契约。

- 采样确认主要开销来自逐 posting 读取、更新文档统计临时文件时的 stdio 定位和刷新。`FtsFieldWriter` 增加一个固定 **4 KiB** 可写块，淘汰及最终完成前写回；文档存在性、精确长度和累计 TF 检查仍完整执行。新增内存缓冲不随文档数增长，写回失败保持错误并阻止候选完成。
- 第一处优化后，剩余热点包括逐 posting 的 RocksDB positions Get。转换改为复用同一 snapshot 的 positions iterator，通常顺序推进；完整 key 不匹配时重新 Seek，保留含 NUL 词项的交错/回退语义。缺失 positions、迭代错误、位置数量及值域错误仍明确失败，单个旧格式 value 的原有大小限制保留。
- 新增跨块、跨词重访、单行尾块、累计 TF 不足/超额，以及中间/末尾缺失 positions 后的固定快照与重新捕获重试测试。没有新增通用缓存框架或修改 RocksDB 写入配置。

使用上一轮冻结 hardening 二进制作旧版基线，冻结并跨机核对当前 **1383** 份一方源码。Quora 522931 文档、6032630 tokens，standard + lowercase、单线程、8 MiB cache；两机各自交替执行旧版/优化版各三轮，每轮新建索引。以下各时间列分别取中位数，分项中位数之和不必等于总耗时中位数。

| 机器 | ingest/flush 秒（旧 → 新） | seal/sync 秒（旧 → 新） | 构建总秒（旧 → 新） | 总耗时减少 |
|---|---:|---:|---:|---:|
| 本机 Apple M6 / macOS 27 | 10.084 → 11.505 | **27.860 → 9.794** | **37.956 → 21.359** | **43.73%** |
| 191 Xeon Platinum 8369B / Linux | 20.927 → 21.048 | **28.078 → 9.120** | **49.120 → 30.140** | **38.64%** |

seal/sync 分别减少 **64.85% / 67.52%**。本机总时间三轮为旧版 37.956 / 39.433 / 36.159 秒、新版 21.359 / 22.234 / 18.433 秒；191 为旧版 48.785 / 49.544 / 49.120 秒、新版 30.360 / 30.140 / 30.074 秒。ingest 路径未改，本机 ingest 与 RocksDB 后台工作有波动；优化收益集中在封存阶段。

- 全部 12 次正式构建最终索引均为 **64500320 bytes**，文档编号映射跨轮次、跨机一致；语料、全量查询、固定查询和 qrels 哈希跨机一致。正式构建未与本任务的编译、测试、采样重叠，未绑核、未清空 OS 页缓存，不宣称冷盘或独占机器结果。
- 每台机器用第三轮新索引执行 mmap 全量 **10000** 查询、8 MiB buffer pool 固定 **1000** 查询。共 **22000** 行结果与各自同机 hardening 参考的完整有序 DocID / 原始 FP32 bits 一致；全量 Recall@100 本机 **0.9480075398652996**、191 **0.9480504565319663**，固定查询均为 **0.9595364078645813**。本轮查询用于验证构建产物，没有重新做所有读取模式的多轮性能比较。
- 两机各 **111** 项常规测试通过：mutable snapshot 3、field 25、container 27、generation 14、segment publication 42；均先由 Ninja 编译，再用测试二进制绝对路径执行。本机 `_zvec` 重建成功，修改范围 clang-format 和 `git diff --check` 通过。Standards / Spec 静态复核分别为 0 项发现。本轮未重跑前文记录的 Mac death-test re-exec 问题。
- 本机和 191 的独立桌面目录均为 `zvec-fts-v1/build-opt-20261118-01/`；本机 `summary.json` 汇总全部指标、测试及差分证据，`results/commands.json` 与 `remote191/results/commands.json` 保存正式命令，原始输出、源码及二进制身份一并保留。未创建新 commit。

### 2026-11-18 hardening 版本 Quora 双机复测

使用 `5d49888` 后的当前 hardening 工作区，冻结并在两机核对 1383 份一方源码。191 在独立目录以 GCC 12.3 Release 重新编译；本机使用当前源码对应的 Apple Clang 21 Release 产物。两端重新构建 Quora custom 索引，再用同一新索引切换 mmap / buffer。语料、查询、qrels、固定 1000 查询输入及新文档 ID 映射均跨机一致。此次测试没有修改实现。

构建为单轮、单线程，cache 配置 8 MiB；522931 文档、6032630 tokens，standard + lowercase。

| 机器 | ingest/flush 秒 | seal/sync 秒 | 构建总秒 | 最终索引 bytes |
|---|---:|---:|---:|---:|
| 本机 Apple M6 / macOS 27 | 10.934 | 28.759 | **39.701** | 64500320 |
| 191 Xeon Platinum 8369B / Linux | 20.806 | 28.222 | **49.051** | 64500320 |

检索为索引级 OR Top100、单线程。全量 10000 查询每模式一轮，固定 1000 查询每模式同进程三轮；下表延迟为全量轮次，最后一列为固定查询三轮 QPS 中位数。

| 机器 | 模式 / pool 预算 | 全量 QPS | p50 ms | p95 ms | p99 ms | 固定查询中位 QPS |
|---|---|---:|---:|---:|---:|---:|
| 本机 | mmap | **293.589** | 3.220 | 6.126 | 8.027 | 298.935 |
| 本机 | buffer 8 MiB | 163.834 | 5.664 | 11.001 | 14.947 | 164.843 |
| 本机 | buffer 64 MiB（工具默认） | 266.012 | 3.541 | 6.745 | 9.088 | 272.512 |
| 191 | mmap | **131.696** | 7.191 | 13.349 | 17.543 | 131.082 |
| 191 | buffer 8 MiB | 71.863 | 13.014 | 25.132 | 33.814 | 71.968 |
| 191 | buffer 64 MiB（工具默认） | 110.485 | 8.562 | 15.959 | 21.289 | 110.744 |

- 两机均为 mmap 最快；buffer 从 8 MiB 调到 64 MiB，全量 QPS 分别增加 62.37% / 53.74%。这些是本轮容量对照，不把 pool 预算解释为整个进程或 OS 页缓存上限。
- 24 份原始输出、78000 查询行全部与各自同机旧参考核对完整有序 DocID 和原始 FP32 score bits，一致。全量 Recall@100：本机 0.948007539865、191 0.948050456532；跨平台浮点/严格同分边界差异沿用 opt6 记录，不声称跨机分数位相同。固定 1000 查询的 Recall@100 均为 0.959536407865。
- 191 直接运行 container 27、generation 14、常规 publication 42，共 83 项通过。Ninja 仅编译。两机测量均在本机内部串行，计时不含打开、输入加载和结果序列化；没有并行的本轮编译或其他本轮性能任务。未绑核、未清空 OS 页缓存，因此首轮不代表冷盘。系统负载和环境另有记录。
- `buffer_pool_used_bytes=0` 是 indexer 析构后的统计，不能推断运行期或峰值占用为零。源码、二进制与输入的前后核对通过。当前生产 writer 门禁仍关闭；本次使用显式 index benchmark 接口。
- 完整参数、单轮输出、三轮结果、分阶段时间、RSS、指纹和独立结果比较保存在本机 `/Users/jiliang.ljl/Desktop/zvec-fts-v1/quora-hardening-20261118-02/`，统一入口 `summary.json`。191 原始证据已回传其 `remote191/`；远端目录为 `/home/jiliang.ljl/Desktop/zvec-fts-v1/quora-hardening-20261118-02/`。原始实验全部保留。

### 2026-11-18 复审后的发布、回收与合并预算修复

- 按用户要求先创建 `codex/fts-v1-hardening`，以 `5d49888` 保存 opt6 实现和 FTS 文档基线。以下修复为其后的工作区改动；生产 custom writer 门禁仍关闭。
- optimize 根据任务的实际输出 descriptor 选择提交协议，补齐普通 compaction 首次生成 custom base、没有独立 rewrite task 的分支。真实混合 compaction 用例在旧判断下双模式失败，修复后通过。
- 同 ID 向量替换只在提交后标记旧文件回收。Segment 持有文件 owner，重开的独立 reader 共享 owner；准备后放弃不会删除旧文件，最后一个旧 reader 释放后才删除，未变化的向量和 FTS 文件继续可读。新增双模式用例先复现旧文件永久残留，再验证上述生命周期。
- Legacy source 复用已有校验扫描，计算最大词项的保留工作区，避免每个很小的旧段固定占用最大允许额度。移除 64 输入硬限制；planner 和 reducer 共用 128 MiB 预算及每输入游标预留。planner 根据格式的保守上界提前分组，reducer 使用打开 source 的上界再次检查。覆盖 8 个混合小源、65 个 custom 源成功和单次 129 源明确拒绝；实际 collection 的 129 个 custom 小段 optimize 原先因预算失败，修复后拆组并在 buffer/mmap 下完成查询与重开验证。没有新增多阶段临时索引。
- 合并词典内的重复 Spool 与 `FtsSpool`，连续同方向读写复用当前位置，读写方向切换仍执行定位屏障，范围检查和粘滞 I/O 错误保留。新增跨缓冲边界、连续追加、覆盖和读写切换测试；不改变磁盘格式、评分或查询执行路径。
- 本机同语料、单线程、8 MiB cache 的单轮 Quora 构建对照：522931 行、6032630 tokens，总耗时 **69.342 → 43.169 秒（减少 37.744%）**，seal/sync **57.336 → 29.329 秒**；ingest/flush 为 11.995 → 13.830 秒。最终体积均为 **64500320 bytes**。新产物的 mmap 全量 10000 查询、buffer 固定 1000 查询均与冻结 opt6 的完整有序 DocID + 原始 FP32 bits 逐字节相同。此次 mmap 全量单轮 280.133 QPS，仅作为新产物读取与一致性记录，不替代 opt6 多轮查询吞吐结论。
- Ninja 仅编译，测试均用产物绝对路径直接执行。349 项常规测试通过：container 27、field 24、reader 3、generation 14、segment publication 42、version publication 7、segment 108、segment helper 17、collection 107。最后 planner 修改后再次通过 generation 14、publication 42 和 collection 相邻回归 2 项；`_zvec` 重建成功，改动范围 clang-format 和 `git diff --check` 通过。Standards / Spec 两轴复审无新增阻断项。
- 两项 checkpoint crash death tests 的 re-exec 子进程收到 SIGKILL，独立重试仍失败，未计为通过。批量启动失败和共享 fixture 干扰的记录保留，最终常规通过证据来自直接启动及对共用路径的旧测试隔离工作目录。所有本轮输出位于桌面 `zvec-fts-v1/hardening-20261118-01/`，汇总为 `validation-summary.json`。
- 仍需单独处理的既有异常路径：同 ID 向量重建在提交前失败时，新候选向量文件可能残留；本轮的旧文件 retirement 不等于已接入新候选所有权或恢复清理。Windows 完整持久 checkpoint 等跨平台验收边界继续沿用下文记录。

### 2026-11-18 mmap 查询优化目标完成（opt6）

- 完成有界 posting batch、4096 DocID 窗口 OR 执行、精确 BM25 批量评分、doclen 热路径及竞争分扫描优化；保留原 AST 累加顺序、严格 Top-K、定位语义和错误传播。仅适用于没有过滤器/候选集的平坦正向多词 OR；其他路径保持原执行器。无磁盘编码、统计、正常打开校验或生产 writer 门禁变更。
- Mac 最终全量 10000 查询：mmap **302.395 QPS**，RocksDB **176.015 QPS**，吞吐提高 **71.80%**。p50/p95/p99 为 **3.124/5.889/7.774 ms**，对照 **3.933/16.521/29.261 ms**。固定 1000 查询三轮中位数 **299.403 / 169.787 QPS**。
- 按用户要求由 subagent 在 191 复测同一份 84 文件冻结 opt6。全量 mmap **131.028 QPS**，RocksDB **67.660 QPS**，吞吐提高约 **93.66%**；p50/p95/p99 **7.232/13.413/17.545 ms**，对照 **9.729/44.108/79.041 ms**。固定 1000 查询三轮中位数 **132.087 / 65.902 QPS**。
- 两机所有正式轮次均与各自旧版 RocksDB 参考的完整有序 DocID/原始 FP32 位一致；Mac Recall@100 **0.9480075399**，191 **0.9480504565**。跨平台细微差异沿用此前浮点及同分边界说明。
- Mac 最终 field 24、reader 3 直接运行通过，`_zvec` 构建成功；相关 format/conjunction 在本轮此前步骤也通过。191 最终 field 24、reader 3、conjunction 10、format 22、bitpacked 124，共 **183 项通过**。修改范围格式检查及两轴静态审查无确认问题。测试和 benchmark 直接运行，Ninja 只编译。
- 正式计时只比较 mmap 和 RocksDB，没有 buffer pool 性能轮次，没有并行构建/采样，未宣称冷盘或专属机器测试。两机复用旧索引，本轮没有新的构建时间或体积结论；custom 构建较慢的问题未优化。生产 writer 仍默认关闭，此处完成的是 Quora 检索性能目标。
- [实现、性能及验证报告](fts-custom-index-v1-batch-optimization.md)。本机桌面 `mmap-opt4/5/6-20261118/`、`remote191-opt4/6-20261118/` 保存阶段记录和最终证据；未创建 commit。

### 2026-11-18 191 Quora 初始 opt2 对照完成

- 按用户要求由 subagent 在 191 独立快照构建并测试冻结 opt2：80 个源码文件及 Quora 原始数据哈希一致；GCC 12.3 Release，直接运行 format 22、field 20、reader 2、bitpacked 124，共 168 项通过。未混入同期 Mac opt3 改动。
- 全量 10000 查询：mmap 39.284 QPS、RocksDB 67.111 QPS；mmap 吞吐低 41.47%，耗时约 1.708 倍。固定 1000 查询三轮中位数分别 38.816 / 65.596 QPS，差距稳定。仅比较 mmap 与 RocksDB，没有 buffer pool 性能对照。
- 两后端全量和所有热态轮次的有序 DocID / 原始 FP32 位完全一致，Recall@100 同为 0.9480504565。Mac 与 191 的文档 ID 映射相同；跨平台共有文档分数差最大 7 ULP，Top100 集合差异均在各自严格同分截断边界。具体浮点求值/标准库原因未隔离，不将跨平台差异混为同机后端不一致。
- 本轮新建：custom 58.467 秒、64,500,320 字节；RocksDB 21.915 秒、124,914,604 字节。custom 缩小 48.36%，构建耗时约 2.668 倍，额外耗时主要落在 seal + sync 阶段。
- [191 初始 opt2 报告](fts-custom-index-v1-191-quora.md)；本机和远端桌面 `zvec-fts-v1/remote191-opt2-20261118/` 保存全部命令、快照、原始输出与差分证据。未绑核、未清空 OS cache，未宣称冷态或隔离机器性能。这是优化目标尚未完成时的历史结果；后续 opt6 已在两机超过 RocksDB，见本页顶部。

### 2026-11-18 第三轮 Mac mmap 推进优化

- `advance` 在已解码块中先检查当前位置，再从近处指数探测和二分，减少 WAND 高频短距离定位成本；不改变跨块校验、评分或上界策略。临时扫描原型已移出生产源码。
- 字段 20、reader 2 项直接运行通过，`fts_bench` / `_zvec` 构建成功，改动范围格式与 whitespace 检查通过。11 份正式结果、29,000 行查询与 v6 的有序 DocID / 原始 FP32 位完全一致。
- Quora 固定 1000 查询三轮中位数：opt2 同轮基线 103.081 QPS，本轮 mmap 114.204 QPS（+10.79%），RocksDB 168.425 QPS。全量 10000：mmap 116.687、RocksDB 174.020 QPS；mmap 仍只有 RocksDB 吞吐的 67.05%，优化目标尚未完成。
- 详见 [第三轮 Mac 优化](fts-custom-index-v1-mmap-opt3.md)。191 的密钥连接已恢复；同期 subagent 已完成冻结 opt2 对照，远端结果不包含本轮块内推进改动。

### 2026-11-18 第二轮 Mac mmap 优化

- 落实上一轮两个主要格式热点：skip decoder 共用无分配字段校验，去除重新编码字符串，并复用已验证的当前块记录；doclen 改为持有原生范围的编码视图，完整检查块边界、presence 和缺字段约束后按行读取，避免展开、复制 128 项数组。评分上界、磁盘格式及门禁未变。
- 新视图用例先红后绿；format 22、field 20、reader 2，共 44 项直接执行通过，`fts_bench` / `_zvec` 构建通过，覆盖非对齐、全 presence、稀疏 / 尾块、异常数据和跨页游标生命周期。改动范围格式检查与 whitespace 检查通过。
- 仅跑 Quora mmap / RocksDB。三轮固定 1000 查询中位数：旧 mmap 同轮复测 84.962 QPS，新 mmap 103.816 QPS（+22.19%），RocksDB 171.015 QPS。全部 10000 查询：mmap 104.821 QPS、RocksDB 176.438 QPS；有序 DocID / FP32 位值完全一致，Recall@100 同为 0.9480075399。原 100 查询性能诊断比值 1.619，通过 ≤2.0 门槛。
- 独立采样：skip 解码从 10.2% 降至 0.4%，重新编码及整块 doclen 解码不再出现，doclen 读取从 14.3% 降至 7.2%。仍有宽松 term 上界导致的候选成本，全量 mmap 吞吐仅为 RocksDB 的 59.41%，没有宣称追平。未重测构建时间 / 大小；191 复查仍认证失败。
- 详见 [第二轮优化结果](fts-custom-index-v1-mmap-opt2.md)。全部工具、源码快照、正式结果及采样在桌面 `zvec-fts-v1/mmap-opt2-20261118/`；未创建 commit。

### 2026-11-18 Quora 查询路径差距诊断

- 后续性能范围按用户最新要求固定为 **Quora、custom mmap 与 RocksDB**，不再跑 buffer pool 性能对照；历史三模式结果保留。
- 原版成对采样与临时计数消融确认：1000 查询 custom 的候选数是 RocksDB 的 2.619 倍、term 评分 2.432 倍、doclen 块解码 6.116 倍，但 posting 块装入仅 1.018 倍。custom skip 解码还调用编码器重新生成字符串，发生约 1087 万次多余编码。
- 三轮诊断耗时中位数：RocksDB 5.846 秒、mmap 11.724 秒；仅收紧 term 上界的实验降至 8.856 秒。启用现有持久化 block max 反而升至 13.453 秒：成功跳块仅 325 次，skip 解析扩大到 4178 万次。上述实验开关没有带入生产。
- 22 份结果与冻结参考的有序 DocID / FP32 位值完全一致；8 个临时插桩文件逐字恢复，80 个实现/工具源码与上一轮冻结清单一致。正常版 `fts_bench` / `_zvec` 重新构建成功，直接运行 format 20、field 19、reader 2，共 41 项通过。本轮交付为归因报告，未修改生产代码或磁盘格式。详见 [查询路径诊断](fts-custom-index-v1-query-path-analysis.md)；工具及证据全部保存在桌面的 `zvec-fts-v1/quora-path-analysis-20261118/`。

### 2026-11-18 Mac mmap 优化与校验策略调整

- 用户明确改为参考向量索引“仅在打开时做有限校验”，已同步主规范 5.1/6.5/7.1/11.1。核对 `MMapFileReadStorage`、`BufferReadStorage` 默认 `checksum_validation=false` 及 `IndexUnpacker` 头尾/目录检查后，FTS 普通 open/预发布保留有界元数据、身份和必要目录检查，取消自动全文件/字段扫描；普通 read 直接持有 native MemoryBlock，不做数据页 SHA、不维护校验副本缓存。`verify_all()` 和字段 `verify()` 保留为显式诊断。磁盘编码、摘要、持久化同步和 generation 所有权不变；普通查询不再保证检测全部数据位翻转。
- 新校验契约的 3 项测试在旧实现直接执行全部失败（读取仍计算哈希、分配校验副本），修改后通过。容器 26、字段 19、reader 2、generation 13、普通 segment publication 40，共 100 项分批直接运行通过；本轮未重试已知受 AMFI 影响的 death re-exec。`_zvec` 正式构建通过。
- 运行时保守 BM25 饱和上界恢复普通重开 reader 的 WAND；不用未验证的持久化 max。严格 FP32 最近舍入的四个误差因子及溢出防护有说明与极值测试，fast-math/扩展精度/非最近舍入/非单位 boost 退化为无穷。差分参考改为无剪枝取全部候选，再按 DocID 重放原 score-only TopK，保持同分顺序。
- profile 发现 WAND 的临近 `advance` 原先在完整剩余 skip 列表二分，触碰远处页；测试先复现额外 4 KiB 校验，改为下一块起指数探测和局部二分后通过。移除页校验后，该测试改以真实范围读取次数约束，仍覆盖近跳、远跳和 EOF。
- 初始样本 Quora 固定 100 查询：v6 mmap 3572.138 ms，单独加宽松运行时 WAND 6818 ms，加入临近跳转 3229.787 ms；按用户要求取消逐页校验后 1463.228 ms。进一步 profile 指向 doclen 通用整数解码，改为验证整段后小端批量复制、大端逐值解码；最终 1203.873 ms，结果始终与 RocksDB 原始 FP32 位值完全一致。RocksDB 同样本 585.450 ms；诊断目标比值 ≤2.0 尚未达到（最终 2.056），不将其误报为性能门槛通过。
- doclen 优化后重新正式构建并直接运行格式 20、字段 19、reader 2，共 41 项通过；覆盖 1/7/8/9/127/128、非对齐、截断、尾随字节和缺字段非零长度。两轴只读复核未发现阻断问题；clang-format 和 diff whitespace 检查通过。没有修改第三方、公共 SDK 或编译选项，没有 commit。
- 用户随后要求性能先只跑 Quora；已 SIGINT 停止 DU custom 构建（退出 130），DU 输出不计入本轮报告。Quora 已重新构建并完成 10,000 条全量查询和固定 1,000 条三轮热态对照，全部 DocID/顺序/FP32 位值与 RocksDB 及 v6 基线一致，Recall@100=0.9480075398652996。mmap 全量 86.497 QPS（原 26.556，提升 3.26 倍），热态第三轮 87.148 QPS；本轮 RocksDB 分别 174.115 / 168.196 QPS，buffer 分别 83.811 / 83.365 QPS。custom 构建由 116.793 秒降到 62.842 秒，RocksDB 9.319 秒；custom 仍为 64,500,320 字节，比 RocksDB 小 47.9%。没有宣称性能已追平。
- 完整结果与方法见 [Mac 优化报告](fts-custom-index-v1-mac-optimization.md)。本轮全部辅助工具/采样/结果在桌面 `zvec-fts-v1/mmap-opt-20261118/`，最终源码、二进制、仅 Quora 的复现命令和汇总已冻结于 `final/`；保留 v6 基线不覆盖。191 的公钥被服务端接受，但本机私钥未解锁，ssh-agent 仍无身份，最新 BatchMode 重试认证失败；等待用户在本机解锁，不读取/索取密钥口令。

### 2026-11-18 续作

用户要求 Ninja 只负责编译，测试直接执行编译产物。以下为当前续作证据，覆盖下文尚未更新的启动故障记录：

- 最终直接重试 `fts_column_indexer_test --gtest_filter='*DeathTest*'`：4 项 re-exec 均收到 signal 9，约 42.9 秒后批次退出 1；没有将这些失败记为通过。普通测试和此前 30 项独立启动的实际退出/恢复矩阵是单独证据。最终 33 个 tracked C++ 文件改动范围、46 个新 C++ 文件的 clang-format 检查及 `git diff --check` 通过。

- v6 有效输入基准全部完成：四个索引均重新构建。du 为 100,001 rows / 21,303,021 tokens，2,000 条 dev 查询在三路径 Top-100 的 DocID、顺序和 FP32 位模式完全一致，Recall@100=0.7857886501。Quora 为 522,931 rows / 6,032,630 tokens，10,000 条 test 查询在三路径精确一致，Recall@100=0.9480075399。du 三遍全量、Quora 固定 1,000 条样本三遍输出跨轮次/模式 SHA-256 一致，热样本与全量对应项也精确相同。输入 SHA-256 未变化，严格 UTF-8 验证通过。完整数据见 [benchmark](fts-custom-index-v1-benchmark.md) 和桌面 `bench-results-v6/summary.json`。custom 空间降低 41.31% / 47.89%，构建与检索明显更慢，默认 writer 保持关闭。

- 基准热样本的等价输入校验发现 fts-bench 原有 JSONL helper 漏解码：`JsonValue` 保留转义字节；现工具内增加独立字符串解码，正确处理 Unicode 代理对、控制字符和仅解码一层，非法转义明确失败、不修改调用方输出。没有修改生产 JSON 库。最小测试先复现两类失败，修复后桌面直接执行 4 项全部通过。Quora corpus 有 18,686 行含转义、queries 有 317 行，du corpus 有 3,579 行。此前 v4 两格式精确一致性仅是对同一错误输入的证据，相关 BEIR 指标已撤回为诊断；正式 v6 已完整重跑。v5 是尚未补全代理对处理的中间工具，不用于最终结果。 正式 CMake/Ninja 构建成功后，直接执行 `fts_bench_input_test` 的 4 项全部通过；含中文、换行、转义反斜杠及 astral Unicode 的 6 条查询在转义/UTF-8 × RocksDB/buffer/mmap 六组 CLI 输出逐字节相同。Standards 复审无问题；Spec 提醒工具仍依赖上游提供合法 UTF-8，未额外拒绝所有非法原始字节，本次两个数据集已按严格 UTF-8 读取验证。

- 完整 du-retrieval 构建暴露重复校验开销：诊断采样中主线程主要耗在词典验证反复计算同一页 SHA-256；该构建主动终止，不计入正式性能结果。新增 reader 私有、最多 128 KiB 的已验证页副本缓存，数据与校验目录页都按完整 key 匹配；副本计入共享 MemoryLimitPool，预算不足仍走原验证路径，淘汰不破坏已返回 range。没有跳过首次校验、放宽 checksum 或改变磁盘格式。
- 上述优化正式重建（含 `_zvec`、fts-bench）后，直接执行 container 26、field 16、generation 12、publication 常规 40，共 94 项全部通过。新增测试验证重复读取不重复 hash、容量上限、淘汰后 range 有效和最后引用释放额度；只读 Spec 复审未发现阻断问题。六条 smoke 查询在 RocksDB / custom buffer / custom mmap 下 DocID、顺序及原始浮点位模式完全一致。完整数据对照仍在执行。
- 全量数据的首条普通查询在 buffer/mmap 都复现范围额度不足：集成层原 1 MiB 上限只能容纳本机约 16 个保守计费的并存范围。新增生产 `OpenCustom` 重开、32 词 OR、4 线程同步起跑回归，先直接执行失败，再将集成层固定上限改为 64 MiB；无预分配、不改变全局 cache。最新正式 `_zvec`/fts-bench 构建成功，generation 13、field 16 全部直接执行通过，独立小预算错误测试保持原约束。前一构建的 reader 2 项（含 153 条差分组合）通过。Standards/Spec 复审无新阻断问题；44 个新文件及 33 个已修改文件的改动范围格式检查、`git diff --check` 通过。
- du-retrieval 全量对照继续发现共用执行层的遍历依赖：RocksDB 与 custom 的 Recall@1/5/10/100 相同，但 294/2000 条查询的 798 个结果有最多 2 ULP 差异，1 条查询近同分排序变化；custom buffer/mmap 的 2000 条查询完全一致。`DisjunctionIterator::score()` 原先按随 WAND 跳跃改变的 `matching_iterators_` 顺序累加。新增 next_doc/advance 最小回归先红，改为按不变子句顺序对当前 doc 的有效匹配累加后，桌面直接编译运行 10 项迭代器测试通过，含 WAND 开/关位级比较。此修复确实改变旧实现的 FP32 结合顺序，必须重跑 RocksDB 查询基线并保留旧结果；没有调整 BM25、boost、统计、collector 同分比较规则或扩大浮点容差。正式重建与全量对照正在进行。
- 评分顺序修复的正式构建及 `_zvec` 成功；直接执行 conjunction 10、reader 2（差分扩充至 162 组）通过。indexer 完整批次 96 项中普通 92 项通过，4 项 death test 的 re-exec 均被 SIGKILL，批次退出 1；未将其记为全部通过。前述独立进程 30 项退出恢复矩阵仍是单独的已有证据，本次评分改动没有重跑该矩阵。

- 直接运行 `fts_field_test` 的 11 项与 `fts_reader_test` 的 1 项均成功。后者覆盖 144 组查询条件，对比 mutable Roaring、legacy BitPacked、custom mmap/buffer 的有限及无限上界，DocID、分数和同分顺序完全一致。
- 新增非破坏性的 `FtsPostingCursor::bound(target, boost)`：在 skip 页中定位目标块，不推进 posting 游标。最小测试先复现旧实现总返回全局边界，修复后双模式通过。
- 新增测试复现并修复三个问题：OR 预读错误在空候选集/空 AND 路径丢失；短语 NOT 未验证位置，三篇文档中应保留两篇却返回零篇；legacy BitPacked 越界 offset 未被 adapter 拒绝。最小复现先失败，修复后均通过。短语 NOT 加入差分后共 153 组条件。
- 百万 TF 重复短语、查询词数/字节/深度预算、位置错误传播和后半段 posting 损坏不能返回部分 Top-K 的测试均通过。最新整体构建及 `_zvec` 成功，clang-format 修改范围检查和 `git diff --check` 通过。
- 本机 Snowball 构建工具改为复用桌面直接编译的同版本源码产物。仅本机 CMake `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` 指向桌面 `prebuilt-snowball.cmake`，没有修改仓库 CMake 或第三方源码。此前试验的 linker launcher 仍为空。
- 桌面 `run-tests.py` 批次的子进程被 SIGKILL，已停止。改为从 shell 直接执行相同测试产物，12 个目标共 418 项分批通过：field 15、reader 2、column indexer 96、conjunction 9、candidate 9、rewriter 30、query 8、recall 42、multi-segment 3、parser 62、legacy reducer 18、BitPacked 124。Indexer 首次 95/96 通过，多字段恢复 death test 的 re-exec 子进程被 SIGKILL；单独重试仍失败一次，再次重试通过。没有将失败批次记为整体成功，也未修改 death test 来避开 re-exec。

### P3 直接转换的局部进展

- 新增 `FtsMutableSnapshot`，在调用方完整 checkpoint 后固定 RocksDB snapshot，所有字段读取共用同一 view。持有 context owner；不修改源 postings、不删除辅助 CF、不提交版本或清理 WAL。
- 原始 doclen 的本机端序键先转为磁盘上的每行 5 字节暂存，再按行序写入；不建立全量行数组。统计从 snapshot 的 stat CF 读取，保留 N/total_tokens；presence 由精确长度 CF 独立恢复。
- 按 `$MAX_TF` 的有序词项读取对应 TF prefix range，使用完整键长度筛出当前词项。这样含 NUL 的词项与短词 doc suffix 在原 CF 中交错时，也能输出正确的 canonical 词序。positions 保留一个 legacy value，限 64 MiB，按 varint 单值消费；不生成完整位置 vector。此限制不能消除 RocksDB Get 内部的旧 value 物化。
- 两项直接转换测试先红后绿并直接运行通过。覆盖 row 255/256 的 native-key 排序、`a` 与 `a\0\0` 的 TF 键交错、缺字段/存在零 token、snapshot 后继续写入、源 postings 不变、损坏 positions 明确报错及修复源数据后重试。P3 的转换器仍未接入生产 seal，持久发布与其生命周期矩阵尚未实施。
- 已在桌面冻结 `legacy_open_0c63ea4`，其 `Version::Load` / manifest codec / pb wire / meta 源码经 `git diff HEAD` 确认未修改。旧 manifest fixture 读取退出 0；带非法 protobuf field-zero 前缀的门禁 fixture 退出 2，明确拒绝。此为当前 HEAD 的读取入口探针，不等于已验证全部历史发行版；实际新 manifest 的再次验证仍待协议落地。临时 probe CMake include 已移除，本机构建只保留 Snowball host-tool include。

### P3 manifest 与持久发布基础（2026-11-18）

- `BlockMeta` 新增可选 `FtsBaseDescriptor`，wire tag 7 为固定 80 字节 LE 记录：format/minimum reader、required features、generation、row count、base ID、完整 metadata SHA-256。legacy 缺失记录仍表示 RocksDB；未知必需版本/特性、非法身份与文档域被拒绝。其他未知可选 block 字段按长度跳过。
- 含 custom base 或已采用显式提交记录的 manifest 使用 `\0ZVEC_FTS_MANIFEST_V1\0` + u64 LE payload 长度 + SHA-256 + 原 protobuf payload。未采用新协议的 legacy 字节保持不变。解析先构造独立候选，失败不改变目标 metadata。逐字节截断/扰动、非法 descriptor 与旧 golden 测试通过；`Version::Load` 保留 `NOT_SUPPORTED`，不再吞成内部错误。
- 新增 `DurableFile`：私有 `.pending` 写入、文件同步、原子改名、目录同步；逐实例 hook 注入四个故障点。改名后的同步失败显式返回 `renamed=true`，由上层决定提交是否不确定。macOS 路径已实测；Windows 使用 `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)`，尚未在 Windows 实测，不据此宣称跨平台验收完成。
- `Version::Save` 改为上述原子持久替换。`VersionManager::commit(candidate)` 先把完整旧 checkpoint 的编号/摘要写入 `manifest.current`，再写新候选 manifest，最后替换权威记录。恢复有权威记录时只按其引用及摘要读取；没有权威记录的 gated manifest 被拒绝。显式协议启用后，即使无 custom 字段也继续保留旧程序门禁。manifest 读写上限 64 MiB，编号按现有 FileHelper 的 uint32 域检查，不截断溢出。
- 提交前复制 schema/segment/block metadata，成功后才移动到 current。最小测试先复现 candidate 修改会提前影响 current 与原始输入，修复后通过。权威记录发布结果不确定时标记 `requires_recovery`，拒绝后续 apply/flush/commit；collection 的写入口联动仍待生产集成。
- 桌面直接编译运行版本发布测试 6 项通过，含 8 个 manifest/authority 故障组合、首次 bootstrap 不确定、失败后更新候选重试、孤立高编号 manifest、已提交文件损坏不回退。正式 Ninja 构建后直接运行：durable 1、publication 5（随后正式重建的 publication 6 项也通过）、FTS manifest 7、旧 manifest golden 21、VersionManager 8、segment row ID 16 均通过。`_zvec` 重建成功。segment 全量 108 项与 collection 全量 107 项也已通过；该次正式构建对应的 8 个目标共 273 项全部通过。
- 实际新 encoder 写出的 `custom-manifest-v1.fixture` 经新 `Version::Load` 往返成功；冻结的 `legacy_open_0c63ea4` 直接打开退出 2，明确拒绝；原 legacy fixture 仍退出 0。探针和 fixture 均在桌面。此证据只覆盖冻结 HEAD 的实际旧读取入口，不代表全部历史发行版。
- 修改范围 clang-format 与 `git diff --check` 通过。本轮未修改仓库 CMake、第三方源码或公共 binding。旧 manifest 暂保留，尚未接入 snapshot pin 与回收队列；collection/segment 准备-提交-发布、完整 checkpoint 的文件同步、reader 延迟回收仍须完成后才能宣称 P3 完成。

### P3 文件生命周期基础

- 新增 `fts_file_lifetime`：candidate 默认在最后一个 owner 释放时清理；提交成功或结果不确定时 `preserve()` 保留文件；已不被权威版本引用时 `retire()` 延迟删除。回收队列节点在准备期分配，删除失败留队并可重试。尚待 collection 持有队列、legacy context 和 snapshot owner 接入，以及重开后的孤立 generation 清理。
- `FtsContainerReader::Open` 可接收文件 owner；范围句柄 pin reader，reader 按成员析构顺序先关闭 storage，再释放文件 owner。最小测试先复现双模式下范围尚存而文件已被删除，接入 owner 后通过；另覆盖 preserve 与删除失败重试。
- 最新正式构建成功后直接运行 container 24、field 15、reader 2、mutable snapshot 2、version publication 6，共 49 项通过，`_zvec` 重建成功。没有再次通过 Ninja 启动测试。

### P3 多字段 generation 准备与重开

- `FtsIndexer::prepare_custom` 从同一 mutable snapshot 输出完整多字段容器，预打开全部字段并完整验证后同步、原子改名、同步目录。使用 RocksDB UUID provider 生成不复用的 base/字段身份，FileHelper 路径同时包含 block、generation 和 base ID；私有 reservation 避免重用已有路径。所有源 CF 保留。
- descriptor 绑定实际持久化 `fts.meta` 的 SHA-256，包含未知可选记录；重开核对 base/segment/block/row domain、所有字段名和持久化分析配置，不根据文件存在推断提交。不可变写入和原地 DDL 返回错误。
- 正式构建后直接运行：多字段往返、双模式四个数据文件发布故障点与继续写入重试、descriptor/config mismatch 三项通过。重开 reader 的生命周期测试先复现提前删除；回收队列改为同路径 live owner 共用后，桌面直接编译的四项全部通过。legacy context 目录 owner 和 segment 级集成测试正在继续。
- 同次正式构建 `_zvec` 成功。container 24、field 15、reader 2 分批直接运行通过；串行 shell 批次在 container 通过后退出 137，field 单独直接重试通过，没有把该失败批次算作成功。辅助静态链接的依赖缺失尝试不计为测试通过。

### P3 segment checkpoint 与发布（2026-11-18 续）

- `Segment::prepare_fts_seal` 先完成全存储 checkpoint，再准备独立 metadata 与 FTS generation；`publish_fts_seal` 仅在调用方持久提交后切换 reader、metadata、sealed 状态。重开按 descriptor 选择 custom reader，并使用 Version 持久化的 mmap 模式。旧源 context owner 在最后查询释放后关闭并回收。
- 自定义 checkpoint 在 manifest/WAL 清理前同步 forward、vector、delete 文件与目录，并等待 RocksDB flush；显式提交协议启用后的后续写入 checkpoint 沿用该路径。目录同步目前 POSIX 实现；Windows 的完整 checkpoint 尚无已验证实现，明确返回 NOT_SUPPORTED，不能据此宣称 Windows 支持完成。
- 修复 forward close 吞掉 flush/footer 错误：最小用例先复现，随后 mem_store 全量 51 项通过。失败保留 writer/buffers，不把不完整 forward 文件当成 checkpoint 完成。
- 新测试先复现不确定提交后仍可写入、checkpoint 同步失败后删除快照被提前确认两个错误。写 WAL 前检查恢复状态；collection 的写入、删除、DDL、optimize、flush 检查同一状态。删除快照的 dirty 状态只在版本提交成功后确认。`VersionManager::checkpoint(candidate)` 保留 legacy wire format，持久化成功后才替换当前 metadata；已采用显式权威记录时复用 commit 协议。
- 最新正式构建及 `_zvec` 成功。直接运行 segment publication 8、version publication 7、VersionManager 8 均通过；上一批 generation 5、mem_store 51、durable 1 与 segment publication 4、version publication 6 共 67 项全部通过。segment 108 与 collection 107 的再次全量回归也均通过；该轮五个目标共 238 项通过。
- `SegmentManager::switch_fts_writer` 已接入准备/提交/发布协调，生产分支受内部默认关闭门禁控制。预先同步旧段、准备 manager map、新后继 writing 段和候选 Version，持久提交成功后仅发布已准备状态；明确失败丢弃后继段，不确定提交保留新旧文件并冻结写入。新增双模式用例通过，当前 segment publication 共 16 项直接运行通过；覆盖后继创建失败、提交前失败与继续写入重试、连续切换以及不确定提交后的完整重开。checkpoint 中途失败与整个 segment 销毁的生命周期边界仍在补验，尚不能称为 P3 完成。生产默认 seal 仍为 legacy；P4 合并/迁移/DDL/snapshot 及 fts-bench 尚未运行。

### P3 checkpoint 重试、恢复与目录生命周期补验

- 最小用例复现 forward close 成功、immutable reader open 失败后，下一次写入覆盖原文件并丢失行。保留完成的 writer 到 replacement 打开成功，checkpoint finishing 状态在接收新 WAL 记录前先重试。两种读取模式通过恢复文件、重试、继续插入、seal 和重开后的完整 DocID 检查。
- 最小用例复现 WAL replay 重复累加已经部分 checkpoint 落盘的 FTS 统计（应 N=3/tokens=4，实为 4/5）。重放完成后有界遍历精确 doclen CF 重建 N/tokens，保持缺字段/存在零 token 的语义；新旧重复写入后的 posting 数据不再次累加统计。两种模式下重开与再次 custom seal 通过。
- 新崩溃测试在 full checkpoint 中途失败后直接 `_Exit`，不靠析构 flush。普通沙箱中的 re-exec 子进程数次被 SIGKILL；获准放宽沙箱后直接运行，两种模式均通过。未修改安全软件、代码签名设置或测试重启机制。
- 最小用例复现整个 segment 销毁会在 FTS query owner 存活时提前删目录。回收队列的子文件/context owner 现在 pin 已登记的父目录，segment 目录也走最后 owner 释放后的回收；legacy/custom、双模式查询跨销毁通过。
- 多字段配置在编码和创建文件前预检累计名称、tokenizer、filter、extra_params 与记录开销；包含大量空 filter 的配置也计费。去掉 schema 比较中的重复编码。预算用例先红后绿，generation 六项通过。
- 本轮正式重建及 `_zvec` 成功；直接运行 segment publication 常规 22 项及 crash 2 项、generation 6、container 24、segment 108、collection 107 全部通过。column indexer 原有 96 项中 93 项通过、3 个 death 子进程 SIGKILL；再次单独运行 death 组四项仍被 SIGKILL，未将这些失败批次记为通过。此前回归通过记录保留，但不能代替本轮失败项。
- P3 尚有平台与集成补验项：Windows full checkpoint 明确未支持；源混合 scalar/vector 的定向自定义发布、完整真实退出阶段矩阵及候选创建中途残留清理仍需补齐。新旧 manifest/快照/orphan 的共同回收随 P4 完成。

### P4 合并输入基础

- 增加 `FtsTermSource` 和字段/词典 writer 的分块词项入口；只请求至多 4 KiB，保留输入 owner 到当前词结束。旧 string_view 入口仍限制 1 MiB；新格式合并可以流式写出更长词，不改变 wire format、ordinal 或 fingerprint 规则。
- 2 MiB+17 字节词项最小测试先红后绿，双模式完整 verify/分块读回通过；补充短词/长词前缀排序、重复和逆序拒绝、短读及来源错误的 sticky 失败。正式构建及 `_zvec` 成功，直接执行 field 16、container 25、reader 2、generation 6，共 49 项通过。
- 新增 `fts_source`：legacy 从 forward presence 和 sealed postings 恢复每行精确长度，并检查逐行 TF 总和与源 total_tokens；使用磁盘暂存，不重分词。custom 按 ordinal 导出词项，超长 key 不物化。源、词项与游标均持有原 generation；错误与 EOF 分开返回。legacy 单 posting value 默认限 8 MiB，仍无法限制 RocksDB 内部先发生的物化。
- 新增 `fts_merge`：新旧统一 source，按 forward 输入顺序与已有删除 bitmap/rank 映射，按块比较词项、流式重编码 posting/positions。普通合并 N=目标物理行数，单输入纯 rewrite 保留源 N；精确长度、presence 和 total_tokens 先校验再构建。最多 64 个输入，预留 term 工作区总预算 128 MiB；另有同时最多一个 legacy positions value（64 MiB），共享缓存和已有删除 bitmap 单独计费。超过预算明确失败，不无界 fallback。
- `FtsIndexer::PrepareMerged` 与 mutable seal 共用已验证的候选 generation 文件发布流程；未变化 custom 字段保留逻辑 ID/incarnation，新 base 使用新身份。正式构建及 `_zvec` 成功，直接运行 generation 11、field 16、reader 2 全部通过，包括混合新旧 + 删除 + 双读取模式、纯转换逐项精确分数、多字段重开，以及此前发布故障/生命周期回归。本轮 column indexer 96 项中 92 项通过，4 个 death 子进程 SIGKILL；单独重试一个 death 用例仍 SIGKILL，不能记为全部通过。
- 新增独立 `RewriteFtsTask` 与 segment source/pure rewrite 接口；旧段从 Arrow forward 批次恢复 presence，保留已有行号、删除标记和评分统计。混合 compaction 已选择统一 reducer；输出容器在写入前绑定最终 segment ID，临时目录改名不改变持久身份。桌面直接运行双模式的单段 rewrite、global DocID 空洞 + 删除 + forward 对齐 + 改名重开共 4 项通过，测试先在旧实现明确失败。
- collection optimize 已接入预打开全部替换段、预分配 manager map、持久候选提交后发布；向量任务和单段 FTS rewrite 的 metadata 按块合并。public collection optimize/reopen 集成测试双模式通过；此前正式 generation 11、publication 新增 6、segment 108、collection 107 项通过。默认 writer 门禁仍关闭。

### P4 多字段 DDL、snapshot 与回收补验

- custom snapshot 复制不可变文件到独占的新目录，保持 basename 与 descriptor，完成文件和目录同步。源 generation 退休并删除后，独立 snapshot 仍能精确读取；已有目标目录拒绝覆盖。
- create/drop/rebuild 生成包含全部有效字段的新容器；只对创建或重建字段扫描 forward 并重新分词，经私有 mutable CF 直接输出 custom source，再统一重编码。未改字段保持 ID/incarnation、N 与 token 统计，重建字段获得新 incarnation。私有 mutable 和中间 custom 文件均由 owner 回收，不经过 legacy BitPacked seal。
- collection FTS DDL 预打开候选完整 segment、同步依赖并预分配 manager map，独立 checkpoint 成功后发布；不确定提交保留候选。新增失败用例阻止 authority 临时文件写入，确认旧 schema/查询继续有效、解除故障后重试成功。多字段 drop 保留旧 query reader 到最后引用释放；最后一个索引删除后可重新创建 custom 索引。
- 关闭后复制整个 collection，移除原件，备份只读重开后 PK/分数完全一致。新用例复现并修复 Segment::CreateAndOpen 中途失败留下目录：仅独占创建的目录可清理，已有目录及其文件保持原样。
- 上述实现正式重建（含 `_zvec`）成功，直接执行 generation 12、publication 常规 34、segment 108、collection 107 项全部通过。本轮额外直接运行两个 checkpoint crash death tests，re-exec 子进程均被 SIGKILL；不能继承先前通过记录宣称本轮 crash 全绿。
- 恢复回收已接入：成功读写恢复后按固定权威文件清单清理严格匹配的 FTS/manifest/delete 残留，只读打开不清理；未知名称、非规范编号与 symlink 保留。每轮最多 4096 项，失败留在重试队列。共享同一路径的进程内 retirement queue，使旧 reader 跨 manager 重开仍能 pin 文件/目录；Version 快照持有 manifest/delete 文件 owner。正式 recovery cleanup 2 项通过；generation 12、container 25、field 16、publication 常规 36、segment 108、collection 107 项再次全部通过，含 `_zvec` 构建成功。
- 新增 2 MiB+17 字节长词的两源归并与删除重映射验证，双模式通过。custom checkpoint 的完整 scalar inverted/vector Flat/删除记录/FTS 组合，经读写及只读重开检索通过。Version publication 原 7 项有一项依赖旧 manifest 永久保留；改为显式持有 Version 快照后，桌面直接编译并执行全部 7 项通过，正式目标在随后重建中更新，直接执行 7 项全部通过。
- 桌面独立工具 `fts_crash_probe` 分别直接启动 prepare/verify，不由 Ninja、Python 子进程或 shell 循环启动。custom 文件、manifest、authority 各 write/sync/rename/dirsync，加 commit 后 publish 前与 publish 后，共 14 个退出点 × buffer/mmap = 28 项全部通过。每项实际 `_Exit` 后重开，检查旧/新权威选择、精确浮点分数、删除过滤及继续插入。shell 循环启动版本被 SIGKILL，已明确区分失败启动与之后独立直接运行通过的证据。
- 独立 `source-checkpoint` 退出补验已在 buffer/mmap 两种模式通过：初始 manifest 仅含 writing block，制造 forward 发布中途失败后实际退出，恢复旧 checkpoint/WAL，核对精确结果、删除和继续写入。真实退出矩阵累计 30 项通过。
- 按 code-review 技能，以固定 HEAD `0c63ea4ea936fcf4b6c352f7955e087143206baf` 对工作区并行做 Standards / Spec 审查。Standards 未发现需修复问题；Spec 找到旧删除快照提前 unlink、数值列 DDL 提交前删除旧 forward 文件两项。均先以直接执行测试复现，再修复：删除文件由 Version owner 退休；显式 authority 集合的数值列 DDL 在独占新段目录上执行，FTS 绑定新段 ID 重写，预打开/同步后提交，旧段在发布后退休。此路径复制 forward/vector，并用 RocksDB checkpoint 固定 scalar 依赖，存在额外磁盘/时间开销。
- 上述两个回归的 buffer/mmap 共 4 项直接运行全部通过；正式 `_zvec` 构建和 version publication 7 项通过。复审新增量化向量 drop 保留旧 raw block 的问题已复现并修复，覆盖关闭/重开；无 FTS 字段后继续使用显式 authority 的 writer 切换已补齐，注入提交失败后保留旧 writer 并成功重试。最新正式 publication 常规 40 项全部通过，`_zvec` 重建成功。前一构建的 segment 108、collection 107 项本轮直接执行全部通过；它们覆盖数值列私有发布，新增量化 drop 与无 FTS 切换修复由最新 publication 目标覆盖。
- fts-bench 已开始增加 `mode=index` 的同路径对照：相同 FtsIndexer mutable 写入和确定性行号，legacy seal_all 与 custom direct prepare 分流，分别计时；严格传播错误，记录逐查询有序 DocID 与原始 float 位模式。辅助工具和数据位于桌面，性能数据尚未产生。尚余跨平台支持与检查、完整 P5 资源测量及启用验收；默认 writer 门禁仍关闭。

## 当前交付范围

从 `0c63ea4ea936fcf4b6c352f7955e087143206baf` 开始。已增加独立格式、容器与字段读写层，查询树接入已完成本机验证。collection 的切换、DDL、混合合并和迁移已接入准备/提交/发布；默认 writer 仍受关闭的内部启用门禁控制。已存在显式 authority 的集合继续采用持久发布协议，没有新增公共配置。

- `fts_format.h` / `fts_format*.cc`：显式 little-endian 整数、严格 uvarint、SHA-256、流式 canonical 词典指纹、带长度的元数据扩展、字段身份与统计、前缀词典块与 oversized 引用、TermMeta 显式 inline 标签、posting/positions/doclen/presence 块。
- 128-doc packed 块复用已有 SIMD dispatch；尾块按实际 count 编码。块首 gap=0，TF 编码 tf-1；不重复存储 doclen。完整块与尾块的布局有明确 flags。
- position_index 块使用显式 64-bit 宽度，每 32 文档保存 positions 与 varint 长度流的双偏移检查点；定位最多累加前面 31 个长度。完整解码检查所有检查点、长度总和与溢出；发布前必须调用完整校验，随机定位只检查当前访问范围。
- `fts_container.h` / `.cc`：把已排序/编码的数据 section 流式写入调用方准备的 FileDumper；校验目录暂存到自动关闭的临时文件，内存不随 posting 数增长。结束时封装 `checksums` 和 `fts.meta`。
- reader 通过 `enable_mmap` 参数选择现有两个后端。该参数将来必须来自 DB 持久配置。范围句柄同时保留 storage owner、页 pin/scratch 和资源额度；没有暴露主动 close。
- 每个逻辑校验粒度保存 SHA-256；每 128 个摘要组成受自身 SHA-256 保护的目录页。目录按需读取。open 校验 metadata 和必要 section 描述，`verify_all()` 流式校验所有数据。
- `fts_dictionary.h` / `.cc`：词典 Writer 使用三个临时 section，按块输出并生成指纹；Reader 提供 ordinal 遍历、seek、seek_ge 和流式 verify。稀疏目录每 128 个记录采样上层 fence，逐层构建/定位，直到顶层不超过 128 条；只保留当前索引页。词项句柄持有文件 generation，超长词每次最多读取 4 KiB。verify 检查上层 fence 与叶层一致、排序、目录连续覆盖、TermMeta 行号与 section 引用、词典指纹。
- `fts_stream.h` / `.cc`：有界临时数据 spool 和游标局部的 section 页缓存；复用范围句柄，保留 generation 与资源额度，读取错误不会被转换为 EOF。
- `fts_positions.h` / `.cc`：每文档位置流与独立 PositionCursor。TF≤128 为一个 counted 子块；更长列表使用 32 字节文档头和分页的 32 字节子块定位记录；顺序遍历、单调 seek 均只解码当前最多 128 个位置。保留重复位置及完整 uint32 值域。
- `fts_field.h` / `fts_field_writer.cc` / `fts_field_reader.cc`：独立逻辑字段 Writer/Reader，协调文档长度、presence、DF/总 TF、位置、词典、指纹与上界。DF=1 内联、DF=2..127 varint、DF≥128 packed128 + 48 字节 skip；TF=1 的单例位置不写外部流。临时原始 posting 和每行 TF 合计落磁盘，按块处理，校验 TF 合计与精确 doclen 一致。
- FP32 上界由现有 BM25Scorer 逐 posting 计算，绑定 N、total_tokens、参数与版本。新 reader 默认返回无限上界；只有完整 verify 在当前评分路径逐项验证安全后才启用有限值。参数不匹配、boost≠1、非默认舍入模式或保存上界偏低均保守退化。P2 必须显式禁用依赖无限上界的剪枝，不允许做 `∞−∞`。
- 打开存储前有界检查 FileDumper 外层 header/footer/目录：限制条目数、目录大小与版本字符串，检查 CRC、名称终止/重复、范围和布局，拒绝未知版本/链式容器。该检查依赖不可变 generation 的路径在打开期间不被外部修改；发布与路径管理仍属于 P3。

`FtsContainerWriter::finish()` 只完成 **PreparedCustomFile 的文件编码工作**，不提供 fsync、独占路径分配、目录同步或数据库版本提交。不得直接把它接入现有 seal 后宣称安全发布完成。

## P0 核对及启用门禁

源码核对结果与主规范一致：

1. `RocksdbContext::prepare_options()` 调用 `OptimizeLevelStyleCompaction()`；8 MiB write buffer 只属于全局 merge operator 分支。FTS 使用 per-CF merge operator，不能沿用这个数值。实际 options 由新增 `CustomFormatBaselinePreservesSealStatistics` 测试通过 `DB::GetOptions()` 输出。
2. `write_opts_.disableWAL=true`。本轮没有改变上层 WAL 或 checkpoint，也不宣称省掉另一份 RocksDB WAL。
3. 新 seal/纯转换应保留字段原有 N/total_tokens；普通 reducer 保留物理目标行数和既有删除过滤口径。新增基线覆盖 row 1 缺字段、row 2 存在但零 token，源 N=3、total_tokens=5。
4. P0 基线的 block 字段为 1..6，manifest 字段为 1..8。P3 已追加 block tag 7，未复用旧编号；旧 decoder 会跳过未知字段，因此采用外层非法 field-zero 前缀门禁，见上文当前验证。
5. 没有验证任何已发布旧二进制会拒绝新 descriptor，因此**没有获准启用新 writer 的兼容版本**。生产开启条件仍为：兼容 reader/可验证的旧程序拒绝机制、安全发布、查询差分、迁移与生命周期验收全部通过。不能仅凭写入新的 minimum reader 字段开启。

本机实测 RocksDB 8.1.1，源码 revision `6a436150417120a3f9732d65a2a5c2b8d19b60fc`。六个 CF（postings、positions、TF、maxTF、doclen、统计）均报告 `write_buffer_size=134217728`、`max_write_buffer_number=6`、`compression=0`、`block_cache capacity=8388608`、`block_size=4096`。这些是有效 options 数值，不代表当前已分配 RSS，也不把缓存容量按 CF 数相乘。尚未测量各层压缩策略对应的实际 SST 压缩率。

可复现评分基线（whitespace + lowercase，k1=1.2、b=0.75）：

| 输入/查询 | 结果 |
|---|---|
| row 0 | `alpha alpha beta` |
| row 1 | 缺字段，不调用字段 insert |
| row 2 | 空文本，存在但零 token |
| row 3 | `beta gamma` |
| `alpha` | row 0: 1.10093081 |
| `alpha OR beta` | row 0: 1.45504308；row 3: 0.434457153 |
| `"alpha beta"` | row 0: 1.45504308 |

这组数据在旧格式 seal 前后保持 N=3、total_tokens=5、结果顺序和既有浮点测试契约。它是语义基线，不是吞吐或空间 benchmark；P0/P5 的大数据性能基线尚未完成。

发布前置项仍需在 P3 修复：

- `Version::Save()` 的直接写入已替换为原子持久发布；需要继续核验 collection/segment 的全存储 checkpoint 联动。
- `VersionManager` 已增加独立候选提交、显式权威记录和不确定状态；旧 manifest 尚待接入延迟回收。旧格式且无权威记录时保留历史恢复兼容路径。
- collection seal 会在 manifest 成功前改变 segment manager、writing pointer 和 metadata。
- FileDumper close 不等于 fsync；源 checkpoint、旧 owner、提交状态不确定和延迟回收需要联合处理。

## 已落实的局部资源限制

这些是基础模块的工作区限制，不是数据库 RSS 承诺：

| 项目 | 当前限制 |
|---|---|
| metadata 编码字节 | 默认 1 MiB；最多 256 字段、4096 section |
| 普通词典块 | 最多 64 词、4096 编码字节；解码词条累计字节单独计费，默认 256 KiB |
| oversized 词 | 显式 uint64 offset/length；本模块不物化外部词字节 |
| 词典 Writer | 当前块、前一个普通词和三个临时 section；超长输入通过 string_view 或 4 KiB 分块 source 消费，不另复制整词 |
| 词典 Reader | 当前索引页、当前解码块；超长词比较/指纹按 4 KiB 分块，句柄不保留整词 |
| 字段 Writer | string 输入最多 1 MiB；不可变 source 入口按 4 KiB 读取超长词；固定 posting/位置块和磁盘暂存逐行/逐词数据 |
| posting / positions 工作块 | 最多 128 条；数组固定大小 |
| skip / 长位置目录 | 固定 48 / 32 字节记录，按页访问；不加载完整列表 |
| position_index 块 | 最多 1376 编码字节；4 个双偏移检查点与 128 个长度 |
| writer 校验目录 | 4096 字节摘要页；收尾按 16 KiB 缓冲复制临时目录文件 |
| reader 范围额度 | 默认每个 reader 共 1 MiB；所有并存范围共享原子计费 |
| 生产 generation 范围额度 | OpenCustom 使用每容器 64 MiB admission 上限，不预分配；多个 segment 可累加，不代表进程 RSS 上限 |
| 已验证页缓存 | 按 2026-11-18 用户决定移除；正常查询复用 native mmap/buffer 句柄，不建立 SHA 校验副本 |
| 范围计费 | 保守预留 `4 * max(native_page_size, layout_granule, 4128)`；跨粒度再预留请求长度的 scratch |
| layout_granule | 1..64 KiB 的 2 次幂；不等同于系统页大小 |

metadata、外层 section 描述和 backend 全局页缓存分别计费。mmap 地址空间不作为 RSS。词典/skip/位置目录页不在 open 时全部载入。新 posting/position 游标已通过 section 范围句柄复用当前页。P2 仍须限制单查询游标总数与生命周期。

## 验证记录

本机 macOS ARM64；Linux、Windows、iOS、Android 和跨端序尚未实测。未改 CMake、公共头、binding 协议或第三方源码。

- 此前基线通过：BitPacked 124、buffer storage 16、旧 FTS indexer 94（含新增统计/实际 options 基线）。没有把这 234 项解释为本轮新增用例。
- 本轮 50 项测试通过标准 Ninja 目标：格式 19、容器/词典 22、字段 9，分批执行。连同此前基线共 284 项通过；不是一次组合命令全部成功。
- `_zvec` 已随新增字段代码编译成功。未改 Python 源码或 binding 契约，未运行 Python pytest。
- 最后组合运行 `ninja -C build.release -j4 unittest.fts_container_test unittest.fts_field_test unittest.fts_format_test _zvec`：容器 22、格式 19 通过，binding 链接成功，字段二进制在测试启动前遭 SIGKILL/137，因此组合命令退出 137。经同一 `/bin/sh` 直接执行同一字段二进制，9 项通过。随后只删除可重新生成的 `build.release/bin/fts_field_test`，保持源码/并发参数不变，执行 `ninja -C build.release -j4 unittest.fts_field_test`，9 项通过且整条命令退出 0。结果支持重链接后的可执行文件状态相关假设，但未确定 OS 层根因，也未宣称消除此环境问题。
- Ninja 仍会报告构建记录不完整并重新构建；未解析或修改构建记录来绕过。失败尝试不记为标准目标通过。
- 词典覆盖 9002 词、跨稀疏目录页、两个超过 1 MiB 的词、NUL/高位字节、空词典、owner 释放后读取、256 KiB 范围预算，以及重新计算合法 checksum 后的错误目录/元数据引用；双模式均通过。位置目录覆盖 golden、1/31/32/33/127/128 文档、64-bit 偏移/长度、零长度、截断、未知宽度和错误检查点。
- 新增双后端验证覆盖三层词典目录（16385 个叶块）、长位置头/目录 golden、DF=1/2/127/128/129、TF=128/129/1000003、百万位置低预算读取、缺字段/存在零 token、多个字段、N 不等于 presence、当前 scorer 分数/上界、偏低上界退化、8 个并发游标、合法 checksum 下的错误位置目录、源 I/O 错误与 writer 失败状态。
- `clang-format -n --Werror` 对新增 C++ 文件和已有测试的新增范围通过；`git diff --check` 通过。没有顺带格式化已有测试的其他部分。
- 初期工具/依赖缺失，以及临时程序/Snowball/测试进程被 SIGKILL 的尝试，不记为通过。独立动态库执行是辅助验证，不能代替标准构建验收；后续标准目标已能执行。
- 本轮独立校验工具按用户要求放在 `~/Desktop/zvec-fts-v1/`。正式单测源码在仓库 `tests/`，正常构建产物留在 `build.release/`，不提交。

## 后续阶段与基准依赖

P2 实现与早期验证记录（后续验收结果见本文顶部 2026-11-18 续作）：

- 新增 `fts_reader` 与 `fts_legacy_reader`。Term/Phrase 查询只使用 generation 所有权句柄和游标；RocksDB Get/MultiGet、Roaring、BitPacked 和 positions key 留在 legacy adapter。短语验证通过独立逻辑位置游标单调对齐，重复词共享底层 legacy value，但不共享游标状态，不再解码完整位置 vector。
- `FtsIndexer` 的 RocksDB context 由最后一个 owner 关闭；保留字段 reader 后关闭 indexer 的最小测试先复现 SIGSEGV，修复后通过。Term、Phrase、AND/OR 汇总错误，搜索返回前检查，避免 EOF 隐藏读取失败。
- 查询限制 1024 个逻辑词项、总词字节 1 MiB、4096 AST 节点、128 层深度；legacy adapter 限制一次批量保留 value 为 64 MiB。这不限制 RocksDB 在 Get/MultiGet 内部已发生的临时物化，也不是整个进程的内存承诺。
- boost 不为 1 时统一使用无限上界。已复现并修复无限上界的 `∞−∞`；另一个最小测试复现大上界相减丢失剩余小分数（应得 8 分，实际提前返回 0）。修复采用独立后缀上界与有证明的浮点累加误差界，不改实际分数运算顺序。
- 在收拢 legacy 查询后，标准组合目标通过：`fts_column_indexer_test` 95、`fts_field_test` 10、`fts_conjunction_opt_test` 8。之后新增的剩余分数测试与修复，直接运行 9 项剪枝测试通过，标准目标仍待后续组合重跑确认。新 `fts_reader_test` 的 Roaring/BitPacked/custom 双模式、有限/无限上界查询差分尚在验证。
- 系统事件已明确记录测试程序及 Snowball 构建工具的 AMFI `Unrecoverable CT signature issue`，确认启动前 SIGKILL 与签名校验有关；触发签名异常的上游原因尚未确定。重新签名、替换 inode、单任务构建、延迟 5 秒和沙箱外执行均未恢复正式目标。桌面 `launch-repro` 中只有 `int main(){return 0;}` 的最小程序也能复现；相同 Snowball 命令脱离 Ninja 直接执行成功。桌面 Ninja 副本能执行已有程序，但仍不能稳定运行刚编译的程序。用户确认存在安全软件，等待软件名称/拦截记录以继续定位。
- 诊断工具均在桌面 `zvec-fts-v1/`；未改安全软件设置或第三方源码。试验用 CMake linker launcher 已恢复为空，避免本地诊断脚本成为构建依赖。所有当前修改范围和新增文件的 clang-format 检查、`git diff --check` 通过。最新差分用例尚未成功编译执行，`_zvec` 尚未随 P2 重建。
- 当时尚待查询差分、嵌套错误与资源限制、目标块上界和短语 NOT 验证；这些 P2 项目现均通过本机验证。P3 新增 `FtsMutableSnapshot` 的直接转换测试正在实施，尚未接入生产 seal。

- P2 接入 generation owner、legacy adapter、Term/Phrase 查询，汇总读取错误，处理无限上界及有界短语验证；再对比既有语法、结果、分数、过滤、candidate_ids 和同分行为。P1 往返测试不能替代查询差分。
- P3 从 RocksDB 原始 TF/positions/精确长度直接转换，恢复 presence，固定源统计，落实持久化发布与失败重试；不得先转 legacy BitPacked 再转 custom。
- P4 的新格式合并可按 ordinal 遍历；已增加分块 term source 入口，下一步接入统一 source reader 与 reducer。
- 用户要求的 fts-bench 对照数据为 `/Users/jiliang.ljl/Project/BEIR/du-retrieval` 和 `/Users/jiliang.ljl/Project/BEIR/quora`。工具/转换脚本及输出放在 `/Users/jiliang.ljl/Desktop/zvec-fts-v1/`，不放 `/tmp`。先逐查询验证召回/结果一致，再报告构建时间、索引大小与检索性能；两种 custom 读取模式均纳入对照。目前尚未运行，未报告性能结论。

当前状态以顶部续作与 P3/P4 补验记录为准：P1/P2 已通过本机验收，P3/P4 的直接 seal、安全发布、混合合并、单段迁移、DDL 和 snapshot/备份已实现并完成本机关键验证；跨平台与全部旧版本兼容门槛尚未完成。fts-bench 已以修正 JSON 输入后的 v6 完整执行，生产 writer 未启用，wildcard/regex/FM 仍延期。
