# FTS custom v1：fts-bench 实测（v6）

日期：2026-11-18。工作区基点：`0c63ea4ea936fcf4b6c352f7955e087143206baf`，结果来自未提交的实现。主规范为 [implementation](fts-custom-index-v1-implementation.md)，功能与故障验证详见 [progress](fts-custom-index-v1-progress.md)。

**两个数据集的全部查询在 RocksDB、custom buffer、custom mmap 下，Top-100 DocID、顺序和原始 FP32 分数位模式完全一致。custom 最终索引更小，但当前构建与检索明显更慢，生产 writer 默认门禁保持关闭。**

本页只使用修复 JSON 输入后的 v6 完整重跑结果。此前 v4 报告因工具漏解码 JSON 字符串已撤回，仅保留作诊断；v5 是中间工具，不参与结论。

## 方法与可复现条件

- macOS 27.0（26A428）、Apple M6、12 个物理/逻辑 CPU、16 GiB 内存；Release 构建。本机以外平台未实测。
- 使用 `fts_bench -mode=index`：两种格式走相同 FtsIndexer mutable 插入，按 corpus 顺序分配 DocID，合并 title/text，单个逻辑 segment。RocksDB 执行现有 `seal_all()`；custom 从辅助 CF 直接构建并完整验证新格式。两者同步依赖、发布 benchmark descriptor 并清理 mutable 后记录成功。该 descriptor 是独立工具协议，collection 的提交/恢复由功能测试覆盖。
- 四个索引均由 `fts_bench_compare_20261118_v6` 从头构建。单写入线程、单查询线程、OR、Top-100；查询计时包含分词与检索，不含结果文件写入。构建各执行一次有效全量样本，所有计时任务串行，无并发编译或其他基准。
- 实际 RocksDB write buffer 为 128 MiB、compression=0、block cache=8 MiB；custom 共享 MemoryLimitPool=8 MiB。custom 每容器范围 admission 上限 64 MiB，无预分配；已验证页副本最多 128 KiB，计入共享池。这些均不是进程 RSS 硬限制。
- du-retrieval：100,001 篇、21,303,021 tokens，`jieba + lowercase`，完整 2,000 条 **dev** 查询。源 Parquet 转为桌面 JSONL；工具约定文件名 `qrels/test.tsv` 实际仍为 dev split。
- Quora：522,931 篇、6,032,630 tokens，`standard + lowercase`；从 15,000 条 query 按 test qrels 选取完整 10,000 条。原始数据集未修改，6 个输入文件 SHA-256 与准备时相同，严格 UTF-8 验证通过。
- du 每条读取路径运行完整查询三遍；Quora 全量查询各运行一遍用于正确性与首轮性能，再按固定种子 `20261118` 抽取同一组 1,000 条 test 查询，各运行三遍测热查询。热轮取 pass 2，不与完整查询的分布混算。首轮是进程内缓存首次使用，**未清空 OS page cache，不能称为真正冷缓存**。
- 最终索引大小是 custom 文件或完整 RocksDB FTS 目录的逻辑文件字节数，不包含共同的 DocID 映射、benchmark descriptor、原文、向量或数据库 WAL。

工具与数据目录：[桌面工作目录](/Users/jiliang.ljl/Desktop/zvec-fts-v1/)。有效结果：[summary.json](/Users/jiliang.ljl/Desktop/zvec-fts-v1/bench-results-v6/summary.json)；[逐条复现命令](/Users/jiliang.ljl/Desktop/zvec-fts-v1/bench-results-v6/commands.sh)；[输入验证](/Users/jiliang.ljl/Desktop/zvec-fts-v1/bench-results-v6/input-verification.json)。重跑应使用新索引目录；工具拒绝覆盖已有目录。

## 召回与精确一致性

两种格式的 `document_ids.tsv` 已逐字节核对相同。完整结果保存有序 `(DocID, FP32 bits)`，不仅比较相关文档召回率。du 共 2,000 条、Quora 共 10,000 条，三条读取路径全部精确一致；du 三遍、Quora 热样本三遍也逐文件 SHA-256 一致，热样本还与完整结果中的相同查询精确一致。

| 数据集（三路径相同） | Recall@1 | Recall@5 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| du-retrieval dev | 0.172573 | 0.477474 | 0.599540 | 0.785789 |
| Quora test | 0.569708 | 0.784792 | 0.844252 | 0.948008 |

## 构建与空间

时间为秒，空间为 MiB（2^20 bytes）。总时间包含输入处理、mutable flush、seal、完整验证、同步、descriptor 发布与清理。

| 数据集 | 格式 | 写入及 flush | seal 及同步 | 总构建 | 最终索引 | 构建峰值 RSS |
|---|---|---:|---:|---:|---:|---:|
| du | rocksdb | 20.445 | 1.337 | 21.785 | 284.932 | 956.641 |
| du | custom | 20.297 | 212.873 | 233.181 | 167.240 | 946.141 |
| quora | rocksdb | 8.803 | 0.492 | 9.298 | 118.040 | 1092.000 |
| quora | custom | 8.804 | 107.982 | 116.793 | 61.512 | 1079.062 |

du：custom 空间降低 41.31%，总构建时间为 RocksDB 的 10.70 倍。
quora：custom 空间降低 47.89%，总构建时间为 RocksDB 的 12.56 倍。

mutable SST 是 seal 前的阶段性快照，受后台 compaction 时序影响，不是峰值或最终空间口径：

| 数据集 | RocksDB 构建时 mutable SST (MiB) | custom 构建时 mutable SST (MiB) |
|---|---:|---:|
| du | 467.951 | 467.951 |
| quora | 276.095 | 290.604 |

custom 文件内部空间分解（MiB）：

| 内容 | du-retrieval | Quora |
|---|---:|---:|
| 词典字节 | 2.175 | 0.456 |
| 词典分层目录 | 0.216 | 0.045 |
| TermMeta | 47.946 | 10.034 |
| postings | 19.476 | 8.018 |
| skip | 3.317 | 1.886 |
| positions | 66.273 | 27.249 |
| 位置长度/检查点索引 | 26.132 | 11.265 |
| 精确 doclen 与 presence | 0.396 | 2.073 |
| checksum、容器、metadata 与 padding 合计 | 1.309 | 0.486 |

最后一行按完整文件大小减公开数据 section 长度计算，未进一步拆开。各 CF SST 原始字节保存在 `*-rocksdb-stats.txt`，未把旧格式嵌入式 doclen/skip 分摊为未经测量的值。临时磁盘峰值、旧 generation 共存峰值尚未测量。

## 检索性能

全量首轮如下，延迟单位 ms。每行使用完整相同查询，未清空 OS cache。

| 数据集 | 读取路径 | 查询数 | QPS | P50 | P95 | P99 | 进程峰值 RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| du | rocksdb | 2000 | 1589.43 | 0.505 | 1.483 | 2.640 | 320.500 |
| du | buffer | 2000 | 170.57 | 5.435 | 10.340 | 14.138 | 267.734 |
| du | mmap | 2000 | 173.98 | 5.317 | 10.072 | 13.558 | 322.359 |
| quora | rocksdb | 10000 | 174.36 | 3.996 | 16.457 | 29.135 | 101.953 |
| quora | buffer | 10000 | 26.48 | 36.158 | 61.874 | 78.726 | 57.547 |
| quora | mmap | 10000 | 26.56 | 36.070 | 61.604 | 78.229 | 72.625 |

热轮（pass 2）：du 为完整 2,000 条；Quora 为固定 1,000 条样本。

| 数据集 | 读取路径 | QPS | P50 ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|---:|
| du | rocksdb | 2001.26 | 0.378 | 1.352 | 2.514 |
| du | buffer | 170.34 | 5.440 | 10.289 | 14.581 |
| du | mmap | 173.65 | 5.313 | 10.103 | 13.903 |
| quora | rocksdb | 169.52 | 4.041 | 18.767 | 32.233 |
| quora | buffer | 26.26 | 36.389 | 62.686 | 85.147 |
| quora | mmap | 25.73 | 37.087 | 63.786 | 84.472 |

重开后的 custom reader 当前保守返回无限评分上界，对应 WAND/跳块剪枝关闭；测量保留此实际行为，没有预先全量 verify 后替换默认重开路径。尚未量化该选择占性能差异的比例。本轮是 match_string 式 OR 查询；短语、布尔、过滤与候选集有功能验证，但未分别压测。

## 修复、评分兼容性与验证边界

基准过程中修复了重复页校验、普通多词查询的范围额度不足，以及 OR 分数依赖 WAND 遍历顺序的问题。页缓存有固定容量并计入共享池；范围额度变更不预分配内存。OR 现按固定子句顺序累加 FP32，保持 BM25 单词贡献、boost、统计及 collector 同分比较器，但**浮点结合顺序有可观测变化**。本报告的 RocksDB 基线是当前共用执行层的 legacy 路径，不能据此声称与旧二进制分数低位或近同分顺序完全一致。此前 v4 的兼容性诊断保留于桌面，不作为正确 BEIR 输入下的最终指标。

fts-bench 原有 JsonValue 字符串未解码，Unicode 转义与换行曾作为字面字符参与分词。新增工具内解码器正确处理代理对及单层转义，没有改生产 JSON 库；4 项正式测试直接运行通过，6 组 Unicode CLI 对照精确一致后才重跑全部索引和查询。

近期正式构建含 `_zvec` 成功，测试按用户要求直接执行编译产物。container 26、field 16、generation 13、publication 常规 40，以及 conjunction 10、reader 2（162 组差分）分批通过。最新 indexer 批次普通 92 项通过，4 项 death test 的 re-exec 遭 SIGKILL，批次退出 1，未记为全绿。另有此前独立直接启动的 30 项实际退出/恢复检查通过，具体阶段及对应构建见 progress。最新 33 个 tracked 文件改动范围、46 个新 C++ 文件的 clang-format 检查与 git diff --check 通过。

### Standards 复审

本轮输入修复及工具 CMake 增量无可行动问题。工具和测试目标受 BUILD_BENCHMARKS 控制，移动 CI 关闭该选项；已核对配置，未宣称跨平台编译运行通过。

### Spec 复审

合法 JSON 转义与代理对未发现阻断问题。一项非阻断限制：工具 helper 不额外验证原始非 ASCII 字节的全部 UTF-8 非法编码；本次 6 个输入已严格验证合法。此前发现的删除快照和数值 DDL 文件生命周期问题已修复并通过回归，见 progress。

本轮增量复审：Standards 0 项；Spec 1 项非阻断输入校验限制。

完成基准后再次直接运行 4 个 indexer death tests，re-exec 仍全部收到 signal 9，42.9 秒后批次退出 1。失败没有被并入通过数量。[源码指纹](/Users/jiliang.ljl/Desktop/zvec-fts-v1/bench-results-v6/source-files-sha256.json) 记录本轮 80 个相关源码/构建文件，二进制 SHA-256 在 summary.json 中。

P1/P2 独立读写及查询、本期 P3/P4 的直接 seal、安全发布、混合合并、单段迁移、DDL 和 snapshot/备份已实施并完成本机关键验证。尚未完成 P5 全部启用门槛：真正 OS 冷缓存、merge 性能、完整读取字节/解码/cache-hit、工作区与临时磁盘峰值、全部目标平台与全部历史发行版拒绝行为。Windows 目录持久同步仍明确返回 NOT_SUPPORTED。**生产 `kCustomWritesEnabled=false` 保持不变，未创建 commit。**
