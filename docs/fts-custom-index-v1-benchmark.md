# FTS 当前实现与 RocksDB 性能对比

> 最新 mmap 热路径修复已完成：[原因与双机对照](fts-custom-index-v1-du-analysis.md)。191 原始 / 上一版 / 当前版 DU mmap 为 **1063.24 / 1039.65 / 1060.22 QPS**，8 MiB buffer 为 **317.24 / 406.67 / 410.90 QPS**；mmap 恢复到原始水平，buffer 保留约 29.5% 的收益。固定 CPU 补测结论一致。Mac 主矩阵因后台负载波动未通过 buffer 性能检查；随后六组配对补测未显示稳定回退，但不能据此给出精确收益。两轮数据及失败状态均保留，当前口径以链接报告最新节为准。

> 上轮相同源码的 RocksDB/custom du 矩阵仍保留于该报告：Mac QPS（RocksDB / mmap / buffer 8 / 64 MiB）为 **2075.55 / 2488.34 / 2161.21 / 2249.51**，191 为 **810.12 / 1059.10 / 316.81 / 908.75**。后续窗口优化及 mmap 修复未重测 RocksDB 或构建时间，不把不同轮次的绝对值拼成新倍率；以下构建和 Quora 主表也仍为历史基线。

> 历史基线说明：以下主表数据采集于 du 热点优化之前。后续已完成短位置编码、定宽追加和批量 doclen 三项通用优化，本机 seal 降低约 24%–26%、查询 QPS 提升约 7%–10%，详见 [优化前后对照](fts-custom-index-v1-du-analysis.md)。优化验证阶段当时未重测 RocksDB/191，现已补充上方 du 双机检索续测；以下旧表保留原始对照口径，不代表优化后 custom 的当前性能。

日期：2026-11-18。对应 `codex/fts-v1-hardening` 的当前未提交工作区，源码和二进制指纹见证据目录。主规范见 [implementation](fts-custom-index-v1-implementation.md)，功能和异常路径验收见 [progress](fts-custom-index-v1-progress.md)。

**当前 custom 的优势是索引体积及 Quora 检索；构建仍比 RocksDB 慢。长文档检索差异较小，不能将 Quora 的提速推广到全部语料。** 本报告使用同版本的新对照，不拼接旧 RocksDB 数据和新 custom 数据。

## 最终实现取舍

- 保留 8 字节私有临时行记录和完整 uint32 长度范围、presence、累计 TF 校验；默认写回缓存从 8 MiB 调为 **4 MiB（1024 × 4 KiB）**，按实际页数缩小并按需分配。
- 当前预算可覆盖 524288 个文档行，与原来 16 字节 / 8 MiB 的覆盖数相同，缓存数据预算减半；不存在 Quora 专用分支。writer 缓存与下文 reader pool 是两笔不同预算。
- 上一轮 Quora 双机的 4 / 8 MiB 对照性能接近，本机长文档的临时表也能完整装入 4 MiB。这是内存预算取舍，不是 4 MiB 对所有规模最优的证明。大表可能较 8 MiB 更早淘汰；超过缓存容量的性能差异尚未量化，2097153 行回归覆盖其正确性。
- RocksDB 编码、普通 mutable 写入、最终 custom 文件编码和评分逻辑不变。生产 custom writer 门禁仍关闭；此次使用显式 index benchmark 接口。

## 方法与口径

- 本机 Apple M6 / macOS 27 / Apple Clang 21 Release；191 为 Xeon Platinum 8369B / Linux / GCC 12.3 Release。同一机器上两种格式使用同一个二进制。RocksDB 指当前源码中的 legacy 存储路径，共用查询执行层，不是旧发行版二进制。
- 每种格式交错新建三次索引，分项分别取中位数；所有构建单线程。RocksDB write buffer 128 MiB、compression=0、block cache 8 MiB，实际值已核对。两种格式先走相同 mutable 插入，再分别执行 legacy seal 或 custom 转换、验证、同步、发布和清理。
- Quora：522931 文档、6032630 tokens，standard + lowercase；每模式全量 10000 条 test 查询一轮，固定 1000 查询同进程三轮。du-retrieval：100001 文档、21303021 tokens，jieba + lowercase；本机每模式完整 2000 条 dev 查询三轮，工具文件名 qrels/test.tsv 仍对应 dev split。
- 查询均为单线程 OR Top100，包含分词与检索，不含打开、输入加载和结果序列化；搜索使用索引持久化的分词配置。各 custom 读取模式使用同一份第三轮索引。
- 最终大小为 custom 文件或完整 RocksDB FTS 目录的逻辑字节，不含共同 DocID 映射、benchmark descriptor、原文、向量和数据库 WAL。峰值 RSS 是整个进程的观测值，不能与 cache 预算等同。mmap 由 OS 管理驻留页，不受 8 MiB pool 预算约束。
- 同机本任务计时串行，不与编译、测试、采样重叠；未绑核、未清 OS cache，不是独占机器或冷盘结果。三轮不足以声明统计显著性。

## 构建时间与索引大小

时间为秒，空间为 MiB（2^20 bytes），各列独立取三轮中位数。

| 机器 / 语料 | 格式 | ingest/flush | seal/sync | 总构建 | 最终索引 MiB | 构建峰值 RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| 本机 / quora | rocksdb | 9.200 | 0.520 | 9.743 | 118.040 | 1036.719 |
| 本机 / quora | custom | 10.164 | 3.145 | 13.215 | 61.512 | 1048.266 |
| 191 / quora | rocksdb | 20.933 | 1.356 | 22.299 | 119.129 | 1157.070 |
| 191 / quora | custom | 20.816 | 5.245 | 26.140 | 61.512 | 1159.762 |
| 本机 / du | rocksdb | 23.916 | 1.446 | 25.368 | 284.932 | 922.875 |
| 本机 / du | custom | 23.372 | 7.098 | 30.489 | 167.240 | 898.016 |

- Quora：custom 索引本机减少 **47.89%**、191 减少 **48.37%**；总构建时间分别增加 **35.64% / 17.23%**。额外成本集中在 seal；本机未修改的 ingest 中位数也有差异，不能将总耗时差全部归因于格式转换。
- du：custom 索引减少 **41.31%**，总构建增加 **20.19%**。临时行缓存减半并不代表整个构建进程的 RSS 减半。

三轮总构建时间范围（秒），用于保留测量波动：

| 机器 / 语料 | RocksDB 范围 | custom 范围 |
|---|---:|---:|
| 本机 / quora | 9.720–11.278 | 12.315–14.878 |
| 191 / quora | 22.143–22.460 | 26.085–26.167 |
| 本机 / du | 24.707–28.005 | 30.274–31.248 |

## Quora 检索性能

全量 10000 查询为单轮，延迟单位 ms；固定 1000 查询为三轮 QPS 中位数，两者查询分布不同，不能互相比作提速。

| 机器 | 读取方式 | 全量 QPS | p50 | p95 | p99 | 峰值 RSS MiB | 固定查询中位 QPS |
|---|---|---:|---:|---:|---:|---:|---:|
| 本机 | RocksDB / 8 MiB block cache | 161.44 | 4.300 | 17.864 | 31.549 | 87.72 | 167.76 |
| 本机 | custom mmap | 287.73 | 3.257 | 6.359 | 8.501 | 58.67 | 297.69 |
| 本机 | custom buffer / 8 MiB | 254.85 | 3.696 | 7.154 | 9.478 | 44.22 | 268.58 |
| 本机 | custom buffer / 64 MiB | 264.09 | 3.554 | 6.812 | 9.065 | 59.36 | 241.77 |
| 191 | RocksDB / 8 MiB block cache | 66.78 | 9.874 | 44.874 | 80.617 | 118.48 | 65.51 |
| 191 | custom mmap | 133.12 | 7.117 | 13.168 | 17.264 | 87.84 | 132.04 |
| 191 | custom buffer / 8 MiB | 72.24 | 12.910 | 25.140 | 33.494 | 82.19 | 70.54 |
| 191 | custom buffer / 64 MiB | 110.82 | 8.550 | 15.854 | 20.971 | 98.25 | 110.35 |

- 相对同机 RocksDB，全量 mmap QPS 为 **1.78× / 1.99×**；buffer 8 MiB 为 **1.58× / 1.08×**，buffer 64 MiB 为 **1.64× / 1.66×**（本机 / 191）。
- 本机 64 MiB 固定查询三轮为 263.96 / 223.43 / 241.77 QPS；8 MiB 为 264.25 / 269.11 / 268.58 QPS。不能据本机数据认定扩大 reader pool 有稳定收益。191 的 64 MiB 在全量和固定查询上均明显快于 8 MiB。
- 当前读取参数及实际 pool 预算已核对。`buffer_pool_used_bytes=0` 来自 indexer 析构后的统计，不能理解为运行期占用为零。报告中的 mmap 和 buffer 优势是当前完整实现对 legacy 的对照，不将历史版本间的查询变化归因于此次 writer 缓存调整。

## du-retrieval 长文档检索

仅本机，完整 2000 查询三轮，QPS 和延迟分位数分别取中位数；RSS 为该进程峰值。

| 读取方式 | QPS 中位数 | 三轮 QPS 范围 | p50 ms | p95 ms | p99 ms | 峰值 RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| RocksDB / 8 MiB block cache | 1936.93 | 1558.61–1974.44 | 0.388 | 1.382 | 2.504 | 320.22 |
| custom mmap | 2023.27 | 1789.79–2213.78 | 0.391 | 1.161 | 1.578 | 317.92 |
| custom buffer / 8 MiB | 1911.68 | 1906.05–1941.01 | 0.433 | 1.175 | 1.578 | 268.22 |
| custom buffer / 64 MiB | 2033.83 | 1997.52–2033.97 | 0.408 | 1.122 | 1.503 | 309.12 |

custom mmap / buffer 8 MiB / buffer 64 MiB 的 QPS 分别为 RocksDB 的 **1.04× / 0.99× / 1.05×**；mmap 和 8 MiB buffer 的范围与 RocksDB 重叠，64 MiB 的中位数优势约 5%。长文档样本支持空间收益，未显示 Quora 那样的大幅查询吞吐收益。

## 一致性、验证与限制

- 共 18 次构建：两机 Quora 各 6 次，本机 du 6 次。对应文档 ID 映射跨格式、轮次一致，Quora 映射和输入指纹跨机一致。custom 大小稳定为 Quora 64500320 bytes、du 175364320 bytes；RocksDB 大小受 SST 组织影响，按同机实测值比较。
- 共 **128000 次查询输出**（含重复轮次）的完整有序 DocID / 原始 FP32 bits 与各自机器的既有参考逐位一致，全部全量查询均有结果。Recall@100：本机 Quora 0.9480075398652996，191 Quora 0.9480504565319663，本机 du 0.7857886501265927；同机四种读取配置相同。跨平台既有浮点/同分边界差异保留，不声称跨机分数逐位相同。
- 本机重新编译 fts_bench、fts_field_test、fts_mutable_snapshot_test、_zvec；先 Ninja 编译目标，再直接运行绝对路径测试，field 27、mutable 4 项通过。相同 8 B / 4 MiB 行为上一轮已通过本机 116、191 118 项测试，覆盖超过缓存容量的淘汰/重读及 uint32 边界；本轮未重复未改变的完整故障套件。修改范围 clang-format、git diff --check 通过。
- 本机使用本轮编译二进制，191 复用上一轮已验证的 8 B / 4 MiB 冻结二进制；1383 个一方源码文件逐一核对，差异仅缓存说明注释，本机编译后也只有注释换行调整。每台机器上的 RocksDB/custom 均使用同一份二进制，输入和二进制指纹在实验前后核验。
- 尚未量化真正冷盘、多线程、merge 性能、短语/过滤组合、超缓存大表吞吐、临时磁盘峰值和旧 generation 共存峰值。本报告不替代生产启用门槛；生产 writer 仍默认关闭，未创建 commit。

## 证据与历史

统一入口：[summary.json](/Users/jiliang.ljl/Desktop/zvec-fts-v1/current-vs-rocksdb-20261118-01/summary.json)；[分析程序](/Users/jiliang.ljl/Desktop/zvec-fts-v1/current-vs-rocksdb-20261118-01/analyze.py)。本机 `results/commands.json`、191 `remote191/results/commands.json` 保存正式命令；原始计时、查询结果、源码和输入指纹位于同一目录，191 原始证据也保留在其桌面同名目录。

[早期 v6 报告](fts-custom-index-v1-benchmark-v6.md) 保留历史输入修复、空间分解和当时的性能数据，其旧性能结论不代表当前实现。此次旧结果仅用于逐位一致性参照，没有参与性能统计。
