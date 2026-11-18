# FTS v1：Quora mmap 批量查询优化

日期：2026-11-18。主规范仍为 [实施设计](fts-custom-index-v1-implementation.md)。本轮在 opt3 工作区基础上完成 opt4/5/6，未创建 commit。Mac 与 191 最终结果均对应同一份 84 文件冻结 opt6 源码；Quora 上 mmap 的吞吐与 p50/p95/p99 均优于同机 RocksDB，完整有序结果及分数位保持一致。

## 改动及原因

此前 mmap 的主要负担并非磁盘读取量。正常打开只做有限校验后，custom 路径仍需独立读取 doclen，而且未验证的持久化评分上界不能用于跳过候选。保守上界使逐文档 WAND 花费大量时间在推进游标、维护 pivot 和计算上界上。opt2 Mac 采样中 WAND 含子调用占 84.6%，`advance` 占 23.5%，`fegetround` 与 `nextafter` 合计约 22%；这些含子调用占比重叠，不能相加。RocksDB 已有整条 posting value 内的 doclen，且采用原来的上界路径，因此文件更大仍可能查得更快。

本轮按三个小步骤调整：

1. **opt4：有界批量 OR。** `FtsPostingCursor::next_batch()` 返回当前最多 128 条 posting 的借用视图：DocID、TF、精确 doclen。游标持有解码数据和长度数组，推进后视图失效；查询层不接触 section、偏移或存储格式。`BatchDisjunction` 使用固定 4096 个 DocID 的窗口，按原 AST 词项顺序累计分数，再按 DocID 顺序交给原 Top-K 收集器。评分窗口为 16 KiB，presence 为 512 字节，另有每词固定大小的批次状态，词数仍受 1024 上限约束，不建立全段文档分数数组。
2. **opt5：缩短 doclen 热路径。** 将同块读取与跨块加载分开，使缓存命中路径更短；精确 little-endian 长度读取内联。保留越界、presence、TF ≤ doclen 和读取错误检查。
3. **opt6：减少窗口竞争分扫描。** 缺席行的分数为零，竞争阈值非负，因此 `score > threshold` 本身就能排除缺席行，无需再次检查 presence。`advance()` 仍执行定位，不使用竞争分剪枝，必须保留零分匹配。

批量入口只用于支持批量游标的 immutable custom reader、没有过滤器或候选集的平坦多词 OR，且子项都是没有 MUST/MUST_NOT/SHOULD 标记的 TERM。其他布尔、短语、过滤、候选集及 legacy 路径沿用既有执行器。WAND 仍然存在；本轮为适合的查询选择了批量执行方式。

BM25 保持 `(boost * IDF) * tf_norm` 的求值顺序和原 AST 累加顺序，重复词项的贡献不合并；Top-K 的严格大于替换规则不变。任何批次读取失败都会传播为搜索失败，不能返回部分 Top-K。没有修改磁盘编码、写入算法、统计语义、生产 writer 门禁或上界可信条件，也没有增加查询 SHA-256 或打开时的全量校验。

## Mac 最终结果

macOS 27.0 / Apple M6（12 CPU、16 GiB 内存），Apple Clang 21 Release；Quora 522,931 文档、6,032,630 tokens，standard + lowercase，OR Top100，单线程，cache 配置 8 MiB。RocksDB 实际 write buffer 为 128 MiB，compression=0。复用原有两份不可变索引，使用同一个冻结二进制切换后端。查询计时包含分词、AST 构建和搜索，不含打开、输入加载和结果序列化。正式计时串行执行，未叠加编译或采样。未清空 OS page cache，首轮不代表冷盘测试。

| 查询集 | 后端 | QPS | 耗时 s | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| 全量 10,000 | RocksDB | 176.015 | 56.813 | 3.933 | 16.521 | 29.261 |
| 全量 10,000 | mmap opt6 | **302.395** | **33.069** | **3.124** | **5.889** | **7.774** |
| 固定 1,000，三轮中位数 | RocksDB | 169.787 | 5.890 | 4.028 | 18.424 | 32.421 |
| 固定 1,000，三轮中位数 | mmap opt6 | **299.403** | **3.340** | **3.153** | **6.225** | **8.151** |

全量吞吐提高 **71.80%**，总查询耗时减少 **41.79%**；p50/p95/p99 分别减少 **20.57% / 64.35% / 73.43%**。固定 1,000 条三轮吞吐提高 **76.34%**。三轮 mmap QPS 为 299.314 / 299.797 / 299.403，RocksDB 为 166.334 / 171.758 / 169.787。表中的热态指标分别取中位数，不拼称某一个实际轮次。

全部 8 份正式结果共 26,000 行查询，与冻结 v6 RocksDB 参考的**完整有序 DocID 和原始 FP32 分数位**逐字节一致。全量 Recall@1/5/10/100 分别为 0.5697081793 / 0.7847922670 / 0.8442518896 / **0.9480075399**，两个后端及旧参考完全相同。

各阶段全量结果如下。每行是该阶段同二进制的两后端对照；阶段之间属于不同轮次，不当成严格隔离实验。

| 阶段 | mmap QPS | RocksDB QPS | mmap / RocksDB |
|---|---:|---:|---:|
| opt3：块内推进 | 116.687 | 174.020 | 0.671 |
| opt4：批量 OR | 215.333 | 179.472 | 1.200 |
| opt5：doclen 热路径 | 240.886 | 176.284 | 1.366 |
| opt6：竞争分扫描 | 302.395 | 176.015 | 1.718 |

本轮未重建索引，构建时间和大小不能算作重新测量。此前 Mac 的 custom 为 62.842 秒、64,500,320 字节，RocksDB 为 9.319 秒、123,773,738 字节；本轮查询优化没有解决 custom 构建更慢的问题。

## 191 最终结果

由 subagent 在独立桌面目录构建、测试。GCC 12.3 Release，Alibaba Cloud Linux 3，Xeon Platinum 8369B，64 vCPU；沿用与 [opt2 报告](fts-custom-index-v1-191-quora.md) 相同的 Quora 输入和只读索引。单线程 Top100、cache 配置 8 MiB，正式计时串行进行，未混入构建或采样。未绑核、未清空 OS cache，前置索引指纹读取会预热缓存，不声称冷盘或完全隔离机器测试。

| 全量 10,000 查询 | RocksDB | mmap opt6 |
|---|---:|---:|
| QPS | 67.660 | **131.028** |
| p50 ms | 9.729 | **7.232** |
| p95 ms | 44.108 | **13.413** |
| p99 ms | 79.041 | **17.545** |

全量吞吐提高约 **93.66%**，耗时从 **147.797 秒**降至 **76.319 秒**，减少 **48.36%**；p50 减少 **25.67%**。固定 1,000 查询三轮 QPS：mmap 为 132.087 / 132.291 / 131.920，RocksDB 为 65.881 / 65.902 / 65.924；中位数 **132.087 / 65.902**，mmap 约为 **2.004 倍**。

两后端的全量及热态所有轮次均与 191 opt2 RocksDB 参考逐位一致；Recall@100 仍为 **0.9480504565**。Mac 和 191 的跨平台浮点/严格同分边界差异沿用 opt2 报告的说明，不将跨平台差异混为同机后端不一致。

最终直接运行 field 24、reader 3、conjunction 10、format 22、bitpacked 124，共 **183 项通过**。运行前后 84 份冻结源码及两份索引的文件指纹一致。根 agent 还对下载回本机的 8 份正式结果、26,000 行查询与旧 191 参考重新做了逐字节核对，全部一致。运行时 dispatch 为 AVX2，pack/unpack wrapper 使用既有 SSE4.1/FastPFOR 路径；本轮没有新增 SIMD codec。

本轮未重建索引。此前 191 新建 custom 为 58.467 秒、64,500,320 字节，RocksDB 为 21.915 秒、124,914,604 字节：custom 小 **48.36%**，但构建耗时约 **2.668 倍**。读取优化未改变这一构建结论。

完整远端报告和原始证据在本机 `/Users/jiliang.ljl/Desktop/zvec-fts-v1/remote191-opt6-20261118/`，191 上对应 `/home/jiliang.ljl/Desktop/zvec-fts-v1/remote191-opt6-20261118/`。`report.md`、`results/summary.json`、全部原始结果和逐位比较记录应与本节一同保存；中间 opt4 的独立结果保留在 `remote191-opt4-20261118/`，不能替代最终 opt6 验证。

## 剩余热点

正式计时后单独进行 opt6 的 10 秒采样，主线程 7471 个样本。`ReadLength` 含子调用占 **44.28%**，posting load 占 **24.17%**，其中 posting 解码占 **19.13%**，ARM scalar unpack 占 **9.58%**。这些调用关系相互包含；比例上升可能来自其他工作减少，不能据此声称绝对耗时回退。内联后 `DocLengthBlockView::at` 不再作为独立调用出现，其工作仍在读取路径中。

该批量 OR 采样中 `fegetround` 和 `nextafter` 均为零。后续优化空间主要是独立 doclen 访问与 posting 解码，而不是恢复全文件预检或增加缓存副本。10 份诊断轮次的结果也与固定 1,000 查询参考逐位一致；诊断计时未计入正式性能表。

## 功能和审查

- opt4 先用 batch API 最小测试复现 `NotSupported`，实现后通过；随后直接运行 field 23、reader 3、conjunction 10 项。
- opt5 直接运行 format 22、field 23、reader 3 项，全部通过；`_zvec` 构建成功。
- 最终 opt6 直接运行 field **24**、reader **3** 项，全部通过；`fts_bench`、`_zvec` 构建成功。新增测试覆盖无限竞争阈值下 `advance` 仍返回零分匹配、当前目标与 EOF。
- 差分测试覆盖窗口边界、重复/缺失词项、大/零 boost、同分 Top-K、inline/short/tail posting、部分 advance 后取 batch、独立游标所有权、父 reader 释放及后半段损坏错误传播；功能测试包含两种读取模式，性能只比较 mmap 和 RocksDB。
- 本轮 14 个变更文件的修改范围格式检查及 `git diff --check` 通过。两个独立 subagent 分别审查 Standards 与 Spec，opt4/5 及 opt6 增量均无确认问题。静态审查不代替功能测试。

## 证据

本机桌面根目录 `/Users/jiliang.ljl/Desktop/zvec-fts-v1/`：

- `mmap-opt4-20261118/`：batch 最小失败/通过测试、首轮性能、原始热点采样、`performance-red.txt` / `performance-green.txt`。
- `mmap-opt5-20261118/`：doclen 改动、测试、同轮性能与原始结果。
- `mmap-opt6-20261118/`：最终 `fts_bench`、84 文件 `frozen-source/` 和源码/二进制指纹、`commands.sh` / `commands.json`、`summary.json`、`comparison-full.json`、`performance-green.txt`、全部 `.txt` / `.tsv.pass*`、`profile-summary.json`、审查增量补丁和格式检查。

源码与产物指纹用于辨识实验版本，没有加入产品打开或查询路径。Ninja 只编译；测试与 benchmark 直接运行。未修改第三方源码，未清理用户文件，未创建 commit。
