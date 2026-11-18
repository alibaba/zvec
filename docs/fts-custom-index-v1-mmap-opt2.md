# FTS v1：第二轮 Mac mmap 优化

日期：2026-11-18。基于 [查询路径诊断](fts-custom-index-v1-query-path-analysis.md)，本轮落实 skip 和 doclen 读取优化。主规范仍为 `fts-custom-index-v1-implementation.md`；本轮没有修改评分、上界策略、磁盘格式或生产 writer 门禁。

## 实现

1. `fts_format_dictionary.cc` 抽出纯字段校验 `ValidPostingSkip`，encoder / decoder 共用；decoder 不再重新编码 48 字节字符串。保留 count、DocID、有限非负分数、非空范围、溢出、长度及版本检查。
2. `FtsPostingCursor::skip` 复用已经装入并验证的当前块记录，供相邻块的前后连续性检查使用；不增加无界缓存，不跳过新记录校验。
3. 新增 codec 内部 `DocLengthBlockView`。打开视图时检查完整编码长度、count、presence padding 和全部缺字段的零长度约束；所有 uint32 值对存在字段均合法，因此只需读取缺字段项做额外值检查。随后只读取所需行的 presence / uint32 长度，不展开 128 项数组、不复制整块。
4. 每个 posting cursor 仍由独立的 `FtsSectionReader` 持有原生范围，视图借用这些不可变字节。跨页时先取得和验证新视图；失败正常传播，不能继续使用失效的旧视图。单项读取显式处理小端编码和非对齐输入；字段 `document()`、显式 `verify()` 与查询游标共用该读取路径。

## 正确性与测试

- 新 doclen 视图测试先运行失败（exit 1），实现后通过。最终 format 22、field 20、reader 2，共 **44 项通过**；全部直接运行编译产物，Ninja 仅编译。
- 覆盖 count 1/7/8/9/127/128、全 presence、稀疏 presence、零 token 与缺字段、完整 uint32 值域、非对齐、尾块、所有截断位置、尾随字节、非法 padding、未请求的缺字段非零长度、单项越界和默认空视图。
- 新字段测试覆盖不同游标独立跨页、父 reader 释放、重复读取、128 行边界及尾行。已有查询差分、并发游标、资源不足和错误传播用例通过。skip 的 golden fixture 保留，并补充 decoder 的非法字段和范围溢出检查。
- 最终 `fts_bench`、`_zvec` 构建通过。7 个本轮修改文件的改动范围 `clang-format -n --Werror`、`git diff --check` 通过。没有修改 thirdparty / CMake，没有创建 commit。
- 正式性能对照的 **11 份**结果全部与冻结 v6 参考逐字节一致；连同短回归、初测和独立采样，共 **17 份、33,200 行查询结果**通过有序 DocID / 原始 FP32 分数位检查。
- 全量 10,000 条查询的 Recall@1/5/10/100 分别为 0.5697081793 / 0.7847922670 / 0.8442518896 / **0.9480075399**，两条路径与参考完全相同。

## 性能结果

Mac（macOS 27.0、Apple M6），Quora 522,931 篇文档；同一份既有 immutable 索引、standard + lowercase、OR Top100、单线程。所有构建与性能运行串行，采样另行运行。只比较 **custom mmap 与 RocksDB**，没有运行 buffer pool 性能对照。第 0 轮不声称 OS 冷缓存。

固定种子 20261118 的 1000 查询，各执行三轮；旧 mmap 二进制也在本轮重新测量。中位数如下：

| 指标 | 优化前 mmap，同轮复测 | 优化后 mmap | RocksDB，同轮 |
|---|---:|---:|---:|
| QPS | 84.962 | **103.816** | 171.015 |
| 1000 查询耗时（秒） | 11.770 | **9.632** | 5.847 |
| p50（毫秒） | 9.387 | **7.426** | 4.003 |
| p95（毫秒） | 31.375 | **27.149** | 18.646 |
| p99（毫秒） | 49.254 | **42.384** | 32.158 |

热态吞吐提升 **22.19%**、耗时减少 **18.16%**。优化后 mmap 三轮分别为 **103.866 / 103.816 / 103.522 QPS**，收益可重复。

全部 10,000 条 test 查询，一轮完整对照：

| 指标 | 优化后 mmap | RocksDB |
|---|---:|---:|
| 总耗时（秒） | 95.401 | 56.677 |
| QPS | **104.821** | **176.438** |
| p50 / p95 / p99（毫秒） | 7.505 / 24.583 / 39.545 | 3.937 / 16.431 / 28.958 |
| Recall@100 | 0.9480075399 | 0.9480075399 |

mmap 全量吞吐仍为 RocksDB 的 **59.41%**，耗时约为 **1.68 倍**，尚未追平。与上一轮全量 86.497 QPS 相比约提升 21.2%；同轮旧版复测的三轮数据是本轮主要优化收益证据。

最初固定前 100 查询的诊断器也重新执行：RocksDB 571.565 ms、mmap 925.510 ms；结果精确一致，耗时比 **1.619**，通过原先 ≤2.0 的自设门槛（上轮为 2.056，失败）。该门槛只表示这一诊断目标达成，不代表已达到 RocksDB 性能。

本轮复用既有索引，未重测构建时间和索引大小；不能将上轮构建数据当成本轮测量。

## 热点变化与剩余工作

最终二进制另行做 10 秒采样，主线程 7,327 samples；不使用采样轮次的耗时作为正式性能结果。以下含子调用，项目之间可能包含，不能相加。

| 调用路径 | 上轮 mmap 采样 | 本轮 mmap 采样 |
|---|---:|---:|
| skip 解码 | 10.2% | **0.4%** |
| 其中 skip 重新编码 | 7.8% | **0 samples** |
| doclen 读取路径 `ReadLength` | 14.3% | **7.2%** |
| 整块 `DecodeDocLengths` | 8.0% | **0 samples** |
| posting `advance` | 28.2% | 23.5% |

新 doclen view 的打开 / 单项读取分别占 1.2% / 0.7%。重复编码和整块展开已退出查询热点，收益与前一轮诊断一致。

当前 `DisjunctionIterator::next_doc_impl` 含子调用占 84.6%，浮点环境检测与向上舍入合计约 22.1%，scalar bit-unpack 约 3.4%。本轮没有减少由宽松 term 上界带来的额外候选；下一步仍应优先研究有保守性契约支持的紧 term 上界和候选推进，避免直接启用已证明会放大 skip 读取的旧 block-max 路径。

## 复现材料和平台范围

桌面目录：`/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-opt2-20261118/`。

- `fts_bench`、`source-files-sha256.json`：最终二进制及 80 个源码 / 工具 / 测试文件的哈希。
- `before-source/`、`before-source-sha256.json`、`changes.patch`：本轮 7 个文件的原始快照和准确差异。
- `commands.sh`、`commands.json`、`summarize.py`、`summary.json`：串行复现命令和完整轮次数据。
- `comparison.json`、`all-result-verification.json`、各 `.tsv.pass*`：全量召回和所有精确结果验证。
- `red-test.txt`、各 `fts_*_test.txt`、`final-fts_format_test.txt`：测试闭环；`mmap-sample.txt`、`profile-summary.json` 为独立采样。

结论仅为本轮 Mac 实测；当时 Linux / Windows / iOS / Android 未验证。191 的现有密钥连接复查仍返回认证失败，因此本轮没有远端性能结果。

后续记录：用户将密钥加入新的 ssh-agent 后，191 已恢复连接；subagent 使用本报告冻结的 opt2 完成 Linux 单测与 Quora 对照，详见 [191 测试报告](fts-custom-index-v1-191-quora.md)。它不包含随后 Mac opt3 的块内推进优化。
