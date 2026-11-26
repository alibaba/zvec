# FTS v1：第三轮 Mac mmap 推进优化

日期：2026-11-18。主规范仍为 `fts-custom-index-v1-implementation.md`。本轮只调整 `FtsPostingCursor::advance` 的块内定位；没有修改磁盘格式、评分、上界可信条件或生产 writer 门禁。

## 改动与验证

第二轮采样中 `advance` 含子调用占 23.5%。它在已经解码的最多 128 条 posting 中，每次都对剩余范围调用 `lower_bound`，包括当前 posting 已满足目标、目标就在下一条的情况。

现在先检查当前 posting；需要前进时，从当前位置开始指数探测，取得近邻区间后再二分。已装入块的末尾 DocID 保证不小于目标，因此二分可以用该末尾作为已知上界。跨块查找、读取校验和错误传播沿用现有实现。

- 直接运行 `fts_field_test` 20 项、`fts_reader_test` 2 项，全部通过；覆盖独立游标、近跳/远跳/EOF、查询差分、并发读取和读取错误。
- `fts_bench`、`_zvec` 构建成功；Ninja 只编译，测试和基准直接运行产物。
- 改动范围 `clang-format -n --Werror` 与 `git diff --check` 通过。
- 11 份正式结果共 29,000 行查询，与冻结 v6 参考的有序 DocID 和原始 FP32 分数位逐字节一致。全量 Recall@100 因结果完全相同仍为 0.9480075398652996。
- 80 文件清单核对：相对 opt2 只有 `fts_field_reader.cc` 改变。此前用于诊断的 `ZVEC_FTS_SCAN_PROBE` 分支已完整移除，查询入口恢复 opt2 原文；扫描原型只保留于桌面独立实验目录，不是生产实现。

## 同轮性能

Mac，macOS 27.0；Quora 522,931 篇文档，standard + lowercase，OR Top100，单线程。复用先前两种不可变索引，缓存配置均为 8 MiB。只比较 mmap 与 RocksDB，正式计时期间没有构建或采样。第 0 轮不声称 OS 冷缓存。

固定种子 20261118 的 1,000 条查询，每种运行三轮，以下为中位数；上一版 opt2 mmap 也在本轮复测。

| 指标 | opt2 mmap 同轮基线 | 本轮 mmap | 本轮 RocksDB |
|---|---:|---:|---:|
| QPS | 103.081 | **114.204** | 168.425 |
| 耗时（秒） | 9.701 | **8.756** | 5.937 |
| p50（毫秒） | 7.461 | 6.822 | 4.053 |
| p95（毫秒） | 27.216 | 23.958 | 18.742 |
| p99（毫秒） | 42.624 | 38.670 | 32.523 |

mmap 热态吞吐提高 **10.79%**，查询耗时减少 **9.74%**。三轮分别为 113.270 / 114.204 / 115.294 QPS。

全部 10,000 条 test 查询，各一轮：

| 指标 | 本轮 mmap | 本轮 RocksDB |
|---|---:|---:|
| QPS | **116.687** | **174.020** |
| 总耗时（秒） | 85.699 | 57.465 |
| p50 / p95 / p99（毫秒） | 6.758 / 22.078 / 35.688 | 3.979 / 16.633 / 29.355 |
| Recall@100 | 0.9480075399 | 0.9480075399 |

mmap 全量吞吐为 RocksDB 的 **67.05%**，耗时约 **1.49 倍**，仍未追平。保守 term 上界导致的额外候选和公共 WAND 开销仍存在。本轮未重测构建时间或索引体积。

## 证据与平台范围

桌面目录：`/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-opt3-20261118/`。

- `fts_bench_advance`、`source-files-sha256.json`、`before-source/`、`advance.patch`：可复现二进制和源码快照。
- `commands.sh` / `commands.json`、`summary.json`、全部 `.txt` / `.tsv.pass*`：串行命令、每轮延迟和逐位一致性证据。
- `advance-field-test.txt`、`fts_reader_test.txt`、`advance-build.txt`、`final-build.txt`：功能与构建记录。

本轮代码只在 Mac 验证。同期 191 subagent 使用冻结 **opt2**，结果应单独引用，不能视为本轮块内优化的 Linux 验证。没有创建 commit。
