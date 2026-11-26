# FTS v1：Mac mmap 优化与 Quora 验证

日期：2026-11-18。仅记录实现与验证结果；主规范仍为 `docs/fts-custom-index-v1-implementation.md`。按用户最新要求，本轮性能报告只包含 Quora。DU custom 构建已收到 SIGINT 并退出 130，DU 输出不纳入本报告。

## 结果

Quora 522,931 篇文档、6,032,630 tokens；全部 10,000 条 test 查询，以及固定种子 20261118 的 1,000 条查询三轮热态结果，在 RocksDB、custom mmap、custom buffer 间 **有序 DocID 和原始 FP32 分数位完全一致**，也与未覆盖的 v6 基线一致。三种方式 Recall@100 均为 **0.9480075398652996**。

| 指标 | 原 custom mmap（v6） | 优化后 custom mmap | 本轮 RocksDB |
|---|---:|---:|---:|
| 构建总耗时（秒） | 116.793 | 62.842 | 9.319 |
| 最终索引（MiB） | 61.512 | 61.512 | 118.040 |
| 全量 10,000 查询 QPS | 26.556 | 86.497 | 174.115 |
| 热态第 3 轮 QPS | 25.735 | 87.148 | 168.196 |
| 全量 p50 / p95 / p99（毫秒） | 36.070 / 61.604 / 78.229 | 9.354 / 28.524 / 44.872 | 3.992 / 16.564 / 29.675 |

mmap 全量吞吐提升 **3.26 倍**，热态提升 **3.39 倍**；构建耗时减少 **46.2%**。custom 文件仍为 **64,500,320 字节**，比本轮 RocksDB 小 **47.9%**。mmap 全量吞吐仍只有 RocksDB 的 **49.7%**，构建仍慢 **6.74 倍**，不能宣称已追平。

buffer 共享同一 custom 编码，全量 **83.811 QPS**，热态三轮分别 **84.835 / 84.283 / 83.365 QPS**。mmap 三轮为 **87.139 / 87.266 / 87.148 QPS**，收益不是单轮偶然值。

## 具体调整

1. 按用户决定对齐向量索引默认策略：正常 open 和候选预打开只检查有界元数据、文件身份、头尾目录及必要长度；正常查询不计算数据页 SHA-256，不保存校验副本缓存。mmap 直接持有原生映射范围，buffer 复用原生 MemoryBlock。范围计费、越界、short read、解码错误、generation owner 与持久化同步保留。
2. 文件编码和摘要未变。完整 `verify_all()` / 字段 `verify()` 留作显式诊断；正常流程不自动执行，不再承诺普通查询检测全部数据位翻转。词典强指纹及打开时元数据摘要仍保留。
3. 普通重开 reader 使用当前 scorer 推导的保守 BM25 饱和上界恢复 WAND；严格 FP32 最近舍入下用四个有证明的舍入因子向上计算，溢出或不支持的浮点环境、非单位 boost 退化为无穷。不信任未经完整验证的持久化 block max，实际评分和同分 collector 不变。
4. `advance` 从下一块指数探测再局部二分，避免相邻跳转触碰远处 skip 页。doclen 数组先整体检查编码范围，再在小端主机批量复制，大端保留逐值解码。

## 诊断证据与剩余差距

初始 v6 主线程采样中 SHA transform 约占 38%；仅打开宽松 WAND 时，完整剩余 skip 列表二分把远处页反复拉入校验路径，SHA 上升至约 60%，固定 100 查询从 3.572 秒退化至约 6.818 秒。临近探测改后 3.230 秒；取消查询哈希后 1.463 秒；doclen 批量解码后 1.204 秒，所有阶段结果精确一致。

该 100 查询诊断器预设 custom/RocksDB 耗时比值 ≤2.0，最终为 2.056，仍未通过该自设性能门槛；不把改善误报为已经追平。

最终版本另行采样（不计入表中耗时）主线程共 2151 samples：`DisjunctionIterator::next_doc_impl` 自身 504（23.4%）、posting `advance` 自身 297（13.8%）、doclen 解码自身 137（6.4%），还有浮点环境/向上舍入和排序开销。SHA 不再出现在查询采样栈；scalar bit unpack 自身 38（1.8%）。这是采样归因，不是可相加的独立消融收益。后续应继续量化候选推进和上界使用，不能把剩余约 2 倍差距简单归因于缺少 ARM NEON bitpacking。x86 SIMD dispatch 和 ARM scalar fallback 未在本轮改动，191 性能尚未验证。

## 验证和实验条件

- macOS 27.0、Apple M6、12 个物理核；单 ingest/query 线程、standard + lowercase、OR Top100。两种索引使用相同输入，查询结果比较不放宽浮点容差。
- RocksDB 实际 write buffer 128 MiB、compression=0、block cache 8 MiB；custom 全局 buffer pool 预算 8 MiB，非进程 RSS 上限。结果中的 pool-used 是 reader 释放后的采样，不解释为查询峰值占用。
- 所有构建、正式搜索串行执行，直接启动编译产物，Ninja 仅编译；pass 0 不声称 OS 冷缓存。旧 RocksDB 全量 174.358 QPS，本轮 174.115 QPS，基线相近。
- 新校验契约 3 项先红后绿。容器 26、字段 19、reader 2、generation 13、普通 publication 40 分批直接执行通过；最后 doclen 改动后重新构建并跑格式 20、字段 19、reader 2 共 41 项通过。`_zvec` 构建、clang-format 和 whitespace 检查通过。已知受安全软件 AMFI 影响的 death re-exec 本轮未重跑；Linux/Windows/iOS/Android 未验证。
- 两轴只读复核未发现阻断问题。没有 commit，生产 custom writer 默认门禁保持关闭。

## 复现材料与远端状态

全部脚本、二进制、采样与结果在 `/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-opt-20261118/final/`：`commands.sh`（仅 Quora 的逐条命令）、`fts_bench`、`source-files-sha256.json`（80 个源码文件及二进制）、`summary.json`、`quora-comparison.json`、完整 TSV 与 `sample.txt`。输入三文件 SHA-256 已与 v6 冻结记录核对一致。旧 v6 数据未改动。

191（11.122.133.191）的公钥已被服务端接受，但私钥口令未加载，ssh-agent 无身份，BatchMode 重试认证失败。需要用户在本机终端执行 `ssh-add ~/.ssh/id_rsa` 解锁后才能继续；不要发送口令。此报告只有 Mac 实测，不包含推测的远端结果。
