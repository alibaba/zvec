# FTS v1：Quora mmap / RocksDB 查询差距诊断

日期：2026-11-18。对应 HEAD `0c63ea4ea936fcf4b6c352f7955e087143206baf` 及当前未提交实现。主规范仍为 `fts-custom-index-v1-implementation.md`。

本轮只分析 Quora 检索，只比较 custom mmap 与 RocksDB；不再运行 buffer pool 性能对照，也没有重测构建时间。实验没有改变生产代码、校验契约或磁盘格式。

## 结论

剩余约 2 倍差距来自两类工作叠加：自定义 reader 的 term 上界较松，导致更多候选、评分和 WAND 循环；同时它每次跳转和评分还承担额外的 skip 解析、文档长度块解码与范围对象管理。

RocksDB 路径先为每个查询词获取整条压缩倒排，持有 `PinnableSlice`，随后在连续内存中遍历；评分所需 TF / doclen 随 posting 块保存并按需解码。因此不能把本次结果理解成“RocksDB 随机读取比 mmap 快”。本轮是热态、单线程 CPU 查询路径的比较，未测 OS 冷缓存或并发伸缩。

较紧 term 上界的受控实验消除了约 **49% 的耗时差额**。但只打开现有 block-max 路径反而变慢：它成功跳块很少，却大幅放大 skip 二分和范围读取。ARM SIMD 不是当前证据支持的首要原因。

## 实验条件与有效性

- 与上一轮正式结果相同的冻结二进制、Quora 两份索引、fixed-seed 1000 查询、单线程、OR Top100、standard + lowercase；没有修改输入或索引。
- 原版二进制对两条路径各做一次 10 秒 `sample`。采样运行的耗时不用于消融比较。
- 临时诊断版增加线程本地整数计数器及三个上界实验开关。四个 1000 查询方案各执行三轮，以下耗时取中位数；计数在三轮中完全相同。
- 持久化上界实验仅用于这份已知正确的冻结索引，临时绕过 reader 的 `bounds_verified_` 使用门槛。它不是已落地优化，也不证明可以直接放宽生产上界契约。
- 所有 **22 份**结果文件（采样 8 份、1000 查询消融 12 份、100 查询无剪枝实验 2 份）均与 v6 参考的有序 DocID、原始 FP32 分数位逐字节一致。本轮没有重新统计 qrels 指标；同一查询结果意味着召回未变。
- 诊断版的计数器和分支存在开销，因此以下数字用于归因，不替代上一轮无插桩的全量 10000 查询正式结果（mmap 86.497 QPS、RocksDB 174.115 QPS）。

## 做了多少额外工作

以下是相同 1000 查询的一轮总计。

| 工作 | RocksDB | custom mmap | 比值 |
|---|---:|---:|---:|
| 耗时中位数 | 5.846 秒 | 11.724 秒 | 2.006× |
| 交给评分的候选文档 | 7,283,221 | 19,074,750 | 2.619× |
| term 评分调用 | 25,701,397 | 62,494,158 | 2.432× |
| term advance 调用 | 97,724,433 | 131,237,777 | 1.343× |
| WAND 循环次数 | 105,008,046 | 150,313,527 | 1.431× |
| block bound 查询 | 27,934,939 | 57,959,222 | 2.075× |
| posting 块装入 / 解码 | 3,693,081 | 3,760,462 | 1.018× |
| doclen 块解码 | 3,061,507 | 18,724,103 | 6.116× |
| 成功 block-max 跳块 | 325 | 0 | — |

custom 的 posting load 包含 38 次 inline，实际 `DecodePostings` 为 3,760,424 次；表中保留两边全部块装入次数。额外候选并没有导致成比例的倒排块解码，但显著增加了评分、doclen 处理和公共 WAND 开销。

### 1. term 上界过松

普通重开 custom reader 不使用未经显式完整验证的持久化 max，而是用当前 BM25 scorer 推导的饱和上界。它保守但较宽。RocksDB 使用封存时实际计算的 term max。两者在这批查询中的上界总和分别为 92,227.27 和 61,931.20，custom 约高 49%，会更早选中 pivot、放过更多低分候选。

代码入口：`fts_field_reader.cc` 的 `usable_bounds`、`conservative_max_score`、`max_score`；`fts_legacy_reader.cc` 的 `Term::max_score`。

| 同一份 custom 索引的实验 | 耗时中位数 | 候选文档 | skip 解析次数 | native range 获取次数 |
|---|---:|---:|---:|---:|
| 当前运行时饱和上界 | 11.724 秒 | 19,074,750 | 10,867,387 | 4,325,162 |
| 仅使用持久化 term max | 8.856 秒 | 7,287,914 | 11,038,143 | 3,772,765 |
| 使用持久化 term + block max | 13.453 秒 | 7,283,221 | 41,781,245 | 16,569,313 |

只收紧 term 上界使查询耗时减少 24.46%，消除与 RocksDB 差额的 48.78%；仍比 RocksDB 慢约 1.515 倍。因此“上界太松”成立，但不是全部原因。

### 2. doclen 的布局与访问粒度不匹配

RocksDB 的 packed posting 同块保存 doc IDs、TF、doclen，`decode_block` 只先解 doc IDs，直到评分才调用 `ensure_tf_decoded` / `ensure_dl_decoded`。解一个 doclen 块得到的是同一词的最多 128 个匹配文档长度。

custom doclen 在字段全局单独保存，按连续 128 个 DocID 分块。`ReadLength` 每换一块就复制长度数组、展开 presence、检查缺字段非零长度，再复制到游标缓存。各个 term 游标独立缓存；稀疏的命中文档会跨越很多全局块，一次解出的 128 个长度只有少数被使用，其他词也不能复用这份解码结果。

当前 custom 共解码 **2,396,497,150 个长度槽位**，对应 62,494,158 次 term 评分。收紧 term 上界后 doclen 块仍有 **11,322,900 次**，比 RocksDB 的 3,061,507 次多约 3.70 倍，说明这不只是候选更多造成的。

代码入口：`fts_field_reader.cc:50` 的 `ReadLength`、`fts_format_blocks.cc:349` 的 `DecodeDocLengths`；legacy 对照为 `posting/bitpacked_posting_list.cc` 的 lazy TF/doclen 解码。

### 3. skip 读取路径重新编码，制造分配与拷贝

`fts_format_dictionary.cc:312` 的 `DecodePostingSkip` 通过 `EncodePostingSkip(skip)` 复用合法性校验。编码器生成 48 字节字符串，逐字节追加整数；读路径因此承担了不需要的分配、扩容和释放。

当前 1000 查询发生 **10,867,387 次** skip 解码，也就有同样次数的重新编码。`advance` 探测、`load` 的当前/前一块校验会多次读取相同 skip；RocksDB 直接访问 pinned value 中的 `BlockMeta` 数组，不做这个重新编码过程。

这是可以独立消除的额外工作：把纯字段校验抽出来供 encoder / decoder 共享，保留全部错误检查，不生成临时编码字符串。此轮只定位，未修改实现。

### 4. block-max 上界查询的二分放大

`FtsPostingCursor::bound` 的区间缓存未命中时，从整个 skip 列表二分。它与 posting 推进共享 skip section 的单范围缓存，远处探测会替换当前范围。此前优化的临近指数探测仅作用于 `advance`，没有覆盖这条 bound 路径。

使用持久化 term + block max 的实验中，候选、评分、advance、成功跳块次数与 RocksDB **完全相同**，耗时却达到 13.453 秒。相对仅使用 term max，它只少了 4,693 个候选，却多了约 3,074 万次 skip 解析和 1,280 万次 native range 获取。Quora 这批查询只有 325 次成功 block-max 跳块，无法抵消查询上界本身的成本。

故不能把“缺少有效 block-max 剪枝”列为本批查询的主要瓶颈，也不能直接开启现有实现作为性能修复。

### 5. eager TF、范围管理及共有浮点上界运算

custom `DecodePostings` 在装入块时同时解 doc IDs 和 TF；legacy 仅在评分时解 TF。当前 custom 解码约 376 万个 posting 块，而 legacy 只有约 306 万块需要 TF/doclen 解码。custom 还检查每个 doc gap / TF，并通过返回值、缓存赋值复制解码块。

custom 的 section cache miss 还需查询 section、范围检查、reserve 额度、创建 lease、取得 native segment、移动 MemoryBlock。它是映射视图管理，并非每次发生磁盘 I/O。计数中的 `range_bytes` 是重复逻辑请求长度之和，不能解释为物理读取量、RSS 或唯一触碰字节数。

`ScoreUpperBound` 的 `fegetround`、`nextafter` / `nextafterf` 是两边共有热点。custom 因 WAND 循环更多而额外付费；单独优化这些函数会同时加速 RocksDB，不能把全部潜在收益归给自定义格式。任何优化必须保留浮点上界的保守性，不调整实际得分。

## 采样热点

原版二进制，RocksDB 主线程 7,348 samples，mmap 主线程 7,336 samples。以下均为**包含子调用**的占比；已去除同名父子帧重复计数，不同项目仍有包含关系，不能相加。

| 调用路径 | mmap | RocksDB |
|---|---:|---:|
| Disjunction `next_doc_impl` | 78.4% | 88.6% |
| posting `advance` | 28.2% | 19.3% |
| custom `ReadLength` | 14.3% | — |
| custom `DecodeDocLengths` | 8.0% | — |
| custom `DecodePostingSkip` | 10.2% | — |
| 其中 `EncodePostingSkip` | 7.8% | — |
| custom `FtsSectionReader::read` | 4.1% | — |
| custom `FtsContainerReader::read` | 1.7% | — |
| `scalar_unpack_uint32_128` | 2.8% | 7.9% |
| `fegetround`，含 stub | 9.2% | 14.5% |
| `nextafter` / `nextafterf`，含 stub | 8.8% | 13.9% |

共有 disjunction 循环、posting 跳转、doclen、skip 重新编码是当前应关注的热点。两条路径都在 Mac ARM 上使用相同的 scalar bit-unpack fallback；当前 custom 的 unpack 采样份额较小，单独增加 NEON 无法据此解释或消除全部约 2 倍差距。查询数据页 SHA 已不在该路径中。

另一个控制实验对固定前 100 查询强制无剪枝：两边完全相同的 28,816,360 个候选、48,831,593 次 term 评分、381,975 个 posting 块，RocksDB 1.290 秒、mmap 1.640 秒，约 1.27 倍。它支持“相同顺序遍历下仍有格式开销，WAND 跳转时差距更大”；该实验改变了执行工作量，只有一轮，不是正式查询性能或可直接采用的优化方案。

## 后续优化顺序

1. **去除 skip 解码中的重新编码**，抽取无分配校验；复用当前/相邻 skip 信息，避免重复解析。变更范围小，不需修改格式或放宽校验。
2. **按所需 DocID 读取 doclen**。当前 wire 已是定宽长度数组，优先做有边界检查的单项读取和 presence 检查，减少整块展开与复制；再评估查询内共享视图。保持 reader 生命周期和有界资源契约，不用扩大 buffer pool 掩盖成本。
3. **恢复有契约支持的紧 term 上界**。在既定有限打开校验策略下明确统计、参数、评分版本、boost 和浮点环境的可用条件，补齐禁剪枝差分；不通过恢复全文件扫描换取上界可用性。本轮诊断开关不能直接合入生产。
4. **优化 bound 游标后再评估 block-max**。优先利用已加载 block、单调目标和独立区间状态，避免全表二分干扰 posting 的 skip 缓存；不能破坏 SHOULD 的定位语义。
5. 在上述工作次数下降后，再测 lazy TF、解码块复制、公共浮点上界计算及 ARM SIMD 的收益。

后续每一步只跑 Quora，比较 mmap / RocksDB，先验证有序 DocID 和 FP32 位值一致，再报告性能。保留当前结果作为不可覆盖基线。

## 可复核材料与清理

桌面目录：`/Users/jiliang.ljl/Desktop/zvec-fts-v1/quora-path-analysis-20261118/`。

- `rocks-sample.txt`、`mmap-sample.txt`、`profile-inclusive-summary.json`：原版成对采样。
- `fts_bench_probe`、`fts_perf_probe.h`、`probe.patch`、`probe-source-sha256.json`：冻结的临时诊断工具。
- `commands.sh`：各实验逐条命令，必须串行执行；只含 mmap / RocksDB。
- 各方案 `.txt`、`.tsv.pass*`、`analyze.py`、`analysis-summary.json`：三轮耗时、操作次数和逐字节结果验证。
- `original-source/`、`original-source-sha256.json`、`restoration.json`：8 个临时插桩文件的完整备份和恢复证据。

临时插桩已逐字恢复；与上一轮冻结清单核对的 **80 个实现/工具源码全部一致**，`src/`、`tools/`、`tests/` 无诊断标记残留，`git diff --check` 通过。正常版 `fts_bench`、`_zvec` 和相关测试重新构建成功；直接运行 format 20、field 19、reader 2，共 **41 项通过**，输出见桌面 `restored-fts_*_test.txt`。未创建 commit。本轮没有测试 191，结论仅适用于本机 Mac。
