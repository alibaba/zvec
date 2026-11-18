# du-retrieval：FTS 构建与查询优化诊断

日期：2026-11-18。本机 Apple M6 / macOS，8 B 临时行、4 MiB writer 缓存。本文先记录已落地优化及实测结果，再保留此前的采样诊断。RocksDB/custom 的上一轮完整对照见 [基线报告](fts-custom-index-v1-benchmark.md)。

## 最新：mmap 命中路径修复，保留 8 MiB 优化

2026-11-18 续作。**191 的 mmap 吞吐已恢复到物理页优化前的水平，8 MiB buffer 的收益保留。Mac 补测未显示稳定回退，但后台负载波动大，不能给出可信的精确提速比例。** 本节对应最终 `candidate-g`；后文 `candidate-v4` 为本次修复前版本。

### 原因与修复

问题集中在共享 reader 的 CPU 路径。上一版虽然保留了 mmap 的逻辑窗口，却把物理页处理及相关状态加入了同一个读取实现；命中读取仍经过常规校验及跨文件的视图访问调用。对 mmap 这种主要消耗 CPU 的路径，这些细小开销和生成代码布局会影响吞吐，不能只检查是否多做了 I/O。

- 独立计数诊断中，原始版、上一版及提取快路径后的中间版 F，完整 2000 查询均为 **43237 次打开、6571821 次读取、5619765 次窗口命中、952056 次未命中**。85.51% 的读取命中已有窗口，回退没有伴随查询工作量或窗口未命中数增加。这组插桩不是最终 G 的计时结果。
- mmap 每次创建 cursor 还多调用一次 `storage_->get()`，产生无用 segment 句柄。现在仅 buffer 初始化物理页偏移时调用；container 打开时仍验证各 section 的存在和长度。单独删除该分配不足以恢复性能，因此没有将其宣称为唯一根因。
- 最终将 `FtsSectionReader::read()` 的短命中路径和 `FtsReadRange::view()` 放入内部头文件，未命中进入独立的 `refill()`。常用缓存状态放在新增对齐状态之前。GCC 最终二进制没有独立的 `read/view` 调用符号；复杂窗口处理留在 `refill`。没有强制内联属性、平台分支、新缓存或语料特判。
- 命中窗口已被 section 边界约束，仍检查相对范围和 **65536 B 单次请求上限**；零长度、非法和超长请求回到原有校验，错误优先级不变。未命中先释放旧 lease，再申请预算；读取失败继续返回错误。新增回归确保“缓存窗口大于请求上限”也不能绕过超长检查，失败后原缓存仍可复用。
- **物理页对齐、跨页计费和单页 pin 优化全部保留**，buffer pool 仍为 8 MiB。RocksDB、文件格式、评分和 writer 配置没有变化。

单独去掉句柄、调整成员位置或只提取慢路径都未完整恢复；最终的小命中路径内联通过 191 的复测。现有证据支持共享 CPU 读取路径的解释，未取得可用硬件性能计数，不能把剩余成本精确归因于某条指令或某类 cache miss。

### 191：同轮三版本对照

版本定义：**原始版**为物理页优化之前；**上一版**为物理页优化后的 `candidate-v4`；**修复版**为当前 G。每种读取模式交错 3 个独立进程 × 3 遍完整 DU 查询，全部 9 遍取中位数，不丢弃首遍。单线程 OR Top100，延迟单位 ms。

| 模式 | 版本 | QPS | p95 | p99 |
|---|---|---:|---:|---:|
| mmap | 原始版 | 1063.24 | 1.959 | 2.671 |
| mmap | 上一版 | 1039.65 | 1.996 | 2.726 |
| mmap | **修复版** | **1060.22** | **1.960** | **2.676** |
| buffer 8 MiB | 原始版 | 317.24 | 5.903 | 9.257 |
| buffer 8 MiB | 上一版 | 406.67 | 4.580 | 6.936 |
| buffer 8 MiB | **修复版** | **410.90** | **4.465** | **6.822** |

- mmap 相对上一版 **+1.98%**，相对原始版 **−0.28%**；最终 9 遍范围 **1056.44–1063.95 QPS**。
- buffer 相对上一版 **+1.04%**，相对原始版 **+29.52%**，p95 相对原始版降低 **24.36%**。本轮未重采 I/O trace，后文减少 33.89% 读取请求的数字仍属于上一版物理页优化的独立诊断。
- 另在相同 CPU 50 上，各版本再跑 3 进程 × 3 遍。mmap 中位数为 **1054.32 → 1034.36 → 1058.53 QPS**；修复版相对原始版 **+0.40%**、相对上一版 **+2.34%**。该补测与主矩阵分别统计；修复版第三进程有 **557.54 / 783.99 QPS** 两个慢轮次，均保留，没有剔除，也不能把绑核等同于独占 CPU。
- 事先使用“mmap 至少达到原始版 99%、buffer 至少达到上一版 99%”作为实用检查线；191 主矩阵和固定 CPU mmap 均通过。该检查线不是统计显著性结论，也不保证每个延迟分位数都改善。

### Mac：保留受干扰结果，另做相邻配对

主矩阵使用相同的 3 进程 × 3 遍方法，但出现明显运行期波动：原始 / 上一版 / 修复版 mmap 中位 QPS 为 **1979.78 / 2306.24 / 2222.09**，buffer 为 **1928.04 / 2081.51 / 1969.70**。修复版 buffer 范围 **1344.99–2114.99 QPS**。**该轮 buffer 相对上一版 −5.37%，性能检查未通过，原始结果和失败状态完整保留。**

观察到 WebKit、音视频、WindowServer 和系统索引进程同时占用 CPU；温控查询未报告告警，不能据此证明没有频率变化。未停止用户进程。为缩短版本间的时间间隔，随后单独做六组成对完整查询，每进程一遍，交替先后顺序；每次 buffer 测量前仍用同一个原始 mmap 程序预热文件页。下表不与主矩阵合并。

| 对照 | 版本 | QPS 中位数 | p95 ms | 六遍 QPS 范围 |
|---|---|---:|---:|---:|
| mmap | 原始版 | 2235.94 | 0.993 | 1533.87–2371.79 |
| mmap | 修复版 | 2406.25 | 0.936 | 2362.97–2422.52 |
| buffer 8 MiB | 上一版 | 2158.97 | 1.021 | 1816.23–2211.81 |
| buffer 8 MiB | 修复版 | 2204.44 | 0.991 | 1722.40–2241.44 |

按每组“修复/对照”计算再取中位数，mmap 为 **1.0648×**、buffer 为 **1.0134×**，均超过预设检查线；buffer 各组比值仍从 **0.826× 到 1.233×**。**补测支持没有稳定性能损失，但不能据此宣称 Mac 必然提升 6.48% 或 1.34%。** 第一轮失败没有被覆盖，Mac 的精确吞吐收益仍需低干扰环境确认。

### 跨语料、正确性与证据

- Quora 固定 1000 查询子集，每版/模式一个进程三遍。191 mmap **149.52 → 151.88 QPS（+1.58%）**；buffer **93.95 → 92.39（−1.66%）**，其 p95 **19.771 → 19.770 ms**，p99 **27.356 → 27.851 ms**。明确保留这组小幅下降，不声称所有语料所有指标都更好。Mac 同轮 mmap **277.75 → 309.30**、buffer **204.91 → 281.89 QPS**，同样受上述负载影响，不用作精确收益结论。
- 两机各先 Ninja 编译具体目标，再通过绝对路径执行 container 28、field 28、format 22、reader 3、mutable snapshot 4，均 **85 项通过**。Mac `_zvec` 同步重建；两机 1396 个一方源码文件逐一匹配最终快照。
- 主矩阵 **240000**、191 固定 CPU **54000**、Mac 成对补测 **48000**，合计 **342000 次正式查询输出**；另有 **44000 次预热输出**。全部有序 DocID / 原始 FP32 bits 与各自机器的原有参考逐位一致。跨平台原有浮点及同分边界差异没有改变，不声称跨机逐位相同。
- 所有临时插桩已移除；本次五个源码/测试文件修改范围 clang-format dry-run、`git diff --check` 通过。没有重新构建索引、重测构建时间或 RocksDB 吞吐，也没有改变生产 writer 门禁或创建 commit。

最终证据：[主矩阵与固定 CPU 汇总](/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-fix-final-20261118-01/summary.json)、[Mac 配对汇总](/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-fix-final-20261118-01/paired-mac/summary.json)、[主矩阵核验](/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-fix-final-20261118-01/analyze.py)、[配对核验](/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-fix-final-20261118-01/analyze-paired.py)。`final/` 为当前 G 源码和程序，`results/`、`remote191/results/`、`remote191/diagnosis/`、`paired-mac/` 保存完整命令和输出；远端原始目录为 `/home/jiliang.ljl/Desktop/zvec-fts-v1/mmap-fix-final-20261118-01/`。

诊断与中间变体另存于 [mmap-fix-20261118-01](/Users/jiliang.ljl/Desktop/zvec-fts-v1/mmap-fix-20261118-01)。该目录的 `final/` 是中间 F，**不代表当前 G**；其计数、单变量实验和早期性能数据不混入本节最终计时。

## 上轮：固定 8 MiB 的物理页窗口优化

2026-11-18 续作。**不增加 buffer pool，du QPS 中位数 Mac 提升 7.71%、191 提升 30.41%；191 物理读取次数减少 33.89%。** 本节对应最终 `candidate-v4` 源码，后文双机矩阵为此次改动之前的基线。控制组仍观察到 mmap QPS 小幅下降（Mac 0.92%、191 2.39%），不能宣称所有读取方式都提速。

### 为什么 191 的 8 MiB 特别慢

1. **主要是重复缺页读取，而不是 doclen 或锁等待。** 旧版完整 2000 查询一次运行中，数据通道发生 48999 次 4096 B `pread64`，请求约 191.40 MiB；独立缓存诊断记录 49053 次 miss、47901 次淘汰，bypass、singleflight wait 和异步提交均为 0。即使总命中率为 98.25%，剩余约 4.9 万次同步读取仍然昂贵。读取主要落在 postings、skips 和 term_meta，doclen 只有约百次。
2. **FTS 逻辑窗口没有对齐底层物理页。** 原实现按 section 内的 `layout_granule` 划窗，但 section 起点通常不是物理页边界。例如起点偏移 64 B 时，一个逻辑 4 KiB 窗口跨两个物理页。191 原生页为 4 KiB，原有窗口经常触发两页读取及连续内存复制，无法走后端单页 pin 路径；Mac 原生页为 16 KiB，错位造成的跨页比例较低。
3. **8 MiB 不是 8 MiB 纯数据页，且两机文件缓存行为不同。** 191 诊断记录页表元数据为 2 MiB，结束快照驻留 1152 页（4.5 MiB）；该快照不是硬性驻留上限。Linux 数据通道使用 `O_DIRECT`，重复查询仍会为被淘汰的页付出 I/O 成本。Mac 保留 `F_NOCACHE`，但实验表明既有文件页缓存状态仍显著影响结果，不能把此前 Mac/191 的差距全部解释为 CPU 或 pool 容量差异。

Mac 的缓存状态专门做了控制：相同旧程序，本轮最初未主动预热时约 **671 QPS**，逐次 mmap 预热后约 **2157 QPS**。这不是冷盘对照，也不能据此断言 `F_NOCACHE` 与 Linux `O_DIRECT` 具有相同观测行为。最终表中每次 Mac 8 MiB 测量前，都先用相同旧程序 mmap 完整查询一遍；每次正式测量仍是新进程、空的进程内 pool。最初未预热的中间候选结果只作为诊断，不混入最终版数据。

### 最终实现

- buffer 读取窗口按**原生物理页**对齐，并裁剪到 section 边界，继续只保留每个 cursor 的一个有界窗口。直接复用既有单页 pin 路径，减少跨页复制和无用读取，不添加第二级缓存、预取或语料专用分支。
- range 的 buffer 临时空间计费按实际物理页交叉判断。一个原生页内跨多个 checksum chunk 的窗口不再被误算为额外连续副本；保留原有保守固定开销、并发预算、失败返回及 generation 生命周期。
- mmap 保留原有 section 相对窗口和计费规则；checksum 格式与显式验证、RocksDB 写入、评分逻辑和最终索引编码均不变。cache 上限仍为 8 MiB，未关闭直接 I/O，也未增加 writer 缓存。

### 同机前后对照

du 为完整 2000 查询，100001 文档、21303021 tokens；单线程 OR Top100，含分词与检索。两机 du buffer 8 MiB 和 mmap 均为交错 **3 个独立进程 × 3 遍，共 9 遍**，第二组反转版本顺序。QPS 与各延迟分别取中位数，不丢弃首遍。Mac 为 Apple M6 / macOS 27 / Clang 21，191 为 Xeon 8369B / Linux 5.10 / GCC 12.3。相同的 1396 个一方源码文件；同机新旧程序读取同一索引，输入及正确性参考沿用下节已核验的跨机数据。

| 机器 | du buffer 8 MiB | QPS | p50 ms | p95 ms | p99 ms | 9 遍 QPS 范围 |
|---|---|---:|---:|---:|---:|---:|
| Mac，逐次预热文件页 | 优化前 | 2149.83 | 0.397 | 0.994 | 1.381 | 2079.23–2185.32 |
| Mac，逐次预热文件页 | 优化后 | **2315.60** | 0.365 | 0.945 | 1.292 | 2293.08–2329.67 |
| 191，直接 I/O | 优化前 | 317.34 | 2.874 | 5.859 | 9.100 | 314.46–322.61 |
| 191，直接 I/O | 优化后 | **413.84** | 2.207 | 4.499 | 6.735 | 402.59–419.12 |

- Mac QPS **+7.71%**，p95 **−4.93%**；191 QPS **+30.41%**，p95 **−23.21%**。
- 191 首遍 QPS 中位数 **315.98 → 404.06**，后续遍 **317.98 → 416.79**，提升并非只出现在预热后的进程内查询。Mac 对应 **2168.63 → 2317.95 / 2137.81 → 2315.58**。
- 8 MiB 下进程峰值 RSS 范围：Mac **268.17–268.33 → 269.39–269.47 MiB**，191 **187.07–187.30 → 180.05–187.22 MiB**。RSS 含分词器、分配器等整个进程，不能与 pool 预算等同，也不能据此宣称整体内存完全不变。

控制组及跨语料检查如下。du mmap 为 9 遍；du 64 MiB 和 Quora 固定 1000 查询子集各为一个进程 3 遍，不将子集数字推广到全量 Quora。

| 机器 | 控制组 | QPS 前 → 后 | 变化 | p95 ms 前 → 后 |
|---|---|---:|---:|---:|
| Mac | du mmap | 2489.62 → 2466.77 | −0.92% | 0.905 → 0.911 |
| 191 | du mmap | 1063.11 → 1037.72 | **−2.39%** | 1.956 → 2.002 |
| Mac | du buffer 64 MiB | 2222.61 → 2318.46 | +4.31% | 0.991 → 0.948 |
| 191 | du buffer 64 MiB | 903.97 → 916.43 | +1.38% | 2.222 → 2.205 |
| Mac | Quora 子集 buffer 8 MiB，逐次预热 | 297.23 → 320.91 | +7.97% | 6.187 → 5.695 |
| 191 | Quora 子集 buffer 8 MiB | 76.92 → 93.13 | +21.08% | 24.626 → 19.845 |

**仍有代价和容量限制。** Mac mmap 的范围重叠；191 mmap 本轮前后范围不重叠，保留这个约 2.4% 的回退，不将其当作已消除的噪声。单独还原窗口或计费的诊断说明公共读取路径也有成本，但尚未精确拆出剩余开销。191 优化后 8 MiB 仍明显慢于同轮 mmap 和 64 MiB，不能说已解决所有容量导致的重复 I/O。64 MiB 控制组不属于本次实现的缓存配置调整；191 的 64 MiB 首遍仍明显慢于后续遍。

### I/O 证据与验证

最终版另行采集一次完整查询的 `strace -f -T -s 0 -e pread64`，只记录调用信息，不抓取数据内容。该采样独立于正式计时：

| 191 数据通道 | 优化前 | 优化后 |
|---|---:|---:|
| 4 KiB 读取请求数 | 48999 | **32394** |
| 请求字节数 MiB | 191.40 | **126.54** |
| 不同读取偏移数 | 9616 | 8959 |
| trace 下读取累计耗时 s | 4.051 | 2.719 |
| postings 完整页读取 | 22040 | 13741 |
| skips 完整页读取 | 12540 | 6524 |
| term_meta 完整页读取 | 8537 | 8111 |

页横跨 section 时归入 mixed/other，按请求范围分类。读取请求数/字节数均减少 **33.89%**。trace 会扰动运行，累计耗时不能当作未插桩查询的精确时间占比。独立 `/usr/bin/time -v` 观测 user/system CPU 约 **2.88/0.23 → 2.79/0.19 s**，文件系统输入 **395960 → 260552**（512 B 单位），与主要减少 I/O 的判断一致；这些是整个进程指标，不冒充纯查询 CPU 时间。

- 两机各先 Ninja 编译具体测试目标，再执行绝对路径：container 28、field 28、format 22、reader 3、mutable snapshot 4，均 **85 项通过**；Mac `_zvec` 同步重建。
- 新增回归覆盖同一物理页内跨 checksum chunk 的复用、零缓存 bypass、mmap 旧边界、物理跨页、65536 B 最大请求、尾页、空/越界读取及紧 range 预算。此前 buffer 的重复读复现失败，修改后通过。
- 最终版 **180000 次正式查询输出 + 14000 次预热输出**，完整有序 DocID / 原始 FP32 bits 与各自机器既有参考逐位一致。跨平台仍遵守下节说明，不声称 Mac/191 原始分数逐位一致。诊断和中间变体不计入此数量。
- 未绑核、未清全机 OS cache、未与同机本任务编译或采样重叠；不是冷盘、多线程或统计显著性结论。本轮只改 reader，复用原索引，没有重新测量构建时间。生产 writer 门禁仍关闭，未创建新 commit。
- 本次五个源码/测试文件的修改范围 clang-format dry-run、`git diff --check` 通过。

证据入口：[summary.json](/Users/jiliang.ljl/Desktop/zvec-fts-v1/buffer8-opt-20261118-01/summary.json)、[统一核验程序](/Users/jiliang.ljl/Desktop/zvec-fts-v1/buffer8-opt-20261118-01/analyze.py)、[核验输出](/Users/jiliang.ljl/Desktop/zvec-fts-v1/buffer8-opt-20261118-01/validation-output.txt)。`candidate-v4/` 为最终本机快照，`results/v4-*` 和 `remote191/results/v4-*` 为最终命令、测试及完整查询输出，`remote191/diagnosis/v4-*` 为最终诊断。远端原始目录为 `/home/jiliang.ljl/Desktop/zvec-fts-v1/buffer8-opt-20261118-01/`。此前 `candidate`、`candidate-v2/v3` 和未带 `v4` 的结果保留为诊断，不代表当前版本。

后续若继续坚持 8 MiB，应分别评估页表元数据的占用和相邻缺页的合并读取；需同时量化 CPU、额外读取和跨语料收益，不能直接按 du 热点分配固定保留区。本次未实现这些候选。

## 上轮检索基线：物理页窗口优化之前

2026-11-18 续测。两机使用完全相同的 1396 个一方源码文件；Mac 复用上一节优化版已验证二进制，191 使用 GCC 12.3 重新编译。Mac 为 Apple M6 / macOS 27，191 为 Xeon Platinum 8369B / Linux 5.10。191 原有语料/查询文件与 Mac 指纹不同，本次将 Mac 输入复制到独立实验目录，逐文件核验 SHA-256 后再构建两种索引，没有使用原有不同字节的输入。

### 方法

- du-retrieval 100001 文档、21303021 tokens、完整 2000 条有 qrels 的查询，jieba + lowercase、单线程、OR Top100。
- 每种模式 **3 个独立进程 × 每进程 3 遍查询，共 9 遍**；模式顺序交错，第二组反转。下表 QPS、延迟分别取全部 9 遍中位数，不丢弃首遍。
- 同机 RocksDB/custom 使用同一二进制；Mac 复用已验证索引，191 用相同输入新建 RocksDB/custom 索引，各 custom 读取模式共用同一索引。构建只是测试准备，本轮不据一次构建比较构建速度。
- 计时含查询分词和检索，不含索引打开、输入加载和结果输出。OS cache 保留、未绑核；每台机器上的本任务计时均串行，不与该机编译或采样重叠。进程首遍不是冷盘试验。
- RocksDB block cache 8 MiB，custom buffer 分别 8/64 MiB。mmap 驻留页由 OS 管理，8 MiB 参数不限制其驻留内存；cache 预算也不等于进程 RSS。

### 吞吐与延迟

延迟单位 ms；“相对 RocksDB”仅比较同一机器、同一轮实验。

| 机器 | 读取模式 | QPS 中位数 | 相对 RocksDB | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|
| Mac | RocksDB / 8 MiB | 2075.55 | 1.000× | 0.367 | 1.293 | 2.386 |
| Mac | custom mmap | 2488.34 | 1.199× | 0.332 | 0.906 | 1.230 |
| Mac | custom buffer / 8 MiB | 2161.21 | 1.041× | 0.397 | 0.996 | 1.359 |
| Mac | custom buffer / 64 MiB | 2249.51 | 1.084× | 0.372 | 0.972 | 1.357 |
| 191 | RocksDB / 8 MiB | 810.12 | 1.000× | 0.910 | 3.349 | 6.693 |
| 191 | custom mmap | 1059.10 | 1.307× | 0.819 | 1.965 | 2.664 |
| 191 | custom buffer / 8 MiB | 316.81 | 0.391× | 2.880 | 5.887 | 9.099 |
| 191 | custom buffer / 64 MiB | 908.75 | 1.122× | 0.974 | 2.220 | 3.113 |

**mmap 在两机均有吞吐和尾延迟优势**：相对 RocksDB，Mac QPS +19.9%，191 +30.7%。Mac 两档 buffer 的 QPS 优势分别为 4.1% / 8.4%，但 p50 略慢于 RocksDB，不能说所有查询都更快。

**191 的 8 MiB buffer 是明显短板**：QPS 比 RocksDB 低 60.9%，p95/p99 也更高；64 MiB 为 8 MiB 的 **2.87× QPS**，达到 RocksDB 的 1.12×，仍低于 mmap。Mac 的 64 MiB 相对 8 MiB 仅快 4.1%。这说明该 du 样本在 191 上对 reader pool 容量更敏感；本轮没有做热点采样，不能仅凭计时把原因归结为某个 I/O 或缓存实现。writer 的 4 MiB 写回缓存不参与这些查询。

### 首遍、后续遍与波动

首遍列为三个独立进程第 0 遍 QPS 的中位数；后续列为第 1/2 遍共六次的中位数。RSS 为三个进程各自峰值的范围，单位 MiB，包含分词器及其他进程内存。

| 机器 | 模式 | 9 遍 QPS 范围 | 进程首遍 QPS | 后续遍 QPS | 峰值 RSS 范围 |
|---|---|---:|---:|---:|---:|
| Mac | RocksDB 8 MiB | 1598.78–2111.68 | 1624.32 | 2076.82 | 320.23–320.31 |
| Mac | mmap | 2019.26–2500.51 | 2395.20 | 2492.11 | 317.97–318.16 |
| Mac | buffer 8 MiB | 2152.45–2173.66 | 2161.21 | 2162.87 | 268.00–268.20 |
| Mac | buffer 64 MiB | 2186.85–2265.25 | 2221.15 | 2253.77 | 308.95–309.05 |
| 191 | RocksDB 8 MiB | 804.34–810.80 | 806.92 | 810.18 | 237.20–237.38 |
| 191 | mmap | 1033.88–1064.86 | 1057.21 | 1060.41 | 231.72–231.87 |
| 191 | buffer 8 MiB | 313.92–319.57 | 314.51 | 319.19 | 187.11–187.24 |
| 191 | buffer 64 MiB | 649.38–913.20 | 649.58 | 909.84 | 202.48–202.54 |

191 的 64 MiB pool 存在明显进程首遍成本，不能只展示后续遍来描述启动表现；8 MiB 则在重复查询后仍慢。9 遍中位数不构成统计显著性证明，本结果也不覆盖冷盘、多线程、短语或过滤查询。

### 一致性与证据

- 共 **144000 次查询输出**：每台机器 72000 次，均与该机器 RocksDB 的完整有序 DocID / 原始 FP32 bits 逐位一致。Mac 同时匹配此前参考；全部查询均有结果。两机 Recall@100 均为 **0.7857886501265938**。
- 跨平台不声称逐位一致：2000 条查询中，1187 条 DocID 排序相同、1964 条 Top100 集合相同；共同命中文档的分数最多相差 5 ULP。36 条集合不同查询涉及的替换项全部位于各自平台的末位同分边界，两机召回率仍相同。两种后端在各自机器上完全一致。
- 文档 ID 映射跨机、跨格式一致。输入、源码和二进制指纹在测试前后检查；191 先 Ninja 编译具体测试目标，再通过绝对路径执行 format / field / reader / container / mutable snapshot，共 **84 项通过**。Mac 对应同一源码的 84 项测试已在上一轮通过，本轮复用冻结二进制。
- 没有修改生产代码、缓存默认值或创建 commit。

证据：[summary.json](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-search-20261118-01/summary.json)、[统一核验程序](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-search-20261118-01/analyze.py)、[跨平台分数明细](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-search-20261118-01/cross-platform-score-details.json)。同目录保存 Mac 实际命令与全部输出，`remote191/results/` 保存 191 实际命令、编译/测试结果、计时和完整查询结果；191 原始证据位于 `/home/jiliang.ljl/Desktop/zvec-fts-v1/du-search-20261118-01/`。

## 已落地优化与实测

保留三项通用优化，无语料名称分支，也未增加缓存预算或配置项：

1. `FtsPositionsWriter` 仅在 TF > 128 时构造实际写出的 header/checkpoint。多块偏移仍取自 data append 之前，位置数量、顺序和 source 失败检查不变。
2. `AppendU32/U64` 在 4/8 字节栈缓冲中显式编码小端整数，再一次追加到字符串，保持原始 wire bytes。
3. `FtsPostingCursor::next_batch` 按已排序 DocID 的 128 行块分组，复用当前已验证 doclen 视图，减少逐 posting 的 `ReadLength` 调用及状态包装。保留整块缺失行检查、每个 posting 的 presence/TF 检查和错误持续传播；没有新增全量 doclen 数组。

### 构建：两项编码优化的组合收益

同机串行交错各三次新建索引，第二轮反转版本顺序；同一输入、单线程、8 B 临时行、4 MiB writer cache。各列分别取中位数，单位秒。两项构建优化一起计时，未单独测量各自贡献。

| 数据集 | seal 优化前 → 后 | seal 降低 | 总构建优化前 → 后 | 总时间降低 | 最终大小，前后相同 |
|---|---:|---:|---:|---:|---:|
| du-retrieval | 6.698 → 5.122 | 23.53% | 27.209 → 25.415 | 6.59% | 175364320 B / 167.240 MiB |
| Quora | 2.975 → 2.216 | 25.50% | 11.937 → 11.060 | 7.35% | 64500320 B / 61.512 MiB |

du seal 三轮范围从 6.651–6.734 s 降至 5.046–5.159 s；Quora 从 2.927–3.000 s 降至 2.196–2.218 s。ingest/flush 中位数分别为 du 20.459 → 20.280 s、Quora 8.943 → 8.854 s；共享导入实现未修改，这部分小幅变化不归因于优化。总时间下降幅度受导入占比限制。

构建使用 `build-opt` 冻结程序，随后 `final` 仅增加 `next_batch` 和其测试；custom seal 的写出/验证路径不调用 `next_batch`。源码快照明确区分两个阶段，未将查询优化算作构建收益。

### 检索：批量 doclen 的收益

优化前后程序读取同一份新构建索引，单线程 OR Top100。du 为完整 2000 查询、同进程三轮；Quora 为固定 1000 查询、同进程三轮。下表 QPS 及延迟分别取中位数，延迟单位 ms。

| 数据集 | 读取模式 | QPS 前 → 后 | QPS 提升 | p95 前 → 后 | p99 前 → 后 |
|---|---|---:|---:|---:|---:|
| du | mmap | 2255.71 → 2435.64 | 7.98% | 1.046 → 0.924 | 1.408 → 1.244 |
| du | buffer 8 MiB | 1970.98 → 2124.73 | 7.80% | 1.128 → 1.005 | 1.629 → 1.441 |
| du | buffer 64 MiB | 2072.01 → 2213.19 | 6.81% | 1.099 → 0.989 | 1.487 → 1.345 |
| Quora 子集 | mmap | 298.33 → 328.61 | 10.15% | 6.251 → 5.534 | 8.374 → 7.410 |
| Quora 子集 | buffer 8 MiB | 269.60 → 293.64 | 8.92% | 6.785 → 6.163 | 9.070 → 8.323 |
| Quora 子集 | buffer 64 MiB | 273.24 → 296.94 | 8.67% | 6.678 → 6.203 | 9.032 → 8.229 |

另测 Quora 全量 10000 查询各一轮：mmap **302.39 → 331.15 QPS（+9.51%）**。全量单轮与固定子集三轮分别报告。buffer 8 MiB 的顺序为新版→旧版，其他模式为旧版→新版；未绑核、未清 OS cache，未与本任务编译/采样重叠。这是本机吞吐对照，不是冷盘、多线程或统计显著性结论；本轮未重测 191 或 RocksDB 性能，不将上一轮绝对时间拼接成新的 RocksDB 倍数。

### 正确性与证据

- 12 次构建的文档映射均与既有参考相同，行数、token 数和最终 custom 大小一致。
- **74000 次查询输出**（含重复轮次）的完整有序 DocID / 原始 FP32 bits 与既有同机参考逐位一致；旧程序也能直接读取新 writer 产物。
- 最终版先 Ninja 编译各测试目标，再直接执行绝对路径二进制：format 22、field 28、reader 3、container 27、mutable snapshot 4，共 **84 项通过**；`fts_bench` 和 `_zvec` 同步重编译。新增测试覆盖 uint32/uint64 极值字节，以及匹配行缺失、非匹配缺失行长度损坏时批读取仍报错且保持错误状态。既有 TF 128/129、百万 positions、跨页/尾块和超缓存行表用例通过。
- 本轮改动范围的 clang-format dry-run 与 `git diff --check` 通过，未重排无关代码。
- RocksDB 编码、普通 mutable 写入、最终 custom 文件格式和评分语义不变。生产 custom writer 门禁仍关闭，未创建新 commit。

证据目录：[`du-opt-20261118-01`](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-opt-20261118-01)。其中 `baseline/`、`build-opt/`、`final/` 各保存二进制及 1396 个源码文件；`build-commands.json`、`search-commands.json` 保存实际命令和退出码，`build-summary.json`、`search-summary.json`、`validation.json`、`identities.json` 保存计时、逐位核验与指纹。

统一入口：[summary.json](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-opt-20261118-01/summary.json)；可重复核验：[verify.py](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-opt-20261118-01/verify.py)，已执行通过。

后续优先验证有界的小 term raw posting 缓冲，以及保持完整结构校验的词典首词比较。它们尚未实现，本轮收益不能归因于这两项；批量导入和执行策略选择仍应独立验证。

## 优化前的诊断结论与优先级

优先做不改变格式、预算和评分语义的局部优化：**省掉短位置列表不用的 header/checkpoint、减少定宽整数逐字节追加、批量读取 doclen**。随后验证有界的小 term raw posting 缓冲，以及词典首词比较的视图化。批量导入和 batch/WAND 选择有更大的影响面，应独立实验。

不建议继续按 du 扩大 writer 缓存，也不建议删除校验、改变 doclen 持久格式或直接把所有 custom OR 改回 WAND。

## 语料与性能基线

同版本工具读取公开 section 统计，term 数由 `term_meta.length / 112` 推导；当前 reader 也校验该等式。

| 指标 | du-retrieval | Quora |
|---|---:|---:|
| 文档数 | 100001 | 522931 |
| tokens | 21303021 | 6032630 |
| 平均 tokens / 文档 | 213.03 | 11.54 |
| 唯一 term 数 | 448882 | 93945 |
| 8 B 临时行表 bytes | 800008 | 4183448 |
| 最终 doclen section bytes | 415633 | 2173435 |
| term_meta section bytes | 50274784 | 10521840 |

du 词表约为 Quora 的 **4.78 倍**，平均文档 token 数约为其 18.5 倍。临时行表只有 **0.76 MiB**，对应 196 个 4 KiB 页槽；4 MiB / 8 MiB 上限下实际槽数相同。最终 doclen 约 **0.40 MiB**。缓存容量无法解释当前主要热点。

已有三轮中位数：custom ingest/flush **23.372 s**、seal/sync **7.098 s**、总构建 **30.489 s**；RocksDB 为 **23.916 / 1.446 / 25.368 s**。custom 最终索引 167.240 MiB，RocksDB 284.932 MiB。即使把整个 custom seal 时间减半，在其他阶段不变的假设下，总构建也只减少约 **11.6%**。

du 完整 2000 查询三轮 QPS 中位数：RocksDB **1936.93**，custom mmap **2023.27**，buffer 8 MiB **1911.68**，buffer 64 MiB **2033.83**。这些差异不能支持所有长文档都能大幅提速的结论。

## 采样方法与边界

- 复用上一轮已验证二进制；单线程、OR Top100、同一输入与索引。构建采样两次，查询 mmap / buffer 8 MiB / RocksDB 各运行完整 2000 查询 25 遍。
- 使用 macOS `sample`，间隔 1 ms。仅分析主线程，分别以 `insert`、`write_field`、`search` 栈为分母，去掉递归重复计数。下表是**条件化的栈样本占比**，不是精确 CPU 时间；包含子调用的比例会重叠，不能直接相加或当成预期提速。
- custom 构建附加较晚，捕获 seal；共享 insert 阶段取自 RocksDB 构建。它们调用同一 `FtsColumnIndexer::insert` 实现。没有把捕获窗口当成完整构建的时间分布。
- 有效样本：insert 8809、custom write_field 7229；查询 search：mmap 12866、buffer 13473、RocksDB 14735。分词/加载/结果写出不混入 search 分母。
- 五次成功采样的构建文档映射与参考一致，**150000 次查询输出**的完整有序 DocID / FP32 bits 与参考逐位一致。采样计时未用于正式性能比较。
- 首次后台启动停留在 dyld 后退出 137，未进入 benchmark；已隔离至 `failed-launch/`，不计入上述样本。

## 构建热点

### 共享导入

分母为 `FtsColumnIndexer::insert` 活跃阶段的 8809 个样本：

| 路径 | 包含子调用的样本占比 |
|---|---:|
| RocksDB Write / WriteImpl | 63.25% |
| MemTable::Add | 49.95% |
| HashSkipListRep::Insert | 36.92% |
| MergeCF | 31.66% |
| PutCF | 25.26% |
| TokenizerPipeline::process | 25.59% |
| JiebaTokenizer::tokenize | 23.95% |

中文分词对导入有实际成本，不能照搬 Quora 分词占比低的结论；但最大的路径仍是 RocksDB memtable 插入。当前已经每文档使用一个跨 CF WriteBatch，单纯把 WriteBatch 拼得更大不会消除逐项 memtable 插入。

中期候选是在**专用批量导入路径**内，以有界批次按 term 合并 bitmap / MAX_TF，减少 Merge 项；positions / TF / doclen 仍保留逐文档内容。必须明确批次可见性、错误返回、WAL/checkpoint 和统计提交时机，不能直接改变普通逐文档 insert 的完成语义。分词并行化也可作为独立实验，但应保持 DocID、位置和输入顺序确定，不改分词结果。

### custom seal

分母为 `FtsMutableSnapshot::write_field` 的 7229 个样本：

| 路径 | 样本占比 | 含义 |
|---|---:|---|
| WriteTerms | 91.22% | 主要成本在按词转换 |
| PositionsWriter::append_document | 23.64% | 包含编码、临时追加等 |
| 定宽整数 AppendU32 / AppendU64 路径 | 17.54% | 两个 helper 按字节 push_back |
| 位置 header/checkpoint 的定宽字段构造 | 11.12% | 上一行的子集，不可再相加 |
| raw posting 追加路径 | 7.44% | 包含 stdio、seek、写入等 |
| raw posting 回读路径 | 14.46% | 与上一行不重叠，合计约 21.9% |
| 行表 load_document_block | 0.18% | 不是当前主要成本 |

叶子样本中 `string::push_back` 及其跳板合计约 19.52%，`__write_nocancel` 为 18.76%。这不表示可以直接获得等比例提速。

**P0：短位置列表只构造实际会写出的内容。** [fts_positions.cc:56](../src/db/index/column/fts_column/fts_positions.cc#L56) 无条件构造 header，[87 行](../src/db/index/column/fts_column/fts_positions.cc#L87) 无条件构造 checkpoint record，但两者仅在 `tf > 128` 时写出。把构造放进相应条件分支即可消除短列表的无用序列化。多块路径的 record offset 必须继续取自 data append **之前**，不能移动代码时误用追加后的偏移。位置顺序、数量和 source 失败检查都保留。尚未统计 TF 分布，不能把 11.12% 全部算作可消除成本。

**P0：定宽编码从多次 push_back 改为一次 append。** [fts_format.cc:42](../src/db/index/column/fts_column/fts_format.cc#L42) 当前每个 uint32/uint64 追加 4/8 次。可用 4/8 字节栈缓冲按同样的小端规则编码，再一次追加；不写入原生结构体，也不依赖机器字节序。先单独验证完整编码字节一致，再与短位置列表优化叠加，以区分收益来源。

**P1：有界的小 term raw posting 缓冲。** 当前 `store_raw` 写入临时文件，`end_term` 又定位并回读：[fts_field_writer.cc:359](../src/db/index/column/fts_column/fts_field_writer.cc#L359)。可尝试按一个 posting block 容量缓存最多 128 个 RawPosting（约 4 KiB），小 term 直接消费，大 term 超限后回退文件。保持单例 inline、位置连续性、最终 DF/IDF 计算和错误状态；不能变成随 DF 无界增长的 vector。先记录 DF 分布及 spill 次数，不能假定 21.9% 路径成本都会消失。

## 查询热点

分母分别为 custom mmap / buffer 的 search 样本：

| 路径 | mmap | buffer 8 MiB |
|---|---:|---:|
| BatchDisjunction::next（含子调用） | 74.72% | 71.75% |
| ReadLength（含子调用） | 33.27% | 36.37% |
| ReadLength 自身 | 20.20% | 17.94% |
| BatchDisjunction::next 自身 | 18.17% | 15.80% |
| 词典 seek | 18.20% | 18.17% |
| DecodeDictionary | 14.67% | 13.13% |
| DecodePostings | 15.74% | 13.34% |

查询循环中约 99.2%–99.5% 的样本位于 search，查询分词不是主要热点。OR 检索不消费 positions 流，优化位置编码主要影响构建和空间，不应据此承诺 OR QPS 提升。

**P1：批量 doclen 访问。** [next_batch](../src/db/index/column/fts_column/fts_field_reader.cc#L475) 对每个 posting 调用 ReadLength，重复处理状态、行域、块序号和返回包装。可以按已排序 DocID 的 128 行块分组，一次装入已验证视图后批量取长度；保留每个实际 posting 的 presence / TF 校验，以及整块缺失行、尾块校验和借用视图生命周期。不新增全量常驻 doclen 数组，也不改最终文件格式。稀疏 posting 跨块较多时收益可能较小，必须按查询验证。

**P1：词典 fence 比较避免整块物化。** [seek_ge](../src/db/index/column/fts_column/fts_dictionary.cc#L681) 每次二分比较会调用 `block()` 解码整个词典块，然后只取第一个 term 比较。优先尝试首词视图、减少重复物化，选定块再展开。必须先确认查询的结构验证契约，保留校验和、边界、oversized term、含 NUL term 和坏索引错误；不以跳过必要验证换取速度。也可比较查询内有界块复用，暂不引入跨查询全局 LRU。

**实验项：稀疏窗口及 batch/WAND 选择。** 当前无 filter/candidate 的纯 term OR 在 custom 直接选择 batch 路径：[fts_column_indexer.cc:287](../src/db/index/column/fts_column/fts_column_indexer.cc#L287)；RocksDB 样本则主要在 WAND `DisjunctionIterator::next_doc_impl`（73.48%）。batch 会填充并遍历 4096 行窗口；可单独测试复用已有 presence bitmap 跳过稀疏空位，但 dense 场景可能增加开销。也应先强制两种执行器做同查询消融，记录 `sum(df)`、候选数、解码 posting 数和块数，证明收益后才考虑通用成本选择。不能从这次采样直接推断 custom WAND 更快，更不能按数据集名称切换。

## 实施顺序与验收

1. 分别测短列表无用构造、定宽 append 两个小改动，再测组合。覆盖 TF 1/128/129、多块 offset、source 少/多值与失败，校验磁盘字节和完整查询结果。
2. query 侧先比较批量 doclen 和词典首词视图；保留原校验，验证稀疏/稠密 presence、尾页、跨页、超长和二进制 term。
3. 取得 DF 分布后决定小 term raw 缓冲是否值得；取得每查询工作量后决定是否增加执行策略选择。
4. 再独立处理批量导入，避免将普通 RocksDB 写入行为变化混入 custom 局部优化。

每个候选都应同时跑 du 和 Quora，mmap / buffer，保持有序 DocID / FP32 bits 一致；构建交错三轮，采样与正式计时分开。候选的实测收益和代价优先于上述推测，不删除逐文档 TF 校验、positions source 检查或持久化校验。

证据：[summary.json](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-analysis-20261118-01/summary.json)、[分阶段栈汇总](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-analysis-20261118-01/profile-summary.json)、[语料与 section 统计](/Users/jiliang.ljl/Desktop/zvec-fts-v1/du-analysis-20261118-01/workload-stats.json)。同目录保存原始 sample、命令、二进制指纹和逐位验证；未创建 commit。
