# zvec 旧矩阵距离路径 Linux x86 正确性与性能验证报告

执行日期：2026-11-06　工作目录：`zvec-x86-regression.04K3Yl`（保留，未清理）

## 1. 环境与版本

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon 6982P-C（KVM 虚拟机，8 核 16 线程，AVX-512/VNNI/AMX 可用） |
| OS / 内核 | Ubuntu 24.04，Linux 6.8.0-87-generic，glibc 2.39 |
| 编译器 | GCC/G++ 13.3.0（三版本相同） |
| CMake / Ninja / Python | 3.30.0 / 1.11.1 / 3.10.9 |
| 构建配置 | Release `-O3 -DNDEBUG`，LTO/OpenMP 关闭，AUTO_DETECT_ARCH=ON、ENABLE_NATIVE=OFF；基线 `-march=x86-64`，分 ISA 文件分别 `core-avx2 / icelake-server / sapphirerapids` 并运行时分派 |
| parent | 499eab892c750ecf60f314b3d24cccad888f5af2 |
| before | f7bd575f09b622a6dd387bcc266d2f3b53fd47c7 |
| fixed | 955ae841ddb8f10b2ae6e82a50720e2419741ccd |
| 工具包来源 | 分支 fix/legacy-matrix-distance-performance @ c21f19ce |
| 基准绑定 CPU | 逻辑 CPU 11（物理核 5，SMT 同胞 10 为最空闲核对） |
| 环境备注 | 机器有其他会话后台负载（load ≈1.4，非基准核），无 LD_LIBRARY_PATH / LD_PRELOAD |

## 2. 为完成执行所做的环境适配（全部为工具包/测试级补丁，未改生产实现、未放宽容差）

1. **fixed 编译阻断**：`tests/turbo/turbo_matrix_test.cc:149` 的 `T(7)` 在 x86 上对 `ailego::Float16` 产生 `Float16(float)` / `Float16(double)` 歧义（macOS ARM64 上 `Float16` 为 `__fp16` 故可编过）。补丁改为 `T(7.0f)`，不改测试语义。补丁保存为 `results/fixed_worktree.patch`。`run.py` 相应改为记录 worktree 差异而非中止。
2. **工具包链接缺陷**：首轮正确性中 Flat/聚类/IVF 共 7 个程序及 parent_write、before_ivf 全部失败（`Invalid index measure SquaredEuclidean`、`Create Metric ... failed`、abort/segfault）。根因是 Ubuntu GCC 默认 `--as-needed`，`libcore_metric.so` / `libcore_knn_cluster.so` 未进入 DT_NEEDED，工厂注册未执行；主仓库顶层 CMakeLists 设有 `-Wl,--no-as-needed` 而工具包未带。已在 `scripts/linux_x86_regression/CMakeLists.txt` 补上（三版本公平应用）。首轮失败日志保留在 `results/attempt1_as_needed_linkbug/`。

## 3. 正确性结果

### 3.1 fixed 44 个测试程序（退出码与 XML 双重核对）

| passed | skipped | disabled | failed | 退出码非零 / 缺 XML |
|---|---|---|---|---|
| **510** | 20 | 25 | **0** | 0 |

- 10 metric、27 Turbo、2 Flat、3 聚类、ivf_legacy_matrix_test、ivf_searcher_test（13 项指定用例）全部 exit=0。
- skipped 主要是 `turbo_neon_*`（ARM 专用，x86 跳过 5+1+1）、fp16/fp32/pq quantizer 的平台条件跳过及 `ivf_legacy_matrix_test` 中 FP32/FP16 无记录量化的 4 项。disabled 为仓库中 `DISABLED_` 基准用例。
- 本次 Linux 计数（510）与 macOS 497 不同属于预期的条件跳过差异。

### 3.2 跨版本旧索引测试（与上表分开统计）

| 项目 | 结果 |
|---|---|
| parent_write（parent 生成 200 份旧索引） | 200 份 `.index` 全部写出；但 parent **自身**查询校验 2 个用例失败（见下） |
| fixed_read_parent（fixed 读取 parent 旧索引） | **exit=0，12 passed / 4 skipped / 0 failed**，28,800 组结果核对通过 |
| before_ivf | exit=0，12 passed / 4 skipped / 0 failed |

**parent 基线问题**：`PersistedColumnBlocksMatchIndependentReference/5` 与 `PersistedRecordQuantizedBlocksMatchIndependentReference/5`（INT8 + InnerProduct）在 dim=128/512 共 10,218 处断言失败，差值均为 32 的整数倍（8192、11520、26624、32768 …），容差 ≈13。这是 parent x86 旧 INT8 IP 内核自身的数值偏差（before/fixed 同用例均通过，fixed 读取 parent 写出的同一份索引也通过），标记为**基线问题**，不影响 fixed 结论。首个复现：`ZVEC_LEGACY_WRITE_FIXTURES=... build-parent/bin/legacy_ivf_comparison --gtest_filter='*/5'`，日志 `results/parent_write.log`。

**结论 1**：fixed 通过本次全部相关正确性回归。
**结论 2**：重构前（parent）生成的 200 份旧索引可被 fixed 正确读取和查询。

## 4. 性能结果

- 完成数量：matrix 504/504、adapter 32/32、ivf 24/24，三版本 × 3 进程全部 exit=0；补测轮（`recheck/results/`）同样 504/32/24 全部完成。
- 单位：matrix/adapter 为线程 CPU 时间 ns/矩阵；ivf 为墙钟 μs/查询。下表为 3 进程中位数，格式 `首轮 / 补测轮`，区间列出首轮三进程 min–max。比例 <1 表示 fixed 更快。`f/p = fixed/parent`，`f/b = fixed/before`。
- 两轮一致性：matrix 的 90 个 >5% 项中 85 个在补测轮复现；adapter/ivf 中位数漂移 ≤0.3%（最大 3.7%）。FP16 多列组合两轮间波动最大（parent 自身亦漂移 10% 以上）。

### 4.1 优先项 1：旧记录量化 INT8/INT4，32×1（adapter，ns）

| 组合 | parent | before | fixed | f/p | f/b | fixed 区间 |
|---|---|---|---|---|---|---|
| INT8 RecordL2 128 | 1486 / 1484 | 1753 / 1642 | **739 / 741** | 0.50 | 0.42 | 738.8–739.1 |
| INT8 RecordL2 512 | 5897 / 5888 | 4709 / 4695 | **2774 / 2773** | 0.47 | 0.59 | 2771–2780 |
| INT8 RecordIP 128 | 1497 / 1506 | 1683 / 1680 | **730 / 730** | 0.49 | 0.43 | 729–731 |
| INT8 RecordIP 512 | 6008 / 5893 | 4948 / 4941 | **2768 / 2767** | 0.46 | 0.56 | 2766–2772 |
| INT4 RecordL2 128 | 1263 / 1291 | 1129 / 1135 | 1463 / 1447 | **1.16 / 1.12** | **1.30** | 1429–1484 |
| INT4 RecordL2 512 | 4946 / 5052 | 3300 / 3282 | 5593 / 5609 | **1.13 / 1.11** | **1.69** | 5341–5737 |
| INT4 RecordIP 128 | 1395 / 1396 | 1117 / 1129 | 1447 / 1486 | **1.04 / 1.06** | **1.30** | 1435–1452 |
| INT4 RecordIP 512 | 5531 / 5511 | 3265 / 3270 | 5744 / 5660 | **1.04 / 1.03** | **1.76** | 5566–5791 |

- INT8 记录量化 32×1：耗时较 parent 减少约 50–54%，较 before 减少 41–58%，两轮稳定。
- **INT4 记录量化 32×1 尚存回退**：较 parent 慢 3–16%，较 before 慢 28–76%，两轮均复现。before 在该路径上原本快于 parent，fixed 退回到略慢于 parent 的水平。
- 32×32 记录量化：INT8 f/p 0.46 但 **f/b 2.0–4.1**（before 9.5–29 μs vs fixed 22–87 μs，两轮复现）；INT4 f/p 0.19–0.27，f/b ≈1.00。

### 4.2 优先项 2：原始矩阵 32×1（matrix，ns）

| 组合 | parent | before | fixed | f/p | f/b |
|---|---|---|---|---|---|
| INT4 IP 32×1 128 | 1393 / 1404 | 3844 / 4031 | 1322 / 1316 | 0.95 / 0.94 | 0.34 |
| INT4 IP 32×1 512 | 5581 / 5695 | 15840 / 16234 | 5229 / 5284 | 0.94 / 0.93 | 0.33 |
| INT8 L2 32×1 128 | 3225 / 3270 | 2408 / 2432 | **709 / 711** | 0.22 | 0.29 |
| INT8 L2 32×1 512 | 12991 / 12858 | 9598 / 9609 | **2808 / 2801** | 0.22 | 0.29 |
| INT8 IP 32×1 128 | 1501 / 1514 | 2320 / 2338 | **689 / 689** | 0.46 | 0.30 |
| INT8 IP 32×1 512 | 6048 / 6068 | 9270 / 9094 | **2724 / 2725** | 0.45 | 0.29 |
| FP16 L2 32×1 128 | 9947 / 9961 | 6880 / 6834 | 9931 / 9938 | 1.00 | **1.44** |
| FP16 L2 32×1 512 | 39911 / 39718 | 27374 / 27214 | 39757 / 39666 | 1.00 | **1.45** |

- INT4 IP 32×1 相对 before 恢复 66%，并略优于 parent（-5%）。INT8 32×1 相对 parent 提升 2.2–4.5 倍。
- **FP16 L2 32×1**：与 parent 持平，但较 before 慢 44–46%（before 在此路径快于 parent，fixed 未保留）。FP16 IP 32×1 则 f/p 0.66、f/b 0.95。

### 4.3 优先项 3：MIPS 适配层 32×1 / 32×32（adapter，ns；仅矩阵适配层，不代表 MIPS IVF 查询）

| 组合 | parent | before | fixed | f/p | f/b |
|---|---|---|---|---|---|
| FP32 MIPS 32×1 128 | 608 / 610 | 5751 / 5767 | 766 / 768 | **1.26** | 0.13 |
| FP32 MIPS 32×1 512 | 2646 / 2679 | 23152 / 23149 | 3090 / 3094 | **1.17** | 0.13 |
| FP32 MIPS 32×32 128/512 | 61012 / 243486 | 58845 / 229191 | 9339 / 36644 | 0.15 | 0.16 |
| INT8 MIPS 32×1 128 | 1343 / 1346 | 7037 / 7043 | 1168 / 1166 | 0.87 | 0.17 |
| INT8 MIPS 32×1 512 | 5305 / 5300 | 25668 / 25567 | 4468 / 4464 | 0.84 | 0.17 |
| INT8 MIPS 32×32 128/512 | 49702 / 198215 | 187172 / 690776 | 24580 / 93314 | 0.49 / 0.47 | 0.13 |
| INT4 MIPS 32×1 128 | 2870 / 2848 | 11828 / 11840 | 2393 / 2455 | 0.83 / 0.86 | 0.20 |
| INT4 MIPS 32×1 512 | 11637 / 11618 | 44681 / 44656 | 9240 / 9542 | 0.79 / 0.82 | 0.21 |
| INT4 MIPS 32×32 128 | 44110 / 43993 | 367398 / 367062 | 48813 / 46999 | **1.11 / 1.07** | 0.13 |
| INT4 MIPS 32×32 512 | 172893 / 172414 | 1368789 / 1370737 | 191693 / 185529 | **1.11 / 1.08** | 0.14 |
| FP16 MIPS 32×1 128 | 21092 / 21044 | 5914 / 5929 | 14800 / 14850 | 0.70 | **2.50** |
| FP16 MIPS 32×1 512 | 84204 / 84288 | 21535 / 21567 | 59087 / 59492 | 0.70 | **2.74** |
| FP16 MIPS 32×32 128/512 | 386444 / 1540453 | 48916 / 168126 | 263524 / 1038976 | 0.68 / 0.67 | **5.39 / 6.18** |

- INT8/INT4 MIPS 32×1 的主要开销（before 慢 5–8 倍）已恢复并优于 parent 13–21%。
- 剩余差距：FP32 MIPS 32×1 较 parent 慢 17–26%；INT4 MIPS 32×32 较 parent 慢 7–11%（两轮复现，fixed 区间 48130–72442 偏宽）。
- **FP16 MIPS**：虽较 parent 快 30%，但较 before 慢 2.5–6.2 倍，两轮稳定复现。

### 4.4 优先项 4：FP32 L2 16×4、4×2/512、32×8/128（matrix，ns）

| 组合 | parent | before | fixed | f/p | f/b |
|---|---|---|---|---|---|
| FP32 L2 16×4 128 | 811 / 810 | 945 / 943 | 678 / 679 | 0.84 | 0.72 |
| FP32 L2 16×4 512 | 3248 / 3231 | 3763 / 3744 | 2702 / 2702 | 0.83 | 0.72 |
| FP32 L2 4×2 512 | 1200 / 1207 | 587 / 586 | 588 / 588 | 0.49 | 1.00 |
| FP32 L2 32×8 128 | 2942 / 2932 | 2700 / 2692 | 2715 / 2696 | 0.92 | 1.00 |

均无回退。其他类型 4×2/512：FP16 IP f/p 1.06→0.83（两轮翻转，判定为波动）、INT4/INT8/FP64/INT16 均 ≤1.0。32×8/128：INT4 L2/IP f/p 1.10/1.13、FP64 IP 1.21、INT16 IP 1.09。

### 4.5 优先项 5：INT4 多查询及其余回退

- INT4 多查询（列数 >1，共 60 组）：相对 before 全部恢复（f/b 0.22–0.49）；相对 parent 大多在 0.75–1.04，**回退项**为 32×8（L2 1.10、IP 1.13–1.20）与 32×16 IP（1.17–1.19；补测 1.11）。fixed 在这些形状区间偏宽（如 32×8 IP/512 为 49453–68306），回退幅度不确定但方向两轮一致。
- 全部 504 组 matrix 分布：相对 parent 271 组快（<0.95）、143 组持平、90 组慢（>1.05，补测 89，85 组两轮复现）；相对 before 290 快 / 152 持平 / 62 慢。
- **继承自重构、本次修复未涉及**（f/p >1.05 且 f/b ≈1.00，47 组）：
  - INT4 1×1 单查询：parent 7–19 ns vs fixed 177–875 ns，**慢 26–47 倍**（L2/IP、128/512 均如此）。
  - INT8 1×1：1.5–1.8 倍；FP16 1×1：2.0–3.2 倍（parent FP16 1×1 精度较低，微基准已跳过其参考断言，不是同精度基线）；FP32 1×1/128：1.15。
  - FP32 L2/IP 单列 4×1、8×1、16×1：1.19–1.62 倍；FP32 L2 16×8：1.17。
  - INT16 L2 16×N（N=1,2,4,8,16）：约 2.0–2.2 倍（f/b 1.03–1.11，略差于 before）。
  - FP64 IP 32×2/4/8/16 与 16×16：1.13–1.31 倍。
- **fixed 同时慢于 parent 与 before**（两轮复现）：FP32 IP 1×1（f/p 1.09–1.35，f/b 1.12）；INT16 L2 16×2/16×8/16×16（f/p ≈2.0–2.2，f/b 1.06–1.11）。
- **fixed 慢于 before 但不慢于 parent**（62 组，主要是 FP16）：FP16 IP 23 组（f/b 1.08–2.01）、FP16 L2 21 组（f/b 1.06–1.53）、INT16 IP 8×N 6 组（1.06–1.38）。即 before 在 FP16 上的提速在 fixed 中未保留。

### 4.6 旧 IVF 查询（24 组，μs/查询，墙钟）

| 类型/度量 | parent | before | fixed | f/p | f/b |
|---|---|---|---|---|---|
| FP32 L2/IP 128/512 | 232–692 | 211–649 | 211–643 | 0.91–0.95 | 0.99–1.00 |
| FP16 L2 128 / 512 | 3152 / 12462 | 2111 / 8266 | 3243 / 12875 | 1.03 | **1.54–1.56** |
| FP16 IP 128 / 512 | 3113 / 12588 | 2094 / 8261 | 2011 / 7904 | 0.65 / 0.63 | 0.96 |
| INT8 列式 L2 128 / 512 | 976 / 3726 | 709 / 2575 | **270 / 829** | 0.28 / 0.22 | 0.38 / 0.32 |
| INT8 列式 IP 128 / 512 | 456 / 1654 | 679 / 2536 | **265 / 843** | 0.58 / 0.51 | 0.39 / 0.33 |
| INT8 记录量化 L2/IP 128 / 512 | 451 / 1601–1617 | 498–509 / 1419–1473 | **278–281 / 838–840** | 0.52–0.62 | 0.55–0.59 |
| INT4 列式 L2/IP 128 / 512 | 410–416 / 1466–1475 | 4991–5032 / 19750–20095 | 431–434 / 1503–1533 | 1.03–1.05 | **0.08–0.09** |
| INT4 记录量化 L2/IP 128 / 512 | 399–412 / 1396–1501 | 354–361 / 967–971 | 432–437 / 1486–1541 | 1.03–1.08 | **1.20–1.59** |

- INT4 列式 IVF：before 慢 12 倍的问题已恢复到与 parent 持平（慢 3–5%，两轮一致，接近阈值）。
- INT8 列式/记录量化 IVF：较 parent 快 1.6–4.5 倍。
- **INT4 记录量化 IVF**：较 parent 慢 3–8%，较 before 慢 20–59%，与 4.1 的 INT4 记录 32×1 适配层结果一致。
- **FP16 L2 IVF**：与 parent 持平，较 before 慢 54–60%。
- 以上旧 IVF 数字仅针对旧列式/旧记录量化兼容路径（8193 条、单倒排、scan_ratio=1），不代表普通行式 IVF Turbo 路径收益。

## 5. 结论

1. **正确性**：fixed 44 个程序 510 passed / 0 failed；fixed 读取 parent 200 份旧索引 12 passed / 0 failed；before IVF 通过。唯一失败为 parent 基线自身的 x86 INT8 IP 数值偏差（已标记基线问题，fixed/before 不受影响）。无崩溃、无缺失 XML。
2. **旧索引兼容**：重构前旧索引可被修复版本正确读取与查询（含 31/32/33 行边界、空倒排、三种存储、过滤、不同 top-k）。
3. **性能恢复**：
   - 已恢复并优于 parent：INT8 记录量化 32×1（耗时 -50%）、INT8 原始 32×1（-55~-78%）、INT8/INT4 MIPS 32×1（-13~-21%）、INT4 原始 32×1（-5~-6%）、INT4 列式 IVF（恢复 12 倍，持平 parent）、FP32 L2 16×4（-16%）、INT8 IVF（-42~-78%）。
   - 相对 parent 尚存回退（两轮复现）：INT4 记录量化 32×1（+3~16%）与 INT4 记录量化 IVF（+3~8%）；FP32 MIPS 32×1（+17~26%）；INT4 MIPS 32×32（+7~11%）；INT4 原始 32×8/32×16 IP（+10~20%，区间偏宽）；INT4 列式 IVF（+3~5%）。
   - 继承自重构、未在本修复范围内：INT4 1×1 单查询慢 26–47 倍，INT8/FP16 1×1 慢 1.5–3.2 倍，FP32 单列 4/8/16×1 慢 1.2–1.6 倍，INT16 L2 16×N 慢 2 倍，FP64 IP 多列慢 1.1–1.3 倍。
   - 相对 before 的回退（before 曾优于 parent，fixed 未保留）：FP16 几乎所有多列形状及 L2 32×1（1.06–2.0 倍）、FP16 MIPS（2.5–6.2 倍）、FP16 L2 IVF（1.54 倍）、INT8 记录量化 32×32（2.0–4.1 倍）、INT4 记录量化 32×1/IVF（1.2–1.8 倍）、INT16 IP 8×N（1.06–1.38 倍）。
4. **无法确认/波动项**：FP16 IP 4×2、8×2、FP16 L2 16×1、8×1/512 等 FP16 多列组合两轮 f/p 在 1.05 两侧翻转，归为波动；FP64 IP 1×1、INT4 L2 16×16/2×2/512 同理。
5. **构建/正确性阻断**：fixed 提交 955ae841 的 `tests/turbo/turbo_matrix_test.cc:149` 在 x86 GCC 下无法编译（见第 2 节），已随本报告一并提交修正（`tests/turbo/turbo_matrix_test.cc` 与工具包 `CMakeLists.txt`）。两者均不涉及生产实现。
6. **适用范围**：本结论限于 metric/Turbo/Flat/聚类/旧 IVF 的独立构建（RABITQ/DISKANN 关闭），KVM 虚拟机单核绑定测量，不含生产规模数据、召回率、并发、ASan/UBSan。不可将旧 IVF 结果泛化到全部 IVF，也不可把 MIPS 适配层提速解释为 MIPS IVF 查询提速。百分比均为耗时变化，非吞吐变化。

## 6. 证据文件

原始证据未入库，位于执行机 `zvec-x86-regression.04K3Yl/linux-x86-results.tar.gz`（5.1 MB），内容如下：

- `results/correctness.json`、各 `*.xml` / `*.log`：正确性。
- `results/{matrix,adapter,ivf}_comparison.csv`、`performance.json`、`*_{parent,before,fixed}_{0,1,2}.log`：首轮性能。
- `recheck/results/`：同条件补测轮完整结果。
- `results/attempt1_as_needed_linkbug/`：首轮链接缺陷导致的失败日志。
- `results/fixed_worktree.patch`：fixed 测试文件最小补丁。
- `results/toolchain.txt`、`lscpu.txt`、`environment.json`、`benchmark_cpu.json`、`perf_start_load.txt`：环境。
- `build-*/CMakeCache.txt`、`compile_commands.json`：编译参数。
- `parent-fixtures/`（未打包，保留在工作目录）：parent 生成的 200 份旧索引。
