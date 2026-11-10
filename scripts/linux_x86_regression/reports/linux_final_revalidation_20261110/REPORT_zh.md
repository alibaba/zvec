# Linux x86 最终复验报告：fix/legacy-matrix-distance-performance 候选 vs Linux final

日期：2026-11-09。执行环境：Linux iZ2zebpiyrokizzrznt3ykZ 6.8.0-87-generic x86_64，Intel Xeon 6982P-C（16 逻辑 CPU，2 线程/核，L2 16 MiB×8，L3 504 MiB），GCC 13.3.0（Ubuntu 13.3.0-6ubuntu2~24.04.1），CMake 3.30.0，Ninja 1.11.1，Python 3.10.9。与此前 Linux final 报告（GCC 13.3 / Xeon 6982P-C）同一类环境。

## 0. 结论

| 项 | 结果 |
|---|---|
| 交接分支 / 工具提交 | `origin/fix/legacy-matrix-distance-performance` = `a934deeade49138b824e2f4a1d70fd4252ddf216`（含 a934deea）；`git diff 008fbdc3..HEAD -- src tests` 为空，候选生产代码即 CODE=008fbdc36737e0b4b70ee2e29a3ff12889f8061a |
| 材料校验 | `final_revalidation/SHA256SUMS` 与 4 个 artifacts 包 `.sha256` 全部通过 |
| 正确性（fixed 44 程序） | older955 510 passed / reference7abc 510 / linux_final 511 / candidate 511；均 20 skipped、25 disabled、0 failed、全部退出 0（与前次 Linux 结果一致） |
| 旧索引兼容 | 四套 parent 各写出 200 个非空 `.index`；fixed 读取均 12 passed / 4 skipped / 16 disabled / 0 failed；candidate 另有 strace 记录成功打开 200 个不同 fixture |
| parent 例外 | 四套 parent_write 均且仅失败已知 INT8 IP /5 两项，错误内容与前次报告一致；未掩盖、未改期望 |
| before_ivf | 四套均 12 passed / 4 skipped / 0 failed |
| x86 代码一致性（candidate vs linux_final） | 指定三 TU 预处理 token 除 `__FILE__` 路径字符串外相同；全部 257 个目标文件加 `-ffile-prefix-map` 后逐字节相同；3 个基准可执行文件 .text 大小/符号地址相同，指令差异屏蔽 RIP 位移后为 0；ARM 新增分支在 x86 完全被排除 |
| 性能：candidate / Linux final | matrix 504 / adapter 32 / IVF 24：两轮都 >5% 的用例 **0 / 0 / 0**；中位比值 1.000/1.000、1.000/0.998、1.002/0.999 |
| 性能：candidate / 7abc | matrix 两轮交集 2 项（FP32 L2 1×1 d128 1.058/1.082，FP32 IP 16×4 d128 1.057/1.056）；adapter/IVF 0 |
| 性能：candidate / 955 | matrix 两轮交集 2 项（FP32 L2 1×1 d128 1.063/1.054，INT8 L2 2×1 d128 1.085/1.072）；adapter/IVF 0 |
| 性能：candidate / before, parent | matrix 两轮交集 15 / 66 项，全部为前次报告已列的 FP64/INT16/1×1 继承项（见 §6）；adapter 0/0，IVF 0/0 |
| 本轮新增回退（相对 Linux final，两轮复现） | **无** |

上述 4 个 >5% 项（vs 955/7abc）均是前次 Linux final 报告已记录的继承项；对应内核在四个 fixed 二进制中归一化反汇编逐条相同（§7）。

## 1. 版本与材料

- 原始仓库 `/home/xingrui/workspace/zvec/ivf_matrix/zvec`，本地分支 HEAD 亦为 a934deea；未 reset，未清理旧实验目录。
- 干净交接工作树 `TASK/handoff`（分离 HEAD a934deea）。`BUNDLE=handoff/scripts/linux_x86_regression`，`PKG=BUNDLE/final_revalidation`。
- 四套 ROOT（均 parent=499eab89、before=f7bd575f、fixed HEAD=955ae841 + 补丁）：

| ROOT | 补丁 | 补丁 SHA256 | fixed src tree | fixed tests tree |
|---|---|---|---|---|
| older955 | baseline_955_test_build.patch | ac28c6f04b2d2674e350e41b5fd47693190abb8f4acaae778947cdead7726767 | 1202d0b4 | 590a2299 |
| reference7abc | reference_7abc_vs_955.patch | 877ae679d1cc558a9eaeeaa5bb90a9d34c776a565940a43c926eb1423426011f | 03fff21e | 623c67ae |
| linux_final | reference_linux_final_vs_955.patch | 5e1d3ee953aa5df174347ebbcfade5e8f29d7299611a6fd2374740a6404a41a1 | 9c7428dc | dffc1e9a |
| candidate | candidate_vs_955.patch | d1a845be014256a597cd79115599c33ef80851ce4583ce724b617501b9676cf8 | fb1a3f7a | dffc1e9a |

  src/tests tree 均与 `versions.json` 记录一致；`git diff --exit-code $CODE -- src tests` 在 candidate/fixed 为空。各 ROOT 的 `applied.patch`、`applied_patch.sha256`、`version_record.txt` 已随证据提交。
- 候选相对 Linux final 的源码差异（`candidate_vs_linux_final_src.diff`）仅 `src/turbo/distance/matrix.h` 三处，全部在 `AILEGO_ARM64_GNU_LIKE` / `__aarch64__` 宏内；tests 无差异。
- DEPS：复用先前已初始化的 `/home/xingrui/workspace/zvec/ivf_matrix/zvec-x86-regression.04K3Yl/fixed`（googletest 1.10.0、RaBitQ 0.7.0），四套共用。
- 构建：`run.py build`，`CC=gcc CXX=g++`，无 CFLAGS/CXXFLAGS/LDFLAGS，`-O3 -march=x86-64 -std=c++17`，Release，无 LTO/OpenMP，`ENABLE_NATIVE=OFF`，`AUTO_DETECT_ARCH=ON`，RABITQ/DISKANN 关闭。12 个构建树的 CMakeCache、compile_commands 与配置/构建日志均在证据中。构建期间未运行性能。

## 2. 正确性与旧索引兼容

四套 `run.py correctness` 的驱动退出码均为 1，原因全部且仅为 parent_write 的两项已知失败（驱动把 parent 失败也计入）。逐一审阅 `correctness.json`、XML、日志后：

| ROOT | fixed 44 程序（passed/skipped/disabled/failed） | before_ivf | parent_write | fixed_read_parent | 审计状态 |
|---|---|---|---|---|---|
| older955 | 510 / 20 / 25 / 0 | 12/4/16/0 | 10/4/16/2（已知 /5 两项，exit 1） | 12/4/16/0 | fixed_pass_with_known_parent_failures |
| reference7abc | 510 / 20 / 25 / 0 | 12/4/16/0 | 同上 | 12/4/16/0 | 同上 |
| linux_final | 511 / 20 / 25 / 0 | 12/4/16/0 | 同上 | 12/4/16/0 | 同上 |
| candidate | 511 / 20 / 25 / 0 | 12/4/16/0 | 同上 | 12/4/16/0 | 同上 |

- 511 vs 510：linux_final/candidate 新增的 stored-moments 测试通过；与前次 Linux 报告一致。skipped 20 中 5 项为 `turbo_neon_distance_test`（x86 上 GTEST_SKIP，0 passed / 5 skipped）。
- parent 失败项：`RawTypes/IVFLegacyMatrixTest.PersistedColumnBlocksMatchIndependentReference/5` 与 `PersistedRecordQuantizedBlocksMatchIndependentReference/5`，GetParam=(4,true)，断言差值与前次报告相同。四套 `parent_write.log/.xml` 原样保留。
- **审计工具说明**：`$PKG/verify_correctness.py` 的 `require_pass` 要求每个程序 passed>0，因而在 x86 上对 `turbo_neon_distance_test`（0 passed/5 skipped）报错（`verify_correctness_original_tool.log`）。未改动交接包工具，而是复制为 `tools/verify_correctness_x86.py`（差异见 `tools/verify_correctness_x86.diff`）：仅允许该一个程序在 skipped>0、disabled==0、failed==0、exit 0 时视为通过，并把例外记录到 `final_correctness_audit.json` 的 `x86_all_skipped_programs`/`audit_tool` 字段；其余检查（44 程序、before_ivf、parent 允许失败名单、200 fixture 非空、读取 12/4/16/0）与原工具一致。四套均以 `--allow-known-parent-int8-ip` 审计通过。
- `fixed_read_parent` 由审计工具补跑（run.py 因 parent 非零跳过）。
- 200 fixture：每套 `results_round1/fixture_sha256.txt`。四套 fixture 内容互不相同的字节为每文件 62–68 个（偏移 0–3、12–15、32–33 及块头位置等，见 `analysis/fixture_cross_root_diff.txt`），文件大小相同，为 parent 写出时的随机 ID/时间戳类头字段，数据块相同；每套 fixed 只读取本套 parent 写出的 fixture，未混用。
- strace：`candidate/results_round1/fixed_read_parent_strace.{txt,log,xml}`，`openat` 成功打开 200 个不同的 `parent-fixtures/*.index`（返回 fd ≥0），结果 12 passed / 4 skipped。

## 3. 性能方法

- CPU：`selected_cpu.txt`=9，SMT 同胞 8；`sched_getaffinity` 允许 0–15；执行前 8/9 空闲。两轮 18:03–18:24，负载均值 0.8–2.3（`load_snapshots.txt`），未同时编译。
- 每套每轮：504 raw matrix / 32 adapter / 24 IVF，内部每版本 3 进程轮换（parent/before/fixed），`taskset -c 9`，取中位；第一轮顺序 older955→reference7abc→linux_final→candidate，第二轮反序。
- `compare_followup.py` 以 BASE/fixed 为 previous_fix（955 / 7abc / Linux final），candidate 根的 parent/before 为对照；`parent_drift`/`before_drift` 为 candidate 根与 BASE 根同一轮 parent/before 之比。
- 四套根之间 parent/before 的最大离散（`analysis/summary.json` `root_drift`）：matrix 中位 1.004，>1.05 的 43/1008（R1）、59/1008（R2），最大 1.16/1.23，全部为微形状；adapter 最大 1.13/1.05；IVF 最大 1.05/1.10。说明单轮 ±5% 量级波动在本机常见，判定以两轮交集为准。
- 单位：matrix/adapter 为线程 CPU ns/matrix，IVF 为 wall-clock us/query。完整表见 `PERFORMANCE_TABLES.md`（A–K），六个 `comparisons/vs_*_round*/` 目录含 CSV 与 JSON，原始日志在各 `results_round*/`。

## 4. candidate / Linux final（核心比较）

| 组 | 用例 | >5% 两轮交集 | 中位 R1/R2 | 最大 R1/R2 |
|---|---|---|---|---|
| matrix 全部 | 504 | 0（R1 9 项、R2 2 项单轮） | 1.000 / 1.000 | 1.102 / 1.053 |
| adapter 全部 | 32 | 0（R1 0、R2 0） | 1.000 / 0.998 | 1.038 / 1.016 |
| IVF 全部 | 24 | 0（R1 0、R2 0） | 1.002 / 0.999 | 1.017 / 1.014 |

分组（均 0 项两轮 >5%）：

| 组 | 用例数 | cand/Linux final 中位 R1/R2 | 最大 R1/R2 | cand/before 中位 | cand/parent 中位 |
|---|---|---|---|---|---|
| FP16 原始矩阵 | 84 | 1.005 / 0.992 | 1.102 / 1.025 | 0.075 / 0.075 | 0.051 / 0.050 |
| FP16 适配器 MIPS | 4 | 0.997 / 0.997 | 1.006 / 1.014 | 1.006 / 1.020 | 0.192 / 0.195 |
| FP16 IVF | 4 | 0.981 / 0.973 | 1.006 / 0.979 | 0.095 / 0.098 | 0.062 / 0.065 |
| FP32 原始矩阵 | 84 | 1.001 / 1.000 | 1.006 / 1.022 | 0.997 / 0.998 | 0.708 / 0.708 |
| FP32 MIPS 适配器 | 4 | 1.000 / 0.998 | 1.002 / 1.010 | 0.123 / 0.122 | 0.389 / 0.412 |
| FP64 原始矩阵 | 84 | 1.000 / 1.000 | 1.030 / 1.021 | 0.581 / 0.579 | 1.001 / 1.001 |
| INT16 原始矩阵 | 84 | 1.000 / 1.000 | 1.033 / 1.011 | 0.949 / 0.948 | 0.977 / 0.977 |
| INT8 单列矩阵 | 24 | 1.001 / 1.001 | 1.010 / 1.007 | 0.477 / 0.478 | 0.567 / 0.566 |
| INT8 矩阵全部 | 84 | 1.000 / 1.000 | 1.010 / 1.053 | 0.372 / 0.370 | 0.451 / 0.454 |
| INT8 记录/融合 MIPS 适配器 | 12 | 1.000 / 1.000 | 1.038 / 1.016 | 0.497 / 0.496 | 0.469 / 0.467 |
| INT4 单列矩阵 | 24 | 0.999 / 0.998 | 1.031 / 1.050 | 0.291 / 0.291 | 0.958 / 0.970 |
| INT4 多列矩阵 | 60 | 1.000 / 1.000 | 1.017 / 1.012 | 0.264 / 0.265 | 0.770 / 0.770 |
| INT4 记录适配器（Record/MIPS） | 12 | 1.000 / 0.995 | 1.014 / 1.008 | 0.532 / 0.527 | 0.388 / 0.387 |
| 旧版 IVF 24 项 | 24 | 1.002 / 0.999 | 1.017 / 1.014 | 0.468 / 0.471 | 0.567 / 0.565 |

第一轮单轮 >5% 的 9 项全部为 FP16（IP 32×1/32×2/4×2/8×2、L2 32×1，d128/d512，1.052–1.102）；第二轮同一批形状反转为 0.918–0.966（例如 FP16 IP 32×1 d512：R1 1853.5 vs 1719.1 ns，R2 1689.3 vs 1840.5 ns），两轮 min/max 区间彼此覆盖（linux_final R1 min 1511.9，candidate R2 min 1506.5），属 FP16 多行形状在本机的双态波动，不是代码差异（两套二进制指令流相同，§7）。第二轮单轮 2 项 INT4 IP 2×1 d128 1.050、INT8 IP 2×2 d128 1.053，第一轮分别 0.997、0.988。

## 5. 必查目标

### 5.1 FP16（全部 84 形状，重点 IP 1×1 d128/d512）

| 用例 | 955 R1/R2 | 7abc | Linux final | candidate | cand/LF | cand/7abc | cand/955 | cand/before | cand/parent |
|---|---|---|---|---|---|---|---|---|---|
| FP16 IP 1×1 d128 | 10.607 / 10.239 | 10.274 / 10.182 | 10.743 / 10.727 | 10.753 / 10.746 | 1.001 / 1.002 | 1.047 / 1.055 | 1.014 / 1.050 | 1.033 / 0.970 | 2.124 / 2.110 |
| FP16 IP 1×1 d512 | 39.871 / 39.837 | 39.977 / 39.792 | 39.841 / 40.129 | 40.097 / 40.031 | 1.006 / 0.998 | 1.003 / 1.006 | 1.006 / 1.005 | 1.004 / 0.999 | 3.237 / 3.212 |
| FP16 L2 1×1 d128 | 12.152 / 12.032 | 11.907 / 11.879 | 11.696 / 11.860 | 11.880 / 11.956 | 1.016 / 1.008 | 0.998 / 1.006 | 0.978 / 0.994 | 0.994 / 0.994 | 2.301 / 2.337 |
| FP16 L2 1×1 d512 | 41.945 / 42.002 | 41.433 / 41.469 | 40.720 / 40.846 | 41.220 / 41.157 | 1.012 / 1.008 | 0.995 / 0.992 | 0.983 / 0.980 | 0.991 / 0.991 | 3.242 / 3.223 |
| FP16 IP 32×32 d128 | 207082.8 / 207051.3 | 8961.1 / 8984.6 | 9432.8 / 8805.3 | 9040.2 / 8967.7 | 0.958 / 1.018 | 1.009 / 0.998 | 0.044 / 0.043 | 0.042 / 0.042 | 0.028 / 0.027 |
| FP16 MIPS 32×32 d128（adapter） | 264079.0 / 263590.0 | 51102.0 / 49408.1 | 50824.7 / 51299.8 | 49378.4 / 50833.0 | 0.972 / 0.991 | 0.966 / 1.029 | 0.187 / 0.193 | 1.002 / 1.041 | 0.127 / 0.132 |
| FP16 IVF IP d128 | 见表 J | | | | 0.98 / 0.97 组中位 | | | 0.095 | 0.062 |

- macOS 报告中的 FP16 IP 1×1 d128 下降（12.09→7.5 ns）来自 ARM64 `neon::inner_product_fp16_distance` 直接调用，该分支在 x86 被预处理排除，x86 本轮 1×1 与 Linux final 持平（1.001/1.002）；相对 7abc 的 +4.7%/+5.5% 为 Linux final 起已存在（Linux final/7abc 同为 1.046/1.054）、1×1 10 ns 量级微形状布局差，不属本轮新增。FP16 1×1 相对 parent 的 2–3× 是 955 起即存在的已知差异（parent 跳过 1×1 oracle、走不同路径）。
- FP16 多行形状相对 955/before/parent 0.04–0.09×，MIPS、IVF 均无退化。

### 5.2 INT4

单列 2–32 行（表 C 节选，cand/LF、cand/7abc、cand/955、cand/before）：INT4 L2 32×1 d128 0.991/0.983、0.787/0.782、0.823/0.827、0.298/0.308；d512 0.991/1.010、0.754/0.752、0.796/0.767、0.285/0.283。INT4 IP 32×1 d128 1.021/0.985、0.800/0.794、0.780/0.755；16×1、8×1、4×1 均 0.74–0.92×7abc/955。2×1 IP d128 0.997/1.050（vs LF）、0.933/0.935（vs 7abc）。
多列：INT4 IP 32×32 d128 30755/30760 ns，cand/LF 1.003/1.003，cand/7abc 0.711/0.718，cand/955 0.786/0.788；L2 32×16 d128 1.002/1.000、0.571/0.566、0.747/0.752；L2 16×16 d128 1.000/1.000、0.753/0.728、0.583/0.592；小多列 2×2/4×2/4×4 cand/LF 0.95–1.02，vs 7abc 0.86–1.00。
记录路径（adapter，类型 6）：

| 用例 | before R1/R2 | 955 | 7abc | Linux final | candidate | cand/LF | cand/7abc | cand/955 | cand/before |
|---|---|---|---|---|---|---|---|---|---|
| INT4rec RecordL2 32×1 d128 | 1129.1 / 1159.3 | 1416.5 / 1409.1 | 1214.9 / 1185.1 | 604.1 / 606.2 | 610.0 / 605.7 | 1.010 / 0.999 | 0.502 / 0.511 | 0.431 / 0.430 | 0.540 / 0.523 |
| INT4rec RecordL2 32×1 d512 | 3283.5 / 3270.3 | 5452.6 / 5387.8 | 3285.4 / 3283.5 | 1754.1 / 1754.5 | 1755.3 / 1752.3 | 1.001 / 0.999 | 0.534 / 0.534 | 0.322 / 0.325 | 0.535 / 0.536 |
| INT4rec RecordIP 32×1 d128 | 1127.5 / 1120.3 | 1408.5 / 1379.9 | 1141.3 / 1132.6 | 583.2 / 583.2 | 587.1 / 585.6 | 1.007 / 1.004 | 0.514 / 0.517 | 0.417 / 0.424 | 0.521 / 0.523 |
| INT4rec RecordIP 32×1 d512 | 3267.6 / 3250.4 | 5624.5 / 5420.1 | 3279.0 / 3273.1 | 1734.2 / 1734.2 | 1732.1 / 1724.0 | 0.999 / 0.994 | 0.528 / 0.527 | 0.308 / 0.318 | 0.530 / 0.530 |
| INT4rec RecordL2 32×32 d128 | 11759.7 / 11746.6 | 11700.6 / 11712.8 | 11846.8 / 11802.7 | 11760.1 / 11896.6 | 11758.7 / 11716.0 | 1.000 / 0.985 | 0.993 / 0.993 | 1.005 / 1.000 | 1.000 / 0.997 |
| INT4rec MIPS 32×32 d128 | 366539.0 / 365530.6 | 48548.7 / 46141.5 | 43217.2 / 43658.9 | 35029.9 / 36456.0 | 35394.2 / 35510.1 | 1.010 / 0.974 | 0.819 / 0.813 | 0.729 / 0.770 | 0.097 / 0.097 |
| IVF INT4 SquaredEuclidean d128 record=1（us/query） | 361.2 / 352.5 | 428.0 / 433.0 | 369.4 / 373.7 | 229.6 / 229.7 | 230.4 / 230.6 | 1.003 / 1.004 | 0.624 / 0.617 | 0.538 / 0.533 | 0.638 / 0.654 |
| IVF INT4 SquaredEuclidean d512 record=1 | 995.0 / 974.9 | 1528.9 / 1489.2 | 964.0 / 974.6 | 566.4 / 571.7 | 572.8 / 573.3 | 1.011 / 1.003 | 0.594 / 0.588 | 0.375 / 0.385 | 0.576 / 0.588 |

记录 32×1 约 0.52–0.54×before 的收益保留；stored-moments 测试通过，语义未变。

### 5.3 此前 13 项整数目标（表 A）

| 用例 | cand/LF R1/R2 | cand/7abc | cand/955 | cand/before |
|---|---|---|---|---|
| INT4 IP 16×2 d128 / d512 | 1.001/0.989、1.001/0.998 | 0.883/0.862、0.867/0.848 | 0.951/0.956、0.952/0.951 | 0.34 |
| INT4 IP 32×32 d128 / d512 | 1.003/1.003、1.001/1.008 | 0.711/0.718、0.694/0.688 | 0.786/0.788、0.797/0.772 | 0.27–0.29 |
| INT4 L2 32×1 d128 / d512 | 0.991/0.983、0.991/1.010 | 0.787/0.782、0.754/0.752 | 0.823/0.827、0.796/0.767 | 0.28–0.31 |
| INT4 L2 32×16 d128 / d512 | 1.002/1.000、1.005/0.998 | 0.571/0.566、0.636/0.563 | 0.747/0.752、0.754/0.754 | 0.23–0.24 |
| INT4 L2 32×2 d128 / d512 | 1.000/0.999、1.001/0.994 | 0.929/0.930、0.938/0.923 | 1.005/0.992、1.011/0.984 | 0.30–0.31 |
| INT4 L2 8×8 d128 / d512 | 0.999/1.002、1.000/1.000 | 0.658/0.610、0.634/0.634 | 0.675/0.661、0.681/0.686 | 0.235 |
| INT8 L2 2×2 d128 | 0.985/0.986 | 0.937/0.948 | 1.013/1.060 | 0.375/0.355 |

12 项两轮快于 7abc 且 ≤955+1.1%；INT8 L2 2×2 d128 与 955 持平（R2 1.060 单轮，R1 1.013，前次报告已记为布局噪声）。

### 5.4 FP32/FP64/INT16、INT8 单列/记录/融合 MIPS

见 §4 分组表与 `PERFORMANCE_TABLES.md` E–I：cand/LF 中位 1.000，最大单轮 1.033（INT16）、1.030（FP64）、1.022（FP32）、1.038（INT8 适配器）。INT8 记录 32×1/32×32、融合 MIPS 32×1/32×32（d128/d512）cand/LF 0.997–1.003，cand/7abc 0.990–1.002，cand/955 0.994–1.003。

## 6. >5% 两轮交集与分类

| 对照 | matrix 两轮交集 | 分类 |
|---|---|---|
| Linux final | 0 | — |
| 7abc | FP32 L2 1×1 d128 1.058/1.082；FP32 IP 16×4 d128 1.057/1.056 | 继承：前者前次 Linux final 报告 §5.5 已列（1.062/1.079×7abc）；后者 Linux final/7abc 同为 1.052/1.060（468.2/473.0 vs 445.0/446.2 ns），candidate/LF 1.005/0.997，即 Linux final 起的既有差异 |
| 955 | FP32 L2 1×1 d128 1.063/1.054；INT8 L2 2×1 d128 1.085/1.072 | 继承：均为前次报告 §5.5 列出项（1.073/1.075、1.070/1.076×955） |
| before | 15 项：INT16 14（L2 16×1/16×2/16×8/16×16、IP 8×1/8×2/8×4，d128/d512）+ FP32 IP 32×4 d512 1.082 | 继承：数量（15）及类型组成与前次 Linux final 报告一致（FP64/INT16 继承列表的子集） |
| parent | 66 项：FP64 18、FP32 17、INT16 17、INT4 6、FP16 4、INT8 4 | 继承：整数/FP16 1×1 与 INT4 2×1 为 955 起即存在的 parent 路径差异；FP64/INT16/FP32 列表与前次 Linux 报告（68 项）一致，减少 2 项 |
| adapter / IVF（任一对照） | 0 | IVF vs parent 仅 R2 单轮 INT4 SquaredEuclidean d128 record=0 1.051（R1 1.027） |

本轮相对 Linux final 没有新增项，相对 955/7abc 没有新出现的项；全部 >5% 项都能在前次 Linux final 报告的列表中找到对应。

## 7. x86 代码一致性与回退定位

详见 `codecheck/CODECHECK_SUMMARY_zh.md`、`codecheck/codecheck.json`、`all_objects.json`、`prefixmap_all.json`：

- 预处理（`g++ -E -P` + token 化）：matrix_sweep.cc 599566 token 完全相同；quantized_integer_metric.cc / mips_euclidean_metric.cc 仅 `__FILE__` 字符串中的工作树路径（`linux_final/` vs `candidate/`）不同，非路径 token 无差异；adapter_sweep.cc 完全相同。
- 目标文件：三 TU + adapter_sweep 的 built.o 全部可执行节字节相同、含重定位反汇编相同；每套 built.o 与按 compile_commands 重编的 reproduced.o 逐字节相同（可复现）。build-fixed 257 个目标文件集合相同，可执行节相同 253/257，差异 4 个均为测试 TU 的路径长度立即数；加 `-ffile-prefix-map=<root>=/ROOT` 后 257/257 逐字节相同。
- 可执行文件：matrix_sweep/adapter_sweep/legacy_ivf_comparison 的 .text 大小相同（1722795/1032427/1434395），符号地址全部相同；原始反汇编差异 4728/345/4083 行全为 RIP 相对位移（.rodata 字符串长度差引起，matrix_sweep .rodata 37620 vs 37556），屏蔽后差异 0 行。
- 回退定位（`analysis/localize/`，`tools/localize_symbols.py`）：对 §6 的 4 个继承项及关键形状，在四个 fixed `matrix_sweep` 中提取内核符号、归一化地址/位移后逐条比较：

| 内核 | 指令数 | 955 / 7abc / LF / candidate 地址 | 与 candidate 相同 |
|---|---|---|---|
| `kernel<float,1,1,false>`（FP32 L2 1×1） | 77 | 0x45550 / 0x46510 / 0x46490 / 0x46490 | 4/4 |
| `kernel<float,16,4,true>`（FP32 IP 16×4） | 433 | 0x71ac0 / 0x6e720 / 0x6b4d0 / 0x6b4d0 | 4/4 |
| `kernel<signed char,2,1,false>`（INT8 L2 2×1） | 182 | 0x5aa20 / 0x586d0 / 0x58650 / 0x58650 | 4/4 |
| `kernel<Float16,1,1,true>`（FP16 IP 1×1） | 77 | 0x45940 / 0x46900 / 0x46880 / 0x46880 | 4/4 |
| `kernel<unsigned char,32,1,false>`（INT4 L2 32×1 入口） | 16 | — | 4/4 |
| `kernel<unsigned char,32,32,true>`（INT4 IP 32×32） | 213/213/214/214 | — | LF=candidate；955/7abc 为修复前旧实现（预期不同） |

  candidate 与 Linux final 在所有抽样内核的地址与指令完全相同，955/7abc 相同指令但地址不同——4 个继承项是函数绝对位置引起的布局效应（前次报告已用 `-falign` 诊断构建证实此类项消失），不是逻辑退化。未对其改代码。

## 8. 未验证范围与注意

- 本工具只覆盖 metric/Turbo/Flat/clustering/legacy IVF 微基准与指定测试，不代表全产品吞吐、召回或并发。
- ARM64 两项改动在 x86 不可见；macOS 结论以 `reports/macos_*` 为准，本报告不复述其数字。
- 未修改交接包内任何脚本；审计使用的 x86 本地副本及 diff 随证据提供。未提交、未推送任何内容；所有工作树为分离 HEAD。
- 全部 TASK 目录（含 12 个构建树、runtime、codecheck 原始 .o/.ii/大反汇编）保留在服务器 `/home/xingrui/workspace/zvec/ivf_matrix/zvec-final-x86.445hP6`，证据包中省略了可从构建树再生的 `.ii`、`*.o`、完整可执行文件反汇编，保留摘要、JSON、diff、命令记录与内核级反汇编。

## 9. 证据索引

- `handoff_commit.txt`、`uname.txt`、`lscpu.txt`、`compiler.txt`、`selected_cpu.txt`、`load_snapshots.txt`、`candidate_vs_linux_final_src.diff`
- `handoff/`：`final_revalidation` 包副本（patches、versions.json、SHA256SUMS、verify_correctness.py）
- `<ROOT>/results_round1/`（正确性 XML/日志、correctness.json、final_correctness_audit.json、fixture_sha256.txt、三组 `*_comparison.csv`、所有基准日志）、`results_round2/`、`parent-fixtures/`（200 .index）、`applied.patch`、`applied_patch.sha256`、`correctness_driver.exit`、`version_record.txt`、驱动日志、`verify_correctness_*.log`、`buildinfo/{parent,before,fixed}/{CMakeCache.txt,compile_commands.json}`
- `comparisons/vs_{older955,reference7abc,linux_final}_round{1,2}/`
- `analysis/`：`PERFORMANCE_TABLES.md`、`summary.md|json`、`groups.md`、`fixture_cross_root_diff.txt`、`localize/`
- `codecheck/`：摘要、JSON、命令记录、per-TU token 摘要与 diff、可执行文件符号表与 RIP 差异 diff
- `tools/`：`analyze_final.py`、`localize_symbols.py`、`verify_correctness_x86.py|.diff`；`scripts/`：run_build/run_correctness/run_perf 与输出
