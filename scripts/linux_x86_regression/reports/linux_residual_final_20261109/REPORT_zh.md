# Linux x86 残余回退验证报告(fix/legacy-matrix-distance-performance @ 9819c706 → 最终候选)

日期:2026-11-09 机器:Intel Xeon 6982P-C(8C/16T,基准 CPU 9,SMT 兄弟 8 空闲)
Linux 6.8.0-87-generic,gcc 13.3.0,Ninja Release `-O3 -march=x86-64`,AUTO_DETECT_ARCH=ON,ENABLE_NATIVE=OFF,无 LTO/OpenMP,
无全局对齐标志(CMakeCache / compile_commands 见 `candidate/buildinfo/`)。perf 硬件计数器不可用(perf_event_paranoid=4)。

本报告仅覆盖 Linux x86。**macOS/ARM64 尚未用最终补丁重新验证**(见 §7)。

## 0. 结论摘要

| 目标 | 结果 |
|---|---|
| 正确性(44 个程序) | 最终候选 511 通过 / 20 跳过 / 25 禁用 / 0 失败,全部退出码 0(955 与 7abc 各 510 通过,新增测试 `TurboMetricAdapter.Int4RecordMatricesPreserveStoredMoments` 通过) |
| 旧索引兼容 | parent_write 仅已知 2 项失败(`RawTypes/IVFLegacyMatrixTest.*/5`, GetParam=(4,true)),200 个 fixture 写出;before_ivf 12/4/0/16;手动 fixed_read_parent 退出 0,12 通过/4 跳过/16 禁用/0 失败,strace 记录 200 个 `.index` 全部打开成功、0 个 ENOENT |
| A:13 项整数矩阵回退 | 12 项两轮都快于 7abc 且不高于 955 +5%;第 13 项 INT8 L2 2x2 d128 两轮为 0.933/1.005×7abc、1.002/1.031×955(与 955 持平,指令流与 955/7abc 完全相同) |
| B:INT4 记录量化 128 维 | RecordL2/RecordIP 32x1 d128:0.52–0.54× before、0.48–0.50× 7abc;d512 0.52–0.53×;旧版 IVF INT4 record 0.57–0.64× before/7abc。32x32 记录路径持平(0.98–1.01) |
| C:已恢复路径保护 | FP16 矩阵/适配器/IVF、INT8 记录 32x1/32x32、FP32 MIPS、INT8 融合 MIPS、INT8 单列:两轮均 0 项 >5%(各组中位数 0.97–1.01×7abc) |
| D:新增 >5%(两轮都复现) | 矩阵 3 项:FP32 L2 1x1 d128(1.07×955 / 1.06–1.08×7abc)、FP32 L2 1x1 d512(1.06×7abc,1.02×955)、INT8 L2 2x1 d128(1.07×955,1.03×7abc)。三者在 955/7abc/FINAL 三个二进制中的指令流逐条相同(归一化反汇编 diff=0),仅为布局效应,且都属于 1x1 / 2x1 微形状;适配器与 IVF 两轮均 0 项 |
| 整体 | 原始矩阵 >5% 项数:9819c706 单轮 23(vs 7abc)/27(vs 955) → 最终候选两轮 2/2;INT4 矩阵中位数 0.87×(955 与 7abc),INT4 单列 0.82–0.84×,INT4 多列 0.87–0.89× |

最终补丁相对 9819c706 **只改 `src/turbo/distance/matrix.h`**(`patches/final_vs_9819c706.patch`,SHA256 见 SHA256SUMS)。
未提交、未推送。

## 1. 校验与版本

- 分支 `fix/legacy-matrix-distance-performance`,HEAD = `9819c706e49d60150886cdefcc8592aa81b46019`(含 7abc63fe)。
- 交接包 `residual-performance-handoff` SHA256SUMS 校验通过;toolkit、deps(`zvec-x86-regression.04K3Yl/fixed`,子模块已初始化)和默认编译标志与前两轮相同。
- 五个版本(全部为 955ae841 分离工作树 + 补丁,补丁 SHA256 与交接包一致):
  - parent 499eab89,before f7bd575f(各根目录内 `parent/`、`before/`)
  - OLDER = 955ae841 + baseline_test_build.patch(`zvec-residual-955.Va6ctj`)
  - REF = 955ae841 + reference_7abc.patch(=7abc63fe,`zvec-residual-7abc.PfRBo1`)
  - RUN = 955ae841 + candidate.patch(=9819c706,SHA256 c711f68e…,`zvec-residual-new.en2dYQ`)
  - FINAL = 955ae841 + final_vs_955ae841.patch(`zvec-residual-final.PZh6kP`,补丁 SHA256 5e1d3ee9…)
- 证据目录命名:`evidence/{older955,reference7abc,candidate}`;9819c706 的第一轮数据在 `candidate_9819c706_round1/`;诊断变体在 `diagnostics/`。

## 2. 9819c706 的验证结果(第一轮,未通过)

9819c706 在 INT4 `ComputePackedColumns<Columns>` 与 `ComputeSingleColumn<WithNorms>` 上加 GCC/x86 专用
`__attribute__((noinline, aligned(64), optimize("align-loops=64")))`,并为 INT4 记录量化 n==1 增加批处理路径。

- 正确性:511 通过 / 0 失败;fixture 读写同上。
- 性能(`candidate_9819c706_round1/vs_*_round1/`):原始矩阵 >5% 项:23(vs 7abc)/ 27(vs 955);适配器 2 项(INT4 MIPS 32x32 1.08–1.10×7abc,但 0.95×955);IVF 0 项。
- 13 项中仅部分改善,且出现新的 >5% 项(INT4 L2 16x16 1.19×,INT4 L2 2x1 1.18×,INT4 IP 32x16 1.18×,INT8 L2 2x2 1.09–1.12× 等)。
- 反汇编(`candidate_9819c706_round1/disasm/`):属性生效(86 个 INT4/INT8 内核符号全部 64 字节对齐),但热循环仍是 53 条指令/8 次查表的标量 shr/and/movzbl 序列,属性只是让布局在不同编译单元间“重新洗牌”,并未消除问题;同时 40 个 INT8 `ComputeSingleColumn` 克隆被 noinline 外联,使 INT8 2x1/2x2/4x1 微形状增加调用开销(INT8 L2 2x2 d128 1.166×955)。
- 目标 B 已达成(INT4 RecordL2/IP 32x1 ≈0.54× before),该部分原样保留在最终补丁中。

结论:对齐属性方案不能收敛,改为降低 INT4 内核指令数(让它不再受布局左右)。

## 3. 诊断变体(单轮,仅用于定位,全部保存在 `diagnostics/`)

| 变体 | 内容 | 结果 |
|---|---|---|
| A(`variantA_restructure_with_attrs`) | 9819c706 + 重构 INT4 多列内核(查询半字节展开 + 每个矩阵字 8 个表行指针),保留属性 | INT4 多列 0.56–0.78× 955;4x4 退化 1.10–1.13;INT8 2x2 仍 1.17–1.20(noinline) |
| B(`variantB_restructure_no_attrs`) | 同 A 但移除两处属性 | INT4 多列与 A 相同(属性无贡献);INT8 2x2 恢复 1.05/1.01;但 INT4 单列 32x1 出现 1.09–1.17(布局噪声) |
| C(`diagC_*`) | B + 对列循环 `#pragma GCC unroll 1` | 修好 4x4(0.83×B),但 4x2/16x4/32x4/2x2 退化 5–19% → 否决 |
| v1(`candidate/variants/v1_*`) | B 的内核,Rows≤4 走旧循环,无属性 | INT4 多列全部 ≤1.00(4x4 ≈1.00),但 INT4 32x1 仍 1.09–1.14 |
| FINAL(v2) | v1 + INT4 单列内核改为“查询表行提升” | INT4 单列 0.75–0.83×,见 §4 |

## 4. 最终补丁内容(`patches/final_vs_9819c706.patch`,仅 matrix.h)

1. 删除两处 `__attribute__((noinline, aligned(64), optimize("align-loops=64")))`(数据表明它既不能稳定 INT4,又让 INT8 微形状变慢)。
2. `ComputePackedColumns<Columns>`:`if constexpr (Rows <= 4)` 保留原始循环(小块被完全展开,展开式会把字节块溢出到栈);`Rows >= 8` 时每个字位置先把查询块展开为 `Columns*8` 个半字节字节(SWAR 偶/奇掩码),每个矩阵字解析一次 8 个表行指针,之后每对半字节只需一次字节加载 + 一次查表。内循环 31 条指令/8 次查表(原 53)。
3. `ComputeSingleColumn`(uint8_t 分支):半字节对表 `PackedDistances()` 对 (a,b) 对称(L2 为 (a-b)²,IP 为 -a·b),因此改为先按查询半字节索引,把 8 个查询表行提升到寄存器,行循环内每次查表只做 shr/and/movzbl/add。行循环 39 条指令/8 次查表(原 53),见 `candidate/disasm/single32_int4_l2_final.asm`。
4. 保留 9819c706 的 x86 INT4 记录 n==1 批处理路径(`quantized_integer_metric.cc`)、7abc63fe 的 `mips_euclidean_metric.cc` 改动与全部测试。

语义:查表值、累加顺序(每个 8 维字一次整数求和后转 FP32)与原内核一致;`turbo_matrix_test`、`quantized_integer_metric_test`、`ivf_legacy_matrix_test` 等 511 项全部通过。

## 5. 性能(两轮,`candidate/vs_{955,7abc}_round{1,2}/`,`candidate/analysis/summary.md`、`tables.md`)

轮次安排:第一轮 OLDER→REF→FINAL,第二轮 FINAL→REF→OLDER(每轮内每个二进制 3 进程轮换,中位数)。
注意:OLDER 根第二轮 IVF 的 parent/before 漂移中位数 0.93(该根第二轮整体偏慢),因此 FINAL/955 的 IVF 第二轮比值偏乐观,以第一轮(漂移 0.996)为准;FINAL/7abc 两轮漂移均 ≈1.00。

### 5.1 >5% 项计数

| 组 | 轮 | vs 955 | vs 7abc | vs before | vs parent | 中位数 FINAL/955 | FINAL/7abc |
|---|---|---|---|---|---|---|---|
| 矩阵 504 | r1 | 2 | 2 | 15 | 68 | 0.997 | 0.999 |
| 矩阵 504 | r2 | 2 | 2 | 15 | 71 | 0.998 | 0.999 |
| 适配器 32 | r1/r2 | 0/0 | 0/0 | 0/0 | 0/0 | 0.847/0.862 | 0.998/0.997 |
| IVF 24 | r1/r2 | 0/0 | 0/0 | 0/0 | 0/0 | 0.990/0.874 | 0.991/0.989 |

(vs before/parent 的矩阵项均为 7abc 之前就存在的 FP64/INT16/1x1 等继承项,与 7abc 报告中的列表一致;FP16 相对 before/parent 为 0.05–0.2×。)

### 5.2 目标 A:13 项(ns/matrix)

| 类型 | 度量 | 形状 | dim | 955 r1/r2 | 7abc r1/r2 | 9819 | FINAL r1/r2 | FINAL/7abc r1/r2 | FINAL/955 r1/r2 | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| INT4 | IP | 16x2 | 128 | 1152.6/1135.7 | 1246.2/1252.4 | 1223.9 | 1084.2/1086.8 | 0.870/0.868 | 0.941/0.957 | 通过 |
| INT4 | IP | 16x2 | 512 | 4551.7/4475.4 | 4945.3/5020.1 | 4753.5 | 4273.2/4268.5 | 0.864/0.850 | 0.939/0.954 | 通过 |
| INT4 | IP | 32x32 | 128 | 36803.9/36944.8 | 41644.9/40937.5 | 42282.7 | 28684.0/28520.7 | 0.689/0.697 | 0.779/0.772 | 通过 |
| INT4 | IP | 32x32 | 512 | 146343.8/146164.5 | 152562.2/161816.4 | 165691.5 | 115494.7/113948.3 | 0.757/0.704 | 0.789/0.780 | 通过 |
| INT4 | L2 | 32x1 | 128 | 1215.5/1182.1 | 1265.7/1223.8 | 1234.3 | 991.9/991.8 | 0.784/0.810 | 0.816/0.839 | 通过 |
| INT4 | L2 | 32x1 | 512 | 5021.9/4653.0 | 4998.5/4985.0 | 4921.3 | 3756.8/3737.9 | 0.752/0.750 | 0.748/0.803 | 通过 |
| INT4 | L2 | 32x16 | 128 | 17893.6/17865.4 | 24264.4/24154.1 | 20066.9 | 13480.2/13466.3 | 0.556/0.558 | 0.753/0.754 | 通过 |
| INT4 | L2 | 32x16 | 512 | 71309.5/71656.2 | 84494.8/96172.2 | 81056.6 | 53768.2/53590.3 | 0.636/0.557 | 0.754/0.748 | 通过 |
| INT4 | L2 | 32x2 | 128 | 2256.1/2224.0 | 2424.3/2400.1 | 2401.1 | 2238.4/2260.6 | 0.923/0.942 | 0.992/1.016 | 通过 |
| INT4 | L2 | 32x2 | 512 | 8857.9/8932.1 | 9682.1/9498.1 | 9588.6 | 8921.1/8915.9 | 0.921/0.939 | 1.007/0.998 | 通过 |
| INT4 | L2 | 8x8 | 128 | 2574.6/2575.3 | 2781.5/2920.6 | 2732.5 | 1828.4/1808.4 | 0.657/0.619 | 0.710/0.702 | 通过 |
| INT4 | L2 | 8x8 | 512 | 10335.3/10403.0 | 12038.1/11184.7 | 10145.5 | 7256.8/7183.6 | 0.603/0.642 | 0.702/0.691 | 通过 |
| INT8 | L2 | 2x2 | 128 | 141.9/141.2 | 152.4/144.8 | 165.4 | 142.1/145.5 | 0.933/1.005 | 1.002/1.031 | 与 955 持平(见下) |

INT8 L2 2x2 d128:最终候选的 INT8 代码与 955/7abc 源码相同(属性已去掉),`kernel<signed char,2,2,false>` 在三个二进制中的 198 条指令逐条相同(`candidate/disasm/residual/`)。7abc 第一轮 152.4、第二轮 144.8 本身波动 5%;FINAL 两轮 142.1/145.5,与 955 的 141.9/141.2 在 3% 内,两轮均未触发 >5%。该项记为“持平、布局噪声”,未作进一步修改(任何修改都只是再一次洗牌)。

### 5.3 目标 B:INT4 记录量化(ns/matrix;IVF 为 us/query)

| 用例 | parent | before | 955 | 7abc | 9819 | FINAL r1/r2 | FINAL/before r1/r2 | FINAL/7abc r1/r2 |
|---|---|---|---|---|---|---|---|---|
| INT4 RecordL2 32x1 d128 | 1181 | 987 | 1354 | 1060 | 527 | 516/514 | 0.523/0.536 | 0.487/0.489 |
| INT4 RecordL2 32x1 d512 | 4647 | 2810 | 5424 | 2799 | 1555 | 1493/1490 | 0.531/0.530 | 0.534/0.532 |
| INT4 RecordIP 32x1 d128 | 1297 | 960 | 1380 | 993 | 516 | 499/496 | 0.519/0.519 | 0.502/0.480 |
| INT4 RecordIP 32x1 d512 | 5131 | 2808 | 5306 | 2805 | 1538 | 1478/1473 | 0.526/0.520 | 0.527/0.519 |
| INT4 RecordL2 32x32 d128/d512 | 40686/160262 | 10003/27869 | 9980/27379 | 10017/27399 | 10035/27392 | 9964/27438 | 0.996/0.985 | 0.995/1.001 |
| INT4 RecordIP 32x32 d128/d512 | 40431/159305 | 9321/27721 | 9341/27298 | 9406/27260 | 9398/27291 | 9303/27401 | 0.998/0.988 | 0.989/1.005 |
| IVF INT4 record IP d128 / d512 | 382/1388 | 309/852 | 404/1395 | 316/852 | 196/493 | 191/485 | 0.619/0.569 | 0.605/0.569 |
| IVF INT4 record L2 d128 / d512 | 368/1304 | 309/850 | 398/1413 | 321/857 | 198/496 | 195/486 | 0.629/0.571 | 0.607/0.567 |

### 5.4 目标 C:受保护路径(两轮内最差 FINAL/7abc)

| 路径 | 用例数 | 两轮都 >5% | 中位数 FINAL/7abc | 最差项 |
|---|---|---|---|---|
| FP16 原始矩阵 | 84 | 0 | 0.987 | FP16 IP 1x1 d128 0.956/1.042 |
| FP16 适配器 MIPS | 4 | 0 | 1.006 | 32x32 d128 1.006/1.020(0.17×955) |
| FP16 IVF | 4 | 0 | 0.966 | IP d128 0.992/0.953(0.09×955) |
| INT8 记录 32x1/32x32 | 8 | 0 | 1.001 | RecordL2 32x32 d512 1.025/0.992 |
| FP32 MIPS | 4 | 0 | 0.998 | 32x32 d128 1.014/0.999 |
| INT8 融合 MIPS | 4 | 0 | 1.000 | 32x1 d128 1.009/0.996 |
| INT8 单列矩阵 | 24 | 0 | 1.000 | INT8 L2 2x1 d128 1.027/1.027(1.070/1.076×955,见 §5.5) |
| INT8 矩阵全部 | 84 | 0 | 1.000 | INT8 L2 4x4 d128 1.048/1.016 |
| INT4 矩阵全部 | 84 | 0 | 0.875 | INT4 IP 4x2 d512 1.000/1.048 |
| FP64/INT16(继承列表) | 168 | 0 | 1.001 | INT16 IP 1x1 d512 1.032/0.999 |
| 旧版 IVF 24 项 | 24 | 0 | 0.997 | INT8 IP d512 raw 0.993/1.026 |

### 5.5 目标 D:两轮都复现的 >5% 项与定位

| 用例 | FINAL/955 r1/r2 | FINAL/7abc r1/r2 | 绝对值(ns) | 定位 |
|---|---|---|---|---|
| FP32 L2 1x1 d128 | 1.073/1.075 | 1.062/1.079 | 955 7.99 → FINAL 8.57 | `kernel<float,1,1,false>`(77 条)及其间接调用的 `squared_euclidean_fp32_distance_{avx2,avx512,scalar}` 在 955/7abc/FINAL 中归一化反汇编 diff=0;补丁不触及 FP32 路径;差值 0.5 ns ≈ 2 周期/调用,属 1x1 继承类布局项 |
| FP32 L2 1x1 d512 | 1.023/1.024 | 1.058/1.059 | 22.7 → 24.0 | 同上(同一内核) |
| INT8 L2 2x1 d128 | 1.070/1.076 | 1.027/1.027 | 73.3 → 78.4 | `kernel<signed char,2,1,false>` 182 条指令三版本完全相同(INT8 `ComputeSingleColumn` 内联,源码同 955);vs 7abc 仅 2.7%;9819c706 单轮时同形状 d512 已为 1.05 |

三项均为微形状(单次 8–80 ns)、代码逐字节相同,是函数在二进制中绝对位置变化引起的前端/分支布局效应,不是逻辑退化。前一轮报告已证明此类项在 `-falign` 诊断构建下消失;本轮不再为它们改代码。

### 5.6 五版本 32x1 / 32x32 / 旧版 IVF 完整表

见 `candidate/analysis/tables.md`(含 parent、before、955、7abc、9819c706、FINAL 两轮与三组比值),以及 INT4 全部多列形状与 INT8 小形状表。

## 6. 残余列表

新增(本轮,两轮复现,均已定位为布局项,代码相同):
- FP32 L2 1x1 d128(vs 955/7abc)、FP32 L2 1x1 d512(vs 7abc)、INT8 L2 2x1 d128(vs 955)。

继承(相对 before/parent 的 FP64/INT16/1x1 等列表):矩阵 vs before 两轮均 15 项,是 7abc 同一列表(16 项)的子集(FP32 IP 1x1 d128 不再超标);vs parent 两轮交集 68 项,其中 65 项在 7abc 列表(87 项)内;新增 3 项 FP32 L2 1x1 d512(1.065×parent,955 为 1.042)、FP64 L2 32x8 d128(1.053,955 为 1.084)、INT16 IP 8x8 d128(1.059,955 为 1.063)均为阈值附近、955 本身同水平。整数 1x1 相对 parent 的 1.1–50× 为 955 起即存在的已知差异(parent 走不同路径),955/7abc/FINAL 三者一致;INT4 IP 2x1/8x1 vs parent 由 955 的 1.13–1.23 降至 1.07–1.17。本轮相对 955/7abc 继承类均 ≤ +3.2%(§5.4 最后两行)。

已关闭:13 项中 12 项(两轮快于 7abc,且 ≤ 955 +2%);INT8 L2 2x2 d128 与 955 持平。INT4 记录量化 128/512 维 32x1 及 IVF INT4 record 全部 ≈0.5–0.64× before。

## 7. 需要注意

- `src/turbo/distance/matrix.h` 是跨平台共享代码:`ComputePackedColumns` 与 `ComputeSingleColumn`(uint8_t 分支)的改动**没有 x86 宏保护**,ARM64/macOS 会编译同一份内核(Clang 不会受影响的只是已被删除的 GCC 属性)。**macOS 必须用 `patches/final_vs_9819c706.patch`(或 `final_vs_955ae841.patch`)重新跑正确性和性能后,才能宣称两平台无退化。** `quantized_integer_metric.cc` 中的 INT4 记录批处理路径仍为 x86-only(来自 9819c706,未改)。
- 两轮之间 OLDER 根第二轮 IVF 整体偏慢(漂移 0.93),已在 §5 说明;对结论无影响(vs 7abc 两轮稳定,vs 955 第一轮为准)。
- 未提交、未推送;所有工作树为分离 HEAD,原始仓库 HEAD 仍为 9819c706。

## 8. 证据与路径

根目录:`/home/xingrui/workspace/zvec/ivf_matrix/zvec-residual-final.PZh6kP/evidence/`
- `patches/`:`final_vs_955ae841.patch`(完整)、`final_vs_7abc63fe.patch`、`final_vs_9819c706.patch`(仅 matrix.h,已验证 `git apply --check` 通过)、`final_matrix_only.patch`、交接包三份补丁副本。
- `candidate/`:FINAL 的 build/correctness/perf 日志,`results_round1/`(44 个 XML+log、correctness.json、parent_write、before_ivf、manual_fixed_read_parent.{xml,log,exit,strace,opened_index_paths.txt}、第一轮 perf CSV/performance.json)、`results_round2/`、`vs_{955,7abc}_round{1,2}/`、`analysis/`(summary.md/json、tables.md)、`disasm/`(新内核反汇编 + `residual/` 三版本归一化对比)、`buildinfo/`(三个 CMakeCache + compile_commands)、`variants/`(v1 补丁与三次 sanity CSV、v2 sanity CSV)、`lscpu.txt`、`environment.json`、驱动脚本。
- `older955/`、`reference7abc/`:两轮 results、日志、反汇编、修订与补丁 SHA256。
- `candidate_9819c706_round1/`:9819c706 的第一轮完整结果、对比与反汇编。
- `diagnostics/`:变体 A/B/C 的补丁、perf CSV、对比与反汇编;`matrix_variantB.h`、`matrix_variantC.h`、`matrix_final.h`。
- 工作树(未打包,含构建产物与 200 个 fixture):`zvec-residual-{955.Va6ctj,7abc.PfRBo1,new.en2dYQ,diagA.shy9h7,diagB.Wdz3Iy,final.PZh6kP}`,均在 `/home/xingrui/workspace/zvec/ivf_matrix/`。
- 打包:`zvec-residual-x86-final-evidence.tar.gz` 与 `.sha256`(路径见交付消息)。
