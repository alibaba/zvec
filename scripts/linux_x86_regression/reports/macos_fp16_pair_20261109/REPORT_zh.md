# ARM64 FP16 1×1 IP 最小修复及验证

2026-11-09，Apple M3 Pro，macOS 26.6.2，AppleClang 15。

**结论：之前完整 sweep 中 FP16 IP 1×1 d128 的 31%～37% 回退已消除。** 本次两轮 d128 为 7.535 / 7.870 ns，相对修复前下降 37.7% / 33.0%，相对 9819c706 快 16.3% / 12.1%。512 维也没有回退。三组完整回归对四个基线均没有两轮共同 >5% 的项。

## 1. 定位与最小改动

原模板 DistanceMatrix<FP16,1,1,IP> 缓存 get_distance_func 返回的 std::function。初始化/选择内核只在第一次发生，但热路径每次仍经过静态初始化守卫、80 字节栈帧、std::function 虚调用及内部函数指针跳转。此前 9819 与修复前候选的计算内核相同，完整 sweep 中的差异与包装层/整程序布局敏感性有关，不能解释成增加了转置或改变了 FP16 算术。
本轮采取直接消除这层可避免开销的办法：仅在 AILEGO_ARM64_GNU_LIKE（__aarch64__ 且 __ARM_NEON）下，M==1、T==Float16、Metric==InnerProduct 时直接调用 neon::inner_product_fp16_distance，然后返回。N 由已有 static_assert 限定为 1。

```cpp
#if defined(AILEGO_ARM64_GNU_LIKE)
if constexpr (M == 1 && std::is_same_v<T, ailego::Float16> &&
              Metric == MetricType::kInnerProduct) {
  neon::inner_product_fp16_distance(m, q, dim, out);
  return;
}
#endif
```

相对上一轮候选仅 matrix.h 增加 13 行（ARM 条件 include + 快速路径）。复用现有 NEON 内核，不改累加顺序、符号、精度或尾部处理；不添加全局对齐/优化选项。运行时工厂 MakeRawDistanceMatrix 和公开分发 API 未修改。上一轮 INT4 大多列 ARM64 条件保留。
ARM64 该内核只使用此平台已有的 NEON 能力，不依赖额外可选 ISA；与注册表的 FP16 raw IP NEON 条目一致。同进程探针确认当前自动分发的实际函数指针就是该函数。x86 和未定义此 ARM 宏的平台继续原代码。

### 定位证据

- 修复前热包装层为 31 条指令（含静态守卫、栈参数与虚调用），std::function 函数指针 thunk 另有 5 次加载与一次间接跳转；修复后基准模板包装函数仅一条直达 NEON 的尾跳转。
- 修复前后 NEON 内核均为 114 条指令，重定位地址归一化后完全相同。见 disasm/{before,final}_fp16_pair.asm、*_neon.normalized 与 fp16_pair_comparison.json。
- 同一二进制、同一输入、相同计时器交错比较 std::function 包装、direct_neon、matrix_candidate。自动分发函数指针相同，并在 0/1/3/4/7/8/9/17/127/128/129/512/513 维比较三条路径的 FP32 输出，逐字节相同。
- 没有硬件性能计数器证据进一步区分先前整程序布局变化究竟影响了哪一类分支预测或指令缓存事件；报告不把某个微架构原因写成已证实。修复依靠减少调用指令/间接跳转，并用完整 sweep 验证效果，不是加入 padding 恰好改变地址。

| 同进程探针中位数（ns/pair） | d128 | d512 |
|---|---:|---:|
| std_function | 8.912 | 45.206 |
| direct_neon | 7.882 | 43.157 |
| matrix_candidate | 7.980 | 43.304 |

探针每维每路径 5 次交错测量，每次自适应至少 20ms 窗口、5 个内部样本取中位数。小型探针的绝对耗时不替代完整 sweep 的数值。

## 2. 正确性与旧索引兼容

| 范围 | 结果 |
|---|---|
| 44 程序 | 498 通过 / 33 跳过 / 25 禁用 / 0 失败；退出码均 0 |
| FP16 直接模板与动态分发一致性 | 现有 SingleColumnExtremesTailsAndNoAllocations 通过；含 0 维、4/8 边界、非整块尾部、128/512 邻近维度、FP16 极值及抵消 |
| 历史记录矩统计 | Int4RecordMatricesPreserveStoredMoments 通过 |
| parent 499eab89 重写索引 | 200 文件；12 通过 / 4 跳过 / 16 禁用 / 0 失败 |
| 本次候选读取上述文件 | 12 通过 / 4 跳过 / 16 禁用 / 0 失败 |

测试范围为 10 metric、27 Turbo、2 flat、3 clustering、ivf_legacy_matrix、IVFSearcher 指定 13 项。已有模板测试同时执行运行时与模板入口，并比较输出和独立参考，不需要修改测试期望来适应新路径。旧索引循环覆盖 28,800 份查询结果列表。未额外宣称全仓/全产品或 sanitizer 验证。

## 3. 完整两轮性能

五版本：955ae841、7abc63fe、9819c706、before（上一轮 ARM64 大多列修复候选）、final（本次 FP16 修复）。对照为保留的静态二进制，本次全部重新运行；没有用过去轮次的计时替代。
两轮各版本各 3 个进程交错运行，第二轮反转起始顺序；每进程内部先取中位数，最后取 3 进程中位数。共 90 个完整基准进程。范围 matrix 504 / adapter 32 / IVF 24。Release -march=armv8 -O3 -DNDEBUG，ENABLE_NATIVE=OFF，无全局对齐/LTO/OpenMP；静态 force_load 链接；统一 QoS + taskpolicy -a。matrix/adapter 线程 CPU 计时，IVF warm scan 墙钟计时。
**macOS 未硬绑某个 CPU 核。** 全部编译和其他性能程序与本轮性能串行；每个版本的原始样本/min/max 均保留。

| FP16 IP 1×1 | 轮 | 9819 ns | 修复前 ns | 本次 ns | 本次/9819 | 本次/修复前 |
|---|---|---:|---:|---:|---:|---:|
| d128 | round1 | 8.998 | 12.086 | 7.535 | 0.837 | 0.623 |
| d512 | round1 | 46.323 | 45.955 | 44.802 | 0.967 | 0.975 |
| d128 | round2 | 8.949 | 11.755 | 7.870 | 0.879 | 0.670 |
| d512 | round2 | 46.056 | 46.143 | 45.458 | 0.987 | 0.985 |

### >5% 项完整计数

| 组 | 对照 | 第一轮 | 第二轮 | 两轮共同 |
|---|---|---:|---:|---:|
| matrix | 9819 | 8 | 0 | 0 |
| matrix | 7abc | 15 | 0 | 0 |
| matrix | 955 | 14 | 0 | 0 |
| matrix | before | 17 | 1 | 0 |
| adapter | 9819 | 0 | 0 | 0 |
| adapter | 7abc | 0 | 0 | 0 |
| adapter | 955 | 0 | 0 | 0 |
| adapter | before | 0 | 0 | 0 |
| ivf | 9819 | 0 | 0 | 0 |
| ivf | 7abc | 0 | 0 | 0 |
| ivf | 955 | 0 | 0 | 0 |
| ivf | before | 0 | 0 | 0 |

原始矩阵第一轮仍有单轮 >5% 项，主要在未改动的 FP64/INT16 路径；第二轮对三个历史基线均 0 项，相对 before 只有一项且与第一轮不重合。因此结果应表述为“无两轮共同复现的 >5% 回退”，而不是“每个单次测量都≤5%”。适配器和 IVF 两轮对四个基线均 0 项。

## 4. 之前的 INT4 修复保持

| 路径 | 相对 9819 中位数 R1 / R2 | 两轮 min～max |
|---|---|---|
| INT4 16×16/32×16/32×32（12项） | 0.976 / 1.001 | 0.939～1.036 |
| INT4 MIPS 32×32（2项） | 1.012 / 1.018 | 1.003～1.020 |
| INT4 单列 Rows≥4 | 0.760 / 0.742 | 0.683～0.795 |
| INT4 RecordL2/IP 32×1 | 0.758 / 0.747 | 0.742～0.765 |
| 旧 IVF INT4 record | 0.783 / 0.773 | 0.750～0.801 |

之前 14 项大多列/MIPS 回退没有再次出现，INT4 单列/记录/旧 IVF 的收益保留。IVF 性能范围为上述 warm 单查询列扫描，不代表全部业务负载。

## 5. x86 影响检查

- ARM 快速路径及其 include 位于 AILEGO_ARM64_GNU_LIKE 条件中，x86 预处理会移除。公开接口、注册表和 x86 原实现未改。
- 修复前/后完整 matrix_sweep 用相同 AppleClang、x86_64 Darwin target、Release 参数分别编译；预处理 token SHA256、完整目标文件 SHA256、所有 __TEXT 节 SHA256 均相同。见 buildinfo/x86_unchanged.json、check_x86_unchanged.py 和 x86_check/ 两个 .o。此处沿用脚本字段 unguarded/guarded，分别表示本次 FP16 修复前/后，前者已经包含上一轮 INT4 ARM64 条件。
- **这是代码路径与交叉编译产物一致性验证，不是 Linux/GCC 服务器实机重跑。** 本轮没有新的 Linux 性能数字，既有 Linux final 实测结果仍来自原交接包；不把 macOS 结果冒充 Linux 实测。

## 6. 补丁与状态

生产改动仅 source/src/turbo/distance/matrix.h。独立工作树基于 9819c706；主分支 c65ac889 干净，尚未提交、推送或合入。
- buildinfo/fp16_pair_only.patch：应用到上一轮已含 ARM64 INT4 大多列保护的候选，只增加本次 13 行。
- buildinfo/final_vs_9819c706.patch：从 9819/c65 开始的完整候选，含 Linux final、INT4 ARM64 保护及本次 FP16 修复。
- buildinfo/macos_fixes_vs_x86_final.patch：应用到已应用 Linux final、尚未加两项 ARM64 修复的源码。
三份补丁按现有基线选一，不要重复应用；详见 PATCHES_README_zh.txt。
回传证据包含两轮完整 CSV/原始日志、同进程分发探针、44 程序 XML/日志、200 fixtures、汇编及 x86 目标文件一致性验证。
