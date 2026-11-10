# 最终矩阵性能修复：报告与证据索引

当前生产代码提交：`008fbdc36737e0b4b70ee2e29a3ff12889f8061a`，分支 `fix/legacy-matrix-distance-performance`。

**服务器执行入口：[FINAL_X86_AGENT_START_HERE_zh.txt](../FINAL_X86_AGENT_START_HERE_zh.txt)。** 分支已包含最终实现，不要把历史证据包中的候选补丁重复应用到当前分支。

下面报告与证据按实验发生时的内容原样保留。里面“尚未提交/推送”“主分支为 c65ac889”等表述是当时状态；本索引和新执行说明描述当前交付状态。失败的中间候选不是当前最终候选。

| 轮次 | 结论 | 报告与完整表格 | 原始证据 |
|---|---|---|---|
| Linux residual final | Linux 正确性511通过；主要整数回退解决，保留微形状布局项 | [报告](linux_residual_final_20261109/REPORT_zh.md) | [原 Linux 交接包](../artifacts/x86-residual-final-handoff.tar.gz) |
| macOS 原样复验 Linux final | 正确性通过；发现12项INT4多列和2项MIPS回退，未通过性能门槛 | [报告](macos_residual_final_20261109/REPORT_zh.md) / [两轮表](macos_residual_final_20261109/PERFORMANCE_TABLES.md) | [证据包](../artifacts/macos-residual-final-evidence-20261109.tar.gz) |
| macOS ARM64 大多列保护 | 14项INT4/MIPS回退消除；FP16 1×1布局相关项仍在 | [报告](macos_arm64_guard_20261109/REPORT_zh.md) / [两轮表](macos_arm64_guard_20261109/PERFORMANCE_TABLES.md) | [证据包](../artifacts/macos-arm64-guard-evidence-20261109.tar.gz) |
| macOS FP16 单对直接调用（最终） | FP16回退消除；全部三组无两轮共同>5%回退，已有INT4收益保留 | [最终报告](macos_fp16_pair_20261109/REPORT_zh.md) / [最终两轮表](macos_fp16_pair_20261109/PERFORMANCE_TABLES.md) | [最终证据包](../artifacts/macos-fp16-pair-evidence-20261109.tar.gz) |
| Linux x86 最终复验（候选 vs Linux final） | 四套正确性 510/510/511/511 通过、200 旧索引×4 通过；两轮 504/32/24 相对 Linux final 无两轮共同 >5% 回退，x86 目标文件与 Linux final 逐字节一致 | [报告](linux_final_revalidation_20261110/REPORT_zh.md) / [两轮表](linux_final_revalidation_20261110/PERFORMANCE_TABLES.md) | [证据包](../artifacts/linux-x86-final-revalidation-results.tar.gz) |

最终 macOS 验证：44个程序，498通过/33跳过/25禁用/0失败；200个parent旧索引读取通过。性能范围为原始矩阵504、MIPS/记录适配器32、旧IVF warm扫描24项，五版本各3进程、两轮。第一轮有单轮波动，因此“无两轮共同>5%回退”不等于每次采样都零回退。

FP16 IP 1×1 d128 两轮：修复前12.086/11.755 ns，修复后7.535/7.870 ns。新增13行ARM64专用路径复用同一个NEON内核、移除模板入口的静态守卫/std::function开销；底层计算指令相同。

ARM64两项改动的x86交叉编译目标文件与Linux final对应实现一致；Linux 服务器已于 2026-11-09 按新说明实测（上表最后一行）：candidate 与 linux_final 在 x86 的 257 个目标文件经 prefix-map 后逐字节相同，两轮性能无两轮共同 >5% 回退，955/7abc/parent/before 对照保留。macOS agent 校验入口：[MACOS_AGENT_VERIFY_LINUX_FINAL_zh.txt](../MACOS_AGENT_VERIFY_LINUX_FINAL_zh.txt)。

每个包旁有 `.sha256` 文件。macOS包内部使用 `EVIDENCE_SHA256SUMS`，原Linux包内部使用 `SHA256SUMS`。报告中的本机绝对路径仅用于溯源；跨机运行使用新服务器说明及仓库工具。
