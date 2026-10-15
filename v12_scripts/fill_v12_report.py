#!/usr/bin/env python3
"""v12 report writer: narrow the v11-4 conclusions in place and append the
v12 section (§8) to comparison_report_ivf_rabitq_dot_reuse.md.

Edits (v11 in-place narrowing, per the v12 review):
  1. §5a method line: drop the "免疫跨调用漂移" claim (same-invocation
     pairing reduces but does not eliminate host-noise drift).
  2. §5a: add a factor-scope caveat block (cluster0/dedshape are replay
     ablations, not drop-in production forms; flat/flat16 are locality
     probes only).
  3. §5b table note: mark the deprecated best-round convention.
  4. §5b conclusion header: scope the failure to the lower_bound-keyed
     candidate selection; do not generalize to "rerank has no benefit".
  5. §5b point 3: same scoping for the refine-replacement claim.
  6. §5b design note: correct the "忠实实现" claim (lower_bound-keyed heap
     skips RaBitQ refine; v12 adds the refine-keyed variant).

Then appends §8 with the v12 experiment record (validation, sweep,
isfinite fix, timing protocol, gated production experiment, delivery).
"""
from pathlib import Path

REPORT = Path('/home/xingrui/bench/rabitq_refactor/'
              'comparison_report_ivf_rabitq_dot_reuse.md')

EDITS = [
    # 1. drift-immunity claim
    ('方法：回放内配对（同一调用含变体行与生产行，免疫跨调用漂移），unpruned',
     '方法：回放内配对（同一调用含变体行与生产行，降低但不消除跨调用漂移——'
     '主机级噪声仍会同轮抬升全部行），unpruned'),
    # 2. factor-scope caveat after the 5a conclusion block
    ('单因素结论不进生产搜索验证\n'
     '  （均 <10%，不足以支撑改造）；后续如需再压缩，方向是位图走查（最大\n'
     '  单因素）或直接降低扫描量。commit a0cac263（feat(ivf): single-factor\n'
     '  coarse-scan variants in scan replay）。',
     '单因素结论不进生产搜索验证\n'
     '  （均 <10%，不足以支撑改造）；后续如需再压缩，方向是位图走查（最大\n'
     '  单因素）或直接降低扫描量。commit a0cac263（feat(ivf): single-factor\n'
     '  coarse-scan variants in scan replay）。\n'
     '- 因素口径备注（v12 收窄）：cluster0/dedshape 是回放内核消融变体\n'
     '  （BatchDataMap 系数布局、去 key-keep 位图走查），非可直接落地形态；\n'
     '  dedshape 行宽 3584B 与生产 3456B 另含布局差异；flat/flat16 仅隔离\n'
     '  映射段局部性与对齐（页对齐连续副本，行序不变）；单因素结论仅用于\n'
     '  归因，不构成生产收益预测。'),
    # 3. best-round convention deprecated
    ('故用最佳轮与逐轮配对比；跨调用绝对值仅作参考。）',
     '故用最佳轮与逐轮配对比；跨调用绝对值仅作参考。）\n'
     '（v12 注：本表"最佳轮"口径已废弃——v12 起统一保留全部轮次并报告\n'
     '中位数与逐轮配对比；此处保留原始口径仅作历史记录。）'),
    # 4. conclusion header scoping
    ('**结论：同召回下无 QPS 收益**【实测】：',
     '**结论（v12 收窄口径）：按 lower_bound 选候选的替代方案失败——'
     '无法达到目标召回；该失败不能推广为"精排无收益"**【实测】：'),
    # 5. point 3 scoping
    ('原始向量精排不能取代 RaBitQ refine（TopK=50 下）。',
     '在按 lower_bound 选候选的前提下，原始向量精排不能取代 RaBitQ refine\n'
     '   （TopK=50 下）；v12（§8）以 refine 键控堆另行验证，结论相反。'),
    # 6. faithful-implementation claim
    ('设计说明：候选集与生产共享 rejected() 判据（lower_bound > 当前堆最差\n'
     '则跳过），差异仅堆大小 N 与计分来源（lower_bound vs refine 后分数），\n'
     '因此本实验即"生产剪枝路径取 Top-N + 原始向量精排"的忠实实现。',
     '设计说明（v12 修正）：候选集与生产共享 rejected() 判据（lower_bound >\n'
     '当前堆最差则跳过），但堆按 lower_bound 维护、跳过了 RaBitQ refine，\n'
     '与生产堆按 refine 分数维护不同——本实验是"lower_bound 选候选的\n'
     'Top-N + 原始向量精排"变体，而非"生产剪枝路径取 Top-N"的忠实实现；\n'
     'v12（§8）以 refine 键控堆补齐该差异。'),
]

SECTION = '''
## 8. v12：精排方案修正与生产验证（refine 键控 Top-N）【实测】

v11-4b 的结论在 v12 被定位为方案缺陷而非方向失败：v11-4b 堆按
lower_bound 维护（跳过 RaBitQ refine），候选覆盖不足；生产堆实际按
refine 分数维护。v12 用 refine 键控堆重做该实验并推进到生产测速。

### 8a 方案修正与逐查询验证

- rrfN（新方案）：保留生产粗扫与下界剪枝；对通过剪枝的候选调用
  scanner->refine()；按 refine 分数维护大小 N 的堆，剪枝阈值也来自该
  堆；对最终 N 个候选读原始向量精确精排返回 Top50。rlbN（对照）：
  v11-4b 原方案（lower_bound 键控）。
- 逐查询验证（同一索引/查询/nprobe，精排前候选 vs 生产 topk=N 的
  ID 与分数，等分须 ID 与分数全同）：N=60/80/100 全部
  `queries=1000 exact=1000 tie=0 id_mismatch=0 score_mismatch=0
  max_score_delta=0`（v12_validate_summary.txt）。
- rlbN 对照召回精确复现 v11-4b（np293：rlb60 82.530% / rlb80 89.730% /
  rlb100 93.470%，与 §5b 表逐位一致）——v11-4b 完全由 lower_bound
  键控解释。

### 8b nprobe 扫描（回放，7 nprobe × {rrf,rlb} × N=60/80/100）

| nprobe | prod50 µs/q | recall | rrf60 µs/q | rrf60 recall | 同 nprobe 配对比 |
|---|---|---|---|---|---|
| 150 | 3719.0 | 98.202% | 3938.5 | 98.738% | x1.054 |
| 200 | 4502.3 | 98.644% | 4603.1 | 99.264% | x1.054 |
| 249 | 5847.7 | 98.886% | 6046.5 | 99.544% | x1.011 |
| 272 | 5620.4 | 98.968% | 6088.9 | 99.636% | x1.061 |
| 293 | 6817.8 | 99.002% | 7007.4 | 99.682% | x1.051 |
| 320 | 7156.8 | 99.042% | 7511.4 | 99.732% | x1.031 |

同 nprobe 下 rrf60 比 prod50 慢 ~5%（refine/q 765k vs 自适应 662k +
精排 111µs/q），收益来自降 nprobe：**rrf60@np200 = 99.264% @
4603.1µs/q**，优于 prod50@np293（99.002% @ 6817.8µs/q，×0.675，
跨调用口径仅指示性）。np200 处 N 对比：rrf60 4603.1/99.264%，
rrf80 4924.3/99.280%，rrf100 5307.8/99.284%——N=60 为前沿点。
rlb60@np200 4378.0µs/q 但召回仅 82.362%（对照失败）。
v8 召回锚点全程精确复现（np272 98.968%、np293 99.002%、
np320 99.042%）。命令/参数/逐轮原始数据：v12_runs/（run_v12_timing.sh、
v12_analysis.out、iv4b_np*.replay.csv 续用 v11 录制，lists.csv 与
topk 无关已证 md5 相同）。

### 8c 验证器 NaN/Inf 漏检修复

样本值、回放计算值与 tolerance 增加 std::isfinite() 检查，非有限值
与负 tolerance 返回非零。负对照（NaN/Inf 样本注入）rc=1、正常对照
rc=0（v12_analysis.out）。commit 见 §8f（独立提交）。

### 8d 计时口径修正

baseline 与各变体执行顺序逐轮轮换，统一预热，5 轮完整配对，全部
轮次保留；报告中位数与逐轮配对比，取消"各配置最佳轮"为主要结果，
不再声称同调用内测量免疫漂移（§5a/§5b 已就地收窄）。

### 8e 生产实验（可关闭入口，t1/t4 QPS）

入口即 yaml `QueryConfig.RefinerConfig` 段（bench 工具已有解析）：
`ScaleFactor: 1.2` → 粗堆扩至 floor(50×1.2)=60（refine 键控，阈值来自
该堆，与已验证的 rrf60 语义一致），kFlat 参考索引对 60 候选精确重排；
删除该段即纯生产路径。生产搜索算法零改动（ivf_index.cc 冲突检查对
默认 rabitq 扫描项 + refiner 放行）。

kFlat 参考索引格式发现：local_builder（FlatBuilder+FileDumper）输出
package 布局，仅 MMapFileReadStorage 消费方可读（IVFIndex::open）；
refiner 走基类 Index::open → MMapFileStorage → IndexMapping，对
package 布局的占位 meta_footer_offset 返回 Unsupported。参考索引必须
经核心接口 API 创建（open kMMAP create_new + add + close，映射容器
布局）——flat_ref_builder 工具（3.09GB，含 id map 供 p_keys 查询）。

召回先行验证（run_v12_prod.sh，先验证后测速）：
`prod50@np293 = 99.002%`（生产复刻，v8 锚点一致）、
`rrf60@np200+refiner = 99.264%`（与回放逐位一致——生产路径忠实
复现已验证方案）。

t1/t4 QPS（5 轮轮换 × 30s，全部保留；共享主机，loadavg 3–19，
r1 遇负载尖峰）：

| 配置 | r1 | r2 | r3 | r4 | r5 | 中位 |
|---|---|---|---|---|---|---|
| prod50 np293 t1 | 101.7 | 151.2 | 138.8 | 132.3 | 138.9 | 138.8 |
| rrf60 np200 t1 | 139.3 | 206.4 | 193.1 | 186.3 | 197.8 | 193.1 |
| prod50 np293 t4 | 332.1 | 620.0 | 548.6 | 597.9 | 568.9 | 568.9 |
| rrf60 np200 t4 | 834.1 | 864.7 | 738.9 | 790.8 | 783.7 | 790.8 |

同轮配对比（rrf60/prod50 QPS）：t1 = 1.370/1.365/1.391/1.408/1.424
（中位 ×1.391）；t4 = 2.512/1.395/1.347/1.323/1.378（中位 ×1.378；
r1 的 2.512 对应 loadavg 19 负载尖峰期，保留不删）。

**结论：召回不低于目标（99.264% vs 99.002%）时，refine→Top-60→精确
精排的生产路径带来 ×1.39（t1）/ ×1.38（t4）QPS，5 轮配对比稳定。**
回放口径 ×0.675（时间）高于端到端比值，差值为回放范围外开销
（探测选择、finalize 等 ~1.4ms/q）——回放时间不能直接当端到端 QPS，
与交付要求一致。v10 的绝对 QPS（137.9/565.8）为不同负载日测量，
跨日绝对值仅作参考。

### 8f 交付

- commit 1（验证器修复）：fix(ivf): reject non-finite values in replay
  sample verification。
- commit 2（精排实验）：feat(ivf): refine-keyed Top-N rerank with
  production verification（replay 工具 + 验证/计时脚本）。
- commit 3（生产实验）：feat(tools): gated refiner production experiment
  for refine-keyed rerank（flat_ref_builder + 生产实验配置/脚本）。
- 脚本：v12_scripts/（patch_v12.py、run_v12_smoke.sh、
  run_v12_validate.sh、run_v12_timing.sh、make_v12_cfgs.py、
  analyze_v12.py、make_v12_prod_cfgs.py、run_v12_prod.sh、
  analyze_v12_prod.py、build_flat_ref.sh、fill_v12_report.py）。
- 原始日志：v12_runs/（v12_validate_summary.txt、v12_analysis.out、
  prod/v12_prod_summary.txt、prod/v12_prod_analysis.out、
  prod/*_r*.log、prod/recall_*.log、flat_ref/build_flat_ref.log）。
- v11 §5a/§5b 已就地收窄（本脚本 EDITS）；v11 低召回结论限定为
  "按 lower_bound 选候选的替代方案失败"。
'''


def main():
    text = REPORT.read_text()
    for old, new in EDITS:
        n = text.count(old)
        assert n == 1, f'edit target count={n} (want 1): {old[:60]!r}'
        text = text.replace(old, new)
    assert '## 8. v12' not in text, 'v12 section already present'
    text = text.rstrip('\n') + '\n' + SECTION
    REPORT.write_text(text)
    print(f'applied {len(EDITS)} narrowing edits + appended v12 section '
          f'-> {REPORT}')


if __name__ == '__main__':
    main()
