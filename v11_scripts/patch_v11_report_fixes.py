#!/usr/bin/env python3
"""v11-3: in-place corrections to comparison_report_ivf_rabitq_dot_reuse.md.

Each replacement is count-asserted (must match exactly once). Anchors avoid
ambiguous quote characters. Corrections per user v11 item 3:
  - ded t4 807.2 vs v9 original 5-round median 765.1 = +5.5% (not "偏差 ≤2%")
  - "1.6 万列表/查询对" wrong: int 293,000/mode + ded 249,000 per replay round
  - delete "粗扫解释了 t1 差距的全部" (1.531 ms end-to-end vs 1.290 ms replay,
    15-25% cross-invocation drift forbids strict subtraction)
  - scan layout: int 3584B/block (32x112B) vs ded 3456B/block (32x108B);
    708B is the full quantized record width (refine only)
  - {51:327} is the 51-60 bucket (10-bucket histogram), not exact rank 51
  - bias claims narrowed to "当前 Top100 样本未见明显总体偏置"
  - f_error is the error-bound coefficient, not the residual norm
"""
from pathlib import Path

REPORT = (Path.home() / 'bench' / 'rabitq_refactor' /
          'comparison_report_ivf_rabitq_dot_reuse.md')

REPLACEMENTS = [
    # 1. exec summary item 2: deviation claim
    ('P1 配对复测复现 v9 中位数（偏差 ≤2%）：t1 137.9 / 136.6 /',
     'P1 配对复测复现 v9 中位数（t1 与集成 t4 偏差 ≤2%；专用 t4 807.2 相对\n'
     '   v9 原始五轮中位 765.1 偏高 +5.5%、相对八轮中位 771.1 偏高 +4.7%，\n'
     '   偏差 ≤2% 的说法不成立【v11 修正，见 §2】）：t1 137.9 / 136.6 /'),
    # 2. exec summary item 4: bucket semantics + bias narrowing
    ('   {51:327}，无一超过 100）；量化误差无全局偏置（GT / 非GT 均值 ≈0，',
     '   {51:327}，即 51–60 分桶（直方图按 10 分桶，非精确名次 51）\n'
     '   【v11 修正】，无一超过 100）；量化误差在当前 Top100 样本未见明显\n'
     '   总体偏置（GT / 非GT 均值 ≈0，'),
    # 3. §2 v9-median-reproduction bullet
    ('  t4 565.7→565.8、577.8→575.2、771.0→807.2（ded t4 偏高 4.7%，其余\n'
     '  ≤2%）。',
     '  t4 565.7→565.8、577.8→575.2、765.1→807.2（【v11 修正】ded t4 相对\n'
     '  v9 原始五轮中位 765.1 偏高 +5.5%、相对八轮中位 771.1 偏高 +4.7%；\n'
     '  原文误作 771.0/+4.7%，执行摘要的偏差 ≤2% 说法不成立；其余\n'
     '  三项 ≤2%）。'),
    # 4. §3 replay scale
    ('真实回放（1.6 万列表/查询对、1000 查询、5 轮交替）：',
     '真实回放（【v11 修正】原 1.6 万列表/查询对的口径有误：每次回放实为\n'
     '集成 293,000 + 专用 249,000 = 542,000 个列表/查询对，v10 以 prod/\n'
     'unpruned 两种配对各执行一次；1000 查询、5 轮交替）：'),
    # 5. §3 end-to-end anchoring
    ('  （1000/174.8）；回放 scan+prep 5.23 / 3.94 ms 占 72% / 69%——粗扫\n'
     '  解释了 t1 差距的全部。',
     '  （1000/174.8），端到端差 1.531 ms；暖态回放 scan+prep 5.23 / 3.94\n'
     '  ms 占 72% / 69%，回放差约 1.290 ms——量级相近，提示粗扫贡献 t1\n'
     '  差距的大部分；但【v11 修正】回放调用间漂移 15–25%，两个差值不可\n'
     '  相减当作严格耗时分解，粗扫解释 t1 差距全部的说法撤回。'),
    # 6. §4 bias narrowing (non-GT line)
    ('  非GT n=50146 mean≈0 std 0.00017——无全局偏置。',
     '  非GT n=50146 mean≈0 std 0.00017——当前 Top100 样本未见明显总体\n'
     '  偏置。'),
    # 7. §4 f_error terminology (integrated)
    ('  近平局（裕度 <1e-4）125/327（38%）。残差（记录 f_error）12.37 vs\n'
     '  12.35——无系统性差异。',
     '  近平局（裕度 <1e-4）125/327（38%）。f_error（量化记录携带的误差界\n'
     '  系数，非残差范数本身【v11 修正】）12.37 vs 12.35——无系统性差异。'),
    # 8. §4 f_error terminology (dedicated)
    ('std 0.00014、裕度 mean 0.00014、近平局 132/249（53%）、残差 12.31 vs\n'
     '12.34——同模式、噪声略低。',
     'std 0.00014、裕度 mean 0.00014、近平局 132/249（53%）、f_error（误差\n'
     '界系数）12.31 vs 12.34——同模式、噪声略低。'),
    # 9. §4 conclusion bias narrowing
    ('结论：排序损失是方差驱动的边界交换，不是偏置。证据：①误差分布零均值',
     '结论：当前 Top100 样本未见明显总体偏置，排序损失与方差驱动的边界交换\n'
     '一致（【v11 修正】收窄原表述）。证据：①误差分布零均值'),
    # 10. §7 scan layout numbers
    ('  块的反转机制）：候选假设为记录布局（turbo 708B 含 u32 前缀 vs\n'
     '  rabitqlib 696B BinDataMap）、缓存/预取行为、对齐加载；需 perf c2c /',
     '  块的反转机制）：候选假设为粗扫块布局（【v11 修正】768 维粗扫实际\n'
     '  读取为集成 3584B/块 = 32×112B，含 u32 模型 ID 前缀 + 96B 1-bit 码\n'
     '  + 12B 系数；原生 3456B/块 = 32×108B BinDataMap 批布局。708B/696B\n'
     '  是完整量化记录宽度，仅参与精排，非粗扫每向量读取量）、缓存/预取\n'
     '  行为、对齐加载；需 perf c2c /'),
]


def main():
    text = REPORT.read_text()
    n = 0
    for old, new in REPLACEMENTS:
        cnt = text.count(old)
        assert cnt == 1, f'anchor matched {cnt} times (expect 1): {old[:60]!r}'
        text = text.replace(old, new)
        n += 1
    REPORT.write_text(text)
    print(f'applied {n}/{len(REPLACEMENTS)} in-place corrections to {REPORT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
