#!/usr/bin/env python3
"""v12: topk=N diagnostic configs at np293 (candidate verification).

Generates np293_topk{N}.yaml from v8_base_np293.yaml by changing TopK and
the QueryParam "topk" field, so the diagnostic exports production topk=N
baseline results for per-query candidate comparison.
"""
from pathlib import Path

V8 = Path('/home/xingrui/bench/rabitq_refactor/v8')
RUN = Path(__file__).parent
RUN.mkdir(parents=True, exist_ok=True)

base = (V8 / 'v8_base_np293.yaml').read_text()
assert base.count('TopK: 50') == 1
assert base.count('"topk":50') == 1

written = []
for k in (60, 80, 100):
    out = RUN / f'np293_topk{k}.yaml'
    if out.exists():
        continue
    text = base.replace('TopK: 50', f'TopK: {k}')
    text = text.replace('"topk":50', f'"topk":{k}')
    text = text.replace('RecallLogDir: ./v8_base_np293.recall',
                        f'RecallLogDir: ./np293_topk{k}.recall')
    out.write_text(text)
    written.append(out.name)
print(f'ok: generated {written or "nothing (all present)"} in {RUN}')
