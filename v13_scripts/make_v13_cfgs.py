#!/usr/bin/env python3
"""v13 configs: np293 topk=52/55 (candidate verification) and int_np160-190
diagnostic yamls (production list export for the nprobe fill-in sweep).
"""
from pathlib import Path

V8 = Path('/home/xingrui/bench/rabitq_refactor/v8')
V11 = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v11_runs')
RUN = Path(__file__).parent
RUN.mkdir(parents=True, exist_ok=True)

written = []

base = (V8 / 'v8_base_np293.yaml').read_text()
assert base.count('TopK: 50') == 1
assert base.count('"topk":50') == 1
for k in (52, 55):
    out = RUN / f'np293_topk{k}.yaml'
    if out.exists():
        continue
    text = base.replace('TopK: 50', f'TopK: {k}')
    text = text.replace('"topk":50', f'"topk":{k}')
    text = text.replace('RecallLogDir: ./v8_base_np293.recall',
                        f'RecallLogDir: ./np293_topk{k}.recall')
    out.write_text(text)
    written.append(out.name)

diag = (V11 / 'int_np200.yaml').read_text()
assert diag.count('"nprobe":200') == 1
for np_ in (160, 170, 180, 190):
    out = RUN / f'int_np{np_}.yaml'
    if out.exists():
        continue
    text = diag.replace('"nprobe":200', f'"nprobe":{np_}')
    text = text.replace('RecallLogDir: ./int_np200.recall',
                        f'RecallLogDir: ./int_np{np_}.recall')
    out.write_text(text)
    written.append(out.name)

print(f'ok: generated {written or "nothing (all present)"} in {RUN}')
