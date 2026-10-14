#!/usr/bin/env python3
"""v11-4b: nprobe configs for the integrated side.

Generates int_np{N}.yaml in this run dir from v8_base_np293.yaml by
changing the nprobe field in the QueryParam JSON (and RecallLogDir).
Existing v8 configs for other nprobes are used directly by the runner.
"""
import sys
from pathlib import Path

V8 = Path("/home/xingrui/bench/rabitq_refactor/v8")
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
RUN.mkdir(parents=True, exist_ok=True)

base = (V8 / "v8_base_np293.yaml").read_text()
assert base.count('"nprobe":293') == 1

written = []
for np in (150, 200, 249):
    out = RUN / f"int_np{np}.yaml"
    if out.exists():
        continue
    text = base.replace('"nprobe":293', f'"nprobe":{np}')
    text = text.replace("RecallLogDir: ./v8_base_np293.recall",
                        f"RecallLogDir: ./int_np{np}.recall")
    out.write_text(text)
    written.append(out.name)

existing = sorted(p.name for p in V8.glob("v8_base_np*.yaml") if p.suffix == ".yaml" and "np" in p.stem and "seg" not in p.stem and p.stem.count(".") == 0)
print(f"ok: generated {written or 'nothing (all present)'} in {RUN}")
print("v8 configs available:", ", ".join(existing))
