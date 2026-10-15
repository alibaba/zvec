#!/usr/bin/env python3
"""Generate v12 production-experiment bench configs.

prod50_np{293} = production reference (no refiner).
rrf60_np{200}  = gated experiment: coarse heap widened to floor(50*1.2)=60
                 (refine-keyed, threshold from that heap -- the rrf60 scheme
                 validated bit-exact in the replay), then exact rerank of the
                 60 candidates on a kFlat reference index holding the
                 original vectors. The gate is the yaml RefinerConfig section:
                 omit it and the search is plain production.
"""
from pathlib import Path

RUN = Path(__file__).parent
FLAT = RUN / 'flat_ref' / 'cohere_1m_flat_ref.index'

IVF_CONFIG = ('{"index_type":"kIVF","metric_type":"kCosine","dimension":768,'
              '"data_type":"DT_FP32","nlist":1024,"niters":20,'
              '"quantizer_param":{"type":"kRabitq","total_bits":7,'
              '"num_clusters":16,"sample_count":0,"niters":20}}')
FLAT_CONFIG = ('{"index_type":"kFlat","metric_type":"kCosine","dimension":768,'
               '"data_type":"DT_FP32"}')
INDEX = '/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index'
QUERIES = '/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt'
GT = '/home/xingrui/bench/dataset/cohere/1m/neighbors.txt'

QUERY = ('{"topk":50,"nprobe":%(nprobe)d,"radius":0,"is_linear":false,'
         '"index_type":"kIVF","fetch_vector":false,"rabitq_pruning":true,'
         '"rabitq_collect_stats":false,"rabitq_segment_timing":false,'
         '"rabitq_reuse_coarse_dot":false}')

CONFIGS = [
    # (name, nprobe, scale_factor) -- scale 0 = no refiner (gate off)
    ('prod50_np293', 293, 0.0),
    ('rrf60_np200', 200, 1.2),
]


def render(name, nprobe, scale, threads):
    def esc(s):
        return s.replace('"', '\\"')

    ivf = esc(IVF_CONFIG)
    query = esc(QUERY % {'nprobe': nprobe})
    lines = [
        'IndexCommon:',
        f'  IndexConfig: "{ivf}"',
        f'  IndexPath: {INDEX}',
        '  TopK: 50',
        f'  QueryFile: {QUERIES}',
        '  QueryType: float',
        '  QueryFirstSep: ;',
        '  QuerySecondSep: " "',
        f'  GroundTruthFile: {GT}',
        f'  RecallLogDir: ./{name}_t{threads}.recall',
        '  RecallThreadCount: 16',
        '  RecallScorePrecision: 1e-4',
        f'  BenchThreadCount: {threads}',
        '  BenchSecs: 30',
        '  BenchIterCount: 10000000',
        '  CompareById: true',
        '  LogLevel: Info',
        'QueryConfig:',
        f'  QueryParam: "{query}"',
    ]
    if scale:
        lines += [
            '  RefinerConfig:',
            f'    ScaleFactor: {scale}',
            '    ReferenceIndex:',
            f'      Config: \'{FLAT_CONFIG}\'',
            f'      Path: {FLAT}',
        ]
    return '\n'.join(lines) + '\n'


out = RUN / 'prod'
out.mkdir(exist_ok=True)
for name, nprobe, scale in CONFIGS:
    for threads in (1, 4):
        path = out / f'v12_{name}_t{threads}.yaml'
        path.write_text(render(name, nprobe, scale, threads))
        print(f'wrote {path.name} (nprobe={nprobe} scale={scale or "off"})')
