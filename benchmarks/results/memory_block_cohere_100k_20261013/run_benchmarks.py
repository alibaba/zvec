import json,pathlib,subprocess,time,itertools,os
O=pathlib.Path(__file__).resolve().parent
while not (O/'VALIDATED').exists() or not (O/'HARNESS_READY').exists():time.sleep(5)
Q='/Users/ali.yzf/zvec/cross_platform_indexes_20261009/data/cohere_test_vector_1000.new.txt'
classes={'hnsw':'HnswStreamer','vamana':'VamanaStreamer','ivf':'IVFSearcher','flat':'FlatSearcher','diskann':'DiskAnnSearcher'}
(O/'runs').mkdir(exist_ok=True)
records=[]
# Alternating AB/BA runs are serialized; no builds run during measurement.
cases=[(a,s,t,512) for a in classes for s in ['MMapFileStorage','BufferStorage'] for t in [1]]
cases += [(a,s,4,512) for a in ['hnsw','vamana'] for s in ['MMapFileStorage','BufferStorage']]
cases += [(a,'BufferStorage',t,64) for a in ['hnsw','vamana'] for t in [1]]
for rep in range(3):
 for alg,stg,threads,pool in cases:
  for version in (['main','current'] if rep%2==0 else ['current','main']):
   name=f'{alg}_{stg}_{threads}t_{pool}m_r{rep+1}_{version}'
   logfile=O/'runs'/f'{name}.log'
   index=O/'cosine_indexes'/f'{alg}.{"index" if alg in ["hnsw","vamana"] else "dump"}'
   actual_stg=stg if alg in ['hnsw','vamana'] else stg.replace('Storage','ReadStorage')
   cmd=[str(O/version/'bench'),classes[alg],actual_stg,str(index),Q,str(O/'data/neighbors_100k.txt'),str(threads),'8',str(pool)]
   env=os.environ.copy();env['BENCH_VALIDATION_QUERIES']='1000' if rep==0 and threads==1 and pool==512 else '100'
   start=time.monotonic()
   if not logfile.exists():
    with logfile.open('w') as f:
     try:r=subprocess.run(cmd,cwd=O,stdout=f,stderr=subprocess.STDOUT,timeout=180,env=env)
     except subprocess.TimeoutExpired:print('TIMEOUT',name,flush=True);raise
    rc=r.returncode
   else:rc=0
   lines=[x for x in logfile.read_text().splitlines() if x.startswith('RESULT ')]
   if rc or not lines:
    print('FAILED',name,rc,logfile.read_text()[-1800:],flush=True)
    raise SystemExit(1)
   row=json.loads(lines[-1][7:]);row.update(version=version,repeat=rep+1,pool_mib=pool,name=name,command=cmd);records.append(row)
   (O/'results.json').write_text(json.dumps(records,indent=2))
   print(f'{name} qps={row["qps"]:.1f} recall={row["recall10"]:.4f} p99={row["p99_us"]:.1f}us rss={row["max_rss_mib"]:.1f}MiB wall={time.monotonic()-start:.1f}s',flush=True)
print('ALL COMPLETE',flush=True)
