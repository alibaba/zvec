import json,os,pathlib,subprocess,time
O=pathlib.Path(__file__).resolve().parent
while 'ALL COMPLETE' not in (O/'runs_resumed.log').read_text():time.sleep(5)
rows=[]
for i,version in enumerate(['current','main','main','current']):
 log=O/f'hnsw_confirmation_{i+1}_{version}.log'
 cmd=[str(O/version/'bench'),'HnswStreamer','MMapFileStorage',str(O/'cosine_indexes/hnsw.index'),'/Users/ali.yzf/zvec/cross_platform_indexes_20261009/data/cohere_test_vector_1000.new.txt',str(O/'data/neighbors_100k.txt'),'1','30','512']
 env=os.environ.copy();env['BENCH_VALIDATION_QUERIES']='1000'
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=120)
 if r.returncode:raise SystemExit(r.returncode)
 row=json.loads(next(l[7:] for l in log.read_text().splitlines() if l.startswith('RESULT ')));row.update(version=version,sequence=i+1,command=cmd);rows.append(row)
 (O/'hnsw_long_confirmation.json').write_text(json.dumps(rows,indent=2));print(version,row['qps'],row['p99_us'],flush=True)
print('CONFIRMATION COMPLETE',flush=True)
