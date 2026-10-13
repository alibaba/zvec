import json,pathlib,statistics,collections
O=pathlib.Path(__file__).resolve().parent
rows=json.loads((O/'results.json').read_text());groups=collections.defaultdict(dict)
for r in rows:
 key=(r['algorithm'],r['storage'],r['threads'],r['pool_mib'])
 groups[key].setdefault(r['version'],[]).append(r)
summary=[]
for key,v in groups.items():
 if len(v)<2:continue
 base=v['main'];cur=v['current'];b=statistics.median(x['qps'] for x in base);c=statistics.median(x['qps'] for x in cur)
 matches=[]
 for r in base:
  for s in cur:
   if r['repeat']==s['repeat']:matches.append(r['result_hash']==s['result_hash'] and r['validation_queries']==s['validation_queries'])
 rec={'algorithm':key[0],'storage':key[1],'threads':key[2],'pool_mib':key[3],'rounds_main':len(base),'rounds_current':len(cur),'qps_main':b,'qps_current':c,'qps_delta_pct':100*(c/b-1),'qps_main_range':[min(x['qps'] for x in base),max(x['qps'] for x in base)],'qps_current_range':[min(x['qps'] for x in cur),max(x['qps'] for x in cur)],'p99_main_us':statistics.median(x['p99_us'] for x in base),'p99_current_us':statistics.median(x['p99_us'] for x in cur),'rss_main_mib':statistics.median(x['max_rss_mib'] for x in base),'rss_current_mib':statistics.median(x['max_rss_mib'] for x in cur),'result_hash_equal':all(matches),'recall10_main':next((r['recall10'] for r in base if r['validation_queries']==1000),None),'recall10_current':next((r['recall10'] for r in cur if r['validation_queries']==1000),None)}
 summary.append(rec)
 print(f'{key} n={len(base)}/{len(cur)} QPS {b:.1f}->{c:.1f} {rec["qps_delta_pct"]:+.1f}% hash={rec["result_hash_equal"]}')
(O/'summary.json').write_text(json.dumps(summary,indent=2));print('runs',len(rows))
