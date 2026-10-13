import csv,json,pathlib,statistics,subprocess
O=pathlib.Path(__file__).resolve().parent
subprocess.run(['python3',str(O/'summarize.py')],check=True)
s=json.loads((O/'summary.json').read_text());raw=json.loads((O/'results.json').read_text())
assert len(raw)==96 and len(s)==16 and all(r['rounds_main']==3 and r['rounds_current']==3 for r in s), 'Measurement matrix incomplete'
with (O/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=s[0].keys(),lineterminator='\n');w.writeheader();w.writerows(s)
lines=['Cohere 100K：IndexStorage MemoryBlock 迁移性能对比','',
'基线：main 13fd088b；迁移分支：ffabef5c。',
'数据：Cohere 1M 的前 100,000 条，原始 768 维 FP32；1,000 条查询，TopK=10。',
'使用标准 CosineFp32Converter / query reformer，精确 Flat Recall@10=100%。',
'机器：Apple M3 Pro，11 核，36 GiB，macOS 26.6.2。两版本使用相同 -O3 Release 参数独立静态编译。',
'共享 main 构建的同一份索引；测量检索，不比较构建吞吐。查询转换在计时前完成。',
'每组 3 轮、每轮 8 秒，顺序为 AB / BA / AB，串行执行；下表为三轮中位数。',
'512 MiB 单线程首轮完整校验/预热 1,000 条查询；其他轮次及 64 MiB 组为 100 条。',
'OS 文件缓存为热缓存；新进程重新建立 BufferPool。桌面后台负载未控制，范围列反映轮次波动。','',
f'有效测量：{len(raw)} 次。各同配置/同轮次的校验结果哈希一致：{all(r["result_hash_equal"] for r in s)}。','',
'算法 | 存储 | 线程 | 池上限 MiB | main QPS | current QPS | 变化 | main/current P99 ms | main/current RSS MiB | main QPS 范围 | current QPS 范围']
for r in s:
 lines.append(f'{r["algorithm"]} | {r["storage"]} | {r["threads"]} | {r["pool_mib"]} | {r["qps_main"]:.2f} | {r["qps_current"]:.2f} | {r["qps_delta_pct"]:+.2f}% | {r["p99_main_us"]/1000:.3f}/{r["p99_current_us"]/1000:.3f} | {r["rss_main_mib"]:.1f}/{r["rss_current_mib"]:.1f} | {r["qps_main_range"][0]:.1f}–{r["qps_main_range"][1]:.1f} | {r["qps_current_range"][0]:.1f}–{r["qps_current_range"][1]:.1f}')
lines+=['','完整 1,000 查询的 Recall@10（main/current）：']
for r in s:
 if r['recall10_main'] is not None:
  lines.append(f'{r["algorithm"]} / {r["storage"]}: {r["recall10_main"]:.4%} / {r["recall10_current"]:.4%}')
lines+=['','解释范围：',
'- Vamana 当前图参数 R=32、L=100、alpha=1.2、ef=200 的基线召回较低，只用于同图上的相对开销比较，不作为召回达标结论。',
'- 64 MiB 缓存组单独看待：索引约 300 MiB，发生大量缓存外读取；这不是完全内存中的性能。',
'- 小缓存组每轮 8 秒内只有约 9–24 次查询，尾延迟与 QPS 只能作为初步趋势，不能视为稳定负载结论。',
'- 未验证完整 1M、Linux、RaBitQ、冷盘性能或构建吞吐；三轮短测不代表统计显著性。',
'- 早期配置检查及被中断的长预热不计入 results.json / summary.csv。',
'','复现材料：',
'bench.cc / compile_bench.py：测量程序与编译方式。',
'build_binaries.py：使用现有 CMake compilation database 分别编译两个提交的 core。',
'configs/*.yaml：最终索引构建参数。',
'run_benchmarks.py：测量矩阵及执行顺序。',
'methodology.json / binary_fingerprints.json：方法、数据与程序指纹。',
'results.json / runs/*.log：逐轮原始结果。']
if (O/'hnsw_long_confirmation.json').exists():
 confirm=json.loads((O/'hnsw_long_confirmation.json').read_text());assert len(confirm)==4
 main=[r for r in confirm if r['version']=='main'];current=[r for r in confirm if r['version']=='current']
 b=statistics.mean(r['qps'] for r in main);c=statistics.mean(r['qps'] for r in current)
 lines+=['','HNSW 单线程 MMap 补充确认：','4 次各 30 秒，顺序 current/main/main/current，每次 1,000 条查询预热与校验。',f'main 平均 QPS {b:.2f}；current 平均 QPS {c:.2f}；变化 {100*(c/b-1):+.2f}%。']
 lines += [f"{r['sequence']}. {r['version']}: QPS={r['qps']:.2f}, P99={r['p99_us']/1000:.3f} ms" for r in confirm]
 assert len({r['result_hash'] for r in confirm})==1
(O/'report.txt').write_text('\n'.join(lines)+'\n')
print('Wrote report.txt and summary.csv')
