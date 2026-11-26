# 191 Quora：opt2 mmap 与 RocksDB 对照

2026-11-18。全量10000查询与固定1000查询的全部三轮中，两后端有序DocID和原始FP32 score bits完全一致。mmap全量吞吐低41.47%，耗时为RocksDB的1.708倍；hot三轮中位吞吐低40.83%。本轮只测冻结opt2，未混入Mac opt3。

## 性能

standard + lowercase、OR Top100、单线程；522931文档，6032630 tokens。实际RocksDB配置为write buffer 128 MiB、compression=0、block cache 8 MiB；custom mmap，index_cache_mb=8。查询计时包含分词和执行，结果序列化在计时外。

| 查询集 | 后端/轮次 | QPS | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| 全量10000 | RocksDB/0 | 67.111 | 9.791 | 44.661 | 80.585 |
| 全量10000 | mmap/0 | 39.284 | 20.087 | 65.471 | 105.172 |
| 固定1000 | RocksDB/0 | 65.659 | 9.813 | 50.695 | 86.791 |
| 固定1000 | RocksDB/1 | 65.596 | 9.927 | 50.599 | 86.697 |
| 固定1000 | RocksDB/2 | 65.452 | 9.876 | 50.580 | 86.952 |
| 固定1000 | mmap/0 | 38.816 | 20.005 | 72.149 | 112.263 |
| 固定1000 | mmap/1 | 38.864 | 19.985 | 72.188 | 113.005 |
| 固定1000 | mmap/2 | 38.740 | 19.909 | 72.892 | 113.124 |

全量耗时149.006秒与254.559秒；hot三轮QPS中位数65.596与38.816。全量open为23.509 ms与0.126 ms，不包含在查询轮次计时内。

| 新建索引 | RocksDB | custom mmap |
|---|---:|---:|
| 总构建秒数 | 21.915 | 58.467 |
| ingest + flush秒数 | 20.559 | 20.711 |
| seal + sync秒数 | 1.355 | 37.731 |
| publication + cleanup秒数 | 0.000387 | 0.024386 |
| 最终索引字节 | 124914604 | 64500320 |
| 目录普通文件字节合计 | 132016948 | 71602810 |

final_index_bytes分别为RocksDB索引目录内文件合计与custom单一不可变文件，均排除共同的ID映射和benchmark manifest。custom缩小48.36%，构建耗时2.668倍。全部文件大小见results/index-sizes.json；du -sb还包含目录元数据。

## 正确性与跨平台差异

- 直接运行二进制：format 22/22、field 20/20、reader 2/2、bitpacked 124/124，共168项通过。
- 同机两后端的全量与hot全部轮次有序DocID及score bits完全相同，见results/comparison-full.json、comparison-hot.json。
- 全量Recall@1/5/10/100：两后端均为0.569866046952 / 0.785164995745 / 0.844394132220 / 0.948050456532。
- Mac与191的document_ids.tsv逐字相同，SHA256均为1dee1b5c524f2a9edd33bc58360c09a0f09c862e5871cab12a7e66450494c527。Mac v6参考结果与Mac opt2也逐字相同。
- 跨平台比较RocksDB：10000查询中9970组次序不同、4312组Top100集合不同；共有DocID中166066个分数位串不同，最大差7 ULP。
- 4312组集合不同的查询，其23740个两边独有结果全部与各自第100名严格同分；共有文档发生顺序反转时，Mac分数最大差1 ULP。
- 已定位FtsResult::operator>只比较score、没有DocID次序（冻结源码fts_column_indexer.h:45）；BM25使用float与std::log。观测差异集中于浮点微差和同分排序/Top100截断。尚未隔离libm、浮点求值/FMA、标准库堆操作各自贡献，不能认定编译器是唯一原因。此现象与本次同机两后端逐位一致性结论分开报告。

## 环境与测量边界

- 191：Alibaba Cloud Linux 3，Linux x86_64；Intel Xeon Platinum 8369B 2.70GHz，64 vCPU / 32 cores / 1 socket / 1 NUMA，KVM。
- GCC12.3.0、CMake3.26.5，Release；默认AUTO_DETECT_ARCH=ON、全局x86-64，LTO=OFF，专用SIMD文件按仓库CMake配置。未额外启用fast-math。
- 链接同一库的独立探针实测runtime_dispatch=avx2；pack/unpack的AVX2 wrapper调用SSE4.1 FastPFOR。
- 初次GCC10.2.1因既有RaBitQ的_mm_loadu_si32不可用失败，改用已安装GCC12的独立构建目录成功，没有修改第三方实现。
- /home位于ext4 /dev/md0，RAID0，成员为4块Ali ECS NVMe Instance Storage，每块3.5T。未测裸设备带宽/IOPS。
- 未绑核，affinity 0–63，未设置专属CPU；结束load average为0.88 / 2.76 / 3.62。没有计时中的profiler或系统负载采样，不能保证每一瞬间无后台竞争。
- 串行顺序：单测→RocksDB build→custom build→RocksDB full→mmap full→全量差分→RocksDB hot三轮→mmap hot三轮→hot差分。未叠加编译或其他FTS试验。
- 未清空OS page cache。pass0是新进程首轮，不代表OS冷缓存；hot在全量后、同进程连续三轮。未测buffer pool，仅能外推至该机该配置。
- 源码基于HEAD 0c63ea4ea936fcf4b6c352f7955e087143206baf，opt2清单80/80文件在运行前后校验一致。独立快照无.git，版本字符串fallback v0.0.0；准确身份以清单和二进制SHA为准。
- fts_bench SHA256：31ef8dce04000e578704fef322bdefde3af952abad0530d343b86661b2128f14。第三方源从远端既有checkout复制，版本记录results/dependency-revisions.txt。

## 证据与复现

本机目录：/Users/jiliang.ljl/Desktop/zvec-fts-v1/remote191-opt2-20261118。
远端目录：/home/jiliang.ljl/Desktop/zvec-fts-v1/remote191-opt2-20261118。
未修改本机生产源码，未清理原checkout或数据，未创建commit。

- commands.sh、run_benchmark.py：构建命令和串行测试协议；results/commands.json：实际argv、工作目录、进程wall time、退出码。
- source-files-sha256.json：冻结源码清单；results/inputs-sha256.json：语料、查询、qrels和固定1000哈希；三份原数据与Mac原件完全一致。
- results/summary.json、comparison-*.json、cross-platform-*.json：性能、同机完整差分/召回、跨平台映射/ULP/同分证据。
- results/*.tsv.pass*：完整原始查询结果；*-build.txt、*-full.txt、*-hot.txt和*test.txt：原始输出。
- results/success.json：完整成功与最终源码/二进制校验；simd-dispatch.txt与environment*.txt、storage-array.txt：环境证据。
