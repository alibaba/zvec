# Buffer Storage 使用指南

Buffer Storage 的目标是：索引大于可用内存时，用明确的缓存预算保留热点页，
降低 RSS 和重复磁盘读取，提高单位内存吞吐。它不保证比 mmap 的绝对 QPS 更高。

## 选择建议

| 场景 | 建议 |
|---|---|
| IVF，查询存在聚类热点，索引明显大于内存预算 | 优先使用 Buffer Storage |
| DiskANN，图页会被重复访问且物理读是瓶颈 | 压测 Buffer Storage |
| 多个 Collection 需要共享有限缓存 | 使用 Buffer Storage |
| 工作集能放入内存，首要目标是最高 QPS/P99 | 使用 mmap |
| 查询近似均匀随机，或 Flat 需要全量扫描 | 使用 mmap |

当前最稳妥的默认落点是“内存受限、有访问偏斜的 IVF”。DiskANN 也能从热点页
缓存中获益，但必须用真实工作负载验证，不能套用 IVF 的 centroid 分布。

## 快速开始

```python
import zvec

# 必须在任何 Collection 操作前调用；预算由进程内所有 Buffer Collection 共享。
zvec.init(memory_limit_mb=2048, query_threads=8)

collection = zvec.create_and_open(
    path="./items.zvec",
    schema=schema,
    option=zvec.CollectionOption(read_only=False, enable_mmap=False),
)
```

打开已有 Collection：

```python
collection = zvec.open(
    "./items.zvec",
    option=zvec.CollectionOption(read_only=True, enable_mmap=False),
)
```

`enable_mmap` 在创建 Collection 时持久化。打开已有 Collection 时传入相反值不会
转换存储格式；切换模式需要新建 Collection 并重新导入、构建索引。

C++：

```cpp
zvec::GlobalConfig::ConfigData config;
config.memory_limit_bytes = 2ULL * 1024 * 1024 * 1024;
auto status = zvec::GlobalConfig::Instance().initialize(config);

zvec::CollectionOptions options{/*read_only=*/false,
                                /*enable_mmap=*/false};
auto result = zvec::Collection::CreateAndOpen("./items.zvec", schema, options);
```

## 如何设置预算

`memory_limit_mb` 是共享 Buffer Pool 的软预算，不是进程 RSS 上限。RSS 还包含
索引元数据、查询工作区、线程栈、运行时和分配器驻留。

建议：

1. 为宿主程序和查询开销预留 25%～40% 内存；
2. IVF 从索引大小的 25%～35% 开始，逐级增加约 25%；
3. 观察 QPS、P99、峰值 RSS 和物理读取，选择满足 SLO 的最小预算；
4. 多租户需要严格隔离时，使用不同进程或容器。

重启后缓存为空。接流量前回放少量代表性热点查询，不要顺序扫描整个索引污染
受限缓存。

## 如何验证收益

保持数据、索引参数、召回、查询序列和并发相同，对比：

```text
A: enable_mmap=True
B: enable_mmap=False + 明确的 memory_limit_mb
```

至少记录 QPS、P95/P99、峰值 RSS、物理读取、Recall/结果指纹，以及：

```text
QPS/GiB = QPS / 峰值 RSS(GiB)
```

没有真实日志时，至少覆盖四类请求：Uniform unique、80/20 热点、Zipf α=1.0、
90/10 exact repeat。主结论使用 80/20 或 Zipf；exact repeat 只表示缓存上界。

## 已验证结果

### IVF：Cohere 1M

1M×768 FP32、3.10 GB IVF、4 查询线程、`nprobe=4`。80/20 centroid
场景中，全部 centroid 里最热 20% 承接约 74% probe：

| 模式 | QPS | P99 | 峰值 RSS | QPS/GiB |
|---|---:|---:|---:|---:|
| Buffer 256 MiB | 319 | 46.8 ms | 303 MiB | 1077 |
| Buffer 512 MiB | 400 | 30.4 ms | 541 MiB | 757 |
| mmap | 1768 | 7.64 ms | 2888 MiB | 627 |

256 MiB Buffer 的绝对 QPS 较低，但只使用约十分之一 RSS，吞吐密度是 mmap 的
1.72 倍；512 MiB 是延迟与内存更均衡的点。Uniform 场景无复用时没有优势。

### DiskANN：Cohere 1M

1M×768、FP16 + PQ384、`list_size=128`、4 查询线程；同一份 2.44 GB 搜索索引，
标准 Recall@10 在 direct/Buffer 间完全一致。关键档位三轮中位数：

| 工作负载 | 512 MiB QPS | 512 MiB P99 | 物理读 | 1,024 MiB QPS |
|---|---:|---:|---:|---:|
| Uniform unique | +18% | -16% | -60% | +98% |
| 80/20 semantic | +39% | -14% | -80% | +86% |
| Zipf α=1.0 | +37% | -9% | -83% | +76% |
| 90/10 exact repeat | +73% | -26% | -97% | +96% |

建议从 `512 MiB` 或预计热工作集的约 60% 开始。更小的 Pool 可能因命中收益不足
而变慢；Pool 能基本容纳工作集时收益最大。主结论看 80/20 与 Zipf，exact repeat
只作上界。测试使用 Linux `O_DIRECT`/libaio；绝对 QPS 仍应在目标机器复测。

详细方法和原始数据：

- [`benchmarks/ivf_buffer_pool/README.md`](benchmarks/ivf_buffer_pool/README.md)
- [`benchmarks/diskann_buffer_pool/REPORT.md`](benchmarks/diskann_buffer_pool/REPORT.md)
