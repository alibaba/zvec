# DiskANN Buffer Storage 本机性能与 RSS 测试报告

测试日期：2026-08-06 至 2026-08-07
状态：已完成问题修复与复测

## 1. 结论

本报告中 `5.06x` QPS 是“300 条查询重复回放、热图页能完全驻留”的缓存
上界，证明 Buffer 路径有效，但不能直接外推到均匀或长尾业务。对外发布通用
DiskANN 性能结论时，应以第 6 节的四工作负载矩阵为准。

在第一期边界内，当前实现可以交付：

- 仅接入 DiskANN。
- `memory_limit_mb` 定义为进程级共享缓存的软预算，不承诺限制整个进程 RSS。
- `enable_mmap=false` 时，DiskANN 通过 Buffer Storage 使用全局 Buffer Pool。
- `enable_mmap=true` 时，DiskANN 的 Buffer Pool 默认不生效，其向量读取不受该缓存预算约束；其他显式接入全局共享池的内存分区仍可继续受控。

| 验收项 | 结果 | 结论 |
|---|---:|---|
| 连续 1,800 次查询后的 RSS 增量 | Buffer Pool `0 KiB`，直读 `8 KiB` | 通过 |
| 128 MiB、`list_size=100` 热点稳态吞吐 | Buffer Pool `1541.52 QPS`，直读 `304.55 QPS` | 热点上界快约 `5.06x` |
| 128 MiB、`list_size=100` 稳态物理读 | Buffer Pool `0 MiB`，直读 `609.36 MiB` | 命中缓存后消除重复磁盘读 |
| 101 MiB 与 256 MiB 压力对照 | 小预算产生淘汰与物理读，大预算不产生 | 软预算行为有效 |
| 查询结果一致性 | 相同配置下结果指纹完全一致 | 通过 |
| 回归测试与格式检查 | 全部通过 | 通过 |

需要明确的是：128 MiB 是共享缓存的逻辑预算，不是 RSS 上限。本次 128 MiB Buffer Pool 场景的峰值 RSS 为 `284.54 MiB`，其中还包括索引元数据、查询工作区、分配器驻留、AIO 缓冲和运行时开销。

## 2. 最终复测结果

### 2.1 标准场景：128 MiB，`list_size=100`

数据集为 160,000 条 128 维 FP32 向量；两种模式使用完全相同的索引，索引大小为 `176,157,600` 字节，SHA256 为 `1d1638ab8c65f8cac44e9438b418fd0ee569d14092cb88d3523e2844d2dda50a`。

| 模式 | 冷启动 QPS | 冷启动 P95 | 稳态 QPS | 稳态 P95 | 峰值 RSS | 稳态物理读 | Self-hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 直读（`enable_mmap=true`） | 307.71 | 3.73 ms | 304.55 | 3.77 ms | 141.34 MiB | 609.36 MiB | 91.33% |
| Buffer Pool（`enable_mmap=false`） | 334.24 | 3.46 ms | 1541.52 | 0.81 ms | 284.54 MiB | 0 MiB | 91.33% |

两种模式的结果指纹均为 `f5c9fc4583025ab0d5277ccfa9c6d36690703a7c61dc18f688143313be871e56`。

Buffer Pool 统计：

| 指标 | 数值 |
|---|---:|
| 命中 | 187,200 |
| 未命中 | 17,496 |
| 命中率 | 91.45% |
| 淘汰 | 0 |
| 二级命中 | 0 |
| 绕过读取 | 0 |

该查询工作集约为 `17,496 × 4 KiB = 68.3 MiB`，因此 128 MiB 已能完整容纳热页。最终结果已正确保留查询时的 `list_size`，旧的参数丢失结果不再作为性能或召回基线。

### 2.2 预算压力场景：`list_size=300`

使用更大的查询工作集对比 101 MiB 与 256 MiB 预算：

| 预算 | 稳态 QPS | 稳态 P95 | 峰值 RSS | 稳态物理读 | 命中率 | 未命中 | 淘汰 | Self-hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 MiB | 216.42 | 6.72 ms | 307.80 MiB | 430.79 MiB | 79.08% | 143,934 | 124,279 | 98.00% |
| 256 MiB | 715.16 | 1.63 ms | 370.11 MiB | 0 MiB | 95.03% | 28,445 | 0 | 98.00% |

两组结果指纹均为 `58d72483bc9f4d2da288902a828db409083f38619636e5f4aadfc8e9811ec11a`。小预算触发持续淘汰和磁盘读取，大预算容纳工作集后不再淘汰，说明预算确实影响缓存驻留和 I/O 行为。

### 2.3 RSS 稳定性

同一个索引对象连续执行 6 批、共 1,800 次查询：

| 累计查询数 | 直读 RSS | Buffer Pool RSS |
|---:|---:|---:|
| 300 | 93.164 MiB | 239.152 MiB |
| 600 | 93.289 MiB | 239.262 MiB |
| 900 | 93.297 MiB | 239.262 MiB |
| 1,800 | 93.297 MiB | 239.262 MiB |

第 2 批到第 6 批的 RSS 增量分别为：直读 `8 KiB`，Buffer Pool `0 KiB`。后端选择日志只输出一次，测试期间没有异常 warning 或 error。

## 3. 预算语义与生效范围

| 配置 | DiskANN 向量读取路径 | 是否进入 Buffer Pool 预算 |
|---|---|---|
| `enable_mmap=false` | `BufferReadStorage` + `VecBufferPool`，阻塞式批量 AIO | 是 |
| `enable_mmap=true` | DiskANN 直读路径；Linux 下实际为 `O_DIRECT` + libaio | 否 |

`memory_limit_mb` 应对用户描述为“进程内共享缓存的软预算”：

- 它约束已显式接入全局 MemoryLimitPool 的缓存分区。
- 它不等于进程 RSS 硬限制，也不覆盖索引元数据、临时查询内存、线程栈、分配器碎片或其他未接入组件。
- `enable_mmap=true` 只表示 DiskANN Buffer Pool 不生效；不能据此推断其他共享缓存分区也失去预算控制。
- 如果未来需要严格限制进程总内存，仍需结合 cgroup、容器限制或进程级资源控制。

可将进程 RSS 粗略理解为：

```text
RSS ≈ Buffer Pool 驻留页
    + 索引元数据和查询工作区
    + AIO/对齐缓冲
    + 分配器驻留与碎片
    + 运行时及其他组件
```

## 4. 已完成修复与验证

| 问题 | 修复结果 | 验证 |
|---|---|---|
| 查询上下文未完整初始化，`list_size` 被回退到默认值 | 初始化上下文 magic，并保留调用方参数 | Self-hit 与结果指纹一致 |
| VisitFilter 销毁链路造成逐查询内存泄漏 | 改为 RAII、幂等销毁，并修复 Bloom 初始化失败路径 | 1,800 次查询 RSS 平台化 |
| AIO 读取在容量仍充足时提前淘汰 | 改为先申请空闲容量，仅在容量不足时淘汰 | 128/256 MiB 容量内均为 0 淘汰 |
| 后端日志由调用方自行去重 | 去重收敛到 `IOBackend` 内部 | 全进程只输出一次 |

回归验证：

- macOS：Buffer Pool 39 项、VisitFilter 3 项，全部通过。
- Linux Release：VisitFilter 3 项、Buffer Pool 40 项、DiskANN AIO 5 项、DiskANN builder/streamer 3 项，全部通过。
- `clang-format 18.1.8 -Werror`：通过。

## 5. 测试设计与环境

测试从公开 Python API 创建、打开和查询 DiskANN 索引，没有绕过上层接口直接调用内部 C++ 类。

主要参数：

- 160,000 条 128 维 FP32 向量，L2 距离。
- 300 条查询，`topk=10`。
- 先执行 300 次冷启动查询，再执行 5 轮、每轮 300 次稳态查询。
- 每个正式场景开始前清理文件页缓存。
- 同时记录 QPS、延迟、`/proc/self/status` RSS、`/proc/self/io` 物理读、Buffer Pool 统计和结果指纹。

环境：Apple M3 Pro；Colima VZ/Rosetta `linux/amd64`；8 vCPU、16 GiB；Ubuntu 24.04；GCC 13.3；Python 3.12；libaio1t64；Release 构建。

Rosetta 环境只适合验证相对行为和正确性，绝对 QPS 不应作为生产容量基线。

## 6. 对外发布工作负载矩阵

DiskANN 的图访问不能用 IVF centroid 占比代替。新的 benchmark 使用完全相同
的请求序列分别测试直读和 Buffer，并覆盖：

| 工作负载 | 用途 |
|---|---|
| `uniform_unique` | 无复用下界 |
| `semantic80` | 20% 相似查询承接 80% 请求，推荐主结果 |
| `zipf1` | 长尾业务 |
| `exact90` | 5 条精确热点承接 90% 请求，只作上界 |

矩阵必须在原生 Linux x86_64 Release 环境串行执行，每个点使用独立进程；固定
`list_size`、并发、Recall 和请求种子，同时记录 QPS、P99、进程峰值 RSS、物理读、
热点集中度和结果指纹：

```sh
python benchmarks/diskann_buffer_pool/run_workload_matrix.py \
  --mmap-collection /data/diskann-direct \
  --buffer-collection /data/diskann-buffer \
  --query-file /data/cohere-queries.npz \
  --buffer-memory-mb 128 256 512 \
  --engine-threads 8 --client-threads 16 \
  --list-size 100 \
  --output-dir /results/diskann-buffer-matrix
```

默认每个点重复三次。`matrix_summary.csv` 输出中位 QPS/P99/RSS、QPS 方差、
物理读取，以及相对直读路径的 QPS、RSS 和 QPS/GiB 比值。

发布时以 `semantic80`/`zipf1` 为主，`uniform_unique` 说明退化边界，
`exact90` 不进入产品 headline。

现有 91.33% 指标是“索引中的查询向量自身是否出现在 top-10”，只是
Self-hit@10，不是对 brute-force ground truth 计算的标准 Recall@10。
它与跨存储模式的结果指纹一起可以证明本次存储对比没有改变结果；
对外发布仍应补充数据集 ground truth Recall@10。

## 7. 剩余工作

以下事项不阻塞第一期软预算交付，但建议在对外开放配置前完成：

1. 对外文档明确 `memory_limit_mb` 的软预算语义，以及 `enable_mmap=true` 时 DiskANN 向量读取绕过 Buffer Pool 的行为。
2. 暴露可观测指标：逻辑容量、已用/已提交页、命中、未命中、淘汰、绕过读取、后端类型以及 `O_DIRECT` 回退状态。
3. 在原生 Linux x86_64 Release 环境运行上述多线程预算矩阵，形成可用于生产容量规划的基线。
4. 写入、脏页、刷盘和崩溃恢复不在本次只读测试范围内；后续接入写入链路时需要单独设计和验证。

## 8. 测试资产

- [benchmark.py](./benchmark.py)：主性能测试。
- [run_workload_matrix.py](./run_workload_matrix.py)：四工作负载串行矩阵与 CSV 汇总。
- [soak.py](./soak.py)：RSS 稳定性测试。
- [Dockerfile](./Dockerfile)：Linux 测试环境。
- `results/fixed_query_*.log`：最终查询复测日志。
- `results/fixed_soak_*.log`：最终 RSS 稳定性日志。
