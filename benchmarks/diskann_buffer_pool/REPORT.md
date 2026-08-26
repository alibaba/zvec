# DiskANN Buffer Storage 本机性能与 RSS 测试报告

测试日期：2026-08-06 至 2026-08-07
状态：已完成问题修复与复测

## 1. 结论

通用结论已用 Cohere 1M、标准 Recall@10 和四种工作负载复测。推荐从
`512 MiB` Pool 开始压测：相对 direct 路径，三轮中位 QPS 在 Uniform、80/20
语义热点、Zipf 和 90/10 精确热点下分别提升 `18%`、`39%`、`37%` 和 `73%`，
测量阶段物理读分别减少 `60%`、`80%`、`83%` 和 `97%`。Recall 完全不变。

旧的 `5.06x` QPS 来自 160K 数据上重复回放 300 条查询，仍只作为“热图页完全
驻留”的功能上界，不再作为对外主结论。

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
| Cohere 1M 标准 Recall@10 | 四种负载下 direct/Buffer 完全一致 | 通过 |
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
| Pool 满后 admission 失败导致查询返回 `-122` | admission 失败时回滚并旁路 direct I/O | 128 MiB 下连续 8,000 次查询完成 |
| 后端日志由调用方自行去重 | 去重收敛到 `IOBackend` 内部 | 全进程只输出一次 |

回归验证：

- macOS：Buffer Pool 39 项、VisitFilter 3 项，全部通过。
- Linux Release：VisitFilter 3 项、Buffer Pool 40 项、DiskANN AIO 6 项；本次另跑 DiskANN reader 3 项、searcher 7 项、builder 4 项，全部通过。
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

## 6. Cohere 1M 四工作负载矩阵

配置：1M×768 FP32，DiskANN FP16 + PQ384，`R=32`，`list_size=128`，4 查询线程；
同一份 2.44 GB 搜索索引分别由 direct 和 Buffer reader 打开。每点独立进程，
先预热 3,000 次，再测量 5,000 次，启动前清理 VM 页缓存。关键档位重复三次并
取中位数。

### 6.1 推荐档位：512 MiB

| 工作负载 | direct QPS | Buffer QPS | QPS 变化 | direct P99 | Buffer P99 | 物理读变化 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform unique | 752 | 887 | **+18%** | 9.21 ms | 7.73 ms | **-60%** | 0.96970 |
| 80/20 semantic | 798 | 1,106 | **+39%** | 8.28 ms | 7.13 ms | **-80%** | 0.97950 |
| Zipf α=1.0 | 824 | 1,131 | **+37%** | 8.18 ms | 7.43 ms | **-83%** | 0.97314 |
| 90/10 exact repeat | 765 | 1,327 | **+73%** | 8.68 ms | 6.47 ms | **-97%** | 0.94350 |

direct 的峰值 RSS 中位数约 `462 MiB`，512 MiB Buffer 约 `958 MiB`，即用约
`495 MiB` 额外 RSS 换取上表的吞吐、尾延迟和磁盘读收益。存储模式没有改变任一
负载的 Recall。

### 6.2 工作集基本驻留：1,024 MiB

| 工作负载 | QPS 提升 | P99 变化 | 物理读减少 | 峰值 RSS |
|---|---:|---:|---:|---:|
| Uniform unique | +98% | -45% | 100% | 1,257 MiB |
| 80/20 semantic | +86% | -29% | 99% | 1,186 MiB |
| Zipf α=1.0 | +76% | -22% | 99% | 1,184 MiB |
| 90/10 exact repeat | +96% | -38% | 99% | 1,062 MiB |

128/256 MiB 的收益不稳定：Pool 太小时，减少的 I/O 可能不足以覆盖缓存管理开销。
因此不能把 Buffer 设成很小就默认开启；应从 `512 MiB` 或预计工作集的约 60%
开始，用真实日志逐档测量。90/10 exact repeat 仍只表示热点上界，产品主结论应以
80/20 semantic 和 Zipf 为主。

测试运行在 Apple M3 Pro 上的 Colima ARM64 Linux VM（4 vCPU、8 GiB），索引位于
VM 内部磁盘，Linux 使用 `O_DIRECT`/libaio。结果适合证明相对收益和正确性；发布
生产容量数字前，仍应在目标 Linux 服务器和实际 NVMe 上复测绝对 QPS。

## 7. 剩余工作

以下事项不阻塞第一期软预算交付，但建议在对外开放配置前完成：

1. 对外文档明确 `memory_limit_mb` 的软预算语义，以及 `enable_mmap=true` 时 DiskANN 向量读取绕过 Buffer Pool 的行为。
2. 暴露可观测指标：逻辑容量、已用/已提交页、命中、未命中、淘汰、绕过读取、后端类型以及 `O_DIRECT` 回退状态。
3. 在目标 Linux 服务器与实际 NVMe 上复跑 Cohere 1M 矩阵，形成生产容量基线。
4. 写入、脏页、刷盘和崩溃恢复不在本次只读测试范围内；后续接入写入链路时需要单独设计和验证。

## 8. 测试资产

- [benchmark.py](./benchmark.py)：主性能测试。
- [run_workload_matrix.py](./run_workload_matrix.py)：四工作负载串行矩阵与 CSV 汇总。
- [`tools/core/diskann_storage_bench.cc`](../../tools/core/diskann_storage_bench.cc)：同一原始索引的 direct/Buffer 对照工具。
- [soak.py](./soak.py)：RSS 稳定性测试。
- [Dockerfile](./Dockerfile)：Linux 测试环境。

原始日志和 CSV 由测试脚本写入本地 `results/`，该目录不会提交到 Git。
