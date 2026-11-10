zvec ScalarPredicate 接入结果
============================

2026-11-10，分支 ``codex/gandiva-scalar-filter``，基于 freshly fetched
upstream/main ``b0d0760a5df786c6f5ddee47b9c8c0d7dc9a0c18``。
机器为 Apple M5 Pro，15 logical CPUs、48 GiB，macOS arm64。
原生代码使用 Apple Clang；Gandiva 使用 LLVM 20.1.8。

实现与启用
----------

DocFilter 的正排表达式优先尝试参数化 ScalarPredicate。编译结果使用固定容量
128 的 LRU；阈值作为每查询参数，数据通过持久化 Arrow snapshot 绑定。
每查询只绑定一次参数，候选按 doc ID 找到 chunk 后只执行对应行。
表达式、编译、绑定或存储不满足条件时，继续使用原 Arrow 路径。

构建开关 ``ZVEC_ENABLE_SCALAR_JIT`` 默认 OFF；开启时依赖 Arrow/Gandiva 21
和 LLVM 20。ON 构建可用 ``ZVEC_SCALAR_JIT=0`` 在新进程中关闭快路。
详细设计、支持范围和平台限制见 INTEGRATION.rst；构建见
../../thirdparty/gandiva_scalar/README.rst。

最终端到端性能
--------------

复用原 issue #868 的 10M 合成数据：62 维 FP32 查询、COSINE/INT8 HNSW，
top21、ef300，``dataTime > 536870911``，约 75% 通过，ratio=0.5 强制正排。
返回全部 scalar 字段；固定 1024 查询池，计时 4096 次，每客户端预热 1024 次。
每种配置交错运行 3 轮，以下为各轮统计量的中位数。API ms 是
Collection::query 耗时；QPS 还包含结果校验与析构。编译发生在预热中。

.. list-table:: 同机同轮对比
   :header-rows: 1

   * - 路径
     - 1 客户端 QPS
     - API ms
     - 8 客户端 QPS
     - API ms
   * - 干净 upstream 正排
     - 473.57
     - 2.1102
     - 411.00
     - 19.4550
   * - 本分支，关闭 JIT 的 Arrow 正排
     - 481.39
     - 2.0758
     - 409.05
     - 19.5478
   * - 参数化 Gandiva JIT
     - 1333.95
     - 0.7480
     - 6793.89
     - 1.1742
   * - 先前 hardcode INT32
     - 1333.31
     - 0.7484
     - 7397.98
     - 1.0778
   * - 本分支，无过滤
     - 2466.87
     - 0.4037
     - 12418.83
     - 0.6410

相对同库 Arrow 正排，JIT 吞吐为 2.77 倍和 16.61 倍。
单并发中位数与 hardcode 接近；8 并发吞吐仍低 8.17%，API 平均延迟高 8.95%。
单并发三轮存在波动：JIT 为 0.7480 / 0.7421 / 0.9796 ms，hardcode 为
0.7484 / 0.7208 / 0.7688 ms，不能把中位数接近解释为所有条件下完全等价。
无过滤的检索行为不同，不把与它的差值全部归因于 predicate 求值。

最终 30 轮共 122880 次计时查询，errors、short_results、filter_violations
均为 0。6 个 JIT 轮次各有 4096 次成功绑定，进程内均只编译 1 次，计时期间
编译次数均为 0。固定阈值与 1024 个变化阈值分别做了完整结果对照，JIT、同库
Arrow、干净 upstream 和旧 hardcode 的 ID、顺序、score、所有 scalar 值一致。

原始统计与每次执行的命令、库/二进制 SHA256 保存在
results/2026-11-10.json。完整 dump 和日志在本地
build-scalar-jit/results/final-ref-fixed 与 final-ref-varying-quality。

接入时处理的准备开销
--------------------

原 case 有 2645 个 chunk。最初每 chunk 调用 Bind，随后改为一次参数绑定加
连续 buffer/offset 描述数组；持久化数组的元数据校验移到 open/schema refresh。
还去掉了循环中的 Table::column() 调用，因为它按值返回 shared_ptr，使并发
查询在同一列的原子引用计数上争用。最终版本循环只借用已有列引用。

独立 micro-benchmark 验证全表元数据 Validate 本身约 144–160 us/查询，
所以不能逐查询重复。该工具公共 Make 仍含完整校验；生产 DocFilter 使用
Segment 已校验快照的私有入口。这里只把 micro-benchmark 用于定位准备开销，
不把它的均匀随机候选访问延迟当作 HNSW 端到端延迟。

正确性和构建
------------

* ON/OFF Release 构建通过。OFF 库没有 LLVM/OpenSSL 动态依赖。
* 16 组新 bridge/binding 测试覆盖类型、NULL/NaN、slice、不同列 chunk 边界、
  独立后续 chunk buffer 生命周期、畸形数据、不同参数绑定与 8 线程读取。
* added-column segment 与已有 float-range collection 回归通过。
* 公共 API smoke 在 4 类数据集上比较 JIT=0/1，每个开关各 512 条完整结果，
  覆盖多段、新增列、删除、group-by 和可写 fallback。

writable smoke 用 ratio=0 避开已经在旧 upstream 库复现的混合 recall-plan
schema 问题；只读和性能测试均为 0.5。根因和测试范围记录在相邻 README.md，
本改动没有修复那个独立问题。

性能表为热编译缓存结果。首次新表达式结构及 LRU 淘汰后的再次使用仍需编译；
阈值改变不会重新编译。当前仅在本机 macOS arm64 执行了运行时验证。
