# Stile：候选行谓词解释器

Stile 是 zvec SQL 执行器的内部模块，位于 `zvec::stile` 命名空间。它将已经绑定的 Arrow boolean 表达式编译为只读求值树，在候选行上直接读取列缓冲区，避免逐行构造 Arrow Scalar 和 ExecBatch。

模块随 zvec 一起构建，复用项目中的 Arrow。Collection 查询由 `DocFilter` 自动接入；业务调用方仍使用原有查询接口，无需直接操作这些内部类型。

## 核心接口

| 接口 | 用途 |
| --- | --- |
| `Initialize()` | 初始化 Arrow compute、记录内置函数身份并预热 LIKE；在绑定表达式和启动独立 Arrow LIKE 工作线程前调用 |
| `ColumnIndex` | 为单列建立行号到 chunk 的定位信息，持有原始数组及缓冲区；不复制列值 |
| `TableIndex::Make()` | 从 Arrow Table 或等长的 `ColumnIndex` 集合构造带 schema 的读取索引 |
| `Predicate::Compile()` | 将绑定到该 schema 的 boolean 表达式编译为谓词；返回 `arrow::Result` |
| `Predicate::matches(row)` | 仅在结果为 true 时返回 true；false、NULL、越界行均返回 false |
| `Predicate::evaluate(row)` | 返回 `Truth::kTrue`、`kFalse` 或 `kNull`；调用方必须保证行号合法 |
| `match_many()` / `evaluate_many()` | 批量求值；当前生产候选链路使用的是单行 `matches()` |

`Compile()` 也会调用 `Initialize()`，但不能替代绑定原始表达式之前的初始化。

## 最小使用示例

下面的 C++17 示例演示 `Initialize → TableIndex → Bind → Compile → matches`。在项目构建中包含 `db/sqlengine/stile/predicate.h` 并链接 `zvec` target；此目录不提供单独的构建入口。

```cpp
#include <cstdint>
#include <iostream>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "db/sqlengine/stile/predicate.h"

arrow::Status example() {
  namespace cp = arrow::compute;
  namespace stile = zvec::stile;
  ARROW_RETURN_NOT_OK(stile::Initialize());

  arrow::Int32Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues({8, 42}));
  ARROW_RETURN_NOT_OK(builder.AppendNull());
  ARROW_ASSIGN_OR_RAISE(auto values, builder.Finish());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::int32())}),
      arrow::ArrayVector{values});
  ARROW_ASSIGN_OR_RAISE(auto index, stile::TableIndex::Make(table, true));

  auto expression = cp::call(
      "greater_equal", {cp::field_ref("value"), cp::literal(int32_t{20})});
  ARROW_ASSIGN_OR_RAISE(auto bound, expression.Bind(*table->schema()));
  stile::CompileOptions options;
  options.validate_full = true;
  ARROW_ASSIGN_OR_RAISE(auto predicate,
                       stile::Predicate::Compile(bound, index, options));

  for (int64_t row = 0; row < predicate->num_rows(); ++row) {
    if (predicate->matches(row)) std::cout << row << '\n';
  }
  return arrow::Status::OK();
}

int main() {
  auto status = example();
  if (status.ok()) return 0;
  std::cerr << status.ToString() << '\n';
  return 1;
}
```

输出为 `1`：值为 42 的行匹配，值为 8 和 NULL 的行不匹配。示例直接传播编译失败；需要回退的调用方应显式处理状态，模块本身不会自动执行 Arrow。

## 表达式与错误处理

支持的主要子集包括 boolean 字段/常量、比较、`and` / `or`、`and_kleene` / `or_kleene`、`invert`、空值检查、`is_in`，以及可证明无损的数值转换。支持情况还取决于绑定后的类型、kernel、options 和数据布局，不能仅凭函数名判断。

- 普通布尔逻辑与 Kleene 逻辑保持 Arrow 各自的三值语义；不能互相替换。
- LIKE 仅使用 Arrow 的大小写敏感前缀、后缀、子串快速路径，例如 `abc%`、`%abc`、`%abc%`。无通配符精确模式、`_` 通配符和忽略大小写等其余模式返回 `NotImplemented`。
- `FunctionRegistry` 可为自定义 Arrow 函数提供 list membership 描述。当前 `DocFilter` 未注册这些描述，不能据此认为 SQL `CONTAIN_*` 已走 Stile。
- `NotImplemented` 表示无法使用当前求值子集；`CapacityError` 表示编译超过硬预算。生产接入对这两类编译状态使用**整表达式 Arrow 回退**，不混合执行部分子树。
- 初始化、schema/数据校验、绑定及其他编译错误必须按调用链传播，不能一概当作正常回退。`DocFilter` 还会在编译前拒绝非 boolean 根节点。

默认编译预算定义在 `CompileOptions` 中：

| 选项 | 默认值 / 行为 |
| --- | --- |
| `max_depth` | 64；超出支持深度返回 `NotImplemented`；另有物理深度 256 的硬限制 |
| `max_nodes` | 16384 |
| `max_operands` | 每个展平 AND/OR 最多 4096 个操作数 |
| `max_set_values` | 累计 65536 个集合值，按去重前计数 |
| `max_set_chunks` | 累计 65536 个集合 chunk，包括空 chunk |
| `max_literal_bytes` | 64 MiB |
| `max_fusion_pairs` | 4096；仅限制比较融合优化，耗尽后继续分别编译 |

对不可信数组使用 `TableIndex::Make(table, true)` 或 `ColumnIndex::Make(..., true)` 做完整数据校验；表达式中的常量和集合由 `CompileOptions::validate_full` 控制。这两个校验入口相互独立。

## 所有权、行号与并发

- `Predicate` 持有 `TableIndex`，后者持有读取所需的数组和缓冲区。编译成功后可释放调用方的 Table/索引引用；缓冲区会保留到最后一个持有者释放。
- 这是只读共享，不是数据复制。调用方不得在编译或求值期间修改共享的底层缓冲区；表达式、options 和函数注册表在编译期间也必须保持稳定。
- 谓词发布后不可变，可供多个线程并发求值；重新准备或销毁谓词仍需调用方同步。
- 行号是 Table/捕获视图内的局部行号，不是 Collection 全局文档 ID。允许各列 chunk 边界不同、空 chunk 和非零 slice offset。
- 虚拟行号列使用 `ColumnIndex::MakeRowNumberChecked()`，类型为 uint64，不分配逐行值缓冲区。
- `evaluate()` 和 `evaluate_many()` 要求所有行号合法；`matches()` 和 `match_many()` 对越界行返回 false。批量 API 的输入、输出数组须容纳 `count` 个元素，仅在 `count == 0` 时可以为空。

## 生产查询如何接入

实现位于 [`planner/doc_filter.cc`](../planner/doc_filter.cc)，流程如下：

1. 查询规划器决定是否形成 single-stage 的 per-candidate forward 表达式。
2. `DocFilter::compute_filter()` 捕获 forward 读取状态，初始化模块、绑定表达式并检查根类型。
3. 对 sealed IPC 提供的 owning `ForwardReadView` 构造列索引和谓词；只有整棵表达式编译成功才使用 Stile。
4. 候选过滤调用 `matches()`；删除、倒排等其他过滤仍由 `DocFilter` 组合处理。

writing/memory、Parquet 等没有 owning view 的路径继续使用 locked fetch；不支持的 sealed IPC 表达式继续从 view 读取并交给 Arrow。bitmap 与倒排执行路径保持原有行为。优化器是否选择 single-stage 与表达式本身是否可编译是两回事，例如 LIKE、IN、EQ 等不能仅凭解释器能力推断其普通 SQL 执行路径。

owning view 只允许读取准备时捕获范围内的行；locked fetch 仍读取 live rows。已编译谓词的缓冲区所有权不等于 MVCC，也不能替代 Collection 对 close、DDL、worker draining 的生命周期同步。

准备完成时的 DEBUG 日志 `Forward filter evaluator: ... backend[stile|arrow]` 可用于确认实际后端，不在逐候选热循环中记录。

## 文件与维护入口

- `predicate.{h,cc}`：编译入口及单行、批量执行 API。
- `table_index.{h,cc}`、`accessors.h`：列布局、chunk 定位与类型化读取。
- `nodes.h`、`value_index.h`：求值节点、比较融合与集合查找。
- `compile_checks.{h,cc}`、`validation.{h,cc}`：内置函数校验、资源预算和数据校验。
- `like.{h,cc}`：LIKE 快速路径；保留 Apache Arrow 的来源说明。
- `functions.{h,cc}`：自定义 list membership 描述注册。

当前兼容性约束为 Arrow 21，禁止 fast-math、finite-only 等破坏 IEEE 浮点语义的配置。升级 Arrow 时须重新验证 kernel 身份、options、无损转换、NaN、signed zero、NULL 和 LIKE 初始化行为，再调整版本约束。

从仓库根目录、已有 CMake 构建目录运行相关测试：

```sh
cmake --build build --target stile_predicate_test forward_filter_read_test forward_filter_collection_test
ctest --test-dir build -R '^(stile_predicate_test|forward_filter_read_test|forward_filter_collection_test)$' --output-on-failure
```

新增求值能力应同时补充 Arrow 差分测试和回退测试；改变生产启用范围还需覆盖 Collection 的后端选择、并发与生命周期测试。

性能工具位于 [`benchmarks/filter`](../../../../benchmarks/filter)。配置 `BUILD_BENCHMARKS=ON` 后构建 `candidate_filter_bench`，使用同一数据目录分别运行待比较版本：

```sh
cmake --build build --target candidate_filter_bench
./build/bin/candidate_filter_bench prepare ./candidate_filter_bench_data 32768
./build/bin/candidate_filter_bench run ./candidate_filter_bench_data 32768 150 4
./build/bin/candidate_filter_bench run ./candidate_filter_bench_data 32768 150 4 bitmap
```

`prepare` 只创建新目录；`run` 只读查询并验证精确结果，包含正常查询准备/编译成本。P50/P95/P99 只统计查询本体，QPS 的壁钟时间还包含结果校验及线程结果收集。该 FLAT 基准是有界验证，不代表所有 ANN、数据规模或查询分布的性能。
