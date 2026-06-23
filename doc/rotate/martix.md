1. 编译指令：
```
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja -j$(nproc)
cd ..
pip install . --no-build-isolation
```

## 自定义旋转矩阵加载

### 功能描述

在 `local_builder` 构建索引时，除了使用默认的 FhtKac 快速随机旋转外，还支持从本地文件加载用户自定义的稠密旋转矩阵，用于替换 Converter 内部创建的 FhtKac 旋转器。该接口仅在 C++ 层（`local_builder` 工具）暴露，不涉及 Python SDK。

### YAML 配置项

在 `construct*.yaml` 的 `ConverterParams` 节中新增 `RotatorMatrixPath` 配置项：

```yaml
ConverterParams:
    cosine.converter.enable_rotate: !!bool true      # 必须为 true，用于初始化 rotator
    # 指定自定义旋转矩阵文件路径（可选）：
    RotatorMatrixPath: "/path/to/custom_matrix.bin"
```

- `enable_rotate: true`：**必填**。Converter 在 `init()` 中据此创建 rotator 并确定 `padded_dim`。
- `RotatorMatrixPath`：可选。指定自定义矩阵的二进制文件路径。若不指定（或留空），则使用 FhtKac 随机旋转。

### 矩阵文件格式（rabitqlib MatrixRotator）

文件为纯二进制格式，无 header、无对齐填充，数据布局与 rabitqlib 中 `MatrixRotator` 的 `load(const char*)` / `save(char*)` 接口一致：

| 属性 | 说明 |
|------|------|
| 数据类型 | `float`（32 位单精度浮点） |
| 布局 | **行主序（Row-Major）**，即 `Eigen::Matrix<float, Dynamic, Dynamic, RowMajor>` |
| 矩阵形状 | `dim × padded_dim`，其中 `padded_dim = ⌈dim / 64⌉ × 64`（向上取 64 的倍数） |
| 文件大小 | 必须恰好为 `dim × padded_dim × sizeof(float)` 字节 |
| 元素定位 | 第 `i` 行第 `j` 列位于字节偏移 `(i × padded_dim + j) × 4` |
| 矩阵要求 | 应为正交矩阵（行向量相互正交且范数为 1），以保证旋转后向量间距离不变 |

**示例（768 维向量）**：
- `dim = 768`（= 12 × 64，已是 64 倍数），`padded_dim = 768`
- 文件大小 = `768 × 768 × 4 = 2,359,296` 字节
- 矩阵形状：768 行 × 768 列

**示例（100 维向量）**：
- `dim = 100`，`padded_dim = 128`（= 2 × 64，向上取整）
- 文件大小 = `100 × 128 × 4 = 51,200` 字节
- 矩阵形状：100 行 × 128 列

### 使用方式

1. 准备矩阵文件：生成一个 `dim × padded_dim` 的正交矩阵，以行主序 float 数组写入二进制文件。
2. 编辑 `construct2.yaml`：
```yaml
ConverterParams:
    cosine.converter.enable_rotate: !!bool true
    RotatorMatrixPath: "/root/data/my_rotator_768x768.bin"
```
3. 构建索引：
```bash
./build/bin/local_builder config/construct2.yaml
```
4. `local_builder` 在 Converter 初始化（FhtKac）之后，会：
   - 读取 `RotatorMatrixPath` 指定的二进制文件
   - 校验文件大小是否等于 `dim × padded_dim × sizeof(float)`
   - 调用 `converter->load_rotator_matrix()` 将自定义矩阵加载到 rotator 中，替换 FhtKac 随机矩阵

### 实现原理

```
Converter::init()           → 检测到 enable_rotate=true，创建 FhtKac rotator，确定 padded_dim
local_builder               → 读取矩阵文件，校验大小，调用 converter->load_rotator_matrix(data, dim, padded_dim)
converter->load_rotator_matrix() → 调用 rotator_->load(data, dim, padded_dim)，用 MatrixRotator 替换 FhtKac
Converter::dump_to_storage() → 将最终的旋转矩阵序列化到 IndexStorage
Reformer::load()            → 从 IndexStorage 加载旋转矩阵（搜索侧自动检测，无需配置）
```

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `src/include/zvec/core/framework/index_converter.h` | `IndexConverter` 基类新增 `load_rotator_matrix()` 虚方法（默认 no-op） |
| `src/core/quantizer/cosine_converter.cc` | `CosineConverter` 覆写 `load_rotator_matrix()`，转发给 `rotator_->load()` |
| `src/core/quantizer/integer_quantizer_converter.cc` | `IntegerStreamingConverter` 覆写 `load_rotator_matrix()`，转发给 `rotator_->load()` |
| `tools/core/local_builder.cc` | `do_build()` 中 `convert_holder()` 后读取 `RotatorMatrixPath`，读文件、校验、调用 `load_rotator_matrix()` |
| `config/construct2.yaml` | 新增 `RotatorMatrixPath` 注释示例 |
