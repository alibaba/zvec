## 单元测试

### 覆盖环境

android, iOS, linux-arm64, linux-x64-clang, linux-x64, macos-26-arm64

### 验证方面

1. /root/code/zvec/tests/core/quantizer/integer_quantizer_reformer_test.cc中
    1. 旋转与逆旋转，覆盖维度：768，200，128，96
    2. 序列化 dump/open roundtrip
2. /root/code/zvec/tests/core/algorithm/hnsw/hnsw_streamer_test.cc
    1. 模仿hnsw+int8实现hnsw+int8+旋转的测试
3. /root/code/zvec/tests/core/algorithm/flat/flat_streamer_test.cc
    1. 模仿flat+int8实现flat+int8+旋转的测试
4. /root/code/zvec/tests/core/algorithm/vamana/vamana_streamer_test.cc
    1. 模仿vamana+int8实现vamana+int8+旋转的测试
5. 在python、C API 中添加相关测试
6. 删除原本的测试

### 初步验证

在推送到CI前，在当前环境linux-x64下进行测试

