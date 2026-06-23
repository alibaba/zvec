### 1. 矩阵存储方式对齐

当前的矩阵存储格式：


┌────────────────────────────────────────────┐
│  Header (12 bytes)                          │
│  ├── type:      uint8_t  (1B)  0=FhtKac, 1=Matrix │
│  ├── padding:   3 bytes                     │
│  ├── origin_dim: uint32_t (4B)  向量维度     │
│  └── reserved:  uint32_t (4B)  = origin_dim │
├────────────────────────────────────────────┤
│  Rotation Blob (变长)                       │
│  ├── FhtKac: 4 × (dim/8) 字节 flip 位数组   │
│  └── Matrix: dim × dim × 4 字节 float 矩阵  │
└────────────────────────────────────────────┘


这个格式不太好，写成以下形式：


┌─────────────────────────────────────────────────────────┐
│  RotatorSerHeader (24 bytes, 固定)                       │
│  ├── magic:         uint32_t  (4B)  = 0x52544F52 ("ROTR")│
│  ├── version:       uint16_t  (2B)  = 1                  │
│  ├── rotator_type:  uint16_t  (2B)  0=Matrix, 1=Fht      │
│  ├── in_dim:        uint32_t  (4B)  输入维度              │
│  ├── out_dim:       uint32_t  (4B)  输出维度              │
│  ├── payload_size:  uint32_t  (4B)  payload 字节数        │
│  └── reserved:      uint32_t  (4B)  = 0                  │
├─────────────────────────────────────────────────────────┤
│  Payload (变长, 由 rotator_type 决定)                     │
│  ├── FhtKac: 4 × (dim/8) 字节 flip 位数组                 │
│  └── Matrix: dim × dim × 4 字节 float 矩阵                │
└────────────────────────────────────────────——————————————┘


### 2. dispatch分发


对于/root/code/zvec/src/core/quantizer/record_rotator.cc，模仿/root/code/zvec/src/ailego/math/euclidean_distance_matrix_fp32_dispatch.cc，使用dispatch的方式来分法函数，进一步提高效率


macos-arm64: 通过
macos-26-arm64: 通过
linux-arm64: 通过
linux-x64: 通过
linux-x64-clang: 失败，报错：
```
99% tests passed, 2 tests failed out of 147

Total Test time (real) = 632.17 sec

The following tests FAILED:
	 61 - collection_test (SEGFAULT)
	103 - segment_helper_test (SEGFAULT)
Errors while running CTest
ninja: build stopped: subcommand failed.
Error: Process completed with exit code 1.
```
windows-2022: 失败，报错：
```
D:\a\zvec\zvec\src\include\zvec/core/framework/index_holder.h(688): note: while compiling class template member function 'uint32_t zvec::core::OnePassIndexHybridHolderBase<int16_t>::Iterator::sparse_count(void) const'
ninja: build stopped: subcommand failed.
WARNING: zvec.dll not found, searching recursively...
D:\a\_temp\329957e4-9010-49f9-809d-356e7f178623.ps1 : zvec.dll not found anywhere under D:\a\zvec\zvec\build
At line:1 char:1
+ . 'D:\a\_temp\329957e4-9010-49f9-809d-356e7f178623.ps1'
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [Write-Error], WriteErrorException
    + FullyQualifiedErrorId : Microsoft.PowerShell.Commands.WriteErrorException,329957e4-9010-49f9-809d-356e7f178623.p 
   s1
 
Error: Process completed with exit code 1.
```
windows-2025: 失败，报错：
```
D:\a\zvec\zvec\src\include\zvec/core/framework/index_holder.h(688): note: while compiling class template member function 'uint32_t zvec::core::OnePassIndexHybridHolderBase<int16_t>::Iterator::sparse_count(void) const'
ninja: build stopped: subcommand failed.
WARNING: zvec.dll not found, searching recursively...
D:\a\_temp\ec57dd75-aa68-42c0-a5f1-90c5e046d7cf.ps1 : zvec.dll not found anywhere under D:\a\zvec\zvec\build
At line:1 char:1
+ . 'D:\a\_temp\ec57dd75-aa68-42c0-a5f1-90c5e046d7cf.ps1'
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [Write-Error], WriteErrorException
    + FullyQualifiedErrorId : Microsoft.PowerShell.Commands.WriteErrorException,ec57dd75-aa68-42c0-a5f1-90c5e046d7cf.p 
   s1
 
Error: Process completed with exit code 1.
```