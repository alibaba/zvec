<p align="right">
  <a href="./README.md">English</a> | 中文
</p>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="zvec logo" />
  </picture>
</div>

<p align="center">
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="代码覆盖率"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml/badge.svg?branch=main" alt="Main"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="许可证"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="PyPI 版本"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/badge/python-3.10%20~%203.14-blue.svg" alt="Python 版本"/></a>
  <a href="https://www.npmjs.com/package/@zvec/zvec"><img src="https://img.shields.io/npm/v/@zvec/zvec.svg" alt="npm 版本"/></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/20830" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20830" alt="alibaba%2Fzvec | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/zh/docs/db/quickstart/">🚀 <strong>快速开始</strong> </a> |
  <a href="https://zvec.org/zh/">🏠 <strong>主页</strong> </a> |
  <a href="https://zvec.org/zh/docs/db/">📚 <strong>文档</strong> </a> |
  <a href="https://zvec.org/zh/docs/db/benchmarks/">📊 <strong>性能报告</strong> </a> |
  <a href="https://deepwiki.com/alibaba/zvec">🔎 <strong>DeepWiki</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/ZvecAI">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** 是一款开源的嵌入式(进程内)向量数据库 — 轻量、极速，可直接嵌入应用程序。以极简的配置提供生产级、低延迟、可扩展的向量检索能力。

> [!IMPORTANT]
> 🚀  **v0.5.0（2026 年 6 月 12 日）**
>
> - **全文检索（FTS）**：原生全文检索能力——可为任意字符串字段挂载 FTS 索引，使用自然语言或结构化表达式检索，无需外接搜索引擎。
> - **混合检索**：在单次查询调用中融合全文、稠密向量和稀疏向量检索，并支持标量过滤与重排。
> - **DiskANN 索引**：全新磁盘索引，将索引主体存于磁盘，大幅降低大规模数据集的内存占用。
> - **生态与平台**：全新官方 [Go](https://github.com/zvec-ai/zvec-go) / [Rust](https://github.com/zvec-ai/zvec-rust) SDK、可视化工具 [Zvec Studio](https://github.com/zvec-ai/zvec-studio)，以及 RISC-V 架构支持。
>
> 👉 [查看更新日志](https://github.com/alibaba/zvec/releases/tag/v0.5.0) | [查看路线图 📍](https://github.com/alibaba/zvec/issues/309)

## 💫 核心特性

- **极致性能**：毫秒级响应，轻松检索数十亿级向量。
- **开箱即用**：[安装](#-安装)后即刻开始搜索，纯本地运行，无需服务器、无需配置、零门槛。
- **稠密 + 稀疏向量**：支持稠密向量、稀疏向量与多向量查询，以及从内存到磁盘、丰富多样的[向量索引类型](https://zvec.org/zh/docs/db/concepts/vector-index/#向量索引类型)。
- **全文检索（FTS）**：原生的基于关键词的全文检索——使用自然语言或结构化表达式检索字符串字段。
- **混合检索**：在单次查询中融合向量语义、全文检索与标量过滤，获得精确结果。
- **持久化存储**：WAL 预写日志保障数据持久性 — 即使进程崩溃或意外断电，数据也不会丢失。
- **并发访问**：支持多进程同时读取同一个 Collection；写入为单进程独占模式。
- **进程内运行**：无需单独部署服务，纯进程内运行。Notebook、高性能服务器、CLI 工具、边缘设备 — 随处可用。

## 📦 安装

Zvec 提供多语言官方 SDK：

- **[Python](https://pypi.org/project/zvec/)**：`pip install zvec`（需 64 位 Python 3.10–3.14）
- **[Node.js](https://www.npmjs.com/package/@zvec/zvec)**：`npm install @zvec/zvec`
- **[Go](https://github.com/zvec-ai/zvec-go)**：高性能的 Go 绑定。
- **[Rust](https://github.com/zvec-ai/zvec-rust)**：高性能的 Rust 绑定。
- **[Dart/Flutter](https://pub.dev/packages/zvec)**：`flutter pub add zvec`

想要图形界面？试试 **[Zvec Studio](https://github.com/zvec-ai/zvec-studio)**，零代码浏览数据与调试查询。

### ✅ 支持的平台

- Linux (x86_64, ARM64)
- macOS (ARM64)
- Windows (x86_64)

### 🛠️ 源码构建

如需从源码构建 Zvec，请参考[源码构建指南](https://zvec.org/zh/docs/db/build/)。

## ⚡ 一分钟上手

```python
import zvec

# 定义 collection schema
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# 创建 collection
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# 插入 documents
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

# 向量相似度检索
results = collection.query(
    zvec.Query(field_name="embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# 查询结果：按相关性排序的 {'id': str, 'score': float, ...} 列表
print(results)
```

## 📈 极致性能

Zvec 提供极致的速度和效率，能够轻松应对高要求的生产环境负载。

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Zvec 性能基准测试" />

有关具体的测试方法、配置及完整结果，请参阅[性能报告](https://zvec.org/zh/docs/db/benchmarks/)。

## 🤝 加入社区

<div align="center">

| 💬 钉钉群 | 📱 微信群 | 🎮 Discord | X (Twitter) |
| :---: | :---: | :---: | :---: |
| <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" width="150" alt="钉钉二维码"/> | <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png" width="150" alt="微信二维码"/> | [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/rKddFBBu9z) | [![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZvecAI)](<https://x.com/ZvecAI>) |
| 扫码加入 | 扫码加入 | 点击加入 | 点击关注 |

</div>

## ❤️ 参与贡献

非常欢迎来自社区的每一份贡献！无论是修复 Bug、新增功能，还是完善文档，都将让 Zvec 变得更好。

请查阅我们的[贡献指南](./CONTRIBUTING.md)开始参与！
