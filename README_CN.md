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
  <a href="https://zvec.org/en/docs/quickstart/">🚀 <strong>快速开始</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>主页</strong> </a> |
  <a href="https://zvec.org/en/docs/">📚 <strong>文档</strong> </a> |
  <a href="https://zvec.org/en/docs/benchmarks/">📊 <strong>性能报告</strong> </a> |
  <a href="https://deepwiki.com/alibaba/zvec">🔎 <strong>DeepWiki</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/ZvecAI">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** 是一款开源的嵌入式(进程内)向量数据库 — 轻量、极速，可直接嵌入应用程序。以极简的配置提供生产级、低延迟、可扩展的向量检索能力。

> [!IMPORTANT]
> 🚀  **v0.3.1 （2026 年 4 月 17 日）**
> - 放宽 Collection 路径限制；改进 Windows 上的路径处理。
> 
> 🚀  **v0.3.0 （2026 年 4 月 3 日）**
>
> - **新平台支持**：支持 **Windows (MSVC)** 和 **Android**。发布了官方 Windows **Python** 和 **Node.js** 安装包。
> - **性能优化**：集成 **RabitQ** 量化以及 **CPU 指令集自适应检测**，优化 SIMD 执行。
> - **生态集成**：提供 **C-API** 用于多种编程语言绑定，以及 **[MCP](https://github.com/zvec-ai/zvec-mcp-server) / [Skill](https://github.com/zvec-ai/zvec-agent-skills)** 集成。
>
> 👉 [查看更新日志](https://github.com/alibaba/zvec/releases/tag/v0.3.0) | [查看路线图 📍](https://github.com/alibaba/zvec/issues/309)

## 💫 核心特性

- **极致性能**：毫秒级响应，轻松检索数十亿级向量。
- **开箱即用**：[安装](#-安装)后即刻开始搜索，无需服务器、无需配置、零门槛。
- **稠密 + 稀疏向量**：支持稠密向量和稀疏向量，提供多向量联合查询的原生支持。
- **混合检索**：向量语义搜索 + 标量条件过滤，获得精确结果。
- **进程内运行**：无需单独部署服务，纯进程内运行。Notebook、高性能服务器、CLI 工具、边缘设备 — 随处可用。

## 📦 安装

### [Python](https://pypi.org/project/zvec/)

**环境要求**：Python 3.10 - 3.14

```bash
pip install zvec
```

### [Node.js](https://www.npmjs.com/package/@zvec/zvec)

```bash
npm install @zvec/zvec
```

### ✅ 支持的平台

- Linux (x86_64, ARM64)
- macOS (ARM64)
- Windows (x86_64)

### 🛠️ 源码构建

如需从源码构建 Zvec，请参考[源码构建指南](https://zvec.org/en/docs/build/)。

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
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# 查询结果：按相关性排序的 {'id': str, 'score': float, ...} 列表
print(results)
```

## 📈 极致性能

Zvec 提供极致的速度和效率，能够轻松应对高要求的生产环境负载。

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Zvec 性能基准测试" />

有关具体的测试方法、配置及完整结果，请参阅[性能报告](https://zvec.org/en/docs/benchmarks/)。

## 🤝 加入社区

<div align="center">

获取最新动态和技术支持：

<div align="center">

| 💬 钉钉群 | 📱 微信群 | 🎮 Discord | X (Twitter) |
| :---: | :---: | :---: | :---: |
| <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" width="150" alt="钉钉二维码"/> | <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png?v=5" width="150" alt="微信二维码"/> | [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/rKddFBBu9z) | [![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZvecAI)](<https://x.com/ZvecAI>) |
| 扫码加入 | 扫码加入 | 点击加入 | 点击关注 |

</div>

</div>

## ❤️ 参与贡献

非常欢迎来自社区的每一份贡献！无论是修复 Bug、新增功能，还是完善文档，都将让 Zvec 变得更好。

请查阅我们的[贡献指南](./CONTRIBUTING.md)开始参与！
