<p align="right">
  English | <a href="./README_CN.md">中文</a>
</p>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="zvec logo" />
  </picture>
</div>

<p align="center">
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="Code Coverage"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml/badge.svg?branch=main" alt="Main"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="PyPI Release"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/badge/python-3.10%20~%203.14-blue.svg" alt="Python Versions"/></a>
  <a href="https://www.npmjs.com/package/@zvec/zvec"><img src="https://img.shields.io/npm/v/@zvec/zvec.svg" alt="npm Release"/></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/20830" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20830" alt="alibaba%2Fzvec | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/en/docs/db/quickstart/">🚀 <strong>Quickstart</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>Home</strong> </a> |
  <a href="https://zvec.org/en/docs/db/">📚 <strong>Docs</strong> </a> |
  <a href="https://zvec.org/en/docs/db/benchmarks/">📊 <strong>Benchmarks</strong> </a> |
  <a href="https://deepwiki.com/alibaba/zvec">🔎 <strong>DeepWiki</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/ZvecAI">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** is an open-source, in-process vector database — lightweight, lightning-fast, and designed to embed directly into applications. Battle-tested within Alibaba Group, it delivers production-grade, low-latency and scalable similarity search with minimal setup.

> [!Important]
> 🚀 **v0.4.0 (May 9, 2026)**
>
> - **Dart/Flutter SDK**: Published the official [zvec](https://github.com/zvec-ai/zvec-dart) Flutter package with FFI bindings. Supports Android (arm64-v8a) and iOS (arm64) — no manual native compilation required.
> - **iOS Build Support**: Added support for building on iOS platforms, expanding cross-platform coverage.
> - **Enlarged topK Limit**: Relaxed the upper bound on topK to support larger-scale recall scenarios.
> - **Bug Fixes**: SQ8 quantizer recall drop; Windows path handling; sparse vector index ordering.
>
> 👉 [Read the Release Notes](https://github.com/alibaba/zvec/releases/tag/v0.4.0) | [View Roadmap 📍](https://github.com/alibaba/zvec/issues/309)

## 💫 Features

- **Blazing Fast**: Searches billions of vectors in milliseconds.
- **Simple, Just Works**: [Install](#-installation) and start searching in seconds. Pure local, no servers, no config, no fuss.
- **Dense + Sparse Vectors**: Work with both dense and sparse embeddings, with native support for multi-vector queries in a single call.
- **Hybrid Search**: Combine semantic similarity with structured filters for precise results.
- **Durable Storage**: Write-ahead logging (WAL) guarantees persistence — data is never lost, even on process crash or power failure.
- **Concurrent Access**: Multiple processes can read the same collection simultaneously; writes are single-process exclusive.
- **Runs Anywhere**: As an in-process library, Zvec runs wherever your code runs — notebooks, servers, CLI tools, or even edge devices.

## 📦 Installation

### [Python](https://pypi.org/project/zvec/)

**Requirements**: Python 3.10 - 3.14

```bash
pip install zvec
```

### [Node.js](https://www.npmjs.com/package/@zvec/zvec)

```bash
npm install @zvec/zvec
```

### ✅ Supported Platforms

- Linux (x86_64, ARM64)
- macOS (ARM64)
- Windows (x86_64)

### 🛠️ Building from Source

If you prefer to build Zvec from source, please check the [Building from Source](https://zvec.org/en/docs/db/build/) guide.

## ⚡ One-Minute Example

```python
import zvec

# Define collection schema
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# Create collection
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# Insert documents
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

# Search by vector similarity
results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# Results: list of {'id': str, 'score': float, ...}, sorted by relevance
print(results)
```

## 📈 Performance at Scale

Zvec delivers exceptional speed and efficiency, making it ideal for demanding production workloads.

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Zvec Performance Benchmarks" />

For detailed benchmark methodology, configurations, and complete results, please see our [Benchmarks documentation](https://zvec.org/en/docs/db/benchmarks/).

## 🤝 Join Our Community

<div align="center">

<div align="center">

| 💬 DingTalk | 📱 WeChat | 🎮 Discord | X (Twitter) |
| :---: | :---: | :---: | :---: |
| <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" width="150" alt="DingTalk QR Code"/> | <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png?v=6" width="150" alt="WeChat QR Code"/> | [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/rKddFBBu9z) | [![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZvecAI)](<https://x.com/ZvecAI>) |
| Scan to join | Scan to join | Click to join | Click to follow |

</div>

</div>

## ❤️ Contributing

We welcome and appreciate contributions from the community! Whether you're fixing a bug, adding a feature, or improving documentation, your help makes Zvec better for everyone.

Check out our [Contributing Guide](./CONTRIBUTING.md) to get started!
