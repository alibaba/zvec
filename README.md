<div align=" center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="zvec logo" />
  </picture>
</div>

<p align="center">
  <a href="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml/badge.svg?branch=main" alt="Linux x64 CI"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/linux_arm64_docker_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_arm64_docker_ci.yml/badge.svg?branch=main" alt="Linux ARM64 CI"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/mac_arm64_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/mac_arm64_ci.yml/badge.svg?branch=main" alt="macOS ARM64 CI"/></a>
  <br>
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="Code Coverage"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="PyPI Release"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/pyversions/zvec.svg" alt="Python Versions"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"/></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/20830" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20830" alt="alibaba%2Fzvec | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/en/docs/quickstart/">🚀 <strong>Quickstart</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>Home</strong> </a> |
  <a href="https://zvec.org/en/docs/">📚 <strong>Docs</strong> </a> |
  <a href="https://zvec.org/en/docs/benchmarks/">📊 <strong>Benchmarks</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/zvec_ai">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** is an open-source, in-process vector database — lightweight, lightning-fast, and designed to embed directly into applications. Built on **Proxima** (Alibaba's battle-tested vector search engine), it delivers production-grade, low-latency, scalable similarity search with minimal setup.

## 💫 Features

- **Blazing Fast**: Searches billions of vectors in milliseconds.
- **Simple, Just Works**: [Install](#-installation) and start searching in seconds. No servers, no config, no fuss.
- **Dense + Sparse Vectors**: Work with both dense and sparse embeddings, with native support for multi-vector queries in a single call.
- **Hybrid Search**: Combine semantic similarity with structured filters for precise results.
- **Runs Anywhere**: As an in-process library, Zvec runs wherever your code runs — notebooks, servers, CLI tools, or even edge devices.

## 📦 Installation

### [Python](https://pypi.org/project/zvec/)

**Requirements**: Python 3.10 - 3.12

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

### 🛠️ Building from Source

If you prefer to build Zvec from source, please check the [Building from Source](https://zvec.org/en/docs/build/) guide.

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

For detailed benchmark methodology, configurations, and complete results, please see our [Benchmarks documentation](https://zvec.org/en/docs/benchmarks/).

## 🤝 Join Our Community

<div align="center">

Stay updated and get support — scan or click:

<table align="center" style="border-collapse: collapse; margin: 16px auto; width: 100%; max-width: 520px;">
  <tr>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">💬 DingTalk</div>
      <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" alt="DingTalk QR Code" width="100" style="border-radius: 8px; border: 1px solid #ddd;">
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">📱 WeChat</div>
      <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png" alt="WeChat QR Code" width="100" style="border-radius: 8px; border: 1px solid #ddd;">
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">🎮 Discord</div>
      <a href="https://discord.gg/rKddFBBu9z" target="_blank" style="display: inline-block; width: 100px; height: 100px; background: #5865F2; border-radius: 8px; text-decoration: none; color: white; font-size: 12px; display: flex; align-items: center; justify-content: center; line-height: 1;">
        Join Server
      </a>
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">🐦 X (Twitter)</div>
      <a href="https://x.com/zvec_ai" target="_blank" style="display: inline-block; width: 100px; height: 100px; background: #000; border-radius: 8px; text-decoration: none; color: white; font-size: 12px; display: flex; align-items: center; justify-content: center; line-height: 1;">
        Follow @zvec_ai
      </a>
    </td>
  </tr>
</table>

</div>

## ❤️ Contributing

We welcome and appreciate contributions from the community! Whether you're fixing a bug, adding a feature, or improving documentation, your help makes Zvec better for everyone.

Check out our [Contributing Guide](./CONTRIBUTING.md) to get started!
