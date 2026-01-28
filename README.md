<h1 align=" center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="zvec logo" />
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/alibaba/zvec/actions" style="text-decoration:none;"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml/badge.svg?branch=main" alt="Linux x64 CI"/></a>
  <a href="https://codecov.io/github/alibaba/zvec" ><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66"  alt="Code Coverage"/></a>
  <a href="https://pypi.org/project/zvec/" style="text-decoration:none;"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="PyPI Release"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE" style="text-decoration: none;"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"/></a>
</p>

# Zvec

**Zvec** is an open-source, in-process vector database — lightweight, lightning-fast, and designed to embed directly into applications. Built on **Proxima** (Alibaba's battle-tested vector search engine), it delivers production-grade, low-latency, scalable similarity search with minimal setup.

[📚 Quick Start](https://zvec.org/en/docs/quickstart/) | 🌐 [Website](https://zvec.org/en/) | [📖 Documentation](https://zvec.org/en/docs/) | [📊 Benchmarks](https://zvec.org/en/docs/benchmarks/) | [💜 Discord](https://discord.gg/rKddFBBu9z)

## 💫 Features

- **Blazing Fast**: Searches billions of vectors in milliseconds.
- **Simple, Just Works**: Install with `pip install zvec` and start searching in seconds. No servers, no config, no fuss.
- **Dense + Sparse Vectors**: Work with both dense and sparse embeddings, with native support for multi-vector queries in a single call.
- **Hybrid Search**: Combine semantic similarity with structured filters for precise results.
- **Runs Anywhere**: As an in-process library, Zvec runs wherever your code runs — notebooks, servers, CLI tools, or even edge devices.

## 📦 Installation

Install Zvec from PyPI with a single command:

```bash
pip install zvec
```

**Requirements**:

- Python 3.10 - 3.12
- **Supported platforms**:
  - Linux (x86_64)
  - macOS (ARM64)

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
collection = zvec.create_and_open(path="./zvec_example", schema=schema,)

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

## ❤️ Contributing

We welcome and appreciate contributions from the community! Whether you're fixing a bug, adding a feature, or improving documentation, your help makes Zvec better for everyone.

Check out our [Contributing Guide](./CONTRIBUTING.md) to get started!
