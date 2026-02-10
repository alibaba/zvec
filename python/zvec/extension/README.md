# zvec Extension - Embedding & Reranking Functions

This directory contains zvec's embedding and reranking function extensions. It provides multiple out-of-the-box implementations and supports custom extensions.

> **Note for users in mainland China:** To download models from Hugging Face more reliably, configure the mirror endpoint before running Python:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```
>
> **Dependencies:** To run the examples in this document, install the following packages first:
>
> ```bash
> pip install openai dashscope sentence-transformers
> ```

## 📚 Table of Contents

- [Overview](#overview)
- [Embedding Functions](#embedding-functions)
  - [Dense Embedding](#dense-embedding)
  - [Sparse Embedding](#sparse-embedding)
- [Reranking Functions](#reranking-functions)
- [Custom Implementation Guide](#custom-implementation-guide)
  - [Custom Embedding Functions](#custom-embedding-functions)
  - [Custom Reranking Functions](#custom-reranking-functions)

---

## Overview

### Embedding Function Types

| Type | Implementation | Description |
|------|---------------|-------------|
| Local Dense Embedding | `DefaultLocalDenseEmbedding` | Uses Sentence Transformers with `all-MiniLM-L6-v2` model (384 dimensions, ~80MB) |
| Local Sparse Embedding | `DefaultLocalSparseEmbedding` | Uses SPLADE `naver/splade-cocondenser-ensembledistil` model (~100MB) |
| BM25 Embedding | `BM25EmbeddingFunction` | Classic BM25 algorithm for sparse embedding |
| Qwen Dense Embedding | `QwenDenseEmbedding` | Uses Qwen Dashscope API |
| Qwen Sparse Embedding | `QwenSparseEmbedding` | Uses Qwen Dashscope API |
| OpenAI Dense Embedding | `OpenAIDenseEmbedding` | Uses OpenAI API |

### Reranking Function Types

| Type | Implementation | Description |
|------|---------------|-------------|
| Local Reranking | `DefaultLocalReRanker` | Uses Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` model (~80MB) |
| Qwen Reranking | `QwenReRanker` | Uses Qwen Dashscope API |
| RRF Reranking | `RrfReRanker` | Reciprocal Rank Fusion for multi-vector retrieval results |
| Weighted Reranking | `WeightedReRanker` | Weighted fusion for multi-vector retrieval results |

---

## Embedding Functions

### Dense Embedding

#### 1. DefaultLocalDenseEmbedding - Local Dense Embedding

Uses the Sentence Transformers library with the `all-MiniLM-L6-v2` model to generate 384-dimensional dense vectors.

**Model Details:**
- Model: `all-MiniLM-L6-v2` (HuggingFace) or `iic/nlp_gte_sentence-embedding_chinese-small` (ModelScope for Chinese)
- Dimensions: 384
- Size: ~80MB

```python
from zvec.extension import DefaultLocalDenseEmbedding

# Basic usage (international users)
embedding_func = DefaultLocalDenseEmbedding()
vector = embedding_func.embed("Hello, world!")
print(f"Dimensions: {len(vector)}")  # 384

# Chinese users: recommended to use ModelScope
embedding_func = DefaultLocalDenseEmbedding(model_source="modelscope")
vector = embedding_func.embed("你好，世界！")

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
vectors = [embedding_func.embed(text) for text in texts]

# Semantic similarity computation
import numpy as np
v1 = embedding_func.embed("The cat sits on the mat")
v2 = embedding_func.embed("A cat is resting on the mat")
similarity = np.dot(v1, v2)  # Normalized vectors, dot product = cosine similarity
print(f"Similarity: {similarity:.4f}")
```

#### 2. QwenDenseEmbedding - Dashscope API Dense Embedding

Uses Qwen's Dashscope embedding API.

**Note:** Requires Dashscope API key, and **dimension must be specified explicitly**.

```python
from zvec.extension import QwenDenseEmbedding

# API key required
embedding_func = QwenDenseEmbedding(
    api_key="your-dashscope-api-key",
    model="text-embedding-v4",   # Optional, uses latest model by default
    dimension=256,               # Required: embedding dimension
)

vector = embedding_func.embed("Vector database")
print(f"Dimensions: {embedding_func.dimension}")  # 256
```

#### 3. OpenAIDenseEmbedding - OpenAI API Dense Embedding

Uses OpenAI's embedding API.

```python
from zvec.extension import OpenAIDenseEmbedding

embedding_func = OpenAIDenseEmbedding(
    api_key="your-openai-api-key",
    model="text-embedding-4",  # Optional, uses latest model by default
    dimension=256,            # Required: embedding dimension
)

vector = embedding_func.embed("Vector database")
```

### Sparse Embedding

#### 1. DefaultLocalSparseEmbedding - Local Sparse Embedding

Uses the SPLADE model to generate sparse vectors, suitable for lexical matching and hybrid retrieval.

**Model Details:**
- Model: `naver/splade-cocondenser-ensembledistil`
- Size: ~100MB
- Output: Sparse dictionary format

```python
from zvec.extension import DefaultLocalSparseEmbedding

# Query embedding (for search queries)
query_embedding = DefaultLocalSparseEmbedding(encoding_type="query")
query_vec = query_embedding.embed("machine learning algorithms")

# Document embedding (for document indexing)
doc_embedding = DefaultLocalSparseEmbedding(encoding_type="document")
doc_vec = doc_embedding.embed("Machine learning is a subfield of artificial intelligence")

# Sparse vector format: {dimension_index: weight}
print(f"Non-zero dimensions: {len(query_vec)}")
print(f"First 5 dimensions: {list(query_vec.items())[:5]}")

# Clear model cache
DefaultLocalSparseEmbedding.clear_cache()
```

#### 2. BM25EmbeddingFunction - BM25 Sparse Embedding

Based on the classic BM25 algorithm.

```python
from zvec.extension import BM25EmbeddingFunction

# Requires a document corpus to build the vocabulary
corpus = [
    "Machine learning is an important branch of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing handles text data"
]

embedding_func = BM25EmbeddingFunction(corpus=corpus)

# Generate query vector
query_vec = embedding_func.embed("deep learning neural networks")
```

#### 3. QwenSparseEmbedding - Dashscope API Sparse Embedding

**Note:** Requires Dashscope API key. Visit [Dashscope Console](https://dashscope.console.aliyun.com/) to get your API key.

```python
from zvec.extension import QwenSparseEmbedding

embedding_func = QwenSparseEmbedding(
    api_key="your-dashscope-api-key",
    dimension=256,  # dashscope api required input dimension
)
sparse_vec = embedding_func.embed("sparse vector")
```

---

## Reranking Functions

Reranking functions are used to re-order retrieval results to improve relevance.

### 1. DefaultLocalReRanker - Local Cross-Encoder Reranking

Uses a Cross-Encoder model for reranking.

**Model Details:**
- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Size: ~80MB

```python
from zvec.extension import DefaultLocalReRanker
from zvec import Doc

# Initialize reranker
reranker = DefaultLocalReRanker(
    query="What are machine learning algorithms",
    topn=5,
    rerank_field="content"  # Specify the field to rerank
)

# Prepare document list
documents = {
    "vector1": [
        Doc(
            id="1",
            fields={
                "content": "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data."
            },
        ),
        Doc(
            id="2",
            fields={
                "content": "The weather is nice today with clear skies and sunshine."
            },
        ),
        Doc(
            id="3",
            fields={
                "content": "Deep learning is a specialized branch of machine learning using neural networks with multiple layers."
            },
        ),
    ],
}

# Perform reranking
reranked_docs = reranker.rerank(documents)

for doc in reranked_docs:
    print(doc)
```

### 2. QwenReRanker - Dashscope API Reranking

**Note:** Requires Dashscope API key. Visit [Dashscope Console](https://dashscope.console.aliyun.com/) to get your API key.

```python
from zvec.extension import QwenReRanker
from zvec import Doc

reranker = QwenReRanker(
    query="What is a vector database",
    model="gte-rerank-v2",
    api_key="your-dashscope-api-key",
    topn=3,
    rerank_field="content",
)

documents = {
    "vector1": [
        Doc(
            id="1",
            fields={
                "content": "Vector databases store and retrieve vectors"
            },
        ),
        Doc(
            id="2",
            fields={
                "content": "Relational databases store structured data"
            },
        ),
        Doc(
            id="3",
            fields={
                "content": "Vector retrieval is based on similarity computation"
            },
        ),
    ],
}

# Perform reranking
reranked_docs = reranker.rerank(documents)

for doc in reranked_docs:
    print(doc)
```

### 3. RrfReRanker - Reciprocal Rank Fusion

Fuses multiple retrieval results using Reciprocal Rank Fusion (RRF). **Specifically designed for multi-vector retrieval scenarios** where you have results from multiple embedding methods (e.g., dense + sparse).

**Note:** This reranker works with ranking positions only, no scores required.

```python
from zvec.extension import RrfReRanker

# Prepare multiple retrieval results (e.g., from dense and sparse embeddings)
dense_results = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
sparse_results = [{"id": "B"}, {"id": "D"}, {"id": "A"}]

reranker = RrfReRanker(topn=3, k=60)  # k is the smoothing parameter

# Fuse results
fused_results = reranker.rerank([dense_results, sparse_results])
```

### 4. WeightedReRanker - Weighted Fusion

Fuses multiple scored retrieval results according to weights. **Specifically designed for multi-vector retrieval scenarios** where each result set has scores.

```python
from zvec.extension import WeightedReRanker
from zvec import Doc

# Prepare multiple retrieval results
documents = {
    "vector1": [
        Doc(
            id="1",
            score=0.8,
        ),
        Doc(
            id="2",
            score=0.7,
        ),
        Doc(
            id="3",
            score=0.75,
        ),
    ],
}

reranker = WeightedReRanker(
    weights=[1.0],  # Weights for each result set
    topn=3
)

# Fuse results
fused_results = reranker.rerank(documents)
print(fused_results)
```

---

## Custom Implementation Guide

### Custom Embedding Functions

zvec provides two base classes and two framework-specific base classes for custom embeddings:

**Protocol Base Classes:**
- `DenseEmbeddingFunction[T]`: Protocol for dense embeddings
- `SparseEmbeddingFunction[T]`: Protocol for sparse embeddings

**Framework-Specific Base Classes:**
- `SentenceTransformerFunctionBase`: Base class for Sentence Transformers models (in `sentence_transformer_function.py`)
- `QwenFunctionBase`: Base class for Qwen Dashscope API (in `qwen_function.py`)

#### Example 1: Custom Dense Embedding from Scratch

```python
from zvec.extension import DenseEmbeddingFunction
from zvec.common.constants import TEXT, DenseVectorType
from typing import Optional
import numpy as np


class MyCustomDenseEmbedding(DenseEmbeddingFunction[TEXT]):
    """Custom dense embedding function example"""
    
    def __init__(self, model_name: str = "custom-model", **kwargs):
        self._model_name = model_name
        self._dimension = 768  # Custom dimension
        self._extra_params = kwargs
        # Initialize your model
        self._model = self._load_model()
    
    @property
    def dimension(self) -> int:
        """Return embedding vector dimension"""
        return self._dimension
    
    @property
    def extra_params(self) -> dict:
        """Return extra parameters"""
        return self._extra_params
    
    def _load_model(self):
        """Load your custom model"""
        # Implement your model loading logic here
        # e.g., return YourModelClass.from_pretrained(self._model_name)
        pass
    
    def embed(self, input: str) -> DenseVectorType:
        """
        Generate dense embedding vector
        
        Args:
            input: Input text
            
        Returns:
            DenseVectorType: List of floats, length = self.dimension
        """
        # Input validation
        if not isinstance(input, str):
            raise TypeError(f"Expected str, got {type(input).__name__}")
        
        input = input.strip()
        if not input:
            raise ValueError("Input cannot be empty")
        
        # Generate embedding using your model
        # embedding = self._model.encode(input)
        # return embedding.tolist()
        
        # Example: return random vector
        return np.random.randn(self._dimension).tolist()
    
    def __call__(self, input: str) -> DenseVectorType:
        """Make the function callable"""
        return self.embed(input)


# Use custom embedding
custom_emb = MyCustomDenseEmbedding(model_name="my-model")
vector = custom_emb.embed("Test text")
print(f"Dimensions: {len(vector)}")
```

#### Example 2: Custom Sparse Embedding from Scratch

```python
from zvec.extension import SparseEmbeddingFunction
from zvec.common.constants import TEXT, SparseVectorType
from typing import Dict


class MyCustomSparseEmbedding(SparseEmbeddingFunction[TEXT]):
    """Custom sparse embedding function example"""
    
    def __init__(self, vocab_size: int = 30000, **kwargs):
        self._vocab_size = vocab_size
        self._extra_params = kwargs
        self._tokenizer = self._load_tokenizer()
    
    @property
    def extra_params(self) -> dict:
        return self._extra_params
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        # Implement your tokenizer loading logic
        pass
    
    def embed(self, input: str) -> SparseVectorType:
        """
        Generate sparse embedding vector
        
        Args:
            input: Input text
            
        Returns:
            SparseVectorType: Dictionary {dimension_index: weight}, contains only non-zero values
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected str, got {type(input).__name__}")
        
        input = input.strip()
        if not input:
            raise ValueError("Input cannot be empty")
        
        # Implement your sparse embedding logic
        # tokens = self._tokenizer.tokenize(input)
        # sparse_vec = self._compute_sparse_representation(tokens)
        
        # Example: return simple term frequency vector
        sparse_vec = {
            100: 0.5,
            250: 1.2,
            500: 0.8
        }
        
        # Ensure sorted by index
        return dict(sorted(sparse_vec.items()))
    
    def __call__(self, input: str) -> SparseVectorType:
        return self.embed(input)


# Use custom sparse embedding
sparse_emb = MyCustomSparseEmbedding(vocab_size=50000)
sparse_vec = sparse_emb.embed("Test text")
print(f"Non-zero dimensions: {len(sparse_vec)}")
```

#### Example 3: Using SentenceTransformerFunctionBase

If you want to use a different Sentence Transformers model, you can inherit from `SentenceTransformerFunctionBase`:

```python
from zvec.extension.sentence_transformer_function import SentenceTransformerFunctionBase
from zvec.extension import DenseEmbeddingFunction
from zvec.common.constants import TEXT, DenseVectorType
from typing import Literal, Optional


class CustomSentenceTransformerEmbedding(
    SentenceTransformerFunctionBase, 
    DenseEmbeddingFunction[TEXT]
):
    """Using custom Sentence Transformer model"""
    
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",  # Use a different model
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        normalize_embeddings: bool = True,
        **kwargs
    ):
        # Initialize base class
        SentenceTransformerFunctionBase.__init__(
            self, 
            model_name=model_name,
            model_source=model_source,
        )
        
        self._normalize_embeddings = normalize_embeddings
        self._extra_params = kwargs
        
        # Load model and get dimension
        model = self._get_model()
        self._dimension = model.get_sentence_embedding_dimension()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def extra_params(self) -> dict:
        return self._extra_params
    
    def embed(self, input: str) -> DenseVectorType:
        if not isinstance(input, str):
            raise TypeError(f"Expected str, got {type(input).__name__}")
        
        input = input.strip()
        if not input:
            raise ValueError("Input cannot be empty")
        
        model = self._get_model()
        embedding = model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings
        )
        
        return embedding.tolist()
    
    def __call__(self, input: str) -> DenseVectorType:
        return self.embed(input)


# Use custom model
# Use larger MPNet model (768 dimensions)
custom_emb = CustomSentenceTransformerEmbedding(
    model_name="all-mpnet-base-v2"
)
vector = custom_emb.embed("High-quality text embedding")
print(f"Dimensions: {len(vector)}")  # 768

# Use multilingual model
multilingual_emb = CustomSentenceTransformerEmbedding(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
```

#### Example 4: Using QwenFunctionBase

If you want to implement custom embeddings using Qwen Dashscope API:

```python
from zvec.extension.qwen_function import QwenFunctionBase
from zvec.extension import DenseEmbeddingFunction
from zvec.common.constants import TEXT, DenseVectorType
from typing import Optional


class CustomQwenEmbedding(QwenFunctionBase, DenseEmbeddingFunction[TEXT]):
    """Custom Qwen embedding implementation"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        **kwargs
    ):
        # Initialize base class with API key
        QwenFunctionBase.__init__(self, api_key=api_key)
        
        self._model = model
        self._extra_params = kwargs
        self._dimension = None  # Will be set after first call
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Get dimension from first embedding call
            test_result = self.embed("test")
            self._dimension = len(test_result)
        return self._dimension
    
    @property
    def extra_params(self) -> dict:
        return self._extra_params
    
    def embed(self, input: str) -> DenseVectorType:
        if not isinstance(input, str):
            raise TypeError(f"Expected str, got {type(input).__name__}")
        
        input = input.strip()
        if not input:
            raise ValueError("Input cannot be empty")
        
        # Use the base class's embed_text method
        result = self._embed_text(
            text=input,
            model=self._model
        )
        
        return result
    
    def __call__(self, input: str) -> DenseVectorType:
        return self.embed(input)


# Use custom Qwen embedding
custom_qwen_emb = CustomQwenEmbedding(
    api_key="your-dashscope-api-key",
    model="text-embedding-v3"
)
vector = custom_qwen_emb.embed("Custom Qwen embedding")
```

### Custom Reranking Functions

Reranking functions need to inherit from the `RerankFunction` base class (exported as `ReRanker`).

#### Example 1: Custom Reranking Function from Scratch

```python
from zvec.extension import ReRanker
from typing import List, Dict, Any, Optional


class MyCustomReRanker(ReRanker):
    """Custom reranking function example"""
    
    def __init__(
        self,
        topn: int = 10,
        model_name: str = "custom-reranker",
        **kwargs
    ):
        self._topn = topn
        self._model_name = model_name
        self._extra_params = kwargs
        self._model = self._load_model()
    
    @property
    def topn(self) -> int:
        """Return top-N"""
        return self._topn
    
    @topn.setter
    def topn(self, value: int):
        """Set top-N"""
        if value <= 0:
            raise ValueError("topn must be positive")
        self._topn = value
    
    @property
    def extra_params(self) -> dict:
        return self._extra_params
    
    def _load_model(self):
        """Load reranking model"""
        # Implement your model loading logic
        pass
    
    def rerank(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        rerank_field: str = "content",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents
        
        Args:
            documents: Document list
            query: Query text (Note: base class doesn't accept query parameter, 
                   implement in subclass if needed)
            rerank_field: Field name to use for reranking
            **kwargs: Extra parameters
            
        Returns:
            Reranked document list, preserves original fields and adds rerank score
        """
        if not documents:
            return []
        
        # Extract content to rerank
        contents = [doc.get(rerank_field, "") for doc in documents]
        
        # Compute reranking scores using your model
        # scores = self._model.predict(query, contents)
        
        # Example: random scores
        import random
        scores = [random.random() for _ in contents]
        
        # Add scores to documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-N
        return scored_docs[:self._topn]
    
    def __call__(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Make the function callable"""
        return self.rerank(documents, **kwargs)


# Use custom reranker
reranker = MyCustomReRanker(topn=5, model_name="my-reranker")

documents = [
    {"id": 1, "content": "Document content 1"},
    {"id": 2, "content": "Document content 2"},
    {"id": 3, "content": "Document content 3"},
]

reranked = reranker.rerank(
    documents,
    query="Query text",
    rerank_field="content"
)

for doc in reranked:
    print(f"ID: {doc['id']}, Score: {doc['rerank_score']:.4f}")
```

#### Example 2: Query-Based Reranker

```python
from zvec.extension import ReRanker
from typing import List, Dict, Any


class QueryBasedReRanker(ReRanker):
    """Reranker that requires query at initialization"""
    
    def __init__(self, query: str, topn: int = 10):
        if not query:
            raise ValueError("Query is required")
        
        self._query = query
        self._topn = topn
    
    @property
    def query(self) -> str:
        return self._query
    
    @property
    def topn(self) -> int:
        return self._topn
    
    @topn.setter
    def topn(self, value: int):
        if value <= 0:
            raise ValueError("topn must be positive")
        self._topn = value
    
    @property
    def extra_params(self) -> dict:
        return {}
    
    def rerank(
        self,
        documents: List[Dict[str, Any]],
        rerank_field: str = "content",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query
        
        Note: query is provided at initialization, not as a parameter
        """
        if not documents:
            return []
        
        # Compute relevance using self._query and document content
        scored_docs = []
        for doc in documents:
            content = doc.get(rerank_field, "")
            # Compute relevance score
            score = self._compute_relevance(self._query, content)
            
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)
        
        # Sort and return top-N
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:self._topn]
    
    def _compute_relevance(self, query: str, content: str) -> float:
        """Compute relevance score (example implementation)"""
        # Simple word overlap score
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words)
        return overlap / (len(query_words) + 1e-6)
    
    def __call__(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        return self.rerank(documents, **kwargs)


# Use
reranker = QueryBasedReRanker(
    query="machine learning algorithms",
    topn=3
)

documents = [
    {"id": 1, "content": "Machine learning is an important AI algorithm"},
    {"id": 2, "content": "Deep learning uses neural networks"},
    {"id": 3, "content": "Supervised learning is a common ML method"},
]

reranked = reranker.rerank(documents, rerank_field="content")
```

#### Example 3: Using QwenFunctionBase for Custom Reranking

```python
from zvec.extension.qwen_function import QwenFunctionBase
from zvec.extension import ReRanker
from typing import List, Dict, Any


class CustomQwenReRanker(QwenFunctionBase, ReRanker):
    """Custom Qwen reranking implementation"""
    
    def __init__(
        self,
        query: str,
        api_key: str,
        topn: int = 10,
        model: str = "gte-rerank",
        **kwargs
    ):
        # Initialize base class
        QwenFunctionBase.__init__(self, api_key=api_key)
        
        if not query:
            raise ValueError("Query is required")
        
        self._query = query
        self._topn = topn
        self._model = model
        self._extra_params = kwargs
    
    @property
    def query(self) -> str:
        return self._query
    
    @property
    def topn(self) -> int:
        return self._topn
    
    @topn.setter
    def topn(self, value: int):
        if value <= 0:
            raise ValueError("topn must be positive")
        self._topn = value
    
    @property
    def extra_params(self) -> dict:
        return self._extra_params
    
    def rerank(
        self,
        documents: List[Dict[str, Any]],
        rerank_field: str = "content",
        **kwargs
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        # Extract contents
        contents = [doc.get(rerank_field, "") for doc in documents]
        
        # Use base class's rerank_text method
        scores = self._rerank_text(
            query=self._query,
            documents=contents,
            model=self._model
        )
        
        # Add scores to documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_docs[:self._topn]
    
    def __call__(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        return self.rerank(documents, **kwargs)


# Use custom Qwen reranker
custom_qwen_reranker = CustomQwenReRanker(
    query="What is a vector database",
    api_key="your-dashscope-api-key",
    topn=5
)
reranked = custom_qwen_reranker.rerank(documents, rerank_field="text")
```

---

## Best Practices

### 1. Hybrid Search (Multi-Vector Retrieval)

Combine dense and sparse embeddings for best retrieval performance:

```python
from zvec.extension import (
    DefaultLocalDenseEmbedding,
    DefaultLocalSparseEmbedding,
    RrfReRanker
)

# Create embedding functions
dense_emb = DefaultLocalDenseEmbedding()
sparse_emb = DefaultLocalSparseEmbedding(encoding_type="query")

# Query text
query = "What is a vector database"

# Generate both embeddings
dense_vec = dense_emb.embed(query)
sparse_vec = sparse_emb.embed(query)

# Retrieve using both vectors separately (pseudo-code)
# dense_results = vector_db.search(dense_vec, topk=100)
# sparse_results = vector_db.search(sparse_vec, topk=100)

# Fuse results using RRF
# reranker = RrfReRanker(topn=10)
# final_results = reranker.rerank([dense_results, sparse_results])
```

### 2. Two-Stage Retrieval

Fast recall first, then precise reranking:

```python
from zvec.extension import (
    DefaultLocalDenseEmbedding,
    DefaultLocalReRanker
)

# Stage 1: Fast recall
dense_emb = DefaultLocalDenseEmbedding()
query_vec = dense_emb.embed("machine learning tutorial")

# Recall top-100 (pseudo-code)
# candidates = vector_db.search(query_vec, topk=100)

# Stage 2: Precise reranking
reranker = DefaultLocalReRanker(
    query="machine learning tutorial",
    topn=10
)
# final_results = reranker.rerank(candidates, rerank_field="content")
```

### 3. Network Configuration for Chinese Users

```python
import os
from zvec.extension import DefaultLocalDenseEmbedding

# Option 1: Use ModelScope
embedding = DefaultLocalDenseEmbedding(model_source="modelscope")

# Option 2: Use Hugging Face mirror in Python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
embedding = DefaultLocalDenseEmbedding(model_source="huggingface")
```

---

## Important Notes

1. **Model Download**: Models will be downloaded on first use. Ensure network connectivity.
2. **Memory Management**: Local models consume memory. Call `clear_cache()` to release memory after use.
3. **API Rate Limiting**: When using API-based functions (Qwen, OpenAI), be mindful of quotas and rate limits.
4. **Thread Safety**: Embedding and reranking functions are thread-safe and can be used in multi-threaded environments.
5. **Multi-Vector Reranking**: `RrfReRanker` and `WeightedReRanker` are specifically designed for fusing results from multiple retrieval methods (e.g., dense + sparse). For single-vector results, use `DefaultLocalReRanker` or `QwenReRanker`.

---

## Related Documentation

- [Embedding Function Protocol](./embedding_function.py)
- [Reranking Function Protocol](./rerank_function.py)
- [Sentence Transformers Base Class](./sentence_transformer_function.py)
- [Qwen Function Base Class](./qwen_function.py)

---

## Contributing

Contributions for custom implementations and improvements are welcome! Please refer to the project's contribution guidelines.
