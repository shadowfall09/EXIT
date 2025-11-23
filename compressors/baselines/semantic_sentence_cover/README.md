# Semantic Sentence Cover

## Overview

Semantic Sentence Cover is a geometric approach for RAG document compression that selects the most relevant sentences from retrieved documents. After selecting the sentences, they are placed back in their original order to form the compressed document.

This is the **first geometric approach for RAG document compression** proposed in the literature.

## Method

The approach consists of three main steps:

### 1. Sentence Embedding Generation

- Split the query into individual sentences
- Split all documents into individual sentences
- Generate embeddings for each sentence individually using an LLM (currently using Qwen3-Embedding-8B equivalent)

### 2. Query Expansion

The original query may be short and may not contain all the information necessary to answer the query. To find more relevant documents, we expand the query by including additional sentences:

- Run BM25 on all sentences, treating each sentence as a "document"
- Include the top scoring sentences as additional "query sentences" for the next step
- The number of sentences included is the **minimum of 10 and 10% of the total number of sentences** across all documents

**Note:** This query expansion methodology draws inspiration from pseudo-relevance feedback for query expansion in information retrieval systems. It helps capture multi-hop relationships without constructing a full knowledge graph, which would be computationally expensive.

### 3. Document Sampling

- For every sentence in all documents, compute its highest similarity to any query sentence
- Select the **top p% of document sentences** with highest similarity
- Use these sentences as context for the RAG retriever
- **Recommended starting value: p = 15%**

**Note:** We believe that sampling around the convex hull formed by the query sentences may give better results. However, this is not computationally tractable due to the scaling properties of convex hulls in higher dimensions (curse of dimensionality).

## Usage

### Quickstart

```python
from compressors import SemanticSentenceCoverCompressor, SearchResult

# Initialize compressor
compressor = SemanticSentenceCoverCompressor(
    embedding_model="Qwen/Qwen3-Embedding-4B",
    query_expansion_min=10,
    query_expansion_ratio=0.1,  # 10%
    document_sampling_ratio=0.15  # 15% (p value)
)

# Prepare documents
documents = [
    SearchResult(
        evi_id=0,
        docid=0,
        title="Document Title",
        text="Document content...",
        score=1.0
    )
]

# Compress
query = "What is the question?"
compressed_docs = compressor.compress(query, documents)
compressed_text = compressed_docs[0].text
```

### Running Experiments

```bash
# Run on all datasets with default parameters (p=15%)
python test_semantic_sentence_cover.py \
    --embedding_model "Qwen/Qwen3-Embedding-4B" \
    --reader_model "meta-llama/Llama-3.1-8B-Instruct" \
    --data_root "/path/to/data/retrieval" \
    --tasks HotpotQA 2wikimultihop musique NQ TQA \
    --top_k 5 \
    --retriever "contriever-msmarco" \
    --document_sampling_ratio 0.15 \
    --output_dir "./outputs/semantic_sentence_cover_results"
```

Or use the SLURM batch script:

```bash
sbatch reproduction_script/test_semantic_sentence_cover.sbatch
```

This will run experiments with p=10%, p=15%, and p=20% to find the optimal sampling ratio.

## Parameters

- `embedding_model`: Sentence embedding model (default: "Qwen/Qwen3-Embedding-4B")
- `query_expansion_min`: Minimum number of sentences for query expansion (default: 10)
- `query_expansion_ratio`: Ratio of total sentences to include in query expansion (default: 0.1 = 10%)
- `document_sampling_ratio`: Ratio of sentences to select for final compression (default: 0.15 = 15%, i.e., p=15%)
