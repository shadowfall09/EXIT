"""Semantic Sentence Cover implementation for RAG document compression.

A geometric approach which selects the sentences from retrieved documents 
which are most relevant to the query. After selecting the sentences, we place 
them back in their original order to form the compressed document.
"""

import torch
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import spacy
from ...base import BaseCompressor, SearchResult


class SemanticSentenceCoverCompressor(BaseCompressor):
    """Semantic Sentence Cover: Geometric approach for RAG document compression.
    
    Steps:
    1. Sentence Embedding Generation: Embed query and document sentences using LLM
    2. Query Expansion: Expand query using BM25 with top scoring sentences
    3. Document Sampling: Select top p% of sentences with highest similarity to any query sentence
    """
    
    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
        device: str = None,
        query_expansion_min: int = 10,
        query_expansion_ratio: float = 0.1,
        document_sampling_ratio: float = 0.15,
        dedup_threshold: float = 0.95,
        cache_dir: str = "./cache"
    ):
        """Initialize Semantic Sentence Cover compressor.
        
        Args:
            embedding_model: Sentence embedding model (using Qwen3-Embedding-8B equivalent)
            device: Device to use (None for auto)
            query_expansion_min: Minimum number of sentences for query expansion
            query_expansion_ratio: Ratio of total sentences to include in query expansion (10%)
            document_sampling_ratio: Ratio of sentences to select (p=15%)
            dedup_threshold: Similarity threshold for de-duplication (default: 0.95)
            cache_dir: Cache directory for models
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_expansion_min = query_expansion_min
        self.query_expansion_ratio = query_expansion_ratio
        self.document_sampling_ratio = document_sampling_ratio
        self.dedup_threshold = dedup_threshold
        
        # Load sentence embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=self.device,
            cache_folder=cache_dir
        )
        
        # Load spacy for sentence splitting
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _deduplicate_sentences(
        self,
        sentences: List[str],
        sentence_to_doc: List[int],
        embeddings: torch.Tensor
    ) -> Tuple[List[str], List[int], torch.Tensor]:
        """Remove near-duplicate sentences using embedding similarity.
        
        Args:
            sentences: List of all sentences
            sentence_to_doc: List mapping each sentence to its document index
            embeddings: Pre-computed sentence embeddings
            
        Returns:
            Tuple of (deduplicated_sentences, deduplicated_sentence_to_doc, deduplicated_embeddings)
        """
        if not sentences or self.dedup_threshold >= 1.0:
            return sentences, sentence_to_doc, embeddings
        
        # Compute pairwise cosine similarity
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        
        # Find duplicates (sentences with similarity above threshold, excluding self-similarity)
        keep_mask = [True] * len(sentences)
        
        for i in range(len(sentences)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(sentences)):
                if keep_mask[j] and similarity_matrix[i, j].item() >= self.dedup_threshold:
                    # Mark the later sentence for removal
                    keep_mask[j] = False
        
        # Filter sentences, mappings, and embeddings
        deduped_sentences = [s for i, s in enumerate(sentences) if keep_mask[i]]
        deduped_sentence_to_doc = [d for i, d in enumerate(sentence_to_doc) if keep_mask[i]]
        deduped_embeddings = embeddings[keep_mask]
        
        return deduped_sentences, deduped_sentence_to_doc, deduped_embeddings
    
    def _expand_query_bm25(
        self,
        query: str,
        all_sentences: List[str]
    ) -> List[str]:
        """Expand query using BM25 top scoring sentences.
        
        Args:
            query: Original query
            all_sentences: All sentences from documents
            
        Returns:
            List of query sentences (original + expanded)
        """
        # Split query into sentences
        query_sentences = self._split_into_sentences(query)
        
        # Tokenize sentences for BM25
        tokenized_corpus = [sent.lower().split() for sent in all_sentences]
        
        # Create BM25 index treating each sentence as a "document"
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Score all sentences
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Determine number of sentences to include
        num_to_include = max(
            self.query_expansion_min,
            int(self.query_expansion_ratio * len(all_sentences))
        )
        
        # Get top scoring sentences
        top_indices = np.argsort(scores)[::-1][:num_to_include]
        top_sentences = [all_sentences[i] for i in top_indices]
        
        # Combine original query sentences with top BM25 sentences
        expanded_query_sentences = query_sentences + top_sentences
        
        return expanded_query_sentences
    
    def _compute_similarity_matrix(
        self,
        query_sentences: List[str],
        doc_embeddings: torch.Tensor
    ) -> np.ndarray:
        """Compute similarity between query and document sentences.
        
        Args:
            query_sentences: Expanded query sentences
            doc_embeddings: Pre-computed document sentence embeddings
            
        Returns:
            Similarity matrix [num_doc_sentences, num_query_sentences]
        """
        # Generate embeddings for query sentences only
        query_embeddings = self.embedding_model.encode(
            query_sentences,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            doc_embeddings.unsqueeze(1),
            query_embeddings.unsqueeze(0),
            dim=2
        )
        
        return similarity.cpu().numpy()
    
    def _sample_documents(
        self,
        document_sentences: List[str],
        similarity_matrix: np.ndarray
    ) -> List[int]:
        """Sample top p% of document sentences based on max similarity.
        
        Args:
            document_sentences: All document sentences
            similarity_matrix: [num_doc_sentences, num_query_sentences]
            
        Returns:
            List of selected sentence indices (in original order)
        """
        # For each document sentence, find highest similarity to any query sentence
        max_similarities = similarity_matrix.max(axis=1)
        
        # Determine number of sentences to select
        num_to_select = max(
            1,
            int(self.document_sampling_ratio * len(document_sentences))
        )
        
        # Get top p% sentences by similarity
        top_indices = np.argsort(max_similarities)[::-1][:num_to_select]
        
        # Sort indices to maintain original order
        selected_indices = sorted(top_indices)
        
        return selected_indices
    
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using Semantic Sentence Cover.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        # Step 1: Sentence Embedding Generation - split all documents into sentences
        all_sentences = []
        sentence_to_doc = []  # Track which document each sentence belongs to
        
        for doc_idx, doc in enumerate(documents):
            # Include title if present
            doc_text = f"{doc.title}\n{doc.text}" if doc.title else doc.text
            sentences = self._split_into_sentences(doc_text)
            
            all_sentences.extend(sentences)
            sentence_to_doc.extend([doc_idx] * len(sentences))
        
        if not all_sentences:
            return [SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text="",
                score=0.0
            )]
        
        # Step 2: Encode all document sentences (single forward pass)
        doc_embeddings = self.embedding_model.encode(
            all_sentences,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Step 3: De-duplicate sentences using pre-computed embeddings
        all_sentences, sentence_to_doc, doc_embeddings = self._deduplicate_sentences(
            all_sentences,
            sentence_to_doc,
            doc_embeddings
        )
        
        # Step 4: Query Expansion using BM25
        expanded_query_sentences = self._expand_query_bm25(query, all_sentences)
        
        # Step 5: Compute similarity between query sentences and document sentences
        similarity_matrix = self._compute_similarity_matrix(
            expanded_query_sentences,
            doc_embeddings
        )
        
        # Step 6: Document Sampling - select top p% sentences
        selected_indices = self._sample_documents(
            all_sentences,
            similarity_matrix
        )
        
        # Step 7: Reconstruct compressed document in original order
        selected_sentences = [all_sentences[i] for i in selected_indices]
        compressed_text = " ".join(selected_sentences)
        
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=compressed_text,
            score=1.0
        )]
