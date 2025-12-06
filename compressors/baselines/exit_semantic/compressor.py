"""EXIT-Semantic implementation: Hybrid approach combining semantic filtering with EXIT.

This compressor combines:
1. Semantic similarity filtering (from Semantic Sentence Cover)
2. Context-aware extraction (from EXIT)

Process:
1. Compute embedding similarities between the whole query and each document sentence
2. Filter out the bottom 50% of sentences, keeping only the top 50% most similar
3. Apply EXIT methodology to the filtered sentences
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
import time
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.cuda.amp import autocast
from functools import lru_cache
import spacy
from ...base import BaseCompressor, SearchResult


class EXITSemanticCompressor(BaseCompressor):
    """EXIT-Semantic: Hybrid compressor combining semantic filtering with EXIT.
    
    This compressor first filters sentences based on semantic similarity to the query,
    then applies EXIT's context-aware extraction to the filtered sentences.
    """
    
    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        checkpoint: str = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
        device: str = None,
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5,
        semantic_filter_ratio: float = 0.5,
        num_hypothetical_documents: int = 0,
        hypothetical_document_model: str = None
    ):
        """Initialize EXIT-Semantic compressor.
        
        Args:
            base_model: Base model path for EXIT
            checkpoint: Path to trained EXIT checkpoint
            embedding_model: Sentence embedding model for semantic filtering
            device: Device to use (None for auto)
            cache_dir: Cache directory for models
            batch_size: Batch size for processing
            threshold: Confidence threshold for EXIT selection
            semantic_filter_ratio: Ratio of sentences to keep after semantic filtering (default: 0.5 = 50%)
            num_hypothetical_documents: Number of hypothetical documents to generate for HyDE (default: 0 = disabled)
            hypothetical_document_model: Model string for generating hypothetical documents (default: None = use base_model)
        """
        self.batch_size = batch_size
        self.threshold = threshold
        self.semantic_filter_ratio = semantic_filter_ratio
        self.num_hypothetical_documents = num_hypothetical_documents
        
        # Initialize timing statistics
        self.timing_stats = {
            'sentence_split': 0.0,
            'hyde_generation': 0.0,
            'query_encoding': 0.0,
            'doc_encoding': 0.0,
            'similarity_compute': 0.0,
            'filtering': 0.0,
            'exit_inference': 0.0,
            'total_compress': 0.0,
            'count': 0
        }
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load sentence embedding model for semantic filtering
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=self.device,
        )
        
        # Load spacy for sentence splitting
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")
        
        # Initialize EXIT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load EXIT model
        model_kwargs = {
            "device_map": "auto" if device is None else device,
            "torch_dtype": torch.bfloat16,
            # "load_in_4bit": True,
            "max_length": 4096,
        }
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )
        
        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint
            )
        else:
            self.model = self.base_model
        
        # Initialize hypothetical document model (use base_model if not specified)
        if hypothetical_document_model is None:
            # Use the same model reference for memory efficiency
            self.hypothetical_document_model = self.model
            self.hypothetical_document_tokenizer = self.tokenizer
        else:
            # Load separate model for hypothetical document generation
            self.hypothetical_document_tokenizer = AutoTokenizer.from_pretrained(
                hypothetical_document_model,
                use_fast=True
            )
            self.hypothetical_document_tokenizer.pad_token = self.hypothetical_document_tokenizer.eos_token
            self.hypothetical_document_tokenizer.padding_side = "left"
            
            self.hypothetical_document_model = AutoModelForCausalLM.from_pretrained(
                hypothetical_document_model,
                **model_kwargs
            )
            self.hypothetical_document_model.eval()
            # if hasattr(self.hypothetical_document_model, 'half'):
            #     self.hypothetical_document_model.half()
            
        # Prepare model
        self.model.eval()
        # if hasattr(self.model, 'half'):
        #     self.model.half()
            
        # Cache device and token IDs for EXIT
        self.exit_device = next(self.model.parameters()).device
        self.yes_token_id = self.tokenizer.encode(
            "Yes",
            add_special_tokens=False
        )[0]
        self.no_token_id = self.tokenizer.encode(
            "No",
            add_special_tokens=False
        )[0]
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _generate_hypothetical_documents(self, query: str) -> List[str]:
        """Generate hypothetical document sentences using HyDE approach.
        
        Args:
            query: The input query
            
        Returns:
            List of hypothetical document sentences
        """
        if self.num_hypothetical_documents <= 0:
            return []
        
        # Create prompt for hypothetical document generation
        prompt = (
            f'<start_of_turn>user\n'
            f'Given the following question, generate {self.num_hypothetical_documents} diverse sentences that could appear in documents '
            f'containing relevant information to answer this question. Each sentence should represent a different way the answer might be presented. '
            f'Make them concise and capture key phrases or concepts related to the question.\n\n'
            f'Question: {query}\n\n'
            f'Generate exactly {self.num_hypothetical_documents} hypothetical document sentences, one per line:<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )
        
        # Generate hypothetical documents
        with torch.cuda.amp.autocast():
            inputs = self.hypothetical_document_tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=2048
            ).to(next(self.hypothetical_document_model.parameters()).device)
            
            with torch.no_grad():
                outputs = self.hypothetical_document_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            generated_text = self.hypothetical_document_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
        # Parse the generated hypothetical documents
        # Split by newlines and take first num_hypothetical_documents non-empty lines
        hypothetical_docs = []
        for line in generated_text.split('\n'):
            line = line.strip()
            # Remove numbering if present (e.g., "1. ", "- ", etc.)
            line = re.sub(r'^[\d]+\.\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line and len(hypothetical_docs) < self.num_hypothetical_documents:
                hypothetical_docs.append(line)
        
        # Debug: print HyDE prompt and output
        import sys
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write("[HyDE DEBUG]\n")
        sys.stderr.write("="*80 + "\n")
        sys.stderr.write(f"Query: {query}\n")
        sys.stderr.write("-"*80 + "\n")
        sys.stderr.write(f"Full Prompt:\n{prompt}\n")
        sys.stderr.write("-"*80 + "\n")
        sys.stderr.write(f"Generated Text:\n{generated_text}\n")
        sys.stderr.write("-"*80 + "\n")
        sys.stderr.write(f"Parsed Hypothetical Documents ({len(hypothetical_docs)}):\n")
        for i, doc in enumerate(hypothetical_docs, 1):
            sys.stderr.write(f"  {i}. {doc}\n")
        sys.stderr.write("="*80 + "\n\n")
        sys.stderr.flush()
        
        return hypothetical_docs[:self.num_hypothetical_documents]
    
    def _semantic_filter(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> Tuple[List[SearchResult], List[int], Dict[str, float]]:
        """Filter sentences based on semantic similarity to the query.
        
        Args:
            query: Input question (treated as a single sentence)
            documents: List of documents to filter
            
        Returns:
            Tuple of (filtered_documents, kept_sentence_indices, timing_dict)
        """
        timing = {}
        
        # Step 1: Split documents into sentences
        t0 = time.time()
        all_sentences = []
        sentence_to_doc = []  # Track which document each sentence belongs to
        sentence_positions = []  # Track position within each document
        
        for doc_idx, doc in enumerate(documents):
            # Include title if present
            doc_text = f"{doc.title}\n{doc.text}" if doc.title else doc.text
            sentences = self._split_into_sentences(doc_text)
            
            for pos, sent in enumerate(sentences):
                all_sentences.append(sent)
                sentence_to_doc.append(doc_idx)
                sentence_positions.append(pos)
        timing['sentence_split'] = time.time() - t0
        
        if not all_sentences:
            return documents, [], timing
        
        # Step 2: Generate hypothetical documents and create expanded query
        t0 = time.time()
        hypothetical_docs = self._generate_hypothetical_documents(query)
        expanded_query = [query] + hypothetical_docs  # Query + hypothetical documents
        timing['hyde_generation'] = time.time() - t0
        
        # Step 3: Encode the expanded query (query + hypothetical documents)
        t0 = time.time()
        expanded_query_embeddings = self.embedding_model.encode(
            expanded_query,
            convert_to_tensor=True,
            device=self.device
        )  # Shape: (num_hypothetical_documents + 1, embedding_dim)
        timing['query_encoding'] = time.time() - t0
        
        # Step 4: Encode all document sentences
        t0 = time.time()
        doc_embeddings = self.embedding_model.encode(
            all_sentences,
            convert_to_tensor=True,
            device=self.device
        )  # Shape: (num_sentences, embedding_dim)
        timing['doc_encoding'] = time.time() - t0
        
        # Step 5: Compute cosine similarity between each document sentence and all expanded query embeddings
        t0 = time.time()
        # Vectorized computation: compute all similarities at once and take max per sentence
        # Shape: (num_sentences, num_expanded_queries)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=1)
        query_norm = F.normalize(expanded_query_embeddings, p=2, dim=1)
        similarity_matrix = doc_norm @ query_norm.T
        similarities = similarity_matrix.max(dim=1).values.cpu().numpy() # Shape: (num_sentences,)
        timing['similarity_compute'] = time.time() - t0
        
        # Step 6: Filter out bottom sentences, keep top by similarity
        t0 = time.time()
        num_to_keep = max(1, int(self.semantic_filter_ratio * len(all_sentences)))
        top_indices = np.argsort(similarities)[::-1][:num_to_keep]
        top_indices_sorted = np.sort(top_indices)  # Maintain original order
        
        # Step 7: Reconstruct documents with only kept sentences
        # Group sentences by document
        doc_sentences = {}
        for idx in top_indices_sorted:
            doc_idx = sentence_to_doc[idx]
            if doc_idx not in doc_sentences:
                doc_sentences[doc_idx] = []
            doc_sentences[doc_idx].append(all_sentences[idx])
        
        # Create filtered documents
        filtered_docs = []
        for doc_idx in sorted(doc_sentences.keys()):
            original_doc = documents[doc_idx]
            filtered_text = " ".join(doc_sentences[doc_idx])
            
            filtered_docs.append(SearchResult(
                evi_id=original_doc.evi_id,
                docid=original_doc.docid,
                title=original_doc.title,
                text=filtered_text,
                score=original_doc.score
            ))
        timing['filtering'] = time.time() - t0
        
        return filtered_docs, top_indices_sorted.tolist(), timing
    
    @lru_cache(maxsize=1024)
    def _generate_prompt(
        self,
        query: str,
        context: str,
        sentence: str
    ) -> str:
        """Generate prompt for EXIT relevance classification."""
        return (
            f'<start_of_turn>user\n'
            f'Query:\n{query}\n'
            f'Full context:\n{context}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )
    
    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict relevance for a batch of sentences using EXIT."""
        prompts = [
            self._generate_prompt(query, context, sentence)
            for query, context, sentence
            in zip(queries, contexts, sentences)
        ]
        
        with torch.cuda.amp.autocast():
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True
            )
            
            inputs = {
                k: v.to(self.exit_device, non_blocking=True)
                for k, v in inputs.items()
            }
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                relevant_logits = torch.stack([
                    next_token_logits[:, self.yes_token_id],
                    next_token_logits[:, self.no_token_id]
                ], dim=1)
                
                probs = torch.softmax(relevant_logits, dim=1)
                predictions = [
                    "Yes" if p else "No"
                    for p in probs.argmax(dim=1).cpu().numpy()
                ]
        
        return predictions, probs
    
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using EXIT-Semantic hybrid approach.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        compress_start = time.time()
        
        # Step 1: Semantic filtering - keep top 50% of sentences by similarity
        filtered_docs, kept_indices, semantic_timing = self._semantic_filter(query, documents)
        
        if not filtered_docs:
            return [SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text="",
                score=0.0
            )]
        
        # Step 2: Apply EXIT to the filtered documents
        exit_start = time.time()
        # Prepare full context from filtered documents
        context = "\n".join(
            f"{doc.title}\n{doc.text}" if doc.title else doc.text
            for doc in filtered_docs
        )
        
        selected_texts = []
        current_doc_id = None
        current_texts = []
        
        # Process each filtered document with EXIT
        for doc in filtered_docs:
            # Start new document
            if current_doc_id != doc.evi_id:
                if current_texts:
                    doc_text = " ".join(current_texts)
                    if doc_text.strip():
                        selected_texts.append(doc_text)
                current_doc_id = doc.evi_id
                current_texts = []
            
            # Split document text into sentences for EXIT processing
            sentences = self._split_into_sentences(doc.text)
            
            # Process sentences in batches
            for i in range(0, len(sentences), self.batch_size):
                batch_sentences = sentences[i:i + self.batch_size]
                batch_queries = [query] * len(batch_sentences)
                batch_contexts = [context] * len(batch_sentences)
                
                # Get predictions for batch
                predictions, probs = self._predict_batch(
                    batch_queries,
                    batch_contexts,
                    batch_sentences
                )
                
                # Add sentences that pass the threshold
                for sent, prob in zip(batch_sentences, probs):
                    if prob[0].item() >= self.threshold:
                        current_texts.append(sent)
        
        exit_time = time.time() - exit_start
        
        # Add last document if exists
        if current_texts:
            doc_text = " ".join(current_texts)
            if doc_text.strip():
                selected_texts.append(doc_text)
        
        # Combine all selected texts
        compressed_text = "\n\n".join(selected_texts)
        
        total_time = time.time() - compress_start
        
        # Update timing statistics
        self.timing_stats['sentence_split'] += semantic_timing.get('sentence_split', 0)
        self.timing_stats['hyde_generation'] += semantic_timing.get('hyde_generation', 0)
        self.timing_stats['query_encoding'] += semantic_timing.get('query_encoding', 0)
        self.timing_stats['doc_encoding'] += semantic_timing.get('doc_encoding', 0)
        self.timing_stats['similarity_compute'] += semantic_timing.get('similarity_compute', 0)
        self.timing_stats['filtering'] += semantic_timing.get('filtering', 0)
        self.timing_stats['exit_inference'] += exit_time
        self.timing_stats['total_compress'] += total_time
        self.timing_stats['count'] += 1
        
        # Log timing to file for every sample
        import sys
        timing_msg = (
            f"[Sample {self.timing_stats['count']}] "
            f"Split:{semantic_timing.get('sentence_split', 0):.3f}s "
            f"HyDE:{semantic_timing.get('hyde_generation', 0):.3f}s "
            f"QEnc:{semantic_timing.get('query_encoding', 0):.3f}s "
            f"DEnc:{semantic_timing.get('doc_encoding', 0):.3f}s "
            f"Sim:{semantic_timing.get('similarity_compute', 0):.3f}s "
            f"Filt:{semantic_timing.get('filtering', 0):.3f}s "
            f"EXIT:{exit_time:.3f}s "
            f"Total:{total_time:.3f}s"
        )
        # Force flush to ensure it's written immediately
        sys.stderr.write(timing_msg + "\n")
        sys.stderr.flush()
        
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=compressed_text,
            score=1.0
        )]
    
    def get_average_timing(self) -> Dict[str, float]:
        """Get average timing statistics across all compressions."""
        if self.timing_stats['count'] == 0:
            return {}
        
        count = self.timing_stats['count']
        return {
            'sentence_split': self.timing_stats['sentence_split'] / count,
            'hyde_generation': self.timing_stats['hyde_generation'] / count,
            'query_encoding': self.timing_stats['query_encoding'] / count,
            'doc_encoding': self.timing_stats['doc_encoding'] / count,
            'similarity_compute': self.timing_stats['similarity_compute'] / count,
            'filtering': self.timing_stats['filtering'] / count,
            'exit_inference': self.timing_stats['exit_inference'] / count,
            'total_compress': self.timing_stats['total_compress'] / count,
            'num_samples': count
        }
    
    def print_average_timing(self):
        """Print average timing statistics."""
        avg_timing = self.get_average_timing()
        if not avg_timing:
            print("No timing statistics available.")
            return
        
        print(f"\n{'='*70}")
        print(f"AVERAGE COMPRESSION TIMING ({avg_timing['num_samples']} samples)")
        print(f"{'='*70}")
        print(f"  1. Sentence Split:      {avg_timing['sentence_split']:.4f}s")
        print(f"  2. HyDE Generation:     {avg_timing['hyde_generation']:.4f}s")
        print(f"  3. Query Encoding:      {avg_timing['query_encoding']:.4f}s")
        print(f"  4. Document Encoding:   {avg_timing['doc_encoding']:.4f}s")
        print(f"  5. Similarity Compute:  {avg_timing['similarity_compute']:.4f}s")
        print(f"  6. Filtering:           {avg_timing['filtering']:.4f}s")
        print(f"  7. EXIT Inference:      {avg_timing['exit_inference']:.4f}s")
        print(f"  ---")
        print(f"  Total Compression:      {avg_timing['total_compress']:.4f}s")
        print(f"{'='*70}")
