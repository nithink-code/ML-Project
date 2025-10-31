"""
Custom Feature Extraction Module
Implements TF-IDF and word embeddings from scratch
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


class TFIDFVectorizer:
    """Custom TF-IDF vectorizer implementation"""
    
    def __init__(self, max_features: Optional[int] = None, min_df: int = 1, 
                 max_df: float = 1.0, use_idf: bool = True, smooth_idf: bool = True,
                 sublinear_tf: bool = False):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        
        self.vocabulary_ = {}
        self.idf_ = None
        self.n_docs_ = 0
        
    def _build_vocabulary(self, tokenized_docs: List[List[str]]):
        """Build vocabulary from documents"""
        # Count document frequencies
        doc_freq = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
        
        # Filter by document frequency
        n_docs = len(tokenized_docs)
        min_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        
        valid_terms = {term for term, freq in doc_freq.items() 
                      if min_count <= freq <= max_count}
        
        # Sort by frequency and limit vocabulary size
        sorted_terms = sorted(valid_terms, key=lambda t: doc_freq[t], reverse=True)
        if self.max_features:
            sorted_terms = sorted_terms[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted_terms)}
        
        return doc_freq
    
    def _compute_idf(self, doc_freq: Counter, n_docs: int):
        """Compute IDF values"""
        idf = np.zeros(len(self.vocabulary_))
        
        for term, idx in self.vocabulary_.items():
            df = doc_freq[term]
            if self.smooth_idf:
                # Smooth IDF: log((n + 1) / (df + 1)) + 1
                idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1
            else:
                # Standard IDF: log(n / df) + 1
                idf[idx] = math.log(n_docs / df) + 1
        
        return idf
    
    def _compute_tf(self, tokens: List[str]) -> np.ndarray:
        """Compute term frequency vector for a document"""
        tf = np.zeros(len(self.vocabulary_))
        token_counts = Counter(tokens)
        
        for term, count in token_counts.items():
            if term in self.vocabulary_:
                idx = self.vocabulary_[term]
                if self.sublinear_tf:
                    # Sublinear TF scaling: 1 + log(tf)
                    tf[idx] = 1 + math.log(count)
                else:
                    # Raw term frequency
                    tf[idx] = count
        
        return tf
    
    def fit(self, tokenized_docs: List[List[str]]):
        """Fit the vectorizer on documents"""
        self.n_docs_ = len(tokenized_docs)
        
        # Build vocabulary and get document frequencies
        doc_freq = self._build_vocabulary(tokenized_docs)
        
        # Compute IDF
        if self.use_idf:
            self.idf_ = self._compute_idf(doc_freq, self.n_docs_)
        else:
            self.idf_ = np.ones(len(self.vocabulary_))
        
        return self
    
    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """Transform documents to TF-IDF vectors"""
        if not self.vocabulary_:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        n_docs = len(tokenized_docs)
        n_features = len(self.vocabulary_)
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for i, tokens in enumerate(tokenized_docs):
            tf = self._compute_tf(tokens)
            tfidf_matrix[i] = tf * self.idf_
        
        # L2 normalization
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        tfidf_matrix = tfidf_matrix / norms
        
        return tfidf_matrix
    
    def fit_transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary terms)"""
        return [term for term, _ in sorted(self.vocabulary_.items(), 
                                          key=lambda x: x[1])]


class Word2VecSimple:
    """Simple word embeddings using co-occurrence matrix and dimensionality reduction"""
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 5, 
                 min_count: int = 5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        self.vocab_size = 0
        
    def _build_vocab(self, tokenized_docs: List[List[str]]):
        """Build vocabulary from documents"""
        word_freq = Counter()
        for tokens in tokenized_docs:
            word_freq.update(tokens)
        
        # Filter by minimum count
        valid_words = [word for word, freq in word_freq.items() 
                      if freq >= self.min_count]
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(valid_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    
    def _build_cooccurrence_matrix(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """Build word co-occurrence matrix"""
        cooc_matrix = np.zeros((self.vocab_size, self.vocab_size))
        
        for tokens in tokenized_docs:
            for i, word in enumerate(tokens):
                if word not in self.word2idx:
                    continue
                
                word_idx = self.word2idx[word]
                
                # Get context window
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    
                    context_word = tokens[j]
                    if context_word in self.word2idx:
                        context_idx = self.word2idx[context_word]
                        # Weight by distance
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        cooc_matrix[word_idx, context_idx] += weight
        
        return cooc_matrix
    
    def _svd_reduction(self, matrix: np.ndarray, k: int) -> np.ndarray:
        """Simple SVD for dimensionality reduction"""
        # Center the matrix
        matrix_centered = matrix - np.mean(matrix, axis=0)
        
        # Use numpy's SVD
        U, S, Vt = np.linalg.svd(matrix_centered, full_matrices=False)
        
        # Keep top k components
        k = min(k, len(S))
        embeddings = U[:, :k] * S[:k]
        
        return embeddings
    
    def fit(self, tokenized_docs: List[List[str]]):
        """Train word embeddings"""
        # Build vocabulary
        self._build_vocab(tokenized_docs)
        
        if self.vocab_size == 0:
            raise ValueError("No valid words found in corpus")
        
        # Build co-occurrence matrix
        cooc_matrix = self._build_cooccurrence_matrix(tokenized_docs)
        
        # Apply dimensionality reduction
        self.embeddings = self._svd_reduction(cooc_matrix, self.embedding_dim)
        
        # L2 normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms
        
        return self
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word"""
        if word in self.word2idx:
            idx = self.word2idx[word]
            return self.embeddings[idx]
        return None
    
    def get_document_embedding(self, tokens: List[str]) -> np.ndarray:
        """Get document embedding by averaging word embeddings"""
        embeddings = []
        for token in tokens:
            emb = self.get_embedding(token)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros(self.embedding_dim)
        
        return np.mean(embeddings, axis=0)
    
    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """Transform documents to embeddings"""
        return np.array([self.get_document_embedding(tokens) 
                        for tokens in tokenized_docs])
