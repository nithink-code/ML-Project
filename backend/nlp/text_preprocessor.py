"""
Custom Text Preprocessing Module
Implements tokenization, normalization, and cleaning without pre-trained models
"""
import re
import string
from typing import List, Set, Optional
from collections import Counter


class TextPreprocessor:
    """Custom text preprocessor for NLP tasks"""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_numbers: bool = False, min_token_length: int = 2):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length
        self.stopwords = self._load_stopwords()
        
    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords"""
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
            'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
            'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
            "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
            "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
            'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
            'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }
        return stopwords
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        # Replace common contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower() if self.lowercase else text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def clean_tokens(self, tokens: List[str]) -> List[str]:
        """Clean and filter tokens"""
        cleaned = []
        for token in tokens:
            # Remove punctuation if specified
            if self.remove_punctuation:
                token = token.translate(str.maketrans('', '', string.punctuation))
            
            # Remove numbers if specified
            if self.remove_numbers:
                token = re.sub(r'\d+', '', token)
            
            # Filter by minimum length
            if len(token) >= self.min_token_length:
                cleaned.append(token)
        
        return cleaned
    
    def simple_stem(self, word: str) -> str:
        """Simple rule-based stemming (Porter-like)"""
        # Remove common suffixes
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 'es', 's', 'ment']
        word = word.lower()
        
        for suffix in suffixes:
            if word.endswith(suffix):
                if len(word) - len(suffix) >= 3:  # Keep reasonable stem length
                    word = word[:-len(suffix)]
                    break
        
        return word
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   stem: bool = False) -> List[str]:
        """Complete preprocessing pipeline"""
        # Tokenize
        tokens = self.tokenize(text)
        
        # Clean tokens
        tokens = self.clean_tokens(tokens)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Stem
        if stem:
            tokens = [self.simple_stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_batch(self, texts: List[str], remove_stopwords: bool = True,
                        stem: bool = False) -> List[List[str]]:
        """Preprocess multiple texts"""
        return [self.preprocess(text, remove_stopwords, stem) for text in texts]


class Vocabulary:
    """Build and manage vocabulary for text corpus"""
    
    def __init__(self, min_freq: int = 1, max_vocab_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.vocab_size = 0
        
    def build_vocab(self, tokenized_texts: List[List[str]]):
        """Build vocabulary from tokenized texts"""
        # Count word frequencies
        for tokens in tokenized_texts:
            self.word_freq.update(tokens)
        
        # Filter by frequency
        valid_words = [word for word, freq in self.word_freq.items() 
                      if freq >= self.min_freq]
        
        # Sort by frequency and limit size
        valid_words = sorted(valid_words, key=lambda w: self.word_freq[w], reverse=True)
        if self.max_vocab_size:
            valid_words = valid_words[:self.max_vocab_size]
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(valid_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        return self
    
    def get_idx(self, word: str) -> Optional[int]:
        """Get index for a word"""
        return self.word2idx.get(word)
    
    def get_word(self, idx: int) -> Optional[str]:
        """Get word for an index"""
        return self.idx2word.get(idx)
