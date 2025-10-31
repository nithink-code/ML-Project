# Custom NLP + KNN Machine Learning Model

## Overview

This project implements a **fully custom Natural Language Processing (NLP) pipeline with K-Nearest Neighbors (KNN) classification** built entirely from scratch. **No pre-trained models are used** - all components are implemented using only NumPy.

## Features

### 🚀 Custom Components (No Pre-trained Models!)

1. **Text Preprocessing** (`text_preprocessor.py`)
   - Custom tokenization
   - Stopword removal (built-in English stopwords list)
   - Simple rule-based stemming
   - Punctuation and number handling
   - Vocabulary building

2. **Feature Extraction** (`feature_extractor.py`)
   - **TF-IDF Vectorizer** - Implemented from scratch
     - Term frequency calculation
     - Inverse document frequency
     - Sublinear TF scaling
     - L2 normalization
   - **Word2Vec-like Embeddings** - Simple co-occurrence based
     - Co-occurrence matrix construction
     - SVD-based dimensionality reduction
     - Document embedding aggregation

3. **KNN Classifier** (`knn_classifier.py`)
   - Multiple distance metrics:
     - Euclidean distance
     - Manhattan distance
     - Cosine distance
     - Minkowski distance
   - Weighted voting (uniform or distance-based)
   - KNN for classification and regression
   - Efficient neighbor search

4. **Evaluation Metrics** (`metrics.py`)
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - Cross-validation
   - Information retrieval metrics
   - Regression metrics (MSE, RMSE, MAE, R²)

## Architecture

```
User Query
    ↓
Text Preprocessing (tokenization, stopword removal, stemming)
    ↓
Feature Extraction (TF-IDF or Word2Vec)
    ↓
KNN Classifier (find k-nearest neighbors)
    ↓
Response Retrieval (return most similar responses)
```

## Installation

```bash
# Only numpy is required!
pip install numpy

# For database operations (optional)
pip install motor pymongo
```

## Usage

### Basic Example

```python
from nlp import (
    TextPreprocessor,
    TFIDFVectorizer,
    KNNClassifier
)
import numpy as np

# 1. Prepare your data
user_queries = [
    "Hello, how are you?",
    "What's the weather like?",
    "Tell me a joke"
]

responses = [
    "I'm doing great!",
    "I don't have weather data.",
    "Why did the chicken cross the road?"
]

# 2. Preprocess text
preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
tokenized_queries = preprocessor.preprocess_batch(
    user_queries, 
    remove_stopwords=True, 
    stem=True
)

# 3. Extract features
vectorizer = TFIDFVectorizer(max_features=5000, use_idf=True)
query_embeddings = vectorizer.fit_transform(tokenized_queries)

# 4. Train KNN
knn = KNNClassifier(n_neighbors=3, metric='cosine', weights='distance')
indices = np.arange(len(user_queries))
knn.fit(query_embeddings, indices)

# 5. Predict for new query
new_query = "Hi there"
new_tokens = preprocessor.preprocess(new_query, remove_stopwords=True, stem=True)
new_embedding = vectorizer.transform([new_tokens])

distances, neighbor_indices = knn.kneighbors(new_embedding, n_neighbors=2)
for dist, idx in zip(distances[0], neighbor_indices[0]):
    print(f"Response: {responses[int(idx)]} (distance: {dist:.4f})")
```

### Training from Database

```python
from nlp.trainer import train_from_db, predict_reply

# Train from MongoDB messages
result = await train_from_db(
    db, 
    model_dir="nlp_model",
    top_k=5,
    feature_method="tfidf",  # or "word2vec"
    distance_metric="cosine"  # or "euclidean", "manhattan"
)

print(f"Trained on {result['pairs_count']} conversation pairs")

# Predict response
prediction = predict_reply("Hello!", model_dir="nlp_model", top_k=3)
print(prediction['results'])
```

## Testing

Run the comprehensive test suite:

```bash
cd backend
python test_nlp_model.py
```

The test suite includes:
- ✅ Text preprocessing tests
- ✅ TF-IDF vectorization tests
- ✅ Word2Vec embedding tests
- ✅ KNN classifier tests with multiple metrics
- ✅ Complete NLP+KNN pipeline tests
- ✅ Cross-validation tests

## Configuration

Environment variables:

```bash
# Model directory
export NLP_MODEL_DIR="nlp_model"

# Number of neighbors
export NLP_TOP_K="5"

# Feature extraction method: "tfidf" or "word2vec"
export FEATURE_METHOD="tfidf"

# Distance metric: "euclidean", "manhattan", "cosine"
export KNN_METRIC="cosine"
```

## Model Artifacts

When trained, the model saves:
- `preprocessor.pkl` - Text preprocessor with settings
- `vectorizer.pkl` - TF-IDF or Word2Vec vectorizer
- `knn.pkl` - Trained KNN model
- `responses.pkl` - Training responses
- `config.pkl` - Model configuration

## Performance

The custom implementation provides:
- **Memory efficiency** - No large pre-trained models
- **Flexibility** - Full control over all components
- **Transparency** - Understand every step of the pipeline
- **Customizability** - Easy to modify for specific use cases

### Comparison with Pre-trained Models

| Feature | Custom NLP+KNN | Pre-trained (SentenceTransformer) |
|---------|---------------|-----------------------------------|
| Model Size | < 1 MB | 80-400 MB |
| Dependencies | numpy only | torch, transformers, etc. |
| Training Required | Yes | No |
| Customization | Full control | Limited |
| Interpretability | High | Low (black box) |
| Performance | Good for domain-specific | Excellent for general |

## Files Structure

```
backend/nlp/
├── __init__.py              # Module exports
├── text_preprocessor.py     # Text preprocessing & tokenization
├── feature_extractor.py     # TF-IDF & Word2Vec
├── knn_classifier.py        # KNN classifier & regressor
├── metrics.py               # Evaluation metrics
└── trainer.py               # Training & prediction pipeline

backend/
└── test_nlp_model.py        # Comprehensive test suite
```

## Algorithm Details

### TF-IDF Calculation

```
TF(t,d) = count(t,d) / total_terms(d)
IDF(t) = log(N / df(t)) + 1
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

With options for:
- Sublinear TF scaling: `1 + log(TF)`
- Smooth IDF: `log((N+1)/(df+1)) + 1`
- L2 normalization

### KNN Distance Metrics

1. **Euclidean**: `√(Σ(xi - yi)²)`
2. **Manhattan**: `Σ|xi - yi|`
3. **Cosine**: `1 - (x·y)/(||x||·||y||)`
4. **Minkowski**: `(Σ|xi - yi|^p)^(1/p)`

### Word Embeddings

1. Build co-occurrence matrix with context window
2. Apply SVD for dimensionality reduction
3. Aggregate word vectors for document representation

## Limitations

- Requires training data (conversation pairs)
- Performance depends on training data quality and size
- May not generalize as well as large pre-trained models
- Requires retraining for new domains

## Future Enhancements

- [ ] Add bigram/n-gram support
- [ ] Implement BM25 ranking
- [ ] Add more advanced stemming (Porter/Snowball)
- [ ] Implement approximate nearest neighbor search for speed
- [ ] Add model versioning and A/B testing
- [ ] Support for multiple languages
- [ ] Add attention-based pooling for document embeddings

## Contributing

This is a custom implementation for educational and production use. Feel free to extend and improve!

## License

MIT License - See LICENSE file for details

## Author

Built from scratch with ❤️ using only NumPy and Python standard library.

---

**No pre-trained models. No magic. Just pure ML implementation.** 🚀
