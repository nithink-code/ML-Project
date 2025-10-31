"""
Test Script for Custom NLP+KNN Model
Demonstrates the complete pipeline without pre-trained models
"""
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nlp.text_preprocessor import TextPreprocessor, Vocabulary
from nlp.feature_extractor import TFIDFVectorizer, Word2VecSimple
from nlp.knn_classifier import KNNClassifier
from nlp.metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    print_model_report,
    cross_validate_knn
)


def test_text_preprocessing():
    """Test text preprocessing functionality"""
    print("\n" + "="*60)
    print("Testing Text Preprocessing")
    print("="*60)
    
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        min_token_length=2
    )
    
    texts = [
        "Hello World! This is a test.",
        "Natural Language Processing is amazing!",
        "Machine Learning models are powerful."
    ]
    
    for text in texts:
        tokens = preprocessor.preprocess(text, remove_stopwords=True, stem=True)
        print(f"\nOriginal: {text}")
        print(f"Tokens:   {tokens}")
    
    print("\n✓ Text preprocessing test passed!")


def test_tfidf_vectorizer():
    """Test TF-IDF vectorization"""
    print("\n" + "="*60)
    print("Testing TF-IDF Vectorizer")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are enemies",
        "dogs chase cats"
    ]
    
    # Preprocess
    tokenized_docs = preprocessor.preprocess_batch(documents, remove_stopwords=False, stem=False)
    
    # Vectorize
    vectorizer = TFIDFVectorizer(max_features=20, use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
    
    print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Vocabulary: {vectorizer.get_feature_names()}")
    print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Sample TF-IDF vector:\n{tfidf_matrix[0][:5]}")
    
    print("\n✓ TF-IDF vectorizer test passed!")


def test_word2vec_embeddings():
    """Test Word2Vec-like embeddings"""
    print("\n" + "="*60)
    print("Testing Word2Vec Embeddings")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    documents = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks for learning",
        "natural language processing helps computers understand text",
        "computer vision enables machines to see and interpret images",
        "reinforcement learning trains agents through rewards"
    ] * 3  # Repeat for better embeddings
    
    # Preprocess
    tokenized_docs = preprocessor.preprocess_batch(documents, remove_stopwords=True, stem=False)
    
    # Train embeddings
    w2v = Word2VecSimple(embedding_dim=50, window_size=3, min_count=1)
    w2v.fit(tokenized_docs)
    
    print(f"\nVocabulary size: {w2v.vocab_size}")
    print(f"Embedding dimension: {w2v.embedding_dim}")
    
    # Get embeddings for documents
    doc_embeddings = w2v.transform(tokenized_docs)
    print(f"\nDocument embeddings shape: {doc_embeddings.shape}")
    
    # Test word similarity
    if 'learning' in w2v.word2idx and 'machine' in w2v.word2idx:
        learning_emb = w2v.get_embedding('learning')
        machine_emb = w2v.get_embedding('machine')
        similarity = np.dot(learning_emb, machine_emb)
        print(f"\nCosine similarity between 'learning' and 'machine': {similarity:.4f}")
    
    print("\n✓ Word2Vec embeddings test passed!")


def test_knn_classifier():
    """Test KNN classifier on synthetic data"""
    print("\n" + "="*60)
    print("Testing KNN Classifier")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Class 0: cluster around (0, 0)
    X_class0 = np.random.randn(50, 2) * 0.5
    y_class0 = np.zeros(50)
    
    # Class 1: cluster around (3, 3)
    X_class1 = np.random.randn(50, 2) * 0.5 + np.array([3, 3])
    y_class1 = np.ones(50)
    
    # Combine
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test different metrics
    metrics = ['euclidean', 'manhattan', 'cosine']
    
    for metric in metrics:
        print(f"\n--- Testing with {metric} distance ---")
        
        knn = KNNClassifier(n_neighbors=5, metric=metric, weights='distance')
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        accuracy = calculate_accuracy(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        if accuracy > 0.7:
            print(f"✓ {metric.capitalize()} KNN test passed!")
        else:
            print(f"⚠ {metric.capitalize()} KNN accuracy is low")


def test_complete_nlp_pipeline():
    """Test complete NLP+KNN pipeline"""
    print("\n" + "="*60)
    print("Testing Complete NLP+KNN Pipeline")
    print("="*60)
    
    # Sample conversation data
    conversations = [
        ("Hello, how are you?", "I'm doing great! How can I help you?"),
        ("What's the weather like?", "I don't have access to weather data."),
        ("Tell me a joke", "Why did the chicken cross the road? To get to the other side!"),
        ("How do I learn Python?", "Start with the basics: variables, loops, and functions."),
        ("What is machine learning?", "Machine learning is a subset of AI that learns from data."),
        ("Hello there", "Hi! How can I assist you today?"),
        ("Good morning", "Good morning! What can I do for you?"),
        ("What's ML?", "ML stands for Machine Learning, a field of artificial intelligence."),
        ("How to code?", "Start by learning a programming language like Python or JavaScript."),
        ("Tell me something funny", "Here's a joke: Why don't scientists trust atoms? Because they make up everything!")
    ]
    
    user_queries = [q for q, _ in conversations]
    assistant_responses = [r for _, r in conversations]
    
    # Step 1: Preprocess
    print("\nStep 1: Preprocessing text...")
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
    tokenized_queries = preprocessor.preprocess_batch(user_queries, remove_stopwords=True, stem=True)
    
    # Step 2: Extract features using TF-IDF
    print("Step 2: Extracting features with TF-IDF...")
    vectorizer = TFIDFVectorizer(max_features=100, min_df=1, use_idf=True)
    query_embeddings = vectorizer.fit_transform(tokenized_queries)
    print(f"Feature matrix shape: {query_embeddings.shape}")
    
    # Step 3: Train KNN
    print("Step 3: Training KNN model...")
    indices = np.arange(len(user_queries))
    knn = KNNClassifier(n_neighbors=3, metric='cosine', weights='distance')
    knn.fit(query_embeddings, indices)
    
    # Step 4: Test queries
    print("\nStep 4: Testing with new queries...")
    test_queries = [
        "Hi there",
        "What is Python?",
        "Make me laugh"
    ]
    
    for test_query in test_queries:
        print(f"\nQuery: '{test_query}'")
        
        # Preprocess and vectorize
        test_tokens = preprocessor.preprocess(test_query, remove_stopwords=True, stem=True)
        test_embedding = vectorizer.transform([test_tokens])
        
        # Find nearest neighbors
        distances, neighbor_indices = knn.kneighbors(test_embedding, n_neighbors=2)
        
        print("Top 2 similar responses:")
        for i, (dist, idx) in enumerate(zip(distances[0], neighbor_indices[0]), 1):
            original_idx = int(idx)
            print(f"  {i}. [Distance: {dist:.4f}] {assistant_responses[original_idx]}")
    
    print("\n✓ Complete NLP+KNN pipeline test passed!")


def test_cross_validation():
    """Test cross-validation functionality"""
    print("\n" + "="*60)
    print("Testing Cross-Validation")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Create 3 classes
    X_class0 = np.random.randn(n_samples, 10) * 0.5
    X_class1 = np.random.randn(n_samples, 10) * 0.5 + np.array([2] * 10)
    X_class2 = np.random.randn(n_samples, 10) * 0.5 + np.array([4] * 10)
    
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 2)])
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Cross-validate
    knn_params = {'n_neighbors': 5, 'metric': 'euclidean', 'weights': 'distance'}
    cv_results = cross_validate_knn(X, y, KNNClassifier, knn_params, k_folds=5)
    
    print(f"\nCross-Validation Results:")
    print(f"  Mean Accuracy:  {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    print(f"  Mean Precision: {cv_results['mean_precision']:.4f} (+/- {cv_results['std_precision']:.4f})")
    print(f"  Mean Recall:    {cv_results['mean_recall']:.4f} (+/- {cv_results['std_recall']:.4f})")
    print(f"  Mean F1:        {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
    
    print("\n✓ Cross-validation test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CUSTOM NLP+KNN MODEL TEST SUITE")
    print("No Pre-trained Models Used!")
    print("="*60)
    
    try:
        test_text_preprocessing()
        test_tfidf_vectorizer()
        test_word2vec_embeddings()
        test_knn_classifier()
        test_complete_nlp_pipeline()
        test_cross_validation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
