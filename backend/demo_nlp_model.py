"""
Demo Script for Custom NLP+KNN Model
Showcases real-world usage without pre-trained models
"""
import numpy as np
import sys
import os
import pickle

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nlp import (
    TextPreprocessor,
    TFIDFVectorizer,
    Word2VecSimple,
    KNNClassifier,
    print_model_report,
    calculate_accuracy
)


def demo_chatbot_similarity():
    """Demonstrate building a simple similarity-based chatbot"""
    print("\n" + "="*70)
    print("DEMO: Simple Chatbot with Custom NLP+KNN")
    print("="*70 + "\n")
    
    # Sample conversation database
    conversation_pairs = [
        # Greetings
        ("hello", "Hi! How can I help you today?"),
        ("hi there", "Hello! What can I do for you?"),
        ("good morning", "Good morning! How are you doing?"),
        ("hey", "Hey there! How can I assist you?"),
        
        # Questions about ML
        ("what is machine learning", "Machine Learning is a field of AI that enables computers to learn from data without explicit programming."),
        ("explain deep learning", "Deep Learning uses neural networks with multiple layers to learn hierarchical representations of data."),
        ("what is AI", "Artificial Intelligence is the simulation of human intelligence by machines, especially computer systems."),
        ("how does ML work", "ML works by finding patterns in data and using those patterns to make predictions or decisions."),
        
        # Python questions
        ("how to learn python", "Start with basics like variables and functions, then practice with small projects. Online courses and documentation are great resources!"),
        ("what is python used for", "Python is used for web development, data science, automation, machine learning, and much more!"),
        ("python tutorial", "Check out Python.org's official tutorial, or try interactive platforms like Codecademy or freeCodeCamp."),
        
        # General help
        ("help me", "I'm here to help! What do you need assistance with?"),
        ("i need help", "Of course! What can I help you with?"),
        ("can you assist", "Absolutely! What would you like to know?"),
        
        # Jokes
        ("tell me a joke", "Why don't programmers like nature? It has too many bugs!"),
        ("make me laugh", "What's a programmer's favorite place to hang out? The Foo Bar!"),
        ("joke please", "Why do Java developers wear glasses? Because they don't C#!"),
    ]
    
    # Separate questions and answers
    questions = [q for q, _ in conversation_pairs]
    answers = [a for _, a in conversation_pairs]
    
    print(f"Training on {len(conversation_pairs)} conversation pairs...\n")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        min_token_length=2
    )
    
    # Preprocess questions
    print("Step 1: Preprocessing questions...")
    tokenized_questions = preprocessor.preprocess_batch(
        questions,
        remove_stopwords=True,
        stem=True
    )
    
    # Create TF-IDF features
    print("Step 2: Creating TF-IDF features...")
    vectorizer = TFIDFVectorizer(
        max_features=1000,
        min_df=1,
        use_idf=True,
        sublinear_tf=True
    )
    question_vectors = vectorizer.fit_transform(tokenized_questions)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature dimension: {question_vectors.shape[1]}")
    
    # Train KNN
    print("\nStep 3: Training KNN model...")
    indices = np.arange(len(questions))
    knn = KNNClassifier(
        n_neighbors=3,
        metric='cosine',
        weights='distance'
    )
    knn.fit(question_vectors, indices)
    print("  Model trained successfully!")
    
    # Interactive demo
    print("\n" + "="*70)
    print("CHATBOT READY! Ask me anything (type 'quit' to exit)")
    print("="*70 + "\n")
    
    test_queries = [
        "hi",
        "what is machine learning?",
        "how do I start with python",
        "i need some help",
        "tell me something funny"
    ]
    
    for query in test_queries:
        print(f"You: {query}")
        
        # Preprocess query
        query_tokens = preprocessor.preprocess(
            query,
            remove_stopwords=True,
            stem=True
        )
        
        # Vectorize
        query_vector = vectorizer.transform([query_tokens])
        
        # Find nearest neighbors
        distances, neighbor_indices = knn.kneighbors(query_vector, n_neighbors=1)
        
        # Get best match
        best_match_idx = int(neighbor_indices[0][0])
        response = answers[best_match_idx]
        confidence = 1 - distances[0][0]  # Convert distance to confidence
        
        print(f"Bot: {response}")
        print(f"     [Confidence: {confidence:.2%}]\n")
    
    print("="*70)
    print("Demo completed! The chatbot successfully matched queries to responses.")
    print("="*70 + "\n")


def demo_document_classification():
    """Demonstrate text classification with custom NLP+KNN"""
    print("\n" + "="*70)
    print("DEMO: Document Classification")
    print("="*70 + "\n")
    
    # Sample documents with categories
    documents = [
        # Sports
        "The team won the championship after a thrilling final match.",
        "The player scored three goals in the first half.",
        "Football fans celebrated the victory in the streets.",
        "The athlete broke the world record in the 100m sprint.",
        "Basketball season starts next month with new players.",
        
        # Technology
        "The new smartphone features advanced AI capabilities.",
        "Machine learning models are improving every year.",
        "Cloud computing enables scalable infrastructure.",
        "The software update includes security patches.",
        "Quantum computers could revolutionize cryptography.",
        
        # Food
        "The restaurant serves delicious Italian cuisine.",
        "This recipe requires fresh vegetables and herbs.",
        "The chef prepared a five-course meal.",
        "Chocolate cake is my favorite dessert.",
        "Organic food is becoming more popular.",
        
        # Science
        "Scientists discovered a new planet in a distant galaxy.",
        "The research paper explains quantum entanglement.",
        "DNA sequencing has advanced medical research.",
        "Climate change affects global weather patterns.",
        "The experiment yielded unexpected results.",
    ]
    
    labels = (
        ['sports'] * 5 +
        ['technology'] * 5 +
        ['food'] * 5 +
        ['science'] * 5
    )
    
    print(f"Dataset: {len(documents)} documents across {len(set(labels))} categories\n")
    
    # Shuffle data
    indices = np.arange(len(documents))
    np.random.shuffle(indices)
    documents = [documents[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Split into train/test
    split_idx = int(0.7 * len(documents))
    train_docs = documents[:split_idx]
    test_docs = documents[split_idx:]
    train_labels = np.array(labels[:split_idx])
    test_labels = np.array(labels[split_idx:])
    
    print(f"Training set: {len(train_docs)} documents")
    print(f"Test set: {len(test_docs)} documents\n")
    
    # Preprocess
    print("Preprocessing...")
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
    train_tokens = preprocessor.preprocess_batch(train_docs, remove_stopwords=True, stem=True)
    test_tokens = preprocessor.preprocess_batch(test_docs, remove_stopwords=True, stem=True)
    
    # Vectorize with TF-IDF
    print("Extracting TF-IDF features...")
    vectorizer = TFIDFVectorizer(max_features=500, use_idf=True)
    train_vectors = vectorizer.fit_transform(train_tokens)
    test_vectors = vectorizer.transform(test_tokens)
    
    # Train KNN
    print("Training KNN classifier...")
    knn = KNNClassifier(n_neighbors=3, metric='cosine', weights='distance')
    knn.fit(train_vectors, train_labels)
    
    # Predict
    print("Making predictions...\n")
    predictions = knn.predict(test_vectors)
    
    # Evaluate
    accuracy = calculate_accuracy(test_labels, predictions)
    
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Accuracy: {accuracy:.2%}\n")
    
    # Show some predictions
    print("Sample Predictions:")
    print("-" * 70)
    for i in range(min(5, len(test_docs))):
        print(f"\nDocument: {test_docs[i][:60]}...")
        print(f"True Label: {test_labels[i]}")
        print(f"Predicted: {predictions[i]}")
        print(f"Correct: {'✓' if predictions[i] == test_labels[i] else '✗'}")
    
    print("\n" + "="*70)
    print("Classification demo completed!")
    print("="*70 + "\n")


def demo_semantic_search():
    """Demonstrate semantic search with Word2Vec embeddings"""
    print("\n" + "="*70)
    print("DEMO: Semantic Search with Custom Word2Vec")
    print("="*70 + "\n")
    
    # Document corpus
    corpus = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning enables computers to learn from data automatically.",
        "Neural networks are inspired by biological neural networks in brains.",
        "Data science combines statistics, programming, and domain knowledge.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses multiple layers to learn complex patterns.",
        "Artificial intelligence aims to create intelligent machines.",
        "The weather today is sunny with clear blue skies.",
        "I love to cook Italian food on weekends.",
        "Exercise and healthy eating contribute to wellness.",
    ] * 3  # Repeat for better embeddings
    
    print(f"Corpus size: {len(corpus)} documents\n")
    
    # Preprocess
    print("Step 1: Preprocessing corpus...")
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
    tokenized_corpus = preprocessor.preprocess_batch(
        corpus,
        remove_stopwords=True,
        stem=False  # Keep full words for better embeddings
    )
    
    # Train Word2Vec
    print("Step 2: Training Word2Vec embeddings...")
    w2v = Word2VecSimple(
        embedding_dim=100,
        window_size=5,
        min_count=1
    )
    w2v.fit(tokenized_corpus)
    print(f"  Vocabulary size: {w2v.vocab_size}")
    print(f"  Embedding dimension: {w2v.embedding_dim}")
    
    # Create document embeddings
    print("\nStep 3: Creating document embeddings...")
    doc_embeddings = w2v.transform(tokenized_corpus)
    
    # Build search index
    print("Step 4: Building search index...")
    knn = KNNClassifier(n_neighbors=3, metric='cosine', weights='uniform')
    indices = np.arange(len(corpus))
    knn.fit(doc_embeddings, indices)
    
    # Search queries
    queries = [
        "programming languages",
        "AI and machine learning",
        "food and cooking"
    ]
    
    print("\n" + "="*70)
    print("SEARCH RESULTS")
    print("="*70)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        # Process query
        query_tokens = preprocessor.preprocess(query, remove_stopwords=True, stem=False)
        query_embedding = w2v.transform([query_tokens])
        
        # Search
        distances, result_indices = knn.kneighbors(query_embedding, n_neighbors=3)
        
        # Display results
        for rank, (dist, idx) in enumerate(zip(distances[0], result_indices[0]), 1):
            doc_idx = int(idx) % 10  # Get unique document
            similarity = 1 - dist
            print(f"{rank}. [Similarity: {similarity:.3f}] {corpus[doc_idx]}")
    
    print("\n" + "="*70)
    print("Semantic search demo completed!")
    print("="*70 + "\n")


def demo_save_and_load():
    """Demonstrate saving and loading the model"""
    print("\n" + "="*70)
    print("DEMO: Save and Load Model")
    print("="*70 + "\n")
    
    model_dir = "demo_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a simple model
    print("Creating and training a simple model...")
    
    docs = ["hello world", "goodbye world", "hello there", "machine learning"]
    preprocessor = TextPreprocessor()
    tokens = preprocessor.preprocess_batch(docs, remove_stopwords=False)
    
    vectorizer = TFIDFVectorizer()
    vectors = vectorizer.fit_transform(tokens)
    
    knn = KNNClassifier(n_neighbors=2, metric='cosine')
    indices = np.arange(len(docs))
    knn.fit(vectors, indices)
    
    # Save
    print(f"\nSaving model to '{model_dir}'...")
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(model_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    
    print("✓ Model saved successfully!")
    
    # Load
    print(f"\nLoading model from '{model_dir}'...")
    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        loaded_preprocessor = pickle.load(f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
        loaded_vectorizer = pickle.load(f)
    with open(os.path.join(model_dir, "knn.pkl"), "rb") as f:
        loaded_knn = pickle.load(f)
    with open(os.path.join(model_dir, "docs.pkl"), "rb") as f:
        loaded_docs = pickle.load(f)
    
    print("✓ Model loaded successfully!")
    
    # Test loaded model
    print("\nTesting loaded model...")
    test_query = "hello"
    test_tokens = loaded_preprocessor.preprocess(test_query, remove_stopwords=False)
    test_vector = loaded_vectorizer.transform([test_tokens])
    _, result_indices = loaded_knn.kneighbors(test_vector, n_neighbors=2)
    
    print(f"Query: '{test_query}'")
    print("Nearest documents:")
    for idx in result_indices[0]:
        print(f"  - {loaded_docs[int(idx)]}")
    
    print("\n✓ Model works correctly after loading!")
    
    # Cleanup
    import shutil
    shutil.rmtree(model_dir)
    print(f"\nCleaned up demo model directory.\n")
    
    print("="*70)
    print("Save/load demo completed!")
    print("="*70 + "\n")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("CUSTOM NLP+KNN MODEL - COMPREHENSIVE DEMO")
    print("Built from scratch - No pre-trained models!")
    print("="*70)
    
    try:
        demo_chatbot_similarity()
        demo_document_classification()
        demo_semantic_search()
        demo_save_and_load()
        
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY! ✓")
        print("="*70)
        print("\nKey Achievements:")
        print("  ✓ Custom text preprocessing")
        print("  ✓ TF-IDF vectorization from scratch")
        print("  ✓ Word2Vec-like embeddings")
        print("  ✓ KNN with multiple distance metrics")
        print("  ✓ Real-world applications demonstrated")
        print("  ✓ Model persistence and loading")
        print("\nAll implemented using only NumPy - no pre-trained models!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    exit(main())
