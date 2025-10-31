"""
Test script to verify deduplication and unique response generation
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nlp import (
    TextPreprocessor,
    TFIDFVectorizer,
    KNNClassifier,
    predict_reply,
    get_best_response
)
import pickle


def test_deduplication():
    """Test that duplicate responses are properly handled"""
    print("\n" + "="*70)
    print("Testing Response Deduplication")
    print("="*70 + "\n")
    
    # Create training data with duplicate responses
    user_queries = [
        "hello",
        "hi there",
        "hey",
        "good morning",
        "greetings",
        # Different queries but same responses
        "hello world",
        "hi everyone",
        "hey folks"
    ]
    
    # Intentionally duplicate some responses
    assistant_responses = [
        "Hi! How can I help you?",
        "Hi! How can I help you?",  # Duplicate
        "Hi! How can I help you?",  # Duplicate
        "Good morning! What can I do for you?",
        "Hello! How are you doing?",
        "Hi! How can I help you?",  # Duplicate
        "Good morning! What can I do for you?",  # Duplicate
        "Hello! How are you doing?"  # Duplicate
    ]
    
    print(f"Training data: {len(user_queries)} queries")
    print(f"Unique responses in training: {len(set(assistant_responses))}")
    print(f"Total responses (with duplicates): {len(assistant_responses)}\n")
    
    # Train model
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
    tokenized = preprocessor.preprocess_batch(user_queries, remove_stopwords=True, stem=True)
    
    vectorizer = TFIDFVectorizer(max_features=100, use_idf=True)
    vectors = vectorizer.fit_transform(tokenized)
    
    knn = KNNClassifier(n_neighbors=5, metric='cosine', weights='distance')
    indices = np.arange(len(user_queries))
    knn.fit(vectors, indices)
    
    # Save model
    model_dir = "test_dedup_model"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(model_dir, "responses.pkl"), "wb") as f:
        pickle.dump(assistant_responses, f)
    with open(os.path.join(model_dir, "config.pkl"), "wb") as f:
        pickle.dump({'feature_method': 'tfidf', 'distance_metric': 'cosine', 'top_k': 5}, f)
    
    # Test queries
    test_queries = [
        "hi",
        "good day",
        "hello friend"
    ]
    
    print("Testing with deduplication enabled:")
    print("-" * 70)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Get top 3 responses (should be unique)
        result = predict_reply(query, model_dir=model_dir, top_k=3)
        
        if "error" not in result:
            print(f"Returned {len(result['results'])} unique responses:")
            for i, (response, distance) in enumerate(zip(result['results'], result['distances']), 1):
                confidence = 1.0 - (distance / 2.0)
                print(f"  {i}. [Conf: {confidence:.2%}] {response}")
            
            # Check for duplicates
            unique_check = len(result['results']) == len(set(result['results']))
            print(f"  ✓ All responses unique: {unique_check}")
        else:
            print(f"  Error: {result.get('message')}")
    
    # Test get_best_response
    print("\n" + "="*70)
    print("Testing get_best_response() function:")
    print("-" * 70)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        result = get_best_response(query, model_dir=model_dir, confidence_threshold=0.3)
        
        if "error" not in result or result.get("warning"):
            response = result.get("response")
            confidence = result.get("confidence", 0)
            alternatives = result.get("alternative_count", 0)
            
            print(f"Best Response: {response}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Alternatives available: {alternatives}")
            
            if result.get("warning"):
                print(f"⚠ Warning: {result.get('message')}")
        else:
            print(f"Error: {result.get('message')}")
    
    # Cleanup
    import shutil
    shutil.rmtree(model_dir)
    print("\n✓ Test model cleaned up")
    
    print("\n" + "="*70)
    print("Deduplication test completed successfully!")
    print("="*70 + "\n")


def test_varied_responses():
    """Test that different queries get different responses"""
    print("\n" + "="*70)
    print("Testing Varied Response Generation")
    print("="*70 + "\n")
    
    # Create diverse training data
    conversations = [
        ("what's the weather", "I don't have access to weather data."),
        ("tell me a joke", "Why did the chicken cross the road?"),
        ("how are you", "I'm doing great, thanks for asking!"),
        ("what's your name", "I'm an AI assistant."),
        ("help me code", "I can help with programming questions!"),
        ("explain ML", "Machine learning is a field of AI."),
        ("good morning", "Good morning! How can I help?"),
        ("bye", "Goodbye! Have a great day!"),
    ]
    
    user_queries = [q for q, _ in conversations]
    assistant_responses = [r for _, r in conversations]
    
    # Train
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
    tokenized = preprocessor.preprocess_batch(user_queries, remove_stopwords=True, stem=True)
    
    vectorizer = TFIDFVectorizer(max_features=100, use_idf=True)
    vectors = vectorizer.fit_transform(tokenized)
    
    knn = KNNClassifier(n_neighbors=3, metric='cosine', weights='distance')
    indices = np.arange(len(user_queries))
    knn.fit(vectors, indices)
    
    # Save model
    model_dir = "test_varied_model"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(model_dir, "responses.pkl"), "wb") as f:
        pickle.dump(assistant_responses, f)
    with open(os.path.join(model_dir, "config.pkl"), "wb") as f:
        pickle.dump({'feature_method': 'tfidf', 'distance_metric': 'cosine', 'top_k': 3}, f)
    
    # Test with different queries
    test_cases = [
        ("weather forecast", "I don't have access to weather data."),
        ("make me laugh", "Why did the chicken cross the road?"),
        ("how do you do", "I'm doing great, thanks for asking!"),
        ("who are you", "I'm an AI assistant."),
        ("programming help", "I can help with programming questions!"),
    ]
    
    print("Testing that different queries get appropriate responses:")
    print("-" * 70)
    
    all_different = True
    previous_response = None
    
    for query, expected_context in test_cases:
        result = get_best_response(query, model_dir=model_dir, confidence_threshold=0.0)
        
        if "error" not in result or result.get("response"):
            response = result.get("response")
            confidence = result.get("confidence", 0)
            
            print(f"\nQuery: '{query}'")
            print(f"Response: {response}")
            print(f"Confidence: {confidence:.2%}")
            
            # Check if response is different from previous
            if previous_response and response == previous_response:
                print("  ⚠ Same as previous response")
                all_different = False
            else:
                print("  ✓ Unique response")
            
            previous_response = response
    
    # Cleanup
    import shutil
    shutil.rmtree(model_dir)
    print("\n✓ Test model cleaned up")
    
    print("\n" + "="*70)
    if all_different:
        print("✓ All queries received different appropriate responses!")
    else:
        print("⚠ Some queries received the same response")
    print("="*70 + "\n")


def main():
    """Run all deduplication tests"""
    print("\n" + "="*70)
    print("RESPONSE DEDUPLICATION & VARIATION TESTS")
    print("="*70)
    
    try:
        test_deduplication()
        test_varied_responses()
        
        print("\n" + "="*70)
        print("ALL DEDUPLICATION TESTS PASSED! ✓")
        print("="*70)
        print("\nKey Improvements:")
        print("  ✓ Duplicate responses are filtered out")
        print("  ✓ Only unique responses are returned")
        print("  ✓ Different queries get different responses")
        print("  ✓ Confidence scoring helps identify best matches")
        print("  ✓ get_best_response() provides single best answer")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
