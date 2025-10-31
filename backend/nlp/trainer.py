import os
import pickle
import asyncio
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Import custom NLP components (no pre-trained models)
from .text_preprocessor import TextPreprocessor, Vocabulary
from .feature_extractor import TFIDFVectorizer, Word2VecSimple
from .knn_classifier import KNNClassifier

MODEL_DIR = os.environ.get("NLP_MODEL_DIR", "nlp_model")
TOP_K = int(os.environ.get("NLP_TOP_K", "5"))
FEATURE_METHOD = os.environ.get("FEATURE_METHOD", "tfidf")  # 'tfidf' or 'word2vec'
DISTANCE_METRIC = os.environ.get("KNN_METRIC", "cosine")  # 'euclidean', 'manhattan', 'cosine'


def _ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


async def train_from_db(db, model_dir: str = MODEL_DIR, top_k: int = TOP_K, 
                       feature_method: str = FEATURE_METHOD,
                       distance_metric: str = DISTANCE_METRIC) -> Dict[str, Any]:
    """
    Collect user->assistant pairs from db.messages, train custom NLP+KNN model,
    and persist artifacts to model_dir.
    
    Args:
        db: Database connection with messages collection
        model_dir: Directory to save model artifacts
        top_k: Number of neighbors for KNN
        feature_method: 'tfidf' or 'word2vec'
        distance_metric: Distance metric for KNN
        
    Returns:
        Dict with training status and metrics
    """
    # Fetch messages (cap to avoid OOM for very large DBs)
    cursor = db.messages.find({}, {"_id": 0}).sort("timestamp", 1)
    messages = await cursor.to_list(length=10000)

    # Build pairs (user_message -> assistant_message) by sequential pairing
    pairs: List[Tuple[str, str]] = []
    last_user: Optional[str] = None
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user:
            pairs.append((last_user, content))
            last_user = None

    if not pairs:
        return {"status": "no_pairs"}

    user_texts = [p[0] for p in pairs]
    assistant_texts = [p[1] for p in pairs]

    # Offload CPU work to thread
    result = await asyncio.to_thread(
        _fit_and_persist, user_texts, assistant_texts, 
        model_dir, top_k, feature_method, distance_metric
    )
    return result


def _fit_and_persist(user_texts: List[str], assistant_texts: List[str], 
                    model_dir: str, top_k: int, feature_method: str,
                    distance_metric: str) -> Dict[str, Any]:
    """
    Train and persist the NLP+KNN model
    """
    _ensure_dir(model_dir)

    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        min_token_length=2
    )
    
    # Preprocess texts
    print(f"Preprocessing {len(user_texts)} texts...")
    tokenized_texts = preprocessor.preprocess_batch(
        user_texts, 
        remove_stopwords=True, 
        stem=True
    )
    
    # Extract features
    print(f"Extracting features using {feature_method}...")
    if feature_method == "tfidf":
        vectorizer = TFIDFVectorizer(
            max_features=5000,
            min_df=1,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        embeddings = vectorizer.fit_transform(tokenized_texts)
    elif feature_method == "word2vec":
        vectorizer = Word2VecSimple(
            embedding_dim=100,
            window_size=5,
            min_count=2
        )
        vectorizer.fit(tokenized_texts)
        embeddings = vectorizer.transform(tokenized_texts)
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")
    
    # Train KNN
    print(f"Training KNN with {distance_metric} metric...")
    n_neighbors = min(len(user_texts), max(1, top_k))
    knn = KNNClassifier(
        n_neighbors=n_neighbors,
        metric=distance_metric,
        weights='distance'  # Weight by inverse distance
    )
    
    # For KNN, we use the embeddings as features and indices as labels
    # This allows us to retrieve the nearest training examples
    indices = np.arange(len(user_texts))
    knn.fit(embeddings, indices)
    
    # Persist artifacts
    print("Saving model artifacts...")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    knn_path = os.path.join(model_dir, "knn.pkl")
    responses_path = os.path.join(model_dir, "responses.pkl")
    config_path = os.path.join(model_dir, "config.pkl")
    
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    with open(responses_path, "wb") as f:
        pickle.dump(assistant_texts, f)
    with open(config_path, "wb") as f:
        pickle.dump({
            'feature_method': feature_method,
            'distance_metric': distance_metric,
            'top_k': top_k
        }, f)
    
    print(f"Training complete! Processed {len(user_texts)} conversation pairs.")
    return {
        "status": "trained",
        "pairs_count": len(user_texts),
        "feature_method": feature_method,
        "distance_metric": distance_metric,
        "vocabulary_size": len(vectorizer.vocabulary_) if feature_method == "tfidf" else vectorizer.vocab_size
    }


def load_index(model_dir: str = MODEL_DIR) -> Tuple[Optional[Any], Optional[List[str]], 
                                                      Optional[Any], Optional[Any], 
                                                      Optional[Dict]]:
    """
    Load persisted model artifacts
    
    Returns:
        Tuple of (knn, responses, preprocessor, vectorizer, config)
    """
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    knn_path = os.path.join(model_dir, "knn.pkl")
    responses_path = os.path.join(model_dir, "responses.pkl")
    config_path = os.path.join(model_dir, "config.pkl")
    
    # Check if all required files exist
    required_files = [preprocessor_path, vectorizer_path, knn_path, responses_path]
    if not all(os.path.exists(f) for f in required_files):
        return None, None, None, None, None
    
    # Load artifacts
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(knn_path, "rb") as f:
        knn = pickle.load(f)
    with open(responses_path, "rb") as f:
        responses = pickle.load(f)
    
    config = None
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    
    return knn, responses, preprocessor, vectorizer, config


def predict_reply(query: str, model_dir: str = MODEL_DIR, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Return top-k assistant responses for the given query using custom NLP+KNN model.
    
    Args:
        query: User query text
        model_dir: Directory with model artifacts
        top_k: Number of similar responses to return
        
    Returns:
        Dict with results (list of unique responses) and distances (list of floats)
    """
    # Load model
    knn, responses, preprocessor, vectorizer, config = load_index(model_dir)
    if knn is None:
        return {"error": "model_not_trained", "message": "Train the model first"}
    
    # Preprocess query
    tokenized_query = preprocessor.preprocess(
        query, 
        remove_stopwords=True, 
        stem=True
    )
    
    # Extract features
    if config and config.get('feature_method') == 'word2vec':
        query_embedding = vectorizer.transform([tokenized_query])
    else:  # tfidf
        query_embedding = vectorizer.transform([tokenized_query])
    
    # Find more neighbors than needed to account for duplicates
    # Request up to 3x top_k to ensure we get enough unique responses
    search_k = min(len(responses), max(top_k * 3, 20))
    distances, indices = knn.kneighbors(query_embedding, n_neighbors=search_k)
    
    # Deduplicate responses while preserving order (best matches first)
    seen_responses = set()
    unique_results = []
    unique_distances = []
    
    for dist, idx in zip(distances[0], indices[0]):
        response = responses[int(idx)]
        
        # Use normalized response for deduplication (case-insensitive, stripped)
        normalized_response = response.strip().lower()
        
        if normalized_response not in seen_responses:
            seen_responses.add(normalized_response)
            unique_results.append(response)  # Keep original formatting
            unique_distances.append(float(dist))
            
            # Stop once we have enough unique responses
            if len(unique_results) >= top_k:
                break
    
    # If we don't have enough results, return what we found
    if not unique_results:
        return {
            "error": "no_results",
            "message": "No matching responses found",
            "results": [],
            "distances": []
        }
    
    return {
        "results": unique_results,
        "distances": unique_distances,
        "feature_method": config.get('feature_method') if config else 'unknown',
        "distance_metric": config.get('distance_metric') if config else 'unknown',
        "total_matches": len(unique_results)
    }


def get_best_response(query: str, model_dir: str = MODEL_DIR, 
                     confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Get the single best response for a query with confidence scoring.
    
    Args:
        query: User query text
        model_dir: Directory with model artifacts
        confidence_threshold: Minimum confidence (0-1) to return a response
        
    Returns:
        Dict with response, confidence, and metadata
    """
    # Get top 3 to ensure we have options after deduplication
    prediction = predict_reply(query, model_dir=model_dir, top_k=3)
    
    if "error" in prediction:
        return prediction
    
    if not prediction.get("results"):
        return {
            "error": "no_response",
            "message": "No suitable response found",
            "response": None,
            "confidence": 0.0
        }
    
    # Get the best match
    best_response = prediction["results"][0]
    best_distance = prediction["distances"][0]
    
    # Convert distance to confidence (0-1 scale)
    # For cosine distance: distance is 0-2, where 0 is perfect match
    # For euclidean/manhattan: normalize based on typical range
    distance_metric = prediction.get("distance_metric", "cosine")
    
    if distance_metric == "cosine":
        # Cosine distance: 0 (identical) to 2 (opposite)
        # Convert to similarity: 1 - (distance / 2)
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / 2.0)))
    else:
        # For euclidean/manhattan, use inverse relationship
        # Assuming most good matches are within distance 1.0
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + best_distance)))
    
    # Check confidence threshold
    if confidence < confidence_threshold:
        return {
            "error": "low_confidence",
            "message": f"Best match has low confidence: {confidence:.2%}",
            "response": best_response,
            "confidence": confidence,
            "warning": "Response may not be relevant"
        }
    
    return {
        "response": best_response,
        "confidence": confidence,
        "distance": best_distance,
        "alternative_count": len(prediction["results"]) - 1,
        "feature_method": prediction.get("feature_method"),
        "distance_metric": prediction.get("distance_metric")
    }