import os
import pickle
import asyncio
from typing import List, Tuple, Optional, Dict, Any

try:
    # SentenceTransformer can be heavy; installed via requirements
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    # Defer import errors until runtime; raise helpful message when used
    SentenceTransformer = None  # type: ignore
    NearestNeighbors = None  # type: ignore

MODEL_NAME = os.environ.get("SENTENCE_EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL_DIR = os.environ.get("NLP_MODEL_DIR", "nlp_model")
TOP_K = int(os.environ.get("NLP_TOP_K", "1"))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


async def train_from_db(db, model_dir: str = MODEL_DIR, top_k: int = TOP_K) -> Dict[str, Any]:
    """Collect user->assistant pairs from `db.messages`, fit a KNN on user embeddings,
    and persist the artifacts to `model_dir`.

    This function runs the DB collection in async code and offloads the CPU work
    (model encoding + fitting) to a thread via `asyncio.to_thread`.
    """
    if SentenceTransformer is None or NearestNeighbors is None:
        return {"error": "missing_dependencies", "message": "Install sentence-transformers and scikit-learn"}

    # Fetch messages (cap to avoid OOM for very large DBs)
    cursor = db.messages.find({}, {"_id": 0}).sort("timestamp", 1)
    messages = await cursor.to_list(length=10000)  # tune this limit as needed

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

    # Offload heavy CPU work to a thread
    result = await asyncio.to_thread(_fit_and_persist, user_texts, assistant_texts, model_dir, top_k, MODEL_NAME)
    return result


def _fit_and_persist(user_texts: List[str], assistant_texts: List[str], model_dir: str, top_k: int, model_name: str) -> Dict[str, Any]:
    _ensure_dir(model_dir)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(user_texts, convert_to_numpy=True, show_progress_bar=False)

    n_neighbors = min(len(user_texts), max(1, top_k))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)

    # Persist artifacts
    knn_path = os.path.join(model_dir, "knn.pkl")
    responses_path = os.path.join(model_dir, "responses.pkl")
    model_name_path = os.path.join(model_dir, "model_name.txt")

    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    with open(responses_path, "wb") as f:
        pickle.dump(assistant_texts, f)
    with open(model_name_path, "w") as f:
        f.write(model_name)

    return {"status": "trained", "pairs_count": len(user_texts)}


def load_index(model_dir: str = MODEL_DIR):
    knn_path = os.path.join(model_dir, "knn.pkl")
    responses_path = os.path.join(model_dir, "responses.pkl")
    model_name_path = os.path.join(model_dir, "model_name.txt")

    if not (os.path.exists(knn_path) and os.path.exists(responses_path)):
        return None, None, None

    knn = pickle.load(open(knn_path, "rb"))
    responses = pickle.load(open(responses_path, "rb"))
    model_name = None
    if os.path.exists(model_name_path):
        try:
            model_name = open(model_name_path, "r").read().strip()
        except Exception:
            model_name = model_name or MODEL_NAME
    else:
        model_name = MODEL_NAME

    model = SentenceTransformer(model_name)
    return knn, responses, model


def predict_reply(query: str, model_dir: str = MODEL_DIR, top_k: int = TOP_K) -> Dict[str, Any]:
    """Return top-k assistant responses for the given `query` using the persisted index.

    Returns a dict with keys: results (list of responses) and distances (list of floats).
    """
    if SentenceTransformer is None or NearestNeighbors is None:
        return {"error": "missing_dependencies", "message": "Install sentence-transformers and scikit-learn"}

    knn, responses, model = load_index(model_dir)
    if knn is None:
        return {"error": "model_not_trained"}

    q_emb = model.encode([query], convert_to_numpy=True)
    n = min(top_k, len(responses))
    dists, idxs = knn.kneighbors(q_emb, n_neighbors=n)
    results = [responses[i] for i in idxs[0]]

    return {"results": results, "distances": dists[0].tolist()}