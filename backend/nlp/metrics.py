"""
Evaluation and Metrics Module
Provides utilities for model evaluation and performance measurement
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy"""
    return np.mean(y_true == y_pred)


def calculate_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                                  average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'weighted', 'macro', 'micro')
        
    Returns:
        Dict with precision, recall, and f1 scores
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if average == 'binary' and len(classes) != 2:
        raise ValueError("Binary averaging requires exactly 2 classes")
    
    # Calculate per-class metrics
    class_metrics = {}
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(y_true == cls)
        }
    
    # Average metrics
    if average == 'binary':
        # Use positive class (second class)
        pos_class = classes[1]
        return {
            'precision': class_metrics[pos_class]['precision'],
            'recall': class_metrics[pos_class]['recall'],
            'f1': class_metrics[pos_class]['f1']
        }
    elif average == 'macro':
        # Unweighted mean
        return {
            'precision': np.mean([m['precision'] for m in class_metrics.values()]),
            'recall': np.mean([m['recall'] for m in class_metrics.values()]),
            'f1': np.mean([m['f1'] for m in class_metrics.values()])
        }
    elif average == 'weighted':
        # Weighted by support
        total_support = sum(m['support'] for m in class_metrics.values())
        if total_support == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        
        return {
            'precision': sum(m['precision'] * m['support'] for m in class_metrics.values()) / total_support,
            'recall': sum(m['recall'] * m['support'] for m in class_metrics.values()) / total_support,
            'f1': sum(m['f1'] * m['support'] for m in class_metrics.values()) / total_support
        }
    else:
        raise ValueError(f"Unknown average method: {average}")


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Returns:
        Confusion matrix where rows are true labels and columns are predicted labels
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    return cm

def cross_validate(model, X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> Dict[str, Any]:
    """
    Perform k-fold cross validation
    
    Args:
        model: Model object with fit and predict methods
        X: Features array
        y: Labels array
        k_folds: Number of folds for cross-validation
        
    Returns:
        Dict containing mean and per-fold metrics
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
        
    fold_size = len(X) // k_folds
    metrics_per_fold = []
    
    for i in range(k_folds):
        # Split data into train and validation
        start_idx = i * fold_size
        end_idx = start_idx + fold_size
        
        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        
        X_train = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Train and predict
        model.fit(X_train)
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        fold_metrics = {
            'accuracy': calculate_accuracy(y_val, y_pred),
            **calculate_precision_recall_f1(y_val, y_pred)
        }
        metrics_per_fold.append(fold_metrics)
    
    # Calculate mean metrics
    mean_metrics = {}
    for metric in metrics_per_fold[0].keys():
        values = [fold[metric] for fold in metrics_per_fold]
        mean_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return {
        'mean_metrics': mean_metrics,
        'fold_metrics': metrics_per_fold
    }
    
    
def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def cross_validate_knn(X: np.ndarray, y: np.ndarray, knn_class, 
                       knn_params: Dict[str, Any], k_folds: int = 5) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for KNN model
    
    Args:
        X: Feature matrix
        y: Labels
        knn_class: KNN classifier class
        knn_params: Parameters for KNN initialization
        k_folds: Number of folds
        
    Returns:
        Dict with cross-validation results
    """
    n_samples = len(X)
    fold_size = n_samples // k_folds
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_scores = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    for fold in range(k_folds):
        # Split data
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else n_samples
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train and evaluate
        knn = knn_class(**knn_params)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        accuracy = calculate_accuracy(y_test, y_pred)
        metrics = calculate_precision_recall_f1(y_test, y_pred, average='weighted')
        
        fold_scores.append(accuracy)
        fold_precisions.append(metrics['precision'])
        fold_recalls.append(metrics['recall'])
        fold_f1s.append(metrics['f1'])
    
    return {
        'mean_accuracy': np.mean(fold_scores),
        'std_accuracy': np.std(fold_scores),
        'mean_precision': np.mean(fold_precisions),
        'std_precision': np.std(fold_precisions),
        'mean_recall': np.mean(fold_recalls),
        'std_recall': np.std(fold_recalls),
        'mean_f1': np.mean(fold_f1s),
        'std_f1': np.std(fold_f1s),
        'fold_scores': fold_scores
    }


def evaluate_text_similarity(query_embeddings: np.ndarray, 
                            candidate_embeddings: np.ndarray,
                            distance_metric: str = 'cosine') -> np.ndarray:
    """
    Calculate similarity scores between query and candidate embeddings
    
    Args:
        query_embeddings: Query embeddings (n_queries, n_features)
        candidate_embeddings: Candidate embeddings (n_candidates, n_features)
        distance_metric: Distance metric to use
        
    Returns:
        Similarity matrix (n_queries, n_candidates)
    """
    if distance_metric == 'cosine':
        # Cosine similarity: dot product of normalized vectors
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10)
        candidate_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10)
        similarity = np.dot(query_norm, candidate_norm.T)
    elif distance_metric == 'euclidean':
        # Negative Euclidean distance as similarity
        similarity = -np.sqrt(np.sum((query_embeddings[:, None, :] - candidate_embeddings[None, :, :]) ** 2, axis=2))
    elif distance_metric == 'manhattan':
        # Negative Manhattan distance as similarity
        similarity = -np.sum(np.abs(query_embeddings[:, None, :] - candidate_embeddings[None, :, :]), axis=2)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    return similarity


def calculate_retrieval_metrics(relevant_docs: List[int], retrieved_docs: List[int],
                                k: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate information retrieval metrics
    
    Args:
        relevant_docs: List of relevant document indices
        retrieved_docs: List of retrieved document indices (ordered by relevance)
        k: Number of top documents to consider (if None, use all)
        
    Returns:
        Dict with precision@k, recall@k, and F1@k
    """
    if k is not None:
        retrieved_docs = retrieved_docs[:k]
    
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)
    
    true_positives = len(relevant_set & retrieved_set)
    
    precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        f'precision@{k if k else "all"}': precision,
        f'recall@{k if k else "all"}': recall,
        f'f1@{k if k else "all"}': f1
    }


def print_model_report(y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> None:
    """
    Print a comprehensive model evaluation report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for the report header
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Report")
    print(f"{'='*60}\n")
    
    # Accuracy
    accuracy = calculate_accuracy(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    
    # Precision, Recall, F1
    metrics = calculate_precision_recall_f1(y_true, y_pred, average='weighted')
    print(f"Weighted Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}\n")
    
    # Confusion Matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"\n{'='*60}\n")
