"""
Custom K-Nearest Neighbors Classifier
Implements KNN from scratch with multiple distance metrics
"""
import numpy as np
from typing import List, Tuple, Optional, Union, Callable
from collections import Counter
import heapq


class KNNClassifier:
    """K-Nearest Neighbors classifier built from scratch"""
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean', 
                 weights: str = 'uniform', algorithm: str = 'brute'):
        """
        Initialize KNN classifier
        
        Args:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
            weights: Weight function ('uniform' or 'distance')
            algorithm: Algorithm to compute neighbors ('brute' or 'kd_tree')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.algorithm = algorithm
        
        self.X_train = None
        self.y_train = None
        self.n_samples = 0
        self.n_features = 0
        
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Manhattan distance"""
        return np.sum(np.abs(x1 - x2))
    
    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Cosine distance (1 - cosine similarity)"""
        dot_product = np.dot(x1, x2)
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        similarity = dot_product / (norm1 * norm2)
        return 1 - similarity
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray, p: int = 3) -> float:
        """Compute Minkowski distance"""
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
    
    def _get_distance_function(self) -> Callable:
        """Get the distance function based on metric"""
        if self.metric == 'euclidean':
            return self._euclidean_distance
        elif self.metric == 'manhattan':
            return self._manhattan_distance
        elif self.metric == 'cosine':
            return self._cosine_distance
        elif self.metric == 'minkowski':
            return self._minkowski_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances from x to all training samples"""
        distance_fn = self._get_distance_function()
        distances = np.array([distance_fn(x, x_train) for x_train in self.X_train])
        return distances
    
    def _get_k_nearest(self, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices and distances of k nearest neighbors"""
        k = min(self.n_neighbors, len(distances))
        
        # Get k smallest distances and their indices
        k_indices = np.argpartition(distances, k-1)[:k]
        k_distances = distances[k_indices]
        
        # Sort them
        sorted_idx = np.argsort(k_distances)
        k_indices = k_indices[sorted_idx]
        k_distances = k_distances[sorted_idx]
        
        return k_indices, k_distances
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNN model
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.n_samples, self.n_features = self.X_train.shape
        
        return self
    
    def predict_single(self, x: np.ndarray) -> Union[int, str]:
        """Predict label for a single sample"""
        # Compute distances
        distances = self._compute_distances(x)
        
        # Get k nearest neighbors
        k_indices, k_distances = self._get_k_nearest(distances)
        
        # Get labels of k nearest neighbors
        k_labels = self.y_train[k_indices]
        
        # Apply weights
        if self.weights == 'uniform':
            # Uniform weights: simple majority vote
            label_counts = Counter(k_labels)
            prediction = label_counts.most_common(1)[0][0]
        else:  # distance weights
            # Weight by inverse distance
            weights = np.zeros(len(k_labels))
            for i, dist in enumerate(k_distances):
                if dist == 0:
                    weights[i] = 1e10  # Very large weight for exact match
                else:
                    weights[i] = 1 / dist
            
            # Weighted vote
            label_weights = {}
            for label, weight in zip(k_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight
            
            prediction = max(label_weights, key=label_weights.get)
        
        return prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for multiple samples
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        X = np.array(X)
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
    
    def predict_proba_single(self, x: np.ndarray) -> dict:
        """Get class probabilities for a single sample"""
        # Compute distances
        distances = self._compute_distances(x)
        
        # Get k nearest neighbors
        k_indices, k_distances = self._get_k_nearest(distances)
        
        # Get labels of k nearest neighbors
        k_labels = self.y_train[k_indices]
        
        # Apply weights
        if self.weights == 'uniform':
            # Count labels
            label_counts = Counter(k_labels)
            total = len(k_labels)
            probas = {label: count / total for label, count in label_counts.items()}
        else:  # distance weights
            # Weight by inverse distance
            weights = np.zeros(len(k_labels))
            for i, dist in enumerate(k_distances):
                if dist == 0:
                    weights[i] = 1e10
                else:
                    weights[i] = 1 / dist
            
            # Weighted probabilities
            label_weights = {}
            for label, weight in zip(k_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight
            
            total_weight = sum(label_weights.values())
            probas = {label: weight / total_weight for label, weight in label_weights.items()}
        
        return probas
    
    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None, 
                   return_distance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k-neighbors of samples
        
        Args:
            X: Query samples
            n_neighbors: Number of neighbors (uses self.n_neighbors if None)
            return_distance: Whether to return distances
            
        Returns:
            Tuple of (distances, indices) if return_distance=True, else just indices
        """
        X = np.array(X)
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        
        all_indices = []
        all_distances = []
        
        for x in X:
            distances = self._compute_distances(x)
            k_indices, k_distances = self._get_k_nearest(distances)
            all_indices.append(k_indices)
            all_distances.append(k_distances)
        
        indices = np.array(all_indices)
        distances = np.array(all_distances)
        
        if return_distance:
            return distances, indices
        else:
            return indices
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


class KNNRegressor(KNNClassifier):
    """K-Nearest Neighbors regressor for continuous values"""
    
    def predict_single(self, x: np.ndarray) -> float:
        """Predict value for a single sample"""
        # Compute distances
        distances = self._compute_distances(x)
        
        # Get k nearest neighbors
        k_indices, k_distances = self._get_k_nearest(distances)
        
        # Get values of k nearest neighbors
        k_values = self.y_train[k_indices]
        
        # Apply weights
        if self.weights == 'uniform':
            # Simple average
            prediction = np.mean(k_values)
        else:  # distance weights
            # Weighted average
            weights = np.zeros(len(k_values))
            for i, dist in enumerate(k_distances):
                if dist == 0:
                    # Exact match: return that value
                    return k_values[i]
                else:
                    weights[i] = 1 / dist
            
            prediction = np.average(k_values, weights=weights)
        
        return prediction
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score
        
        Args:
            X: Test data
            y: True values
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
