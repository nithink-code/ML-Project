"""
NLP Module for Custom Machine Learning Pipeline
Provides text preprocessing, feature extraction, KNN classification, and evaluation
"""

from .text_preprocessor import TextPreprocessor, Vocabulary
from .feature_extractor import TFIDFVectorizer, Word2VecSimple
from .knn_classifier import KNNClassifier, KNNRegressor
from .metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    calculate_confusion_matrix,
    calculate_mse,
    calculate_rmse,
    calculate_mae,
    calculate_r2_score,
    cross_validate_knn,
    evaluate_text_similarity,
    calculate_retrieval_metrics,
    print_model_report
)
from .trainer import train_from_db, load_index, predict_reply, get_best_response

__all__ = [
    # Text preprocessing
    'TextPreprocessor',
    'Vocabulary',
    
    # Feature extraction
    'TFIDFVectorizer',
    'Word2VecSimple',
    
    # Classification
    'KNNClassifier',
    'KNNRegressor',
    
    # Metrics
    'calculate_accuracy',
    'calculate_precision_recall_f1',
    'calculate_confusion_matrix',
    'calculate_mse',
    'calculate_rmse',
    'calculate_mae',
    'calculate_r2_score',
    'cross_validate_knn',
    'evaluate_text_similarity',
    'calculate_retrieval_metrics',
    'print_model_report',
    
    # Training
    'train_from_db',
    'load_index',
    'predict_reply',
    'get_best_response',
]

__version__ = '1.0.0'
