"""
Model Evaluation Script
Tests the accuracy of the NLP model
"""
import numpy as np
from nlp.feature_extractor import TFIDFVectorizer, Word2VecSimple
from nlp.metrics import calculate_accuracy, calculate_precision_recall_f1, cross_validate
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

class ModelEvaluator:
    def __init__(self, max_features=5000, embedding_dim=100):
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.tfidf = TFIDFVectorizer(max_features=max_features)
        self.w2v = Word2VecSimple(embedding_dim=embedding_dim)
    
    def predict(self, model, X):
        """Make predictions using the given model and features"""
        try:
            return model.predict(X)
        except AttributeError:
            return np.argmax(model.predict_proba(X), axis=1)
    
    def evaluate_model(self, texts, labels, test_size=0.2, random_state=42):
        """
        Evaluate model performance using both TF-IDF and Word2Vec features
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        
        results = {}
        
        # TF-IDF Features
        print("Evaluating TF-IDF features...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)
        
        y_pred_tfidf = self.predict(self.tfidf, X_test_tfidf)
        tfidf_metrics = calculate_precision_recall_f1(y_test, y_pred_tfidf)
        tfidf_cv = cross_validate(self.tfidf, X_train_tfidf, y_train)
        
        results['tfidf'] = {
            'metrics': tfidf_metrics,
            'cross_validation': tfidf_cv
        }
        
        # Word2Vec Features
        print("\nEvaluating Word2Vec features...")
        X_train_w2v = self.w2v.fit_transform(X_train)
        X_test_w2v = self.w2v.transform(X_test)
        
        y_pred_w2v = self.predict(self.w2v, X_test_w2v)
        w2v_metrics = calculate_precision_recall_f1(y_test, y_pred_w2v)
        w2v_cv = cross_validate(self.w2v, X_train_w2v, y_train)
        
        results['word2vec'] = {
            'metrics': w2v_metrics,
            'cross_validation': w2v_cv
        }
        
        # Print results
        self._print_results(results)
        return results
    
    def _print_results(self, results):
        """Print formatted evaluation results"""
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            cv = model_results['cross_validation']
            
            print(f"\n{model_name.upper()} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Cross-validation accuracy: {cv['mean_metrics']['accuracy']['mean']:.4f} ± {cv['mean_metrics']['accuracy']['std']:.4f}")

def load_training_data(data_path):
    """Load training data from JSON file"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

if __name__ == "__main__":
    try:
        # Specify the path to your training data JSON file
        data_path = "training_data.json"
        texts, labels = load_training_data(data_path)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(texts, labels)
        
        # Save results to file
        output_path = Path("test_reports") / "model_evaluation_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")