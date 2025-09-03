"""
Performance Predictor for SACOPOA ML Module
===========================================

Machine learning models for predicting student performance,
final grades, and course outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore')


class PerformancePredictor:
    """
    Machine learning predictor for student performance analysis.
    
    Capabilities:
    - Predict final grades based on early assessments
    - Forecast performance trends
    - Estimate probability of passing/failing
    - Multi-step ahead predictions
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the performance predictor.
        
        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boost', 'linear', 'ridge')
        """
        self.model_type = model_type
        self.models = {
            'performance': None,
            'grade_category': None,
            'pass_probability': None
        }
        self.feature_importance = {}
        self.training_metrics = {}
        self.is_trained = False
        
        # Initialize models based on type
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models based on specified type."""
        if self.model_type == 'random_forest':
            self.models['performance'] = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            self.models['grade_category'] = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            
        elif self.model_type == 'gradient_boost':
            self.models['performance'] = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            self.models['grade_category'] = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            
        elif self.model_type == 'linear':
            self.models['performance'] = LinearRegression()
            self.models['grade_category'] = LinearRegression()
            
        elif self.model_type == 'ridge':
            self.models['performance'] = Ridge(alpha=1.0)
            self.models['grade_category'] = Ridge(alpha=1.0)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y_performance: np.ndarray, 
              y_grade_category: np.ndarray = None, 
              feature_names: List[str] = None) -> Dict[str, float]:
        """
        Train the performance prediction models.
        
        Args:
            X: Feature matrix
            y_performance: Performance scores (continuous)
            y_grade_category: Grade categories (discrete)
            feature_names: Names of features
            
        Returns:
            Dictionary with training metrics
        """
        # Split data for validation
        X_train, X_test, y_perf_train, y_perf_test = train_test_split(
            X, y_performance, test_size=0.2, random_state=42
        )
        
        # Train performance model
        self.models['performance'].fit(X_train, y_perf_train)
        
        # Evaluate performance model
        y_pred = self.models['performance'].predict(X_test)
        perf_metrics = {
            'performance_mse': mean_squared_error(y_perf_test, y_pred),
            'performance_r2': r2_score(y_perf_test, y_pred),
            'performance_mae': mean_absolute_error(y_perf_test, y_pred)
        }
        
        # Train grade category model if provided
        if y_grade_category is not None:
            _, _, y_grade_train, y_grade_test = train_test_split(
                X, y_grade_category, test_size=0.2, random_state=42
            )
            
            self.models['grade_category'].fit(X_train, y_grade_train)
            y_grade_pred = self.models['grade_category'].predict(X_test)
            
            grade_metrics = {
                'grade_mse': mean_squared_error(y_grade_test, y_grade_pred),
                'grade_r2': r2_score(y_grade_test, y_grade_pred),
                'grade_mae': mean_absolute_error(y_grade_test, y_grade_pred)
            }
            perf_metrics.update(grade_metrics)
        
        # Store feature importance for tree-based models
        if hasattr(self.models['performance'], 'feature_importances_'):
            self.feature_importance['performance'] = self.models['performance'].feature_importances_
        
        if hasattr(self.models['grade_category'], 'feature_importances_'):
            self.feature_importance['grade_category'] = self.models['grade_category'].feature_importances_
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.models['performance'], X, y_performance, cv=5)
        perf_metrics['cv_mean'] = cv_scores.mean()
        perf_metrics['cv_std'] = cv_scores.std()
        
        self.training_metrics = perf_metrics
        self.is_trained = True
        
        return perf_metrics
    
    def predict_performance(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict student performance metrics.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Dictionary with various predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        
        # Performance predictions
        predictions['performance_scores'] = self.models['performance'].predict(X)
        
        # Grade category predictions
        if self.models['grade_category'] is not None:
            predictions['grade_categories'] = self.models['grade_category'].predict(X)
        
        # Calculate pass probability (assuming 40% threshold)
        predictions['pass_probability'] = self._calculate_pass_probability(
            predictions['performance_scores']
        )
        
        # Calculate confidence intervals for tree-based models
        if hasattr(self.models['performance'], 'estimators_'):
            predictions['prediction_intervals'] = self._calculate_prediction_intervals(X)
        
        return predictions
    
    def predict_early_performance(self, X_partial: np.ndarray, 
                                assessment_index: int) -> Dict[str, np.ndarray]:
        """
        Predict final performance based on early assessments.
        
        Args:
            X_partial: Feature matrix with only early assessments
            assessment_index: Index of the last available assessment
            
        Returns:
            Dictionary with early predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # For early prediction, we need to handle missing assessments
        # This is a simplified approach - can be enhanced with more sophisticated methods
        X_filled = self._fill_missing_assessments(X_partial, assessment_index)
        
        # Make predictions
        predictions = self.predict_performance(X_filled)
        
        # Add uncertainty estimates for early predictions
        predictions['early_prediction_confidence'] = self._calculate_early_confidence(
            assessment_index
        )
        
        return predictions
    
    def get_feature_importance(self, model_name: str = 'performance') -> Dict[str, float]:
        """Get feature importance for interpretability."""
        if model_name not in self.feature_importance:
            return {}
        
        importance = self.feature_importance[model_name]
        # Return as dictionary if feature names are available
        return dict(enumerate(importance))
    
    def _calculate_pass_probability(self, performance_scores: np.ndarray) -> np.ndarray:
        """Calculate probability of passing based on performance scores."""
        # Sigmoid transformation around 40% threshold
        threshold = 40.0
        # Use logistic function to convert scores to probabilities
        prob = 1 / (1 + np.exp(-(performance_scores - threshold) / 10))
        return prob
    
    def _calculate_prediction_intervals(self, X: np.ndarray, 
                                     confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate prediction intervals for tree-based models."""
        if not hasattr(self.models['performance'], 'estimators_'):
            return {}
        
        # Get predictions from all estimators
        estimator_predictions = np.array([
            estimator.predict(X) for estimator in self.models['performance'].estimators_
        ])
        
        # Calculate percentiles for confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(estimator_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(estimator_predictions, upper_percentile, axis=0)
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def _fill_missing_assessments(self, X_partial: np.ndarray, 
                                assessment_index: int) -> np.ndarray:
        """Fill missing assessment scores for early prediction."""
        # Simple strategy: use average of available assessments
        # More sophisticated methods could use time series forecasting
        
        X_filled = X_partial.copy()
        
        # Find assessment columns (assuming they're the last columns)
        # This is a simplified approach - should be parameterized
        assessment_start = X_partial.shape[1] - assessment_index - 1
        
        for i in range(X_partial.shape[0]):
            available_scores = X_partial[i, assessment_start:assessment_start + assessment_index + 1]
            avg_score = np.mean(available_scores[available_scores > 0])
            
            # Fill remaining assessments with average (if any available)
            if not np.isnan(avg_score):
                for j in range(assessment_index + 1, X_partial.shape[1] - assessment_start):
                    if assessment_start + j < X_filled.shape[1]:
                        X_filled[i, assessment_start + j] = avg_score
        
        return X_filled
    
    def _calculate_early_confidence(self, assessment_index: int) -> float:
        """Calculate confidence level for early predictions."""
        # Confidence decreases with fewer assessments
        max_assessments = 10  # Assuming maximum 10 assessments
        confidence = min(1.0, (assessment_index + 1) / max_assessments)
        return confidence
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'models': self.models,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained model."""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        summary = {
            "model_type": self.model_type,
            "training_metrics": self.training_metrics,
            "feature_importance_available": bool(self.feature_importance),
            "models_trained": list(self.models.keys())
        }
        
        return summary