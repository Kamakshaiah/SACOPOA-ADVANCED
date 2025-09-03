"""
Data Processor for SACOPOA ML Module
====================================

Handles data preprocessing, feature engineering, and data validation
for machine learning models in the SACOPOA system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Data processing utility for SACOPOA machine learning operations.
    
    Handles:
    - Data cleaning and validation
    - Feature engineering for student performance data
    - Data transformation for ML models
    - Integration with existing SACOPOA data structures
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def process_student_data(self, student_df: pd.DataFrame, co_po_matrix: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Process raw student data for ML analysis.
        
        Args:
            student_df: DataFrame with student assessment data
            co_po_matrix: Optional CO-PO mapping matrix
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Validate input data
        self._validate_student_data(student_df)
        
        # Extract features
        features = self._extract_features(student_df)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(student_df)
        
        # Generate feature matrix
        feature_matrix = self._create_feature_matrix(features, performance_metrics)
        
        # Add CO-PO related features if matrix is provided
        if co_po_matrix is not None:
            co_po_features = self._extract_co_po_features(student_df, co_po_matrix)
            feature_matrix = pd.concat([feature_matrix, co_po_features], axis=1)
        
        return {
            'features': feature_matrix,
            'raw_data': student_df,
            'performance_metrics': performance_metrics,
            'feature_names': list(feature_matrix.columns),
            'student_ids': student_df['Roll Number'].tolist() if 'Roll Number' in student_df.columns else None
        }
    
    def _validate_student_data(self, df: pd.DataFrame) -> None:
        """Validate the structure and content of student data."""
        required_columns = ['Roll Number', 'Name']
        assessment_columns = [col for col in df.columns if col.startswith('A')]
        
        if not assessment_columns:
            raise ValueError("No assessment columns found. Expected columns starting with 'A' (e.g., A1, A2, A3)")
        
        # Check for missing required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate assessment data
        for col in assessment_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    raise ValueError(f"Assessment column {col} contains non-numeric data")
    
    def _extract_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract basic features from student data."""
        assessment_cols = [col for col in df.columns if col.startswith('A')]
        assessment_data = df[assessment_cols].fillna(0)
        
        features = {
            'total_score': assessment_data.sum(axis=1).values,
            'average_score': assessment_data.mean(axis=1).values,
            'max_score': assessment_data.max(axis=1).values,
            'min_score': assessment_data.min(axis=1).values,
            'score_std': assessment_data.std(axis=1).fillna(0).values,
            'score_range': (assessment_data.max(axis=1) - assessment_data.min(axis=1)).values,
            'num_assessments': len(assessment_cols),
            'assessment_scores': assessment_data.values
        }
        
        return features
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        assessment_cols = [col for col in df.columns if col.startswith('A')]
        assessment_data = df[assessment_cols].fillna(0)
        
        # Assume maximum score per assessment is 100 (can be parameterized)
        max_score_per_assessment = 100
        total_possible = len(assessment_cols) * max_score_per_assessment
        
        metrics = {
            'percentages': (assessment_data.sum(axis=1) / total_possible * 100).values,
            'pass_fail': (assessment_data.sum(axis=1) >= (total_possible * 0.4)).astype(int).values,
            'excellence': (assessment_data.sum(axis=1) >= (total_possible * 0.85)).astype(int).values,
            'grade_categories': self._categorize_performance(assessment_data.sum(axis=1) / total_possible * 100),
            'improvement_trend': self._calculate_trend(assessment_data.values),
            'consistency_score': self._calculate_consistency(assessment_data.values)
        }
        
        return metrics
    
    def _categorize_performance(self, percentages: pd.Series) -> np.ndarray:
        """Categorize student performance into standard bands."""
        categories = np.zeros(len(percentages))
        categories[percentages >= 85] = 4  # Excellent
        categories[(percentages >= 70) & (percentages < 85)] = 3  # Good
        categories[(percentages >= 55) & (percentages < 70)] = 2  # Average
        categories[(percentages >= 40) & (percentages < 55)] = 1  # Below Average
        categories[percentages < 40] = 0  # Poor
        
        return categories
    
    def _calculate_trend(self, assessment_scores: np.ndarray) -> np.ndarray:
        """Calculate performance trend for each student."""
        trends = []
        for scores in assessment_scores:
            if len(scores) < 2:
                trends.append(0)
                continue
            
            # Simple linear trend calculation
            x = np.arange(len(scores))
            trend = np.polyfit(x, scores, 1)[0]  # Slope of linear fit
            trends.append(trend)
        
        return np.array(trends)
    
    def _calculate_consistency(self, assessment_scores: np.ndarray) -> np.ndarray:
        """Calculate consistency score for each student."""
        consistency_scores = []
        for scores in assessment_scores:
            if len(scores) < 2:
                consistency_scores.append(1.0)
                continue
            
            # Consistency as inverse of coefficient of variation
            mean_score = np.mean(scores)
            if mean_score == 0:
                consistency_scores.append(0.0)
            else:
                cv = np.std(scores) / mean_score
                consistency = max(0, 1 - cv)  # Higher is more consistent
                consistency_scores.append(consistency)
        
        return np.array(consistency_scores)
    
    def _create_feature_matrix(self, features: Dict[str, np.ndarray], 
                             metrics: Dict[str, Any]) -> pd.DataFrame:
        """Create the final feature matrix for ML models."""
        feature_dict = {
            'total_score': features['total_score'],
            'average_score': features['average_score'],
            'max_score': features['max_score'],
            'min_score': features['min_score'],
            'score_std': features['score_std'],
            'score_range': features['score_range'],
            'percentage': metrics['percentages'],
            'improvement_trend': metrics['improvement_trend'],
            'consistency_score': metrics['consistency_score'],
            'grade_category': metrics['grade_categories']
        }
        
        # Add individual assessment scores
        assessment_scores = features['assessment_scores']
        for i in range(assessment_scores.shape[1]):
            feature_dict[f'assessment_{i+1}'] = assessment_scores[:, i]
        
        return pd.DataFrame(feature_dict)
    
    def _extract_co_po_features(self, student_df: pd.DataFrame, 
                              co_po_matrix: pd.DataFrame) -> pd.DataFrame:
        """Extract CO-PO related features."""
        assessment_cols = [col for col in student_df.columns if col.startswith('A')]
        assessment_data = student_df[assessment_cols].fillna(0)
        
        # Simple CO-PO feature extraction (can be enhanced based on specific mapping)
        co_po_features = {}
        
        # Assume each assessment maps to a CO (simplified)
        if len(assessment_cols) <= len(co_po_matrix):
            for i, col in enumerate(assessment_cols):
                if i < len(co_po_matrix):
                    # Calculate weighted PO scores for this assessment
                    po_weights = co_po_matrix.iloc[i].values[1:]  # Skip CO column
                    weighted_scores = assessment_data[col].values * np.sum(po_weights)
                    co_po_features[f'co_{i+1}_po_weighted'] = weighted_scores
        
        return pd.DataFrame(co_po_features) if co_po_features else pd.DataFrame()
    
    def prepare_for_training(self, feature_matrix: pd.DataFrame, 
                           target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for machine learning training."""
        # Handle missing values
        feature_matrix_clean = feature_matrix.fillna(feature_matrix.median())
        
        # Separate features and target
        if target_column and target_column in feature_matrix_clean.columns:
            X = feature_matrix_clean.drop(columns=[target_column])
            y = feature_matrix_clean[target_column]
        else:
            X = feature_matrix_clean
            y = None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = list(X.columns)
        
        return X_scaled, y.values if y is not None else None
    
    def transform_new_data(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scalers."""
        # Handle missing values
        feature_matrix_clean = feature_matrix.fillna(feature_matrix.median())
        
        # Apply same scaling
        X_scaled = self.scaler.transform(feature_matrix_clean)
        
        return X_scaled