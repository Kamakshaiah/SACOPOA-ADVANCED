"""
Risk Analyzer for SACOPOA ML Module
===================================

Early warning system and risk assessment for identifying
students at risk of poor performance or failure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class RiskAnalyzer:
    """
    Risk assessment and early warning system for student performance.
    
    Capabilities:
    - Identify at-risk students early in the course
    - Multi-level risk categorization
    - Intervention recommendation timing
    - Risk factor analysis and explanation
    """
    
    def __init__(self, risk_threshold: float = 0.4):
        """
        Initialize the risk analyzer.
        
        Args:
            risk_threshold: Performance threshold below which students are considered at risk
        """
        self.risk_threshold = risk_threshold
        self.models = {
            'binary_risk': RandomForestClassifier(n_estimators=100, random_state=42),
            'multi_level_risk': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.feature_importance = {}
        self.risk_categories = {
            0: 'Low Risk',
            1: 'Medium Risk', 
            2: 'High Risk',
            3: 'Critical Risk'
        }
        self.is_trained = False
        
    def prepare_risk_labels(self, performance_scores: np.ndarray, 
                          grade_categories: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Prepare risk labels from performance data.
        
        Args:
            performance_scores: Student performance percentages
            grade_categories: Optional grade category data
            
        Returns:
            Dictionary with binary and multi-level risk labels
        """
        # Binary risk (at-risk vs not at-risk)
        binary_risk = (performance_scores < (self.risk_threshold * 100)).astype(int)
        
        # Multi-level risk categorization
        multi_level_risk = np.zeros(len(performance_scores), dtype=int)
        multi_level_risk[performance_scores < 25] = 3  # Critical Risk
        multi_level_risk[(performance_scores >= 25) & (performance_scores < 40)] = 2  # High Risk
        multi_level_risk[(performance_scores >= 40) & (performance_scores < 60)] = 1  # Medium Risk
        multi_level_risk[performance_scores >= 60] = 0  # Low Risk
        
        return {
            'binary_risk': binary_risk,
            'multi_level_risk': multi_level_risk
        }
    
    def train(self, X: np.ndarray, performance_scores: np.ndarray,
              feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train risk assessment models.
        
        Args:
            X: Feature matrix
            performance_scores: Performance percentages
            feature_names: Names of features
            
        Returns:
            Training metrics and model performance
        """
        # Prepare risk labels
        risk_labels = self.prepare_risk_labels(performance_scores)
        
        # Split data
        X_train, X_test, y_binary_train, y_binary_test = train_test_split(
            X, risk_labels['binary_risk'], test_size=0.2, random_state=42, stratify=risk_labels['binary_risk']
        )
        
        _, _, y_multi_train, y_multi_test = train_test_split(
            X, risk_labels['multi_level_risk'], test_size=0.2, random_state=42, stratify=risk_labels['multi_level_risk']
        )
        
        # Train binary risk model
        self.models['binary_risk'].fit(X_train, y_binary_train)
        binary_pred = self.models['binary_risk'].predict(X_test)
        binary_prob = self.models['binary_risk'].predict_proba(X_test)[:, 1]
        
        # Train multi-level risk model
        self.models['multi_level_risk'].fit(X_train, y_multi_train)
        multi_pred = self.models['multi_level_risk'].predict(X_test)
        
        # Calculate metrics
        metrics = {
            'binary_risk': {
                'accuracy': np.mean(binary_pred == y_binary_test),
                'auc_score': roc_auc_score(y_binary_test, binary_prob),
                'classification_report': classification_report(y_binary_test, binary_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_binary_test, binary_pred).tolist()
            },
            'multi_level_risk': {
                'accuracy': np.mean(multi_pred == y_multi_test),
                'classification_report': classification_report(y_multi_test, multi_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_multi_test, multi_pred).tolist()
            }
        }
        
        # Store feature importance
        self.feature_importance['binary_risk'] = self.models['binary_risk'].feature_importances_
        self.feature_importance['multi_level_risk'] = self.models['multi_level_risk'].feature_importances_
        
        self.is_trained = True
        return metrics
    
    def assess_risk(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Assess risk for students.
        
        Args:
            X: Feature matrix
            
        Returns:
            Comprehensive risk assessment
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before risk assessment")
        
        # Binary risk predictions
        binary_risk_prob = self.models['binary_risk'].predict_proba(X)[:, 1]
        binary_risk_pred = self.models['binary_risk'].predict(X)
        
        # Multi-level risk predictions
        multi_risk_pred = self.models['multi_level_risk'].predict(X)
        multi_risk_prob = self.models['multi_level_risk'].predict_proba(X)
        
        # Risk factor analysis
        risk_factors = self._analyze_risk_factors(X, binary_risk_prob)
        
        # Intervention urgency
        intervention_urgency = self._calculate_intervention_urgency(
            binary_risk_prob, multi_risk_pred
        )
        
        return {
            'binary_risk_probability': binary_risk_prob,
            'binary_risk_prediction': binary_risk_pred,
            'multi_level_risk_prediction': multi_risk_pred,
            'multi_level_risk_probability': multi_risk_prob,
            'risk_categories': [self.risk_categories[pred] for pred in multi_risk_pred],
            'risk_factors': risk_factors,
            'intervention_urgency': intervention_urgency,
            'high_risk_students': np.where(multi_risk_pred >= 2)[0].tolist(),
            'critical_risk_students': np.where(multi_risk_pred == 3)[0].tolist()
        }
    
    def early_risk_assessment(self, X_partial: np.ndarray, 
                            assessment_index: int) -> Dict[str, Any]:
        """
        Perform early risk assessment with limited data.
        
        Args:
            X_partial: Partial feature matrix
            assessment_index: Index of last available assessment
            
        Returns:
            Early risk assessment with confidence measures
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before risk assessment")
        
        # Fill missing assessments for early prediction
        X_filled = self._fill_missing_for_risk_assessment(X_partial, assessment_index)
        
        # Get risk assessment
        risk_assessment = self.assess_risk(X_filled)
        
        # Adjust confidence based on available data
        early_confidence = self._calculate_early_risk_confidence(assessment_index)
        risk_assessment['early_assessment_confidence'] = early_confidence
        
        # Flag students for immediate attention
        immediate_attention = self._identify_immediate_attention_students(
            risk_assessment, early_confidence
        )
        risk_assessment['immediate_attention_students'] = immediate_attention
        
        return risk_assessment
    
    def _analyze_risk_factors(self, X: np.ndarray, risk_probabilities: np.ndarray) -> List[Dict]:
        """Analyze key risk factors for each student."""
        feature_importance = self.feature_importance['binary_risk']
        
        risk_factors = []
        for i, prob in enumerate(risk_probabilities):
            # Get student's feature values
            student_features = X[i]
            
            # Calculate risk contribution of each feature
            risk_contributions = feature_importance * np.abs(student_features)
            
            # Get top risk factors
            top_factors_idx = np.argsort(risk_contributions)[-5:][::-1]
            
            student_risk_factors = {
                'student_index': i,
                'overall_risk_probability': prob,
                'top_risk_factors': [
                    {
                        'factor_index': int(idx),
                        'importance': float(feature_importance[idx]),
                        'value': float(student_features[idx]),
                        'contribution': float(risk_contributions[idx])
                    }
                    for idx in top_factors_idx
                ]
            }
            
            risk_factors.append(student_risk_factors)
        
        return risk_factors
    
    def _calculate_intervention_urgency(self, binary_risk_prob: np.ndarray, 
                                      multi_risk_pred: np.ndarray) -> np.ndarray:
        """Calculate intervention urgency scores."""
        urgency = np.zeros(len(binary_risk_prob))
        
        # Base urgency on risk probability and category
        urgency = binary_risk_prob * 0.6  # Base from probability
        
        # Add urgency based on risk category
        urgency[multi_risk_pred == 1] += 0.2  # Medium risk
        urgency[multi_risk_pred == 2] += 0.4  # High risk  
        urgency[multi_risk_pred == 3] += 0.6  # Critical risk
        
        # Cap at 1.0
        urgency = np.minimum(urgency, 1.0)
        
        return urgency
    
    def _fill_missing_for_risk_assessment(self, X_partial: np.ndarray, 
                                        assessment_index: int) -> np.ndarray:
        """Fill missing data for early risk assessment."""
        # Simple approach: use median of available assessments
        # Could be enhanced with more sophisticated imputation
        
        X_filled = X_partial.copy()
        
        for i in range(X_partial.shape[0]):
            # Find assessment columns (simplified approach)
            # This assumes assessment columns are at the end
            assessment_values = X_partial[i, -assessment_index-1:]
            if len(assessment_values) > 0:
                median_score = np.median(assessment_values[assessment_values > 0])
                if not np.isnan(median_score):
                    # Fill remaining columns with median
                    remaining_cols = X_filled.shape[1] - assessment_index - 1
                    X_filled[i, -remaining_cols:] = median_score
        
        return X_filled
    
    def _calculate_early_risk_confidence(self, assessment_index: int) -> float:
        """Calculate confidence for early risk assessment."""
        # Confidence increases with more available assessments
        max_assessments = 10  # Assuming maximum 10 assessments
        base_confidence = 0.3  # Minimum confidence
        max_confidence = 0.9   # Maximum confidence
        
        confidence = base_confidence + (max_confidence - base_confidence) * (
            assessment_index / max_assessments
        )
        
        return min(confidence, max_confidence)
    
    def _identify_immediate_attention_students(self, risk_assessment: Dict, 
                                             confidence: float) -> List[int]:
        """Identify students needing immediate attention."""
        immediate_attention = []
        
        binary_risk_prob = risk_assessment['binary_risk_probability']
        multi_risk_pred = risk_assessment['multi_level_risk_prediction']
        
        for i, (prob, category) in enumerate(zip(binary_risk_prob, multi_risk_pred)):
            # Flag for immediate attention if:
            # 1. High risk probability even with low confidence
            # 2. Critical risk category
            # 3. High risk category with decent confidence
            
            if (prob > 0.8) or (category == 3) or (category == 2 and confidence > 0.5):
                immediate_attention.append(i)
        
        return immediate_attention
    
    def generate_risk_report(self, risk_assessment: Dict, 
                           student_ids: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        if student_ids is None:
            student_ids = [f"Student_{i}" for i in range(len(risk_assessment['binary_risk_probability']))]
        
        # Summary statistics
        risk_summary = {
            'total_students': len(risk_assessment['binary_risk_probability']),
            'high_risk_count': len(risk_assessment['high_risk_students']),
            'critical_risk_count': len(risk_assessment['critical_risk_students']),
            'immediate_attention_count': len(risk_assessment.get('immediate_attention_students', [])),
            'average_risk_probability': float(np.mean(risk_assessment['binary_risk_probability'])),
            'risk_distribution': {
                category: int(np.sum(risk_assessment['multi_level_risk_prediction'] == level))
                for level, category in self.risk_categories.items()
            }
        }
        
        # Individual student details
        student_details = []
        for i, student_id in enumerate(student_ids):
            detail = {
                'student_id': student_id,
                'risk_probability': float(risk_assessment['binary_risk_probability'][i]),
                'risk_category': risk_assessment['risk_categories'][i],
                'intervention_urgency': float(risk_assessment['intervention_urgency'][i]),
                'is_high_risk': i in risk_assessment['high_risk_students'],
                'is_critical_risk': i in risk_assessment['critical_risk_students'],
                'needs_immediate_attention': i in risk_assessment.get('immediate_attention_students', [])
            }
            student_details.append(detail)
        
        return {
            'summary': risk_summary,
            'student_details': student_details,
            'assessment_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_feature_importance(self, model_type: str = 'binary_risk') -> Dict[int, float]:
        """Get feature importance for risk factors."""
        if model_type not in self.feature_importance:
            return {}
        
        importance = self.feature_importance[model_type]
        return dict(enumerate(importance))