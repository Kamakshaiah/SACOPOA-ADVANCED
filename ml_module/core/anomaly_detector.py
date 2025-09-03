"""
Anomaly Detector for SACOPOA ML Module
======================================

Anomaly detection system for identifying unusual patterns
in student performance and assessment data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Anomaly detection system for student performance analysis.
    
    Capabilities:
    - Statistical anomaly detection
    - Machine learning-based anomaly detection
    - Performance pattern anomalies
    - Assessment irregularity detection
    """
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'one_class_svm', 'statistical')
            contamination: Expected proportion of anomalies in the data
        """
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.detector = None
        self.statistical_thresholds = {}
        self.is_fitted = False
        
        # Initialize detector based on method
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the anomaly detection model."""
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'one_class_svm':
            self.detector = OneClassSVM(nu=self.contamination, kernel='rbf')
        elif self.method == 'statistical':
            # Statistical method doesn't need a specific detector
            pass
        else:
            raise ValueError(f"Unsupported detection method: {self.method}")
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Fit the anomaly detection model.
        
        Args:
            X: Feature matrix (normal data for training)
            feature_names: Names of features
            
        Returns:
            Fitting statistics and thresholds
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method in ['isolation_forest', 'one_class_svm']:
            # Fit ML-based detector
            self.detector.fit(X_scaled)
            
            # Get anomaly scores for threshold setting
            anomaly_scores = self.detector.decision_function(X_scaled)
            
            fit_stats = {
                'method': self.method,
                'contamination': self.contamination,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'anomaly_score_mean': float(np.mean(anomaly_scores)),
                'anomaly_score_std': float(np.std(anomaly_scores))
            }
            
        elif self.method == 'statistical':
            # Fit statistical thresholds
            self._fit_statistical_thresholds(X_scaled, feature_names)
            
            fit_stats = {
                'method': self.method,
                'thresholds': self.statistical_thresholds,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
        
        self.is_fitted = True
        return fit_stats
    
    def _fit_statistical_thresholds(self, X: np.ndarray, feature_names: List[str] = None):
        """Fit statistical thresholds for anomaly detection."""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Calculate statistical thresholds (mean ± 2 or 3 standard deviations)
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            
            # IQR-based thresholds (more robust to outliers)
            q1, q3 = np.percentile(feature_data, [25, 75])
            iqr = q3 - q1
            
            self.statistical_thresholds[feature_name] = {
                'mean': mean,
                'std': std,
                'lower_bound_2std': mean - 2 * std,
                'upper_bound_2std': mean + 2 * std,
                'lower_bound_3std': mean - 3 * std,
                'upper_bound_3std': mean + 3 * std,
                'iqr_lower': q1 - 1.5 * iqr,
                'iqr_upper': q3 + 1.5 * iqr,
                'extreme_iqr_lower': q1 - 3 * iqr,
                'extreme_iqr_upper': q3 + 3 * iqr
            }
    
    def detect_anomalies(self, X: np.ndarray, 
                        feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Detect anomalies in the data.
        
        Args:
            X: Feature matrix to analyze
            feature_names: Names of features
            
        Returns:
            Anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before anomaly detection")
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        if self.method in ['isolation_forest', 'one_class_svm']:
            results = self._detect_ml_anomalies(X_scaled)
        elif self.method == 'statistical':
            results = self._detect_statistical_anomalies(X_scaled, feature_names)
        
        # Add detailed analysis
        results.update(self._analyze_anomalies(X, X_scaled, results['anomaly_mask']))
        
        return results
    
    def _detect_ml_anomalies(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using ML methods."""
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.detector.predict(X_scaled)
        anomaly_mask = predictions == -1
        
        # Get anomaly scores
        anomaly_scores = self.detector.decision_function(X_scaled)
        
        return {
            'anomaly_mask': anomaly_mask,
            'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
            'anomaly_scores': anomaly_scores,
            'n_anomalies': int(np.sum(anomaly_mask)),
            'anomaly_rate': float(np.mean(anomaly_mask))
        }
    
    def _detect_statistical_anomalies(self, X_scaled: np.ndarray, 
                                    feature_names: List[str] = None) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_scaled.shape[1])]
        
        anomaly_mask = np.zeros(len(X_scaled), dtype=bool)
        feature_anomalies = {}
        anomaly_scores = np.zeros(len(X_scaled))
        
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.statistical_thresholds:
                continue
                
            thresholds = self.statistical_thresholds[feature_name]
            feature_data = X_scaled[:, i]
            
            # Detect anomalies using IQR method (more robust)
            feature_anomaly_mask = (
                (feature_data < thresholds['iqr_lower']) |
                (feature_data > thresholds['iqr_upper'])
            )
            
            # Extreme anomalies
            extreme_anomaly_mask = (
                (feature_data < thresholds['extreme_iqr_lower']) |
                (feature_data > thresholds['extreme_iqr_upper'])
            )
            
            feature_anomalies[feature_name] = {
                'mild_anomalies': np.where(feature_anomaly_mask & ~extreme_anomaly_mask)[0].tolist(),
                'extreme_anomalies': np.where(extreme_anomaly_mask)[0].tolist(),
                'anomaly_count': int(np.sum(feature_anomaly_mask))
            }
            
            # Update overall anomaly mask
            anomaly_mask |= feature_anomaly_mask
            
            # Calculate anomaly scores (distance from normal range)
            for j, value in enumerate(feature_data):
                if value < thresholds['iqr_lower']:
                    score = (thresholds['iqr_lower'] - value) / (thresholds['mean'] - thresholds['iqr_lower'])
                elif value > thresholds['iqr_upper']:
                    score = (value - thresholds['iqr_upper']) / (thresholds['iqr_upper'] - thresholds['mean'])
                else:
                    score = 0
                
                anomaly_scores[j] = max(anomaly_scores[j], score)
        
        return {
            'anomaly_mask': anomaly_mask,
            'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
            'anomaly_scores': anomaly_scores,
            'n_anomalies': int(np.sum(anomaly_mask)),
            'anomaly_rate': float(np.mean(anomaly_mask)),
            'feature_anomalies': feature_anomalies
        }
    
    def _analyze_anomalies(self, X_original: np.ndarray, X_scaled: np.ndarray, 
                         anomaly_mask: np.ndarray) -> Dict[str, Any]:
        """Analyze detected anomalies in detail."""
        if not np.any(anomaly_mask):
            return {
                'anomaly_analysis': 'No anomalies detected',
                'severity_distribution': {},
                'anomaly_patterns': []
            }
        
        anomaly_data = X_original[anomaly_mask]
        normal_data = X_original[~anomaly_mask]
        
        # Severity analysis
        severity_scores = self._calculate_anomaly_severity(X_scaled, anomaly_mask)
        
        # Pattern analysis
        patterns = self._identify_anomaly_patterns(anomaly_data, normal_data)
        
        # Statistical comparison
        comparison = self._compare_anomalies_to_normal(anomaly_data, normal_data)
        
        return {
            'anomaly_analysis': {
                'severity_scores': severity_scores,
                'severity_distribution': self._categorize_severity(severity_scores),
                'statistical_comparison': comparison
            },
            'anomaly_patterns': patterns,
            'anomalous_students': np.where(anomaly_mask)[0].tolist()
        }
    
    def _calculate_anomaly_severity(self, X_scaled: np.ndarray, 
                                  anomaly_mask: np.ndarray) -> List[float]:
        """Calculate severity scores for anomalies."""
        if not hasattr(self.detector, 'decision_function'):
            # For statistical method, use distance from mean
            center = np.mean(X_scaled[~anomaly_mask], axis=0)
            distances = np.linalg.norm(X_scaled[anomaly_mask] - center, axis=1)
            # Normalize to 0-1 scale
            max_distance = np.max(distances) if len(distances) > 0 else 1
            return (distances / max_distance).tolist()
        else:
            # For ML methods, use decision function
            scores = self.detector.decision_function(X_scaled[anomaly_mask])
            # Convert to severity (lower decision function = higher severity)
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score != min_score:
                severity = (max_score - scores) / (max_score - min_score)
            else:
                severity = np.ones_like(scores)
            return severity.tolist()
    
    def _categorize_severity(self, severity_scores: List[float]) -> Dict[str, int]:
        """Categorize anomalies by severity."""
        if not severity_scores:
            return {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        scores = np.array(severity_scores)
        
        return {
            'low': int(np.sum(scores <= 0.25)),
            'medium': int(np.sum((scores > 0.25) & (scores <= 0.5))),
            'high': int(np.sum((scores > 0.5) & (scores <= 0.75))),
            'critical': int(np.sum(scores > 0.75))
        }
    
    def _identify_anomaly_patterns(self, anomaly_data: np.ndarray, 
                                 normal_data: np.ndarray) -> List[Dict[str, Any]]:
        """Identify patterns in anomalous data."""
        if len(anomaly_data) == 0:
            return []
        
        patterns = []
        
        # Feature-wise analysis
        for i in range(anomaly_data.shape[1]):
            anomaly_feature = anomaly_data[:, i]
            normal_feature = normal_data[:, i]
            
            # Statistical tests
            if len(normal_feature) > 0:
                # Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = stats.mannwhitneyu(
                        anomaly_feature, normal_feature, alternative='two-sided'
                    )
                    
                    if p_value < 0.05:  # Significant difference
                        pattern = {
                            'feature_index': i,
                            'pattern_type': 'significant_difference',
                            'anomaly_mean': float(np.mean(anomaly_feature)),
                            'normal_mean': float(np.mean(normal_feature)),
                            'p_value': float(p_value),
                            'effect_size': float(np.abs(np.mean(anomaly_feature) - np.mean(normal_feature)) / 
                                               np.std(normal_feature)) if np.std(normal_feature) > 0 else 0
                        }
                        patterns.append(pattern)
                except:
                    pass
        
        # Correlation analysis
        if len(anomaly_data) > 1 and anomaly_data.shape[1] > 1:
            try:
                anomaly_corr = np.corrcoef(anomaly_data.T)
                normal_corr = np.corrcoef(normal_data.T)
                
                # Find significantly different correlations
                corr_diff = np.abs(anomaly_corr - normal_corr)
                significant_pairs = np.where(corr_diff > 0.3)  # Threshold for significant difference
                
                for i, j in zip(significant_pairs[0], significant_pairs[1]):
                    if i < j:  # Avoid duplicates
                        patterns.append({
                            'pattern_type': 'correlation_anomaly',
                            'feature_pair': [int(i), int(j)],
                            'anomaly_correlation': float(anomaly_corr[i, j]),
                            'normal_correlation': float(normal_corr[i, j]),
                            'correlation_difference': float(corr_diff[i, j])
                        })
            except:
                pass
        
        return patterns
    
    def _compare_anomalies_to_normal(self, anomaly_data: np.ndarray, 
                                   normal_data: np.ndarray) -> Dict[str, Any]:
        """Compare anomalous and normal data statistically."""
        if len(anomaly_data) == 0 or len(normal_data) == 0:
            return {}
        
        comparison = {}
        
        for i in range(anomaly_data.shape[1]):
            anomaly_feature = anomaly_data[:, i]
            normal_feature = normal_data[:, i]
            
            comparison[f'feature_{i}'] = {
                'anomaly_mean': float(np.mean(anomaly_feature)),
                'normal_mean': float(np.mean(normal_feature)),
                'anomaly_std': float(np.std(anomaly_feature)),
                'normal_std': float(np.std(normal_feature)),
                'mean_difference': float(np.mean(anomaly_feature) - np.mean(normal_feature)),
                'anomaly_range': [float(np.min(anomaly_feature)), float(np.max(anomaly_feature))],
                'normal_range': [float(np.min(normal_feature)), float(np.max(normal_feature))]
            }
        
        return comparison
    
    def detect_performance_anomalies(self, student_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Specialized method for detecting performance anomalies.
        
        Args:
            student_data: DataFrame with student performance data
            
        Returns:
            Performance-specific anomaly analysis
        """
        # Extract assessment columns
        assessment_cols = [col for col in student_data.columns if col.startswith('A')]
        
        if not assessment_cols:
            return {'error': 'No assessment columns found'}
        
        assessment_data = student_data[assessment_cols].values
        
        # Detect various types of performance anomalies
        anomalies = {}
        
        # 1. Sudden drops in performance
        anomalies['sudden_drops'] = self._detect_performance_drops(assessment_data)
        
        # 2. Unusual consistency patterns
        anomalies['consistency_anomalies'] = self._detect_consistency_anomalies(assessment_data)
        
        # 3. Score progression anomalies
        anomalies['progression_anomalies'] = self._detect_progression_anomalies(assessment_data)
        
        # 4. Outlier scores
        anomalies['score_outliers'] = self._detect_score_outliers(assessment_data)
        
        return {
            'performance_anomalies': anomalies,
            'summary': self._summarize_performance_anomalies(anomalies),
            'student_flags': self._flag_anomalous_students(anomalies, len(student_data))
        }
    
    def _detect_performance_drops(self, assessment_data: np.ndarray) -> List[Dict]:
        """Detect sudden drops in student performance."""
        drops = []
        
        for i, student_scores in enumerate(assessment_data):
            for j in range(1, len(student_scores)):
                drop = student_scores[j-1] - student_scores[j]
                if drop > 20:  # Threshold for significant drop
                    drops.append({
                        'student_index': i,
                        'assessment_from': j-1,
                        'assessment_to': j,
                        'drop_magnitude': float(drop),
                        'severity': 'high' if drop > 30 else 'medium'
                    })
        
        return drops
    
    def _detect_consistency_anomalies(self, assessment_data: np.ndarray) -> List[Dict]:
        """Detect unusual consistency patterns."""
        anomalies = []
        
        for i, student_scores in enumerate(assessment_data):
            cv = np.std(student_scores) / np.mean(student_scores) if np.mean(student_scores) > 0 else 0
            
            # Very high or very low consistency can be anomalous
            if cv > 0.5 or cv < 0.05:
                anomalies.append({
                    'student_index': i,
                    'consistency_score': float(cv),
                    'type': 'highly_variable' if cv > 0.5 else 'unusually_consistent',
                    'scores': student_scores.tolist()
                })
        
        return anomalies
    
    def _detect_progression_anomalies(self, assessment_data: np.ndarray) -> List[Dict]:
        """Detect unusual progression patterns."""
        anomalies = []
        
        for i, student_scores in enumerate(assessment_data):
            if len(student_scores) < 3:
                continue
                
            # Fit linear trend
            x = np.arange(len(student_scores))
            slope, intercept = np.polyfit(x, student_scores, 1)
            
            # Detect unusual patterns
            if slope > 10:  # Very strong improvement
                anomalies.append({
                    'student_index': i,
                    'pattern_type': 'rapid_improvement',
                    'slope': float(slope),
                    'total_improvement': float(student_scores[-1] - student_scores[0])
                })
            elif slope < -10:  # Very strong decline
                anomalies.append({
                    'student_index': i,
                    'pattern_type': 'rapid_decline',
                    'slope': float(slope),
                    'total_decline': float(student_scores[0] - student_scores[-1])
                })
        
        return anomalies
    
    def _detect_score_outliers(self, assessment_data: np.ndarray) -> List[Dict]:
        """Detect outlier scores in assessments."""
        outliers = []
        
        # For each assessment, find outliers
        for j in range(assessment_data.shape[1]):
            assessment_scores = assessment_data[:, j]
            
            # IQR method
            q1, q3 = np.percentile(assessment_scores, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = np.where(
                (assessment_scores < lower_bound) | (assessment_scores > upper_bound)
            )[0]
            
            for idx in outlier_indices:
                outliers.append({
                    'student_index': int(idx),
                    'assessment_index': j,
                    'score': float(assessment_scores[idx]),
                    'outlier_type': 'low' if assessment_scores[idx] < lower_bound else 'high',
                    'deviation_from_median': float(assessment_scores[idx] - np.median(assessment_scores))
                })
        
        return outliers
    
    def _summarize_performance_anomalies(self, anomalies: Dict) -> Dict[str, Any]:
        """Summarize performance anomaly findings."""
        return {
            'total_sudden_drops': len(anomalies.get('sudden_drops', [])),
            'total_consistency_anomalies': len(anomalies.get('consistency_anomalies', [])),
            'total_progression_anomalies': len(anomalies.get('progression_anomalies', [])),
            'total_score_outliers': len(anomalies.get('score_outliers', [])),
            'severity_breakdown': {
                'high': len([d for d in anomalies.get('sudden_drops', []) if d.get('severity') == 'high']),
                'medium': len([d for d in anomalies.get('sudden_drops', []) if d.get('severity') == 'medium'])
            }
        }
    
    def _flag_anomalous_students(self, anomalies: Dict, total_students: int) -> List[int]:
        """Flag students with multiple anomalies."""
        student_anomaly_counts = {}
        
        # Count anomalies per student
        for anomaly_type, anomaly_list in anomalies.items():
            for anomaly in anomaly_list:
                student_idx = anomaly.get('student_index')
                if student_idx is not None:
                    student_anomaly_counts[student_idx] = student_anomaly_counts.get(student_idx, 0) + 1
        
        # Flag students with multiple anomalies
        flagged_students = [
            student_idx for student_idx, count in student_anomaly_counts.items()
            if count >= 2  # Threshold for flagging
        ]
        
        return flagged_students
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection setup."""
        return {
            'method': self.method,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted,
            'statistical_thresholds_available': bool(self.statistical_thresholds) if self.method == 'statistical' else None
        }