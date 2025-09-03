"""
Performance Clustering for SACOPOA ML Module
============================================

Clustering algorithms for identifying student groups,
performance patterns, and course difficulty analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

warnings.filterwarnings('ignore')


class PerformanceClustering:
    """
    Clustering analysis for student performance and course patterns.
    
    Capabilities:
    - Student performance clustering
    - Learning pattern identification
    - Course difficulty analysis
    - Cohort comparison and segmentation
    """
    
    def __init__(self, clustering_method: str = 'kmeans'):
        """
        Initialize the clustering analyzer.
        
        Args:
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        """
        self.clustering_method = clustering_method
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.cluster_model = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.clustering_metrics = {}
        self.is_fitted = False
        
        # Initialize clustering model
        self._initialize_clustering_model()
    
    def _initialize_clustering_model(self):
        """Initialize clustering model based on specified method."""
        if self.clustering_method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        elif self.clustering_method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
        elif self.clustering_method == 'hierarchical':
            self.cluster_model = AgglomerativeClustering(n_clusters=4)
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
    
    def fit_student_clusters(self, X: np.ndarray, 
                           performance_scores: np.ndarray = None) -> Dict[str, Any]:
        """
        Fit clustering model on student data.
        
        Args:
            X: Feature matrix
            performance_scores: Optional performance scores for validation
            
        Returns:
            Clustering results and metrics
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction if needed
        if X_scaled.shape[1] > 10:
            X_reduced = self.pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Fit clustering model
        if self.clustering_method == 'kmeans':
            # Find optimal number of clusters
            optimal_k = self._find_optimal_clusters(X_reduced)
            self.cluster_model.n_clusters = optimal_k
            
        self.cluster_labels = self.cluster_model.fit_predict(X_reduced)
        
        # Get cluster centers for kmeans
        if hasattr(self.cluster_model, 'cluster_centers_'):
            self.cluster_centers = self.cluster_model.cluster_centers_
        
        # Calculate clustering metrics
        if len(np.unique(self.cluster_labels)) > 1:
            silhouette = silhouette_score(X_reduced, self.cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_reduced, self.cluster_labels)
            
            self.clustering_metrics = {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'n_clusters': len(np.unique(self.cluster_labels)),
                'cluster_sizes': [int(np.sum(self.cluster_labels == i)) 
                                for i in np.unique(self.cluster_labels)]
            }
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(X, performance_scores)
        
        self.is_fitted = True
        
        return {
            'cluster_labels': self.cluster_labels,
            'clustering_metrics': self.clustering_metrics,
            'cluster_analysis': cluster_analysis,
            'reduced_features': X_reduced
        }
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method and silhouette score."""
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        # Find elbow point (simplified)
        if len(silhouette_scores) > 0:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3  # Default
        
        return min(optimal_k, len(X) // 4)  # Ensure reasonable cluster size
    
    def _analyze_clusters(self, X: np.ndarray, 
                        performance_scores: np.ndarray = None) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        if self.cluster_labels is None:
            return {}
        
        cluster_analysis = {}
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = X[cluster_mask]
            
            # Basic statistics
            analysis = {
                'cluster_id': int(cluster_id),
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(X) * 100),
                'feature_means': cluster_data.mean(axis=0).tolist(),
                'feature_stds': cluster_data.std(axis=0).tolist()
            }
            
            # Performance analysis if available
            if performance_scores is not None:
                cluster_performance = performance_scores[cluster_mask]
                analysis.update({
                    'avg_performance': float(np.mean(cluster_performance)),
                    'performance_std': float(np.std(cluster_performance)),
                    'min_performance': float(np.min(cluster_performance)),
                    'max_performance': float(np.max(cluster_performance)),
                    'pass_rate': float(np.mean(cluster_performance >= 40) * 100),
                    'excellence_rate': float(np.mean(cluster_performance >= 85) * 100)
                })
            
            cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def predict_cluster(self, X_new: np.ndarray) -> np.ndarray:
        """Predict cluster for new data points."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Transform new data
        X_scaled = self.scaler.transform(X_new)
        
        if hasattr(self.pca, 'components_'):
            X_reduced = self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Predict clusters
        if hasattr(self.cluster_model, 'predict'):
            return self.cluster_model.predict(X_reduced)
        else:
            # For DBSCAN, use distance-based assignment
            return self._assign_to_nearest_cluster(X_reduced)
    
    def _assign_to_nearest_cluster(self, X: np.ndarray) -> np.ndarray:
        """Assign points to nearest cluster for algorithms without predict method."""
        if self.cluster_centers is None:
            # Calculate cluster centers from training data
            unique_clusters = np.unique(self.cluster_labels)
            centers = []
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Exclude noise points in DBSCAN
                    cluster_mask = self.cluster_labels == cluster_id
                    center = np.mean(X[cluster_mask], axis=0)
                    centers.append(center)
            self.cluster_centers = np.array(centers)
        
        # Assign to nearest cluster
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def identify_learning_patterns(self, X: np.ndarray, 
                                 assessment_scores: np.ndarray) -> Dict[str, Any]:
        """Identify learning patterns and trajectories."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before pattern analysis")
        
        patterns = {}
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_assessments = assessment_scores[cluster_mask]
            
            # Calculate learning trajectory
            trajectory = self._calculate_learning_trajectory(cluster_assessments)
            
            # Identify pattern type
            pattern_type = self._classify_learning_pattern(trajectory)
            
            patterns[f'cluster_{cluster_id}'] = {
                'pattern_type': pattern_type,
                'trajectory': trajectory,
                'consistency': self._calculate_pattern_consistency(cluster_assessments),
                'improvement_rate': trajectory['slope'] if trajectory else 0
            }
        
        return patterns
    
    def _calculate_learning_trajectory(self, assessment_scores: np.ndarray) -> Dict[str, float]:
        """Calculate learning trajectory for a group of students."""
        if assessment_scores.size == 0:
            return {}
        
        # Average scores across assessments
        avg_scores = np.mean(assessment_scores, axis=0)
        
        # Fit linear trend
        x = np.arange(len(avg_scores))
        if len(avg_scores) > 1:
            slope, intercept = np.polyfit(x, avg_scores, 1)
        else:
            slope, intercept = 0, avg_scores[0] if len(avg_scores) > 0 else 0
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'start_performance': float(avg_scores[0]) if len(avg_scores) > 0 else 0,
            'end_performance': float(avg_scores[-1]) if len(avg_scores) > 0 else 0,
            'total_improvement': float(avg_scores[-1] - avg_scores[0]) if len(avg_scores) > 0 else 0
        }
    
    def _classify_learning_pattern(self, trajectory: Dict[str, float]) -> str:
        """Classify learning pattern based on trajectory."""
        if not trajectory:
            return 'Unknown'
        
        slope = trajectory['slope']
        total_improvement = trajectory['total_improvement']
        
        if slope > 1.0:
            return 'Strong Improver'
        elif slope > 0.2:
            return 'Gradual Improver'
        elif abs(slope) <= 0.2:
            return 'Consistent Performer'
        elif slope > -1.0:
            return 'Gradual Decliner'
        else:
            return 'Strong Decliner'
    
    def _calculate_pattern_consistency(self, assessment_scores: np.ndarray) -> float:
        """Calculate consistency of learning pattern."""
        if assessment_scores.size == 0:
            return 0.0
        
        # Calculate coefficient of variation for each student
        cv_scores = []
        for scores in assessment_scores:
            if np.mean(scores) > 0:
                cv = np.std(scores) / np.mean(scores)
                cv_scores.append(cv)
        
        # Return average consistency (lower CV = higher consistency)
        if cv_scores:
            avg_cv = np.mean(cv_scores)
            consistency = max(0, 1 - avg_cv)  # Convert to consistency score
            return float(consistency)
        else:
            return 0.0
    
    def analyze_course_difficulty(self, course_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze difficulty patterns across multiple courses."""
        difficulty_analysis = {}
        
        for course_name, data in course_data.items():
            # Fit clustering for this course
            clustering_result = self.fit_student_clusters(data['features'], data['performance'])
            
            # Calculate difficulty metrics
            avg_performance = np.mean(data['performance'])
            std_performance = np.std(data['performance'])
            pass_rate = np.mean(data['performance'] >= 40) * 100
            
            # Classify difficulty
            if avg_performance >= 75 and std_performance <= 15:
                difficulty_level = 'Easy'
            elif avg_performance >= 60 and std_performance <= 20:
                difficulty_level = 'Moderate'
            elif avg_performance >= 45:
                difficulty_level = 'Challenging'
            else:
                difficulty_level = 'Very Difficult'
            
            difficulty_analysis[course_name] = {
                'difficulty_level': difficulty_level,
                'avg_performance': float(avg_performance),
                'performance_std': float(std_performance),
                'pass_rate': float(pass_rate),
                'cluster_distribution': clustering_result['clustering_metrics']['cluster_sizes'],
                'learning_patterns': self.identify_learning_patterns(
                    data['features'], data.get('assessment_scores', np.array([]))
                )
            }
        
        return difficulty_analysis
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results."""
        if not self.is_fitted:
            return {"status": "Model not fitted"}
        
        return {
            "clustering_method": self.clustering_method,
            "clustering_metrics": self.clustering_metrics,
            "cluster_labels": self.cluster_labels.tolist() if self.cluster_labels is not None else None,
            "is_fitted": self.is_fitted
        }