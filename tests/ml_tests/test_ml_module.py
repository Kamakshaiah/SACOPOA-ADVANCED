"""
Test cases for SACOPOA ML Module
===============================

Comprehensive test suite for all ML components.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import ml_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ml_module.sacopoa_ml import SACOPOAMLAnalyzer
from ml_module.core.predictor import PerformancePredictor
from ml_module.core.risk_analyzer import RiskAnalyzer
from ml_module.core.clustering import PerformanceClustering
from ml_module.core.recommender import InterventionRecommender
from ml_module.core.anomaly_detector import AnomalyDetector
from ml_module.utils.data_processor import DataProcessor
from ml_module.utils.visualization import MLVisualizer


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def setup_method(self):
        """Set up test data."""
        self.processor = DataProcessor()
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample student data for testing."""
        return pd.DataFrame({
            'Roll Number': ['21CSE001', '21CSE002', '21CSE003', '21CSE004', '21CSE005'],
            'Name': ['Student1', 'Student2', 'Student3', 'Student4', 'Student5'],
            'A1': [85, 78, 92, 65, 45],
            'A2': [80, 75, 88, 70, 50],
            'A3': [88, 80, 90, 68, 48],
            'A4': [85, 82, 91, 72, 52],
            'A5': [87, 79, 89, 69, 47]
        })
    
    def test_process_student_data(self):
        """Test student data processing."""
        result = self.processor.process_student_data(self.sample_data)
        
        assert 'features' in result
        assert 'performance_metrics' in result
        assert 'raw_data' in result
        assert 'feature_names' in result
        assert 'student_ids' in result
        
        # Check feature matrix dimensions
        assert result['features'].shape[0] == len(self.sample_data)
        assert result['features'].shape[1] > 0
        
        # Check performance metrics
        metrics = result['performance_metrics']
        assert 'percentages' in metrics
        assert 'pass_fail' in metrics
        assert 'grade_categories' in metrics
        assert len(metrics['percentages']) == len(self.sample_data)
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        features = self.processor._extract_features(self.sample_data)
        
        assert 'total_score' in features
        assert 'average_score' in features
        assert 'max_score' in features
        assert 'min_score' in features
        assert 'score_std' in features
        
        # Check array lengths
        assert len(features['total_score']) == len(self.sample_data)
    
    def test_data_validation(self):
        """Test data validation."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'A1': [85, 78, 'invalid', 65, 45],
            'A2': [80, 75, 88, 70, 50]
        })
        
        with pytest.raises(ValueError):
            self.processor._validate_student_data(invalid_data)
    
    def test_prepare_for_training(self):
        """Test data preparation for ML training."""
        processed_data = self.processor.process_student_data(self.sample_data)
        X, y = self.processor.prepare_for_training(
            processed_data['features'], target_column='percentage'
        )
        
        assert X.shape[0] == len(self.sample_data)
        assert y is not None
        assert len(y) == len(self.sample_data)


class TestPerformancePredictor:
    """Test cases for PerformancePredictor."""
    
    def setup_method(self):
        """Set up test data."""
        self.predictor = PerformancePredictor()
        self.X, self.y = self._create_test_data()
    
    def _create_test_data(self):
        """Create test data for predictor."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.uniform(30, 95, n_samples)  # Performance scores
        
        return X, y
    
    def test_initialization(self):
        """Test predictor initialization."""
        assert self.predictor.model_type == 'random_forest'
        assert not self.predictor.is_trained
        assert 'performance' in self.predictor.models
    
    def test_training(self):
        """Test model training."""
        grade_categories = np.random.randint(0, 5, len(self.y))
        
        metrics = self.predictor.train(self.X, self.y, grade_categories)
        
        assert self.predictor.is_trained
        assert 'performance_mse' in metrics
        assert 'performance_r2' in metrics
        assert 'cv_mean' in metrics
        assert metrics['performance_r2'] >= 0  # R² can be negative for very poor models
    
    def test_prediction(self):
        """Test performance prediction."""
        grade_categories = np.random.randint(0, 5, len(self.y))
        self.predictor.train(self.X, self.y, grade_categories)
        
        predictions = self.predictor.predict_performance(self.X)
        
        assert 'performance_scores' in predictions
        assert 'pass_probability' in predictions
        assert len(predictions['performance_scores']) == len(self.X)
        assert len(predictions['pass_probability']) == len(self.X)
    
    def test_early_prediction(self):
        """Test early performance prediction."""
        grade_categories = np.random.randint(0, 5, len(self.y))
        self.predictor.train(self.X, self.y, grade_categories)
        
        X_partial = self.X[:, :5]  # Only first 5 features
        early_pred = self.predictor.predict_early_performance(X_partial, 2)
        
        assert 'performance_scores' in early_pred
        assert 'early_prediction_confidence' in early_pred


class TestRiskAnalyzer:
    """Test cases for RiskAnalyzer."""
    
    def setup_method(self):
        """Set up test data."""
        self.risk_analyzer = RiskAnalyzer()
        self.X, self.performance_scores = self._create_test_data()
    
    def _create_test_data(self):
        """Create test data for risk analysis."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        performance_scores = np.random.uniform(20, 95, n_samples)
        
        return X, performance_scores
    
    def test_initialization(self):
        """Test risk analyzer initialization."""
        assert self.risk_analyzer.risk_threshold == 0.4
        assert not self.risk_analyzer.is_trained
    
    def test_risk_label_preparation(self):
        """Test risk label preparation."""
        labels = self.risk_analyzer.prepare_risk_labels(self.performance_scores)
        
        assert 'binary_risk' in labels
        assert 'multi_level_risk' in labels
        assert len(labels['binary_risk']) == len(self.performance_scores)
        assert len(labels['multi_level_risk']) == len(self.performance_scores)
    
    def test_training(self):
        """Test risk analyzer training."""
        metrics = self.risk_analyzer.train(self.X, self.performance_scores)
        
        assert self.risk_analyzer.is_trained
        assert 'binary_risk' in metrics
        assert 'multi_level_risk' in metrics
        assert 'accuracy' in metrics['binary_risk']
    
    def test_risk_assessment(self):
        """Test risk assessment."""
        self.risk_analyzer.train(self.X, self.performance_scores)
        
        assessment = self.risk_analyzer.assess_risk(self.X)
        
        assert 'binary_risk_probability' in assessment
        assert 'multi_level_risk_prediction' in assessment
        assert 'risk_categories' in assessment
        assert 'high_risk_students' in assessment
        assert len(assessment['binary_risk_probability']) == len(self.X)


class TestPerformanceClustering:
    """Test cases for PerformanceClustering."""
    
    def setup_method(self):
        """Set up test data."""
        self.clustering = PerformanceClustering()
        self.X, self.performance_scores = self._create_test_data()
    
    def _create_test_data(self):
        """Create test data for clustering."""
        np.random.seed(42)
        n_samples = 50
        n_features = 8
        
        X = np.random.randn(n_samples, n_features)
        performance_scores = np.random.uniform(30, 95, n_samples)
        
        return X, performance_scores
    
    def test_initialization(self):
        """Test clustering initialization."""
        assert self.clustering.clustering_method == 'kmeans'
        assert not self.clustering.is_fitted
    
    def test_clustering_fit(self):
        """Test clustering model fitting."""
        results = self.clustering.fit_student_clusters(self.X, self.performance_scores)
        
        assert self.clustering.is_fitted
        assert 'cluster_labels' in results
        assert 'clustering_metrics' in results
        assert 'cluster_analysis' in results
        assert len(results['cluster_labels']) == len(self.X)
    
    def test_cluster_prediction(self):
        """Test cluster prediction for new data."""
        self.clustering.fit_student_clusters(self.X, self.performance_scores)
        
        new_data = np.random.randn(5, self.X.shape[1])
        predictions = self.clustering.predict_cluster(new_data)
        
        assert len(predictions) == 5
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)


class TestInterventionRecommender:
    """Test cases for InterventionRecommender."""
    
    def setup_method(self):
        """Set up test data."""
        self.recommender = InterventionRecommender()
        self.X, self.risk_levels, self.performance_scores = self._create_test_data()
    
    def _create_test_data(self):
        """Create test data for intervention recommendation."""
        np.random.seed(42)
        n_samples = 30
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        risk_levels = np.random.randint(0, 4, n_samples)
        performance_scores = np.random.uniform(25, 90, n_samples)
        
        return X, risk_levels, performance_scores
    
    def test_initialization(self):
        """Test recommender initialization."""
        assert len(self.recommender.intervention_types) > 0
        assert 'tutoring' in self.recommender.intervention_types
        assert not self.recommender.is_trained
    
    def test_recommendation_generation(self):
        """Test intervention recommendation generation."""
        recommendations = self.recommender.recommend_interventions(
            self.X, self.risk_levels, self.performance_scores
        )
        
        assert len(recommendations) == len(self.X)
        
        for rec in recommendations:
            assert 'student_index' in rec
            assert 'risk_level' in rec
            assert 'recommendations' in rec
            assert 'priority_interventions' in rec
    
    def test_base_recommendations(self):
        """Test base recommendation logic."""
        # Test different risk levels
        low_risk_recs = self.recommender._get_base_recommendations(0)
        critical_risk_recs = self.recommender._get_base_recommendations(3)
        
        assert len(low_risk_recs) > 0
        assert len(critical_risk_recs) > len(low_risk_recs)
        
        # Critical risk should have more urgent interventions
        critical_urgencies = [rec['urgency'] for rec in critical_risk_recs]
        assert 'critical' in critical_urgencies or 'high' in critical_urgencies


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""
    
    def setup_method(self):
        """Set up test data."""
        self.detector = AnomalyDetector()
        self.X = self._create_test_data()
    
    def _create_test_data(self):
        """Create test data for anomaly detection."""
        np.random.seed(42)
        n_samples = 100
        n_features = 8
        
        # Create normal data
        normal_data = np.random.randn(n_samples, n_features)
        
        # Add some outliers
        outliers = np.random.randn(10, n_features) * 3 + 5
        
        X = np.vstack([normal_data, outliers])
        return X
    
    def test_initialization(self):
        """Test anomaly detector initialization."""
        assert self.detector.method == 'isolation_forest'
        assert self.detector.contamination == 0.1
        assert not self.detector.is_fitted
    
    def test_fitting(self):
        """Test anomaly detector fitting."""
        fit_stats = self.detector.fit(self.X)
        
        assert self.detector.is_fitted
        assert 'method' in fit_stats
        assert 'n_samples' in fit_stats
        assert 'n_features' in fit_stats
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        self.detector.fit(self.X)
        
        results = self.detector.detect_anomalies(self.X)
        
        assert 'anomaly_mask' in results
        assert 'anomaly_indices' in results
        assert 'n_anomalies' in results
        assert 'anomaly_rate' in results
        assert len(results['anomaly_mask']) == len(self.X)
    
    def test_performance_anomaly_detection(self):
        """Test performance-specific anomaly detection."""
        # Create sample student DataFrame
        student_df = pd.DataFrame({
            'Roll Number': [f'S{i:03d}' for i in range(20)],
            'Name': [f'Student_{i}' for i in range(20)],
            'A1': np.random.uniform(60, 90, 20),
            'A2': np.random.uniform(55, 85, 20),
            'A3': np.random.uniform(50, 80, 20)
        })
        
        # Add some anomalous patterns
        student_df.loc[0, 'A2'] = 20  # Sudden drop
        student_df.loc[1, ['A1', 'A2', 'A3']] = [95, 95, 95]  # Unusually consistent
        
        results = self.detector.detect_performance_anomalies(student_df)
        
        assert 'performance_anomalies' in results
        assert 'summary' in results
        assert 'student_flags' in results


class TestSACOPOAMLAnalyzer:
    """Test cases for main ML analyzer."""
    
    def setup_method(self):
        """Set up test data."""
        self.analyzer = SACOPOAMLAnalyzer()
        self.student_df = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample student data."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'Roll Number': [f'21CSE{i:03d}' for i in range(20)],
            'Name': [f'Student_{i}' for i in range(20)],
            'A1': np.random.uniform(50, 95, 20),
            'A2': np.random.uniform(45, 90, 20),
            'A3': np.random.uniform(55, 85, 20),
            'A4': np.random.uniform(50, 88, 20)
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        assert hasattr(self.analyzer, 'data_processor')
        assert hasattr(self.analyzer, 'predictor')
        assert hasattr(self.analyzer, 'risk_analyzer')
        assert hasattr(self.analyzer, 'clustering')
        assert hasattr(self.analyzer, 'recommender')
        assert hasattr(self.analyzer, 'anomaly_detector')
        assert not self.analyzer.is_trained
    
    def test_analyze_course_data(self):
        """Test comprehensive course data analysis."""
        results = self.analyzer.analyze_course_data(self.student_df)
        
        assert self.analyzer.is_trained
        assert 'processed_data' in results
        assert 'predictions' in results
        assert 'risk_assessment' in results
        assert 'clustering' in results
        assert 'interventions' in results
        assert 'anomalies' in results
        assert 'insights' in results
    
    def test_student_profile_generation(self):
        """Test student profile generation."""
        # First run analysis
        self.analyzer.analyze_course_data(self.student_df)
        
        profile = self.analyzer.get_student_profile(0)
        
        assert 'student_index' in profile
        assert 'performance' in profile
        assert 'risk_assessment' in profile
        assert 'cluster_info' in profile
        assert profile['student_index'] == 0
    
    def test_early_warning_report(self):
        """Test early warning report generation."""
        # First run analysis
        self.analyzer.analyze_course_data(self.student_df)
        
        early_warning = self.analyzer.generate_early_warning_report(2)
        
        assert 'assessment_point' in early_warning
        assert 'confidence_level' in early_warning
        assert 'early_predictions' in early_warning
        assert 'early_risk_assessment' in early_warning
        assert early_warning['assessment_point'] == 2


# Test runner
def run_tests():
    """Run all tests."""
    print("Running SACOPOA ML Module Tests...")
    print("=" * 40)
    
    # Test classes
    test_classes = [
        TestDataProcessor,
        TestPerformancePredictor,
        TestRiskAnalyzer,
        TestPerformanceClustering,
        TestInterventionRecommender,
        TestAnomalyDetector,
        TestSACOPOAMLAnalyzer
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        # Get test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Create test instance
                test_instance = test_class()
                test_instance.setup_method()
                
                # Run test method
                getattr(test_instance, test_method)()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{test_method}")
    
    # Summary
    print(f"\n" + "=" * 40)
    print(f"Test Results Summary:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        for test in failed_tests:
            print(f"  - {test}")
    else:
        print(f"\n🎉 All tests passed!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)