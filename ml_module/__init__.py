"""
SACOPOA Machine Learning Module
===============================

Advanced machine learning capabilities for Student Assessment & CO-PO Analysis Tool.

This module provides:
- Student performance prediction
- Risk assessment and early warning systems  
- Performance clustering and pattern analysis
- Recommendation engine for educational interventions
- Anomaly detection for unusual performance patterns

Author: SACOPOA Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SACOPOA Development Team"

# Import main ML components
from .core.predictor import PerformancePredictor
from .core.risk_analyzer import RiskAnalyzer
from .core.clustering import PerformanceClustering
from .core.recommender import InterventionRecommender
from .core.anomaly_detector import AnomalyDetector
from .utils.data_processor import DataProcessor
from .utils.visualization import MLVisualizer

__all__ = [
    'PerformancePredictor',
    'RiskAnalyzer', 
    'PerformanceClustering',
    'InterventionRecommender',
    'AnomalyDetector',
    'DataProcessor',
    'MLVisualizer'
]