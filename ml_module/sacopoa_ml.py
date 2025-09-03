"""
SACOPOA ML Integration Module
============================

Main integration module that combines all ML components
for comprehensive student performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .core.predictor import PerformancePredictor
from .core.risk_analyzer import RiskAnalyzer
from .core.clustering import PerformanceClustering
from .core.recommender import InterventionRecommender
from .core.anomaly_detector import AnomalyDetector
from .utils.data_processor import DataProcessor
from .utils.visualization import MLVisualizer

warnings.filterwarnings('ignore')


class SACOPOAMLAnalyzer:
    """
    Main class integrating all ML components for comprehensive analysis.
    
    This class provides a unified interface for:
    - Performance prediction
    - Risk assessment
    - Clustering analysis
    - Intervention recommendations
    - Anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ML analyzer.
        
        Args:
            config: Configuration dictionary for ML components
        """
        self.config = config or {}
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.predictor = PerformancePredictor(
            model_type=self.config.get('predictor_model', 'random_forest')
        )
        self.risk_analyzer = RiskAnalyzer(
            risk_threshold=self.config.get('risk_threshold', 0.4)
        )
        self.clustering = PerformanceClustering(
            clustering_method=self.config.get('clustering_method', 'kmeans')
        )
        self.recommender = InterventionRecommender()
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_method', 'isolation_forest'),
            contamination=self.config.get('contamination', 0.1)
        )
        self.visualizer = MLVisualizer(
            style=self.config.get('viz_style', 'educational')
        )
        
        # Analysis results storage
        self.results = {}
        self.is_trained = False
        
    def analyze_course_data(self, student_df: pd.DataFrame, 
                          co_po_matrix: pd.DataFrame = None,
                          run_all_analyses: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive ML analysis on course data.
        
        Args:
            student_df: DataFrame with student assessment data
            co_po_matrix: Optional CO-PO mapping matrix
            run_all_analyses: Whether to run all analysis types
            
        Returns:
            Dictionary containing all analysis results
        """
        print("Starting comprehensive ML analysis...")
        
        # Step 1: Process data
        print("Processing data...")
        processed_data = self.data_processor.process_student_data(student_df, co_po_matrix)
        features = processed_data['features']
        performance_metrics = processed_data['performance_metrics']
        
        # Prepare data for ML
        X, y = self.data_processor.prepare_for_training(
            features, target_column='percentage'
        )
        
        # Store processed data
        self.results['processed_data'] = processed_data
        
        if not run_all_analyses:
            return self.results
        
        # Step 2: Performance prediction
        print("Training performance prediction models...")
        prediction_metrics = self.predictor.train(
            X, performance_metrics['percentages'], 
            performance_metrics['grade_categories'],
            feature_names=self.data_processor.feature_names
        )
        
        predictions = self.predictor.predict_performance(X)
        self.results['predictions'] = {
            'actual': performance_metrics['percentages'],
            'predicted': predictions['performance_scores'],
            'pass_probability': predictions['pass_probability'],
            'metrics': prediction_metrics
        }
        
        # Step 3: Risk assessment
        print("Analyzing student risk factors...")
        risk_metrics = self.risk_analyzer.train(X, performance_metrics['percentages'])
        risk_assessment = self.risk_analyzer.assess_risk(X)
        
        self.results['risk_assessment'] = {
            'risk_probabilities': risk_assessment['binary_risk_probability'],
            'risk_categories': risk_assessment['risk_categories'],
            'high_risk_students': risk_assessment['high_risk_students'],
            'critical_risk_students': risk_assessment['critical_risk_students'],
            'metrics': risk_metrics
        }
        
        # Step 4: Clustering analysis
        print("Performing clustering analysis...")
        clustering_results = self.clustering.fit_student_clusters(
            X, performance_metrics['percentages']
        )
        
        # Learning pattern analysis
        if 'assessment_scores' in processed_data:
            learning_patterns = self.clustering.identify_learning_patterns(
                X, processed_data['assessment_scores']
            )
            clustering_results['learning_patterns'] = learning_patterns
        
        self.results['clustering'] = clustering_results
        
        # Step 5: Intervention recommendations
        print("Generating intervention recommendations...")
        intervention_recs = self.recommender.recommend_interventions(
            X, 
            risk_assessment['multi_level_risk_prediction'],
            performance_metrics['percentages']
        )
        
        self.results['interventions'] = {
            'recommendations': intervention_recs,
            'summary': self.recommender.generate_intervention_report(
                intervention_recs, processed_data.get('student_ids')
            )
        }
        
        # Step 6: Anomaly detection
        print("Detecting anomalies...")
        anomaly_stats = self.anomaly_detector.fit(X)
        anomaly_results = self.anomaly_detector.detect_anomalies(X)
        
        # Performance-specific anomaly detection
        perf_anomalies = self.anomaly_detector.detect_performance_anomalies(student_df)
        
        self.results['anomalies'] = {
            'general_anomalies': anomaly_results,
            'performance_anomalies': perf_anomalies,
            'fit_statistics': anomaly_stats
        }
        
        # Step 7: Generate summary insights
        print("Generating insights...")
        self.results['insights'] = self._generate_insights()
        
        self.is_trained = True
        print("Analysis complete!")
        
        return self.results
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate high-level insights from all analyses."""
        insights = {
            'summary': {},
            'key_findings': [],
            'recommendations': [],
            'alerts': []
        }
        
        # Performance insights
        if 'predictions' in self.results:
            pred_metrics = self.results['predictions']['metrics']
            insights['summary']['prediction_accuracy'] = pred_metrics.get('performance_r2', 0)
            
            if pred_metrics.get('performance_r2', 0) > 0.8:
                insights['key_findings'].append(
                    "High prediction accuracy suggests stable performance patterns"
                )
            elif pred_metrics.get('performance_r2', 0) < 0.5:
                insights['key_findings'].append(
                    "Low prediction accuracy indicates high performance variability"
                )
        
        # Risk insights
        if 'risk_assessment' in self.results:
            risk_data = self.results['risk_assessment']
            critical_count = len(risk_data['critical_risk_students'])
            high_risk_count = len(risk_data['high_risk_students'])
            
            insights['summary']['critical_risk_students'] = critical_count
            insights['summary']['high_risk_students'] = high_risk_count
            
            if critical_count > 0:
                insights['alerts'].append(
                    f"{critical_count} students at critical risk - immediate intervention needed"
                )
            
            if (critical_count + high_risk_count) > 0.3 * len(risk_data['risk_probabilities']):
                insights['alerts'].append(
                    "High proportion of at-risk students - review course design"
                )
        
        # Clustering insights
        if 'clustering' in self.results:
            cluster_data = self.results['clustering']
            n_clusters = cluster_data['clustering_metrics']['n_clusters']
            
            insights['summary']['student_groups'] = n_clusters
            
            if 'learning_patterns' in cluster_data:
                for cluster_id, pattern in cluster_data['learning_patterns'].items():
                    pattern_type = pattern.get('pattern_type', 'Unknown')
                    if 'Decliner' in pattern_type:
                        insights['alerts'].append(
                            f"Cluster {cluster_id}: Declining performance pattern detected"
                        )
                    elif 'Improver' in pattern_type:
                        insights['key_findings'].append(
                            f"Cluster {cluster_id}: Positive improvement pattern identified"
                        )
        
        # Intervention insights
        if 'interventions' in self.results:
            intervention_data = self.results['interventions']['summary']
            critical_cases = intervention_data.get('critical_cases', 0)
            
            if critical_cases > 0:
                insights['recommendations'].append(
                    f"Priority: Address {critical_cases} critical intervention cases"
                )
            
            # Most common intervention needs
            freq_interventions = intervention_data.get('intervention_frequency', {})
            if freq_interventions:
                top_intervention = max(freq_interventions.items(), key=lambda x: x[1])
                insights['recommendations'].append(
                    f"Most needed intervention: {top_intervention[0]} ({top_intervention[1]} students)"
                )
        
        # Anomaly insights
        if 'anomalies' in self.results:
            anomaly_data = self.results['anomalies']['general_anomalies']
            anomaly_count = anomaly_data.get('n_anomalies', 0)
            
            insights['summary']['anomalous_students'] = anomaly_count
            
            if anomaly_count > 0:
                insights['key_findings'].append(
                    f"{anomaly_count} students show unusual performance patterns"
                )
            
            # Performance anomaly summary
            perf_anomalies = self.results['anomalies']['performance_anomalies']
            if 'summary' in perf_anomalies:
                sudden_drops = perf_anomalies['summary'].get('total_sudden_drops', 0)
                if sudden_drops > 0:
                    insights['alerts'].append(
                        f"{sudden_drops} instances of sudden performance drops detected"
                    )
        
        return insights
    
    def generate_early_warning_report(self, assessment_index: int) -> Dict[str, Any]:
        """
        Generate early warning report based on partial assessment data.
        
        Args:
            assessment_index: Index of last completed assessment
            
        Returns:
            Early warning report
        """
        if not self.is_trained:
            raise ValueError("Must run full analysis before generating early warnings")
        
        # Get original data
        processed_data = self.results['processed_data']
        X = processed_data['features'].values
        
        # Create partial feature matrix
        X_partial = X.copy()
        # Zero out future assessments (simplified approach)
        # This would need to be more sophisticated in practice
        
        # Early predictions
        early_predictions = self.predictor.predict_early_performance(X_partial, assessment_index)
        
        # Early risk assessment
        early_risk = self.risk_analyzer.early_risk_assessment(X_partial, assessment_index)
        
        # Generate report
        report = {
            'assessment_point': assessment_index,
            'confidence_level': early_predictions.get('early_prediction_confidence', 0.5),
            'early_predictions': early_predictions,
            'early_risk_assessment': early_risk,
            'immediate_attention_students': early_risk.get('immediate_attention_students', []),
            'recommendations': self._generate_early_recommendations(early_risk, assessment_index)
        }
        
        return report
    
    def _generate_early_recommendations(self, early_risk: Dict, assessment_index: int) -> List[str]:
        """Generate recommendations for early intervention."""
        recommendations = []
        
        immediate_attention = early_risk.get('immediate_attention_students', [])
        if immediate_attention:
            recommendations.append(
                f"Immediate action needed for {len(immediate_attention)} students"
            )
        
        high_risk = early_risk.get('high_risk_students', [])
        if high_risk:
            recommendations.append(
                f"Monitor {len(high_risk)} high-risk students closely"
            )
        
        if assessment_index < 3:  # Early in course
            recommendations.append(
                "Consider additional practice sessions for struggling students"
            )
            recommendations.append(
                "Implement peer support systems"
            )
        
        return recommendations
    
    def create_visualizations(self, save_dir: str = './ml_outputs/') -> Dict[str, Any]:
        """
        Create comprehensive visualizations of ML results.
        
        Args:
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with visualization objects
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        visualizations = {}
        
        # Performance predictions
        if 'predictions' in self.results:
            pred_data = self.results['predictions']
            viz = self.visualizer.plot_performance_predictions(
                pred_data['actual'],
                pred_data['predicted']
            )
            visualizations['performance_predictions'] = viz
            self.visualizer.save_figure(viz, f'{save_dir}/performance_predictions.html')
        
        # Risk assessment
        if 'risk_assessment' in self.results:
            risk_data = self.results['risk_assessment']
            viz = self.visualizer.plot_risk_assessment(
                risk_data['risk_probabilities'],
                risk_data['risk_categories']
            )
            visualizations['risk_assessment'] = viz
            self.visualizer.save_figure(viz, f'{save_dir}/risk_assessment.html')
        
        # Clustering
        if 'clustering' in self.results:
            cluster_data = self.results['clustering']
            viz = self.visualizer.plot_clustering_results(
                cluster_data['reduced_features'],
                cluster_data['cluster_labels']
            )
            visualizations['clustering'] = viz
            self.visualizer.save_figure(viz, f'{save_dir}/clustering_analysis.html')
        
        # Intervention recommendations
        if 'interventions' in self.results:
            intervention_data = self.results['interventions']['recommendations']
            viz = self.visualizer.plot_intervention_recommendations(intervention_data)
            visualizations['interventions'] = viz
            self.visualizer.save_figure(viz, f'{save_dir}/intervention_recommendations.html')
        
        # Anomaly detection
        if 'anomalies' in self.results:
            anomaly_data = self.results['anomalies']['general_anomalies']
            processed_data = self.results['processed_data']
            
            viz = self.visualizer.plot_anomaly_detection(
                processed_data['features'].values,
                anomaly_data['anomaly_mask']
            )
            visualizations['anomalies'] = viz
            self.visualizer.save_figure(viz, f'{save_dir}/anomaly_detection.html')
        
        # Dashboard
        dashboard = self.visualizer.create_ml_dashboard(self.results)
        visualizations['dashboard'] = dashboard
        self.visualizer.save_figure(dashboard, f'{save_dir}/ml_dashboard.html')
        
        # Static plots for reports
        self.visualizer.create_static_plots(self.results, save_dir)
        
        return visualizations
    
    def export_results(self, filepath: str, format: str = 'excel') -> None:
        """
        Export analysis results to file.
        
        Args:
            filepath: Output file path
            format: Export format ('excel', 'json', 'csv')
        """
        if format == 'excel':
            self._export_to_excel(filepath)
        elif format == 'json':
            import json
            with open(filepath, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._convert_numpy_for_json(self.results)
                json.dump(json_results, f, indent=2)
        elif format == 'csv':
            self._export_to_csv(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_excel(self, filepath: str):
        """Export results to Excel file with multiple sheets."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # Summary sheet
            if 'insights' in self.results:
                summary_data = {
                    'Metric': [],
                    'Value': []
                }
                
                for key, value in self.results['insights']['summary'].items():
                    summary_data['Metric'].append(key.replace('_', ' ').title())
                    summary_data['Value'].append(str(value))
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Predictions sheet
            if 'predictions' in self.results:
                pred_df = pd.DataFrame({
                    'Student_ID': range(len(self.results['predictions']['actual'])),
                    'Actual_Score': self.results['predictions']['actual'],
                    'Predicted_Score': self.results['predictions']['predicted'],
                    'Pass_Probability': self.results['predictions']['pass_probability']
                })
                pred_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Risk assessment sheet
            if 'risk_assessment' in self.results:
                risk_df = pd.DataFrame({
                    'Student_ID': range(len(self.results['risk_assessment']['risk_probabilities'])),
                    'Risk_Probability': self.results['risk_assessment']['risk_probabilities'],
                    'Risk_Category': self.results['risk_assessment']['risk_categories']
                })
                risk_df.to_excel(writer, sheet_name='Risk_Assessment', index=False)
    
    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        else:
            return obj
    
    def _export_to_csv(self, filepath: str):
        """Export main results to CSV file."""
        if 'processed_data' not in self.results:
            raise ValueError("No processed data available for export")
        
        # Combine main results into single DataFrame
        data = {}
        
        # Add student IDs
        student_ids = self.results['processed_data'].get('student_ids')
        if student_ids:
            data['Student_ID'] = student_ids
        else:
            data['Student_ID'] = range(len(self.results['predictions']['actual']))
        
        # Add predictions
        if 'predictions' in self.results:
            data['Actual_Performance'] = self.results['predictions']['actual']
            data['Predicted_Performance'] = self.results['predictions']['predicted']
            data['Pass_Probability'] = self.results['predictions']['pass_probability']
        
        # Add risk assessment
        if 'risk_assessment' in self.results:
            data['Risk_Probability'] = self.results['risk_assessment']['risk_probabilities']
            data['Risk_Category'] = self.results['risk_assessment']['risk_categories']
        
        # Add clustering
        if 'clustering' in self.results:
            data['Cluster_ID'] = self.results['clustering']['cluster_labels']
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def get_student_profile(self, student_index: int) -> Dict[str, Any]:
        """
        Get comprehensive profile for a specific student.
        
        Args:
            student_index: Index of the student
            
        Returns:
            Student profile with all ML analysis results
        """
        if not self.is_trained:
            raise ValueError("Must run analysis before generating student profiles")
        
        profile = {
            'student_index': student_index,
            'student_id': None,
            'performance': {},
            'risk_assessment': {},
            'cluster_info': {},
            'interventions': {},
            'anomalies': {}
        }
        
        # Student ID
        student_ids = self.results['processed_data'].get('student_ids')
        if student_ids and student_index < len(student_ids):
            profile['student_id'] = student_ids[student_index]
        
        # Performance data
        if 'predictions' in self.results:
            profile['performance'] = {
                'actual_score': float(self.results['predictions']['actual'][student_index]),
                'predicted_score': float(self.results['predictions']['predicted'][student_index]),
                'pass_probability': float(self.results['predictions']['pass_probability'][student_index])
            }
        
        # Risk assessment
        if 'risk_assessment' in self.results:
            profile['risk_assessment'] = {
                'risk_probability': float(self.results['risk_assessment']['risk_probabilities'][student_index]),
                'risk_category': self.results['risk_assessment']['risk_categories'][student_index],
                'is_high_risk': student_index in self.results['risk_assessment']['high_risk_students'],
                'is_critical_risk': student_index in self.results['risk_assessment']['critical_risk_students']
            }
        
        # Cluster information
        if 'clustering' in self.results:
            cluster_id = self.results['clustering']['cluster_labels'][student_index]
            profile['cluster_info'] = {
                'cluster_id': int(cluster_id),
                'cluster_analysis': self.results['clustering']['cluster_analysis'].get(f'cluster_{cluster_id}', {})
            }
        
        # Interventions
        if 'interventions' in self.results:
            student_interventions = [
                rec for rec in self.results['interventions']['recommendations']
                if rec['student_index'] == student_index
            ]
            if student_interventions:
                profile['interventions'] = student_interventions[0]
        
        # Anomalies
        if 'anomalies' in self.results:
            anomaly_mask = self.results['anomalies']['general_anomalies']['anomaly_mask']
            if student_index < len(anomaly_mask):
                profile['anomalies'] = {
                    'is_anomalous': bool(anomaly_mask[student_index]),
                    'anomaly_score': float(self.results['anomalies']['general_anomalies']['anomaly_scores'][student_index])
                }
        
        return profile