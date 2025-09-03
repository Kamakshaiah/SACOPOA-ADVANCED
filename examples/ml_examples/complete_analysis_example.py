"""
Example: Complete SACOPOA ML Analysis
=====================================

This example demonstrates how to use the SACOPOA ML module
for comprehensive student performance analysis.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import ml_module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_module.sacopoa_ml import SACOPOAMLAnalyzer


def create_sample_data():
    """Create sample student data for demonstration."""
    np.random.seed(42)
    
    # Create sample student data
    n_students = 50
    n_assessments = 5
    
    # Generate student data
    students_data = {
        'Roll Number': [f'21CSE{str(i+1).zfill(3)}' for i in range(n_students)],
        'Name': [f'Student_{i+1}' for i in range(n_students)]
    }
    
    # Generate assessment scores with realistic patterns
    base_performance = np.random.normal(70, 15, n_students)  # Base ability
    base_performance = np.clip(base_performance, 30, 95)  # Realistic range
    
    for i in range(n_assessments):
        # Add some variation and trends
        assessment_scores = base_performance + np.random.normal(0, 8, n_students)
        
        # Add some realistic patterns
        if i > 0:  # Progressive difficulty or learning
            trend = np.random.normal(2, 3, n_students)  # Slight improvement trend
            assessment_scores += trend
        
        # Ensure realistic score ranges
        assessment_scores = np.clip(assessment_scores, 0, 100)
        students_data[f'A{i+1}'] = assessment_scores
    
    student_df = pd.DataFrame(students_data)
    
    # Create sample CO-PO matrix
    co_po_matrix = pd.DataFrame({
        'CO': ['CO1', 'CO2', 'CO3', 'CO4', 'CO5'],
        'PO1': [3, 2, 1, 2, 3],
        'PO2': [2, 3, 2, 1, 2],
        'PO3': [1, 2, 3, 3, 1],
        'PO4': [2, 1, 2, 3, 2],
        'PSO1': [3, 2, 1, 2, 1],
        'PSO2': [1, 3, 2, 1, 3]
    })
    
    return student_df, co_po_matrix


def main():
    """Main example function."""
    print("SACOPOA Machine Learning Analysis Example")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample data...")
    student_df, co_po_matrix = create_sample_data()
    
    print(f"Created data for {len(student_df)} students with {len([col for col in student_df.columns if col.startswith('A')])} assessments")
    
    # Initialize ML analyzer
    print("\nInitializing ML analyzer...")
    config = {
        'predictor_model': 'random_forest',
        'risk_threshold': 0.4,
        'clustering_method': 'kmeans',
        'anomaly_method': 'isolation_forest',
        'contamination': 0.1,
        'viz_style': 'educational'
    }
    
    ml_analyzer = SACOPOAMLAnalyzer(config)
    
    # Run comprehensive analysis
    print("\nRunning comprehensive ML analysis...")
    results = ml_analyzer.analyze_course_data(student_df, co_po_matrix)
    
    # Display results summary
    print("\n" + "="*50)
    print("ANALYSIS RESULTS SUMMARY")
    print("="*50)
    
    # Performance prediction results
    if 'predictions' in results:
        pred_metrics = results['predictions']['metrics']
        print(f"\nPerformance Prediction:")
        print(f"  - R² Score: {pred_metrics.get('performance_r2', 0):.3f}")
        print(f"  - Mean Absolute Error: {pred_metrics.get('performance_mae', 0):.2f}")
        print(f"  - Cross-validation Score: {pred_metrics.get('cv_mean', 0):.3f} ± {pred_metrics.get('cv_std', 0):.3f}")
    
    # Risk assessment results
    if 'risk_assessment' in results:
        risk_data = results['risk_assessment']
        print(f"\nRisk Assessment:")
        print(f"  - High Risk Students: {len(risk_data['high_risk_students'])}")
        print(f"  - Critical Risk Students: {len(risk_data['critical_risk_students'])}")
        print(f"  - Average Risk Probability: {np.mean(risk_data['risk_probabilities']):.3f}")
    
    # Clustering results
    if 'clustering' in results:
        cluster_data = results['clustering']
        print(f"\nClustering Analysis:")
        print(f"  - Number of Clusters: {cluster_data['clustering_metrics']['n_clusters']}")
        print(f"  - Silhouette Score: {cluster_data['clustering_metrics'].get('silhouette_score', 0):.3f}")
        
        # Cluster sizes
        cluster_sizes = cluster_data['clustering_metrics']['cluster_sizes']
        print(f"  - Cluster Sizes: {cluster_sizes}")
    
    # Intervention recommendations
    if 'interventions' in results:
        intervention_summary = results['interventions']['summary']
        print(f"\nIntervention Recommendations:")
        print(f"  - Total Students: {intervention_summary['summary']['total_students']}")
        print(f"  - Critical Cases: {intervention_summary['summary']['critical_cases']}")
        print(f"  - High Priority Cases: {intervention_summary['summary']['high_priority_cases']}")
        
        # Most common interventions
        freq_interventions = intervention_summary.get('intervention_frequency', {})
        if freq_interventions:
            top_3 = sorted(freq_interventions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  - Top Interventions Needed:")
            for intervention, count in top_3:
                print(f"    * {intervention}: {count} students")
    
    # Anomaly detection
    if 'anomalies' in results:
        anomaly_data = results['anomalies']['general_anomalies']
        print(f"\nAnomaly Detection:")
        print(f"  - Anomalous Students: {anomaly_data['n_anomalies']}")
        print(f"  - Anomaly Rate: {anomaly_data['anomaly_rate']:.1%}")
        
        # Performance anomalies
        perf_anomalies = results['anomalies']['performance_anomalies']
        if 'summary' in perf_anomalies:
            perf_summary = perf_anomalies['summary']
            print(f"  - Sudden Performance Drops: {perf_summary.get('total_sudden_drops', 0)}")
            print(f"  - Consistency Anomalies: {perf_summary.get('total_consistency_anomalies', 0)}")
    
    # Key insights
    if 'insights' in results:
        insights = results['insights']
        print(f"\nKey Insights:")
        
        for finding in insights.get('key_findings', []):
            print(f"  • {finding}")
        
        print(f"\nRecommendations:")
        for rec in insights.get('recommendations', []):
            print(f"  • {rec}")
        
        if insights.get('alerts'):
            print(f"\nAlerts:")
            for alert in insights['alerts']:
                print(f"  ⚠ {alert}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    visualizations = ml_analyzer.create_visualizations('./examples/ml_outputs/')
    print(f"Visualizations saved to './examples/ml_outputs/'")
    
    # Export results
    print(f"\nExporting results...")
    ml_analyzer.export_results('./examples/ml_outputs/analysis_results.xlsx', 'excel')
    ml_analyzer.export_results('./examples/ml_outputs/analysis_results.json', 'json')
    ml_analyzer.export_results('./examples/ml_outputs/analysis_results.csv', 'csv')
    print(f"Results exported to './examples/ml_outputs/'")
    
    # Example: Student profile
    print(f"\nExample Student Profile:")
    student_profile = ml_analyzer.get_student_profile(0)
    print(f"Student ID: {student_profile.get('student_id', 'N/A')}")
    print(f"Actual Performance: {student_profile['performance']['actual_score']:.1f}%")
    print(f"Predicted Performance: {student_profile['performance']['predicted_score']:.1f}%")
    print(f"Risk Category: {student_profile['risk_assessment']['risk_category']}")
    print(f"Cluster: {student_profile['cluster_info']['cluster_id']}")
    print(f"Is Anomalous: {student_profile['anomalies']['is_anomalous']}")
    
    # Example: Early warning system
    print(f"\nExample Early Warning (after 3 assessments):")
    early_warning = ml_analyzer.generate_early_warning_report(2)  # After 3rd assessment
    print(f"Confidence Level: {early_warning['confidence_level']:.2f}")
    print(f"Students Needing Immediate Attention: {len(early_warning['immediate_attention_students'])}")
    
    for rec in early_warning['recommendations']:
        print(f"  • {rec}")
    
    print(f"\nAnalysis complete! Check the output files for detailed results.")


if __name__ == "__main__":
    main()