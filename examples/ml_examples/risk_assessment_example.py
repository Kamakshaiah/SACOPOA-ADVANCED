"""
Example: Risk Assessment and Early Warning System
================================================

This example demonstrates the risk assessment and early warning
capabilities of the SACOPOA ML module.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import ml_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ml_module.core.risk_analyzer import RiskAnalyzer
from ml_module.utils.data_processor import DataProcessor
from ml_module.utils.visualization import MLVisualizer


def create_diverse_student_data():
    """Create student data with diverse risk patterns."""
    np.random.seed(42)
    
    n_students = 40
    
    # Create different student types
    student_types = {
        'high_performers': 10,
        'steady_average': 15,
        'declining': 8,
        'struggling': 7
    }
    
    students_data = {
        'Roll Number': [f'21CS{str(i+1).zfill(3)}' for i in range(n_students)],
        'Name': [f'Student_{i+1}' for i in range(n_students)]
    }
    
    # Generate assessment scores based on student types
    assessment_scores = []
    current_index = 0
    
    # High performers (consistent good performance)
    for i in range(student_types['high_performers']):
        scores = np.random.normal(85, 5, 5)  # 5 assessments
        scores = np.clip(scores, 75, 100)
        assessment_scores.append(scores)
    current_index += student_types['high_performers']
    
    # Steady average performers
    for i in range(student_types['steady_average']):
        scores = np.random.normal(65, 8, 5)
        scores = np.clip(scores, 45, 85)
        assessment_scores.append(scores)
    current_index += student_types['steady_average']
    
    # Declining performers (start well, decline)
    for i in range(student_types['declining']):
        initial_score = np.random.uniform(75, 90)
        decline_rate = np.random.uniform(3, 8)
        scores = [initial_score - j * decline_rate + np.random.normal(0, 3) for j in range(5)]
        scores = [max(score, 25) for score in scores]  # Don't go below 25
        assessment_scores.append(scores)
    current_index += student_types['declining']
    
    # Struggling students (consistently low)
    for i in range(student_types['struggling']):
        scores = np.random.normal(35, 10, 5)
        scores = np.clip(scores, 15, 55)
        assessment_scores.append(scores)
    
    # Add assessment scores to DataFrame
    for j in range(5):  # 5 assessments
        students_data[f'A{j+1}'] = [scores[j] for scores in assessment_scores]
    
    return pd.DataFrame(students_data)


def main():
    """Main risk assessment example."""
    print("SACOPOA Risk Assessment & Early Warning Example")
    print("=" * 55)
    
    # Create sample data
    print("Creating diverse student data...")
    student_df = create_diverse_student_data()
    print(f"Created data for {len(student_df)} students with different risk patterns")
    
    # Initialize components
    data_processor = DataProcessor()
    risk_analyzer = RiskAnalyzer(risk_threshold=0.4)
    visualizer = MLVisualizer()
    
    # Process data
    print("\nProcessing student data...")
    processed_data = data_processor.process_student_data(student_df)
    
    # Prepare features for ML
    X, y = data_processor.prepare_for_training(
        processed_data['features'], target_column='percentage'
    )
    
    performance_scores = processed_data['performance_metrics']['percentages']
    
    print(f"Processed {len(X)} student records with {X.shape[1]} features")
    
    # Train risk assessment model
    print("\nTraining risk assessment model...")
    training_metrics = risk_analyzer.train(X, performance_scores)
    
    print("Training Results:")
    print(f"  - Binary Risk Model Accuracy: {training_metrics['binary_risk']['accuracy']:.3f}")
    print(f"  - Binary Risk AUC Score: {training_metrics['binary_risk']['auc_score']:.3f}")
    print(f"  - Multi-level Risk Accuracy: {training_metrics['multi_level_risk']['accuracy']:.3f}")
    
    # Perform risk assessment
    print("\nPerforming comprehensive risk assessment...")
    risk_assessment = risk_analyzer.assess_risk(X)
    
    # Display detailed results
    print(f"\nRisk Assessment Results:")
    print(f"  - Students Analyzed: {len(risk_assessment['binary_risk_probability'])}")
    print(f"  - High Risk Students: {len(risk_assessment['high_risk_students'])}")
    print(f"  - Critical Risk Students: {len(risk_assessment['critical_risk_students'])}")
    print(f"  - Average Risk Probability: {np.mean(risk_assessment['binary_risk_probability']):.3f}")
    
    # Risk category distribution
    risk_categories = risk_assessment['risk_categories']
    category_counts = {cat: risk_categories.count(cat) for cat in set(risk_categories)}
    print(f"\nRisk Category Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(risk_categories)) * 100
        print(f"  - {category}: {count} students ({percentage:.1f}%)")
    
    # Generate comprehensive risk report
    student_ids = student_df['Roll Number'].tolist()
    risk_report = risk_analyzer.generate_risk_report(risk_assessment, student_ids)
    
    print(f"\nRisk Report Summary:")
    summary = risk_report['summary']
    print(f"  - Total Students: {summary['total_students']}")
    print(f"  - Immediate Attention Needed: {summary['immediate_attention_count']}")
    print(f"  - Average Risk Probability: {summary['average_risk_probability']:.3f}")
    
    # Show high-risk students details
    print(f"\nHigh-Risk Students Details:")
    high_risk_details = [
        detail for detail in risk_report['student_details']
        if detail['is_high_risk'] or detail['is_critical_risk']
    ]
    
    for detail in high_risk_details[:10]:  # Show first 10
        print(f"  - {detail['student_id']}: {detail['risk_category']} "
              f"(Probability: {detail['risk_probability']:.3f}, "
              f"Urgency: {detail['intervention_urgency']:.2f})")
    
    # Early warning simulation
    print(f"\n" + "="*55)
    print("EARLY WARNING SYSTEM SIMULATION")
    print("="*55)
    
    # Simulate early warning after each assessment
    assessment_points = [1, 2, 3, 4]  # After 2nd, 3rd, 4th, 5th assessment
    
    for assessment_index in assessment_points:
        print(f"\nEarly Warning After Assessment {assessment_index + 1}:")
        
        # Create partial data (only up to current assessment)
        X_partial = X.copy()
        # This is a simplified approach - in practice, would need more sophisticated handling
        
        early_assessment = risk_analyzer.early_risk_assessment(X_partial, assessment_index)
        
        immediate_attention = early_assessment['immediate_attention_students']
        confidence = early_assessment['early_assessment_confidence']
        
        print(f"  - Assessment Confidence: {confidence:.2f}")
        print(f"  - Students Needing Immediate Attention: {len(immediate_attention)}")
        
        if immediate_attention:
            print(f"  - Critical Students (Indices): {immediate_attention[:5]}...")  # Show first 5
        
        # Early intervention recommendations
        if len(immediate_attention) > 0:
            print(f"  - Recommended Actions:")
            print(f"    • Schedule individual meetings with {len(immediate_attention)} students")
            print(f"    • Implement tutoring support")
            print(f"    • Monitor progress closely")
        else:
            print(f"  - No immediate interventions needed at this point")
    
    # Feature importance analysis
    print(f"\n" + "="*55)
    print("RISK FACTOR ANALYSIS")
    print("="*55)
    
    feature_importance = risk_analyzer.get_feature_importance()
    print(f"\nTop Risk Factors (Feature Importance):")
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = [
        'total_score', 'average_score', 'max_score', 'min_score', 'score_std',
        'score_range', 'percentage', 'improvement_trend', 'consistency_score',
        'grade_category'
    ]
    
    for i, (feature_idx, importance) in enumerate(sorted_features[:5]):
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
        print(f"  {i+1}. {feature_name}: {importance:.3f}")
    
    # Risk factor analysis for specific students
    print(f"\nRisk Factor Analysis for Critical Students:")
    critical_students = risk_assessment['critical_risk_students'][:3]  # First 3 critical students
    
    for student_idx in critical_students:
        if student_idx < len(student_ids):
            student_id = student_ids[student_idx]
            risk_prob = risk_assessment['binary_risk_probability'][student_idx]
            
            print(f"\n  Student {student_id} (Risk: {risk_prob:.3f}):")
            
            # Get student's feature values
            student_features = X[student_idx]
            
            # Show key risk indicators
            print(f"    - Performance Percentage: {student_features[6]:.1f}%")  # Assuming percentage is index 6
            print(f"    - Improvement Trend: {student_features[7]:.3f}")  # Assuming trend is index 7
            print(f"    - Consistency Score: {student_features[8]:.3f}")  # Assuming consistency is index 8
    
    # Visualization
    print(f"\nGenerating risk assessment visualizations...")
    
    # Create output directory
    output_dir = './examples/ml_outputs/risk_assessment/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Risk assessment visualization
    risk_viz = visualizer.plot_risk_assessment(
        risk_assessment['binary_risk_probability'],
        risk_assessment['risk_categories'],
        student_ids
    )
    
    visualizer.save_figure(risk_viz, f'{output_dir}/risk_assessment_chart.html')
    
    print(f"Risk assessment visualization saved to '{output_dir}'")
    
    # Export risk report
    risk_df = pd.DataFrame(risk_report['student_details'])
    risk_df.to_excel(f'{output_dir}/risk_assessment_report.xlsx', index=False)
    
    print(f"Risk assessment report exported to '{output_dir}/risk_assessment_report.xlsx'")
    
    # Summary recommendations
    print(f"\n" + "="*55)
    print("SUMMARY RECOMMENDATIONS")
    print("="*55)
    
    total_high_risk = len(risk_assessment['high_risk_students'])
    total_critical = len(risk_assessment['critical_risk_students'])
    total_students = len(student_df)
    
    print(f"\n1. Immediate Actions Required:")
    if total_critical > 0:
        print(f"   • {total_critical} students need critical intervention")
        print(f"   • Schedule emergency academic counseling sessions")
        print(f"   • Implement intensive tutoring programs")
    
    print(f"\n2. Monitoring and Support:")
    if total_high_risk > 0:
        print(f"   • {total_high_risk} students require close monitoring")
        print(f"   • Establish weekly check-ins")
        print(f"   • Provide additional learning resources")
    
    print(f"\n3. Early Warning System:")
    print(f"   • Implement automated risk assessment after each evaluation")
    print(f"   • Set up alert system for declining performance trends")
    print(f"   • Train faculty on risk indicator recognition")
    
    risk_percentage = ((total_high_risk + total_critical) / total_students) * 100
    print(f"\n4. Overall Assessment:")
    print(f"   • {risk_percentage:.1f}% of students are at elevated risk")
    
    if risk_percentage > 30:
        print(f"   • Consider reviewing course design and teaching methods")
        print(f"   • Implement class-wide support measures")
    elif risk_percentage > 20:
        print(f"   • Focus on targeted interventions")
        print(f"   • Enhance support systems")
    else:
        print(f"   • Continue current approach with minor adjustments")
    
    print(f"\nRisk assessment analysis complete!")


if __name__ == "__main__":
    main()