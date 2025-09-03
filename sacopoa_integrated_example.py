"""
SACOPOA Main Application Integration Example
===========================================

Example showing how to integrate the ML module with the main SACOPOA application.
This demonstrates the complete workflow from data loading to analysis and reporting.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import ML module
from ml_module.sacopoa_ml import SACOPOAMLAnalyzer


class SACOPOAIntegratedAnalyzer:
    """
    Integrated analyzer that combines traditional SACOPOA analysis 
    with advanced ML capabilities.
    """
    
    def __init__(self):
        """Initialize the integrated analyzer."""
        self.ml_analyzer = SACOPOAMLAnalyzer({
            'predictor_model': 'random_forest',
            'risk_threshold': 0.4,
            'clustering_method': 'kmeans',
            'anomaly_method': 'isolation_forest',
            'viz_style': 'professional'
        })
        
        self.traditional_results = {}
        self.ml_results = {}
        self.integrated_insights = {}
    
    def load_course_data(self, excel_file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load course data from Excel file (SACOPOA format).
        
        Args:
            excel_file_path: Path to Excel file with student data
            
        Returns:
            Dictionary with student data and CO-PO matrix
        """
        try:
            # Load student data sheet
            student_df = pd.read_excel(excel_file_path, sheet_name=0)
            
            # Try to load CO-PO matrix sheet
            try:
                co_po_matrix = pd.read_excel(excel_file_path, sheet_name=1)
            except:
                co_po_matrix = None
                print("Warning: CO-PO matrix sheet not found, proceeding without it")
            
            return {
                'student_data': student_df,
                'co_po_matrix': co_po_matrix
            }
            
        except Exception as e:
            raise ValueError(f"Error loading course data: {str(e)}")
    
    def run_traditional_analysis(self, student_df: pd.DataFrame, 
                                co_po_matrix: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run traditional SACOPOA analysis.
        
        Args:
            student_df: Student data DataFrame
            co_po_matrix: Optional CO-PO matrix
            
        Returns:
            Traditional analysis results
        """
        print("Running traditional SACOPOA analysis...")
        
        # Basic statistical analysis
        assessment_cols = [col for col in student_df.columns if col.startswith('A')]
        
        results = {
            'total_students': len(student_df),
            'total_assessments': len(assessment_cols),
            'student_statistics': {},
            'assessment_statistics': {},
            'co_po_analysis': {},
            'performance_categories': {}
        }
        
        # Student-wise statistics
        for idx, student in student_df.iterrows():
            student_id = student.get('Roll Number', f'Student_{idx}')
            scores = [student[col] for col in assessment_cols if pd.notna(student[col])]
            
            if scores:
                total_score = sum(scores)
                avg_score = total_score / len(scores)
                percentage = (total_score / (len(assessment_cols) * 100)) * 100
                
                results['student_statistics'][student_id] = {
                    'total_score': total_score,
                    'average_score': avg_score,
                    'percentage': percentage,
                    'grade': self._calculate_grade(percentage)
                }
        
        # Assessment-wise statistics
        for col in assessment_cols:
            scores = student_df[col].dropna()
            results['assessment_statistics'][col] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'pass_rate': float((scores >= 40).mean() * 100)
            }
        
        # Performance categorization
        percentages = [stats['percentage'] for stats in results['student_statistics'].values()]
        results['performance_categories'] = {
            'Excellent (≥85%)': sum(1 for p in percentages if p >= 85),
            'Good (70-84%)': sum(1 for p in percentages if 70 <= p < 85),
            'Average (55-69%)': sum(1 for p in percentages if 55 <= p < 70),
            'Below Average (40-54%)': sum(1 for p in percentages if 40 <= p < 55),
            'Poor (<40%)': sum(1 for p in percentages if p < 40)
        }
        
        # Basic CO-PO analysis if matrix provided
        if co_po_matrix is not None:
            results['co_po_analysis'] = self._basic_co_po_analysis(
                results['student_statistics'], co_po_matrix
            )
        
        self.traditional_results = results
        return results
    
    def run_ml_analysis(self, student_df: pd.DataFrame, 
                       co_po_matrix: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run advanced ML analysis.
        
        Args:
            student_df: Student data DataFrame
            co_po_matrix: Optional CO-PO matrix
            
        Returns:
            ML analysis results
        """
        print("Running advanced ML analysis...")
        
        # Run comprehensive ML analysis
        ml_results = self.ml_analyzer.analyze_course_data(student_df, co_po_matrix)
        
        self.ml_results = ml_results
        return ml_results
    
    def generate_integrated_insights(self) -> Dict[str, Any]:
        """
        Generate insights by combining traditional and ML analysis.
        
        Returns:
            Integrated insights and recommendations
        """
        print("Generating integrated insights...")
        
        insights = {
            'performance_insights': {},
            'risk_insights': {},
            'learning_patterns': {},
            'recommendations': [],
            'alerts': [],
            'quality_metrics': {}
        }
        
        # Performance insights
        if self.traditional_results and self.ml_results:
            # Compare traditional vs ML predictions
            traditional_avg = np.mean([
                stats['percentage'] for stats in self.traditional_results['student_statistics'].values()
            ])
            
            if 'predictions' in self.ml_results:
                ml_avg = np.mean(self.ml_results['predictions']['actual'])
                prediction_accuracy = self.ml_results['predictions']['metrics'].get('performance_r2', 0)
                
                insights['performance_insights'] = {
                    'traditional_average': traditional_avg,
                    'ml_average': ml_avg,
                    'prediction_accuracy': prediction_accuracy,
                    'consistency': abs(traditional_avg - ml_avg) < 5.0
                }
        
        # Risk insights
        if 'risk_assessment' in self.ml_results:
            risk_data = self.ml_results['risk_assessment']
            total_students = len(risk_data['risk_probabilities'])
            high_risk_count = len(risk_data['high_risk_students'])
            critical_risk_count = len(risk_data['critical_risk_students'])
            
            insights['risk_insights'] = {
                'total_at_risk': high_risk_count + critical_risk_count,
                'risk_percentage': ((high_risk_count + critical_risk_count) / total_students) * 100,
                'immediate_action_needed': critical_risk_count,
                'monitoring_required': high_risk_count
            }
            
            # Risk-based recommendations
            if critical_risk_count > 0:
                insights['alerts'].append(f"CRITICAL: {critical_risk_count} students need immediate intervention")
            
            if (high_risk_count + critical_risk_count) > total_students * 0.3:
                insights['recommendations'].append("High proportion of at-risk students - review course design")
        
        # Learning patterns
        if 'clustering' in self.ml_results:
            cluster_data = self.ml_results['clustering']
            insights['learning_patterns'] = {
                'student_groups': cluster_data['clustering_metrics']['n_clusters'],
                'group_quality': cluster_data['clustering_metrics'].get('silhouette_score', 0),
                'patterns_identified': cluster_data.get('learning_patterns', {})
            }
        
        # Quality metrics
        if self.traditional_results:
            total_students = self.traditional_results['total_students']
            categories = self.traditional_results['performance_categories']
            
            pass_rate = ((total_students - categories.get('Poor (<40%)', 0)) / total_students) * 100
            excellence_rate = (categories.get('Excellent (≥85%)', 0) / total_students) * 100
            
            insights['quality_metrics'] = {
                'pass_rate': pass_rate,
                'excellence_rate': excellence_rate,
                'quality_index': (pass_rate * 0.6 + excellence_rate * 0.4)  # Weighted quality index
            }
            
            # Quality-based recommendations
            if pass_rate < 80:
                insights['recommendations'].append("Low pass rate - implement additional support measures")
            if excellence_rate < 15:
                insights['recommendations'].append("Low excellence rate - consider enrichment activities")
        
        # ML-based recommendations
        if 'insights' in self.ml_results:
            ml_insights = self.ml_results['insights']
            insights['recommendations'].extend(ml_insights.get('recommendations', []))
            insights['alerts'].extend(ml_insights.get('alerts', []))
        
        self.integrated_insights = insights
        return insights
    
    def generate_comprehensive_report(self, output_dir: str = './reports/') -> Dict[str, str]:
        """
        Generate comprehensive report with all analysis results.
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Dictionary with file paths of generated reports
        """
        print("Generating comprehensive reports...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        
        # Export traditional analysis
        if self.traditional_results:
            traditional_file = os.path.join(output_dir, 'traditional_analysis.xlsx')
            self._export_traditional_results(traditional_file)
            report_files['traditional_analysis'] = traditional_file
        
        # Export ML analysis
        if self.ml_results:
            ml_file = os.path.join(output_dir, 'ml_analysis.xlsx')
            self.ml_analyzer.export_results(ml_file, 'excel')
            report_files['ml_analysis'] = ml_file
        
        # Generate visualizations
        if self.ml_results:
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            self.ml_analyzer.create_visualizations(viz_dir)
            report_files['visualizations'] = viz_dir
        
        # Export integrated insights
        insights_file = os.path.join(output_dir, 'integrated_insights.json')
        import json
        with open(insights_file, 'w') as f:
            json.dump(self.integrated_insights, f, indent=2, default=str)
        report_files['integrated_insights'] = insights_file
        
        return report_files
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade from percentage."""
        if percentage >= 85:
            return 'A'
        elif percentage >= 70:
            return 'B'
        elif percentage >= 55:
            return 'C'
        elif percentage >= 40:
            return 'D'
        else:
            return 'F'
    
    def _basic_co_po_analysis(self, student_stats: Dict, co_po_matrix: pd.DataFrame) -> Dict:
        """Basic CO-PO analysis (simplified)."""
        # This is a simplified implementation
        # In practice, would need more sophisticated CO-PO mapping
        
        analysis = {
            'co_attainment': {},
            'po_attainment': {},
            'average_attainment': 0
        }
        
        # Calculate average performance as proxy for attainment
        avg_performance = np.mean([stats['percentage'] for stats in student_stats.values()])
        analysis['average_attainment'] = avg_performance / 100  # Convert to 0-1 scale
        
        return analysis
    
    def _export_traditional_results(self, filepath: str):
        """Export traditional analysis results to Excel."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Student statistics
            student_data = []
            for student_id, stats in self.traditional_results['student_statistics'].items():
                student_data.append({
                    'Student_ID': student_id,
                    'Total_Score': stats['total_score'],
                    'Average_Score': stats['average_score'],
                    'Percentage': stats['percentage'],
                    'Grade': stats['grade']
                })
            
            pd.DataFrame(student_data).to_excel(writer, sheet_name='Student_Statistics', index=False)
            
            # Assessment statistics
            assessment_data = []
            for assessment, stats in self.traditional_results['assessment_statistics'].items():
                assessment_data.append({
                    'Assessment': assessment,
                    'Mean': stats['mean'],
                    'Std_Dev': stats['std'],
                    'Min': stats['min'],
                    'Max': stats['max'],
                    'Pass_Rate': stats['pass_rate']
                })
            
            pd.DataFrame(assessment_data).to_excel(writer, sheet_name='Assessment_Statistics', index=False)


def create_demo_data() -> pd.DataFrame:
    """Create demonstration data for testing."""
    np.random.seed(42)
    
    # Create realistic student data
    n_students = 40
    student_data = {
        'Roll Number': [f'21CSE{str(i+1).zfill(3)}' for i in range(n_students)],
        'Name': [f'Student_{i+1}' for i in range(n_students)]
    }
    
    # Create different performance groups
    high_performers = 10
    average_performers = 20
    struggling_students = 10
    
    all_scores = []
    
    # High performers (80-95 range)
    for i in range(high_performers):
        base_score = np.random.uniform(80, 95)
        scores = [base_score + np.random.normal(0, 3) for _ in range(5)]
        scores = [max(0, min(100, score)) for score in scores]
        all_scores.append(scores)
    
    # Average performers (55-75 range)
    for i in range(average_performers):
        base_score = np.random.uniform(55, 75)
        scores = [base_score + np.random.normal(0, 5) for _ in range(5)]
        scores = [max(0, min(100, score)) for score in scores]
        all_scores.append(scores)
    
    # Struggling students (25-50 range)
    for i in range(struggling_students):
        base_score = np.random.uniform(25, 50)
        scores = [base_score + np.random.normal(0, 8) for _ in range(5)]
        scores = [max(0, min(100, score)) for score in scores]
        all_scores.append(scores)
    
    # Add assessment scores
    for j in range(5):
        student_data[f'A{j+1}'] = [scores[j] for scores in all_scores]
    
    return pd.DataFrame(student_data)


def main():
    """Main integration example."""
    print("SACOPOA Integrated Analysis Example")
    print("=" * 45)
    
    # Create demonstration data
    print("Creating demonstration data...")
    demo_data = create_demo_data()
    print(f"Created data for {len(demo_data)} students with realistic performance patterns")
    
    # Initialize integrated analyzer
    print("\nInitializing integrated analyzer...")
    analyzer = SACOPOAIntegratedAnalyzer()
    
    # Run traditional analysis
    print("\n" + "="*45)
    traditional_results = analyzer.run_traditional_analysis(demo_data)
    
    print("\nTraditional Analysis Results:")
    print(f"  Total Students: {traditional_results['total_students']}")
    print(f"  Total Assessments: {traditional_results['total_assessments']}")
    
    print("\n  Performance Categories:")
    for category, count in traditional_results['performance_categories'].items():
        percentage = (count / traditional_results['total_students']) * 100
        print(f"    {category}: {count} students ({percentage:.1f}%)")
    
    # Run ML analysis
    print("\n" + "="*45)
    ml_results = analyzer.run_ml_analysis(demo_data)
    
    print("\nML Analysis Results:")
    if 'predictions' in ml_results:
        pred_metrics = ml_results['predictions']['metrics']
        print(f"  Prediction Accuracy (R²): {pred_metrics.get('performance_r2', 0):.3f}")
        print(f"  Mean Absolute Error: {pred_metrics.get('performance_mae', 0):.2f}")
    
    if 'risk_assessment' in ml_results:
        risk_data = ml_results['risk_assessment']
        print(f"  High Risk Students: {len(risk_data['high_risk_students'])}")
        print(f"  Critical Risk Students: {len(risk_data['critical_risk_students'])}")
    
    if 'clustering' in ml_results:
        cluster_data = ml_results['clustering']
        print(f"  Student Groups Identified: {cluster_data['clustering_metrics']['n_clusters']}")
        print(f"  Clustering Quality: {cluster_data['clustering_metrics'].get('silhouette_score', 0):.3f}")
    
    # Generate integrated insights
    print("\n" + "="*45)
    insights = analyzer.generate_integrated_insights()
    
    print("\nIntegrated Insights:")
    
    if 'performance_insights' in insights:
        perf_insights = insights['performance_insights']
        print(f"  Performance Consistency: {'✓' if perf_insights.get('consistency', False) else '✗'}")
        print(f"  Prediction Accuracy: {perf_insights.get('prediction_accuracy', 0):.3f}")
    
    if 'risk_insights' in insights:
        risk_insights = insights['risk_insights']
        print(f"  Students at Risk: {risk_insights.get('total_at_risk', 0)} ({risk_insights.get('risk_percentage', 0):.1f}%)")
        print(f"  Immediate Action Needed: {risk_insights.get('immediate_action_needed', 0)}")
    
    if 'quality_metrics' in insights:
        quality = insights['quality_metrics']
        print(f"  Pass Rate: {quality.get('pass_rate', 0):.1f}%")
        print(f"  Excellence Rate: {quality.get('excellence_rate', 0):.1f}%")
        print(f"  Quality Index: {quality.get('quality_index', 0):.1f}")
    
    # Recommendations and alerts
    if insights.get('recommendations'):
        print(f"\n  Key Recommendations:")
        for rec in insights['recommendations'][:3]:  # Show top 3
            print(f"    • {rec}")
    
    if insights.get('alerts'):
        print(f"\n  Alerts:")
        for alert in insights['alerts']:
            print(f"    ⚠ {alert}")
    
    # Generate reports
    print("\n" + "="*45)
    print("Generating comprehensive reports...")
    
    try:
        report_files = analyzer.generate_comprehensive_report('./demo_reports/')
        
        print("\nReports Generated:")
        for report_type, filepath in report_files.items():
            print(f"  {report_type}: {filepath}")
        
        print(f"\n✓ Integration example completed successfully!")
        print(f"  - Traditional SACOPOA analysis: ✓")
        print(f"  - Advanced ML analysis: ✓")
        print(f"  - Integrated insights: ✓")
        print(f"  - Comprehensive reports: ✓")
        
    except Exception as e:
        print(f"Error generating reports: {e}")
    
    print(f"\nNext Steps:")
    print(f"  1. Review generated reports in './demo_reports/'")
    print(f"  2. Integrate this workflow into your SACOPOA application")
    print(f"  3. Customize analysis parameters for your specific needs")
    print(f"  4. Train faculty on interpreting ML insights")


if __name__ == "__main__":
    main()