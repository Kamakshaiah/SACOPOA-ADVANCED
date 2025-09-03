# SACOPOA Machine Learning Module

## Overview

The SACOPOA Machine Learning Module is an advanced analytics extension for the Student Assessment & CO-PO Analysis Tool. It provides comprehensive machine learning capabilities for educational data analysis, student performance prediction, risk assessment, and intervention recommendations.

## Features

### 🎯 Core ML Capabilities

- **Performance Prediction**: Predict student final grades and outcomes based on early assessments
- **Risk Assessment**: Identify students at risk of poor performance with early warning systems
- **Clustering Analysis**: Group students by performance patterns and learning behaviors  
- **Intervention Recommendations**: AI-powered suggestions for educational interventions
- **Anomaly Detection**: Identify unusual patterns in student performance data

### 📊 Advanced Analytics

- **Learning Pattern Analysis**: Identify improvement, decline, and consistency patterns
- **Feature Importance**: Understand which factors most impact student success
- **Statistical Analysis**: Comprehensive statistical metrics and correlations
- **Visualization**: Interactive charts and dashboards for insights
- **Early Warning System**: Progressive risk assessment as courses progress

## Installation

### Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

### Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Import the Module**:
   ```python
   from ml_module.sacopoa_ml import SACOPOAMLAnalyzer
   ```

## Quick Start

### Basic Analysis

```python
import pandas as pd
from ml_module.sacopoa_ml import SACOPOAMLAnalyzer

# Load your student data
student_df = pd.read_excel('student_data.xlsx')

# Initialize ML analyzer
analyzer = SACOPOAMLAnalyzer()

# Run comprehensive analysis
results = analyzer.analyze_course_data(student_df)

# Generate visualizations
visualizations = analyzer.create_visualizations('./output/')

# Export results
analyzer.export_results('analysis_results.xlsx', 'excel')
```

### Individual Component Usage

```python
from ml_module.core.risk_analyzer import RiskAnalyzer
from ml_module.utils.data_processor import DataProcessor

# Process data
processor = DataProcessor()
processed_data = processor.process_student_data(student_df)

# Risk assessment
risk_analyzer = RiskAnalyzer()
X, y = processor.prepare_for_training(processed_data['features'])
risk_analyzer.train(X, processed_data['performance_metrics']['percentages'])
risk_assessment = risk_analyzer.assess_risk(X)
```

## Data Format

### Input Data Requirements

Your Excel file should contain:

#### Student Data Sheet
| Column | Description | Example |
|--------|-------------|---------|
| Roll Number | Student identifier | 21CSE001 |
| Name | Student name | John Doe |
| A1, A2, A3... | Assessment scores | 85, 78, 92 |

#### CO-PO Matrix Sheet (Optional)
| CO | PO1 | PO2 | PO3 | PSO1 | PSO2 |
|----|-----|-----|-----|------|------|
| CO1 | 3 | 2 | 1 | 2 | 3 |
| CO2 | 2 | 3 | 2 | 1 | 2 |

## Module Components

### 1. Data Processor (`data_processor.py`)
- Data validation and cleaning
- Feature engineering
- Statistical metric calculation
- Data transformation for ML models

### 2. Performance Predictor (`predictor.py`)
- Multiple ML algorithms (Random Forest, Gradient Boosting, Linear Regression)
- Performance score prediction
- Pass/fail probability estimation
- Early prediction with confidence intervals

### 3. Risk Analyzer (`risk_analyzer.py`)
- Binary and multi-level risk classification
- Early warning system
- Risk factor analysis
- Intervention urgency calculation

### 4. Performance Clustering (`clustering.py`)
- Student grouping by performance patterns
- Learning trajectory analysis
- Course difficulty assessment
- Pattern identification (improvers, decliners, consistent performers)

### 5. Intervention Recommender (`recommender.py`)
- Personalized intervention suggestions
- Resource requirement estimation
- Implementation timeline planning
- Success probability estimation

### 6. Anomaly Detector (`anomaly_detector.py`)
- Statistical and ML-based anomaly detection
- Performance pattern anomalies
- Unusual student behavior identification
- Data quality assessment

### 7. Visualization (`visualization.py`)
- Interactive Plotly charts
- Static matplotlib plots
- Comprehensive dashboards
- Export capabilities

## Analysis Results

### Performance Predictions
- Actual vs predicted performance scores
- Pass probability estimates
- Prediction confidence intervals
- Model accuracy metrics

### Risk Assessment
- Risk probability for each student
- Risk categories (Low, Medium, High, Critical)
- High-risk student identification
- Early warning alerts

### Clustering Analysis
- Student group identification
- Cluster characteristics
- Learning pattern analysis
- Performance trajectory classification

### Intervention Recommendations
- Personalized intervention suggestions
- Priority and urgency levels
- Resource requirements
- Implementation timelines

### Anomaly Detection
- Unusual performance patterns
- Statistical outliers
- Sudden performance changes
- Data quality issues

## Configuration Options

```python
config = {
    'predictor_model': 'random_forest',  # 'random_forest', 'gradient_boost', 'linear', 'ridge'
    'risk_threshold': 0.4,              # Performance threshold for risk (40%)
    'clustering_method': 'kmeans',       # 'kmeans', 'dbscan', 'hierarchical'
    'anomaly_method': 'isolation_forest', # 'isolation_forest', 'one_class_svm', 'statistical'
    'contamination': 0.1,               # Expected proportion of anomalies
    'viz_style': 'educational'          # 'educational', 'professional', 'minimal'
}

analyzer = SACOPOAMLAnalyzer(config)
```

## Examples

The `examples/ml_examples/` directory contains:

- `complete_analysis_example.py`: Full ML analysis workflow
- `risk_assessment_example.py`: Detailed risk assessment and early warning
- Additional specialized examples

## Testing

Run the test suite:

```bash
python tests/ml_tests/test_ml_module.py
```

## Output Files

### Visualizations
- Interactive HTML charts
- Static PNG/PDF plots for reports
- Comprehensive ML dashboard

### Reports
- Excel files with multiple sheets
- JSON format for API integration
- CSV files for further analysis

### Analysis Results
- Student-level predictions and assessments
- Class-level summary statistics
- Intervention recommendations
- Risk assessment reports

## Integration with SACOPOA

This ML module is designed to integrate seamlessly with the existing SACOPOA system:

1. **Data Compatibility**: Uses the same Excel format as SACOPOA
2. **Workflow Integration**: Can be called from SACOPOA main application
3. **Report Generation**: Outputs compatible with SACOPOA reporting system
4. **Visualization**: Charts can be embedded in SACOPOA interface

## Performance Considerations

- **Memory Usage**: Optimized for datasets up to 1000 students
- **Processing Time**: Analysis typically completes in 1-5 minutes
- **Scalability**: Can handle multiple courses simultaneously
- **Resource Requirements**: 4GB RAM recommended for large datasets

## Best Practices

### Data Quality
- Ensure consistent assessment scoring
- Handle missing data appropriately
- Validate data before analysis
- Remove obvious data entry errors

### Model Interpretation
- Consider prediction confidence levels
- Review feature importance
- Validate results with domain expertise
- Use multiple metrics for evaluation

### Intervention Planning
- Prioritize high-risk students
- Consider resource constraints
- Monitor intervention effectiveness
- Adjust strategies based on results

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Format Issues**: Check Excel file structure
3. **Memory Errors**: Reduce dataset size or increase RAM
4. **Visualization Issues**: Check browser compatibility for HTML outputs

### Performance Optimization

- Use smaller datasets for testing
- Enable multiprocessing where available
- Consider feature selection for large datasets
- Cache results for repeated analysis

## API Reference

### Main Analyzer Class

```python
class SACOPOAMLAnalyzer:
    def analyze_course_data(student_df, co_po_matrix=None)
    def generate_early_warning_report(assessment_index)
    def create_visualizations(save_dir)
    def export_results(filepath, format)
    def get_student_profile(student_index)
```

### Core Components

Each component has detailed documentation in its respective file:
- `PerformancePredictor`: Machine learning models for prediction
- `RiskAnalyzer`: Risk assessment and early warning
- `PerformanceClustering`: Student grouping and pattern analysis
- `InterventionRecommender`: Intervention suggestion system
- `AnomalyDetector`: Anomaly and outlier detection

## Future Enhancements

### Planned Features
- Deep learning models for complex pattern recognition
- Natural language processing for qualitative feedback analysis
- Real-time streaming analysis
- Advanced time series forecasting
- Integration with learning management systems

### Extensibility
The modular design allows for easy extension:
- Custom ML algorithms
- Additional visualization types
- New intervention strategies
- Domain-specific analysis modules

## Support and Contribution

### Documentation
- Comprehensive code documentation
- Example scripts and tutorials
- Best practices guide
- API reference

### Testing
- Unit tests for all components
- Integration tests
- Performance benchmarks
- Validation with real datasets

## License

This module is part of the SACOPOA system and follows the same licensing terms.

## Changelog

### Version 1.0.0
- Initial release with core ML capabilities
- Performance prediction and risk assessment
- Clustering and anomaly detection
- Comprehensive visualization system
- Integration with SACOPOA data format