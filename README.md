version https://git-lfs.github.com/spec/v1
oid sha256:fd13fbd4cc3898c0355212bb84ef8ec9fed9a9c6b4f745242a6a05dbf81ca50c
size 17996

# SACOPOA - Student Assessment & CO-PO Analysis Tool
## Comprehensive Documentation

### Table of Contents
1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Data Input Requirements](#data-input-requirements)
4. [Main Menu Systems](#main-menu-systems)
5. [Calculation Methodologies](#calculation-methodologies)
6. [Export Features](#export-features)
7. [Multi-Course Analysis](#multi-course-analysis)
8. [Performance Analytics](#performance-analytics)
9. [Quality Metrics & Rubrics](#quality-metrics--rubrics)
10. [Advanced Analysis Features](#advanced-analysis-features)
11. [Troubleshooting](#troubleshooting)

---

## Overview

SACOPOA (Student Assessment & CO-PO Analysis) is a comprehensive educational assessment tool designed for academic institutions to analyze student performance against Course Outcomes (CO) and Program Outcomes (PO). The application provides detailed analytics, comparative analysis, and reporting capabilities for educational quality assurance and continuous improvement.

### Key Features
- **CO-PO Attainment Analysis**: Calculate and analyze Course Outcome to Program Outcome mappings
- **Multi-Course Comparison**: Compare performance across multiple courses
- **Performance Analytics**: Advanced statistical analysis and visualization
- **Quality Metrics**: Comprehensive quality assessment and rubrics
- **Export Capabilities**: Professional reports in Excel, Word, and PDF formats
- **Licensed Software**: 30-day trial with full functionality, then licensed features

---

## Installation & Setup

### System Requirements
- Windows 10/11
- Python 3.7+ (if running from source)
- Minimum 4GB RAM
- 1GB free disk space

### Quick Start (Executable)
1. Download `sacopoa.exe` from the `dist/sacopoa/` folder
2. Double-click to run the application
3. No additional installation required

### Running from Source
```bash
pip install -r requirements.txt
python sacopoa.py
```

### Required Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl
- python-docx
- reportlab
- tkinter (usually included with Python)

---

## Data Input Requirements

### Student Data Format
The application expects Excel files with the following structure:

#### Sheet 1: Student Data
| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| Roll Number | Student identifier | Text/Number | 21CSE001 |
| Name | Student name | Text | John Doe |
| A1 | Assessment 1 marks | Number | 85 |
| A2 | Assessment 2 marks | Number | 78 |
| A3 | Assessment 3 marks | Number | 92 |
| ... | Additional assessments | Number | ... |

#### Sheet 2: CO-PO Matrix
| CO | PO1 | PO2 | PO3 | ... | PSO1 | PSO2 | ... |
|----|-----|-----|-----|-----|------|------|-----|
| CO1 | 3 | 2 | 1 | ... | 2 | 3 | ... |
| CO2 | 2 | 3 | 2 | ... | 1 | 2 | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

#### Mapping Values
- **3**: Strong correlation
- **2**: Moderate correlation  
- **1**: Weak correlation
- **0**: No correlation

---

## Main Menu Systems

### 1. Single Course Analysis

#### 1.1 Basic Analysis
**Purpose**: Analyze individual course performance with fundamental metrics.

**Features**:
- Student performance calculation
- CO attainment analysis
- PO attainment calculation
- Basic statistical summaries

**Methodology**:
```
Student Total = Î£(Assessment Marks)
Student Percentage = (Total Marks / Maximum Possible) Ã— 100
CO Attainment = Average percentage of students achieving CO threshold
PO Attainment = Weighted average of CO attainments based on CO-PO mapping
```

#### 1.2 Learner Categorization
**Purpose**: Classify students based on performance thresholds.

**Categories**:
- **Excellent**: â‰¥85%
- **Good**: 70-84%
- **Average**: 55-69%
- **Below Average**: 40-54%
- **Poor**: <40%

**Methodology**:
```
For each threshold T:
  Category Count = Number of students with percentage â‰¥ T
  Category Percentage = (Category Count / Total Students) Ã— 100
```

#### 1.3 Advanced Analysis
**Purpose**: Detailed statistical analysis with multiple perspectives.

**Features**:
- PO-PSO Separation Analysis
- Comparative Analysis
- Gap Analysis
- Student Profiling
- Comprehensive Insights

**Calculations**:
```
PO Separation Analysis:
  PO_only = Î£(CO_attainment Ã— PO_mapping) / Î£(PO_mapping)
  
PSO Separation Analysis:  
  PSO_only = Î£(CO_attainment Ã— PSO_mapping) / Î£(PSO_mapping)
  
Gap Analysis:
  Performance Gap = Target Attainment - Actual Attainment
  Improvement Required = Max(0, Gap)
```

### 2. Visualization & Charts

#### 2.1 Summary Statistics
**Purpose**: Provide statistical overview of course performance.

**Metrics Calculated**:
```
Mean = Î£(Student Percentages) / Number of Students
Median = Middle value when students sorted by percentage
Standard Deviation = âˆš(Î£(xi - Î¼)Â² / N)
Quartiles = 25th, 50th, 75th percentile values
Range = Maximum - Minimum percentage
```

#### 2.2 Category Charts
**Purpose**: Visual representation of student distribution.

**Chart Types**:
- **Pie Chart**: Shows percentage distribution of categories
- **Bar Chart**: Compares category counts
- **Histogram**: Shows performance distribution

**Calculation for Pie Chart**:
```
For each category:
  Slice Angle = (Category Count / Total Students) Ã— 360Â°
  Slice Percentage = (Category Count / Total Students) Ã— 100%
```

#### 2.3 Advanced Visualizations
**Purpose**: Complex analytical charts for detailed insights.

**Types**:
- **Comparative Analysis**: Multi-threshold scenario comparison
- **Correlation Analysis**: Assessment and outcome relationships
- **Trend Analysis**: Performance progression tracking
- **Pattern Analysis**: Statistical pattern identification

### 3. Multi-Course Analysis

#### 3.1 Course Loading
**Purpose**: Load and manage multiple course datasets for comparison.

**Supported Formats**:
- Individual Excel files
- Zip archives containing multiple files
- Directory-based loading
- Batch processing capabilities

**Data Validation**:
```
For each course file:
  1. Validate Excel format
  2. Check required sheets (Student Data, CO-PO Matrix)
  3. Verify column structure
  4. Validate data types
  5. Check for missing critical data
```

#### 3.2 Course Comparison
**Purpose**: Compare performance metrics across multiple courses.

**Comparison Types**:
- **Overall Performance**: Average course performance
- **PO Comparison**: Program Outcome attainment comparison
- **PSO Comparison**: Program Specific Outcome comparison  
- **Assessment Comparison**: Assessment-wise performance analysis

**Calculation for Overall Comparison**:
```
For each course:
  Course Average = Î£(Student Percentages) / Number of Students
  Pass Rate = (Students with â‰¥40% / Total Students) Ã— 100%
  Excellence Rate = (Students with â‰¥85% / Total Students) Ã— 100%
  Quality Index = Weighted combination of metrics
```

#### 3.3 Performance Analytics
**Purpose**: Advanced statistical analysis across courses.

**Analytics Types**:

##### 3.3.1 Top Performing Courses
**Methodology**:
```
Excellence Threshold = User-defined (default 75%)
Course Score = Weighted average of:
  - Average Performance (40%)
  - Pass Rate (25%)
  - Excellence Rate (25%)
  - Consistency Score (10%)

Top Performers = Courses with Score â‰¥ Excellence Threshold
```

##### 3.3.2 Difficulty Analysis
**Methodology**:
```
Difficulty Score = 100 - Average Performance
Difficulty Categories:
  - Very High: D-Score > 50
  - High: D-Score 35-50
  - Moderate: D-Score 25-35
  - Low: D-Score 15-25
  - Very Low: D-Score â‰¤ 15
```

##### 3.3.3 Course Clustering
**Purpose**: Group courses with similar characteristics.

**Algorithm**:
```
Features = [Average Performance, Pass Rate, Excellence Rate, Std Deviation]
Normalization: Feature_norm = (Feature - Min) / (Max - Min)
Simple K-means clustering with k=3 groups:
  - High Performance Cluster
  - Medium Performance Cluster  
  - Low Performance Cluster
```

### 4. Quality Metrics & Rubrics

#### 4.1 Quality Index Calculation
**Purpose**: Comprehensive quality assessment for each course.

**Quality Index Formula**:
```
QI = Weighted sum of:
  - Average Performance (30%)
  - Pass Rate (25%)
  - Excellence Rate (20%)
  - Consistency (15%)
  - Assessment Balance (10%)

Consistency = 100 - Standard Deviation
Assessment Balance = 100 - Coefficient of Variation of Assessment Averages
```

**Quality Grades**:
- **Excellent**: QI â‰¥ 85
- **Very Good**: QI 75-84
- **Good**: QI 65-74
- **Satisfactory**: QI 55-64
- **Needs Improvement**: QI < 55

#### 4.2 Quality Distribution Analysis
**Purpose**: Analyze quality distribution across courses.

**Metrics**:
```
Grade Distribution = Count of courses in each quality grade
Quality Range = Maximum QI - Minimum QI
Average QI = Î£(Course QI) / Number of Courses
Quality Variance = Î£(QI - Average QI)Â² / Number of Courses
```

---

## Calculation Methodologies

### 1. CO Attainment Calculation

#### Direct Method
```
For each CO:
  CO_Attainment = Î£(Assessment marks for CO) / (Number of assessments Ã— Max marks per assessment)
  CO_Percentage = CO_Attainment Ã— 100
```

#### Indirect Method (from overall performance)
```
Student_CO_Attainment = Student_Total_Percentage
Course_CO_Attainment = Average of all student CO attainments
```

### 2. PO Attainment Calculation

#### Weighted Average Method
```
For each PO:
  PO_Attainment = Î£(CO_Attainment Ã— CO_PO_Mapping_Value) / Î£(CO_PO_Mapping_Value)
  
Where:
  CO_PO_Mapping_Value âˆˆ {0, 1, 2, 3}
  Only non-zero mappings are considered in calculation
```

#### Target-Based Method
```
PO_Achievement = (Actual_PO_Attainment / Target_PO_Attainment) Ã— 100
Target typically set at 60% for most institutions
```

### 3. Statistical Calculations

#### Basic Statistics
```
Mean (Î¼) = Î£(xi) / n
Variance (ÏƒÂ²) = Î£(xi - Î¼)Â² / n
Standard Deviation (Ïƒ) = âˆš(ÏƒÂ²)
Coefficient of Variation = (Ïƒ / Î¼) Ã— 100
```

#### Percentiles and Quartiles
```
For sorted data [x1, x2, ..., xn]:
Q1 = Value at position (n+1)/4
Q2 (Median) = Value at position (n+1)/2  
Q3 = Value at position 3(n+1)/4
IQR = Q3 - Q1
```

#### Correlation Analysis
```
Pearson Correlation Coefficient:
r = Î£((xi - xÌ„)(yi - È³)) / âˆš(Î£(xi - xÌ„)Â² Ã— Î£(yi - È³)Â²)

Where:
  xi, yi = individual assessment scores
  xÌ„, È³ = mean assessment scores
  r âˆˆ [-1, 1]
```

### 4. Performance Metrics

#### Pass Rate Calculation
```
Pass_Rate = (Number of students with percentage â‰¥ 40) / Total students Ã— 100
```

#### Excellence Rate Calculation  
```
Excellence_Rate = (Number of students with percentage â‰¥ 85) / Total students Ã— 100
```

#### Consistency Measure
```
Consistency = 100 - Standard_Deviation_of_Percentages
Higher consistency indicates more uniform performance
```

---

## Export Features

### 1. Single Course Reports

#### Excel Export
**Sheets Generated**:
- **Summary**: Overall course statistics
- **Student Details**: Individual student performance
- **CO Analysis**: Course outcome attainment
- **PO Analysis**: Program outcome attainment
- **Charts Data**: Data for visualization

#### Word Export
**Sections**:
- Executive Summary
- Course Overview
- Performance Analysis
- Recommendations
- Detailed Tables

#### PDF Export
**Content**:
- Professional formatted report
- Charts and graphs
- Statistical summaries
- Quality metrics

### 2. Multi-Course Reports

#### Comparison Reports
**Sheets**:
- **Overall Comparison**: Summary across all courses
- **PO Comparison**: Program outcome comparison
- **PSO Comparison**: Program specific outcome comparison
- **Assessment Comparison**: Assessment-wise analysis
- **Comprehensive Summary**: Detailed multi-course analysis

#### Analytics Reports
**Content**:
- Top performing courses analysis
- Difficulty level assessment
- Course clustering results
- Quality metrics comparison
- Improvement recommendations

---

## Advanced Analysis Features

### 1. Correlation Analysis

#### Assessment Correlation
**Purpose**: Analyze relationships between different assessments.

**Methodology**:
```
For assessments A1, A2:
Correlation_Matrix[A1][A2] = Pearson_Correlation(A1_scores, A2_scores)

Interpretation:
  |r| > 0.7: Strong correlation
  0.3 < |r| â‰¤ 0.7: Moderate correlation
  |r| â‰¤ 0.3: Weak correlation
```

#### CO-PO Correlation
**Purpose**: Analyze correlation patterns in CO-PO mappings.

**Analysis**:
```
For each PO pair (POi, POj):
  Correlation = Pearson_Correlation(COi_mappings, COj_mappings)
  
Heat map visualization shows:
  - Strong positive correlations (>0.7): Dark red
  - Moderate correlations (0.3-0.7): Orange/Yellow
  - Weak correlations (<0.3): Light colors
```

### 2. Trend Analysis

#### Performance Progression
**Purpose**: Track performance changes across assessments.

**Methodology**:
```
For each student:
  Trend_Slope = Linear_Regression_Slope(Assessment_sequence, Marks_sequence)
  
Classification:
  Improving: Slope > 0.1
  Stable: -0.1 â‰¤ Slope â‰¤ 0.1
  Declining: Slope < -0.1
```

#### Class Performance Trends
**Analysis**:
```
Assessment_Means = [Mean(A1), Mean(A2), ..., Mean(An)]
Overall_Trend = Linear_Regression_Slope([1,2,...,n], Assessment_Means)

Variability_Trend = Trend in standard deviations across assessments
```

### 3. Risk Assessment

#### Student Risk Analysis
**Purpose**: Identify students at risk of poor performance.

**Risk Factors**:
```
Risk_Score = Weighted sum of:
  - Current Performance (40%)
  - Trend Analysis (30%)
  - Consistency (20%)
  - Assessment Pattern (10%)

Risk Categories:
  High Risk: Score < 40
  Medium Risk: Score 40-60
  Low Risk: Score > 60
```

#### Course Risk Assessment
**Methodology**:
```
Course_Risk = Function of:
  - Percentage of failing students
  - Performance variability
  - Assessment difficulty patterns
  - Historical comparison (if available)
```

---

## Quality Metrics & Rubrics

### 1. Academic Quality Standards

#### Performance Bands
```
Excellent: 85-100%
  - Demonstrates exceptional understanding
  - Exceeds learning outcomes
  - Consistently high performance

Very Good: 75-84%
  - Shows strong understanding
  - Meets all learning outcomes
  - Good consistency

Good: 65-74%
  - Adequate understanding
  - Meets most learning outcomes
  - Acceptable performance

Satisfactory: 55-64%
  - Basic understanding
  - Meets minimum requirements
  - Room for improvement

Needs Improvement: <55%
  - Below minimum standards
  - Requires intervention
  - Significant improvement needed
```

#### Quality Indicators
```
Course Quality Index = Weighted average of:
  1. Learning Outcome Achievement (30%)
  2. Student Satisfaction (25%)
  3. Assessment Effectiveness (20%)
  4. Teaching Quality (15%)
  5. Resource Utilization (10%)
```

### 2. Institutional Benchmarks

#### Comparative Standards
```
Industry Standard Benchmarks:
  - Pass Rate: â‰¥80%
  - Excellence Rate: â‰¥25%
  - Average Performance: â‰¥70%
  - PO Attainment: â‰¥60%
  - Student Satisfaction: â‰¥4.0/5.0
```

#### Accreditation Requirements
```
NBA/NAAC Standards:
  - Minimum 60% attainment for each PO
  - Documented evidence of assessment
  - Continuous improvement process
  - Stakeholder feedback integration
```

---

## Troubleshooting

### Common Issues

#### 1. Data Import Problems
**Issue**: "File format not supported"
**Solution**: 
- Ensure Excel file (.xlsx format)
- Check for correct sheet names
- Verify column headers match expected format

#### 2. Calculation Errors
**Issue**: "Cannot calculate CO-PO attainment"
**Solution**:
- Verify CO-PO matrix has numeric values (0,1,2,3)
- Check for missing or invalid data
- Ensure assessment columns start with 'A'

#### 3. Export Failures
**Issue**: "Permission denied when saving"
**Solution**:
- Close any open Excel/Word files
- Check file permissions
- Try saving to different location

#### 4. Performance Issues
**Issue**: "Application running slowly"
**Solution**:
- Check file sizes (recommended <5MB per course)
- Close unnecessary applications
- Restart the application

### Data Validation Checklist

```
Before Analysis:
â˜ Student data sheet exists
â˜ CO-PO matrix sheet exists
â˜ Roll Number column present
â˜ Assessment columns (A1, A2, etc.) present
â˜ CO-PO matrix has proper structure
â˜ Numeric data in assessment columns
â˜ No empty critical cells
â˜ File size reasonable (<10MB)
```

### Technical Support

For technical issues:
1. Check this documentation first
2. Verify data format requirements
3. Review error messages carefully
4. Contact system administrator if needed

---

## Version Information

**Current Version**: 2.0.1
**Last Updated**: August 2025
**Compatibility**: Windows 10/11, Python 3.7+
**License**: Commercial (30-day trial available)

---

## Appendices

### Appendix A: Sample Data Templates
Available in `Sample_Courses/` directory:
- Sample_Course_Data.xlsx
- Sample_CO_PO_Matrix.xlsx
- Multi_Course_Bundle.zip

### Appendix B: Mathematical Formulas Reference
Detailed mathematical formulations for all calculations used in the application.

### Appendix C: Export Templates
Standard formats for all export types with field descriptions.

### Appendix D: Quality Assurance Guidelines
Best practices for data preparation, analysis, and interpretation.

---

*This documentation is maintained and updated regularly. For the latest version, please check the application's help menu or contact support.*
