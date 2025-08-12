# SACOPOA (COPO Attainment Calculator)

SACOPOA is a software application for Course Outcome Program Outcome (COPO) attainment caliculations based on student marks.

## Features

- Input marks for multiple students
- Calculates percentage of students achieving different mark thresholds (eg. 60%, 50%, 40%)
- Displays average marks for the class
- Determines class attainment level based on performance
- User-friendly graphical interface

## Attainment Levels

The application calculates attainment levels based on the percentage of students scoring 60% or above:
- Level 3 (High): ≥ 70% of students score 60% or above
- Level 2 (Medium): ≥ 60% of students score 60% or above
- Level 1 (Low): ≥ 50% of students score 60% or above
- Level 0 (Very Low): < 50% of students score 60% or above

The advanced application has the ability to work with dynamic levels. Users can define the levels initially before analysis. 

## How to Run

1. Run setup file
3. Upload student marks (should be a Excel workbook)
4. App automatically calculate the attainments with download buttons for reports.
5. The app has implementation for slow-learner identification (single and multiple threashold values)
6. Does advanced analysis using scenarios for competitive analysis.
7. Has menus for plots (with save plot methods)
8. Fully GUI application with all necessary implementations for COPO attainment analysis. 

## Calculation Methodology

### 1. Assessment Attainment Calculation

For each assessment (A1, A2, A3, etc.), the attainment percentage is calculated as:

```
Assessment Attainment (%) = (Average Marks / Maximum Marks) × 100
```

Where:
- **Average Marks** = Sum of all student marks for that assessment ÷ Number of students
- **Maximum Marks** = Highest possible marks for that assessment

### 2. Course Outcome (CO) Attainment Calculation

CO attainment is calculated by mapping assessments to specific course outcomes:

1. **Assessment-CO Mapping**: Each assessment is mapped to one or more COs using a binary matrix (1 = mapped, 0 = not mapped)

2. **CO Attainment Formula**:
   ```
   CO Attainment (%) = Average of all mapped assessment attainments
   ```

   For example, if CO1 is mapped to assessments A1 and A3:
   ```
   CO1 Attainment = (A1 Attainment + A3 Attainment) ÷ 2
   ```

3. **Special Cases**:
   - If no assessments are mapped to a CO: CO Attainment = 0%
   - If only one assessment is mapped: CO Attainment = That assessment's attainment

### 3. Program Outcome (PO) Attainment Calculation

PO attainment is calculated using the CO-PO mapping matrix with weighted averaging:

1. **CO-PO Mapping Matrix**: Shows the relationship strength between COs and POs on a scale of 0-3:
   - 0 = No relationship
   - 1 = Low relationship
   - 2 = Medium relationship  
   - 3 = High relationship

2. **PO Attainment Formula**:
   ```
   PO Attainment (%) = Σ(CO Attainment × Mapping Weight) ÷ Σ(Mapping Weight)
   ```

   For example, if PO1 is mapped to:
   - CO1 with weight 3 (CO1 attainment = 75%)
   - CO2 with weight 2 (CO2 attainment = 80%)
   - CO3 with weight 1 (CO3 attainment = 70%)

   ```
   PO1 Attainment = (75×3 + 80×2 + 70×1) ÷ (3+2+1)
                  = (225 + 160 + 70) ÷ 6
                  = 455 ÷ 6 = 75.83%
   ```

### 4. Attainment Level Conversion

The application converts percentage attainments to categorical levels:

#### For Individual CO-PO Relationships:
- **Level 3 (High)**: ≥ 70% attainment
- **Level 2 (Medium)**: 60% ≤ attainment < 70%
- **Level 1 (Low)**: 50% ≤ attainment < 60%
- **Level 0 (Not Attained)**: < 50% attainment

#### For Final PO Attainments:
Same scale as above, applied to the calculated weighted average.

### 5. CO-PO Percentage Matrix Calculation

The percentage contribution of each CO to each PO is calculated as:

```
CO-PO Percentage = (CO Attainment × Mapping Weight) ÷ 3
```

This normalizes the contribution based on the maximum possible mapping weight (3).

### 6. Student Performance Metrics

The application also calculates comprehensive student performance statistics:

- **Total Marks**: Sum of marks across all assessments for each student
- **Percentage**: (Total Marks ÷ Maximum Possible Total) × 100
- **Class Statistics**:
  - Mean performance
  - Median performance
  - Standard deviation
  - Minimum and maximum performance
  - 25th and 75th percentiles

### 7. Data Input Requirements

The application expects three Excel sheets:

1. **Sheet 1 (Student Marks)**:
   - Columns: Roll Number, Name, A1, A2, A3, ... (assessment columns)
   - Each row represents one student

2. **Sheet 2 (CO-PO Matrix)**:
   - Rows: CO1, CO2, CO3, ... (course outcomes)
   - Columns: PO1, PO2, PSO1, PSO2, ... (program outcomes)
   - Values: 0-3 indicating relationship strength

3. **Sheet 3 (Assessment-CO Mapping)**:
   - Columns: Assessment, CO1, CO2, CO3, ...
   - Values: 1 (mapped) or 0 (not mapped)

### 8. Quality Assurance

The application includes validation mechanisms:
- Ensures all assessments have corresponding CO mappings
- Validates that CO-PO matrix dimensions match the number of COs and POs
- Checks for proper data types and ranges in all input matrices

This methodology ensures a comprehensive and standardized approach to COPO attainment calculation, providing transparency in how course and program outcomes are measured and evaluated.
