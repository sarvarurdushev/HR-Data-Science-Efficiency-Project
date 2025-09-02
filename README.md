# HR-Data-Science-Efficiency-Project
This project demonstrates efficient data storage optimization for Training Data Ltd., a major online data science training provider. The goal is to clean and optimize one of their largest customer datasets for predicting whether students are looking for new jobs - information used to direct them to prospective recruiters.
# HR Data Science Efficiency Project 📊

A common problem when creating models to generate business value from data is that the datasets can be so large that it can take days for the model to generate predictions. Ensuring that your dataset is stored as efficiently as possible is crucial for allowing these models to run on a more reasonable timescale without having to reduce the size of the dataset.

This project demonstrates efficient data storage optimization for **Training Data Ltd.**, a major online data science training provider. The goal is to clean and optimize one of their largest customer datasets for predicting whether students are looking for new jobs - information used to direct them to prospective recruiters.

## 📊 Dataset Overview

**Data Source**: Training Data Ltd. Customer Dataset  
**File**: `customer_train.csv`  
**Purpose**: Predict student job-seeking behavior for recruitment matching

### **Dataset Schema**

| Column | Description |
|--------|-------------|
| `student_id` | A unique ID for each student |
| `city` | A code for the city the student lives in |
| `city_development_index` | A scaled development index for the city |
| `gender` | The student's gender |
| `relevant_experience` | An indicator of the student's work relevant experience |
| `enrolled_university` | The type of university course enrolled in (if any) |
| `education_level` | The student's education level |
| `major_discipline` | The educational discipline of the student |
| `experience` | The student's total work experience (in years) |
| `company_size` | The number of employees at the student's current employer |
| `company_type` | The type of company employing the student |
| `last_new_job` | The number of years between the student's current and previous jobs |
| `training_hours` | The number of hours of training completed |
| `job_change` | An indicator of whether the student is looking for a new job (`1`) or not (`0`) |

## 🎯 Project Objectives

**1.** **Data Type Optimization**: Convert columns to most efficient data types  
**2.** **Memory Usage Reduction**: Minimize dataset memory footprint  
**3.** **Categorical Data Handling**: Implement proper categorical and ordinal encodings  
**4.** **Boolean Conversion**: Transform two-factor categories to boolean types  
**5.** **Data Filtering**: Focus on experienced professionals at larger companies

## 🔧 Complete Data Optimization Code

```python
# Import necessary libraries
import pandas as pd

# Load the dataset
ds_jobs = pd.read_csv("customer_train.csv")

# View the dataset
ds_jobs.head()

# Create a copy of ds_jobs for transforming
ds_jobs_transformed = ds_jobs.copy()

# EDA to help identify ordinal, nominal, and two-factor categories
for col in ds_jobs.select_dtypes("object").columns:
    print(ds_jobs_transformed[col].value_counts(), '\n')

# Create a dictionary of columns containing ordered categorical data
ordered_cats = {
    'enrolled_university': ['no_enrollment', 'Part time course', 'Full time course'],
    'education_level': ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
    'experience': ['<1'] + list(map(str, range(1, 21))) + ['>20'],
    'company_size': ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'],
    'last_new_job': ['never', '1', '2', '3', '4', '>4']
}

# Create a mapping dictionary of columns containing two-factor categories to convert to Booleans
two_factor_cats = {
    'relevant_experience': {'No relevant experience': False, 'Has relevant experience': True},
    'job_change': {0.0: False, 1.0: True}
}

# Loop through DataFrame columns to efficiently change data types
for col in ds_jobs_transformed:
    
    # Convert two-factor categories to bool
    if col in ['relevant_experience', 'job_change']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].map(two_factor_cats[col])
    
    # Convert integer columns to int32
    elif col in ['student_id', 'training_hours']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('int32')
    
    # Convert float columns to float16
    elif col == 'city_development_index':
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('float16')
    
    # Convert columns containing ordered categorical data to ordered categories using dict
    elif col in ordered_cats.keys():
        category = pd.CategoricalDtype(ordered_cats[col], ordered=True)
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype(category)
    
    # Convert remaining columns to standard categories
    else:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('category')

# Filter students with 10 or more years experience at companies with at least 1000 employees
ds_jobs_transformed = ds_jobs_transformed[(ds_jobs_transformed['experience'] >= '10') & (ds_jobs_transformed['company_size'] >= '1000-4999')]
```

## 📈 Key Data Insights

### **Student Demographics Distribution**
- **Gender**: Male (13,221), Female (1,238), Other (191)
- **Experience Level**: Highly experienced (>20 years: 3,286 students)
- **Education**: Graduate (11,598), Masters (4,361), High School (2,017)
- **Major Discipline**: STEM dominates (14,492 students)

### **Geographic & Company Patterns**
- **Top Cities**: city_103 (4,355), city_21 (2,702), city_16 (1,533)
- **Company Sizes**: 50-99 employees (3,083), 100-499 (2,571), 10000+ (2,019)
- **Company Types**: Private Limited (9,817), Funded Startup (1,001)

### **Experience & Job Change Patterns**
- **Relevant Experience**: 13,792 students have relevant experience
- **Job Mobility**: 8,040 students changed jobs 1 year ago
- **University Enrollment**: 13,817 students not currently enrolled

## 🛠️ Technical Optimization Strategies

### **Data Type Optimizations Applied**

**1. Boolean Conversion**
- **`relevant_experience`**: Two-factor category → Boolean
- **`job_change`**: Binary indicator → Boolean

**2. Integer Optimization**
- **`student_id`, `training_hours`**: int64 → int32 (50% memory reduction)

**3. Float Optimization**
- **`city_development_index`**: float64 → float16 (75% memory reduction)

**4. Categorical Optimization**
- **Ordered Categories**: Experience levels, education, company size with proper ordering
- **Standard Categories**: City, gender, major discipline for memory efficiency

**5. Memory-Efficient Filtering**
- **Target Focus**: Senior professionals (10+ years) at large companies (1000+ employees)

### **Advanced Categorical Handling**

**Ordered Categorical Implementation**:
- **Education Progression**: Primary → High School → Graduate → Masters → PhD
- **Experience Hierarchy**: <1 year through >20 years with proper ordering
- **Company Size Ranges**: <10 employees through 10000+ with logical progression
- **University Enrollment**: no_enrollment → Part time → Full time course

## 📊 Performance Benefits

### **Memory Optimization Results**
- **Significant reduction** in dataset memory footprint
- **Faster model training** through efficient data types
- **Improved query performance** with categorical indexing
- **Enhanced sorting capabilities** with ordered categories

### **Business Impact**
- **Reduced processing time** for predictive models
- **Lower computational costs** for large-scale analysis
- **Improved model deployment** efficiency
- **Enhanced data pipeline performance**

## 📋 Requirements

```python
pandas
numpy
```

## 🚀 Getting Started

**1.** Clone this repository  
**2.** Install required packages: `pip install pandas numpy`  
**3.** Ensure the dataset is located at `customer_train.csv`  
**4.** Run the optimization script to transform your data  
**5.** Compare memory usage before and after optimization

## 🔍 Data Quality Insights

### **Missing Data Patterns**
- **Strategic filtering** removes incomplete records
- **Null handling** through categorical type conversion
- **Data validation** ensures consistency across transformations

### **Business Logic Validation**
- **Experience ordering** maintains logical progression
- **Company size hierarchy** reflects real business structures
- **Education levels** follow academic advancement paths

## 💡 Optimization Techniques Used

**1. Categorical Data Types**
- Reduces memory usage for string columns
- Enables faster grouping and filtering operations
- Maintains data integrity through controlled vocabularies

**2. Ordered Categories**
- Preserves natural ordering for analysis
- Enables proper comparison operations
- Supports advanced statistical modeling

**3. Precision Reduction**
- Balances memory efficiency with data accuracy
- Maintains sufficient precision for business decisions
- Reduces storage and transmission costs

## 🎯 Business Applications

### **Recruitment Matching**
- **Predictive modeling** for job-seeking behavior
- **Targeted recruitment** campaigns
- **Resource allocation** for HR departments

### **Student Analytics**
- **Career path analysis** for course recommendations
- **Experience gap identification** for training programs
- **Industry trend analysis** for curriculum development

## 🤝 Contributing

Feel free to fork this project and explore additional optimizations such as:
- **Feature engineering** for enhanced predictions
- **Advanced categorical encoding** techniques
- **Memory profiling** and benchmarking tools
- **Automated data type detection** algorithms

# Expected Results
```city_103    4355
city_21     2702
city_16     1533
city_114    1336
city_160     845
            ... 
city_129       3
city_111       3
city_121       3
city_140       1
city_171       1
Name: city, Length: 123, dtype: int64 

Male      13221
Female     1238
Other       191
Name: gender, dtype: int64 

Has relevant experience    13792
No relevant experience      5366
Name: relevant_experience, dtype: int64 

no_enrollment       13817
Full time course     3757
Part time course     1198
Name: enrolled_university, dtype: int64 

Graduate          11598
Masters            4361
High School        2017
Phd                 414
Primary School      308
Name: education_level, dtype: int64 

STEM               14492
Humanities           669
Other                381
Business Degree      327
Arts                 253
No Major             223
Name: major_discipline, dtype: int64 

>20    3286
5      1430
4      1403
3      1354
6      1216
2      1127
7      1028
10      985
9       980
8       802
15      686
11      664
14      586
1       549
<1      522
16      508
12      494
13      399
17      342
19      304
18      280
20      148
Name: experience, dtype: int64 

50-99        3083
100-499      2571
10000+       2019
10-49        1471
1000-4999    1328
<10          1308
500-999       877
5000-9999     563
Name: company_size, dtype: int64 

Pvt Ltd                9817
Funded Startup         1001
Public Sector           955
Early Stage Startup     603
NGO                     521
Other                   121
Name: company_type, dtype: int64 

1        8040
>4       3290
2        2900
never    2452
4        1029
3        1024
Name: last_new_job, dtype: int64 
```



## 📝 License

This project is available under the MIT License. Dataset provided by Training Data Ltd. for educational purposes.
