# ForensicOps: MLOps for Forensic Audit Analytics

## Project Overview
- This MLOps project implements a forensic audit system for healthcare insurance coverage data under the Affordable Care Act (ACA). The system combines statistical analysis (Benford's Law) and ML techniques to detect potential anomalies and patterns in healthcare enrollment data across US states. 

- By analyzing changes in insurance coverage rates, enrollment numbers, and program participation before and after the ACA implementation, the project helps identify potential irregularities and provides insights into the effectiveness of healthcare reforms.

- This project implements a Machine Learning Operations (MLOps) pipeline for financial data analysis and model training. It uses Apache Airflow for workflow orchestration and MLflow for experiment tracking.

## Project Structure

```
fin/
├── code files/
│   ├── model.py
│   ├── app.py
│   └── fin2.py
├── data/
│   └── states.csv
└── airflow_home/
    ├── dags/
    │   ├── mlops_pipeline.py
    │   └── create_model.py
    ├── logs/
    ├── airflow.db
    ├── airflow.cfg
    └── webserver_config.py
```

## Features

- Automated ML pipeline using Apache Airflow
- Financial data preprocessing and analysis
- Neural Network model training and evaluation
- Experiment tracking with MLflow
- Production-grade workflow orchestration
- Benford's Law analysis for fraud detection
- Multiple ML model comparison and evaluation

## Technical Stack

- Python 3.x
- Apache Airflow
- MLflow
- Scikit-learn
- Pandas
- Neural Networks (MLPClassifier)
- Benfordslaw
- Seaborn
- Matplotlib

## Prerequisites

- Python 3.x
- Apache Airflow
- MLflow
- Required Python packages (install via pip):
  ```bash
  pip install pandas scikit-learn mlflow apache-airflow benfordslaw seaborn matplotlib
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fin
   ```

2. Set up Airflow:
   ```bash
   export AIRFLOW_HOME=/path/to/airflow_home
   airflow db init
   ```

3. Start Airflow services:
   ```bash
   airflow webserver -p 8080
   airflow scheduler
   ```

4. Start MLflow tracking server:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

## Usage

1. The ML pipeline is implemented as an Airflow DAG in `airflow_home/dags/mlops_pipeline.py`
2. The pipeline consists of four main tasks:
   - Data preprocessing
   - Data scaling
   - Data splitting
   - Model training

3. To trigger the pipeline:
   - Access the Airflow web interface at `http://localhost:8080`
   - Navigate to the DAGs view
   - Find the `ml_pipeline` DAG
   - Trigger the DAG manually or wait for scheduled execution

## Model Architecture

The project uses a Multi-layer Perceptron (MLP) classifier with the following specifications:
- 3 hidden layers (150, 100, 50 neurons)
- ReLU activation function
- Adam optimizer
- Standardized input features

## Dataset

### Context
This project analyzes health insurance coverage data from the Affordable Care Act (ACA), also known as Obamacare. The ACA was enacted in two parts:
- The Patient Protection and Affordable Care Act (signed March 23, 2010)
- The Health Care and Education Reconciliation Act (signed March 30, 2010)

### Data Source
The dataset (`data/states.csv`) provides comprehensive health insurance coverage data for all US states and the nation, including:

#### Key Variables:
- Uninsured rates (2010 and 2015)
- Health insurance coverage changes (2010-2015)
- Employer health insurance coverage (2015)
- Marketplace health insurance coverage (2016)
- Marketplace tax credits (2016)
- State Medicaid expansion status (2016)
- Medicaid enrollment (2013 and 2016)
- Medicare enrollment (2016)

#### Data Preprocessing:
The data undergoes several preprocessing steps:
- Removal of null values
- Cleaning of percentage and dollar values
- Dropping of unnecessary columns
- Conversion of categorical variables to numerical format

This dataset enables analysis of the ACA's impact on healthcare coverage across different states and various insurance programs.

## Analysis Methods

### Benford's Law Analysis
The project implements Benford's Law analysis for fraud detection and data validation:
- First digit analysis
- Second digit analysis
- Last digit analysis
- Second-to-last digit analysis

This analysis helps identify potential anomalies or irregularities in the healthcare enrollment data.

### Machine Learning Models
The project implements and compares multiple machine learning models:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes
- Multi-layer Perceptron (MLP)
- XGBoost
- AdaBoost

Each model's performance is evaluated using:
- Accuracy scores
- AUC-ROC curves
- Confusion matrices
- Classification reports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- Dataset Source: [Health Insurance Coverage Data](https://www.kaggle.com/datasets/hhs/health-insurance?resource=download&select=states.csv) from the U.S. Department of Health and Human Services (HHS) via Kaggle.