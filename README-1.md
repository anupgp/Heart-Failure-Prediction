Team 02 DS

Content
* [Purpose & Overview](#Purpose-&-Overview)
* [Goals & Objectives](#Goals-&-Objectives)
* [Techniques & Technologies](Techniques-&-Technologies)
* [Key Findings & Results](Key-Findings-&-Results)
* [Visuals & Feature Insights](Visuals-&-Feature-Insights)
* [Risks, Limitations & Next Steps](Risks,-Limitations-&-Next-Steps)
* [Reproducibility](Reproducibility)
* [Credits](Credits)

## Purpose & Overview
This project focuses on predicting the risk of heart failure in patients based on clinical and demographic data.

Business Problem
We are a data science team supporting healthcare decision-makers in preventing heart-related mortality. Our task is to build a model that predicts the likelihood of heart failure, so that medical professionals can intervene early for high-risk patients.

We use a dataset containing 12 clinical features and a binary target DEATH_EVENT, indicating whether the patient experienced heart failure during follow-up.

Dataset Features
Age

Anaemia

Creatinine Phosphokinase (CPK)

Diabetes

Ejection Fraction

High Blood Pressure

Platelets

Serum Creatinine

Serum Sodium

Sex

Smoking

Time (days of follow-up)

Target: DEATH_EVENT (1 = heart failure occurred, 0 = survived)

## Goals & Objectives
Build interpretable and high-performance predictive models (Logistic Regression, Random Forest).

Achieve at least 85% accuracy on test data.

Prioritize recall to minimize false negatives — we want to flag all high-risk patients.

Provide a reproducible notebook and saved model file for deployment.

Metric	Logistic Regression	Random Forest
Accuracy	0.86	0.91
Precision	...	...
Recall	...	...
F1 Score	...	...
ROC-AUC	...	...

## Techniques & Technologies
Programming Language: Python 3.11

Notebook: Jupyter

Libraries: pandas, scikit-learn, matplotlib, seaborn, joblib, shap

Preprocessing Steps:
Checked for nulls (none present)

Scaled numeric features (only for Logistic Regression)

Converted categorical variables (e.g., sex, smoking) using label encoding

Split data using train_test_split with fixed random seed

Modeling Techniques:
Logistic Regression (baseline)

Random Forest Classifier (final model)

Performance measured using confusion_matrix, classification_report, roc_auc_score

Visualized model performance using ROC curves and SHAP plots

## Key Findings & Results
The Random Forest model provided the best performance with a test accuracy of ~91%, and strong precision/recall balance.

Logistic Regression also performed well (86% accuracy), but with slightly lower recall.

Serum Creatinine, Ejection Fraction, and Age were among the most predictive features.


## Visuals & Feature Insights
Feature Importance (Random Forest): Serum Creatinine, Age, Ejection Fraction, Time

SHAP Summary Plot: Illustrates the marginal contribution of each feature to heart failure risk.

Example Visuals:

. Age vs DEATH_EVENT

. Serum Creatinine distributions

. SHAP value bee swarm


## Risks, Limitations & Next Steps
Limitations:

Relatively small dataset (299 patients) may limit generalizability.

Binary outcomes don’t capture severity of heart conditions.

No data on medications, comorbidities, or lifestyle beyond smoking.

Next Steps:

Experiment with ensemble models (e.g., XGBoost, LightGBM)

Investigate survival analysis instead of binary prediction

Deploy a prediction dashboard for clinicians

Consider oversampling (SMOTE) for balancing if imbalance increases in future datasets

## Reproducibility
............
## Credits 
Full list of us memebers ...........
