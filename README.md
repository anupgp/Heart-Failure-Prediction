
# Data Science Team 2 

# 🫀 Heart Disease Prediction Model
This project uses a combined heart disease dataset to develop a predictive model for diagnosing the presence of heart disease based on clinical features. A Random Forest Classifier is trained and evaluated for performance using accuracy, precision, recall, F1, Confustion Matric. The results are further illustrated with data visualizations and analysis, supporting explainability and clinical insight.


## ✅ Purpose & Overview
The objective of this project is to:
- Develop a machine learning model for predicting heart disease.
- Identify significant predictors contributing to heart disease.
- Provide intuitive visualizations to support model interpretability for both technical and non-technical audiences.

A cleaned dataset combining five heart-related datasets serves as the input, and Random Forest Plot regression was chosen for data modeling. 

## Dataset Overview 
The final dataset contains 918 unique patient records drawn from five publicly available sources:

| Dataset             | Observations |
|---------------------|--------------|
| Cleveland           | 303          |
| Hungarian           | 294          |
| Switzerland         | 123          |
| Long Beach VA       | 200          |
| Stalog (Heart)      | 270          |

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/)  
Citation: *fedesoriano*, [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/fedesoriano/heart-failure-prediction)


## 📊 Key Variables & Feature Descriptions
| Column Name       | Description                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------|
| Age               | Age of the patient (years)                                                                   |
| Sex               | Gender (1 = Male, 0 = Female)                                                                |
| ChestPainType     | One-hot encoded; Includes TA, ATA, NAP, and ASY (first category dropped)                     |
| RestingBP         | Resting blood pressure (mm Hg)                                                               |
| Cholesterol       | Serum cholesterol (mg/dl)                                                                    |
| FastingBS         | Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)                                            |
| RestingECG        | Resting ECG results: 0 = Normal, 1 = ST Abnormality, 2 = LVH                                 |
| MaxHR             | Maximum heart rate achieved                                                                  |
| ExerciseAngina    | Exercise-induced angina (1 = Yes, 0 = No)                                                    |
| Oldpeak           | ST depression induced by exercise compared to rest                                           |
| ST_Slope          | Slope of the ST segment: 0 = Up, 1 = Flat, 2 = Down                                          |
| HeartDisease      | Target variable (1 = Heart Disease, 0 = Normal) 

## Exploratory Data Analysis 

![Summary of age and gender influence on heart disease] (figures/Age vs Heart Disease and Heart Disease vs. Gender .png)

The figure contains two subplots analyzing the relationships between age, gender, and heart disease. The first subplot, **Age vs Heart Disease Distribution**, presents a histogram illustrating the counts of individuals with and without heart disease across different age groups. It shows that people without heart disease (blue bars) are fairly evenly distributed across ages, with some peaks around ages 48 and 55. In contrast, individuals with heart disease (orange bars) tend to be concentrated in the middle age range, particularly between ages 50 and 60, with a peak around age 55. This indicates that the occurrence of heart disease generally increases with age, especially in the 50-60 age bracket. The second subplot, **Heart Disease vs Gender**, highlights that males (represented by '1') have a significantly higher count of heart disease cases—458—compared to females, who have only 50 cases. Males without heart disease number 143, suggesting a higher prevalence of heart disease among males in this dataset. Overall, the data suggests that heart disease prevalence rises with age and is more common in males than females.


![Cholesterol Distribution by Gender and Patient Group] (figures/cholesterol-male-vs-female-mti.png)

The top left plot displays the density and distribution of cholesterol levels for male patients, with the accompanying histogram illustrating the frequency of specific cholesterol values. In the top right plot, a similar density and histogram representation is provided for another subgroup of male patients, highlighting potential variations within this group. The bottom left plot shows the cholesterol distribution for female patients, with a pink-shaded histogram overlay indicating the spread of cholesterol levels within this group. Lastly, the bottom right plot presents the density and histogram of cholesterol levels for female patients, emphasizing differences compared to the male groups. The use of light blue and pink shades helps differentiate between the various patient groups or categories within the dataset.


! [Resting BP distribution by Gender and Patient Group] (figures/resting-bp-male-vs-female-mti.png)


## 🧪 Model Development

The model selection process follows these steps: 

1. **Data Cleaning & Feature Encoding**: Categorical values are encoded using mapping and one-hot encoding.  
2. **Scaling**: Features are standardized using `StandardScaler` for model fitting.  
3. **Train-Test Split**: Data is split into train (80%) and test (20%) sets with stratification.  
4. **Model Training**: A `LogisticRegression` model is used in a `Pipeline`.  
5. **Hyperparameter Tuning**: Performed using `GridSearchCV` with 5-fold cross-validation



## 📈 Model Evaluation


- **Confusion Matrix**:

  |               | Predicted 0 | Predicted 1 |
  |---------------|-------------|-------------|
  | Actual 0      |     68      |     14      |
  | Actual 1      |     8       |     94      |

- **Classification Report**:

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.870 |
| Precision | 0.875 |
| Recall    | 0.892 |
| F1 Score  | 0.883 |
| ROC AUC   | 0.931 |

## 🔍 Exploratory Data Analysis

Key trends and insights:
- **Age**: Normally distributed across the population.  
- **Sex Distribution**: Majority male, indicating gender imbalance.  
- **Chest Pain Type**: Most common is 'ASY' (Asymptomatic).  
- **Feature Correlation**: Oldpeak and ST_Slope show notable associations with heart disease.

## 🧠 Feature Importance

Planned visualizations:
- Bar graph visualization for basic statistical analysis   
- Heatmap showing correlations among features  
- Boxplots comparing key variables across disease status groups

## 🎨 Data Visualization Objectives

1. **Explore Data Distributions**  
   - Age, sex, chest pain by disease outcome

2. **Examine Feature Relationships**  
   - Boxplots (e.g., Oldpeak vs HeartDisease)  
   - Correlation heatmap

3. **Showcase Feature Importance**  
   - Bar chart of logistic regression coefficients  
   - Interactive dashboard using Plotly Dash

## 🎯 Audience

This project is designed for:
- **Data scientists & analysts** in healthcare  
- **Medical researchers** and healthcare professionals  
- **Machine learning students**  
- Anyone interested in clinical decision support tools

## 📚 Tech Stack & Libraries

| Library        | Purpose                                  |
|----------------|------------------------------------------|
| `pandas`       | Data manipulation                        |
| `numpy`        | Numerical computations                   |
| `seaborn`, `matplotlib` | Data visualization                 |
| `scikit-learn` | Modeling, evaluation, hyperparameter tuning |


## Ethical Considerations

- The dataset is anonymized, posing low privacy risks.  
- Ensure visualizations are inclusive (colorblind-friendly palettes, clear labeling).  
- Communicate model limitations and avoid overreliance in a real-world clinical setting.

## Future Enhancements

- Implement more complex models such as Random Forest or XGBoost for comparison.  
- Deploy a real-time prediction interface using Streamlit or Dash.  
- Perform feature selection or dimensionality reduction using PCA or Lasso.

## References

- **UCI Datasets**: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/  
- **Kaggle Dataset (Fedesoriano)**: https://www.kaggle.com/fedesoriano/heart-failure-prediction

## Links to Individual Recordings 

- Anup Pillai 
- Fatema Banihashem 
- Mohd Tazim Ishraque 
- Khoren Avetisyan 
- Khrystyna Platko 1 
