# Heart-Failure-Prediction
This project aims to analyze the Heart Failure Dataset to classify the most significant disease predictors.

REVIEWING THE DATASET 
1. What are the key variables and attributes in your dataset?
The dataset contains demographic and clinical variables related to heart health, including:  

Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
    TA: Typical Angina — chest pain with characteristic symptoms
    ATA: Atypical Angina — chest pain with some but not all typical features
    NAP: Non-Anginal Pain — chest pain not related to angina
    ASY: Asymptomatic — no chest pain but other signs of risk
RestingBP: resting blood pressure [mm Hg]
    Elevated blood pressure is a risk factor for heart disease.
Cholesterol: serum cholesterol [mm/dl]
    Higher cholesterol levels are associated with increased cardiovascular risk.
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
    Elevated blood sugar is linked to diabetes and heart disease risk.
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
    Normal: Normal ECG readings.
    ST: Presence of ST-T wave abnormalities, indicating possible ischemia.
    LVH: Signs of left ventricular hypertrophy, which can indicate increased cardiac workload.
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
    Measure of cardiovascular fitness 
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
    Downsloping can provide insights into ischemic changes.
HeartDisease: output class [1: heart disease, 0: Normal]


2. How can we explore the relationships between different variables?
To perform regression analysis, we can fit a model to explore relationships between variables, visualize the data with scatter plots to identify patterns, and evaluate the model's statistics to assess significance, helping us understand how predictors influence the target variable.

3. Are there any patterns or trends in the data that we can identify?
The age distribution appears to be relatively normal, suggesting that the ages of the subjects are spread in a balanced manner around a central value.
The majority of subjects are male, indicating a gender imbalance with more males than females in the dataset.
The "ASY" (Atypical Angina) chest pain type is predominant among the subjects, meaning that this symptom is the most common chest pain classification in the dataset.

4. Who is the intended audience?

The intended audience for this analysis and code are data analysts, data scientists, medical researchers, or students interested in predictive modeling of heart disease. It could also be aimed at healthcare professionals exploring the relationships between patient features and heart disease risk. 

5. What is the question our analysis is trying to answer?
We aim to understand the key features that contribute to heart disease risk and evaluate the predictive performance of the logistic regression model in classifying patients. 

6. Are there any specific libraries or frameworks that are well-suited to our project requirements?

Yes, several libraries and frameworks are well-suited to this project focused on data analysis, feature exploration, and predictive modeling of heart disease:

pandas: For data manipulation and cleaning.
seaborn and matplotlib: For data visualization and understanding feature distributions and relationships.
scikit-learn (sklearn): For machine learning tasks including data splitting, model training (like logistic regression), evaluation metrics, and feature importance analysis.
numpy: For numerical operations, such as calculating absolute coefficients for feature importance.



DATA VISUALIZATION

1. What are the main goals and objectives of our visualization project?

Explore Data Distributions: Visualize the distribution of key variables such as age, gender, and chest pain type to understand their patterns and prevalence within the dataset.

Identify Relationships Between Variables: Use boxplots and other visualizations to examine how predictor variables like age and resting blood pressure relate to the presence of heart disease.

Highlight Feature Importance:  Show the impact of different predictor variables on heart disease risk through visualizations of the logistic regression coefficients. 

2. How can we tailor the visualization to effectively communicate with our audience?

Use Clear and Simple Visuals: Choose straightforward plots (e.g., bar charts, boxplots) that clearly illustrate the relationships and distributions without unnecessary complexity.

Choose High-Contrast Color Schemes:  Use color palettes that are distinguishable for color-blind users, such as ColorBrewer schemes (e.g., "Set2" or "Dark2"). Avoid relying solely on color to convey important information; include patterns or labels as well.

Label Clearly and Consistently: Ensure all axes, legends, and titles are descriptive and easy to read. Use large, legible font sizes and clear labels to improve readability.

Include Descriptive Titles and Annotations:  Add contextual information directly on the visualizations to guide interpretation, especially for viewers unfamiliar with the data.

Ensure Accessibility of Digital Formats: When sharing reports or dashboards, use accessible formats like HTML or PDF with selectable text.

3. What type of visualization best suits our data and objectives (e.g., bar chart, scatter plot, heatmap)?
4. How can we iterate on our design to address feedback and make iterative improvements?
5. What best practices can we follow to promote inclusivity and diversity in our visualization design?
6. How can we ensure that our visualization accurately represents the underlying data without misleading or misinterpreting information?
7. Are there any privacy concerns or sensitive information that need to be addressed in our visualization?