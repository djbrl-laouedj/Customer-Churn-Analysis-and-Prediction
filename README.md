# Customer Churn Prediction

## 🔍 Project Overview

**Customer churn** is a major challenge for many businesses, especially in the banking sector.
This project aims to predict whether a customer is likely **to leave the company** using supervised machine learning models.

The project focuses on :

- Handling imbalanced classification problems

- Comparing multiple machine learning models

- Optimizing hyperparameters

- Evaluating models using appropriate metrics such as ROC-AUC, F1-score, and Recall

## Models Used

The following models were implemented and compared :

- Logistic Regression

- Decision Tree

- Random Forest

- Bagging Classifier

- K-Nearest Neighbors (KNN)

- XGBoost

Finally, hyperparameter optimization was performed using RandomizedSearchCV on Random Forest & XGBoost.

## Evaluation Strategy

Since customer churn is a highly imbalanced classification problem, accuracy alone is not sufficient.

The main evaluation metric used is :

- ROC-AUC (which measures the model’s ability to rank customers from most likely to churn to least likely).

Additional metrics :

- F1-score

- Recall

- Confusion Matrix

- ROC Curve

## Project Structure
```
Customer-Churn-Analysis-and-Prediction/

│── my_streamlit_app_vf.py # Streamlit interface

│── CustomerChurn_ML.ipynb # Project notebook

│── requirements.txt # Exact tested dependency versions

│── README.md # Documentation (English)

│── README_FR.md # Documentation (French)

│── .gitignore # Ignored files

└── Caixa Banco.csv # Bank Customers Data
```

## Installation

1. Clone the repository
```
git clone https://github.com/djbrl-laouedj/Customer-Churn-Analysis-and-Prediction.git
```
```
cd Customer-Churn-Analysis-and-Prediction
```

2. Create a virtual environment (recommended)
```
python -m venv venv
```
```
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

## How to Use the Project

Running the notebooks

Open the Jupyter notebooks in the notebooks/ folder

Run the cells sequentially to :

- Load and preprocess the data

- Train multiple models

- Perform hyperparameter tuning

- Evaluate and compare model performance

- Streamlit interface

## If you want to run the Streamlit demo :

**Google colab :**

Create an account: https://ngrok.com

Get your auth token: https://dashboard.ngrok.com/get-started/your-authtoken

Add the following code at the end of your script:
```
from pyngrok import ngrok
ngrok.set_auth_token("<YOUR_NGROK_TOKEN>")
```
Run Streamlit:
```
!streamlit run my_streamlit_app_vf.py &>/dev/null &
```
Expose the app:
```
public_url = ngrok.connect(8501)
public_url
```
To restart ngrok cleanly if needed:
```
from pyngrok import ngrok
try:
    ngrok.kill()
except:
    pass
```

**Vs Code :**

``````
streamlit run my_streamlit_app_vf.py
``````

## User Guide

This application is organized into **two main pages**, accessible from the left navigation menu :

<img width="398" height="250" alt="image" src="https://github.com/user-attachments/assets/538f4faa-b250-4f22-94fe-d21930a38f3d" />

### Page 1 — Customer Churn Prediction**

This page allows users to predict the churn risk of an individual customer and understand the reasoning behind the prediction.

**1. Model Selection**

<img width="358" height="302" alt="image" src="https://github.com/user-attachments/assets/cee391bf-6821-44ac-9915-281bb1fbe9d3" />

**Choose a machine learning model from the dropdown menu :**

- XGBoost

- Random Forest

- Bagging

The selected model will be used to compute the churn probability.

**2. Decision Threshold**

<img width="285" height="104" alt="image" src="https://github.com/user-attachments/assets/95c3528c-4ac4-460d-a75e-e3a00247a319" />

**Adjust the decision threshold using the slider.**

This threshold represents a business parameter used to decide whether a customer is considered at risk.

A higher threshold makes the decision more conservative.

**3. Customer Profile Input**

<img width="285" height="421" alt="image" src="https://github.com/user-attachments/assets/1e09f4ff-7517-4df5-b963-77ab34ca6358" />

<img width="272" height="469" alt="image" src="https://github.com/user-attachments/assets/0fc3a0eb-57a6-414c-844f-6c228b1b8b38" />

**Users can simulate a customer by adjusting the following parameters :**

- Credit score

- Age

- Tenure

- Account balance

- Number of products

- Estimated salary

- Country

- Gender

- Credit card ownership

- Active membership status

These inputs represent the customer’s profile used by the model to generate a prediction.

**4. Run the Prediction**

<img width="923" height="244" alt="image" src="https://github.com/user-attachments/assets/025dc7a2-fb1b-42ba-80bd-1bc8a72c1f93" />

Click the **“Analyze risk”** button to launch the prediction.

The application displays :

- Predicted churn probability

- Decision threshold

- Final decision (Low risk / High risk), based on probability vs threshold

### Prediction Explanation (Local Explainability)

After the prediction, the application explains why the model produced this result.

**Feature Impact (Local)**

<img width="490" height="329" alt="image" src="https://github.com/user-attachments/assets/b5fa2921-be8f-481b-afae-aac0a12da5ab" />

**A list of the most influential variables is displayed.**

Each variable shows :

- Its direction of impact (increasing or decreasing churn risk)

- Its relative contribution to the final prediction

This helps users understand which customer characteristics drive the churn risk.

### Model Explainability (Global – SHAP)

To go beyond individual predictions, the application provides global explainability :

<img width="889" height="416" alt="image" src="https://github.com/user-attachments/assets/c17ad917-3dc6-485d-96e9-8acc2a915901" />

**A SHAP-based feature importance visualization**

Shows which variables are most influential across the entire dataset

Helps identify structural churn drivers such as :

- Customer activity

- Age

- Number of products

- Geographic factors

This section is useful for model interpretation and strategic decision-making.

### Page 2 — Data Monitoring & EDA

This page provides a global analytical view of customer churn.

<img width="844" height="235" alt="image" src="https://github.com/user-attachments/assets/0844c7ce-c803-4b58-97c1-8e13efa86c99" />

**Key Performance Indicators (KPIs)**

- Overall churn rate

- Number of customers analyzed

- Country with the highest churn

- Most critical customer segment

### Exploratory Data Analysis

The page includes multiple visualizations :

**Global churn distribution**

<img width="921" height="414" alt="image" src="https://github.com/user-attachments/assets/76437006-137a-4ff9-b423-b9930893421d" />

**Churn rate by gender**

<img width="858" height="399" alt="image" src="https://github.com/user-attachments/assets/d87d3904-5948-43ac-8894-b17887ee1ba3" />

**Churn rate by country**

<img width="858" height="387" alt="image" src="https://github.com/user-attachments/assets/a20131c5-33e2-4958-95e8-ed23b1584ada" />

**Segment-based churn heatmaps (age × number of products)**

<img width="886" height="408" alt="image" src="https://github.com/user-attachments/assets/570d1945-2afd-44fd-ac06-b1b1cfe236b4" />

**Customer profile distributions**

<img width="904" height="481" alt="image" src="https://github.com/user-attachments/assets/bad7aac2-f129-4aac-bb76-f69ea7723a2f" />

**Summary Insights : **

<img width="912" height="186" alt="image" src="https://github.com/user-attachments/assets/ee34c908-a298-4ccc-904e-b7adddb59c13" />

A synthesis section highlights key findings, such as :

- Inactive customers have significantly higher churn rates

- Churn increases after age 40, especially with fewer products

- Certain countries show higher churn risk

- Customers with only one product are more fragile

## Notes

XGBoost automatically runs on CPU or GPU depending on the environment.

⚠️ Hyperparameter tuning can take several minutes depending on the model and hardware.

## 👤 Authors

This project was developed by **Djebril Laouedj**,

final-year student in **Big Data & Artificial Intelligence** at **ECE Paris**.
