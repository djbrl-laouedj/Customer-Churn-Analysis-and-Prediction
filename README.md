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



## Notes

XGBoost automatically runs on CPU or GPU depending on the environment.

⚠️ Hyperparameter tuning can take several minutes depending on the model and hardware.

## 👤 Authors

This project was developed by **Djebril Laouedj**,

final-year student in **Big Data & Artificial Intelligence** at **ECE Paris**.
