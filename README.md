# Heart-Disease-Prediction-Project
End-to-end ML web app for heart disease risk prediction using Python, Scikit-learn, and Flask with real-time user input and model comparison.
Heart Disease Prediction System

A machine learning–based web application that predicts the risk of heart disease using patient medical data. The system analyzes important health parameters and provides an instant prediction through a simple and interactive web interface.

Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can help in timely treatment and prevention.
This project builds and compares multiple machine learning models to predict the likelihood of heart disease. The best-performing model is integrated with a Flask web application to create a complete end-to-end solution.


Technologies Used :
 Python
 Flask
 Scikit-learn
 Pandas
 NumPy
 Matplotlib / Seaborn
 HTML / CSS
 Machine Learning Workflow
 Data Collection
 Heart disease dataset with clinical parameters
 Data Preprocessing
 Handling missing values
 Encoding categorical data
 Feature scaling and normalization
 Exploratory Data Analysis (EDA)
 Correlation analysis
 Distribution plots
 Feature importance visualization
 Model Training
 Logistic Regression
 K-Nearest Neighbors (with normalization)
 Decision Tree Classifier
 Random Forest (with hyperparameter tuning)
 AdaBoost Classifier
 Model Evaluation
 Accuracy score
 Confusion matrix
 Training vs testing performance
 Cross-validation
 Model Selection
 Best-performing model saved as model.pkl
 Deployment
 Flask backend integration
 User input form via HTML
 Instant prediction output

Input Parameters

The system takes medical inputs such as:
1) Age
2) Sex
3) Chest Pain Type
4) Resting Blood Pressure
5) Cholesterol
6) Fasting Blood Sugar
7) Maximum Heart Rate
8) Exercise Induced Angina
9) ST Depression (Oldpeak)

Other clinical features

Project Structure
Heart-Disease-Prediction/
│
├── app.py                # Flask application
├── model.pkl             # Trained ML model
├── dataset/              # Dataset files
├── templates/            # HTML pages
├── static/               # CSS and assets
├── requirements.txt      # Dependencies
└── README.md

How to Run the Project :

Clone the repository:
 git clone https://github.com/your-username/Heart-Disease-Prediction.git

Navigate to the project folder
 cd Heart-Disease-Prediction

Install dependencies:
 pip install -r requirements.txt

Run the application:
 python app.py

Open in browser
 http://127.0.0.1:5000/

Objective:
 The objective of this project is to demonstrate how machine learning can be applied in the healthcare domain for early disease risk prediction and to build a complete real-world ML application from model development to deployment.

Future Enhancements:

 1) Improve accuracy using advanced models
 2) Add more medical features
 3) Deploy the application on cloud (Heroku / Render / AWS)
 4) Add user authentication and history tracking
 5) Create a mobile-friendly interface
