# app.py
from flask import Flask, request, render_template, send_file
import os
import pickle
import pandas as pd
from predict import make_prediction   # import helper function

app = Flask(__name__)

# --- Load the model ---
model = None
project_dir = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(project_dir, 'model', 'random_forest_model.pkl'),
    os.path.join(project_dir, 'random_forest_model.pkl'),
]

for path in candidate_paths:
    if os.path.exists(path):
        with open(path, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"✅ Loaded model from: {path}")
        break

if model is None:
    raise FileNotFoundError(
        "❌ Model file not found. Checked paths: " + ", ".join(candidate_paths) +
        ".\nPlace `random_forest_model.pkl` in a `model/` folder or next to `app.py`."
    )

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        prediction, probability = make_prediction(request.form, model)
        return render_template('index.html',
                               prediction=prediction,
                               probability=f"{probability:.2f}" if probability is not None else None)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

@app.route('/download_model')
def download_model():
    for path in candidate_paths:
        if os.path.exists(path):
            return send_file(path, as_attachment=True,
                             download_name='random_forest_model.pkl')
    return "❌ Model file not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
