# predict.py
import pandas as pd

def make_prediction(form_data, model):
    """
    Convert form inputs into a DataFrame and run prediction.
    Returns (prediction, probability).
    """
    # Collect inputs safely
    inputs = [
        float(form_data.get('age', 0)),
        int(form_data.get('sex', 0)),
        int(form_data.get('cp', 0)),
        float(form_data.get('trestbps', 0)),
        float(form_data.get('chol', 0)),
        int(form_data.get('fbs', 0)),
        int(form_data.get('restecg', 0)),
        float(form_data.get('thalach', 0)),
        int(form_data.get('exang', 0)),
        float(form_data.get('oldpeak', 0)),
        int(form_data.get('slope', 0)),
        int(form_data.get('ca', 0)),
        int(form_data.get('thal', 0))
    ]

    # Build DataFrame
    input_data = pd.DataFrame(
        [inputs],
        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                 'ca', 'thal']
    )

    # Predict
    prediction = model.predict(input_data)[0]

    # Probability (if supported)
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0].max()

    return prediction, probability