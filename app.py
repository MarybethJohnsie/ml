from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

# Загружаем сохранённый sklearn Pipeline
model = load('test_mode.joblib')

cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Собираем значения из формы
    int_features = [x for x in request.form.values()]

    # DataFrame с правильными именами колонок
    data_unseen = pd.DataFrame([int_features], columns=cols)

    # Приводим числовые колонки к числам
    data_unseen['age'] = pd.to_numeric(data_unseen['age'])
    data_unseen['bmi'] = pd.to_numeric(data_unseen['bmi'])
    data_unseen['children'] = pd.to_numeric(data_unseen['children'])

    # Предсказание через sklearn (без PyCaret)
    prediction = model.predict(data_unseen)
    prediction = int(round(float(prediction[0])))

    return render_template('home.html',
    pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])[cols]

    prediction = model.predict(data_unseen)
    output = float(prediction[0])
    return jsonify({"prediction": output})

if __name__ == '__main__':
    app.run(debug=True)
