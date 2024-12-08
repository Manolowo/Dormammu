from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', prediction=None)

# Cargar el modelo de regresión lineal
with open('../models/regresion/linear_regression_model.pkl', 'rb') as file:
    regression_model = pickle.load(file)

@app.route('/regresion_prediction')
def regresion_prediction():
    return render_template('regresion_prediction.html', prediction=None)

@app.route('/predict_regresion', methods=['POST'])
def predict1():
    # Obtener valores del formulario
    cuentas = float(request.form['cuentas'])
    dualidad = float(request.form['dualidad'])
    col_media = float(request.form['col_media'])
    eeccnac_media = float(request.form['eeccnac_media'])

    # Realizar la predicción
    input_data = [[cuentas, dualidad, col_media, eeccnac_media]]
    prediction = regression_model.predict(input_data)[0]

    # Renderizar la página con el resultado
    return render_template('regresion_prediction.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)