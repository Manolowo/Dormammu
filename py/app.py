from flask import Flask, render_template, request
import pickle
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', prediction=None)

# Cargar el modelo de regresión lineal
with open('../models/regresion/linear_regression_model.pkl', 'rb') as file:
    regression_model = pickle.load(file)

model_svm = joblib.load('../models/clasification/SVM/svm_model.pkl')
scaler = joblib.load('../models/clasification/SVM/scaler.pkl')

@app.route('/regresion_prediction')
def regresion_prediction():
    return render_template('regresion_prediction.html', prediction1=None, prediction_svm = None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction1 = None
    prediction_svm = None
    
    if request.method == 'POST':
        # Obtener los valores del formulario
        cuentas = int(request.form['cuentas'])
        dualidad = int(request.form['dualidad'])
        col_media = float(request.form['col_media'])
        eeccnac_media = float(request.form['eeccnac_media'])
        
        # Realizar la predicción
        input_data = [[cuentas, dualidad, col_media, eeccnac_media]]
        prediction1 = regression_model.predict(input_data)[0]
        
        # Predicción con el modelo SVM
        data = [[cuentas, col_media, eeccnac_media, dualidad]]
        data_scaled = scaler.transform(data)
        prediction_svm = model_svm.predict(data_scaled)[0]
        
    return render_template('regresion_prediction.html', prediction1=prediction1, prediction_svm=prediction_svm)

if __name__ == '__main__':
    app.run(debug=True)