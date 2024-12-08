# linear_regression_model.py
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Cargar los datos
file_path = 'C:/Users/diego/OneDrive/Escritorio/Dormammu/data-center/Base_clientes_monopoly.xlsx'
df_raw = pd.read_excel(file_path)

from data_cleaner import clean_data

df_cleaned = clean_data(df_raw)

print(df_cleaned.columns)

# Variables independientes y dependiente
X = df_cleaned[['Cuentas', 'Dualidad', 'Col_Media', 'EeccNac_Media']]
y = df_cleaned['UsoL1_Media']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
lr = LinearRegression()
lr.fit(X_train, y_train)

# Guardar el modelo entrenado como archivo .pkl
with open('models/regresion/linear_regression_model.pkl', 'wb') as file:
    pickle.dump(lr, file)

print("Modelo de regresión lineal guardado correctamente.")

y_pred_lr = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print(f"Evaluación - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
