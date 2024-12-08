# linear_regression_model.py
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
model_path = 'models/regresion/linear_regression_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as file:
    pickle.dump(lr, file)

print("Modelo de regresión lineal guardado correctamente.")

# Predicciones en el conjunto de prueba
y_pred_lr = lr.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred_lr)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print(f"Evaluación - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# Generar el gráfico
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación entre valores reales y predicciones')
plt.legend()

# Guardar el gráfico
graph_path = 'C:/Users/diego/OneDrive/Escritorio/Dormammu/plots/regression_plot.png'
os.makedirs(os.path.dirname(graph_path), exist_ok=True)
plt.savefig(graph_path)
print(f"Gráfico guardado en: {graph_path}")

plt.close()