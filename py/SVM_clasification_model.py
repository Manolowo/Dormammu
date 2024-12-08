import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar los datos
file_path = 'C:/Users/diego/OneDrive/Escritorio/Dormammu/data-center/Base_clientes_monopoly.xlsx'
df_raw = pd.read_excel(file_path)

from data_cleaner import clean_data

df_cleaned = clean_data(df_raw)

print(df_cleaned.columns)

# Calcular el umbral como la media de 'UsoL1_Media'
umbral = df_cleaned['UsoL1_Media'].mean()

# Crear la nueva variable objetivo
df_cleaned['Target'] = (df_cleaned['UsoL1_Media'] > umbral).astype(int)

# Variables independientes y objetivo
X = df_cleaned[['Cuentas', 'Col_Media', 'EeccNac_Media', 'Dualidad']]
y = df_cleaned['Target']

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Mostrar métricas de evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado
model_filename = 'models/clasification/SVM/svm_model.pkl'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
joblib.dump(model, model_filename)
print(f'Modelo guardado como {model_filename}')

# Guardar el scaler
scaler_filename = 'models/clasification/SVM/scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f'Scaler guardado como {scaler_filename}')

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')

# Guardar el gráfico de la matriz de confusión
graph_path = 'C:/Users/diego/OneDrive/Escritorio/Dormammu/plots/confusion_matrix_svm.png'
os.makedirs(os.path.dirname(graph_path), exist_ok=True)
plt.savefig(graph_path)
print(f"Gráfico guardado en: {graph_path}")

plt.close()