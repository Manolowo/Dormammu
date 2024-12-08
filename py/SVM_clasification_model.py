import pandas as pd
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

X = df_cleaned[['Cuentas', 'Col_Media', 'EeccNac_Media', 'Dualidad']]
y = df_cleaned['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Mostrar métricas de evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado como archivo .pkl
model_filename = 'models/clasification/SVM/svm_model.pkl'
joblib.dump(model, model_filename)

print(f'Modelo guardado como {model_filename}')

# Guardar también el scaler
scaler_filename = 'models/clasification/SVM/scaler.pkl'
joblib.dump(scaler, scaler_filename)

print(f'Scaler guardado como {scaler_filename}')
