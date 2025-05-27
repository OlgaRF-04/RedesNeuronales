# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 📂 Cargar los datos
df = pd.read_csv('data/dataBasePrestDigital.csv', sep=';')

# 🧹 Renombrar columnas para facilitar el trabajo
df.columns = [
    'mes', 'cliente', 'estado_cliente', 'rango_edad', 'genero',
    'rango_sueldo', 'procedencia', 'canal_digital', 'transacciones_mes',
    'promedio_transacciones_3m', 'recurrencia_campania', 'frecuencia_campania',
    'tiene_tarjeta', 'prom_cons_banco_3m', 'prom_saldo_banco_3m',
    'prom_saldo_tc_3m', 'prom_saldo_prest_3m', 'sow_tc', 'sow_prestamo', 'compra_prestamo_digital'
]

# 🗓️ Convertir mes a datetime
df['mes'] = pd.to_datetime(df['mes'].astype(str), format='%Y%m')

# Reemplazar comas por puntos en columnas numéricas (si es necesario)
numeric_cols = ['transacciones_mes', 'promedio_transacciones_3m',
                'prom_cons_banco_3m', 'prom_saldo_banco_3m',
                'prom_saldo_tc_3m', 'prom_saldo_prest_3m', 'sow_tc', 'sow_prestamo']

for col in numeric_cols:
    if df[col].dtype == object:  # Solo si es tipo texto (string)
        df[col] = df[col].str.replace(',', '.').astype(float)
    else:
        df[col] = df[col].astype(float)

# 🎯 Variable objetivo: usamos prom_saldo_prest_3m como proxy del monto del préstamo
X = df[['rango_edad', 'genero', 'procedencia', 'canal_digital', 'transacciones_mes',
        'promedio_transacciones_3m', 'recurrencia_campania', 'frecuencia_campania',
        'tiene_tarjeta', 'prom_cons_banco_3m', 'prom_saldo_banco_3m',
        'sow_tc', 'sow_prestamo']]
y = df['prom_saldo_prest_3m']

# ⚙️ Preprocesamiento: codificación y normalización
categorical_cols = ['rango_edad', 'genero', 'procedencia', 'canal_digital', 'tiene_tarjeta']
numerical_cols = X.columns.drop(categorical_cols).tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)

# 🔀 Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 📈 Regresión Lineal Simple (usando transacciones_mes)
X_simple = df[['transacciones_mes']].values
y_simple = df['prom_saldo_prest_3m'].values

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)

# 📉 Graficar regresión lineal
plt.figure(figsize=(10,5))
plt.scatter(X_test_lr, y_test_lr, color='blue', label='Real')
plt.plot(X_test_lr, y_pred_lr, color='red', label='Predicción LR')
plt.title('Regresión Lineal: Transacciones vs Saldo Promedio Préstamo')
plt.xlabel('Transacciones Mensuales')
plt.ylabel('Saldo Promedio Préstamo')
plt.legend()
plt.grid(True)
plt.savefig('images/regresion_lineal.png')
plt.show()

# 🤖 Red Neuronal Artificial (ANN)
model_ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Regresión
])

model_ann.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar
history = model_ann.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_test, y_test), verbose=0)

# 📊 Evaluar resultados
loss = history.history['loss']
val_loss = history.history['val_loss']

# 📉 Gráfica de pérdida
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Evolución del error durante entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('images/loss_vs_epochs.png')
plt.show()

# 📊 Comparar predicciones vs reales
y_pred_ann = model_ann.predict(X_test).flatten()

plt.scatter(y_test, y_pred_ann)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Red Neuronal: Predicción vs Real')
plt.grid(True)
plt.savefig('images/prediccion_vs_real.png')
plt.show()