# 🧠 Predicción de Monto de Préstamos Digitales en Perú

> Proyecto de análisis predictivo aplicando regresión lineal y redes neuronales artificiales (ANN) para predecir el monto promedio de préstamos digitales utilizando datos de clientes peruanos.

---

## 🎯 Objetivo

El objetivo principal de este proyecto es **predecir el monto promedio de préstamos digitales** concedidos a clientes bancarios en Perú, utilizando técnicas de machine learning como:

- Regresión Lineal Simple
- Redes Neuronales Artificiales (ANN)

Este modelo puede ser utilizado por instituciones financieras para identificar patrones de consumo, evaluar riesgos crediticios y personalizar ofertas de productos financieros.

---

## 📁 Dataset

### Fuente:
Archivo local: `data/dataBasePrestDigital.csv`

### Descripción:
Contiene información transaccional y demográfica de clientes peruanos con datos históricos mensuales sobre su comportamiento digital, consumo financiero y posesión de productos.

### Variables seleccionadas:
| Variable | Tipo | Descripción |
|---------|------|-------------|
| `cliente` | int | ID único del cliente |
| `mes` | datetime | Mes de la observación |
| `transacciones_mes` | float | Transacciones digitales en el mes |
| `promedio_transacciones_3m` | float | Promedio de transacciones últimos 3 meses |
| `prom_cons_banco_3m` | float | Promedio consumo bancario últimos 3 meses |
| `prom_saldo_banco_3m` | float | Promedio saldo bancario últimos 3 meses |
| `sow_tc`, `sow_prestamo` | float | Share of wallet - tarjeta crédito y préstamos |
| `rango_edad`, `genero`, `procedencia` | str | Características demográficas |
| `canal_digital`, `tiene_tarjeta` | str | Comportamiento digital |

### Variable objetivo (`y`):
Se usó `prom_saldo_prest_3m` como proxy del **monto promedio del préstamo en los últimos 3 meses**.

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Python**
- **Pandas**: Manipulación y limpieza de datos.
- **NumPy**: Operaciones numéricas.
- **Matplotlib**: Visualización de resultados.
- **Scikit-Learn**: Modelado, métricas y preprocesamiento.
- **TensorFlow/Keras**: Entrenamiento de red neuronal artificial.
- **GitHub**: Control de versiones y documentación.

---

## 🧪 Modelos Implementados

### 1. **Regresión Lineal Simple**
- **Variable independiente (`X`)**: `transacciones_mes`
- **Variable dependiente (`y`)**: `prom_saldo_prest_3m`
- **Métrica**: Gráfico visual comparativo entre valores reales y predichos.

### 2. **Red Neuronal Artificial (ANN)**
- **Arquitectura**:
  - Capa de entrada: Tamaño dinámico según variables procesadas
  - Capa oculta: 64 neuronas, activación `ReLU`
  - Capa intermedia: 32 neuronas, activación `ReLU`
  - Capa de salida: 1 neurona (regresión)
- **Función de pérdida**: `Mean Squared Error (MSE)`
- **Métrica**: `Mean Absolute Error (MAE)`
- **Optimizador**: `Adam`
- **Epochs**: 50
- **Batch Size**: 32

---

## 📈 Resultados y Visualizaciones

Las gráficas generadas durante la ejecución se guardan automáticamente en la carpeta `/images`.

### 1. `regresion_lineal.png`
- Muestra la relación entre las transacciones mensuales y el monto promedio del préstamo.
- Incluye línea de tendencia del modelo de regresión lineal.

### 2. `loss_vs_epochs.png`
- Gráfica del error (MSE) durante el entrenamiento vs validación.
- Útil para detectar overfitting o underfitting.

### 3. `prediccion_vs_real.png`
- Comparativa entre valores reales y predichos por la red neuronal.
- Ideal para validar la precisión del modelo.

---

## 🧰 Estructura del Proyecto

prestamos-digitales-peru/
│
├── data/
│ └── dataBasePrestDigital.csv # Archivo fuente de datos
├── images/ # Gráficas generadas
├── models/ # Opcional: guardar modelos entrenados
├── README.md # Documentación del proyecto
└── prestamos_analisis.py # Script principal de análisis y modelado

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Clonar el repositorio

git clone https://github.com/tu-usuario/prestamos-digitales-peru.git 
cd prestamos-digitales-peru

### 2. Instalar dependencias

pip install pandas numpy matplotlib scikit-learn tensorflow

### 3. Ejecutar el script

python prestamos_analisis.py

## 🧠 Conclusiones

- El modelo de red neuronal artificial mostró una mayor capacidad para capturar relaciones complejas entre las características del cliente y el monto del préstamo.
- La regresión lineal fue útil para explorar correlaciones iniciales y servir como baseline.
- Se recomienda seguir trabajando en:
- Selección de características más precisa.
- Optimización de hiperparámetros.
- Validación cruzada con más meses de datos.
