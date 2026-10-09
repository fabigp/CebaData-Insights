# Modelos Basados en Árboles (XGBoost y LightGBM)

### 1. Descripción Teórica y Fundamentos

**XGBoost (eXtreme Gradient Boosting)** y **LightGBM** son algoritmos de aprendizaje automático basados en la técnica de *Gradient Boosting*. En lugar de depender de un único modelo matemático complejo, estos métodos construyen un "ensamble" (conjunto) de cientos de árboles de decisión pequeños de forma secuencial. La clave de su éxito radica en que cada nuevo árbol se diseña específicamente para corregir los errores residuales del árbol inmediatamente anterior.

* **XGBoost:** Utiliza derivadas matemáticas de segundo orden para optimizar la dirección del aprendizaje y cuenta con mecanismos internos de regularización (L1 y L2) para evitar que el modelo memorice los datos (*overfitting*).
* **LightGBM:** A diferencia de los árboles tradicionales que crecen por niveles, LightGBM crece por hojas (*leaf-wise*), lo que le permite capturar patrones más complejos y ejecutarse con mayor rapidez y eficiencia de memoria.

### 2. Factibilidad y Alineación con el Reto AgroCebada 2026

**Factibilidad: Alta.** Son el estándar de la industria para datos tabulares no lineales.

* **Manejo de anomalías:** Son sumamente robustos ante datos faltantes o ruido (por ejemplo, vacíos causados por nubosidad en las imágenes satelitales).
* **Alineación con los criterios de evaluación:** La principal ventaja para este reto es que poseen un atributo nativo de **Importancia de Variables (Feature Importance)**. Esto permite extraer directamente la influencia porcentual de cada métrica, respondiendo a la perfección las preguntas guía del jurado: *"¿Qué índice es más importante en la predicción?"* y *"¿Qué variable climática tiene mayor peso?"*.



### 3. Estrategia de Implementación con el Dataset del Proyecto

Dado que el archivo `Conjunto_datos_BASICO_AgroCebada2026.csv` contiene series temporales (múltiples fechas por parcela), el modelo requiere un preprocesamiento previo de "aplanamiento".

**Paso a paso de implementación:**

1. **Feature Engineering (Ingeniería de Características):** Agrupar el dataset por `ID_POLIGONO` y extraer estadísticas descriptivas del ciclo de cultivo. Por ejemplo: `NDVI_max`, `Temperatura_mean`, `Precipitacion_sum`.
2. **Cruce de datos:** Unir estas nuevas columnas con la variable objetivo `RENDIMIENTO_T_HA` de las parcelas de entrenamiento (70% de las 197 parcelas).


3. **Entrenamiento y Extracción de Importancia (Ejemplo en Python):**

```python
import xgboost as xgb

# Configuración del modelo (controlando la profundidad para evitar sobreajuste)
modelo_xgb = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=100)
modelo_xgb.fit(X_train, y_train)

# Obtener la importancia de las variables para justificar el modelo en el reporte
importancia_variables = modelo_xgb.feature_importances_

```

# Regresión por Procesos Gaussianos (GPR)

### 1. Descripción Teórica y Fundamentos

La **Regresión por Procesos Gaussianos (Gaussian Process Regression - GPR)** es un modelo probabilístico, no paramétrico y basado en el teorema de Bayes. A diferencia de los modelos tradicionales que buscan encontrar "la fórmula matemática única" que mejor se ajuste a los datos, el GPR considera una distribución sobre *todas* las funciones posibles que podrían explicar los datos.

Funciona a través de una "Función Kernel" (como la Función de Base Radial - RBF). El Kernel evalúa la similitud entre las parcelas; la premisa es empírica: si la parcela A tiene condiciones satelitales y climáticas muy similares a la parcela B, entonces sus rendimientos agrícolas estarán altamente correlacionados.

### 2. Factibilidad y Alineación con el Reto AgroCebada 2026

**Factibilidad: Muy Alta (Especialmente estratégico para el modelo de negocio de FIRA).**

* **Eficacia en Datasets Pequeños:** El GPR tiene un desempeño excepcional en conjuntos de datos reducidos. Dado que el volumen de entrenamiento está limitado a aproximadamente 138 parcelas, algoritmos masivos como redes neuronales fallarían, mientras que el GPR extraerá el máximo valor de los datos disponibles.


* **Estimación de Incertidumbre y Política Pública:** Esta es su ventaja competitiva para el reto. El GPR no solo predice un valor fijo, sino que entrega una **distribución de probabilidad (Media y Desviación Estándar)**. Esto se vincula directamente con la sección requerida sobre *"Uso de la ciencia de datos en la inclusión financiera del sector agro"* y la evaluación de riesgos. Permite concluir en el reporte que: *“Parcelas con menor incertidumbre productiva representan un menor riesgo de impago, facilitando la viabilidad de créditos y reduciendo tasas de interés”*.



### 3. Estrategia de Implementación con el Dataset del Proyecto

A diferencia de los árboles de decisión, el GPR es sumamente sensible a la magnitud de los datos (las diferencias de escala entre la precipitación en mm y el NDVI que va de 0 a 1).

**Paso a paso de implementación:**

1. **Reducción de Dimensionalidad:** Dado que el GPR es complejo computacionalmente y no filtra el ruido automáticamente, se recomienda alimentarlo **solo** con el top de variables más importantes (previamente identificadas mediante un análisis de correlación o usando el Feature Importance de XGBoost).
2. **Escalado de Datos:** Es obligatorio estandarizar los datos (media 0, varianza 1) usando herramientas como `StandardScaler`.
3. **Entrenamiento y Predicción Probabilística (Ejemplo en Python):**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

# 1. Escalar los datos (Crucial para GPR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Definir el Kernel (Similitud base + Manejo de ruido blanco)
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
modelo_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
modelo_gpr.fit(X_train_scaled, y_train)

# 3. Predicción con medición de Incertidumbre
# Al poner return_std=True obtenemos la estimación y el riesgo asociado
predicciones_rendimiento, incertidumbre = modelo_gpr.predict(X_test_scaled, return_std=True)

```