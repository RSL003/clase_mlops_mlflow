# Databricks notebook source
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # Predicción de Enfermedades del Corazón
# MAGIC
# MAGIC En este ejemplo, trabajaremos con el famoso conjunto de datos "UCI Heart Disease". Este conjunto de datos contiene un conjunto de atributos relacionados con pacientes potencialmente afectados por una enfermedad cardiovascular (ECV). Las ECV son una de las principales causas de mortalidad, pero se estima que hasta un 90% de las ECV podrían ser prevenibles. Un diagnóstico temprano podría ser esencial en la mayoría de los casos y la inteligencia artificial puede lograr este objetivo.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clasificación Binaria en Aprendizaje Automático
# MAGIC
# MAGIC ### Entrenamiento de un Clasificador Binario
# MAGIC 1. **Preparación de Datos**: Divide el conjunto de datos en entrenamiento y prueba.
# MAGIC 2. **Selección del Modelo**: Elige un modelo de clasificación binaria (ej. regresión logística).
# MAGIC 3. **Entrenamiento del Modelo**: Entrena el modelo usando una biblioteca como sklearn.
# MAGIC 4. **Ajuste de Hiperparámetros**: Mejora el rendimiento ajustando los hiperparámetros.
# MAGIC
# MAGIC ### Evaluación del Clasificador Binario
# MAGIC 1. **Matriz de Confusión**: Muestra predicciones correctas e incorrectas divididas en dos clases.
# MAGIC 2. **Precisión (Precision)**: Proporción de identificaciones positivas correctas.
# MAGIC    - `Precisión = VP / (VP + FP)`
# MAGIC 3. **Recall (Sensibilidad)**: Proporción de positivos reales identificados correctamente.
# MAGIC    - `Recall = VP / (VP + FN)`
# MAGIC 4. **Puntuación F1 (F1 Score)**: Promedio ponderado de precisión y recall.
# MAGIC    - `F1 = 2 * (Precisión * Recall) / (Precisión + Recall)`
# MAGIC 5. **Curva ROC y AUC**: Rendimiento del clasificador y medida agregada de rendimiento.
# MAGIC 6. **Validación Cruzada**: Evalúa la robustez del modelo.

# COMMAND ----------

# leemos los datos
data = pd.read_csv('../data/raw/heart.csv')

# COMMAND ----------

# Vamos a echar un ojo a los datos que tenemos
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **age**: edad en años.
# MAGIC 2. **sex**: sexo (1 = masculino; 0 = femenino).
# MAGIC 3. **cp**: tipo de dolor de pecho.
# MAGIC    - Valor 1: angina típica.
# MAGIC    - Valor 2: angina atípica.
# MAGIC    - Valor 3: dolor no anginoso.
# MAGIC    - Valor 4: asintomático.
# MAGIC 4. **trestbps**: presión arterial en reposo (en mm Hg al ingresar al hospital).
# MAGIC 5. **chol**: colesterol sérico en mg/dl.
# MAGIC 6. **fbs**: azúcar en sangre en ayunas > 120 mg/dl (1 = verdadero; 0 = falso).
# MAGIC 7. **restecg**: resultados electrocardiográficos en reposo.
# MAGIC    - Valor 0: normal.
# MAGIC    - Valor 1: con anormalidad de onda ST-T (inversiones de onda T y/o elevación o depresión del ST de > 0.05 mV).
# MAGIC    - Valor 2: muestra probable o definitiva hipertrofia ventricular izquierda según los criterios de Estes.
# MAGIC 8. **thalach**: frecuencia cardíaca máxima alcanzada.
# MAGIC 9. **exang**: angina inducida por el ejercicio (1 = sí; 0 = no).
# MAGIC 10. **oldpeak**: depresión del segmento ST inducida por el ejercicio en relación al reposo.
# MAGIC 11. **slope**: la pendiente del segmento ST durante el pico del ejercicio.
# MAGIC     - Valor 1: ascendente.
# MAGIC     - Valor 2: plano.
# MAGIC     - Valor 3: descendente.
# MAGIC 12. **ca**: número de vasos principales (0-3) coloreados por fluoroscopia.
# MAGIC 13. **thal**: 3 = normal; 6 = defecto fijo; 7 = defecto reversible.
# MAGIC 14. **target**: variable objetivo, se refiere a la presencia de enfermedad cardíaca en el paciente.

# COMMAND ----------

data.info()

# COMMAND ----------

data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC La edad media es de 54 años, los adultos tienen más probabilidades que los jóvenes de sufrir enfermedades cardiovasculares.

# COMMAND ----------

# vamos echar un ojo al target
data.target.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC El dataset esta balanceado

# COMMAND ----------

# train test split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=5)

# COMMAND ----------

# Vamos a hacer un plot de las distirbuciones de todas las columnas
train_set.hist(bins=50, figsize=(20, 15))

# COMMAND ----------

# MAGIC %md
# MAGIC Las características tienen diferentes escalas, por lo que es una buena idea realizar un standard scaling.

# COMMAND ----------

# Creeamos un pipeline para hacer un one hot encoding de las variables categoricas y además, hacemos un median target encoding para los valores nulos.
# finalmente estandarizamos las escalas de las columnas
cat_attr = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
num_attr = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr),
    ("cat", OneHotEncoder(), cat_attr)
])

# COMMAND ----------

x_train = train_set.drop("target", axis=1)
y_train = train_set.target

# COMMAND ----------

x_train_pr = full_pipeline.fit_transform(x_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Entrenamiento del modelo y evaluacion

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a probar con un simple clasificador binario como base, en este caso un Clasificador estocástico por descenso de gradiente (SGD)

# COMMAND ----------

# La forma recomendada de comenzar a usar el seguimiento de MLflow con Python es utilizar la API autolog() de MLflow.
# Con las capacidades de autologging de MLflow, una sola línea de código registra automáticamente el modelo resultante, los parámetros utilizados para crear el modelo y una puntuación del modelo.
mlflow.sklearn.autolog()

# Esta vez quiero usar un experimento específico
mlflow.set_experiment("Experimento de Enfermedad Cardíaca")

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

sgd_clf = SGDClassifier(random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Casi siempre una buena de evaluar el modelo es realizando una validacion cruzada

# COMMAND ----------

scores = cross_val_score(sgd_clf, x_train_pr, y_train, cv=3, scoring="accuracy")
scores.mean()

# COMMAND ----------

mlflow.log_metric("cv_score", scores.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC "No es una gran metrica. Creo que es mejor idea evaluar nuestro modelo utilizando otras herramientas como la matriz de confusión.

# COMMAND ----------

# es como la validacion cruzada pero te devuelve las predcciones
preds = cross_val_predict(sgd_clf, x_train_pr, y_train, cv=3)

# COMMAND ----------

# Ahora ploteamos la matriz de confusión
cm = confusion_matrix(y_train, preds)
ConfusionMatrixDisplay(cm).plot()

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a ver la precisión, recall y F1 Score

# COMMAND ----------

precision = precision_score(y_train, preds)
print("Precision: ", precision)
recall = recall_score(y_train, preds)
print("Recall: ", recall)
f1 = f1_score(y_train, preds)
print("F1 score: ", f1)

# COMMAND ----------

# Otra metrica top para los clasificadores es el roc auc score
roc_auc = roc_auc_score(y_train, preds)
print("roc auc score:", roc_auc)

# COMMAND ----------

# logueamos todas nuestras métricas
mlflow.log_metric("Precision", precision)
mlflow.log_metric("Recall", recall)
mlflow.log_metric("F1 score", f1)
mlflow.log_metric("roc auc score", roc_auc)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC Pienso que es mejor empezar a usar modelos mas potentes como un RandomForestClassifier

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

rf_clf = RandomForestClassifier(random_state=42)
rf_preds = cross_val_predict(rf_clf, x_train_pr, y_train, cv=3)

# COMMAND ----------

cm = confusion_matrix(y_train, rf_preds)
ConfusionMatrixDisplay(cm).plot()

# COMMAND ----------

precision = precision_score(y_train, rf_preds)
print("Precision: ", precision)
recall = recall_score(y_train, rf_preds)
print("Recall: ", recall)
f1 = f1_score(y_train, rf_preds)
print("F1 score: ", f1)
roc_auc = roc_auc_score(y_train, rf_preds)
print("roc auc score:", roc_auc)

# COMMAND ----------

mlflow.log_metric("Precision", precision)
mlflow.log_metric("Recall", recall)
mlflow.log_metric("F1 score", f1)
mlflow.log_metric("roc auc score", roc_auc)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC Un poco mejor que nuestro modelo baseline. vamos a entrenar un random forest usando todo el dataset y a ver que tal sale

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(x_train_pr, y_train)

# COMMAND ----------

x_test = test_set.drop("target", axis=1)
y_test = test_set.target

# COMMAND ----------

x_test_pr = full_pipeline.transform(x_test)
final_preds = forest_clf.predict(x_test_pr)

# COMMAND ----------

# Printeamos las métricas
print("Precision: ", precision_score(y_test, final_preds))
print("Recall: ", recall_score(y_test, final_preds))
print("F1 score: ", f1_score(y_test, final_preds))
print("roc auc score:", roc_auc_score(y_test, final_preds))

# COMMAND ----------

precision = precision_score(y_test, final_preds)
print("Precision: ", precision)
recall = recall_score(y_test, final_preds)
print("Recall: ", recall)
f1 = f1_score(y_test, final_preds)
print("F1 score: ", f1)
roc_auc = roc_auc_score(y_test, final_preds)
print("roc auc score:", roc_auc)

# COMMAND ----------

mlflow.log_metric("Precision", precision)
mlflow.log_metric("Recall", recall)
mlflow.log_metric("F1 score", f1)
mlflow.log_metric("roc auc score", roc_auc)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

! mlflow ui

# COMMAND ----------

# MAGIC %md
# MAGIC Este fue un ejemplo sencillo de cómo evaluar un clasificador, ¡sin embargo, el resultado es bueno! La validación cruzada es un buen método para evaluar modelos, pero como dividimos nuestro conjunto de datos en 3 partes, el modelo tuvo pocos datos para lograr buenos resultados. ¡Con el conjunto de datos completo alcanzamos un buen resultado!