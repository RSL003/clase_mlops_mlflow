# Databricks notebook source
# MAGIC %md
# MAGIC # ❤️ Proyecto: Predicción de Enfermedades Cardiovasculares
# MAGIC
# MAGIC ## 🎯 Objetivos del Proyecto
# MAGIC
# MAGIC En este proyecto práctico aprenderás a:
# MAGIC
# MAGIC 1. **Trabajar con datos médicos reales** del UCI Heart Disease Dataset
# MAGIC 2. **Construir un pipeline completo** de Machine Learning
# MAGIC 3. **Evaluar modelos de clasificación binaria** con múltiples métricas
# MAGIC 4. **Comparar diferentes algoritmos** (SGD vs Random Forest)
# MAGIC 5. **Registrar experimentos** con MLflow de forma profesional
# MAGIC 6. **Interpretar resultados médicos** con responsabilidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ❤️ ¿Por qué es importante?
# MAGIC
# MAGIC Las **Enfermedades Cardiovasculares (ECV)** son la principal causa de muerte en el mundo:
# MAGIC
# MAGIC - 💔 Causan **17.9 millones** de muertes al año
# MAGIC - ⚠️ **90% son prevenibles** con detección temprana
# MAGIC - 🏥 Un diagnóstico temprano puede **salvar vidas**
# MAGIC - 🤖 La IA puede ayudar a **detectar patrones** que salven vidas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 El Dataset: UCI Heart Disease
# MAGIC
# MAGIC - **303 pacientes** con datos clínicos
# MAGIC - **14 características** médicas (edad, presión arterial, colesterol, etc.)
# MAGIC - **Variable objetivo**: Presencia de enfermedad cardíaca (0/1)
# MAGIC - **Tipo de problema**: Clasificación binaria
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Ejercicio Especial: Completa los Comentarios
# MAGIC
# MAGIC **🚨 IMPORTANTE**: A lo largo de este notebook verás comentarios como:
# MAGIC
# MAGIC ```python
# MAGIC # TODO: [Completa aquí] ¿Qué hace esta función?
# MAGIC ```
# MAGIC
# MAGIC **Tu misión** es completar estos comentarios explicando:
# MAGIC - ¿Qué hace el código?
# MAGIC - ¿Por qué es importante?
# MAGIC - ¿Qué resultado esperamos?
# MAGIC
# MAGIC Esto te ayudará a:
# MAGIC - ✅ Entender profundamente cada paso
# MAGIC - ✅ Practicar documentación de código
# MAGIC - ✅ Prepararte para proyectos reales
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 ¡Empecemos!

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Paso 2: Configuración de MLflow
# MAGIC
# MAGIC Configuramos el experimento donde registraremos todos nuestros entrenamientos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Paso 3: Importar Librerías
# MAGIC
# MAGIC Importamos todas las herramientas necesarias para el proyecto.

# COMMAND ----------

import mlflow
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# TODO: [Completa aquí] ¿Qué debes hacer con esta variable email?
email = 'rafaelsanchezupo@gmail.com'  # ⚠️ CAMBIAR POR TU EMAIL DE DATABRICKS

# Validación
if not email:
    print("⚠️ CAMBIAR POR TU EMAIL DE DATABRICKS")
else:
    # TODO: [Completa aquí] ¿Para qué sirve set_tracking_uri?
    mlflow.set_tracking_uri("databricks")
    
    # TODO: [Completa aquí] ¿Qué hace set_experiment?
    experiment_name = f"/Users/{email}/5-prediccion-infarto"
    mlflow.set_experiment(experiment_name)
    
    print("=" * 70)
    print("✅ MLflow configurado correctamente")
    print("=" * 70)
    print(f"📊 Experimento: {experiment_name}")
    print(f"❤️  Proyecto: Predicción de Enfermedades Cardiovasculares")
    print("=" * 70)

# COMMAND ----------

# Librerías básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: [Completa aquí] ¿Para qué sirve train_test_split? Para dividir los datos entre entranamiento y test
from sklearn.model_selection import train_test_split

# TODO: [Completa aquí] ¿Qué es un Pipeline en sklearn? Un Pipeline de sklearn encadena preprocesamiento y #modelo en un solo objeto, evita fugas de datos y simplifica el código.
from sklearn.pipeline import Pipeline

# Preprocesamiento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# TODO: [Completa aquí] ¿Qué modelos vamos a usar? Random forest para clasificacion y SGDC para #clasificacion tambien
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO: [Completa aquí] ¿Para qué sirve cross_val_score? #Para realizar validacion cruzada
from sklearn.model_selection import cross_val_score, cross_val_predict

# TODO: [Completa aquí] ¿Qué métricas usaremos y por qué? Precision, recall, f1, confusion matrix, roc auc, accuracy, classification report, roc_curve. Porque nos dan informacion sobre el modelo y como se comporta ante nuevos datos.
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve
)

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("✅ Todas las librerías importadas correctamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ❤️ Contexto del Proyecto
# MAGIC
# MAGIC ### El Problema
# MAGIC
# MAGIC En este proyecto trabajaremos con el **UCI Heart Disease Dataset**, uno de los datasets más importantes en medicina predictiva. 
# MAGIC
# MAGIC ### 📊 Datos Clave sobre ECV
# MAGIC
# MAGIC - 💔 **Principal causa de muerte** a nivel mundial
# MAGIC - 📈 **17.9 millones** de muertes anuales
# MAGIC - ⚠️ **90% son prevenibles** con detección temprana
# MAGIC - 🎯 **Diagnóstico temprano** = vidas salvadas
# MAGIC - 🤖 **IA** puede detectar patrones imperceptibles para humanos
# MAGIC
# MAGIC ### 🎯 Nuestro Objetivo
# MAGIC
# MAGIC Construir un modelo de Machine Learning que pueda **predecir la presencia de enfermedad cardiovascular** basándose en características clínicas del paciente.
# MAGIC
# MAGIC ### ⚕️ Responsabilidad Ética
# MAGIC
# MAGIC > ⚠️ **IMPORTANTE**: Este es un proyecto educativo. Los modelos médicos reales requieren:
# MAGIC > - Validación clínica exhaustiva
# MAGIC > - Aprobación regulatoria
# MAGIC > - Supervisión médica profesional
# MAGIC > - Consideraciones éticas y legales
# MAGIC
# MAGIC Nuestro objetivo es aprender las técnicas, no reemplazar el criterio médico.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Fundamentos: Clasificación Binaria
# MAGIC
# MAGIC ### 🎯 ¿Qué es Clasificación Binaria?
# MAGIC
# MAGIC Un problema donde el modelo debe elegir entre **dos clases**:
# MAGIC - ✅ Clase Positiva (1): Tiene enfermedad cardíaca
# MAGIC - ❌ Clase Negativa (0): No tiene enfermedad cardíaca
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 Pipeline de Entrenamiento
# MAGIC
# MAGIC ```
# MAGIC 1. Preparación de Datos
# MAGIC    ↓
# MAGIC 2. Selección del Modelo
# MAGIC    ↓
# MAGIC 3. Entrenamiento
# MAGIC    ↓
# MAGIC 4. Ajuste de Hiperparámetros
# MAGIC    ↓
# MAGIC 5. Evaluación
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Métricas de Evaluación
# MAGIC
# MAGIC #### 1. **Matriz de Confusión**
# MAGIC
# MAGIC |                    | Predicción: No (0) | Predicción: Sí (1) |
# MAGIC |--------------------|--------------------|--------------------|
# MAGIC | **Real: No (0)**   | TN (Verdadero Neg) | FP (Falso Positivo)|
# MAGIC | **Real: Sí (1)**   | FN (Falso Negativo)| TP (Verdadero Pos) |
# MAGIC
# MAGIC #### 2. **Precision (Precisión)**
# MAGIC ```
# MAGIC Precision = TP / (TP + FP)
# MAGIC ```
# MAGIC - "De los que predije como enfermos, ¿cuántos realmente lo están?"
# MAGIC - **Importante cuando**: Los falsos positivos son costosos
# MAGIC
# MAGIC #### 3. **Recall (Sensibilidad)**
# MAGIC ```
# MAGIC Recall = TP / (TP + FN)
# MAGIC ```
# MAGIC - "De todos los enfermos reales, ¿cuántos detecté?"
# MAGIC - **Importante cuando**: No podemos perder ningún caso positivo (medicina)
# MAGIC
# MAGIC #### 4. **F1-Score**
# MAGIC ```
# MAGIC F1 = 2 × (Precision × Recall) / (Precision + Recall)
# MAGIC ```
# MAGIC - Balance entre Precision y Recall
# MAGIC - Útil cuando las clases están desbalanceadas
# MAGIC
# MAGIC #### 5. **ROC-AUC**
# MAGIC - Curva ROC: Representa el trade-off entre True Positive Rate y False Positive Rate
# MAGIC - AUC: Área bajo la curva (0.5 = aleatorio, 1.0 = perfecto)
# MAGIC
# MAGIC #### 6. **Validación Cruzada**
# MAGIC - Divide datos en K partes (folds)
# MAGIC - Entrena K veces, cada vez con un fold diferente como test
# MAGIC - Promedia resultados para obtener estimación robusta
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 ¿Qué métrica usar?
# MAGIC
# MAGIC | Situación | Métrica Principal |
# MAGIC |-----------|-------------------|
# MAGIC | Clases balanceadas | Accuracy |
# MAGIC | No perder positivos (medicina) | **Recall** ⭐ |
# MAGIC | Evitar falsos positivos | Precision |
# MAGIC | Balance general | F1-Score |
# MAGIC | Comparar modelos | ROC-AUC |
# MAGIC
# MAGIC **En medicina, típicamente priorizamos Recall** porque es mejor detectar un caso falso positivo que perder un verdadero positivo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📁 Paso 4: Cargar y Explorar los Datos
# MAGIC
# MAGIC Cargamos el dataset y realizamos un análisis exploratorio inicial.

# COMMAND ----------

# TODO: [Completa aquí] ¿De dónde cargamos los datos y qué contiene este archivo? Contiene los datos sobre enfermedades cardiobasculares
data = pd.read_csv('../data/raw/heart.csv')

print("=" * 70)
print("❤️  DATASET CARGADO: UCI HEART DISEASE")
print("=" * 70)
print(f"📊 Número de muestras: {len(data)}")
print(f"📈 Número de características: {len(data.columns) - 1}")  # -1 porque target no es característica
print(f"🎯 Variable objetivo: target (0 = No enfermedad, 1 = Enfermedad)")
print("=" * 70)

# COMMAND ----------

# TODO: [Completa aquí] ¿Qué nos muestra head() y por qué es útil? Nos muestra las variables y las primeras lineas
print("🔍 Primeras 5 filas del dataset:\n")
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📋 Diccionario de Datos
# MAGIC
# MAGIC Cada columna representa una característica médica importante:
# MAGIC
# MAGIC #### 👤 Datos Demográficos
# MAGIC 1. **age**: Edad del paciente en años
# MAGIC 2. **sex**: Sexo (1 = masculino, 0 = femenino)
# MAGIC
# MAGIC #### 💊 Síntomas y Diagnóstico
# MAGIC 3. **cp**: Tipo de dolor de pecho (Chest Pain)
# MAGIC    - 0: Angina típica
# MAGIC    - 1: Angina atípica
# MAGIC    - 2: Dolor no anginoso
# MAGIC    - 3: Asintomático
# MAGIC
# MAGIC 4. **exang**: Angina inducida por ejercicio (1 = sí, 0 = no)
# MAGIC
# MAGIC #### 🩺 Mediciones Clínicas
# MAGIC 5. **trestbps**: Presión arterial en reposo (mm Hg)
# MAGIC 6. **chol**: Colesterol sérico (mg/dl)
# MAGIC 7. **fbs**: Azúcar en sangre en ayunas > 120 mg/dl (1 = verdadero, 0 = falso)
# MAGIC 8. **thalach**: Frecuencia cardíaca máxima alcanzada
# MAGIC
# MAGIC #### 📊 Resultados de Pruebas
# MAGIC 9. **restecg**: Resultados electrocardiográficos en reposo
# MAGIC    - 0: Normal
# MAGIC    - 1: Anormalidad de onda ST-T
# MAGIC    - 2: Hipertrofia ventricular izquierda
# MAGIC
# MAGIC 10. **oldpeak**: Depresión del segmento ST inducida por ejercicio
# MAGIC
# MAGIC 11. **slope**: Pendiente del segmento ST durante ejercicio
# MAGIC     - 0: Ascendente
# MAGIC     - 1: Plano
# MAGIC     - 2: Descendente
# MAGIC
# MAGIC 12. **ca**: Número de vasos principales coloreados por fluoroscopia (0-3)
# MAGIC
# MAGIC 13. **thal**: Resultados de prueba de talasemia
# MAGIC     - 1: Normal
# MAGIC     - 2: Defecto fijo
# MAGIC     - 3: Defecto reversible
# MAGIC
# MAGIC #### 🎯 Variable Objetivo
# MAGIC 14. **target**: Presencia de enfermedad cardíaca
# MAGIC     - 0: No enfermedad
# MAGIC     - 1: Enfermedad presente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **💡 Tip**: En medicina, cada una de estas características tiene significado clínico. Un buen data scientist debe entender el dominio del problema.

# COMMAND ----------

# TODO: [Completa aquí] ¿Qué información nos da info() sobre el dataset? Nos dice las variables, su tipo y cuantos valores nulos contienen.
print("ℹ️  Información general del dataset:\n")
data.info()

# COMMAND ----------

# TODO: [Completa aquí] ¿Qué estadísticas nos proporciona describe()?  
print("📊 Estadísticas descriptivas:\n")
data.describe()

# COMMAND ----------

#Nos describe valores estadisticos de las variables tales como la media, la desviacion estandar... para las variables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 Observaciones Clave
# MAGIC
# MAGIC **TODO: [Completa aquí] Basándote en las estadísticas, ¿qué observaciones puedes hacer sobre:**
# MAGIC - La edad media de los pacientes
# MAGIC - El rango de valores de cada característica
# MAGIC - La presencia de valores nulos
# MAGIC - Características que pueden necesitar normalización

# COMMAND ----------

# TODO: [Completa aquí] ¿Por qué es importante analizar la distribución del target?Porque nos da informacion sobre el balance del dataset
print("🎯 Distribución de la variable objetivo:\n")
target_counts = data.target.value_counts()
print(target_counts)
print(f"\nProporción:")
print(f"  - Sin enfermedad (0): {target_counts[0]/len(data)*100:.1f}%")
print(f"  - Con enfermedad (1): {target_counts[1]/len(data)*100:.1f}%")

# TODO: Calcula proporciones por clase.
# TODO: Decide si el dataset está balanceado y explica por qué importa. Lo esta, y es importante porque si no el modelo puede estar sesgado #hacia la clase mayoritaria.



# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Análisis Exploratorio Visual
# MAGIC
# MAGIC Visualicemos las relaciones entre características para entender mejor los datos.

# COMMAND ----------

# TODO: Crea visualizaciones para explorar patrones del dataset.
# Ideas:
# - Distribución de edad por diagnóstico
# - Colesterol vs presión arterial coloreado por target
# - Tipo de dolor de pecho vs diagnóstico
# - Matriz de correlación

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Distribución de edad por target
axes[0, 0].hist([data[data.target==0]['age'], data[data.target==1]['age']], 
                bins=20, label=['Sin enfermedad', 'Con enfermedad'], alpha=0.7)
axes[0, 0].set_xlabel('Edad')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Edad por Diagnóstico')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Colesterol vs Presión Arterial
scatter = axes[0, 1].scatter(data['chol'], data['trestbps'], c=data['target'], 
                             alpha=0.6, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel('Colesterol (mg/dl)')
axes[0, 1].set_ylabel('Presión Arterial en Reposo (mm Hg)')
axes[0, 1].set_title('Colesterol vs Presión Arterial')
plt.colorbar(scatter, ax=axes[0, 1], label='Target')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Tipo de dolor de pecho por diagnóstico
cp_target = pd.crosstab(data['cp'], data['target'], normalize='index') * 100
cp_target.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
axes[1, 0].set_xlabel('Tipo de Dolor de Pecho')
axes[1, 0].set_ylabel('Porcentaje (%)')
axes[1, 0].set_title('Tipo de Dolor de Pecho vs Diagnóstico')
axes[1, 0].legend(['Sin enfermedad', 'Con enfermedad'])
axes[1, 0].set_xticklabels(['Típica', 'Atípica', 'No anginoso', 'Asintomático'], rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Gráfico 4: Matriz de correlación (top 10 características)
top_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'ca', 'target']
corr_matrix = data[top_features].corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            ax=axes[1, 1], cbar_kws={'label': 'Correlación'})
axes[1, 1].set_title('Matriz de Correlación')

plt.tight_layout()
plt.show()

print("✅ Visualizaciones generadas")
print("\n💡 TODO: [Completa aquí] ¿Qué patrones observas en los gráficos?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔀 Paso 5: Dividir los Datos
# MAGIC
# MAGIC Dividimos el dataset en conjuntos de entrenamiento y prueba.

# COMMAND ----------

# TODO: Divide los datos en train y test.
# Pistas:
# - Usa train_test_split
# - Define test_size
# - Usa random_state para reproducibilidad

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

print("=" * 70)
print("🔀 DIVISIÓN DE DATOS")
print("=" * 70)
print(f"📊 Total de muestras: {len(data)}")
print(f"📚 Conjunto de entrenamiento: {len(train_set)} ({len(train_set)/len(data)*100:.1f}%)")
print(f"🧪 Conjunto de prueba: {len(test_set)} ({len(test_set)/len(data)*100:.1f}%)")
print("=" * 70)
print("✅ Datos divididos correctamente")

# COMMAND ----------

import matplotlib.pyplot as plt

# TODO: [Completa aquí] ¿Qué nos muestran los histogramas y por qué son importantes?
print("📊 Generando histogramas de distribuciones...\n")
train_set.hist(bins=50, figsize=(20, 15))
plt.suptitle('Distribución de Características - Conjunto de Entrenamiento', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()

print("💡 TODO: [Completa aquí] ¿Qué características tienen distribuciones muy diferentes?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📏 Observación Importante: Escalas Diferentes
# MAGIC
# MAGIC **TODO: [Completa aquí] ¿Por qué es un problema que las características tengan escalas diferentes?**
# MAGIC
# MAGIC Revisa columnas como `age`, `trestbps`, `chol`, `thalach` y `oldpeak`.
# MAGIC
# MAGIC **TODO: [Completa aquí] ¿Qué transformación aplicarías para que las variables numéricas sean comparables?**
# MAGIC
# MAGIC **TODO: [Completa aquí] ¿Por qué esto ayuda a algunos modelos de Machine Learning?**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Paso 6: Construir el Pipeline de Preprocesamiento
# MAGIC
# MAGIC Crearemos un pipeline que prepare los datos automáticamente.

# COMMAND ----------

# TODO: Clasifica las columnas en categóricas y numéricas.
num_attr = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_attr = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

print("📋 Clasificación de variables:")
print(f"   📊 Numéricas ({len(num_attr)}): {num_attr}")
print(f"   🏷️  Categóricas ({len(cat_attr)}): {cat_attr}")

# TODO: Crea un ColumnTransformer que combine numéricas y categóricas.
# Pista: usa OneHotEncoder para variables categóricas.
# full_pipeline = ColumnTransformer([
#     (...),
#     (...),
# ])

# TODO: [Completa aquí] ¿Qué es ColumnTransformer y por qué lo necesitamos?
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr),      # Pipeline para numéricas
    ("cat", OneHotEncoder(), cat_attr)    # One-Hot Encoding para categóricas
])

print("\n✅ Pipeline de preprocesamiento creado")
print("\n💡 El pipeline:")
print("   1. Rellena valores nulos con la mediana (numéricas)")
print("   2. Estandariza variables numéricas (media=0, std=1)")
print("   3. Aplica One-Hot Encoding a variables categóricas")
print("\n💡 TODO: [Completa aquí] ¿Por qué One-Hot Encoding para categóricas?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Paso 7: Preparar X e y
# MAGIC
# MAGIC Separamos características (X) de la variable objetivo (y).

# COMMAND ----------

# TODO: Separa características (X) y target (y) del conjunto de entrenamiento.
x_train = train_set.drop('target', axis=1)
y_train = train_set['target']

print("=" * 70)
print("🎯 SEPARACIÓN DE CARACTERÍSTICAS Y TARGET")
print("=" * 70)
print(f"X_train shape: {x_train.shape} (características)")
print(f"y_train shape: {y_train.shape} (target)")
print("=" * 70)

# COMMAND ----------

# TODO: Aplica el pipeline al conjunto de entrenamiento.
x_train_pr = full_pipeline.fit_transform(x_train)

print("🔧 Pipeline aplicado al conjunto de entrenamiento")
print(f"   Shape original: {x_train.shape}")
print(f"   Shape procesado: {x_train_pr.shape}")
print(f"\n💡 TODO: [Completa aquí] ¿Por qué cambió el número de columnas?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Paso 8: Entrenamiento y Evaluación de Modelos
# MAGIC
# MAGIC Entrenaremos dos modelos y compararemos sus resultados usando MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔵 Modelo 1: SGD Classifier (Baseline)
# MAGIC
# MAGIC **TODO: [Completa aquí] ¿Qué es SGD (Stochastic Gradient Descent) y cómo funciona?**
# MAGIC
# MAGIC Empezaremos con un modelo simple como baseline (línea base) para tener una referencia.

# COMMAND ----------

# TODO: Activa autolog de MLflow.
# mlflow.sklearn.autolog()
mlflow.sklearn.autolog()

# Iniciar experimento con MLflow
with mlflow.start_run(run_name="SGD Classifier - Baseline") as run:
    
    print("=" * 70)
    print("🚀 ENTRENANDO MODELO BASELINE: SGD CLASSIFIER")
    print("=" * 70)
    
    # TODO: [Completa aquí] ¿Qué es random_state y por qué lo usamos?
    sgd_clf = SGDClassifier(random_state=42)
    
    # TODO: [Completa aquí] ¿Qué es validación cruzada con 3 folds?
    print("\n🔄 Realizando validación cruzada (3-fold)...")
    scores = cross_val_score(sgd_clf, x_train_pr, y_train, cv=3, scoring="accuracy")
    
    print(f"\n📊 Accuracy por fold:")
    for i, score in enumerate(scores, 1):
        print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\n📈 Accuracy promedio: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Registrar métrica en MLflow
    mlflow.log_metric("cv_accuracy_mean", scores.mean())
    mlflow.log_metric("cv_accuracy_std", scores.std())
    
    print("\n✅ Modelo SGD entrenado con validación cruzada")
    print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Evaluación del Modelo SGD

# COMMAND ----------

# TODO: [Completa aquí] ¿Por qué usamos cross_val_predict en lugar de solo predecir? Para poder comparar cross validation de distintos #hiperparametros para el mismo modelo y asi elegir los hiperparametros
print("🔮 Obteniendo predicciones con validación cruzada...")
preds = cross_val_predict(sgd_clf, x_train_pr, y_train, cv=3)
print(f"✅ {len(preds)} predicciones generadas")

# COMMAND ----------

# TODO: [Completa aquí] ¿Qué nos muestra la matriz de confusión? Los falsos postivos y negativos
print("📊 Generando matriz de confusión...\n")
print("La matriz de confusión muestra cuántas predicciones fueron correctas (TN y TP)")
print("y cuántas fueron incorrectas (FP y FN), comparando las etiquetas reales con las predichas.")
print("Permite ver visualmente qué tipo de errores comete el modelo.\n")

cm = confusion_matrix(y_train, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Sin enfermedad', 'Con enfermedad'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title('Matriz de Confusión - SGD Classifier', fontsize=14, fontweight='bold')
plt.grid(False)
plt.show()

print(f"\n📋 Interpretación de la matriz:")
print(f"   TN (Verdaderos Negativos): {cm[0,0]} - Correctamente identificados como sanos")
print(f"   FP (Falsos Positivos): {cm[0,1]} - Sanos clasificados como enfermos")
print(f"   FN (Falsos Negativos): {cm[1,0]} - Enfermos clasificados como sanos ⚠️")
print(f"   TP (Verdaderos Positivos): {cm[1,1]} - Correctamente identificados como enfermos")
print(f"\n💡 TODO: [Completa aquí] ¿Qué tipo de error es más grave en medicina?")

# COMMAND ----------

# TODO: [Completa aquí] ¿Qué mide cada una de estas métricas? La precision de la prediccion, el porcentaje de verdaderos positivos detectados,el balance entre las 2 primeras y la capacidad de discrminacion del modelo.
print("=" * 70)
print("📊 MÉTRICAS DEL MODELO SGD")
print("=" * 70)

precision = precision_score(y_train, preds)
recall = recall_score(y_train, preds)
f1 = f1_score(y_train, preds)
roc_auc = roc_auc_score(y_train, preds)

print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   → De los que predije como enfermos, el {precision*100:.1f}% realmente lo están")
print(f"\n❤️  Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"   → Detecté el {recall*100:.1f}% de todos los enfermos reales")
print(f"\n⚖️  F1 Score: {f1:.4f}")
print(f"   → Balance entre precision y recall")
print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
print(f"   → Capacidad de discriminación del modelo")
print("=" * 70)

print("\n💡 TODO: [Completa aquí] ¿Qué métrica es más importante en medicina y por qué?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🌲 Modelo 2: Random Forest Classifier
# MAGIC
# MAGIC **TODO: [Completa aquí] ¿Qué es Random Forest y por qué suele funcionar mejor que modelos simples?**
# MAGIC
# MAGIC Ahora probemos un modelo más potente y comparemos resultados.

# COMMAND ----------

# Continuar con MLflow (nuevo run para Random Forest)
with mlflow.start_run(run_name="Random Forest Classifier") as run:
    
    print("=" * 70)
    print("🚀 ENTRENANDO MODELO: RANDOM FOREST CLASSIFIER")
    print("=" * 70)
    
    # TODO: [Completa aquí] ¿Qué hiperparámetros podríamos ajustar en Random Forest?
    rf_clf = RandomForestClassifier(random_state=42)
    
    # TODO: [Completa aquí] ¿Por qué volvemos a hacer validación cruzada?
    print("\n🔄 Realizando validación cruzada (3-fold)...")
    rf_scores = cross_val_score(rf_clf, x_train_pr, y_train, cv=3, scoring="accuracy")
    
    print(f"\n📊 Accuracy por fold:")
    for i, score in enumerate(rf_scores, 1):
        print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\n📈 Accuracy promedio: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    
    # Obtener predicciones
    rf_preds = cross_val_predict(rf_clf, x_train_pr, y_train, cv=3)
    
    # Registrar en MLflow
    mlflow.log_metric("cv_accuracy_mean", rf_scores.mean())
    mlflow.log_metric("cv_accuracy_std", rf_scores.std())
    
    print("\n✅ Modelo Random Forest entrenado")
    print("=" * 70)

# COMMAND ----------

# TODO: [Completa aquí] ¿Esperamos que la matriz de confusión sea mejor? ¿Por qué?
print("📊 Matriz de Confusión - Random Forest\n")

cm_rf = confusion_matrix(y_train, rf_preds)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=['Sin enfermedad', 'Con enfermedad'])
fig, ax = plt.subplots(figsize=(8, 6))
disp_rf.plot(ax=ax, cmap='Greens')
plt.title('Matriz de Confusión - Random Forest', fontsize=14, fontweight='bold')
plt.grid(False)
plt.show()

print(f"\n📋 Interpretación:")
print(f"   TN: {cm_rf[0,0]} | FP: {cm_rf[0,1]}")
print(f"   FN: {cm_rf[1,0]} | TP: {cm_rf[1,1]}")
print(f"\n💡 TODO: [Completa aquí] ¿Mejoró respecto a SGD? ¿En qué?")

# COMMAND ----------

# TODO: [Completa aquí] ¿Cómo se comparan estas métricas con las de SGD? Random Forest supera a SGD en todas #las métricas, destacando el Recall (+12.8 puntos: 82.7% vs 69.9%), lo que significa que detecta muchos más #pacientes enfermos que el modelo lineal baseline.
print("=" * 70)
print("📊 MÉTRICAS DEL MODELO RANDOM FOREST")
print("=" * 70)

rf_precision = precision_score(y_train, rf_preds)
rf_recall = recall_score(y_train, rf_preds)
rf_f1 = f1_score(y_train, rf_preds)
rf_roc_auc = roc_auc_score(y_train, rf_preds)

print(f"🎯 Precision: {rf_precision:.4f} ({rf_precision*100:.2f}%)")
print(f"❤️  Recall: {rf_recall:.4f} ({rf_recall*100:.2f}%)")
print(f"⚖️  F1 Score: {rf_f1:.4f}")
print(f"📈 ROC-AUC Score: {rf_roc_auc:.4f}")
print("=" * 70)

# Comparación con SGD
print("\n📊 COMPARACIÓN SGD vs RANDOM FOREST")
print("=" * 70)
comparison = pd.DataFrame({
    'Métrica': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'SGD': [precision, recall, f1, roc_auc],
    'Random Forest': [rf_precision, rf_recall, rf_f1, rf_roc_auc],
    'Diferencia': [
        rf_precision - precision,
        rf_recall - recall,
        rf_f1 - f1,
        rf_roc_auc - roc_auc
    ]
})
print(comparison.to_string(index=False))
print("=" * 70)

print("\n💡 TODO: [Completa aquí] ¿Qué modelo funciona mejor y por qué?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Paso 9: Entrenamiento Final y Evaluación en Test
# MAGIC
# MAGIC Ahora entrenaremos el modelo con **todo el conjunto de entrenamiento** y evaluaremos en el conjunto de prueba (que nunca ha visto).

# COMMAND ----------

# TODO: [Completa aquí] ¿Por qué ahora entrenamos con todos los datos de train? Porque es el ultimo entrenamiento antes de prediccion en test, #no un cross validation
print("🎯 Entrenamiento final con todo el conjunto de entrenamiento\n")

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(x_train_pr, y_train)

print("✅ Modelo Random Forest entrenado con", len(x_train_pr), "muestras")
print("📊 El modelo ahora tiene más datos para aprender patrones")

# COMMAND ----------

# TODO: Separa características y target del conjunto de prueba.
x_test = test_set.drop('target', axis=1)
y_test = test_set['target']

print("🧪 Conjunto de prueba preparado:")
print(f"   X_test: {x_test.shape}")
print(f"   y_test: {y_test.shape}")

# COMMAND ----------

# TODO: Transforma el conjunto de prueba y genera predicciones.
x_test_pr = full_pipeline.transform(x_test)
final_preds = forest_clf.predict(x_test_pr)

print("🔮 Predicciones en conjunto de prueba generadas")
print(f"   Datos procesados: {x_test_pr.shape}")
print(f"   Predicciones: {len(final_preds)}")
print("\n💡 TODO: [Completa aquí] ¿Por qué NO debemos usar fit_transform en test?")

# COMMAND ----------

# TODO: [Completa aquí] ¿Qué significan estos resultados en el conjunto de prueba? Que el modelo se #desempeña bien con una precision del 84%
print("=" * 70)
print("🎉 RESULTADOS FINALES EN CONJUNTO DE PRUEBA")
print("=" * 70)

final_precision = precision_score(y_test, final_preds)
final_recall = recall_score(y_test, final_preds)
final_f1 = f1_score(y_test, final_preds)
final_roc_auc = roc_auc_score(y_test, final_preds)
final_accuracy = accuracy_score(y_test, final_preds)

print(f"🎯 Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"🎯 Precision: {final_precision:.4f} ({final_precision*100:.2f}%)")
print(f"❤️  Recall: {final_recall:.4f} ({final_recall*100:.2f}%)")
print(f"⚖️  F1 Score: {final_f1:.4f}")
print(f"📈 ROC-AUC: {final_roc_auc:.4f}")
print("=" * 70)

# Matriz de confusión final
print("\n📊 Matriz de Confusión Final:\n")
cm_final = confusion_matrix(y_test, final_preds)
disp_final = ConfusionMatrixDisplay(cm_final, display_labels=['Sin enfermedad', 'Con enfermedad'])
fig, ax = plt.subplots(figsize=(8, 6))
disp_final.plot(ax=ax, cmap='YlGnBu')
plt.title('Matriz de Confusión - Conjunto de Prueba', fontsize=14, fontweight='bold')
plt.grid(False)
plt.show()

# Reporte de clasificación completo
print("\n📋 REPORTE DE CLASIFICACIÓN DETALLADO:")
print("=" * 70)
print(classification_report(y_test, final_preds, 
                          target_names=['Sin enfermedad', 'Con enfermedad'],
                          digits=4))
print("=" * 70)

print("\n💡 TODO: [Completa aquí] ¿El modelo generaliza bien? ¿Cómo lo sabes?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 ¡Proyecto Completado!
# MAGIC
# MAGIC Completa esta sección cuando termines el notebook.
# MAGIC
# MAGIC ### ✅ Comprueba que has trabajado
# MAGIC
# MAGIC 1. [ ] Exploración de datos médicos reales
# MAGIC 2. [ ] Pipeline de preprocesamiento
# MAGIC 3. [ ] Entrenamiento y comparación de modelos
# MAGIC 4. [ ] Evaluación con métricas de clasificación
# MAGIC 5. [ ] Validación cruzada
# MAGIC 6. [ ] Registro de experimentos en MLflow
# MAGIC 7. [ ] Interpretación de resultados con matrices de confusión
# MAGIC
# MAGIC ### 📊 Conclusiones Clave
# MAGIC
# MAGIC **TODO: Basándote en tus resultados, escribe tus conclusiones:**
# MAGIC
# MAGIC 1. **¿Qué modelo funcionó mejor?**
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC 2. **¿Por qué crees que funcionó mejor?**
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC 3. **¿El modelo es suficientemente bueno para uso médico?**
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC 4. **¿Qué métrica es más importante en este caso?**
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC 5. **¿Qué mejorarías del modelo?**
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎓 Conceptos Aprendidos
# MAGIC
# MAGIC #### 📚 Machine Learning
# MAGIC - Clasificación binaria
# MAGIC - Validación cruzada
# MAGIC - Train/test split
# MAGIC - Overfitting vs generalización
# MAGIC
# MAGIC #### 🔧 Preprocesamiento
# MAGIC - Normalización (StandardScaler)
# MAGIC - One-Hot Encoding
# MAGIC - Imputación de valores nulos
# MAGIC - Pipelines de transformación
# MAGIC
# MAGIC #### 📊 Evaluación
# MAGIC - Accuracy, Precision, Recall, F1
# MAGIC - Matriz de confusión
# MAGIC - ROC-AUC
# MAGIC - Trade-offs entre métricas
# MAGIC
# MAGIC #### 🤖 Modelos
# MAGIC - SGD Classifier
# MAGIC - Random Forest
# MAGIC - Comparación de modelos
# MAGIC - Selección de modelo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Reflexiones Importantes
# MAGIC
# MAGIC #### ⚕️ Sobre Medicina e IA
# MAGIC
# MAGIC **TODO: Reflexiona sobre:**
# MAGIC
# MAGIC 1. **Falsos Negativos vs Falsos Positivos**
# MAGIC    - En medicina, ¿cuál es más grave y por qué?
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC 2. **Responsabilidad Ética**
# MAGIC    - ¿Puede un modelo de IA tomar decisiones médicas solo?
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC 3. **Interpretabilidad**
# MAGIC    - ¿Por qué es importante que los médicos entiendan cómo decide el modelo?
# MAGIC    - [Tu respuesta aquí]
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC #### 🎯 Desafíos Adicionales
# MAGIC
# MAGIC 1. **Optimización de Hiperparámetros**
# MAGIC    ```python
# MAGIC    # TODO: Implementa GridSearchCV
# MAGIC    from sklearn.model_selection import GridSearchCV
# MAGIC    param_grid = {
# MAGIC        'n_estimators': [50, 100, 200],
# MAGIC        'max_depth': [5, 10, 15, None],
# MAGIC        'min_samples_split': [2, 5, 10]
# MAGIC    }
# MAGIC    ```
# MAGIC
# MAGIC 2. **Prueba Otros Modelos**
# MAGIC    - Gradient Boosting
# MAGIC    - XGBoost
# MAGIC    - SVM
# MAGIC    - Neural Networks
# MAGIC
# MAGIC 3. **Feature Engineering**
# MAGIC    - Crea nuevas características
# MAGIC    - Analiza importancia de características
# MAGIC    - Selecciona las más relevantes
# MAGIC
# MAGIC 4. **Análisis de Errores**
# MAGIC    - ¿Qué pacientes se clasifican mal?
# MAGIC    - ¿Hay patrones en los errores?
# MAGIC    - ¿Cómo mejorar?
# MAGIC
# MAGIC 5. **Curva ROC**
# MAGIC    - Implementa y visualiza la curva ROC
# MAGIC    - Encuentra el threshold óptimo
# MAGIC    - Compara AUC de diferentes modelos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📖 Recursos para Seguir Aprendiendo
# MAGIC
# MAGIC - **Scikit-learn**: [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
# MAGIC - **MLflow**: [Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
# MAGIC - **Kaggle**: Notebooks sobre Heart Disease
# MAGIC - **Papers**: Sobre IA en diagnóstico médico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🌟 ¡Felicidades!
# MAGIC
# MAGIC Has completado un proyecto completo de Machine Learning en el dominio médico. Las habilidades que has desarrollado son aplicables a muchos otros problemas de clasificación.
# MAGIC
# MAGIC **Puntos clave para recordar:**
# MAGIC - ✅ Siempre explora tus datos primero
# MAGIC - ✅ Usa validación cruzada para evaluar
# MAGIC - ✅ Compara múltiples modelos
# MAGIC - ✅ Elige métricas según el problema
# MAGIC - ✅ En medicina, prioriza recall sobre precision
# MAGIC - ✅ Documenta todo con MLflow
# MAGIC - ✅ Piensa en las implicaciones éticas
# MAGIC
# MAGIC **¡Sigue practicando y construyendo proyectos! 🚀❤️**

# COMMAND ----------

# 🎨 CÓDIGO DE EJEMPLO PARA LOS DESAFÍOS
# Descomenta y completa para explorar más

# ========================================
# TODO: DESAFÍO 1 - Grid Search
# ========================================
# from sklearn.model_selection import GridSearchCV
# param_grid = {...}
# grid_search = GridSearchCV(...)
# grid_search.fit(...)

# ========================================
# TODO: DESAFÍO 2 - Curva ROC
# ========================================
# from sklearn.metrics import roc_curve, auc
# y_proba = ...
# fpr, tpr, thresholds = roc_curve(...)
# TODO: visualiza la curva ROC.

# ========================================
# TODO: DESAFÍO 3 - Importancia de características
# ========================================
# feature_names = ...
# feature_importance = ...
# TODO: identifica e interpreta las características más importantes.

# ========================================
# TODO: DESAFÍO 4 - Comparar más modelos
# ========================================
# modelos = {...}
# resultados = []
# for nombre, modelo in modelos.items():
#     with mlflow.start_run(run_name=f"{nombre} - Heart Disease"):
#         modelo.fit(x_train_pr, y_train)
#         preds = modelo.predict(x_test_pr)
#         
#         acc = accuracy_score(y_test, preds)
#         prec = precision_score(y_test, preds)
#         rec = recall_score(y_test, preds)
#         f1_s = f1_score(y_test, preds)
#         
#         resultados.append({
#             'Modelo': nombre,
#             'Accuracy': acc,
#             'Precision': prec,
#             'Recall': rec,
#             'F1-Score': f1_s
#         })
# 
# df_resultados = pd.DataFrame(resultados).sort_values('Recall', ascending=False)
# print("\n📊 Ranking de Modelos (ordenado por Recall):")
# print(df_resultados.to_string(index=False))

# ========================================
# DESAFÍO 5: Análisis de Errores
# ========================================
# # TODO: [Completa aquí] ¿Qué pacientes son difíciles de clasificar?
# # Encontrar errores
# errors_mask = final_preds != y_test
# errors_df = test_set[errors_mask].copy()
# errors_df['prediccion'] = final_preds[errors_mask]
# 
# print(f"\nTotal de errores: {errors_mask.sum()}")
# print(f"\nPacientes mal clasificados:")
# print(errors_df[['age', 'sex', 'cp', 'trestbps', 'chol', 'target', 'prediccion']])
# 
# # Analizar características de los errores
# print("\n¿Los errores tienen características en común?")
# print(errors_df.describe())

print("💡 Descomenta el código del desafío que quieras explorar")