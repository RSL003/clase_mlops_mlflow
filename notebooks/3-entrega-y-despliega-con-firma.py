# Databricks notebook source
# MAGIC %md
# MAGIC # 📦 MLflow: Entrega y Despliega Modelos con Firma
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC En este notebook aprenderás a:
# MAGIC
# MAGIC 1. **Configurar MLflow** para el seguimiento de experimentos de ML
# MAGIC 2. **Entrenar un modelo** de Random Forest con scikit-learn
# MAGIC 3. **Definir firmas de modelos** (Model Signatures) para documentar entrada/salida
# MAGIC 4. **Registrar métricas, parámetros y artefactos** en MLflow
# MAGIC 5. **Guardar modelos con metadatos completos** para despliegue en producción
# MAGIC
# MAGIC ## 📚 Conceptos Clave
# MAGIC
# MAGIC ### ¿Qué es MLflow?
# MAGIC MLflow es una plataforma open-source para gestionar el ciclo de vida completo de Machine Learning:
# MAGIC - **Tracking**: Registro de experimentos, parámetros y métricas
# MAGIC - **Projects**: Empaquetado de código reutilizable
# MAGIC - **Models**: Gestión y despliegue de modelos
# MAGIC - **Registry**: Almacén centralizado de modelos versionados
# MAGIC
# MAGIC ### ¿Qué son las Firmas de Modelos?
# MAGIC Las **Model Signatures** definen el esquema de entrada y salida del modelo:
# MAGIC - Validan que los datos tengan el formato correcto
# MAGIC - Documentan el contrato del modelo
# MAGIC - Facilitan la integración en producción
# MAGIC - Previenen errores de inferencia
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 ¡Empecemos!

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# IMPORTANTE: Sustituye esto con tu email real de Databricks
# Ejemplo: 'nombre.apellido@dominio.com'
# ⚠️ IMPORTANTE: Cambia esto por tu email de Databricks
email = 'tu_email_de_registro_en_databricks'

# Configurar el experimento
experiment_name = f"/Users/{email}/3-entrega-y-despliega-con-firma"
mlflow.set_experiment(experiment_name)

print(f"✓ Experimento configurado: {experiment_name}")
print("✓ MLflow listo para registrar métricas, parámetros y modelos")



# COMMAND ----------

# Importación de librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Scikit-learn: framework de ML
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# MLflow: gestión de modelos y experimentos
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

print("✓ Librerías importadas correctamente")

# COMMAND ----------

# Carga del dataset de diabetes
# Este dataset contiene información médica de 442 pacientes
# Objetivo: predecir la progresión de la diabetes un año después del baseline
dataset = load_diabetes()

print("📊 Información del Dataset:")
print(f"   - Número de muestras: {dataset.data.shape[0]}")
print(f"   - Número de características: {dataset.data.shape[1]}")
print(f"   - Características: {dataset.feature_names}")
print(f"   - Target: Progresión de la diabetes (variable continua)")

# COMMAND ----------

# División del dataset en conjuntos de entrenamiento y prueba
# - X: características (features)
# - y: variable objetivo (target)
# - test_size por defecto es 0.25 (25% para test)
# - random_state asegura reproducibilidad

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42
)

print("📂 Datos divididos:")
print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
print(f"   - Prueba: {X_test.shape[0]} muestras")
print(f"   - Proporción: {(X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]))*100:.1f}% / {(X_test.shape[0]/(X_train.shape[0]+X_test.shape[0]))*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definición de Esquema de Entrada/Salida en MLflow
# MAGIC
# MAGIC En MLflow, es fundamental definir esquemas de entrada y salida para garantizar que los datos manejados por el modelo son consistentes y están en el formato esperado. Estos esquemas se visualizan en la interfaz de usuario de MLflow, facilitando la gestión y comprensión de los modelos.
# MAGIC
# MAGIC ### Pasos para Definir Esquemas en MLflow
# MAGIC
# MAGIC 1. **Desactivar Auto-guardado del Modelo:** Al guardar el modelo, usa `log_models=False` para desactivar el auto-guardado y permitir la especificación manual de esquemas.
# MAGIC
# MAGIC 2. **Definir el Esquema de Entrada:**
# MAGIC    - Utiliza `Schema([])` para crear un esquema.
# MAGIC    - Incluye cada característica esperada por el modelo con su tipo de dato correspondiente (`double` en este ejemplo).
# MAGIC    - Ejemplo:
# MAGIC      ```python
# MAGIC      input_schema = Schema([
# MAGIC        ColSpec("double", "age"),
# MAGIC        ColSpec("double", "sex"),
# MAGIC        ColSpec("double", "bmi"),
# MAGIC        ColSpec("double", "bp"),
# MAGIC        // Continúa con las demás características
# MAGIC      ])
# MAGIC      ```
# MAGIC
# MAGIC 3. **Definir el Esquema de Salida:**
# MAGIC    - Define el formato y tipo de dato de la salida del modelo.
# MAGIC    - Ejemplo:
# MAGIC      ```python
# MAGIC      output_schema = Schema([ColSpec("long", "objetivo")])
# MAGIC      ```
# MAGIC
# MAGIC 4. **Crear la Firma del Modelo:**
# MAGIC    - Combina los esquemas de entrada y salida.
# MAGIC    - Ejemplo:
# MAGIC      ```python
# MAGIC      signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# MAGIC      ```
# MAGIC
# MAGIC ### Importancia
# MAGIC
# MAGIC - **Validación de Datos:** Asegura que los datos de entrada y salida sean consistentes con lo esperado por el modelo.
# MAGIC - **Facilita la Integración:** Los esquemas claros ayudan a integrar el modelo en sistemas de producción.
# MAGIC - **Mejora la Fiabilidad:** Previene errores y mejora la confiabilidad del modelo.
# MAGIC
# MAGIC Con estos pasos, se puede establecer un proceso claro y estructurado para la definición de esquemas en modelos de MLflow, esencial para el manejo efectivo del ciclo de vida de modelos de machine learning.
# MAGIC

# COMMAND ----------

# Definición del esquema de entrada
# Cada ColSpec define: tipo de dato y nombre de la característica
# "double" equivale a float64 en Python
input_schema = Schema([
    ColSpec("double", "age"),      # Edad
    ColSpec("double", "sex"),      # Sexo
    ColSpec("double", "bmi"),      # Índice de masa corporal
    ColSpec("double", "bp"),       # Presión arterial promedio
    ColSpec("double", "s1"),       # Colesterol total sérico
    ColSpec("double", "s2"),       # LDL (colesterol malo)
    ColSpec("double", "s3"),       # HDL (colesterol bueno)
    ColSpec("double", "s4"),       # Colesterol total / HDL
    ColSpec("double", "s5"),       # Log del nivel de triglicéridos
    ColSpec("double", "s6"),       # Nivel de glucosa en sangre
])

# Definición del esquema de salida
# El modelo predice un valor continuo (progresión de diabetes)
output_schema = Schema([ColSpec("double", "target")])

# Creación de la firma del modelo
# La firma documenta el contrato de entrada/salida del modelo
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

print("✓ Esquema de entrada definido: 10 características médicas")
print("✓ Esquema de salida definido: 1 valor de predicción (continuo)")
print("✓ Firma del modelo creada exitosamente")

# COMMAND ----------

# ========================================
# ENTRENAMIENTO Y REGISTRO DEL MODELO
# ========================================

# Activar autologging de MLflow para sklearn
# log_models=False porque registraremos el modelo manualmente con firma
mlflow.sklearn.autolog(log_models=False)

# Configurar el experimento
mlflow.set_experiment(f"/Users/{email}/3-entrega-y-despliega-con-firma")

# Iniciar una nueva ejecución (run)
with mlflow.start_run(run_name="RandomForest con Firma") as run:
    
    print("🚀 Iniciando entrenamiento del modelo...\n")
    
    # =====================================
    # 1. CONFIGURACIÓN DE HIPERPARÁMETROS
    # =====================================
    n_estimators = 100  # Número de árboles en el bosque
    max_depth = 10      # Profundidad máxima de cada árbol
    max_features = 5    # Número máximo de características por árbol
    random_state = 42   # Semilla para reproducibilidad
    
    print("⚙️  Hiperparámetros:")
    print(f"   - Árboles: {n_estimators}")
    print(f"   - Profundidad máxima: {max_depth}")
    print(f"   - Características máximas: {max_features}\n")
    
    # =====================================
    # 2. ENTRENAMIENTO DEL MODELO
    # =====================================
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1  # Usar todos los cores disponibles
    )
    
    rf.fit(X_train, y_train)
    print("✓ Modelo entrenado\n")
    
    # =====================================
    # 3. PREDICCIONES Y EVALUACIÓN
    # =====================================
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Métricas en conjunto de entrenamiento
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Métricas en conjunto de prueba
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("📊 Métricas de Evaluación:")
    print("\n   ENTRENAMIENTO:")
    print(f"   - MSE:  {mse_train:.2f}")
    print(f"   - RMSE: {rmse_train:.2f}")
    print(f"   - MAE:  {mae_train:.2f}")
    print(f"   - R²:   {r2_train:.4f}")
    
    print("\n   PRUEBA:")
    print(f"   - MSE:  {mse_test:.2f}")
    print(f"   - RMSE: {rmse_test:.2f}")
    print(f"   - MAE:  {mae_test:.2f}")
    print(f"   - R²:   {r2_test:.4f}\n")
    
    # =====================================
    # 4. REGISTRO EN MLFLOW
    # =====================================
    
    # Registrar hiperparámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.25)
    
    # Registrar métricas de entrenamiento
    mlflow.log_metric("train_mse", mse_train)
    mlflow.log_metric("train_rmse", rmse_train)
    mlflow.log_metric("train_mae", mae_train)
    mlflow.log_metric("train_r2", r2_train)
    
    # Registrar métricas de prueba
    mlflow.log_metric("test_mse", mse_test)
    mlflow.log_metric("test_rmse", rmse_test)
    mlflow.log_metric("test_mae", mae_test)
    mlflow.log_metric("test_r2", r2_test)
    
    # Calcular y registrar overfitting score
    overfitting = abs(r2_train - r2_test)
    mlflow.log_metric("overfitting_score", overfitting)
    
    print("✓ Parámetros y métricas registrados\n")
    
    # =====================================
    # 5. VISUALIZACIONES
    # =====================================
    
    # Gráfico 1: Residuales del conjunto de prueba
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    residuals = y_pred_test - y_test
    ax1.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicciones', fontsize=12)
    ax1.set_ylabel('Residuales', fontsize=12)
    ax1.set_title('Análisis de Residuales - Conjunto de Prueba', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    mlflow.log_figure(fig1, "residuals_plot.png")
    plt.close(fig1)
    
    # Gráfico 2: Valores reales vs predicciones
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Valores Reales', fontsize=12)
    ax2.set_ylabel('Predicciones', fontsize=12)
    ax2.set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    mlflow.log_figure(fig2, "predictions_vs_actual.png")
    plt.close(fig2)
    
    # Gráfico 3: Importancia de características
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importancia', fontsize=12)
    ax3.set_ylabel('Características', fontsize=12)
    ax3.set_title('Importancia de Características', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    mlflow.log_figure(fig3, "feature_importance.png")
    plt.close(fig3)
    
    print("✓ Visualizaciones generadas y registradas\n")
    
    # =====================================
    # 6. REGISTRO DEL MODELO CON FIRMA
    # =====================================
    
    # Registrar el modelo con su firma
    # Esto documenta el contrato de entrada/salida del modelo
    mlflow.sklearn.log_model(
        rf, 
        "random_forest_model", 
        signature=signature,
        input_example=X_train[:5]  # Ejemplo de entrada para referencia
    )
    
    print("✓ Modelo registrado con firma\n")
    
    # =====================================
    # 7. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    print("=" * 60)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Experimento: {experiment_name}")
    print(f"R² Score (Test): {r2_test:.4f}")
    print(f"RMSE (Test): {rmse_test:.2f}")
    print("=" * 60)

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Resumen de lo Aprendido
# MAGIC
# MAGIC ### 1. **MLflow Tracking**
# MAGIC - ✅ Configuración de experimentos
# MAGIC - ✅ Registro de parámetros e hiperparámetros
# MAGIC - ✅ Registro de métricas de evaluación
# MAGIC - ✅ Guardado de visualizaciones como artefactos
# MAGIC
# MAGIC ### 2. **Firmas de Modelos (Model Signatures)**
# MAGIC Las firmas documentan el contrato de entrada/salida del modelo:
# MAGIC - **Input Schema**: Define las características que el modelo espera (tipo y nombre)
# MAGIC - **Output Schema**: Define el formato de las predicciones
# MAGIC - **Beneficios**:
# MAGIC   - Validación automática de datos en inferencia
# MAGIC   - Documentación clara del modelo
# MAGIC   - Prevención de errores en producción
# MAGIC   - Facilita la integración con sistemas externos
# MAGIC
# MAGIC ### 3. **Mejores Prácticas**
# MAGIC - 📊 Registrar múltiples métricas (MSE, RMSE, MAE, R²)
# MAGIC - 📈 Comparar entrenamiento vs prueba (detectar overfitting)
# MAGIC - 🎨 Crear visualizaciones informativas
# MAGIC - 📝 Incluir ejemplos de entrada con el modelo
# MAGIC - 🏷️ Usar nombres descriptivos para experimentos y runs
# MAGIC
# MAGIC ### 4. **Próximos Pasos**
# MAGIC 1. Acceder a la UI de MLflow para visualizar el experimento
# MAGIC 2. Comparar diferentes runs con distintos hiperparámetros
# MAGIC 3. Registrar el mejor modelo en el Model Registry
# MAGIC 4. Desplegar el modelo para inferencia en producción
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **💡 Consejo**: Experimenta con diferentes valores de hiperparámetros y compara los resultados en la interfaz de MLflow para encontrar la mejor configuración.