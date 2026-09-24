# Databricks notebook source
# MAGIC %md
# MAGIC # Introducción a MLflow - Registro de Experimentos
# MAGIC
# MAGIC Este notebook ilustra cómo utilizar **MLflow**, una herramienta esencial para la gestión del ciclo de vida del machine learning, para registrar varios componentes de un experimento.
# MAGIC
# MAGIC ## ¿Qué es MLflow?
# MAGIC
# MAGIC MLflow es una plataforma de código abierto para gestionar el ciclo de vida completo del machine learning, incluyendo:
# MAGIC - Experimentación
# MAGIC - Reproducibilidad
# MAGIC - Despliegue
# MAGIC - Registro centralizado de modelos
# MAGIC
# MAGIC ## Funcionalidades Principales
# MAGIC
# MAGIC ### 1. Registro de Parámetros
# MAGIC `mlflow.log_param()` se emplea para registrar parámetros clave-valor, que son esenciales para la configuración del experimento.
# MAGIC
# MAGIC ### 2. Registro de Métricas
# MAGIC `mlflow.log_metric()` permite registrar y actualizar métricas a lo largo del experimento. Estas métricas son cruciales para evaluar el rendimiento del modelo.
# MAGIC
# MAGIC ### 3. Registro de Artefactos
# MAGIC `mlflow.log_artifact()` se usa para registrar archivos como artefactos. Estos pueden ser:
# MAGIC - Archivos de texto
# MAGIC - Imágenes
# MAGIC - Modelos entrenados
# MAGIC - Gráficos y visualizaciones
# MAGIC
# MAGIC ## Objetivo del Notebook
# MAGIC
# MAGIC En este notebook aprenderás a:
# MAGIC 1. Iniciar un experimento de MLflow
# MAGIC 2. Registrar parámetros y métricas
# MAGIC 3. Guardar artefactos
# MAGIC 4. Finalizar correctamente un experimento

# COMMAND ----------

# Importar librerías necesarias
from random import randint, random
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# Configurar el experimento
# IMPORTANTE: Reemplaza con tu email de registro en Databricks
email = 'rafaelsanchezupo@gmail.com'

# Establecer el nombre del experimento
mlflow.set_experiment(f"/Users/{email}/1-introduccion-mlflow")

# Iniciar una nueva ejecución (run) del experimento
mlflow.start_run()

print(f"Experimento iniciado: /Users/{email}/1-introduccion-mlflow")
print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# 1. Registrar un parámetro (clave-valor)
# Los parámetros son valores de configuración del experimento
param_value = randint(0, 100)
mlflow.log_param("param1", param_value)
print(f"Parámetro registrado: param1 = {param_value}")

# 2. Registrar métricas
# Las métricas pueden actualizarse durante la ejecución
# MLflow mantiene el historial de todas las actualizaciones
metric_1 = random()
metric_2 = random() + 1
metric_3 = random() + 2

mlflow.log_metric("foo", metric_1)
mlflow.log_metric("foo", metric_2)
mlflow.log_metric("foo", metric_3)
print(f"Métricas registradas: {metric_1:.4f}, {metric_2:.4f}, {metric_3:.4f}")

# 3. Registrar artefactos (archivos)
# Los artefactos son archivos relacionados con el experimento
artifact_path = f"/Workspace/Users/{email}/clase_mlops_mlflow/notebooks/artifacts/kitten.jpg"

try:
    mlflow.log_artifact(artifact_path)
    print(f"Artefacto registrado: {artifact_path}")
except Exception as e:
    print(f"Error al registrar artefacto: {e}")
    print("Asegúrate de que el archivo existe en la ruta especificada")

# COMMAND ----------

# Finalizar la ejecución del experimento
# IMPORTANTE: Siempre debes llamar a end_run() para cerrar correctamente el experimento
mlflow.end_run()
print("Experimento finalizado correctamente")
print("Puedes revisar los resultados en la UI de MLflow")