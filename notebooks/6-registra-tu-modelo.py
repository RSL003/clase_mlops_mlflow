# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 MLflow Model Registry - Guía Completa
# MAGIC
# MAGIC ## 🎯 Objetivo del Notebook
# MAGIC
# MAGIC Este notebook demuestra cómo utilizar **MLflow Model Registry** para gestionar el ciclo de vida completo de modelos de Machine Learning. Aprenderás a:
# MAGIC
# MAGIC - 🔍 Buscar y comparar experimentos
# MAGIC - 🏆 Seleccionar el mejor modelo basado en métricas
# MAGIC - 📝 Registrar modelos en el Model Registry
# MAGIC - 🏷️ Gestionar versiones y etapas (Staging, Production)
# MAGIC - 📊 Visualizar y comparar métricas
# MAGIC - 🔄 Implementar mejores prácticas de MLOps
# MAGIC
# MAGIC ## 📚 Contexto
# MAGIC
# MAGIC MLflow Model Registry es un repositorio centralizado que proporciona:
# MAGIC - **Control de versiones** de modelos
# MAGIC - **Gestión de etapas** del ciclo de vida (None → Staging → Production → Archived)
# MAGIC - **Anotaciones y descripciones** para cada versión
# MAGIC - **Trazabilidad completa** desde entrenamiento hasta producción
# MAGIC
# MAGIC ---

# COMMAND ----------

# 📦 Instalación de dependencias
!pip install mlflow scikit-learn pandas matplotlib seaborn --quiet --upgrade

# Nota: En Databricks, descomentar la siguiente línea
# dbutils.library.restartPython()

# COMMAND ----------

# 📚 Importación de librerías necesarias
import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Configuración de visualización
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Librerías importadas correctamente")
print(f"📌 Versión de MLflow: {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuración de MLflow
# MAGIC
# MAGIC Configuramos la conexión con el servidor de MLflow y verificamos que todo funciona correctamente.

# COMMAND ----------

# 🔧 Configuración del cliente de MLflow
try:
    # Configurar tracking URI (ajustar según tu configuración)
    # mlflow.set_tracking_uri("http://localhost:5000")  # Para servidor local
    # mlflow.set_tracking_uri("databricks")  # Para Databricks

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks")
    
    # Crear instancia del cliente MLflow
    client = MlflowClient()
    
    print("✅ Cliente MLflow creado correctamente")
    print(f"📍 Tracking URI: {mlflow.get_tracking_uri()}")
    
except Exception as e:
    print(f"❌ Error al crear el cliente MLflow: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Exploración de Experimentos
# MAGIC
# MAGIC Primero, vamos a listar todos los experimentos disponibles para identificar cuál queremos analizar.

# COMMAND ----------

# 📋 Listar todos los experimentos disponibles
experiments = client.search_experiments()

print(f"🔢 Total de experimentos encontrados: {len(experiments)}\n")

# Crear DataFrame para mejor visualización
experiments_data = []
for exp in experiments:
    experiments_data.append({
        'ID': exp.experiment_id,
        'Nombre': exp.name,
        'Lifecycle': exp.lifecycle_stage,
        'Artifact Location': exp.artifact_location
    })

df_experiments = pd.DataFrame(experiments_data)
print("📊 Experimentos disponibles:")
df_experiments

# COMMAND ----------

# 🎯 Seleccionar experimento específico
# IMPORTANTE: Reemplazar "<experiment-id>" con el ID real de tu experimento
# Puedes usar el nombre del experimento o su ID

# Opción 1: Por ID
experiment_id = "<experiment-id>"  # Ejemplo: "0", "1", "2", etc.

# Opción 2: Por nombre (descomenta si prefieres usar el nombre)
# experiment = mlflow.get_experiment_by_name("nombre-del-experimento")
# experiment_id = experiment.experiment_id

try:
    experiment = mlflow.get_experiment(experiment_id=experiment_id)
    
    print(f"✅ Experimento seleccionado:")
    print(f"   📌 ID: {experiment.experiment_id}")
    print(f"   📛 Nombre: {experiment.name}")
    print(f"   📂 Artifact Location: {experiment.artifact_location}")
    print(f"   🔄 Lifecycle Stage: {experiment.lifecycle_stage}")
    
except Exception as e:
    print(f"❌ Error: No se encontró el experimento con ID '{experiment_id}'")
    print(f"   Detalles: {str(e)}")
    print("\n💡 Tip: Usa el ID de uno de los experimentos listados arriba")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Análisis de Runs del Experimento
# MAGIC
# MAGIC Ahora vamos a obtener todas las ejecuciones (runs) del experimento seleccionado y analizarlas en detalle.

# COMMAND ----------

# 🔎 Buscar todas las ejecuciones del experimento
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.mse ASC"]  # Ordenar por MSE (menor es mejor)
)

print(f"🔢 Total de runs encontrados: {len(runs)}\n")

if len(runs) == 0:
    print("⚠️ No se encontraron runs en este experimento.")
    print("💡 Asegúrate de haber ejecutado entrenamientos en este experimento primero.")
else:
    print(f"✅ Se encontraron {len(runs)} ejecuciones para analizar")

# COMMAND ----------

# 📈 Crear DataFrame con información de todos los runs
runs_data = []

for run in runs:
    run_data = {
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'status': run.info.status,
        'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
        'duration_min': round((run.info.end_time - run.info.start_time) / 60000, 2) if run.info.end_time else None,
    }
    
    # Añadir métricas disponibles
    for metric_key, metric_value in run.data.metrics.items():
        run_data[metric_key] = round(metric_value, 4)
    
    # Añadir parámetros importantes
    for param_key, param_value in run.data.params.items():
        run_data[f'param_{param_key}'] = param_value
    
    runs_data.append(run_data)

df_runs = pd.DataFrame(runs_data)

print("📊 Resumen de todas las ejecuciones:")
print(f"Columnas disponibles: {list(df_runs.columns)}\n")
df_runs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Visualización de Métricas
# MAGIC
# MAGIC Visualizamos las métricas de todos los runs para identificar patrones y el mejor modelo.

# COMMAND ----------

# 📊 Visualización de métricas comparativas
if len(runs) > 0:
    # Identificar columnas de métricas (excluir columnas de identificación y parámetros)
    metric_columns = [col for col in df_runs.columns 
                     if col not in ['run_id', 'run_name', 'status', 'start_time', 'duration_min'] 
                     and not col.startswith('param_')]
    
    if len(metric_columns) > 0:
        # Crear subplots para cada métrica
        n_metrics = len(metric_columns)
        fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(15, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_columns[:3]):  # Mostrar hasta 3 métricas
            if metric in df_runs.columns:
                ax = axes[idx] if n_metrics > 1 else axes[0]
                df_runs_sorted = df_runs.sort_values(by=metric)
                ax.barh(range(len(df_runs_sorted)), df_runs_sorted[metric], color='steelblue')
                ax.set_xlabel(metric.upper())
                ax.set_ylabel('Runs')
                ax.set_title(f'Comparación de {metric.upper()}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas descriptivas
        print("\n📈 Estadísticas de métricas:")
        print(df_runs[metric_columns].describe())
    else:
        print("⚠️ No se encontraron métricas en los runs")
else:
    print("⚠️ No hay runs para visualizar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏆 Selección del Mejor Modelo
# MAGIC
# MAGIC Utilizamos la métrica MSE (Mean Squared Error) para seleccionar el mejor modelo. Menor MSE = Mejor modelo.

# COMMAND ----------

# 🏅 Identificar el mejor modelo basado en MSE
try:
    # Obtener todas las métricas MSE
    metrics = [run.data.metrics.get("mse") for run in runs if run.data.metrics.get("mse") is not None]
    
    if not metrics:
        raise ValueError("No se encontró la métrica 'mse' en los runs. Verifica que tus modelos tengan esta métrica.")
    
    # Encontrar el mejor MSE (menor valor)
    best_mse = min(metrics)
    
    # Obtener el run con el mejor MSE
    best_run = [run for run in runs if run.data.metrics.get("mse") == best_mse][0]
    
    print("🏆 MEJOR MODELO IDENTIFICADO")
    print("=" * 60)
    print(f"📊 MSE: {best_mse:.4f}")
    print(f"🆔 Run ID: {best_run.info.run_id}")
    print(f"📛 Run Name: {best_run.info.run_name}")
    print(f"📅 Fecha: {datetime.fromtimestamp(best_run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Duración: {(best_run.info.end_time - best_run.info.start_time) / 60000:.2f} min")
    print("\n📈 Todas las métricas:")
    for metric_name, metric_value in best_run.data.metrics.items():
        print(f"   • {metric_name}: {metric_value:.4f}")
    
    print("\n⚙️ Parámetros utilizados:")
    for param_name, param_value in best_run.data.params.items():
        print(f"   • {param_name}: {param_value}")
    
    print("\n📊 Comparación con otros modelos:")
    print(f"   • Mejor MSE: {best_mse:.4f}")
    print(f"   • Peor MSE: {max(metrics):.4f}")
    print(f"   • MSE promedio: {np.mean(metrics):.4f}")
    print(f"   • Mejora vs promedio: {((np.mean(metrics) - best_mse) / np.mean(metrics) * 100):.2f}%")
    
except Exception as e:
    print(f"❌ Error al seleccionar el mejor modelo: {str(e)}")
    print("\n💡 Verifica que tus runs tengan la métrica 'mse' registrada")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Registro del Modelo en Model Registry
# MAGIC
# MAGIC El **Model Registry** es un componente crítico para MLOps que permite:
# MAGIC - ✅ Versionado automático de modelos
# MAGIC - ✅ Gestión de ciclo de vida (Staging → Production)
# MAGIC - ✅ Trazabilidad completa
# MAGIC - ✅ Colaboración entre equipos

# COMMAND ----------

# 📝 Registrar el mejor modelo en el Model Registry
try:
    # Definir nombre descriptivo para el modelo
    model_name = "diabetes-prediction-model"
    
    # Construir URI del modelo
    model_uri = f"runs:/{best_run.info.run_id}/model"
    
    # Registrar el modelo
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "stage": "development",
            "algorithm": best_run.data.params.get("algorithm", "unknown"),
            "mse": str(best_mse),
            "registered_by": "mlops-team",
            "purpose": "diabetes prediction"
        }
    )
    
    print("✅ MODELO REGISTRADO EXITOSAMENTE")
    print("=" * 60)
    print(f"📛 Nombre del modelo: {model_name}")
    print(f"🔢 Versión: {model_version.version}")
    print(f"🆔 Run ID: {best_run.info.run_id}")
    print(f"📍 Source: {model_version.source}")
    print(f"🏷️ Stage actual: {model_version.current_stage}")
    print(f"📅 Fecha de registro: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🔗 URI del modelo: {model_uri}")
    
except Exception as e:
    print(f"❌ Error al registrar el modelo: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Inspección del Modelo Registrado
# MAGIC
# MAGIC Verificamos que el modelo se haya registrado correctamente y exploramos su metadata.

# COMMAND ----------

# 🔍 Obtener información completa del modelo registrado
try:
    model_info = client.get_registered_model(name=model_name)
    
    print("📋 INFORMACIÓN DEL MODELO REGISTRADO")
    print("=" * 60)
    print(f"📛 Nombre: {model_info.name}")
    print(f"📅 Creación: {datetime.fromtimestamp(model_info.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Última actualización: {datetime.fromtimestamp(model_info.last_updated_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Descripción: {model_info.description if model_info.description else 'Sin descripción'}")
    print(f"🔢 Versiones totales: {len(model_info.latest_versions)}")
    
    print("\n📦 Versiones disponibles:")
    for version in model_info.latest_versions:
        print(f"   • Versión {version.version} - Stage: {version.current_stage}")
    
except Exception as e:
    print(f"❌ Error al obtener información del modelo: {str(e)}")

# COMMAND ----------

# 📊 Visualización detallada del modelo
model_info

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📥 Carga del Modelo para Uso

# COMMAND ----------

# 📥 Cargar el modelo desde el registry
try:
    # Opción 1: Cargar desde el run específico
    model = mlflow.sklearn.load_model(model_uri)
    
    # Opción 2: Cargar desde el registry por nombre y versión
    # model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version.version}")
    
    # Opción 3: Cargar la versión en una etapa específica
    # model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    
    print("✅ Modelo cargado exitosamente")
    print(f"📦 Tipo de modelo: {type(model).__name__}")
    print(f"🔧 Algoritmo: {type(model).__module__}.{type(model).__name__}")
    
    # Mostrar atributos del modelo si están disponibles
    if hasattr(model, 'n_features_in_'):
        print(f"📊 Número de features: {model.n_features_in_}")
    
except Exception as e:
    print(f"❌ Error al cargar el modelo: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 Inferencia con el Modelo
# MAGIC
# MAGIC Ahora vamos a usar el modelo cargado para hacer predicciones sobre el dataset de diabetes.

# COMMAND ----------

# 🔮 Realizar predicciones con el modelo
try:
    # Cargar dataset de diabetes
    diabetes = datasets.load_diabetes()
    X_test = diabetes.data
    y_true = diabetes.target
    
    # Hacer predicciones
    predictions = model.predict(X_test)
    
    print("✅ PREDICCIONES REALIZADAS")
    print("=" * 60)
    print(f"📊 Número de predicciones: {len(predictions)}")
    print(f"📈 Predicción mínima: {predictions.min():.2f}")
    print(f"📈 Predicción máxima: {predictions.max():.2f}")
    print(f"📈 Predicción promedio: {predictions.mean():.2f}")
    print(f"📈 Desviación estándar: {predictions.std():.2f}")
    
    # Calcular error
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    print(f"\n📊 MÉTRICAS DE EVALUACIÓN:")
    print(f"   • MSE: {mse:.4f}")
    print(f"   • MAE: {mae:.4f}")
    print(f"   • R² Score: {r2:.4f}")
    
    # Mostrar algunas predicciones de ejemplo
    print(f"\n🔍 Ejemplos de predicciones (primeras 5):")
    for i in range(min(5, len(predictions))):
        print(f"   {i+1}. Real: {y_true[i]:.2f} | Predicción: {predictions[i]:.2f} | Error: {abs(y_true[i] - predictions[i]):.2f}")
    
except Exception as e:
    print(f"❌ Error durante la inferencia: {str(e)}")
    raise

# COMMAND ----------

# 📊 Visualización de predicciones vs valores reales
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico 1: Scatter plot de predicciones vs valores reales
axes[0].scatter(y_true, predictions, alpha=0.6, edgecolors='k', s=50)
axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel('Valores Reales', fontsize=12)
axes[0].set_ylabel('Predicciones', fontsize=12)
axes[0].set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Distribución de errores
errors = y_true - predictions
axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Error = 0')
axes[1].set_xlabel('Error (Real - Predicción)', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_title('Distribución de Errores', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📊 Análisis de errores:")
print(f"   • Error medio: {errors.mean():.4f}")
print(f"   • Error absoluto medio: {abs(errors).mean():.4f}")
print(f"   • Error máximo: {errors.max():.4f}")
print(f"   • Error mínimo: {errors.min():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏷️ Gestión de Metadatos y Tags
# MAGIC
# MAGIC Los tags nos permiten añadir información contextual al modelo para facilitar su gestión y búsqueda.

# COMMAND ----------

# 🏷️ Añadir tags y metadatos al modelo
try:
    # Tags de validación
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="validation_status",
        value="passed"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="prediction_works",
        value="true"
    )
    
    # Tags técnicos
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="framework",
        value="scikit-learn"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="mse_score",
        value=f"{mse:.4f}"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="r2_score",
        value=f"{r2:.4f}"
    )
    
    # Tags de negocio
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="use_case",
        value="diabetes-progression-prediction"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="approved_by",
        value="data-science-team"
    )
    
    print("✅ Tags añadidos exitosamente al modelo")
    print(f"📝 Total de tags añadidos: 7")
    
except Exception as e:
    print(f"❌ Error al añadir tags: {str(e)}")

# COMMAND ----------

# 🔍 Obtener información actualizada de la versión del modelo
model_version_info = client.get_model_version(
    name=model_name,
    version=model_version.version
)

print("📋 INFORMACIÓN COMPLETA DE LA VERSIÓN DEL MODELO")
print("=" * 60)
print(f"📛 Nombre: {model_version_info.name}")
print(f"🔢 Versión: {model_version_info.version}")
print(f"🏷️ Stage: {model_version_info.current_stage}")
print(f"📍 Source: {model_version_info.source}")
print(f"🆔 Run ID: {model_version_info.run_id}")
print(f"📅 Creación: {datetime.fromtimestamp(model_version_info.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📅 Última actualización: {datetime.fromtimestamp(model_version_info.last_updated_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"👤 Usuario: {model_version_info.user_id}")
print(f"📝 Descripción: {model_version_info.description if model_version_info.description else 'Sin descripción'}")

# COMMAND ----------

# 🏷️ Visualizar todos los tags del modelo
print("\n📌 TAGS DEL MODELO:")
print("=" * 60)
for tag_key, tag_value in model_version_info.tags.items():
    print(f"   • {tag_key}: {tag_value}")

# Crear DataFrame con los tags para mejor visualización
tags_df = pd.DataFrame([
    {'Tag': key, 'Valor': value} 
    for key, value in model_version_info.tags.items()
])

print("\n📊 Tabla de Tags:")
tags_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Gestión de Etapas del Ciclo de Vida
# MAGIC
# MAGIC MLflow soporta las siguientes etapas:
# MAGIC - **None**: Etapa inicial al registrar el modelo
# MAGIC - **Staging**: Modelo en fase de pruebas/validación
# MAGIC - **Production**: Modelo aprobado para producción
# MAGIC - **Archived**: Modelo archivado (no en uso)
# MAGIC
# MAGIC ### Transición a Staging

# COMMAND ----------

# MAGIC %md
# MAGIC El modelo ha pasado todas las validaciones. ¡Es hora de moverlo a **Staging** para pruebas más exhaustivas! 🎉

# COMMAND ----------

# 🚀 Transición del modelo a Staging
try:
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False  # No archivar versiones anteriores en Staging
    )
    
    print("✅ MODELO MOVIDO A STAGING")
    print("=" * 60)
    print(f"📛 Modelo: {model_name}")
    print(f"🔢 Versión: {model_version.version}")
    print(f"🏷️ Nueva etapa: Staging")
    print(f"📅 Fecha de transición: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 El modelo ahora está listo para pruebas de integración y validación")
    
except Exception as e:
    print(f"❌ Error al mover el modelo a Staging: {str(e)}")

# COMMAND ----------

# 🔄 Actualizar información de la versión del modelo
model_version_info = client.get_model_version(
    name=model_name,
    version=model_version.version
)

print("📋 ESTADO ACTUALIZADO DEL MODELO")
print("=" * 60)
print(f"📛 Nombre: {model_version_info.name}")
print(f"🔢 Versión: {model_version_info.version}")
print(f"🏷️ Stage actual: {model_version_info.current_stage}")
print(f"📅 Última actualización: {datetime.fromtimestamp(model_version_info.last_updated_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# ✅ Verificar que el modelo está en Staging
current_stage = model_version_info.current_stage

if current_stage == "Staging":
    print("✅ CONFIRMADO: El modelo está en Staging")
    print(f"🎯 Stage actual: {current_stage}")
    print("\n📋 Próximos pasos recomendados:")
    print("   1. Ejecutar pruebas de integración")
    print("   2. Validar rendimiento con datos reales")
    print("   3. Obtener aprobación del equipo")
    print("   4. Mover a Production cuando esté listo")
else:
    print(f"⚠️ Stage actual: {current_stage}")
    print("   Se esperaba 'Staging'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Añadir Descripción al Modelo
# MAGIC
# MAGIC Las descripciones ayudan a documentar el propósito y características del modelo.

# COMMAND ----------

# 📝 Actualizar descripción del modelo registrado
description = f"""
# Modelo de Predicción de Diabetes

## Descripción
Modelo de regresión entrenado para predecir la progresión de diabetes utilizando 
características médicas de pacientes.

## Métricas de Rendimiento
- MSE: {mse:.4f}
- MAE: {mae:.4f}
- R² Score: {r2:.4f}

## Información Técnica
- Framework: scikit-learn
- Algoritmo: {type(model).__name__}
- Features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}
- Dataset: Diabetes dataset (sklearn)

## Estado
- Versión: {model_version.version}
- Stage: Staging
- Fecha de registro: {datetime.now().strftime('%Y-%m-%d')}
- Validación: Aprobada

## Uso
```python
import mlflow
model = mlflow.pyfunc.load_model("models:/{model_name}/Staging")
predictions = model.predict(data)
```
"""

try:
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=description
    )
    
    print("✅ Descripción añadida exitosamente al modelo")
    print(f"📝 Longitud de la descripción: {len(description)} caracteres")
    
except Exception as e:
    print(f"❌ Error al añadir descripción: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Ejemplo: Transición a Production
# MAGIC
# MAGIC Cuando el modelo ha sido validado completamente en Staging, podemos moverlo a Production.

# COMMAND ----------

# 🚀 Mover modelo a Production (DESCOMENTA CUANDO ESTÉ LISTO)
"""
# Una vez validado en Staging, movemos a Production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True  # Archivar versiones anteriores en Production
)

print("✅ Modelo movido a Production")
print(f"🎉 El modelo v{model_version.version} ahora está en producción")

# Cargar modelo desde Production
production_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
"""

print("💡 Para mover a Production, descomenta el código de arriba")
print("📋 Checklist antes de mover a Production:")
print("   ✅ Validación completa en Staging")
print("   ✅ Pruebas de integración exitosas")
print("   ✅ Aprobación del equipo de ML")
print("   ✅ Documentación completa")
print("   ✅ Plan de rollback definido")
print("   ✅ Monitoreo configurado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Listar Todas las Versiones del Modelo
# MAGIC
# MAGIC Visualizamos todas las versiones registradas del modelo y sus etapas.

# COMMAND ----------

# 📋 Listar todas las versiones del modelo
try:
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    print(f"📊 VERSIONES DEL MODELO: {model_name}")
    print("=" * 80)
    print(f"🔢 Total de versiones: {len(all_versions)}\n")
    
    # Crear DataFrame con todas las versiones
    versions_data = []
    for version in all_versions:
        versions_data.append({
            'Versión': version.version,
            'Stage': version.current_stage,
            'Run ID': version.run_id[:8] + '...',
            'Creación': datetime.fromtimestamp(version.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M'),
            'Usuario': version.user_id
        })
    
    df_versions = pd.DataFrame(versions_data)
    df_versions = df_versions.sort_values('Versión', ascending=False)
    
    print(df_versions.to_string(index=False))
    
    # Mostrar resumen por stage
    print("\n📈 Resumen por Stage:")
    stage_counts = df_versions['Stage'].value_counts()
    for stage, count in stage_counts.items():
        print(f"   • {stage}: {count} versión(es)")
    
except Exception as e:
    print(f"❌ Error al listar versiones: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Búsqueda Avanzada de Modelos
# MAGIC
# MAGIC Podemos buscar modelos específicos usando filtros avanzados.

# COMMAND ----------

# 🔎 Búsqueda de modelos con filtros
print("🔍 EJEMPLOS DE BÚSQUEDA DE MODELOS\n")

# Ejemplo 1: Buscar modelos en Staging
staging_models = client.search_model_versions(
    filter_string=f"name='{model_name}' and current_stage='Staging'"
)
print(f"📊 Modelos en Staging: {len(staging_models)}")
for model in staging_models:
    print(f"   • Versión {model.version}")

# Ejemplo 2: Buscar modelos en Production
production_models = client.search_model_versions(
    filter_string=f"name='{model_name}' and current_stage='Production'"
)
print(f"\n🚀 Modelos en Production: {len(production_models)}")
for model in production_models:
    print(f"   • Versión {model.version}")

# Ejemplo 3: Listar todos los modelos registrados
print("\n📚 Todos los modelos registrados en el sistema:")
all_registered_models = client.search_registered_models()
for idx, model in enumerate(all_registered_models[:10], 1):  # Mostrar primeros 10
    print(f"   {idx}. {model.name} - {len(model.latest_versions)} versión(es)")

if len(all_registered_models) > 10:
    print(f"   ... y {len(all_registered_models) - 10} más")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💡 Mejores Prácticas y Consejos
# MAGIC
# MAGIC Resumen de mejores prácticas para trabajar con MLflow Model Registry.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 Mejores Prácticas para Model Registry
# MAGIC
# MAGIC #### 1. 📝 Nomenclatura
# MAGIC - Usa nombres descriptivos y consistentes para los modelos
# MAGIC - Ejemplo: `{use_case}-{algorithm}-model` → `diabetes-prediction-model`
# MAGIC
# MAGIC #### 2. 🏷️ Tags y Metadatos
# MAGIC - **Técnicos**: framework, algorithm, metrics
# MAGIC - **Negocio**: use_case, owner, approved_by
# MAGIC - **Validación**: validation_status, prediction_works
# MAGIC - **Performance**: mse_score, r2_score, accuracy
# MAGIC
# MAGIC #### 3. 🚀 Gestión de Etapas
# MAGIC ```
# MAGIC None → Staging → Production → Archived
# MAGIC ```
# MAGIC - **None**: Modelo recién registrado
# MAGIC - **Staging**: Validación y pruebas
# MAGIC - **Production**: Modelo activo en producción
# MAGIC - **Archived**: Modelo obsoleto
# MAGIC
# MAGIC #### 4. 📊 Versionado
# MAGIC - Cada registro crea una nueva versión automáticamente
# MAGIC - Mantén solo las versiones necesarias en cada etapa
# MAGIC - Archiva versiones antiguas en Production
# MAGIC
# MAGIC #### 5. 📝 Documentación
# MAGIC - Añade descripciones detalladas a cada versión
# MAGIC - Documenta cambios entre versiones
# MAGIC - Incluye ejemplos de uso
# MAGIC - Especifica dependencias y requisitos
# MAGIC
# MAGIC #### 6. 🔍 Trazabilidad
# MAGIC - Vincula modelos con sus runs de entrenamiento
# MAGIC - Mantén logs de transiciones de etapa
# MAGIC - Documenta criterios de validación
# MAGIC
# MAGIC #### 7. 🔄 Rollback
# MAGIC - Siempre ten un plan de rollback
# MAGIC - Mantén la versión anterior en Staging
# MAGIC - Archiva en lugar de eliminar
# MAGIC
# MAGIC #### 8. 🎛️ Cargar Modelos
# MAGIC ```python
# MAGIC # Por run ID
# MAGIC model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
# MAGIC
# MAGIC # Por nombre y versión
# MAGIC model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
# MAGIC
# MAGIC # Por nombre y etapa
# MAGIC model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
# MAGIC ```
# MAGIC
# MAGIC #### 9. 🔐 Seguridad
# MAGIC - Controla accesos al Model Registry
# MAGIC - Audita cambios de etapa
# MAGIC - Valida modelos antes de moverlos a Production
# MAGIC
# MAGIC #### 10. 📈 Monitoreo
# MAGIC - Monitorea el rendimiento de modelos en Production
# MAGIC - Configura alertas para degradación de métricas
# MAGIC - Realiza validaciones periódicas

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎓 Resumen del Notebook
# MAGIC
# MAGIC ### ✅ Lo que hemos aprendido
# MAGIC
# MAGIC 1. **🔍 Exploración de Experimentos**: Listar y analizar experimentos disponibles
# MAGIC 2. **📊 Análisis de Runs**: Comparar métricas y rendimiento de diferentes entrenamientos
# MAGIC 3. **🏆 Selección del Mejor Modelo**: Identificar el modelo óptimo basado en métricas
# MAGIC 4. **📝 Registro de Modelos**: Registrar modelos en el Model Registry
# MAGIC 5. **🏷️ Gestión de Metadatos**: Añadir tags y descripciones para organización
# MAGIC 6. **🚀 Ciclo de Vida**: Mover modelos entre etapas (None → Staging → Production)
# MAGIC 7. **🔮 Inferencia**: Cargar y usar modelos registrados para predicciones
# MAGIC 8. **📊 Visualización**: Crear gráficos para análisis de rendimiento
# MAGIC 9. **🔍 Búsqueda**: Utilizar filtros para encontrar modelos específicos
# MAGIC 10. **💡 Mejores Prácticas**: Implementar patrones profesionales de MLOps
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC 1. **Integración CI/CD**: Automatizar el despliegue de modelos
# MAGIC 2. **Monitoreo**: Implementar seguimiento de modelos en producción
# MAGIC 3. **A/B Testing**: Comparar versiones de modelos en producción
# MAGIC 4. **Drift Detection**: Detectar degradación del modelo
# MAGIC 5. **Reentrenamiento Automático**: Pipeline automático de reentrenamiento
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
# MAGIC - [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
# MAGIC - [MLOps Best Practices](https://ml-ops.org/)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 ¡Felicidades! Has completado el tutorial de MLflow Model Registry**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛠️ Utilidades y Comandos Rápidos
# MAGIC
# MAGIC Funciones de utilidad para operaciones comunes con MLflow.

# COMMAND ----------

# 🛠️ UTILIDADES - Funciones helper para operaciones comunes

def list_all_registered_models():
    """Lista todos los modelos registrados con su información"""
    models = client.search_registered_models()
    data = []
    for model in models:
        data.append({
            'Nombre': model.name,
            'Versiones': len(model.latest_versions),
            'Última actualización': datetime.fromtimestamp(
                model.last_updated_timestamp / 1000
            ).strftime('%Y-%m-%d %H:%M')
        })
    return pd.DataFrame(data)


def get_model_by_stage(model_name, stage="Production"):
    """Carga un modelo específico por su etapa"""
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        print(f"✅ Modelo '{model_name}' cargado desde {stage}")
        return model
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def compare_model_versions(model_name, version1, version2):
    """Compara dos versiones de un modelo"""
    v1 = client.get_model_version(model_name, version1)
    v2 = client.get_model_version(model_name, version2)
    
    comparison = {
        'Versión': [version1, version2],
        'Stage': [v1.current_stage, v2.current_stage],
        'Run ID': [v1.run_id[:8] + '...', v2.run_id[:8] + '...'],
        'Creación': [
            datetime.fromtimestamp(v1.creation_timestamp / 1000).strftime('%Y-%m-%d'),
            datetime.fromtimestamp(v2.creation_timestamp / 1000).strftime('%Y-%m-%d')
        ]
    }
    
    return pd.DataFrame(comparison)


def archive_old_versions(model_name, keep_latest_n=3):
    """Archiva versiones antiguas, manteniendo las N más recientes"""
    versions = client.search_model_versions(f"name='{model_name}'")
    versions_sorted = sorted(versions, key=lambda x: x.version, reverse=True)
    
    archived_count = 0
    for version in versions_sorted[keep_latest_n:]:
        if version.current_stage != "Archived":
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
            archived_count += 1
    
    print(f"✅ {archived_count} versiones archivadas")
    print(f"📊 Manteniendo las {keep_latest_n} versiones más recientes")


# Ejemplo de uso (comentado)
"""
# Listar todos los modelos
df_models = list_all_registered_models()
display(df_models)

# Cargar modelo de Production
prod_model = get_model_by_stage("diabetes-prediction-model", "Production")

# Comparar versiones
comparison = compare_model_versions("diabetes-prediction-model", "1", "2")
display(comparison)

# Archivar versiones antiguas
archive_old_versions("diabetes-prediction-model", keep_latest_n=3)
"""

print("✅ Funciones de utilidad cargadas")
print("\n📚 Funciones disponibles:")
print("   • list_all_registered_models()")
print("   • get_model_by_stage(model_name, stage)")
print("   • compare_model_versions(model_name, v1, v2)")
print("   • archive_old_versions(model_name, keep_latest_n)")