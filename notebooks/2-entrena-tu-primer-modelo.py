# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Tu Primer Modelo con MLflow
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC En este notebook aprenderás a:
# MAGIC
# MAGIC 1. **Instalar y configurar MLflow** para el seguimiento de experimentos
# MAGIC 2. **Entrenar tu primer modelo de Machine Learning** con scikit-learn
# MAGIC 3. **Usar MLflow Autolog** para registro automático de parámetros y métricas
# MAGIC 4. **Registrar métricas personalizadas** y visualizaciones
# MAGIC 5. **Organizar experimentos** de forma profesional
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 ¿Qué es MLflow?
# MAGIC
# MAGIC **MLflow** es una plataforma open-source que te ayuda a gestionar el ciclo de vida completo de Machine Learning:
# MAGIC
# MAGIC | Componente | Descripción |
# MAGIC |------------|-------------|
# MAGIC | **Tracking** | Registra experimentos, parámetros, métricas y artefactos |
# MAGIC | **Projects** | Empaqueta código de ML en formato reutilizable |
# MAGIC | **Models** | Gestiona y despliega modelos en diferentes plataformas |
# MAGIC | **Registry** | Almacén centralizado para gestionar el ciclo de vida de modelos |
# MAGIC
# MAGIC ### 🔑 Conceptos Clave
# MAGIC
# MAGIC - **Experiment**: Agrupa múltiples runs relacionados (ej: "Optimización de Random Forest")
# MAGIC - **Run**: Una ejecución individual de tu código de ML con sus parámetros y resultados
# MAGIC - **Parameters**: Configuración del modelo (hiperparámetros)
# MAGIC - **Metrics**: Valores que miden el rendimiento del modelo (accuracy, MSE, etc.)
# MAGIC - **Artifacts**: Archivos generados (modelos, gráficos, datasets)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎬 ¡Comencemos!

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Paso 1: Configuración del Experimento
# MAGIC
# MAGIC Antes de entrenar, necesitamos configurar nuestro experimento en MLflow.
# MAGIC
# MAGIC ### 📝 Instrucciones:
# MAGIC 1. **Sustituye** `'tu_email_de_registro_en_databricks'` con tu email real
# MAGIC 2. **Ejemplo**: `'nombre.apellido@ejemplo.com'`
# MAGIC
# MAGIC En Databricks, los experimentos siguen la convención de: `/Users/{email}/{nombre_experimento}`

# COMMAND ----------

import mlflow
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# ⚠️ IMPORTANTE: Cambia esto por tu email de Databricks
email = 'rafaelsanchezupo@gmail.com'

# Configurar el experimento
experiment_name = f"/Users/{email}/2-entrena-tu-primer-modelo"
mlflow.set_experiment(experiment_name)

print("=" * 60)
print("✅ MLflow configurado correctamente")
print("=" * 60)
print(f"📊 Experimento: {experiment_name}")
print(f"🔗 Versión MLflow: {mlflow.__version__}")
print("=" * 60)
print("\n💡 Tip: Accede a la UI de MLflow para ver los resultados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Paso 3: Importar Librerías
# MAGIC
# MAGIC Importamos todas las librerías necesarias para:
# MAGIC - Manipulación de datos (pandas, numpy)
# MAGIC - Visualización (matplotlib)
# MAGIC - Machine Learning (scikit-learn)
# MAGIC - Seguimiento de experimentos (MLflow)

# COMMAND ----------

# Librerías de datos y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: ML framework
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Todas las librerías importadas correctamente")
print(f"   - Pandas: {pd.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - Matplotlib: Configurado")
print(f"   - Scikit-learn: Listo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Paso 4: Cargar y Explorar el Dataset
# MAGIC
# MAGIC Usaremos el **Diabetes Dataset** de scikit-learn:
# MAGIC
# MAGIC ### 📋 Características del Dataset:
# MAGIC - **442 muestras** de pacientes
# MAGIC - **10 características** médicas (edad, sexo, IMC, presión arterial, etc.)
# MAGIC - **Objetivo**: Predecir la progresión de la diabetes un año después
# MAGIC
# MAGIC Este es un problema de **regresión** donde predecimos valores continuos.

# COMMAND ----------

# Cargar el dataset de diabetes
dataset = load_diabetes()

# Información del dataset
print("=" * 60)
print("📊 INFORMACIÓN DEL DATASET")
print("=" * 60)
print(f"📦 Número de muestras: {dataset.data.shape[0]}")
print(f"📈 Número de características: {dataset.data.shape[1]}")
print(f"\n🔍 Características disponibles:")
for i, feature in enumerate(dataset.feature_names, 1):
    print(f"   {i:2d}. {feature}")
print(f"\n🎯 Variable objetivo: Progresión de la diabetes (continua)")
print(f"   - Rango: [{dataset.target.min():.1f}, {dataset.target.max():.1f}]")
print(f"   - Media: {dataset.target.mean():.1f}")
print(f"   - Desv. estándar: {dataset.target.std():.1f}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔀 Paso 5: Dividir los Datos
# MAGIC
# MAGIC Dividiremos el dataset en dos conjuntos:
# MAGIC - **Entrenamiento (75%)**: Para entrenar el modelo
# MAGIC - **Prueba (25%)**: Para evaluar el rendimiento en datos no vistos
# MAGIC
# MAGIC **¿Por qué es importante?**
# MAGIC - Evita el **overfitting** (sobreajuste)
# MAGIC - Simula el rendimiento en producción
# MAGIC - Permite una evaluación honesta del modelo

# COMMAND ----------

# Separar características (X) y objetivo (y)
X = dataset.data
y = dataset.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% para prueba
    random_state=42,     # Semilla para reproducibilidad
    shuffle=True         # Mezclar los datos
)

# Información de la división
print("=" * 60)
print("🔀 DIVISIÓN DE DATOS")
print("=" * 60)
print(f"📊 Datos totales: {len(X)} muestras")
print(f"\n📚 Conjunto de Entrenamiento:")
print(f"   - Muestras: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Shape: {X_train.shape}")
print(f"\n🧪 Conjunto de Prueba:")
print(f"   - Muestras: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"   - Shape: {X_test.shape}")
print("=" * 60)
print("✅ Datos preparados para el entrenamiento")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Paso 6: Entrenar el Modelo con MLflow
# MAGIC
# MAGIC Ahora viene la parte emocionante: ¡entrenar nuestro primer modelo con MLflow!
# MAGIC
# MAGIC ### 🎯 ¿Qué haremos?
# MAGIC
# MAGIC 1. **Activar MLflow Autolog**: Registra automáticamente parámetros, métricas y el modelo
# MAGIC 2. **Definir hiperparámetros**: Configuración del Random Forest
# MAGIC 3. **Entrenar el modelo**: Fit con los datos de entrenamiento
# MAGIC 4. **Evaluar**: Calcular métricas en el conjunto de prueba
# MAGIC 5. **Registrar métricas personalizadas**: Agregar información adicional
# MAGIC 6. **Crear visualizaciones**: Gráficos para análisis
# MAGIC
# MAGIC ### 🔑 MLflow Autolog
# MAGIC
# MAGIC `mlflow.sklearn.autolog()` es mágico porque registra automáticamente:
# MAGIC - ✅ Todos los hiperparámetros del modelo
# MAGIC - ✅ Métricas de evaluación (R², MSE, etc.)
# MAGIC - ✅ El modelo entrenado
# MAGIC - ✅ Información del entorno
# MAGIC
# MAGIC ¡Todo con una sola línea de código!

# COMMAND ----------

# ========================================
# ENTRENAMIENTO CON MLFLOW
# ========================================

# Activar autolog de MLflow para sklearn
# Esto registra automáticamente parámetros, métricas y el modelo
mlflow.sklearn.autolog()

# Iniciar una nueva ejecución (run) con nombre descriptivo
with mlflow.start_run(run_name="Random Forest - Primera versión") as run:
    
    print("=" * 70)
    print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    
    # =====================================
    # 1. DEFINIR HIPERPARÁMETROS
    # =====================================
    n_estimators = 100    # Número de árboles en el bosque
    max_depth = 6         # Profundidad máxima de cada árbol
    max_features = 3      # Número de características a considerar por split
    random_state = 42     # Semilla para reproducibilidad
    
    print("\n⚙️  HIPERPARÁMETROS DEL MODELO:")
    print(f"   - Número de árboles (n_estimators): {n_estimators}")
    print(f"   - Profundidad máxima (max_depth): {max_depth}")
    print(f"   - Características máximas (max_features): {max_features}")
    print(f"   - Semilla aleatoria (random_state): {random_state}")
    
    # =====================================
    # 2. CREAR Y ENTRENAR EL MODELO
    # =====================================
    print("\n🔨 Creando modelo Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1  # Usar todos los cores del CPU
    )
    
    print("📚 Entrenando modelo con", len(X_train), "muestras...")
    rf.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente")
    
    # =====================================
    # 3. HACER PREDICCIONES
    # =====================================
    print("\n🔮 Realizando predicciones...")
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    print("✅ Predicciones completadas")
    
    # =====================================
    # 4. CALCULAR MÉTRICAS
    # =====================================
    print("\n📊 CALCULANDO MÉTRICAS DE EVALUACIÓN:")
    
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
    
    print("\n   📚 CONJUNTO DE ENTRENAMIENTO:")
    print(f"      - MSE:  {mse_train:.2f}")
    print(f"      - RMSE: {rmse_train:.2f}")
    print(f"      - MAE:  {mae_train:.2f}")
    print(f"      - R²:   {r2_train:.4f} ({r2_train*100:.2f}%)")
    
    print("\n   🧪 CONJUNTO DE PRUEBA:")
    print(f"      - MSE:  {mse_test:.2f}")
    print(f"      - RMSE: {rmse_test:.2f}")
    print(f"      - MAE:  {mae_test:.2f}")
    print(f"      - R²:   {r2_test:.4f} ({r2_test*100:.2f}%)")
    
    # Calcular overfitting
    overfitting_score = abs(r2_train - r2_test)
    print(f"\n   ⚠️  Overfitting Score: {overfitting_score:.4f}")
    if overfitting_score < 0.05:
        print("      ✅ Excelente - Poco overfitting")
    elif overfitting_score < 0.10:
        print("      ⚡ Bueno - Overfitting moderado")
    else:
        print("      ⚠️  Cuidado - Posible overfitting")
    
    # =====================================
    # 5. REGISTRAR PARÁMETROS Y MÉTRICAS PERSONALIZADAS
    # =====================================
    print("\n📝 Registrando información adicional en MLflow...")
    
    # Registrar parámetros (aunque autolog ya lo hace, agregamos algunos extras)
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)
    mlflow.log_param("dataset_size", len(X))
    mlflow.log_param("train_test_split", "75-25")
    
    # Registrar métricas adicionales
    mlflow.log_metric("mse", mse_test)
    mlflow.log_metric("train_mse", mse_train)
    mlflow.log_metric("test_mse", mse_test)
    mlflow.log_metric("train_r2", r2_train)
    mlflow.log_metric("test_r2", r2_test)
    mlflow.log_metric("overfitting_score", overfitting_score)
    
    # =====================================
    # 6. CREAR VISUALIZACIONES
    # =====================================
    print("\n📈 Generando visualizaciones...")
    
    # Figura con 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis del Modelo Random Forest', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Residuales del conjunto de prueba
    residuals_test = y_pred_test - y_test
    axes[0, 0].scatter(y_pred_test, residuals_test, alpha=0.6, edgecolors='k')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicciones', fontsize=11)
    axes[0, 0].set_ylabel('Residuales', fontsize=11)
    axes[0, 0].set_title('Residuales - Conjunto de Prueba', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Predicciones vs Valores Reales (Prueba)
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', label='Predicciones')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Línea perfecta')
    axes[0, 1].set_xlabel('Valores Reales', fontsize=11)
    axes[0, 1].set_ylabel('Predicciones', fontsize=11)
    axes[0, 1].set_title('Predicciones vs Valores Reales', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Distribución de residuales
    axes[1, 0].hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuales', fontsize=11)
    axes[1, 0].set_ylabel('Frecuencia', fontsize=11)
    axes[1, 0].set_title('Distribución de Residuales', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Gráfico 4: Importancia de características
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1, 1].set_xlabel('Importancia', fontsize=11)
    axes[1, 1].set_ylabel('Características', fontsize=11)
    axes[1, 1].set_title('Importancia de Características', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Guardar la figura en MLflow
    mlflow.log_figure(fig, "model_analysis.png")
    print("✅ Visualizaciones guardadas en MLflow")
    
    # También guardar solo el gráfico de residuales (compatible con código original)
    fig_residuals = plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals_test)), residuals_test, alpha=0.6, edgecolors='k')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Observación", fontsize=12)
    plt.ylabel("Residual", fontsize=12)
    plt.title("Residuos", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    mlflow.log_figure(fig_residuals, "residuals_plot.png")
    plt.close('all')
    
    # =====================================
    # 7. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    
    print("\n" + "=" * 70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name}")
    print(f"🎯 R² Score (Test): {r2_test:.4f} ({r2_test*100:.2f}%)")
    print(f"📉 RMSE (Test): {rmse_test:.2f}")
    print(f"📈 MAE (Test): {mae_test:.2f}")
    print("=" * 70)
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Ve a la UI de MLflow para ver los resultados")
    print("   2. Experimenta con diferentes hiperparámetros")
    print("   3. Compara múltiples runs para encontrar el mejor modelo")
    print("=" * 70)

# Finalizar la ejecución
mlflow.end_run()
print("\n✅ Run finalizado correctamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 ¡Felicitaciones! Has completado tu primer modelo con MLflow
# MAGIC
# MAGIC ### 📊 ¿Qué hemos logrado?
# MAGIC
# MAGIC ✅ **Instalado y configurado MLflow**
# MAGIC ✅ **Cargado y explorado un dataset**
# MAGIC ✅ **Dividido datos en entrenamiento y prueba**
# MAGIC ✅ **Entrenado un modelo Random Forest**
# MAGIC ✅ **Evaluado con múltiples métricas**
# MAGIC ✅ **Registrado todo automáticamente con MLflow**
# MAGIC ✅ **Creado visualizaciones profesionales**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Conceptos Clave Aprendidos
# MAGIC
# MAGIC ### 1. **MLflow Tracking**
# MAGIC - `mlflow.set_experiment()`: Organiza tus experimentos
# MAGIC - `mlflow.start_run()`: Inicia una ejecución
# MAGIC - `mlflow.sklearn.autolog()`: Registro automático
# MAGIC - `mlflow.log_param()`: Registra parámetros manualmente
# MAGIC - `mlflow.log_metric()`: Registra métricas
# MAGIC - `mlflow.log_figure()`: Guarda visualizaciones
# MAGIC
# MAGIC ### 2. **Métricas de Regresión**
# MAGIC - **MSE** (Mean Squared Error): Penaliza errores grandes
# MAGIC - **RMSE** (Root MSE): En las mismas unidades que el target
# MAGIC - **MAE** (Mean Absolute Error): Error promedio absoluto
# MAGIC - **R²** (Coeficiente de determinación): Porcentaje de varianza explicada (0-1)
# MAGIC
# MAGIC ### 3. **Overfitting**
# MAGIC Cuando el modelo funciona muy bien en entrenamiento pero mal en prueba:
# MAGIC - **Overfitting Score = |R² train - R² test|**
# MAGIC - < 0.05: ✅ Excelente
# MAGIC - 0.05-0.10: ⚡ Aceptable
# MAGIC - \> 0.10: ⚠️ Problema
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Próximos Pasos
# MAGIC
# MAGIC ### 1. **Explora la UI de MLflow**
# MAGIC ```python
# MAGIC # En terminal local ejecuta:
# MAGIC mlflow ui
# MAGIC ```
# MAGIC Luego abre: http://localhost:5000
# MAGIC
# MAGIC ### 2. **Experimenta con Hiperparámetros**
# MAGIC Prueba diferentes valores y compara resultados:
# MAGIC - `n_estimators`: 50, 100, 200
# MAGIC - `max_depth`: 3, 6, 10, None
# MAGIC - `max_features`: 3, 5, 'sqrt', 'log2'
# MAGIC
# MAGIC ### 3. **Prueba Otros Modelos**
# MAGIC - Gradient Boosting
# MAGIC - XGBoost
# MAGIC - LightGBM
# MAGIC - Redes Neuronales
# MAGIC
# MAGIC ### 4. **Optimización de Hiperparámetros**
# MAGIC - Grid Search
# MAGIC - Random Search
# MAGIC - Bayesian Optimization
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Tips Profesionales
# MAGIC
# MAGIC ### ✨ Mejores Prácticas
# MAGIC 1. **Siempre usa nombres descriptivos** para experimentos y runs
# MAGIC 2. **Registra el contexto**: qué cambios hiciste y por qué
# MAGIC 3. **Compara múltiples runs**: MLflow facilita la comparación
# MAGIC 4. **Documenta tus decisiones**: agrega tags y notas
# MAGIC 5. **Versiona tus datos**: registra qué datos usaste
# MAGIC
# MAGIC ### 🎨 Mejora tus Visualizaciones
# MAGIC ```python
# MAGIC # Agregar tags a un run
# MAGIC mlflow.set_tag("modelo", "RandomForest")
# MAGIC mlflow.set_tag("version", "1.0")
# MAGIC mlflow.set_tag("objetivo", "baseline")
# MAGIC
# MAGIC # Registrar archivos adicionales
# MAGIC mlflow.log_artifact("path/to/file.csv")
# MAGIC ```
# MAGIC
# MAGIC ### 📈 Monitoreo en Producción
# MAGIC - Registra métricas de negocio, no solo técnicas
# MAGIC - Compara predicciones con valores reales
# MAGIC - Detecta drift en los datos
# MAGIC - Reentrana periódicamente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Recursos Adicionales
# MAGIC
# MAGIC - 📖 [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
# MAGIC - 🎥 [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
# MAGIC - 💬 [Comunidad MLflow](https://github.com/mlflow/mlflow/discussions)
# MAGIC - 📚 [Scikit-learn Documentation](https://scikit-learn.org/)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌟 ¡Sigue Aprendiendo!
# MAGIC
# MAGIC Este es solo el comienzo. MLflow tiene muchas más capacidades:
# MAGIC - **MLflow Projects**: Reproducibilidad
# MAGIC - **MLflow Models**: Despliegue multi-plataforma
# MAGIC - **Model Registry**: Gestión del ciclo de vida
# MAGIC - **Model Serving**: APIs REST automáticas
# MAGIC
# MAGIC **¡Continúa experimentando y construyendo modelos increíbles! 🚀**

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 🧪 Ejercicio Práctico: ¡Tu Turno!
# MAGIC
# MAGIC Ahora que has aprendido lo básico, te reto a:
# MAGIC
# MAGIC ### 🎯 Desafío 1: Mejora el Modelo
# MAGIC Crea un nuevo run experimentando con estos hiperparámetros:
# MAGIC ```python
# MAGIC n_estimators = 200
# MAGIC max_depth = 10
# MAGIC max_features = 5
# MAGIC ```
# MAGIC
# MAGIC **Pregunta**: ¿El modelo mejoró? ¿Por qué crees que sí o no?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Desafío 2: Prueba Grid Search
# MAGIC Encuentra la mejor combinación de hiperparámetros:
# MAGIC ```python
# MAGIC from sklearn.model_selection import GridSearchCV
# MAGIC
# MAGIC param_grid = {
# MAGIC     'n_estimators': [50, 100, 200],
# MAGIC     'max_depth': [3, 6, 10],
# MAGIC     'max_features': [3, 5, 7]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Desafío 3: Compara Modelos
# MAGIC Entrena al menos 3 modelos diferentes:
# MAGIC - Random Forest (ya lo hiciste ✅)
# MAGIC - Gradient Boosting
# MAGIC - Linear Regression
# MAGIC
# MAGIC Compara sus resultados en MLflow UI y decide cuál es mejor.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Desafío 4: Agrega Tags
# MAGIC Mejora la organización de tus experimentos:
# MAGIC ```python
# MAGIC mlflow.set_tag("tipo_modelo", "ensemble")
# MAGIC mlflow.set_tag("fecha", "2025-10-12")
# MAGIC mlflow.set_tag("autor", "tu_nombre")
# MAGIC mlflow.set_tag("estado", "experimental")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **💪 ¡Acepta el desafío y conviértete en un experto de MLflow!**

# COMMAND ----------

# 🎨 CÓDIGO DE EJEMPLO PARA LOS DESAFÍOS
# Descomenta y modifica según necesites

# ========================================
# DESAFÍO 1: Modelo Mejorado
# ========================================
# mlflow.sklearn.autolog()
# with mlflow.start_run(run_name="Random Forest - Mejorado"):
#     rf_mejorado = RandomForestRegressor(
#         n_estimators=200,
#         max_depth=10,
#         max_features=5,
#         random_state=42,
#         n_jobs=-1
#     )
#     rf_mejorado.fit(X_train, y_train)
#     y_pred = rf_mejorado.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     print(f"R² Score: {r2:.4f}")

# ========================================
# DESAFÍO 2: Grid Search
# ========================================
# from sklearn.model_selection import GridSearchCV
# 
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 6, 10],
#     'max_features': [3, 5, 7]
# }
# 
# mlflow.sklearn.autolog()
# with mlflow.start_run(run_name="Grid Search - Random Forest"):
#     grid_search = GridSearchCV(
#         RandomForestRegressor(random_state=42, n_jobs=-1),
#         param_grid,
#         cv=5,
#         scoring='r2',
#         verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     print(f"Mejores parámetros: {grid_search.best_params_}")
#     print(f"Mejor R² Score: {grid_search.best_score_:.4f}")

# ========================================
# DESAFÍO 3: Comparar Modelos
# ========================================
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# 
# modelos = {
#     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#     "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
#     "Linear Regression": LinearRegression()
# }
# 
# mlflow.sklearn.autolog()
# for nombre, modelo in modelos.items():
#     with mlflow.start_run(run_name=nombre):
#         modelo.fit(X_train, y_train)
#         y_pred = modelo.predict(X_test)
#         r2 = r2_score(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         print(f"{nombre}: R²={r2:.4f}, RMSE={rmse:.2f}")

print("💡 Descomenta el código que quieras probar y ejecútalo")