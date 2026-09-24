# Databricks notebook source
# MAGIC %md
# MAGIC # 🌸 Ejercicio: Clasificación de Iris con MLflow
# MAGIC
# MAGIC ## 🎯 Objetivos del Ejercicio
# MAGIC
# MAGIC En este ejercicio práctico aprenderás a:
# MAGIC
# MAGIC 1. **Trabajar con un problema de clasificación** multiclase
# MAGIC 2. **Aplicar MLflow** a un modelo de árbol de decisión
# MAGIC 3. **Registrar experimentos** de forma profesional
# MAGIC 4. **Evaluar y comparar** diferentes configuraciones
# MAGIC 5. **Crear visualizaciones** para clasificación
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌺 El Dataset Iris
# MAGIC
# MAGIC El **Iris Dataset** es uno de los datasets más famosos en Machine Learning, introducido por Ronald Fisher en 1936.
# MAGIC
# MAGIC ### 📊 Características del Dataset
# MAGIC
# MAGIC | Aspecto | Descripción |
# MAGIC |---------|-------------|
# MAGIC | **Muestras** | 150 flores (50 por clase) |
# MAGIC | **Características** | 4 medidas físicas (cm) |
# MAGIC | **Clases** | 3 especies de iris |
# MAGIC | **Tipo** | Clasificación multiclase |
# MAGIC | **Dificultad** | ⭐⭐ (Principiante) |
# MAGIC
# MAGIC ### 🌸 Las 3 Especies de Iris
# MAGIC
# MAGIC 1. **Iris Setosa** (Clase 0)
# MAGIC 2. **Iris Versicolor** (Clase 1)
# MAGIC 3. **Iris Virginica** (Clase 2)
# MAGIC
# MAGIC ### 📏 Las 4 Características
# MAGIC
# MAGIC 1. **Sepal Length** (Longitud del sépalo)
# MAGIC 2. **Sepal Width** (Ancho del sépalo)
# MAGIC 3. **Petal Length** (Longitud del pétalo)
# MAGIC 4. **Petal Width** (Ancho del pétalo)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Tu Misión
# MAGIC
# MAGIC Construir un **Árbol de Decisión** que pueda clasificar correctamente las especies de iris basándose en sus medidas físicas, y documentar todo el proceso con MLflow.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 ¡Empecemos!

# COMMAND ----------

import mlflow
import warnings
warnings.filterwarnings('ignore')


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# ⚠️ IMPORTANTE: Cambia esto por tu email de Databricks
email = 'rafaelsanchezupo@gmail.com'

# Validar que el email no esté vacío
if not email:
    print("⚠️  ADVERTENCIA: Debes configurar tu email antes de continuar")
    print("   Cambia la variable 'email' por tu email de Databricks")
else:
    # Configurar el experimento
    experiment_name = f"/Users/{email}/4-ejercicio-the-irish"
    mlflow.set_experiment(experiment_name)
    
    print("=" * 60)
    print("✅ MLflow configurado correctamente")
    print("=" * 60)
    print(f"📊 Experimento: {experiment_name}")
    print(f"🌸 Dataset: Iris (Clasificación)")
    print(f"🤖 Modelo: Decision Tree Classifier")
    print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Paso 1: Configuración de MLflow
# MAGIC
# MAGIC Configuramos el experimento donde se registrarán todas las ejecuciones.
# MAGIC
# MAGIC **🔴 IMPORTANTE**: Cambia el email vacío por tu email de Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Paso 2: Importar Librerías
# MAGIC
# MAGIC Importamos todas las herramientas necesarias para clasificación.

# COMMAND ----------

# Librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para tracking
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("✅ Todas las librerías importadas correctamente")
print("   - MLflow: Listo para tracking")
print("   - Scikit-learn: Modelos y métricas")
print("   - Matplotlib & Seaborn: Visualizaciones")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌺 Paso 3: Cargar y Explorar el Dataset Iris
# MAGIC
# MAGIC Cargaremos el famoso dataset de flores Iris y exploraremos sus características.

# COMMAND ----------

# TODO: Carga el dataset Iris y explora su estructura.
# Pistas:
# - Usa datasets.load_iris()
# - Crea un DataFrame con dataset.data y dataset.feature_names
# - Añade la columna target/especie
# - Muestra shape, clases, primeras filas y estadísticas descriptivas

# Crear un DataFrame para mejor visualización
df = pd.DataFrame(
    data=dataset.data,
    columns=dataset.feature_names
)
df['species'] = dataset.target
df['species_name'] = df['species'].map({
    0: 'Setosa',
    1: 'Versicolor', 
    2: 'Virginica'
})

print("=" * 70)
print("🌸 DATASET IRIS - INFORMACIÓN GENERAL")
print("=" * 70)
print(f"📦 Número total de muestras: {len(df)}")
print(f"📊 Características (features): {len(dataset.feature_names)}")
print(f"🎯 Clases: {len(dataset.target_names)}")
print(f"\n🌺 Distribución de clases:")
print(df['species_name'].value_counts().to_string())

print(f"\n📏 Características del dataset:")
for i, feature in enumerate(dataset.feature_names, 1):
    print(f"   {i}. {feature}")

print("\n" + "=" * 70)
print("📊 PRIMERAS 5 MUESTRAS:")
print("=" * 70)
print(df.head().to_string())

print("\n" + "=" * 70)
print("📈 ESTADÍSTICAS DESCRIPTIVAS:")
print("=" * 70)
print(df.describe().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Visualización Exploratoria del Dataset
# MAGIC
# MAGIC Visualicemos las relaciones entre las características para entender mejor los datos.

# COMMAND ----------

# Crear visualizaciones exploratorias
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🌸 Análisis Exploratorio del Dataset Iris', fontsize=16, fontweight='bold')

# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# ...
# plt.show()

# Gráfico 2: Distribución de características por especie
ax2 = plt.subplot(2, 2, 2)
df_melted = df[['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'petal width (cm)', 'species_name']].melt(
    id_vars='species_name',
    var_name='feature',
    value_name='value'
)
sns.violinplot(data=df_melted, x='feature', y='value', hue='species_name', ax=ax2)
ax2.set_xlabel('Característica', fontsize=11)
ax2.set_ylabel('Valor (cm)', fontsize=11)
ax2.set_title('Distribución de Características por Especie', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Especie')

# Gráfico 3: Boxplot de longitud del pétalo
ax3 = plt.subplot(2, 2, 3)
df.boxplot(column='petal length (cm)', by='species_name', ax=ax3)
ax3.set_xlabel('Especie', fontsize=11)
ax3.set_ylabel('Petal Length (cm)', fontsize=11)
ax3.set_title('Distribución de Longitud del Pétalo', fontweight='bold')
plt.sca(ax3)
plt.xticks(rotation=45)

# Gráfico 4: Correlación entre características
ax4 = plt.subplot(2, 2, 4)
correlation = df[dataset.feature_names].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4,
            cbar_kws={'label': 'Correlación'})
ax4.set_title('Matriz de Correlación', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Visualizaciones generadas")
print("\n🔍 Observaciones clave:")
print("   - Las características del pétalo separan mejor las especies")
print("   - Setosa es claramente separable de las otras dos")
print("   - Versicolor y Virginica tienen mayor solapamiento")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔀 Paso 4: Preparar los Datos
# MAGIC
# MAGIC Dividimos el dataset en conjuntos de entrenamiento y prueba.

# COMMAND ----------

# TODO: Separa características (X) y etiquetas (y).
# X = ...
# y = ...

# TODO: Divide el dataset en entrenamiento y prueba.
# Pistas:
# - Usa train_test_split
# - Reserva una parte para test
# - Usa random_state para reproducibilidad
# - Considera stratify=y para mantener la proporción de clases

print("=" * 60)
print("🔀 DIVISIÓN DEL DATASET")
print("=" * 60)
print(f"📊 Total de muestras: {len(X)}")
print(f"\n📚 Conjunto de Entrenamiento:")
print(f"   - Muestras: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Shape: {X_train.shape}")
print(f"   - Distribución de clases:")
unique, counts = np.unique(y_train, return_counts=True)
for class_id, count in zip(unique, counts):
    print(f"     • {dataset.target_names[class_id]}: {count} muestras")

print(f"\n🧪 Conjunto de Prueba:")
print(f"   - Muestras: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"   - Shape: {X_test.shape}")
print(f"   - Distribución de clases:")
unique, counts = np.unique(y_test, return_counts=True)
for class_id, count in zip(unique, counts):
    print(f"     • {dataset.target_names[class_id]}: {count} muestras")

print("=" * 60)
print("✅ Datos preparados para entrenamiento")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌳 Paso 5: Entrenar Árbol de Decisión con MLflow
# MAGIC
# MAGIC Ahora entrenaremos un modelo de **Decision Tree** y registraremos todo con MLflow.
# MAGIC
# MAGIC ### 🔑 Hiperparámetros del Árbol de Decisión
# MAGIC
# MAGIC - **max_depth**: Profundidad máxima del árbol (evita overfitting)
# MAGIC - **max_features**: Número máximo de características a considerar por split
# MAGIC - Valores más altos = modelo más complejo = mayor riesgo de overfitting

# COMMAND ----------

# ========================================
# TODO: ENTRENAMIENTO CON MLFLOW
# ========================================

# TODO: Activa autologging de MLflow.
# mlflow.sklearn.autolog()

# Iniciar run con nombre descriptivo
with mlflow.start_run(run_name="Decision Tree - Iris Classifier") as run:
    
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
    print("=" * 70)
    
    # =====================================
    # 1. DEFINIR HIPERPARÁMETROS
    # =====================================
    max_depth = 10        # Profundidad máxima del árbol
    max_features = 2      # Número máximo de características por split
    min_samples_split = 2 # Mínimo de muestras para hacer un split
    min_samples_leaf = 1  # Mínimo de muestras en hoja
    random_state = 42     # Reproducibilidad
    
    print("\n⚙️  HIPERPARÁMETROS:")
    print(f"   - max_depth: {max_depth}")
    print(f"   - max_features: {max_features}")
    print(f"   - min_samples_split: {min_samples_split}")
    print(f"   - min_samples_leaf: {min_samples_leaf}")
    
    # =====================================
    # 2. CREAR Y ENTRENAR EL MODELO
    # =====================================
    print("\n🌳 Creando Árbol de Decisión...")
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    print("📚 Entrenando con", len(X_train), "muestras...")
    dt.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente")
    
    # =====================================
    # 3. HACER PREDICCIONES
    # =====================================
    print("\n🔮 Realizando predicciones...")
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    # Probabilidades para análisis adicional
    y_pred_proba_test = dt.predict_proba(X_test)
    
    print("✅ Predicciones completadas")
    
    # =====================================
    # 4. CALCULAR MÉTRICAS
    # =====================================
    print("\n📊 CALCULANDO MÉTRICAS DE CLASIFICACIÓN:")
    
    # Métricas de entrenamiento
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    
    # Métricas de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print("\n   📚 ENTRENAMIENTO:")
    print(f"      - Accuracy:  {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
    print(f"      - Precision: {precision_train:.4f}")
    print(f"      - Recall:    {recall_train:.4f}")
    print(f"      - F1-Score:  {f1_train:.4f}")
    
    print("\n   🧪 PRUEBA:")
    print(f"      - Accuracy:  {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"      - Precision: {precision_test:.4f}")
    print(f"      - Recall:    {recall_test:.4f}")
    print(f"      - F1-Score:  {f1_test:.4f}")
    
    print(f"\n   🔄 CROSS-VALIDATION (5-fold):")
    print(f"      - Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Calcular overfitting
    overfitting = accuracy_train - accuracy_test
    print(f"\n   ⚠️  Overfitting Score: {overfitting:.4f}")
    if overfitting < 0.05:
        print("      ✅ Excelente - Poco overfitting")
    elif overfitting < 0.10:
        print("      ⚡ Bueno - Overfitting moderado")
    else:
        print("      ⚠️  Cuidado - Posible overfitting")
    
    # =====================================
    # 5. REGISTRAR EN MLFLOW
    # =====================================
    print("\n📝 Registrando información en MLflow...")
    
    # Registrar hiperparámetros adicionales
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("dataset", "Iris")
    mlflow.log_param("n_classes", len(dataset.target_names))
    
    # Registrar métricas
    mlflow.log_metric("train_accuracy", accuracy_train)
    mlflow.log_metric("test_accuracy", accuracy_test)
    mlflow.log_metric("train_precision", precision_train)
    mlflow.log_metric("test_precision", precision_test)
    mlflow.log_metric("train_recall", recall_train)
    mlflow.log_metric("test_recall", recall_test)
    mlflow.log_metric("train_f1", f1_train)
    mlflow.log_metric("test_f1", f1_test)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("cv_std_accuracy", cv_std)
    mlflow.log_metric("overfitting_score", overfitting)
    
    # =====================================
    # 6. CREAR VISUALIZACIONES
    # =====================================
    print("\n📈 Generando visualizaciones...")
    
    # Figura con múltiples análisis
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Matriz de Confusión
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    ax1.set_title('Matriz de Confusión', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Verdadero', fontsize=10)
    ax1.set_xlabel('Predicción', fontsize=10)
    
    # 2. Visualización del Árbol
    ax2 = plt.subplot(2, 3, 2)
    plot_tree(dt, 
              feature_names=dataset.feature_names,
              class_names=dataset.target_names,
              filled=True,
              rounded=True,
              ax=ax2,
              fontsize=8)
    ax2.set_title('Estructura del Árbol de Decisión', fontweight='bold', fontsize=12)
    
    # 3. Importancia de Características
    ax3 = plt.subplot(2, 3, 3)
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=True)
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importancia', fontsize=10)
    ax3.set_title('Importancia de Características', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Comparación de Métricas
    ax4 = plt.subplot(2, 3, 4)
    metrics_comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Entrenamiento': [accuracy_train, precision_train, recall_train, f1_train],
        'Prueba': [accuracy_test, precision_test, recall_test, f1_test]
    })
    x = np.arange(len(metrics_comparison))
    width = 0.35
    ax4.bar(x - width/2, metrics_comparison['Entrenamiento'], width, label='Entrenamiento', alpha=0.8)
    ax4.bar(x + width/2, metrics_comparison['Prueba'], width, label='Prueba', alpha=0.8)
    ax4.set_xlabel('Métrica', fontsize=10)
    ax4.set_ylabel('Valor', fontsize=10)
    ax4.set_title('Comparación Train vs Test', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_comparison['Métrica'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # 5. Predicciones por clase
    ax5 = plt.subplot(2, 3, 5)
    class_accuracy = []
    for i, class_name in enumerate(dataset.target_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax5.bar(dataset.target_names, class_accuracy, alpha=0.7, edgecolor='black')
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy por Clase', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Distribución de confianza en predicciones
    ax6 = plt.subplot(2, 3, 6)
    max_probas = np.max(y_pred_proba_test, axis=1)
    ax6.hist(max_probas, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(max_probas.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {max_probas.mean():.3f}')
    ax6.set_xlabel('Confianza de Predicción', fontsize=10)
    ax6.set_ylabel('Frecuencia', fontsize=10)
    ax6.set_title('Distribución de Confianza', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🌸 Análisis Completo del Modelo Decision Tree - Iris', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar en MLflow
    mlflow.log_figure(fig, "model_analysis_complete.png")
    plt.show()
    
    print("✅ Visualizaciones guardadas")
    
    # =====================================
    # 7. REPORTE DE CLASIFICACIÓN
    # =====================================
    print("\n" + "=" * 70)
    print("📋 REPORTE DE CLASIFICACIÓN DETALLADO")
    print("=" * 70)
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=dataset.target_names,
        digits=4
    ))
    
    # =====================================
    # 8. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    
    print("=" * 70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name if email else 'No configurado'}")
    print(f"🎯 Accuracy (Test): {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"🌟 F1-Score (Test): {f1_test:.4f}")
    print(f"📊 Cross-Val Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print("=" * 70)

mlflow.end_run()
print("\n✅ Experimento finalizado correctamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 ¡Excelente Trabajo! Ejercicio Completado
# MAGIC
# MAGIC Completa esta sección cuando termines el notebook.
# MAGIC
# MAGIC ### ✅ Comprueba que has trabajado
# MAGIC
# MAGIC 1. [ ] Exploración del dataset Iris
# MAGIC 2. [ ] Separación train/test
# MAGIC 3. [ ] Entrenamiento de un Árbol de Decisión
# MAGIC 4. [ ] Evaluación con métricas de clasificación
# MAGIC 5. [ ] Registro del experimento en MLflow
# MAGIC 6. [ ] Interpretación de resultados
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Conceptos Clave - Clasificación
# MAGIC
# MAGIC ### 🎯 Métricas de Clasificación
# MAGIC
# MAGIC | Métrica | Descripción | Cuándo Usarla |
# MAGIC |---------|-------------|---------------|
# MAGIC | **Accuracy** | % de predicciones correctas | Clases balanceadas |
# MAGIC | **Precision** | % de positivos correctos | Importante evitar falsos positivos |
# MAGIC | **Recall** | % de positivos encontrados | Importante encontrar todos los positivos |
# MAGIC | **F1-Score** | Media armónica de P y R | Balance entre precisión y recall |
# MAGIC
# MAGIC ### 🌳 Árbol de Decisión
# MAGIC
# MAGIC **Ventajas:**
# MAGIC - ✅ Fácil de interpretar y visualizar
# MAGIC - ✅ No requiere normalización de datos
# MAGIC - ✅ Maneja datos numéricos y categóricos
# MAGIC - ✅ Captura relaciones no lineales
# MAGIC
# MAGIC **Desventajas:**
# MAGIC - ⚠️ Propenso a overfitting
# MAGIC - ⚠️ Inestable ante pequeños cambios
# MAGIC - ⚠️ Puede crear sesgos con clases desbalanceadas
# MAGIC
# MAGIC ### 📊 Matriz de Confusión
# MAGIC
# MAGIC ```
# MAGIC                 Predicción
# MAGIC               Setosa  Versi  Virgin
# MAGIC Real Setosa     [TP]   [FP]   [FP]
# MAGIC      Versi      [FN]   [TP]   [FP]
# MAGIC      Virgin     [FN]   [FN]   [TP]
# MAGIC ```
# MAGIC
# MAGIC - **TP (True Positive)**: Predicción correcta
# MAGIC - **FP (False Positive)**: Error tipo I
# MAGIC - **FN (False Negative)**: Error tipo II
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Desafíos Adicionales
# MAGIC
# MAGIC ### 🎯 Desafío 1: Optimiza los Hiperparámetros
# MAGIC
# MAGIC Prueba diferentes configuraciones y encuentra la mejor:
# MAGIC
# MAGIC ```python
# MAGIC # Experimenta con:
# MAGIC max_depth = [3, 5, 10, None]
# MAGIC max_features = [2, 3, 4, 'sqrt']
# MAGIC min_samples_split = [2, 5, 10]
# MAGIC ```
# MAGIC
# MAGIC **TODO:** ¿Qué combinación da el mejor balance entre accuracy y overfitting?
# MAGIC
# MAGIC ### 🎯 Desafío 2: Implementa Grid Search
# MAGIC
# MAGIC Automatiza la búsqueda del mejor modelo con `GridSearchCV`.
# MAGIC
# MAGIC ### 🎯 Desafío 3: Compara con Otros Modelos
# MAGIC
# MAGIC Entrena y compara Random Forest, SVM, KNN o Logistic Regression.
# MAGIC
# MAGIC ### 🎯 Desafío 4: Análisis de Errores
# MAGIC
# MAGIC **TODO:** ¿Qué flores se confunden más y por qué?
# MAGIC
# MAGIC ### 🎯 Desafío 5: Reducción de Dimensionalidad
# MAGIC
# MAGIC Usa solo 2 características y visualiza:
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.decomposition import PCA
# MAGIC
# MAGIC # Reducir a 2D
# MAGIC pca = PCA(n_components=2)
# MAGIC X_pca = pca.fit_transform(X)
# MAGIC
# MAGIC # Entrenar y visualizar límites de decisión
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Tips para Mejorar
# MAGIC
# MAGIC ### 🎨 Mejora la Organización en MLflow
# MAGIC
# MAGIC ```python
# MAGIC # Agregar tags descriptivos
# MAGIC mlflow.set_tag("tipo_modelo", "decision_tree")
# MAGIC mlflow.set_tag("dataset", "iris")
# MAGIC mlflow.set_tag("objetivo", "clasificacion_multiclase")
# MAGIC mlflow.set_tag("version", "1.0")
# MAGIC mlflow.set_tag("autor", "tu_nombre")
# MAGIC
# MAGIC # Agregar notas
# MAGIC mlflow.set_tag("notas", "Primer experimento baseline")
# MAGIC ```
# MAGIC
# MAGIC ### 📊 Experimenta con Visualizaciones
# MAGIC
# MAGIC ```python
# MAGIC # Límites de decisión en 2D
# MAGIC from matplotlib.colors import ListedColormap
# MAGIC # ... código para visualizar límites ...
# MAGIC
# MAGIC # Curvas de aprendizaje
# MAGIC from sklearn.model_selection import learning_curve
# MAGIC # ... código para curvas de aprendizaje ...
# MAGIC ```
# MAGIC
# MAGIC ### 🔄 Validación Cruzada Estratificada
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import StratifiedKFold
# MAGIC
# MAGIC skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# MAGIC cv_scores = cross_val_score(dt, X, y, cv=skf)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Recursos Adicionales
# MAGIC
# MAGIC ### 📖 Documentación
# MAGIC - [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
# MAGIC - [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
# MAGIC - [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
# MAGIC
# MAGIC ### 🎥 Tutoriales
# MAGIC - Interpretación de matrices de confusión
# MAGIC - Árboles de decisión explicados visualmente
# MAGIC - Optimización de hiperparámetros
# MAGIC
# MAGIC ### 📊 Datasets Similares
# MAGIC - **Wine Quality**: Clasificación de vinos
# MAGIC - **Digits**: Reconocimiento de dígitos escritos
# MAGIC - **Breast Cancer**: Diagnóstico médico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🌟 Próximos Pasos
# MAGIC
# MAGIC 1. **Completa los desafíos** propuestos
# MAGIC 2. **Compara tus resultados** en MLflow UI
# MAGIC 3. **Documenta tus hallazgos** con tags y notas
# MAGIC 4. **Experimenta con otros datasets** de clasificación
# MAGIC 5. **Aprende sobre ensembles** (Random Forest, Gradient Boosting)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 ¡Felicitaciones!
# MAGIC
# MAGIC Has completado con éxito el ejercicio de clasificación de Iris con MLflow. Ahora tienes las habilidades para:
# MAGIC
# MAGIC - ✅ Trabajar con problemas de clasificación multiclase
# MAGIC - ✅ Evaluar modelos con métricas apropiadas
# MAGIC - ✅ Interpretar matrices de confusión
# MAGIC - ✅ Visualizar resultados de clasificación
# MAGIC - ✅ Documentar experimentos profesionalmente
# MAGIC
# MAGIC **¡Sigue practicando y mejorando tus habilidades en Machine Learning! 🚀🌸**

# COMMAND ----------

# ========================================
# ENTRENAMIENTO CON MLFLOW
# ========================================
# Librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para tracking
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Configurar email y experimento
email = 'rafaelsanchezupo@gmail.com'
experiment_name = f"/Users/{email}/4-ejercicio-the-irish"

# Cargar el dataset Iris
dataset = datasets.load_iris()

# Activar autologging de MLflow
mlflow.sklearn.autolog()
# Separar características (X) y etiquetas (y)
X = dataset.data
y = dataset.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% para prueba
    random_state=42,     # Reproducibilidad
    stratify=y           # Mantener proporción de clases
)
# Iniciar run con nombre descriptivo
with mlflow.start_run(run_name="Decision Tree - Iris Classifier") as run:
    
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
    print("=" * 70)
    
    # =====================================
    # 1. DEFINIR HIPERPARÁMETROS
    # =====================================
    max_depth = 3        # Profundidad máxima del árbol
    max_features = 2      # Número máximo de características por split
    min_samples_split = 2 # Mínimo de muestras para hacer un split
    min_samples_leaf = 1  # Mínimo de muestras en hoja
    random_state = 42     # Reproducibilidad
    
    print("\n⚙️  HIPERPARÁMETROS:")
    print(f"   - max_depth: {max_depth}")
    print(f"   - max_features: {max_features}")
    print(f"   - min_samples_split: {min_samples_split}")
    print(f"   - min_samples_leaf: {min_samples_leaf}")
    
    # =====================================
    # 2. CREAR Y ENTRENAR EL MODELO
    # =====================================
    print("\n🌳 Creando Árbol de Decisión...")
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    print("📚 Entrenando con", len(X_train), "muestras...")
    dt.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente")
    
    # =====================================
    # 3. HACER PREDICCIONES
    # =====================================
    print("\n🔮 Realizando predicciones...")
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    # Probabilidades para análisis adicional
    y_pred_proba_test = dt.predict_proba(X_test)
    
    print("✅ Predicciones completadas")
    
    # =====================================
    # 4. CALCULAR MÉTRICAS
    # =====================================
    print("\n📊 CALCULANDO MÉTRICAS DE CLASIFICACIÓN:")
    
    # Métricas de entrenamiento
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    
    # Métricas de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print("\n   📚 ENTRENAMIENTO:")
    print(f"      - Accuracy:  {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
    print(f"      - Precision: {precision_train:.4f}")
    print(f"      - Recall:    {recall_train:.4f}")
    print(f"      - F1-Score:  {f1_train:.4f}")
    
    print("\n   🧪 PRUEBA:")
    print(f"      - Accuracy:  {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"      - Precision: {precision_test:.4f}")
    print(f"      - Recall:    {recall_test:.4f}")
    print(f"      - F1-Score:  {f1_test:.4f}")
    
    print(f"\n   🔄 CROSS-VALIDATION (5-fold):")
    print(f"      - Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Calcular overfitting
    overfitting = accuracy_train - accuracy_test
    print(f"\n   ⚠️  Overfitting Score: {overfitting:.4f}")
    if overfitting < 0.05:
        print("      ✅ Excelente - Poco overfitting")
    elif overfitting < 0.10:
        print("      ⚡ Bueno - Overfitting moderado")
    else:
        print("      ⚠️  Cuidado - Posible overfitting")
    
    # =====================================
    # 5. REGISTRAR EN MLFLOW
    # =====================================
    print("\n📝 Registrando información en MLflow...")
    
    # Registrar hiperparámetros adicionales
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("dataset", "Iris")
    mlflow.log_param("n_classes", len(dataset.target_names))
    
    # Registrar métricas
    mlflow.log_metric("train_accuracy", accuracy_train)
    mlflow.log_metric("test_accuracy", accuracy_test)
    mlflow.log_metric("train_precision", precision_train)
    mlflow.log_metric("test_precision", precision_test)
    mlflow.log_metric("train_recall", recall_train)
    mlflow.log_metric("test_recall", recall_test)
    mlflow.log_metric("train_f1", f1_train)
    mlflow.log_metric("test_f1", f1_test)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("cv_std_accuracy", cv_std)
    mlflow.log_metric("overfitting_score", overfitting)
    
    # Crear figura para visualizaciones
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Matriz de Confusión
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    ax1.set_title('Matriz de Confusión', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Verdadero', fontsize=10)
    ax1.set_xlabel('Predicción', fontsize=10)
    
    # 2. Visualización del Árbol
    ax2 = plt.subplot(2, 3, 2)
    plot_tree(dt, 
              feature_names=dataset.feature_names,
              class_names=dataset.target_names,
              filled=True,
              rounded=True,
              ax=ax2,
              fontsize=8)
    ax2.set_title('Estructura del Árbol de Decisión', fontweight='bold', fontsize=12)
    
    # 3. Importancia de Características
    ax3 = plt.subplot(2, 3, 3)
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=True)
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importancia', fontsize=10)
    ax3.set_title('Importancia de Características', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Comparación de Métricas
    ax4 = plt.subplot(2, 3, 4)
    metrics_comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Entrenamiento': [accuracy_train, precision_train, recall_train, f1_train],
        'Prueba': [accuracy_test, precision_test, recall_test, f1_test]
    })
    x = np.arange(len(metrics_comparison))
    width = 0.35
    ax4.bar(x - width/2, metrics_comparison['Entrenamiento'], width, label='Entrenamiento', alpha=0.8)
    ax4.bar(x + width/2, metrics_comparison['Prueba'], width, label='Prueba', alpha=0.8)
    ax4.set_xlabel('Métrica', fontsize=10)
    ax4.set_ylabel('Valor', fontsize=10)
    ax4.set_title('Comparación Train vs Test', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_comparison['Métrica'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # 5. Predicciones por clase
    ax5 = plt.subplot(2, 3, 5)
    class_accuracy = []
    for i, class_name in enumerate(dataset.target_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax5.bar(dataset.target_names, class_accuracy, alpha=0.7, edgecolor='black')
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy por Clase', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Distribución de confianza en predicciones
    ax6 = plt.subplot(2, 3, 6)
    max_probas = np.max(y_pred_proba_test, axis=1)
    ax6.hist(max_probas, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(max_probas.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {max_probas.mean():.3f}')
    ax6.set_xlabel('Confianza de Predicción', fontsize=10)
    ax6.set_ylabel('Frecuencia', fontsize=10)
    ax6.set_title('Distribución de Confianza', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🌸 Análisis Completo del Modelo Decision Tree - Iris', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar en MLflow
    mlflow.log_figure(fig, "model_analysis_complete1.png")
    plt.show()
    
    print("✅ Visualizaciones guardadas")
    
    # =====================================
    # 7. REPORTE DE CLASIFICACIÓN
    # =====================================
    print("\n" + "=" * 70)
    print("📋 REPORTE DE CLASIFICACIÓN DETALLADO")
    print("=" * 70)
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=dataset.target_names,
        digits=4
    ))
    
    # =====================================
    # 8. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    
    print("=" * 70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name if email else 'No configurado'}")
    print(f"🎯 Accuracy (Test): {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"🌟 F1-Score (Test): {f1_test:.4f}")
    print(f"📊 Cross-Val Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print("=" * 70)

mlflow.end_run()
print("\n✅ Experimento finalizado correctamente")

# COMMAND ----------

# DBTITLE 1,Experimento 2 - Árbol de Decisión Iris (Copia 1)
# ========================================
# ENTRENAMIENTO CON MLFLOW
# ========================================
# Librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para tracking
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Configurar email y experimento
email = 'rafaelsanchezupo@gmail.com'
experiment_name = f"/Users/{email}/4-ejercicio-the-irish"

# Cargar el dataset Iris
dataset = datasets.load_iris()

# Activar autologging de MLflow
mlflow.sklearn.autolog()
# Separar características (X) y etiquetas (y)
X = dataset.data
y = dataset.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% para prueba
    random_state=42,     # Reproducibilidad
    stratify=y           # Mantener proporción de clases
)
# Iniciar run con nombre descriptivo
with mlflow.start_run(run_name="Decision Tree - Iris Classifier") as run:
    
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
    print("=" * 70)
    
    # =====================================
    # 1. DEFINIR HIPERPARÁMETROS
    # =====================================
    max_depth = 5        # Profundidad máxima del árbol
    max_features = 3      # Número máximo de características por split
    min_samples_split = 5 # Mínimo de muestras para hacer un split
    min_samples_leaf = 2  # Mínimo de muestras en hoja
    random_state = 42     # Reproducibilidad
    
    print("\n⚙️  HIPERPARÁMETROS:")
    print(f"   - max_depth: {max_depth}")
    print(f"   - max_features: {max_features}")
    print(f"   - min_samples_split: {min_samples_split}")
    print(f"   - min_samples_leaf: {min_samples_leaf}")
    
    # =====================================
    # 2. CREAR Y ENTRENAR EL MODELO
    # =====================================
    print("\n🌳 Creando Árbol de Decisión...")
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    print("📚 Entrenando con", len(X_train), "muestras...")
    dt.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente")
    
    # =====================================
    # 3. HACER PREDICCIONES
    # =====================================
    print("\n🔮 Realizando predicciones...")
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    # Probabilidades para análisis adicional
    y_pred_proba_test = dt.predict_proba(X_test)
    
    print("✅ Predicciones completadas")
    
    # =====================================
    # 4. CALCULAR MÉTRICAS
    # =====================================
    print("\n📊 CALCULANDO MÉTRICAS DE CLASIFICACIÓN:")
    
    # Métricas de entrenamiento
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    
    # Métricas de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print("\n   📚 ENTRENAMIENTO:")
    print(f"      - Accuracy:  {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
    print(f"      - Precision: {precision_train:.4f}")
    print(f"      - Recall:    {recall_train:.4f}")
    print(f"      - F1-Score:  {f1_train:.4f}")
    
    print("\n   🧪 PRUEBA:")
    print(f"      - Accuracy:  {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"      - Precision: {precision_test:.4f}")
    print(f"      - Recall:    {recall_test:.4f}")
    print(f"      - F1-Score:  {f1_test:.4f}")
    
    print(f"\n   🔄 CROSS-VALIDATION (5-fold):")
    print(f"      - Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Calcular overfitting
    overfitting = accuracy_train - accuracy_test
    print(f"\n   ⚠️  Overfitting Score: {overfitting:.4f}")
    if overfitting < 0.05:
        print("      ✅ Excelente - Poco overfitting")
    elif overfitting < 0.10:
        print("      ⚡ Bueno - Overfitting moderado")
    else:
        print("      ⚠️  Cuidado - Posible overfitting")
    
    # =====================================
    # 5. REGISTRAR EN MLFLOW
    # =====================================
    print("\n📝 Registrando información en MLflow...")
    
    # Registrar hiperparámetros adicionales
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("dataset", "Iris")
    mlflow.log_param("n_classes", len(dataset.target_names))
    
    # Registrar métricas
    mlflow.log_metric("train_accuracy", accuracy_train)
    mlflow.log_metric("test_accuracy", accuracy_test)
    mlflow.log_metric("train_precision", precision_train)
    mlflow.log_metric("test_precision", precision_test)
    mlflow.log_metric("train_recall", recall_train)
    mlflow.log_metric("test_recall", recall_test)
    mlflow.log_metric("train_f1", f1_train)
    mlflow.log_metric("test_f1", f1_test)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("cv_std_accuracy", cv_std)
    mlflow.log_metric("overfitting_score", overfitting)
    
    # Crear figura para visualizaciones
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Matriz de Confusión
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    ax1.set_title('Matriz de Confusión', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Verdadero', fontsize=10)
    ax1.set_xlabel('Predicción', fontsize=10)
    
    # 2. Visualización del Árbol
    ax2 = plt.subplot(2, 3, 2)
    plot_tree(dt, 
              feature_names=dataset.feature_names,
              class_names=dataset.target_names,
              filled=True,
              rounded=True,
              ax=ax2,
              fontsize=8)
    ax2.set_title('Estructura del Árbol de Decisión', fontweight='bold', fontsize=12)
    
    # 3. Importancia de Características
    ax3 = plt.subplot(2, 3, 3)
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=True)
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importancia', fontsize=10)
    ax3.set_title('Importancia de Características', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Comparación de Métricas
    ax4 = plt.subplot(2, 3, 4)
    metrics_comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Entrenamiento': [accuracy_train, precision_train, recall_train, f1_train],
        'Prueba': [accuracy_test, precision_test, recall_test, f1_test]
    })
    x = np.arange(len(metrics_comparison))
    width = 0.35
    ax4.bar(x - width/2, metrics_comparison['Entrenamiento'], width, label='Entrenamiento', alpha=0.8)
    ax4.bar(x + width/2, metrics_comparison['Prueba'], width, label='Prueba', alpha=0.8)
    ax4.set_xlabel('Métrica', fontsize=10)
    ax4.set_ylabel('Valor', fontsize=10)
    ax4.set_title('Comparación Train vs Test', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_comparison['Métrica'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # 5. Predicciones por clase
    ax5 = plt.subplot(2, 3, 5)
    class_accuracy = []
    for i, class_name in enumerate(dataset.target_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax5.bar(dataset.target_names, class_accuracy, alpha=0.7, edgecolor='black')
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy por Clase', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Distribución de confianza en predicciones
    ax6 = plt.subplot(2, 3, 6)
    max_probas = np.max(y_pred_proba_test, axis=1)
    ax6.hist(max_probas, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(max_probas.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {max_probas.mean():.3f}')
    ax6.set_xlabel('Confianza de Predicción', fontsize=10)
    ax6.set_ylabel('Frecuencia', fontsize=10)
    ax6.set_title('Distribución de Confianza', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🌸 Análisis Completo del Modelo Decision Tree - Iris', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar en MLflow
    mlflow.log_figure(fig, "model_analysis_complete1.png")
    plt.show()
    
    print("✅ Visualizaciones guardadas")
    
    # =====================================
    # 7. REPORTE DE CLASIFICACIÓN
    # =====================================
    print("\n" + "=" * 70)
    print("📋 REPORTE DE CLASIFICACIÓN DETALLADO")
    print("=" * 70)
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=dataset.target_names,
        digits=4
    ))
    
    # =====================================
    # 8. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    
    print("=" * 70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name if email else 'No configurado'}")
    print(f"🎯 Accuracy (Test): {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"🌟 F1-Score (Test): {f1_test:.4f}")
    print(f"📊 Cross-Val Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print("=" * 70)

mlflow.end_run()
print("\n✅ Experimento finalizado correctamente")

# COMMAND ----------

# DBTITLE 1,Experimento 3 - Árbol de Decisión Iris (Copia 2)
# ========================================
# ENTRENAMIENTO CON MLFLOW
# ========================================
# Librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para tracking
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Configurar email y experimento
email = 'rafaelsanchezupo@gmail.com'
experiment_name = f"/Users/{email}/4-ejercicio-the-irish"

# Cargar el dataset Iris
dataset = datasets.load_iris()

# Activar autologging de MLflow
mlflow.sklearn.autolog()
# Separar características (X) y etiquetas (y)
X = dataset.data
y = dataset.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% para prueba
    random_state=42,     # Reproducibilidad
    stratify=y           # Mantener proporción de clases
)
# Iniciar run con nombre descriptivo
with mlflow.start_run(run_name="Decision Tree - Iris Classifier") as run:
    
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
    print("=" * 70)
    
    # =====================================
    # 1. DEFINIR HIPERPARÁMETROS
    # =====================================
    max_depth = 10        # Profundidad máxima del árbol
    max_features = 4      # Número máximo de características por split
    min_samples_split = 10 # Mínimo de muestras para hacer un split
    min_samples_leaf = 4  # Mínimo de muestras en hoja
    random_state = 42     # Reproducibilidad
    
    print("\n⚙️  HIPERPARÁMETROS:")
    print(f"   - max_depth: {max_depth}")
    print(f"   - max_features: {max_features}")
    print(f"   - min_samples_split: {min_samples_split}")
    print(f"   - min_samples_leaf: {min_samples_leaf}")
    
    # =====================================
    # 2. CREAR Y ENTRENAR EL MODELO
    # =====================================
    print("\n🌳 Creando Árbol de Decisión...")
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    print("📚 Entrenando con", len(X_train), "muestras...")
    dt.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente")
    
    # =====================================
    # 3. HACER PREDICCIONES
    # =====================================
    print("\n🔮 Realizando predicciones...")
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    # Probabilidades para análisis adicional
    y_pred_proba_test = dt.predict_proba(X_test)
    
    print("✅ Predicciones completadas")
    
    # =====================================
    # 4. CALCULAR MÉTRICAS
    # =====================================
    print("\n📊 CALCULANDO MÉTRICAS DE CLASIFICACIÓN:")
    
    # Métricas de entrenamiento
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    
    # Métricas de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print("\n   📚 ENTRENAMIENTO:")
    print(f"      - Accuracy:  {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
    print(f"      - Precision: {precision_train:.4f}")
    print(f"      - Recall:    {recall_train:.4f}")
    print(f"      - F1-Score:  {f1_train:.4f}")
    
    print("\n   🧪 PRUEBA:")
    print(f"      - Accuracy:  {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"      - Precision: {precision_test:.4f}")
    print(f"      - Recall:    {recall_test:.4f}")
    print(f"      - F1-Score:  {f1_test:.4f}")
    
    print(f"\n   🔄 CROSS-VALIDATION (5-fold):")
    print(f"      - Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Calcular overfitting
    overfitting = accuracy_train - accuracy_test
    print(f"\n   ⚠️  Overfitting Score: {overfitting:.4f}")
    if overfitting < 0.05:
        print("      ✅ Excelente - Poco overfitting")
    elif overfitting < 0.10:
        print("      ⚡ Bueno - Overfitting moderado")
    else:
        print("      ⚠️  Cuidado - Posible overfitting")
    
    # =====================================
    # 5. REGISTRAR EN MLFLOW
    # =====================================
    print("\n📝 Registrando información en MLflow...")
    
    # Registrar hiperparámetros adicionales
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("dataset", "Iris")
    mlflow.log_param("n_classes", len(dataset.target_names))
    
    # Registrar métricas
    mlflow.log_metric("train_accuracy", accuracy_train)
    mlflow.log_metric("test_accuracy", accuracy_test)
    mlflow.log_metric("train_precision", precision_train)
    mlflow.log_metric("test_precision", precision_test)
    mlflow.log_metric("train_recall", recall_train)
    mlflow.log_metric("test_recall", recall_test)
    mlflow.log_metric("train_f1", f1_train)
    mlflow.log_metric("test_f1", f1_test)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("cv_std_accuracy", cv_std)
    mlflow.log_metric("overfitting_score", overfitting)
    
    # Crear figura para visualizaciones
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Matriz de Confusión
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    ax1.set_title('Matriz de Confusión', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Verdadero', fontsize=10)
    ax1.set_xlabel('Predicción', fontsize=10)
    
    # 2. Visualización del Árbol
    ax2 = plt.subplot(2, 3, 2)
    plot_tree(dt, 
              feature_names=dataset.feature_names,
              class_names=dataset.target_names,
              filled=True,
              rounded=True,
              ax=ax2,
              fontsize=8)
    ax2.set_title('Estructura del Árbol de Decisión', fontweight='bold', fontsize=12)
    
    # 3. Importancia de Características
    ax3 = plt.subplot(2, 3, 3)
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=True)
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importancia', fontsize=10)
    ax3.set_title('Importancia de Características', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Comparación de Métricas
    ax4 = plt.subplot(2, 3, 4)
    metrics_comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Entrenamiento': [accuracy_train, precision_train, recall_train, f1_train],
        'Prueba': [accuracy_test, precision_test, recall_test, f1_test]
    })
    x = np.arange(len(metrics_comparison))
    width = 0.35
    ax4.bar(x - width/2, metrics_comparison['Entrenamiento'], width, label='Entrenamiento', alpha=0.8)
    ax4.bar(x + width/2, metrics_comparison['Prueba'], width, label='Prueba', alpha=0.8)
    ax4.set_xlabel('Métrica', fontsize=10)
    ax4.set_ylabel('Valor', fontsize=10)
    ax4.set_title('Comparación Train vs Test', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_comparison['Métrica'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # 5. Predicciones por clase
    ax5 = plt.subplot(2, 3, 5)
    class_accuracy = []
    for i, class_name in enumerate(dataset.target_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax5.bar(dataset.target_names, class_accuracy, alpha=0.7, edgecolor='black')
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy por Clase', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Distribución de confianza en predicciones
    ax6 = plt.subplot(2, 3, 6)
    max_probas = np.max(y_pred_proba_test, axis=1)
    ax6.hist(max_probas, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(max_probas.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {max_probas.mean():.3f}')
    ax6.set_xlabel('Confianza de Predicción', fontsize=10)
    ax6.set_ylabel('Frecuencia', fontsize=10)
    ax6.set_title('Distribución de Confianza', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🌸 Análisis Completo del Modelo Decision Tree - Iris', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar en MLflow
    mlflow.log_figure(fig, "model_analysis_complete1.png")
    plt.show()
    
    print("✅ Visualizaciones guardadas")
    
    # =====================================
    # 7. REPORTE DE CLASIFICACIÓN
    # =====================================
    print("\n" + "=" * 70)
    print("📋 REPORTE DE CLASIFICACIÓN DETALLADO")
    print("=" * 70)
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=dataset.target_names,
        digits=4
    ))
    
    # =====================================
    # 8. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    
    print("=" * 70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name if email else 'No configurado'}")
    print(f"🎯 Accuracy (Test): {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"🌟 F1-Score (Test): {f1_test:.4f}")
    print(f"📊 Cross-Val Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print("=" * 70)

mlflow.end_run()
print("\n✅ Experimento finalizado correctamente")

# COMMAND ----------

# DBTITLE 1,Experimento 4 - Grid Search con Múltiples Hiperparámetros
# ========================================
# ENTRENAMIENTO CON MLFLOW - GRID SEARCH
# ========================================
# Librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para tracking
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Configurar email y experimento
email = 'rafaelsanchezupo@gmail.com'
experiment_name = f"/Users/{email}/4-ejercicio-the-irish"

# Cargar el dataset Iris
dataset = datasets.load_iris()

# Activar autologging de MLflow
mlflow.sklearn.autolog()
# Separar características (X) y etiquetas (y)
X = dataset.data
y = dataset.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% para prueba
    random_state=42,     # Reproducibilidad
    stratify=y           # Mantener proporción de clases
)

# Iniciar run con nombre descriptivo
with mlflow.start_run(run_name="Grid Search - Decision Tree") as run:
    
    print("=" * 70)
    print("🔍 GRID SEARCH - BÚSQUEDA DE HIPERPARÁMETROS")
    print("=" * 70)
    
    # =====================================
    # 1. DEFINIR GRID DE HIPERPARÁMETROS
    # =====================================
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'max_features': [2, 3, 4],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("\n⚙️  CONFIGURACIÓN DE BÚSQUEDA:")
    print(f"   - Parámetros a explorar: {len(param_grid)}")
    print(f"   - max_depth: {param_grid['max_depth']}")
    print(f"   - max_features: {param_grid['max_features']}")
    print(f"   - min_samples_split: {param_grid['min_samples_split']}")
    print(f"   - min_samples_leaf: {param_grid['min_samples_leaf']}")
    print(f"   - Combinaciones totales: {4 * 3 * 3 * 3} = 108 modelos")
    print(f"   - Cross-validation: 5 folds")
    print(f"   - Total entrenamientos: 540")
    
    # =====================================
    # 2. EJECUTAR GRID SEARCH
    # =====================================
    print("\n🌳 Creando Grid Search...")
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    print("📚 Entrenando con", len(X_train), "muestras...")
    print("⏳ Esto puede tomar un momento...\n")
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo encontrado
    dt = grid_search.best_estimator_
    print("\n✅ Grid Search completado exitosamente")
    
    print("\n🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
    for param, value in grid_search.best_params_.items():
        print(f"   - {param}: {value}")
    
    # =====================================
    # 3. HACER PREDICCIONES
    # =====================================
    print("\n🔮 Realizando predicciones con mejor modelo...")
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    # Probabilidades para análisis adicional
    y_pred_proba_test = dt.predict_proba(X_test)
    
    print("✅ Predicciones completadas")
    
    # =====================================
    # 4. CALCULAR MÉTRICAS
    # =====================================
    print("\n📊 CALCULANDO MÉTRICAS DE CLASIFICACIÓN:")
    
    # Métricas de entrenamiento
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    
    # Métricas de prueba
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    # Cross-validation
    cv_mean = grid_search.best_score_
    cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    
    print("\n   📚 ENTRENAMIENTO:")
    print(f"      - Accuracy:  {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
    print(f"      - Precision: {precision_train:.4f}")
    print(f"      - Recall:    {recall_train:.4f}")
    print(f"      - F1-Score:  {f1_train:.4f}")
    
    print("\n   🧪 PRUEBA:")
    print(f"      - Accuracy:  {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"      - Precision: {precision_test:.4f}")
    print(f"      - Recall:    {recall_test:.4f}")
    print(f"      - F1-Score:  {f1_test:.4f}")
    
    print(f"\n   🔄 CROSS-VALIDATION (5-fold):")
    print(f"      - Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Calcular overfitting
    overfitting = accuracy_train - accuracy_test
    print(f"\n   ⚠️  Overfitting Score: {overfitting:.4f}")
    if overfitting < 0.05:
        print("      ✅ Excelente - Poco overfitting")
    elif overfitting < 0.10:
        print("      ⚡ Bueno - Overfitting moderado")
    else:
        print("      ⚠️  Cuidado - Posible overfitting")
    
    # Top 5 configuraciones
    print("\n🔝 TOP 5 MEJORES CONFIGURACIONES:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nsmallest(5, 'rank_test_score')[[
        'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
    ]]
    
    for idx, row in top_5.iterrows():
        rank = int(row['rank_test_score'])
        score = row['mean_test_score']
        std = row['std_test_score']
        params = row['params']
        print(f"   #{rank}: Accuracy = {score:.4f} ± {std:.4f}")
        print(f"        {params}")
    
    # =====================================
    # 5. REGISTRAR EN MLFLOW
    # =====================================
    print("\n📝 Registrando información en MLflow...")
    
    # Registrar parámetros del grid search
    mlflow.log_param("grid_search", True)
    mlflow.log_param("total_combinations", 108)
    mlflow.log_param("dataset", "Iris")
    mlflow.log_param("n_classes", len(dataset.target_names))
    
    # Registrar métricas
    mlflow.log_metric("train_accuracy", accuracy_train)
    mlflow.log_metric("test_accuracy", accuracy_test)
    mlflow.log_metric("train_precision", precision_train)
    mlflow.log_metric("test_precision", precision_test)
    mlflow.log_metric("train_recall", recall_train)
    mlflow.log_metric("test_recall", recall_test)
    mlflow.log_metric("train_f1", f1_train)
    mlflow.log_metric("test_f1", f1_test)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("cv_std_accuracy", cv_std)
    mlflow.log_metric("overfitting_score", overfitting)
    
    # Crear figura para visualizaciones
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Matriz de Confusión
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    ax1.set_title('Matriz de Confusión', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Verdadero', fontsize=10)
    ax1.set_xlabel('Predicción', fontsize=10)
    
    # 2. Visualización del Árbol
    ax2 = plt.subplot(2, 3, 2)
    plot_tree(dt, 
              feature_names=dataset.feature_names,
              class_names=dataset.target_names,
              filled=True,
              rounded=True,
              ax=ax2,
              fontsize=8)
    ax2.set_title('Mejor Árbol (Grid Search)', fontweight='bold', fontsize=12)
    
    # 3. Importancia de Características
    ax3 = plt.subplot(2, 3, 3)
    feature_importance = pd.DataFrame({
        'feature': dataset.feature_names,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=True)
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importancia', fontsize=10)
    ax3.set_title('Importancia de Características', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Comparación de Métricas
    ax4 = plt.subplot(2, 3, 4)
    metrics_comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Entrenamiento': [accuracy_train, precision_train, recall_train, f1_train],
        'Prueba': [accuracy_test, precision_test, recall_test, f1_test]
    })
    x = np.arange(len(metrics_comparison))
    width = 0.35
    ax4.bar(x - width/2, metrics_comparison['Entrenamiento'], width, label='Entrenamiento', alpha=0.8)
    ax4.bar(x + width/2, metrics_comparison['Prueba'], width, label='Prueba', alpha=0.8)
    ax4.set_xlabel('Métrica', fontsize=10)
    ax4.set_ylabel('Valor', fontsize=10)
    ax4.set_title('Comparación Train vs Test', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_comparison['Métrica'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # 5. Predicciones por clase
    ax5 = plt.subplot(2, 3, 5)
    class_accuracy = []
    for i, class_name in enumerate(dataset.target_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax5.bar(dataset.target_names, class_accuracy, alpha=0.7, edgecolor='black')
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy por Clase', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Distribución de confianza en predicciones
    ax6 = plt.subplot(2, 3, 6)
    max_probas = np.max(y_pred_proba_test, axis=1)
    ax6.hist(max_probas, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(max_probas.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {max_probas.mean():.3f}')
    ax6.set_xlabel('Confianza de Predicción', fontsize=10)
    ax6.set_ylabel('Frecuencia', fontsize=10)
    ax6.set_title('Distribución de Confianza', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🔍 Grid Search - Mejor Modelo Decision Tree', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar en MLflow
    mlflow.log_figure(fig, "grid_search_best_model.png")
    plt.show()
    
    print("✅ Visualizaciones guardadas")
    
    # =====================================
    # 7. REPORTE DE CLASIFICACIÓN
    # =====================================
    print("\n" + "=" * 70)
    print("📋 REPORTE DE CLASIFICACIÓN DETALLADO")
    print("=" * 70)
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=dataset.target_names,
        digits=4
    ))
    
    # =====================================
    # 8. INFORMACIÓN FINAL
    # =====================================
    run_id = run.info.run_id
    
    print("=" * 70)
    print("🎉 GRID SEARCH COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name if email else 'No configurado'}")
    print(f"🎯 Accuracy (Test): {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"🌟 F1-Score (Test): {f1_test:.4f}")
    print(f"📊 Cross-Val Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"🏆 Mejores params: {grid_search.best_params_}")
    print("=" * 70)

mlflow.end_run()
print("\n✅ Experimento finalizado correctamente")

# COMMAND ----------

# 🎨 CÓDIGO DE EJEMPLO PARA LOS DESAFÍOS
# Descomenta y experimenta con diferentes configuraciones

# ========================================
# TODO: DESAFÍO 1 - Optimización manual
# ========================================
configuraciones = [
     {'max_depth': 3, 'max_features': 2},
    {'max_depth': 5, 'max_features': 3},
    {'max_depth': 10, 'max_features': 4},
   {'max_depth': None, 'max_features': 'sqrt'},
 ]
# 
mlflow.sklearn.autolog()
for i, config in enumerate(configuraciones):
     with mlflow.start_run(run_name=f"DT Config {i+1}"):
        dt = DecisionTreeClassifier(**config, random_state=42)
        dt.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, dt.predict(X_test))
print(f"Config {i+1}: {config} -> Accuracy: {accuracy:.4f}")

# ========================================
# TODO: DESAFÍO 2 - Grid Search
# ========================================
from sklearn.model_selection import GridSearchCV
# 
param_grid = {
     'max_depth': [3, 5, 10, None],
     'max_features': [2, 3, 4],
     'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
 }
# 
mlflow.sklearn.autolog()
with mlflow.start_run(run_name="Grid Search - Decision Tree"):
     grid_search = GridSearchCV(
         DecisionTreeClassifier(random_state=42),
         param_grid,
         cv=5,
         scoring='accuracy',
         verbose=1,
         n_jobs=-1
     )
     grid_search.fit(X_train, y_train)
     
     print(f"Mejores parámetros: {grid_search.best_params_}")
     print(f"Mejor accuracy (CV): {grid_search.best_score_:.4f}")
     print(f"Accuracy en test: {accuracy_score(y_test, grid_search.predict(X_test)):.4f}")

# ========================================
# TODO: DESAFÍO 3 - Comparación de modelos
# ========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# 
modelos = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
 }
# 
resultados = []
mlflow.sklearn.autolog()
# 
for nombre, modelo in modelos.items():
   with mlflow.start_run(run_name=f"{nombre} - Iris"):
      modelo.fit(X_train, y_train)
      y_pred = modelo.predict(X_test)
#         
      acc = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred, average='weighted')
#         
      resultados.append({
         'Modelo': nombre,
          'Accuracy': acc,
         'F1-Score': f1
       })
print(f"{nombre}: Accuracy={acc:.4f}, F1={f1:.4f}")
# 
# # Visualizar comparación
df_resultados = pd.DataFrame(resultados).sort_values('Accuracy', ascending=False)
print("\n📊 Ranking de Modelos:")
print(df_resultados.to_string(index=False))

# ========================================
# TODO: DESAFÍO 4 - Análisis de errores
# ========================================
# incorrect_indices = ...
# TODO: inspecciona las muestras mal clasificadas.

# ========================================
# TODO: DESAFÍO 5 - PCA y visualización 2D
# ========================================
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
#     X_pca, y, test_size=0.25, random_state=42, stratify=y
# )
# 
# # Entrenar modelo con datos reducidos
# dt_pca = DecisionTreeClassifier(max_depth=5, random_state=42)
# dt_pca.fit(X_train_pca, y_train_pca)
# 
# # Visualizar límites de decisión
# from matplotlib.colors import ListedColormap
# 
# h = 0.02  # step size
# x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
# y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# 
# Z = dt_pca.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# 
# plt.figure(figsize=(10, 8))
# plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
#                       edgecolors='black', s=80, cmap=plt.cm.RdYlBu)
# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
# plt.title('Límites de Decisión con PCA (2D)')
# plt.colorbar(scatter)
# plt.show()
# 
# print(f"Accuracy con PCA: {accuracy_score(y_test_pca, dt_pca.predict(X_test_pca)):.4f}")

print("💡 Descomenta y ejecuta el código del desafío que quieras explorar")

# COMMAND ----------

# DBTITLE 1,Desafío 5 - PCA y Visualización de Límites de Decisión 2D
# ========================================
# DESAFÍO 5: PCA Y VISUALIZACIÓN 2D
# ========================================
# Librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# MLflow
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Configurar email y experimento
email = 'rafaelsanchezupo@gmail.com'
experiment_name = f"/Users/{email}/4-ejercicio-the-irish"

# Cargar dataset
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

print("=" * 70)
print("🎨 DESAFÍO 5: REDUCCIÓN DE DIMENSIONALIDAD CON PCA")
print("=" * 70)

# =====================================
# 1. APLICAR PCA
# =====================================
print("\n📐 Reduciendo dimensionalidad de 4D a 2D con PCA...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"✅ Reducción completada")
print(f"\n📊 Varianza explicada:")
print(f"   - PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"   - PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"   - Total: {pca.explained_variance_ratio_.sum():.2%}")

# Dividir en train/test
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.25, random_state=42, stratify=y
)

# =====================================
# 2. ENTRENAR MODELO CON DATOS REDUCIDOS
# =====================================
print("\n🌳 Entrenando Decision Tree con 2 componentes principales...")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="PCA 2D - Decision Tree") as run:
    
    # Entrenar modelo
    dt_pca = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_pca.fit(X_train_pca, y_train_pca)
    
    # Predicciones
    y_pred_pca = dt_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
    
    print(f"✅ Modelo entrenado")
    print(f"🎯 Accuracy con PCA (2D): {accuracy_pca:.4f} ({accuracy_pca*100:.2f}%)")
    
    # Registrar información adicional
    mlflow.log_param("n_components", 2)
    mlflow.log_param("pca_variance_ratio_pc1", pca.explained_variance_ratio_[0])
    mlflow.log_param("pca_variance_ratio_pc2", pca.explained_variance_ratio_[1])
    mlflow.log_param("total_variance_explained", pca.explained_variance_ratio_.sum())
    mlflow.log_metric("accuracy_pca_2d", accuracy_pca)
    
    # =====================================
    # 3. VISUALIZAR LÍMITES DE DECISIÓN
    # =====================================
    print("\n📊 Generando visualizaciones...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Subplot 1: Límites de decisión
    ax1 = plt.subplot(2, 3, 1)
    
    # Crear malla para límites de decisión
    h = 0.02  # tamaño del paso
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predicciones en la malla
    Z = dt_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plotear límites de decisión
    ax1.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                          edgecolors='black', s=80, cmap=plt.cm.RdYlBu)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=11)
    ax1.set_title('Límites de Decisión en 2D', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Distribución por clase
    ax2 = plt.subplot(2, 3, 2)
    colors = ['red', 'green', 'blue']
    for i, (color, target_name) in enumerate(zip(colors, dataset.target_names)):
        mask = y == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    c=color, label=target_name, 
                    edgecolors='black', s=80, alpha=0.7)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=11)
    ax2.set_title('Distribución por Especie', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Varianza explicada
    ax3 = plt.subplot(2, 3, 3)
    pc_labels = ['PC1', 'PC2']
    variance_values = pca.explained_variance_ratio_
    bars = ax3.bar(pc_labels, variance_values, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, variance_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11)
    ax3.set_ylabel('Varianza Explicada', fontsize=11)
    ax3.set_title('Varianza por Componente', fontweight='bold', fontsize=12)
    ax3.set_ylim([0, max(variance_values) * 1.2])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Matriz de confusión
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_test_pca, y_pred_pca)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    ax4.set_title('Matriz de Confusión (PCA 2D)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Verdadero', fontsize=10)
    ax4.set_xlabel('Predicción', fontsize=10)
    
    # Subplot 5: Train vs Test split visualization
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                c=y_train_pca, cmap=plt.cm.RdYlBu,
                marker='o', s=80, alpha=0.6, 
                edgecolors='black', label='Train')
    ax5.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                c=y_test_pca, cmap=plt.cm.RdYlBu,
                marker='s', s=120, alpha=0.9, 
                edgecolors='black', linewidth=2, label='Test')
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=11)
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=11)
    ax5.set_title('Train (círculos) vs Test (cuadrados)', fontweight='bold', fontsize=12)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Comparación 4D vs 2D
    ax6 = plt.subplot(2, 3, 6)
    
    # Entrenar modelo con datos originales para comparar
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    dt_full = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_full.fit(X_train_full, y_train_full)
    accuracy_full = accuracy_score(y_test_full, dt_full.predict(X_test_full))
    
    comparison = pd.DataFrame({
        'Configuración': ['4D Original', '2D PCA'],
        'Accuracy': [accuracy_full, accuracy_pca],
        'Dimensiones': [4, 2]
    })
    
    bars = ax6.bar(comparison['Configuración'], comparison['Accuracy'], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, comparison['Accuracy']):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11)
    
    ax6.set_ylabel('Accuracy', fontsize=11)
    ax6.set_title('Comparación: 4D vs 2D', fontweight='bold', fontsize=12)
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Agregar texto con información
    info_text = f"Pérdida de accuracy: {(accuracy_full - accuracy_pca)*100:.1f}%\nGanancia: Visualización 2D"
    ax6.text(0.5, 0.05, info_text, transform=ax6.transAxes,
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('🎨 Análisis PCA: Reducción de Dimensionalidad 4D → 2D', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar en MLflow
    mlflow.log_figure(fig, "pca_2d_analysis.png")
    plt.show()
    
    print("✅ Visualizaciones guardadas en MLflow")
    
    # =====================================
    # 4. REPORTE DETALLADO
    # =====================================
    print("\n" + "=" * 70)
    print("📋 REPORTE DE CLASIFICACIÓN CON PCA 2D")
    print("=" * 70)
    print(classification_report(
        y_test_pca, 
        y_pred_pca, 
        target_names=dataset.target_names,
        digits=4
    ))
    
    # =====================================
    # 5. ANÁLISIS COMPARATIVO
    # =====================================
    print("\n" + "=" * 70)
    print("🔍 ANÁLISIS COMPARATIVO")
    print("=" * 70)
    print(f"\n📊 Dataset Original (4D):")
    print(f"   - Dimensiones: {X.shape[1]} características")
    print(f"   - Accuracy: {accuracy_full:.4f} ({accuracy_full*100:.2f}%)")
    
    print(f"\n🎨 Dataset con PCA (2D):")
    print(f"   - Dimensiones: 2 componentes principales")
    print(f"   - Varianza retenida: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"   - Accuracy: {accuracy_pca:.4f} ({accuracy_pca*100:.2f}%)")
    print(f"   - Pérdida de accuracy: {(accuracy_full - accuracy_pca)*100:.2f}%")
    
    print(f"\n💡 Conclusiones:")
    diff = accuracy_full - accuracy_pca
    if diff < 0.05:
        print(f"   ✅ Excelente: Mínima pérdida de accuracy ({diff*100:.1f}%)")
        print(f"   ✅ PCA es viable para reducir dimensionalidad")
        print(f"   ✅ Ganamos visualización interpretable en 2D")
    elif diff < 0.10:
        print(f"   ⚡ Bueno: Pérdida moderada de accuracy ({diff*100:.1f}%)")
        print(f"   ⚡ Trade-off aceptable para visualización")
    else:
        print(f"   ⚠️  Pérdida significativa de accuracy ({diff*100:.1f}%)")
        print(f"   ⚠️  Considerar más componentes o features originales")
    
    # Información final
    run_id = run.info.run_id
    
    print("\n" + "=" * 70)
    print("🎉 DESAFÍO 5 COMPLETADO")
    print("=" * 70)
    print(f"🆔 Run ID: {run_id}")
    print(f"📊 Experimento: {experiment_name}")
    print(f"🎯 Accuracy con PCA: {accuracy_pca:.4f}")
    print(f"🎨 Varianza retenida: {pca.explained_variance_ratio_.sum():.2%}")
    print("=" * 70)

mlflow.end_run()
print("\n✅ Experimento PCA finalizado correctamente")
print("\n🌟 ¡Has completado con éxito el Desafío 5!")
print("   Ahora puedes visualizar e interpretar el modelo en 2D")