# Databricks notebook source
# MAGIC %md
# MAGIC # Semana 01 · Ciclo de vida completo de un modelo con MLflow
# MAGIC
# MAGIC En esta práctica llevarás un caso de clasificación desde los datos hasta una API local:
# MAGIC
# MAGIC `datos → experimentos → comparación → gate → modelo ganador → Registry → API → observabilidad`
# MAGIC
# MAGIC El objetivo no es conseguir el número más alto, sino construir una decisión **reproducible,
# MAGIC auditable y desplegable**. El dataset cardiovascular es únicamente didáctico: el resultado
# MAGIC no puede usarse para diagnóstico, tratamiento ni decisiones sobre pacientes.
# MAGIC
# MAGIC **Entorno:** Databricks Free Edition, compute serverless y experimento asociado a este notebook.
# MAGIC No necesitas tarjeta de crédito, cluster propio, Docker, `uv`, `pyproject.toml` ni tokens.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mapa y criterios de terminación
# MAGIC
# MAGIC Al acabar debes poder enseñar en MLflow:
# MAGIC
# MAGIC 1. un lote de al menos seis runs comparables, con datos, parámetros, métricas y artefactos;
# MAGIC 2. una regla de selección escrita **antes** de mirar el test;
# MAGIC 3. un único candidato evaluado sobre test y marcado como ganador;
# MAGIC 4. versiones en Unity Catalog y un ciclo `Challenger → Champion → rollback`;
# MAGIC 5. una API HTTP local que carga el `model.pkl` del run ganador y supera pruebas válidas e inválidas;
# MAGIC 6. un run de despliegue con latencia, volumen de peticiones, errores y evidencia de la respuesta.
# MAGIC
# MAGIC Tiempo recomendado: 45 min de construcción guiada + 120–180 min de práctica autónoma y debrief.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencias
# MAGIC
# MAGIC Ejecuta `%pip` al principio. Si Databricks reinicia Python, continúa desde la celda de imports.
# MAGIC Las dependencias se instalan en la sesión del notebook; no se crea ningún entorno o fichero de
# MAGIC proyecto en el repositorio.

# COMMAND ----------

# MAGIC %pip install -q --upgrade "scikit-learn==1.9.0" "matplotlib==3.11.1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cómo trabajar con la versión sin resolver
# MAGIC
# MAGIC Ejecuta las celdas en orden y completa cada `TODO`. Los bloques se detienen con
# MAGIC `NotImplementedError` para que un fallo no genere evidencia engañosa. Consulta la guía del
# MAGIC alumno antes de mirar la solución. Mantén el test cerrado hasta haber elegido el ganador.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración e identidad del lote

# COMMAND ----------

import json
import pickle
import re
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

STUDENT_ALIAS = "TODO-alias-anonimo"
BATCH_ID = uuid.uuid4().hex[:8]
RANDOM_STATE = 42
MIN_VALIDATION_RECALL = 0.72
DATASET_PATH = "../../../data/raw/heart.csv"

# TODO 1: valida el alias con una expresión regular y configura el tracking de Databricks.
raise NotImplementedError("TODO 1 · configuración")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Contrato y controles de datos
# MAGIC
# MAGIC Permite nulos porque el pipeline los imputará, pero detente si faltan columnas, el dataset está
# MAGIC vacío o el target no es binario. Construye un `quality_report` serializable como JSON.

# COMMAND ----------

data = pd.read_csv(DATASET_PATH)
TARGET = "target"
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# TODO 2: valida el contrato y crea X, y y quality_report.
# Debe incluir filas, duplicados, distribución del target, nulos y checks ejecutados.
raise NotImplementedError("TODO 2 · calidad de datos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Divide 60 % train, 20 % validación y 20 % test, siempre estratificado

# COMMAND ----------

# TODO 3: crea X_train, X_valid, X_test, y_train, y_valid e y_test.
# No uses X_test ni y_test hasta la sección 9.
raise NotImplementedError("TODO 3 · split")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pipeline y funciones de evaluación

# COMMAND ----------

categorical_columns = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_columns = [c for c in EXPECTED_FEATURES if c not in categorical_columns]


def build_pipeline(config: dict) -> Pipeline:
    # TODO 4a: ColumnTransformer con imputación y OneHotEncoder, seguido del RandomForest.
    raise NotImplementedError("TODO 4a · pipeline")


def evaluate_classifier(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict:
    # TODO 4b: accuracy, precision, recall, F1, ROC AUC y latencia por fila.
    raise NotImplementedError("TODO 4b · métricas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Diseña al menos seis candidatos comparables

# COMMAND ----------

# TODO 5: cambia de forma controlada n_estimators, max_depth, min_samples_leaf
# y/o class_weight. Todos deben incluir random_state.
CANDIDATES = []
assert len(CANDIDATES) >= 6, "Necesitas al menos seis candidatos."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tracking profundo: un run por candidato
# MAGIC
# MAGIC Por cada candidato registra tags, parámetros, versiones, inputs de train/validación, tarjeta de
# MAGIC datos, calidad, riesgos, métricas, reporte por clase, matriz de confusión, modelo MLflow con
# MAGIC firma/input example, `deployment/model.pkl` y contrato de inferencia.

# COMMAND ----------

dataset_card = {
    "source": "TODO",
    "purpose": "TODO",
    "features": EXPECTED_FEATURES,
    "target": TARGET,
    "split": {"train": 0.60, "validation": 0.20, "test": 0.20, "random_state": RANDOM_STATE},
    "known_limitations": ["TODO", "TODO", "TODO"],
}
risk_register = {
    "version": "s01-lifecycle",
    "risks": [
        {"id": "R1", "risk": "TODO", "impact": "high", "owner": "TODO", "mitigation": "TODO"},
        {"id": "R2", "risk": "TODO", "impact": "high", "owner": "TODO", "mitigation": "TODO"},
        {"id": "R3", "risk": "TODO", "impact": "high", "owner": "TODO", "mitigation": "TODO"},
        {"id": "R4", "risk": "TODO", "impact": "high", "owner": "TODO", "mitigation": "TODO"},
    ],
}
assert len(dataset_card["known_limitations"]) >= 3
assert len(risk_register["risks"]) >= 4

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Inputs de datos
# MAGIC
# MAGIC Crea dos objetos `mlflow.data` con nombres distintos. Comprueba sus nombres antes de lanzar runs.

# COMMAND ----------

# TODO 6a: crea train_dataset y validation_dataset con mlflow.data.from_pandas.
# Incluye target alineado con cada split y usa DATASET_PATH como source.
raise NotImplementedError("TODO 6a · datasets de MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Artefactos del candidato
# MAGIC
# MAGIC Implementa una función pequeña que registre firma/input example, `model_info.model_uri`, pickle,
# MAGIC tamaño, matriz de confusión y contrato. Podrás probarla dentro de un único run antes del bucle.

# COMMAND ----------

def log_candidate_artifacts(
    model: Pipeline, predictions: np.ndarray, candidate_index: int
) -> str:
    # TODO 6b: log_model(name='model'), tag candidate.model_uri, model.pkl,
    # pickle_size_bytes, confusion matrix e inference_contract.json. Devuelve model_uri.
    raise NotImplementedError("TODO 6b · artefactos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c. Runs comparables
# MAGIC
# MAGIC Ahora el bucle se ocupa sólo de identidad, parámetros, inputs, gobierno, entrenamiento, métricas
# MAGIC y de llamar al helper de artefactos. Conserva `experiment_id` y cada `run_id`.

# COMMAND ----------

experiment_id = None
candidate_run_ids = []

# TODO 6c: implementa un start_run por configuración, registra tags/params/inputs/JSON,
# entrena, calcula validation.*, classification_report y llama log_candidate_artifacts.
raise NotImplementedError("TODO 6c · runs de candidatos")
assert len(candidate_run_ids) == len(CANDIDATES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Busca y compara exclusivamente los runs de tu alias y `BATCH_ID`

# COMMAND ----------

# TODO 7: usa mlflow.search_runs con tres tags en filter_string y muestra parámetros,
# métricas de validación, latencia y run_id en un DataFrame `comparison`.
raise NotImplementedError("TODO 7 · comparación")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Aplica el gate sin mirar test
# MAGIC
# MAGIC Filtra por recall mínimo; ordena por F1 DESC, ROC AUC DESC y latencia ASC. Conserva
# MAGIC `BEST_RUN_ID`, `BEST_MODEL_URI` y un `selection_report` que diga explícitamente que test no
# MAGIC participó en la selección.

# COMMAND ----------

# TODO 8: aplica el gate. Si no hay elegibles, falla sin rebajar el umbral a posteriori.
raise NotImplementedError("TODO 8 · gate y ganador")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Carga el modelo desde MLflow, evalúa test una sola vez y completa el run ganador

# COMMAND ----------

# TODO 9: load_model(BEST_MODEL_URI), calcula test.*, reabre BEST_RUN_ID,
# registra el input de testing, métricas, tags de selección y selection/decision.json.
raise NotImplementedError("TODO 9 · evaluación final")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Registra el modelo ganador en Unity Catalog
# MAGIC
# MAGIC Usa el catálogo y esquema activos, un nombre de tres niveles y tu alias saneado. Añade
# MAGIC descripción y tags. Registra referencia+ganador, asigna `Challenger`, haz smoke test,
# MAGIC promociona, comprueba rollback y deja el ganador como `Champion`. Si recibes `PERMISSION_DENIED`, consulta la
# MAGIC guía: no pegues credenciales ni cambies a recursos de pago.

# COMMAND ----------

# TODO 10: set_registry_uri('databricks-uc'), register_model, descripciones,
# tags, Challenger, promoción, rollback, re-promoción y variable champion.
raise NotImplementedError("TODO 10 · Model Registry")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Carga `models:/<nombre>@Champion` y realiza un smoke test con tres filas

# COMMAND ----------

# TODO 11: prueba que el consumidor puede cargar por alias y predecir tres filas.
raise NotImplementedError("TODO 11 · smoke test del Registry")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. API local con el pickle ganador
# MAGIC
# MAGIC Resuelve el run de `Champion`, descarga `deployment/model.pkl` y levanta un
# MAGIC `ThreadingHTTPServer` en `127.0.0.1` con puerto automático. Implementa `GET /health` y
# MAGIC `POST /predict`; valida `instances`, columnas ausentes y columnas extra.

# COMMAND ----------

# TODO 12a: resuelve champion.run_id, descarga runs:/.../deployment/model.pkl,
# deserializa api_model y comprueba que predice una fila.
raise NotImplementedError("TODO 12a · cargar pickle")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 12b. Handler y contrato
# MAGIC
# MAGIC Define `PredictionHandler` con `_send_json`, `GET /health` y `POST /predict`. Valida objeto raíz,
# MAGIC lista no vacía de objetos, columnas exactas y errores 400. Instrumenta contadores y latencias.

# COMMAND ----------

API_STATS = {"requests_total": 0, "requests_success": 0, "requests_rejected": 0, "requests_error": 0, "latencies_ms": []}
API_STATS_LOCK = threading.Lock()

# TODO 12b: define PredictionHandler y actualiza API_STATS en cada respuesta.
raise NotImplementedError("TODO 12b · handler")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 12c. Arranque y limpieza idempotente
# MAGIC
# MAGIC Arranca en `127.0.0.1` y puerto 0. Define `stop_api()` antes de probar para que un `finally` pueda
# MAGIC cerrar servidor, socket e hilo. No uses `0.0.0.0`.

# COMMAND ----------

# TODO 12c: crea api_server, API_PORT, api_thread, API_STOPPED y stop_api().
raise NotImplementedError("TODO 12c · ciclo de servidor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Pruebas de contrato y de error

# COMMAND ----------

# TODO 13: crea call_json con try/finally para apagar siempre la API y verifica:
# GET /health -> 200; POST válido -> 200; sólo age -> 400; JSON no-object -> 400.
raise NotImplementedError("TODO 13 · pruebas de API")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Registra un run separado para la evidencia del despliegue

# COMMAND ----------

# TODO 14: registra versión/alias/run origen, requests totales, éxitos, rechazadas,
# latencias y deployment/evidence.json. Conserva DEPLOYMENT_RUN_ID.
# No registres el payload de entrada: podría contener información sensible.
raise NotImplementedError("TODO 14 · observabilidad de deployment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Apaga la API y entrega el resumen del ciclo

# COMMAND ----------

# TODO 15: shutdown, server_close, join y assert de que el hilo terminó.
# Imprime experiment_id, batch, ganador, modelo/versión/alias y deployment run.
raise NotImplementedError("TODO 15 · cierre")