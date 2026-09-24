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

# MAGIC %pip install -q --upgrade "mlflow[databricks]==3.14.0" "scikit-learn==1.9.0" "pandas==3.0.3" "matplotlib==3.11.1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración e identidad del lote
# MAGIC
# MAGIC MLflow crea/reutiliza el experimento de este notebook. `BATCH_ID` separa esta ejecución de
# MAGIC intentos anteriores y hace que la selección compare sólo candidatos entrenados con el mismo
# MAGIC split. Usa un alias anónimo: acabará en tags y en el nombre del modelo registrado.

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
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

STUDENT_ALIAS = "pareja-07"
assert re.fullmatch(r"[a-z0-9_-]{3,24}", STUDENT_ALIAS), (
    "Usa 3–24 caracteres: minúsculas, números, _ o -, sin datos personales."
)

BATCH_ID = uuid.uuid4().hex[:8]
RANDOM_STATE = 42
MIN_VALIDATION_RECALL = 0.72
DATASET_PATH = "../../../data/raw/heart.csv"

mlflow.set_tracking_uri("databricks")
print(
    f"MLflow={mlflow.__version__} · sklearn={sklearn.__version__} · "
    f"alias={STUDENT_ALIAS} · batch={BATCH_ID}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Contrato y controles de datos
# MAGIC
# MAGIC Antes de entrenar, comprueba el contrato mínimo. Los valores ausentes están permitidos porque
# MAGIC la tubería los imputa; una columna ausente, un target no binario o un dataset vacío no lo están.
# MAGIC Registraremos el resumen y el esquema para poder explicar con qué datos se produjo cada run.

# COMMAND ----------

data = pd.read_csv(DATASET_PATH)
TARGET = "target"
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

missing_columns = sorted(set(EXPECTED_FEATURES + [TARGET]) - set(data.columns))
assert not missing_columns, f"Faltan columnas obligatorias: {missing_columns}"
assert len(data) > 0, "El dataset está vacío."
assert data[TARGET].notna().all(), "El target no puede contener valores nulos."
assert set(data[TARGET].unique()) <= {0, 1}, "El target debe ser binario."

X = data[EXPECTED_FEATURES].copy()
y = data[TARGET].astype(int)
quality_report = {
    "rows": int(len(data)),
    "columns": int(data.shape[1]),
    "duplicate_rows": int(data.duplicated().sum()),
    "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
    "missing_by_column": {k: int(v) for k, v in X.isna().sum().items()},
    "checks": {
        "required_columns": "passed",
        "non_empty": "passed",
        "binary_target": "passed",
    },
}
display(pd.DataFrame({"dtype": X.dtypes.astype(str), "missing": X.isna().sum()}))
print(json.dumps(quality_report, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train, validación y test: tres funciones distintas
# MAGIC
# MAGIC - **Train** ajusta los parámetros aprendidos.
# MAGIC - **Validación** compara configuraciones y aplica el gate.
# MAGIC - **Test** se abre una sola vez, después de congelar el ganador.
# MAGIC
# MAGIC Usar test para elegir produciría una estimación optimista. El split estratificado y sus
# MAGIC semillas quedan registrados como parámetros del run.

# COMMAND ----------

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

split_report = pd.DataFrame(
    {
        "rows": [len(X_train), len(X_valid), len(X_test)],
        "positive_rate": [y_train.mean(), y_valid.mean(), y_test.mean()],
    },
    index=["train", "validation", "test"],
)
display(split_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pipeline y funciones de evaluación
# MAGIC
# MAGIC La imputación, codificación y clasificación viajan juntas dentro del `Pipeline`. Esto evita
# MAGIC que entrenamiento y serving apliquen transformaciones distintas. La latencia se mide como
# MAGIC señal operativa, aunque en un notebook compartido no es un benchmark de producción.

# COMMAND ----------

categorical_columns = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_columns = [c for c in EXPECTED_FEATURES if c not in categorical_columns]


def build_pipeline(config: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(**config)),
        ]
    )


def evaluate_classifier(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict:
    started = time.perf_counter()
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    latency_ms_per_row = 1000 * (time.perf_counter() - started) / len(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "latency_ms_per_row": float(latency_ms_per_row),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Plan del experimento
# MAGIC
# MAGIC Variamos varios hiperparámetros de forma controlada. Todos los candidatos usan exactamente el
# MAGIC mismo dataset, split, pipeline y métricas. `random_state` forma parte de la configuración: una
# MAGIC ejecución reproducible no depende de recordar una semilla fuera de MLflow.

# COMMAND ----------

CANDIDATES = [
    {"n_estimators": 40, "max_depth": 3, "min_samples_leaf": 4, "class_weight": None, "random_state": RANDOM_STATE},
    {"n_estimators": 80, "max_depth": 4, "min_samples_leaf": 3, "class_weight": None, "random_state": RANDOM_STATE},
    {"n_estimators": 120, "max_depth": 5, "min_samples_leaf": 2, "class_weight": None, "random_state": RANDOM_STATE},
    {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": RANDOM_STATE},
    {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 1, "class_weight": "balanced", "random_state": RANDOM_STATE},
    {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": RANDOM_STATE},
]
assert len(CANDIDATES) >= 6
display(pd.DataFrame(CANDIDATES))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tracking profundo: un run por candidato
# MAGIC
# MAGIC Cada run registra cinco capas de evidencia:
# MAGIC
# MAGIC - **identidad:** nombre, alias, lote, fase y propósito;
# MAGIC - **reproducibilidad:** hiperparámetros, semillas y versiones;
# MAGIC - **datos:** digest y contexto de train/validación con `log_input`;
# MAGIC - **resultado:** métricas, matriz de confusión y reporte por clase;
# MAGIC - **despliegue:** modelo MLflow con firma/input example y un `model.pkl` explícito.

# COMMAND ----------

dataset_card = {
    "source": "data/raw/heart.csv (caso histórico del repositorio)",
    "purpose": "Aprendizaje de MLOps; no es un sistema clínico.",
    "features": EXPECTED_FEATURES,
    "target": TARGET,
    "split": {"train": 0.60, "validation": 0.20, "test": 0.20, "random_state": RANDOM_STATE},
    "known_limitations": [
        "No representa una población clínica real ni actual.",
        "No se ha auditado equidad, calibración, deriva ni validez externa.",
        "No puede usarse para diagnóstico, tratamiento o decisiones sobre pacientes.",
    ],
}
risk_register = {
    "version": "s01-lifecycle",
    "risks": [
        {"id": "R1", "risk": "Uso clínico indebido", "impact": "high", "owner": "product_owner", "mitigation": "Restringir el caso a docencia y exigir revisión humana."},
        {"id": "R2", "risk": "Sesgo o falta de representatividad", "impact": "high", "owner": "data_owner", "mitigation": "Auditar procedencia, subgrupos y validez externa antes de cualquier piloto."},
        {"id": "R3", "risk": "Diferencia entre preprocessing de training y serving", "impact": "high", "owner": "ml_engineer", "mitigation": "Empaquetar preprocessing y modelo en un único Pipeline y probar el contrato."},
        {"id": "R4", "risk": "Secretos o datos sensibles en MLflow", "impact": "high", "owner": "team", "mitigation": "No registrar tokens, PII, correos o prompts sensibles."},
    ],
}

train_dataset = mlflow.data.from_pandas(
    X_train.assign(target=y_train.to_numpy()), source=DATASET_PATH, name="heart_train"
)
validation_dataset = mlflow.data.from_pandas(
    X_valid.assign(target=y_valid.to_numpy()), source=DATASET_PATH, name="heart_validation"
)

experiment_id = None
candidate_run_ids = []

for index, config in enumerate(CANDIDATES, start=1):
    run_name = f"{STUDENT_ALIAS}-{BATCH_ID}-candidate-{index:02d}"
    with mlflow.start_run(run_name=run_name) as run:
        experiment_id = run.info.experiment_id
        candidate_run_ids.append(run.info.run_id)
        mlflow.set_tags(
            {
                "course": "MUIAAP-operacion-modelos",
                "course.week": "01",
                "student.alias": STUDENT_ALIAS,
                "batch.id": BATCH_ID,
                "lifecycle.phase": "candidate",
                "use_case": "didactic-heart-classification",
                "risk.tier": "not-for-clinical-use",
            }
        )
        mlflow.log_params(config)
        mlflow.log_params(
            {
                "split.random_state": RANDOM_STATE,
                "split.train_rows": len(X_train),
                "split.validation_rows": len(X_valid),
                "split.test_rows_reserved": len(X_test),
                "selection.min_validation_recall": MIN_VALIDATION_RECALL,
                "library.sklearn": sklearn.__version__,
            }
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(validation_dataset, context="validation")
        mlflow.log_dict(dataset_card, "governance/dataset_card.json")
        mlflow.log_dict(quality_report, "governance/data_quality.json")
        mlflow.log_dict(risk_register, "governance/risk_register.json")

        model = build_pipeline(config)
        fit_started = time.perf_counter()
        model.fit(X_train, y_train)
        mlflow.log_metric("train.fit_seconds", time.perf_counter() - fit_started)

        validation_metrics = evaluate_classifier(model, X_valid, y_valid)
        mlflow.log_metrics({f"validation.{k}": v for k, v in validation_metrics.items()})

        predictions = model.predict(X_valid)
        report = classification_report(y_valid, predictions, output_dict=True, zero_division=0)
        mlflow.log_dict(report, "evaluation/validation_classification_report.json")

        signature = infer_signature(X_valid, model.predict(X_valid))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_valid.head(3),
        )
        mlflow.set_tag("candidate.model_uri", model_info.model_uri)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pickle_path = temp_path / "model.pkl"
            with pickle_path.open("wb") as file:
                pickle.dump(model, file)
            mlflow.log_artifact(str(pickle_path), artifact_path="deployment")
            mlflow.log_metric("model.pickle_size_bytes", pickle_path.stat().st_size)

            figure, axis = plt.subplots(figsize=(4.5, 4.5))
            ConfusionMatrixDisplay.from_predictions(y_valid, predictions, ax=axis, colorbar=False)
            axis.set_title(f"Validation · candidate {index:02d}")
            figure.tight_layout()
            figure_path = temp_path / "validation_confusion_matrix.png"
            figure.savefig(figure_path, dpi=140)
            plt.close(figure)
            mlflow.log_artifact(str(figure_path), artifact_path="evaluation")

        mlflow.log_dict(
            {"ordered_features": EXPECTED_FEATURES, "request_key": "instances"},
            "deployment/inference_contract.json",
        )

print(f"Registrados {len(candidate_run_ids)} candidatos en el experimento {experiment_id}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparación por API y por interfaz
# MAGIC
# MAGIC Abre **Experiments** y usa la comparación visual. Después repite la búsqueda por código: esa
# MAGIC consulta puede automatizarse y deja explícito qué lote y qué columnas sostienen la decisión.

# COMMAND ----------

filter_string = (
    f"tags.student.alias = '{STUDENT_ALIAS}' AND "
    f"tags.batch.id = '{BATCH_ID}' AND "
    "tags.lifecycle.phase = 'candidate'"
)
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=filter_string,
    max_results=100,
)

metric_columns = [
    "metrics.validation.f1",
    "metrics.validation.recall",
    "metrics.validation.precision",
    "metrics.validation.roc_auc",
    "metrics.validation.latency_ms_per_row",
    "metrics.model.pickle_size_bytes",
]
columns_to_show = [
    "run_id",
    "tags.mlflow.runName",
    "tags.candidate.model_uri",
    "params.n_estimators",
    "params.max_depth",
    "params.min_samples_leaf",
    "params.class_weight",
    *metric_columns,
]
comparison = runs.reindex(columns=columns_to_show).copy()
for column in metric_columns:
    comparison[column] = pd.to_numeric(comparison[column], errors="coerce")
display(comparison.sort_values("metrics.validation.f1", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Gate y elección reproducible del ganador
# MAGIC
# MAGIC Regla predefinida:
# MAGIC
# MAGIC 1. descartar candidatos con `validation.recall < MIN_VALIDATION_RECALL`;
# MAGIC 2. entre los elegibles, maximizar F1;
# MAGIC 3. desempatar por ROC AUC;
# MAGIC 4. usar menor latencia sólo como tercer desempate.
# MAGIC
# MAGIC Un gate no demuestra que el modelo sea seguro: convierte una preferencia en una condición
# MAGIC visible. Aquí priorizamos recall para discutir el coste de falsos negativos.

# COMMAND ----------

eligible = comparison[
    comparison["metrics.validation.recall"] >= MIN_VALIDATION_RECALL
].copy()
assert not eligible.empty, (
    "Ningún candidato supera el gate. No rebajes el umbral después de ver los resultados: "
    "revisa el plan experimental y genera nueva evidencia."
)

ranked = eligible.sort_values(
    by=[
        "metrics.validation.f1",
        "metrics.validation.roc_auc",
        "metrics.validation.latency_ms_per_row",
    ],
    ascending=[False, False, True],
)
winner = ranked.iloc[0]
BEST_RUN_ID = str(winner["run_id"])
BEST_MODEL_URI = str(winner["tags.candidate.model_uri"])

selection_report = {
    "batch_id": BATCH_ID,
    "candidate_count": int(len(comparison)),
    "eligible_count": int(len(eligible)),
    "gate": {"metric": "validation.recall", "operator": ">=", "value": MIN_VALIDATION_RECALL},
    "ranking": ["validation.f1 DESC", "validation.roc_auc DESC", "validation.latency_ms_per_row ASC"],
    "winner_run_id": BEST_RUN_ID,
    "test_was_used_for_selection": False,
}
print(json.dumps(selection_report, indent=2))
display(ranked.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Abrir test una sola vez y cerrar la decisión
# MAGIC
# MAGIC Cargamos el artefacto desde MLflow, no la variable que quedó en memoria. Así comprobamos que el
# MAGIC modelo persistido es utilizable. Las métricas de test se añaden exclusivamente al run ganador.

# COMMAND ----------

winner_model = mlflow.sklearn.load_model(BEST_MODEL_URI)
test_metrics = evaluate_classifier(winner_model, X_test, y_test)
test_dataset = mlflow.data.from_pandas(
    X_test.assign(target=y_test.to_numpy()), source=DATASET_PATH, name="heart_test"
)

with mlflow.start_run(run_id=BEST_RUN_ID):
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_metrics({f"test.{k}": v for k, v in test_metrics.items()})
    mlflow.set_tags(
        {
            "selection.status": "winner",
            "selection.rule": "recall_gate_then_f1_roc_auc_latency",
            "selection.test_used_once": "true",
        }
    )
    mlflow.log_dict(selection_report, "selection/decision.json")

print("Ganador:", BEST_RUN_ID)
print(json.dumps({f"test.{k}": v for k, v in test_metrics.items()}, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model Registry en Unity Catalog
# MAGIC
# MAGIC MLflow 3 usa el Registry de Unity Catalog. El nombre tiene tres niveles:
# MAGIC `catalog.schema.model`. Tomamos el catálogo y esquema activos de la sesión y añadimos el alias
# MAGIC del equipo para evitar colisiones. Registraremos una referencia de rollback y el ganador como
# MAGIC `Challenger`; tras un smoke test practicaremos promoción, rollback y re-promoción. Registrar no
# MAGIC equivale a desplegar: crea una versión gobernada.
# MAGIC
# MAGIC Si aparece `PERMISSION_DENIED`, confirma que el notebook está en tu workspace Free Edition y
# MAGIC que el catálogo/esquema activos permiten `CREATE MODEL`; prueba `main.default` desde Catalog
# MAGIC Explorer. No cambies a un servicio de pago ni pegues credenciales.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
catalog_name, schema_name = spark.sql(
    "SELECT current_catalog(), current_schema()"
).first()
safe_alias = re.sub(r"[^a-z0-9_]", "_", STUDENT_ALIAS.lower())
REGISTERED_MODEL_NAME = f"{catalog_name}.{schema_name}.heart_classifier_{safe_alias}"

rollback_row = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
ROLLBACK_MODEL_URI = str(rollback_row["tags.candidate.model_uri"])
rollback_registered = mlflow.register_model(
    model_uri=ROLLBACK_MODEL_URI,
    name=REGISTERED_MODEL_NAME,
)
registered = mlflow.register_model(
    model_uri=BEST_MODEL_URI,
    name=REGISTERED_MODEL_NAME,
)
registry_client = MlflowClient(registry_uri="databricks-uc")
registry_client.update_registered_model(
    name=REGISTERED_MODEL_NAME,
    description=(
        "Clasificador didáctico de la semana 01. Prohibido uso clínico. "
        "Ganador elegido mediante gate de recall sobre validación."
    ),
)
registry_client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=rollback_registered.version,
    description="Referencia para practicar rollback; no autorizada para uso clínico.",
)
registry_client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=registered.version,
    description=f"Run ganador {BEST_RUN_ID}; batch {BATCH_ID}.",
)
registry_client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=rollback_registered.version,
    key="validation_status",
    value="rollback_demo_only",
)
registry_client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=rollback_registered.version,
    key="clinical_use",
    value="forbidden",
)
registry_client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered.version,
    key="validation_status",
    value="passed_didactic_gate",
)
registry_client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered.version,
    key="clinical_use",
    value="forbidden",
)
registry_client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="Challenger",
    version=registered.version,
)

challenger_uri = f"models:/{REGISTERED_MODEL_NAME}@Challenger"
challenger_model = mlflow.sklearn.load_model(challenger_uri)
assert len(challenger_model.predict(X_test.head(3))) == 3

registry_client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Champion", registered.version)
promoted = registry_client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Champion")
assert str(promoted.version) == str(registered.version)
registry_client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Champion", rollback_registered.version)
rolled_back = registry_client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Champion")
assert str(rolled_back.version) == str(rollback_registered.version)
rollback_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@Champion")
assert len(rollback_model.predict(X_test.head(3))) == 3
registry_client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Champion", registered.version)
champion = registry_client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Champion")
assert str(champion.version) == str(registered.version)
print(
    f"Registry verificado: rollback={rollback_registered.version}, "
    f"Champion final={champion.version} (run={champion.run_id})."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Smoke test desde el alias `Champion`
# MAGIC
# MAGIC El consumidor apunta al alias y no a una versión fija. Cambiar `Champion` permite promocionar
# MAGIC otra versión sin modificar el código consumidor. El `input_example` y la firma ayudan a
# MAGIC detectar incompatibilidades de contrato.

# COMMAND ----------

champion_uri = f"models:/{REGISTERED_MODEL_NAME}@Champion"
champion_model = mlflow.sklearn.load_model(champion_uri)
smoke_predictions = champion_model.predict(X_test.head(3))
assert len(smoke_predictions) == 3
print("URI:", champion_uri)
display(pd.DataFrame({"prediction": smoke_predictions}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Despliegue local y gratuito de `model.pkl`
# MAGIC
# MAGIC Free Edition limita recursos y un servidor iniciado en el driver **no es público ni
# MAGIC persistente**. Para aprender el contrato de serving, levantamos una API HTTP sólo en
# MAGIC `127.0.0.1` y la probamos desde el mismo notebook. No abre puertos externos y se apaga al final.
# MAGIC
# MAGIC Endpoints:
# MAGIC
# MAGIC - `GET /health` devuelve estado y versión del modelo;
# MAGIC - `POST /predict` acepta `{"instances": [{...}]}` y devuelve clase y probabilidad.

# COMMAND ----------

source_run_id = champion.run_id or BEST_RUN_ID
pickle_uri = f"runs:/{source_run_id}/deployment/model.pkl"
local_pickle_path = mlflow.artifacts.download_artifacts(artifact_uri=pickle_uri)
with open(local_pickle_path, "rb") as file:
    api_model = pickle.load(file)


API_STATS = {"requests_total": 0, "requests_success": 0, "requests_rejected": 0, "requests_error": 0, "latencies_ms": []}
API_STATS_LOCK = threading.Lock()


class PredictionHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict, started: float) -> None:
        elapsed_ms = 1000 * (time.perf_counter() - started)
        with API_STATS_LOCK:
            API_STATS["requests_total"] += 1
            API_STATS["latencies_ms"].append(elapsed_ms)
            if status < 400:
                API_STATS["requests_success"] += 1
            elif status < 500:
                API_STATS["requests_rejected"] += 1
            else:
                API_STATS["requests_error"] += 1
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        started = time.perf_counter()
        if self.path != "/health":
            self._send_json(404, {"error": "not_found"}, started)
            return
        self._send_json(
            200,
            {
                "status": "ok",
                "model": REGISTERED_MODEL_NAME,
                "version": str(champion.version),
                "alias": "Champion",
            },
            started,
        )

    def do_POST(self) -> None:
        started = time.perf_counter()
        if self.path != "/predict":
            self._send_json(404, {"error": "not_found"}, started)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request_payload = json.loads(self.rfile.read(content_length))
            if not isinstance(request_payload, dict):
                raise ValueError("El cuerpo JSON debe ser un objeto.")
            instances = request_payload.get("instances")
            if not isinstance(instances, list) or not instances:
                raise ValueError("'instances' debe ser una lista no vacía.")
            if not all(isinstance(instance, dict) for instance in instances):
                raise ValueError("Cada elemento de 'instances' debe ser un objeto.")
            frame = pd.DataFrame(instances)
            missing = sorted(set(EXPECTED_FEATURES) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(EXPECTED_FEATURES))
            if missing or extra:
                raise ValueError(f"Contrato inválido; missing={missing}, extra={extra}")
            frame = frame[EXPECTED_FEATURES]
            predictions = api_model.predict(frame)
            probabilities = api_model.predict_proba(frame)[:, 1]
            self._send_json(
                200,
                {
                    "predictions": [int(value) for value in predictions],
                    "probabilities": [float(value) for value in probabilities],
                    "model_version": str(champion.version),
                },
                started,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": "invalid_request", "detail": str(error)}, started)
        except Exception:
            self._send_json(500, {"error": "internal_error"}, started)

    def log_message(self, format: str, *args) -> None:
        return


api_server = ThreadingHTTPServer(("127.0.0.1", 0), PredictionHandler)
API_PORT = api_server.server_address[1]
api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
api_thread.start()
print(f"API local activa en http://127.0.0.1:{API_PORT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Pruebas de contrato: éxito, error y health
# MAGIC
# MAGIC Un despliegue no termina cuando el proceso arranca. Comprobamos el health check, una predicción
# MAGIC válida y el rechazo controlado de columnas incompletas y JSON no-object. Un `finally` apaga el
# MAGIC servidor incluso si falla una aserción. La API no devuelve tracebacks ni datos internos.

# COMMAND ----------

def call_json(method: str, path: str, payload: object | None = None) -> tuple[int, dict, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{API_PORT}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=10) as response:
            status = response.status
            result = json.loads(response.read())
    except HTTPError as error:
        status = error.code
        result = json.loads(error.read())
    elapsed_ms = 1000 * (time.perf_counter() - started)
    return status, result, elapsed_ms


API_STOPPED = False


def stop_api() -> None:
    global API_STOPPED
    if not API_STOPPED:
        api_server.shutdown()
        api_server.server_close()
        api_thread.join(timeout=5)
        API_STOPPED = True


try:
    health_status, health_body, health_ms = call_json("GET", "/health")
    example_instance = json.loads(X_test.head(1).to_json(orient="records"))[0]
    predict_status, predict_body, predict_ms = call_json(
        "POST", "/predict", {"instances": [example_instance]}
    )
    invalid_status, invalid_body, invalid_ms = call_json(
        "POST", "/predict", {"instances": [{"age": 55}]}
    )
    non_object_status, non_object_body, non_object_ms = call_json(
        "POST", "/predict", [example_instance]
    )

    assert health_status == 200 and health_body["status"] == "ok"
    assert predict_status == 200 and len(predict_body["predictions"]) == 1
    assert invalid_status == 400 and invalid_body["error"] == "invalid_request"
    assert non_object_status == 400 and non_object_body["error"] == "invalid_request"
finally:
    stop_api()

print("health:", health_status, health_body)
print("predict:", predict_status, predict_body)
print("missing columns:", invalid_status, invalid_body)
print("non-object JSON:", non_object_status, non_object_body)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Evidencia del despliegue y señales de operación
# MAGIC
# MAGIC Creamos un run separado porque desplegar es un evento distinto de entrenar. En un sistema real
# MAGIC registraríamos ventanas agregadas, no cada payload sensible. Aquí guardamos contadores,
# MAGIC latencias, versión servida y resultados de pruebas sin almacenar la petición completa.

# COMMAND ----------

deployment_evidence = {
    "mode": "local_notebook_http",
    "public_endpoint": False,
    "persistent": False,
    "model_registry_name": REGISTERED_MODEL_NAME,
    "model_version": str(champion.version),
    "model_alias": "Champion",
    "source_run_id": source_run_id,
    "tests": {
        "health": {"status": health_status},
        "valid_prediction": {"status": predict_status, "response_keys": sorted(predict_body)},
        "invalid_contract": {"status": invalid_status, "error": invalid_body.get("error")},
        "non_object_json": {"status": non_object_status, "error": non_object_body.get("error")},
    },
    "limitations": [
        "Sólo accesible desde el driver del notebook.",
        "Se detiene al terminar la sesión; no tiene SLA ni autenticación.",
        "No es apta para tráfico o datos reales.",
    ],
}

with mlflow.start_run(run_name=f"{STUDENT_ALIAS}-{BATCH_ID}-local-deployment") as deployment_run:
    mlflow.set_tags(
        {
            "course.week": "01",
            "student.alias": STUDENT_ALIAS,
            "batch.id": BATCH_ID,
            "lifecycle.phase": "deployment-test",
            "deployment.mode": "local-notebook",
            "deployment.public": "false",
            "model.source_run_id": source_run_id,
        }
    )
    mlflow.log_params(
        {
            "registered_model": REGISTERED_MODEL_NAME,
            "model_version": str(champion.version),
            "model_alias": "Champion",
        }
    )
    mlflow.log_metrics(
        {
            "api.requests_total": API_STATS["requests_total"],
            "api.requests_success": API_STATS["requests_success"],
            "api.requests_rejected": API_STATS["requests_rejected"],
            "api.requests_error": API_STATS["requests_error"],
            "api.latency_mean_ms": float(np.mean(API_STATS["latencies_ms"])),
            "api.latency_max_ms": float(np.max(API_STATS["latencies_ms"])),
            "api.health_latency_ms": health_ms,
            "api.predict_latency_ms": predict_ms,
            "api.invalid_latency_ms": invalid_ms,
        }
    )
    mlflow.log_dict(deployment_evidence, "deployment/evidence.json")
    DEPLOYMENT_RUN_ID = deployment_run.info.run_id

print("Run de despliegue:", DEPLOYMENT_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Apagado y cierre del ciclo
# MAGIC
# MAGIC Verificamos el apagado idempotente y cerramos la evidencia. Comprueba en la UI:
# MAGIC
# MAGIC - seis candidatos del mismo `batch.id`;
# MAGIC - el ganador con métricas de test y `selection.status=winner`;
# MAGIC - el modelo en Catalog Explorer con una versión y alias `Champion`;
# MAGIC - el run `deployment-test` con métricas y `deployment/evidence.json`.
# MAGIC
# MAGIC Preguntas para el debrief: ¿qué haría falta para rollback?, ¿qué monitorizarías por ventana?,
# MAGIC ¿qué aprobación debería preceder a producción?, ¿por qué esta API local no es producción?

# COMMAND ----------

stop_api()
assert not api_thread.is_alive(), "El servidor no se apagó correctamente."

summary = {
    "experiment_id": experiment_id,
    "batch_id": BATCH_ID,
    "candidate_runs": len(candidate_run_ids),
    "winner_run_id": BEST_RUN_ID,
    "registered_model": REGISTERED_MODEL_NAME,
    "registered_version": str(champion.version),
    "alias": "Champion",
    "deployment_run_id": DEPLOYMENT_RUN_ID,
    "api_stopped": API_STOPPED,
}
print(json.dumps(summary, indent=2))