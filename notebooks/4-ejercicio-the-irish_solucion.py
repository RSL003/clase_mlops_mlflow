# Databricks notebook source
!pip install mlflow --quiet
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets
 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

# COMMAND ----------

dataset = datasets.load_iris()

# COMMAND ----------

X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

mlflow.sklearn.autolog()
mlflow.set_experiment("Iris Experiment")

# COMMAND ----------

with mlflow.start_run():
    max_depth = 10
    max_features = 2

    dt = DecisionTreeClassifier(max_depth = max_depth, max_features = max_features)
    dt.fit(X_train, y_train)
    y_preds = dt.predict(X_test)
    acc = accuracy_score(y_test, y_preds)

  # Logueamos los parametros
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)

    mlflow.log_metric("accuracy", acc)

mlflow.end_run()