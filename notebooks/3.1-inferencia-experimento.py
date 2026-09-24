# Databricks notebook source
import requests

# COMMAND ----------

url = 'http://127.0.0.1:5001/invocations'

input_data = {
    "dataframe_split" :{
    "columns": ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5','s6'],
    "data": [ [ 0.03807591,  0.05068012,  0.06169621,  0.02187239, -0.0442235 , -0.03482076, -0.04340085, -0.00259226,  0.01990749, -0.01764613]]
    }
}

# COMMAND ----------

resp = requests.post(url, json = input_data)

# COMMAND ----------

resp.text