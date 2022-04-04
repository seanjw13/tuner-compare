# Databricks notebook source
#dbutils.notebook.run("launch-sagemaker-tuner", timeout_seconds=60*60)

# COMMAND ----------

import json

with open("default_job_settings.json") as fp:
  mtj_settings = json.load(fp)

# COMMAND ----------

token = dbutils.secrets.get(scope="sewi", key="api-token")
headers = {"Authorization" : f"Bearer {token}"} 

r = requests.post("https://e2-demo-west.cloud.databricks.com/api/2.1/jobs/create", headers=headers, json=mtj_settings)
