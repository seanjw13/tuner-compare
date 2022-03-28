# Databricks notebook source
from pyspark.sql.types import *
from sklearn.model_selection import train_test_split

def prepare_model_data(data_path, test_size=0.2):
  
  schema = StructType([
    StructField("fixed_acidity", DoubleType(), False),
    StructField("volatile_acidity", DoubleType(), True),
    StructField("citric_acid", DoubleType(), True),
    StructField("residual_sugar", DoubleType(), True),
    StructField("chlorides", DoubleType(), True),
    StructField("free_sulfur_dioxide", DoubleType(), True),
    StructField("total_sulphur_dioxide", DoubleType(), True),
    StructField("density", DoubleType(), True),
    StructField("pH", DoubleType(), True),
    StructField("sulphates", DoubleType(), True),
    StructField("alcohol", DoubleType(), True),
    StructField("quality", IntegerType(), True)])

  wine_df = spark.read.csv(path=data_path,
                           schema=schema,
                           header=True,
                           sep=";")

  clean_wine_df = wine_df.dropna("any").toPandas()

  train_x, test_x = train_test_split(clean_wine_df, 
                                     test_size=test_size, 
                                     shuffle=True)

  train_y = train_x.pop("quality")
  test_y = test_x.pop("quality")

  return train_x, test_x, train_y, test_y

# COMMAND ----------

import mlflow
from hyperopt import STATUS_OK
from sklearn.ensemble import GradientBoostingRegressor as gbr

def objective(params):
  
    with mlflow.start_run():
      # Create ensemble model
      model = gbr(learning_rate=params['lr'],
                  n_estimators=int(params['n_est']),
                  max_depth=int(params['max_depth']))
    
      model.fit(train_x, train_y)
      eval_results = model.score(test_x, test_y)
      mlflow.sklearn.log_model(model, 'wine_xgb')
    
    return {'loss': eval_results, 'status': STATUS_OK}

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials

def tune_model_databricks(parallelism=8, max_evals=100, h_params=None):
  
  default_space = {
             'lr': hp.loguniform('learning_rate', -5, -1),
             'max_depth': hp.quniform("max_depth", 2, 10, 1),
             'n_est': hp.quniform("n_estimators", 10, 200, 1)
  }
  
  space = h_params if h_params else default_space

  trial = SparkTrials(parallelism=parallelism)

  best_model = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trial)
  
  return best_model

# COMMAND ----------

train_x, test_x, train_y, test_y = prepare_model_data('dbfs:/databricks-datasets/wine-quality/winequality-red.csv')
#best_model = tune_model_databricks()

# COMMAND ----------

local_data_path = "/dbfs/FileStore/data/"

# COMMAND ----------

# Download dataframes to local file system before uploading to S3.
train_x.to_csv(local_data_path + "train_x.csv")
train_y.to_csv(local_data_path + "train_y.csv")
test_x.to_csv(local_data_path + "test_x.csv")
test_y.to_csv(local_data_path + "test_y.csv")

# COMMAND ----------

sm_client = boto3.client("sagemaker", region_name="us-west-2")
sess = sagemaker.session.Session(boto3.session.Session(region_name="us-west-2"), sagemaker_client=sm_client)

# COMMAND ----------

sess.upload_data(path=local_data_path + "train_x.csv")

# COMMAND ----------

# MAGIC %sh pip install boto3 sagemaker==2.78.0

# COMMAND ----------

import boto3
import sagemaker
from sagemaker.sklearn import SKLearn

sm_client = boto3.client("sagemaker", region_name="us-west-2")
sess = sagemaker.session.Session(boto3.session.Session(region_name="us-west-2"), sagemaker_client=sm_client)

estimator = SKLearn(entry_point="entry_point.py", 
                    source_dir="train_dir",
                    framework_version="0.23-1",
                    role="arn:aws:iam::997819012307:role/service-role/AmazonSageMaker-ExecutionRole-20200318T125733",
                    region="us-west-2",
                    instance_count=1,
                    instance_type="ml.m5.large",
                    sagemaker_session=sess)

#sagemaker.tuner.HyperparameterTuner(estimator=estimator,
#                                    objective_metric_name,
#                                    hyperparameter_ranges,
#                                    metric_definitions,
#                                    max_jobs,
#                                    max_parallel_jobs)

estimator.fit({'training':'s3://sagemaker-us-west-2-997819012307/data/train.csv',
               'test': 's3://sagemaker-us-west-2-997819012307/data/test.csv'})
#tuner.fit(wait=True)

# COMMAND ----------

a = 1.3434343
print("{0:.3f}".format(a))

# COMMAND ----------


