# Databricks notebook source
import pandas as pd

train_data = pd.read_csv("/dbfs/FileStore/data/train.csv")
test_data = pd.read_csv("/dbfs/FileStore/data/test.csv")

# COMMAND ----------

from hyperopt import STATUS_OK
from train_dir.train import train_model

def objective(params):

  eval_results = train_model(train_data, test_data,
                             n_est=int(params['n_est']),
                             max_depth=params['max_depth'],
                             lr=params['lr'])
    
  return {'loss': eval_results, 'status': STATUS_OK}

# COMMAND ----------

from hyperopt import fmin, hp, tpe, space_eval, SparkTrials

def tune_model_databricks(parallelism=1, max_evals=4, h_params=None):
  
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
  
  objective_metric = space_eval(space, best_model)
  
  return {"objective_metric": objective_metric, 
          "best_params" : best_model}

# COMMAND ----------

best_model = tune_model_databricks()
