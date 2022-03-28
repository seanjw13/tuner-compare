# Databricks notebook source
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

# COMMAND ----------

from sagemaker.parameter import ContinuousParameter, IntegerParameter

metrics = [{"Name": "R2", "Regex": "R2:(.*?);"}]
hyperparameters = {'max_depth' : IntegerParameter(2, 10),
                   'lr' : ContinuousParameter(.0001, .01),
                   'n_est': IntegerParameter(10, 200)}

tuner = sagemaker.tuner.HyperparameterTuner(estimator=estimator,
                                            objective_metric_name="R2",
                                            hyperparameter_ranges=hyperparameters,
                                            metric_definitions=metrics,
                                            max_jobs=4,
                                            max_parallel_jobs=1)

tuner.fit({'training':'s3://sagemaker-us-west-2-997819012307/data/train.csv',
           'test': 's3://sagemaker-us-west-2-997819012307/data/test.csv'})

# COMMAND ----------


