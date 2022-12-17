import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from typing import List
from pyspark.ml.classification import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def get_spark_session(app_name:str = "Scorecard"):
    spark = SparkSession.builder.appName(app_name).config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()
    return spark

# def read_csv_data(filepath:str):
#     df = self.spark.read.option("header",True).csv(filepath, inferSchema=True)
#     return df

# def remove_cols(remove_cols_list):
#     self.df = self.df.drop(*remove_cols_list)