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

def time_difference_calculator(datetime1,datetime2):

    from datetime import datetime
    import pytz
    # Getting the time in America/New_York timezone
    timezone_newyork= pytz.timezone('America/New_York')
    # Getting the time in Europe/London timezone
    timezone_london = pytz.timezone("Europe/London")
    # input the date time of the newyork in the format Year, Month, Day, Hour, Minute, Second
    newyorkDateTime = datetime(2013, 3, 15, 20, 5, 10)
    #input date time of the london
    londonDateTime = datetime(2013, 3, 15, 20, 5, 10)
    # Localize the given date, according to the timezone objects
    datewith_tz_newyork1 = timezone_newyork.localize(datetime1)
    datewith_tz_newyork2 = timezone_newyork.localize(datetime2)
    # datewith_tz_london = timezone_london.localize(londonDateTime)
    # These are now, effectively no longer the same *date* after being localized
    # print("The date and time with timezone newyork:", datewith_tz_newyork)
    # print("The date and time with timezone london:", datewith_tz_london)
    difference_1 = int(datewith_tz_newyork1.strftime('%z'))
    difference_2 = int(datewith_tz_newyork1.strftime('%z'))
    # comparingthe date with different timezonesafter they are in the same timezone
    # if(difference > difference2):
    #     print('Given Two Times of two different Time Zones are equal',(difference-difference2)/100,'hours')
    # else:
    #     print('Given Two Times of two different Time Zones are not equal by',(difference2-difference)/100,'hours')
    return difference_1-difference_2
# def read_csv_data(filepath:str):
#     df = self.spark.read.option("header",True).csv(filepath, inferSchema=True)
#     return df

# def remove_cols(remove_cols_list):
#     self.df = self.df.drop(*remove_cols_list)