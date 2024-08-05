import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SquareIntegers").getOrCreate()

user_input = input("Enter a set of integers separated by spaces: ")
integers = list(map(int, user_input.split()))

rdd = spark.sparkContext.parallelize(integers)

def square(x):
    return x * x

squared_rdd = rdd.map(square)
squared_numbers = squared_rdd.collect()

print("Original Numbers:", integers)
print("Squared Numbers:", squared_numbers)

spark.stop()