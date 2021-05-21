import os
import sys
import glob
import threading
import avro
import json
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader
from pyspark.sql.avro.functions import from_avro
from pyspark.sql.types import *
from pyspark.sql.functions import *

from hdf5_getters import *
import pyspark
from pyspark.sql import SparkSession


# Just a word count sanity test to make sure that pyspark works as expected
if __name__ == "__main__":
    # create Spark context with necessary configuration
    sc = SparkSession.builder.appName('PySpark ML').master('local[*]').getOrCreate()

    sparkContext = sc.sparkContext
    sparkContext.setLogLevel("OFF")
    # sc.setLogLevel("OFF")

    parquetFile = sc.read.parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetFileTuple")

    # Parquet files can also be used to create a temporary view and then used in SQL statements.
    parquetFile.printSchema()
    parquetFile.show(20, True, True)
    print("Sanity check counter ", parquetFile.count())
    print("Describe info ", parquetFile.describe().show())
    print(len(parquetFile.columns))

    # parquetFile.filter(col('label') != -1).show()

    # parquetFile.withColumn("tester", lit("USE")).show(4)

    # parquetFile.withColumn("tester", when("USE").when).show(4)

    # parquetFile.withColumn("new_gender", when(col("gender") == = "M", "Male").when(col("gender") == = "F", "Female").otherwise("Unknown"))
    # parquetFile.withColumn('label', when(col('year') == 0, -1).when(col('year') < 2000, 0).otherwise(1)).show(20)