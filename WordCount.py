import pyspark
from pyspark.sql import SparkSession

# Just a word count sanity test to make sure that pyspark works as expected
if __name__ == "__main__":
    # conf = SparkConf().setAppName("PySpark Word Count").setMaster("local")
    # # create Spark context with necessary configuration
    # sc = SparkContext(conf=conf)
    sc = SparkSession.builder.appName('PySpark Word Count').master("local[*]").getOrCreate()
    sparkContext = sc.sparkContext

    # read data from text file and split each line into words
    words = sparkContext.textFile("/home/skalogerakis/Projects/MillionSongBigData/MyDocs/WordCountSanity").flatMap(lambda line: line.split(" "))

    # count the occurrence of each word
    wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

    wordCounts.foreach(print)