import os
import sys
import glob
import threading
import avro
import json
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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

    parquetFile = sc.read.parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetTimeBig")

    # Parquet files can also be used to create a temporary view and then used in SQL statements.
    parquetFile.printSchema()
    parquetFile.show(20, True, True)
    print("Sanity check counter ", parquetFile.count())
    print("Describe info ", parquetFile.describe().show())
    print(len(parquetFile.columns))

    print("Count 1s :", parquetFile.filter(col('label') == 1).count())
    print("Count 0s", parquetFile.filter(col('label') == 0).count())
    print("Sanity check sum 1s and 0s",
          parquetFile.filter(col('label') == 1).count() + parquetFile.filter(col('label') == 0).count())



    feature_selector = parquetFile.select("artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo",
                                          "time_signature_confidence",
                                          "artist_playmeid", "artist_7digitalid", "release_7digitalid",
                                          "track_7digitalid", "key", "loudness", "mode",
                                          "mode_confidence", "time_signature", "label")

    feature_selector.show(1)
    feature_selector.describe().show()

    columns = ["artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo",
                                          "time_signature_confidence",
                                          "artist_playmeid", "artist_7digitalid", "release_7digitalid",
                                          "track_7digitalid", "key", "loudness", "mode",
                                          "mode_confidence", "time_signature", "label"]

    vector_col = "features"
    assembler = VectorAssembler(inputCols=columns,
                                outputCol=vector_col)
    # myGraph_vector = assembler.transform(feature_selector).select(vector_col)

    df2 = assembler.transform(feature_selector)

    df2.show()

    trainingData, testData = df2.randomSplit(weights = [0.80, 0.20], seed = 1234567)

    lr = LogisticRegression(featuresCol='features',labelCol='label',maxIter=2, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(trainingData)

    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))


    predictions = lrModel.transform(testData)
    predictions.show()



    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary


    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    # trainingSummary.roc.show()
    # print("areaUnderROC: " + str(trainingSummary.areaUnderROC))


    lp = predictions.select("label","prediction")
    count_total = predictions.count()
    correct = lp.filter(col('label') == col('prediction')).count()
    wrong = lp.filter(col('label') != col('prediction')).count()
    truep = lp.filter(col('prediction') == 0).filter(col('label') == col('prediction')).count()
    falseN = lp.filter(col('prediction') == 0.0).filter(col('label') != col('prediction')).count()
    falseP = lp.filter(col('prediction') == 1.0).filter(col("label") != col("prediction")).count()
    ratioWrong = float(wrong) / float(count_total)
    ratioCorrect = float(correct) / float(count_total)

    print("Total count ", count_total)
    print("Correct prediction ",correct)
    print("Wrongly Identified: ", wrong)
    print("True Positive: ", truep)
    print("False Negative: ", falseN)
    print("False Positive: ", falseP)
    print("ratioWrong: ", ratioWrong)
    print("ratioCorrect: ", ratioCorrect)


    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)

    # evaluator = BinaryClassificationEvaluator().setLabelCol("label")
    # accuracy = evaluator.evaluate(predictions)

    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    # print("Accuracy: ", accuracy)







