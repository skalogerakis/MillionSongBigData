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
import pandas as pd

from hdf5_getters import *
import pyspark
from pyspark.sql import SparkSession


def evaluate_metrics(predictions):
    lp = predictions.select("label", "prediction")
    count_total = predictions.count()
    correct = lp.filter(col('label') == col('prediction')).count()
    wrong = lp.filter(col('label') != col('prediction')).count()
    truep = lp.filter(col('prediction') == 0).filter(col('label') == col('prediction')).count()
    falseN = lp.filter(col('prediction') == 0.0).filter(col('label') != col('prediction')).count()
    falseP = lp.filter(col('prediction') == 1.0).filter(col("label") != col("prediction")).count()
    ratioWrong = float(wrong) / float(count_total)
    ratioCorrect = float(correct) / float(count_total)

    print("Total count ", count_total)
    print("Correct prediction ", correct)
    print("Wrongly Identified: ", wrong)
    print("True Positive: ", truep)
    print("False Negative: ", falseN)
    print("False Positive: ", falseP)
    print("ratioWrong: ", ratioWrong)
    print("ratioCorrect: ", ratioCorrect)


def logistic_regression(trainingData, testData):
    # Always the same prediction. ROC: 0.5
    # lr = LogisticRegression(featuresCol='scaled_features',labelCol='label',maxIter=100, regParam=0.3, elasticNetParam=0.8)
    # A little better, still not very good. ROC: 0.663
    # lr = LogisticRegression(featuresCol='scaled_features', labelCol='label', maxIter=100, regParam=0.1,
    #                         elasticNetParam=0.8)
    # ROC: 0.60
    # lr = LogisticRegression(featuresCol='scaled_features', labelCol='label', maxIter=20, regParam=0.01,
    #                         elasticNetParam=0.3)
    # ROC: 0.61
    # lr = LogisticRegression(featuresCol='scaled_features', labelCol='label', maxIter=20, regParam=1e-10,
    #                         elasticNetParam=0.2)
    # ROC 0.65. It seems that in the most iterations things are getting improved
    # lr = LogisticRegression(featuresCol='scaled_features', labelCol='label', maxIter=100, regParam=1e-10,
    #                         elasticNetParam=0.2)
    # ROC: 0.66 The best one. However, to improve we must move to other choices
    lr = LogisticRegression(featuresCol='scaled_features', labelCol='label', maxIter=100, regParam=0.01,
                            elasticNetParam=0.2)

    # Fit the model
    lrModel = lr.fit(trainingData)

    import matplotlib.pyplot as plt
    import numpy as np

    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    beta = np.sort(lrModel.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()

    trainingSummary = lrModel.summary
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'], roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

    pr = trainingSummary.pr.toPandas()
    plt.plot(pr['recall'], pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

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

    evaluate_metrics(predictions)

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)

    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))


# Simple Decision tree performs poorly because it is too weak given the range of different features
def decision_tree_classifier(trainingData, testData):
    from pyspark.ml.classification import DecisionTreeClassifier

    # ROC 0.615
    dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=3)

    # ROC 0.611
    # dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=10)

    dtModel = dt.fit(trainingData)
    predictions = dtModel.transform(testData)
    predictions.show(10)

    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))


def random_forest_classifier(trainingData, testData):
    from pyspark.ml.classification import RandomForestClassifier

    # ROC: 0.72
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label')

    rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', numTrees=50, maxDepth=30, maxBins=32)

    rfModel = rf.fit(trainingData)
    predictions = rfModel.transform(testData)
    predictions.show(10)

    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


def gradient_boosted_tree_classifier(trainingData, testData):
    # ROC: 0.70
    from pyspark.ml.classification import GBTClassifier
    gbt = GBTClassifier(featuresCol='scaled_features', labelCol='label', maxIter=10)
    gbtModel = gbt.fit(trainingData)
    predictions = gbtModel.transform(testData)
    predictions.show(10)

    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [2, 4, 6])
                 .addGrid(gbt.maxBins, [20, 60])
                 .addGrid(gbt.maxIter, [10, 20])
                 .build())
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
    cvModel = cv.fit(trainingData)
    predictions = cvModel.transform(testData)
    evaluator.evaluate(predictions)

    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


def bayes_classifier(trainingData, testData):
    from pyspark.ml.classification import NaiveBayes

    dt = NaiveBayes(featuresCol='scaled_features', labelCol='label', smoothing=0.00001)

    dtModel = dt.fit(trainingData)
    predictions = dtModel.transform(testData)
    predictions.show(10)

    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))


# Just a word count sanity test to make sure that pyspark works as expected
if __name__ == "__main__":
    # create Spark context with necessary configuration
    sc = SparkSession.builder.appName('PySpark ML').master('local[*]').getOrCreate()

    sparkContext = sc.sparkContext
    sparkContext.setLogLevel("OFF")
    # sc.setLogLevel("OFF")

    parquetFile = sc.read.parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetAfterProcess")

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

    pd.DataFrame(parquetFile.take(5), columns=parquetFile.columns).transpose()

    # https://www.quora.com/What-are-the-best-ways-to-predict-data-once-you-have-your-input-splitted-into-train-cross_validation-and-test-sets
    trainingData, validationData, testData = parquetFile.randomSplit(weights=[0.60, 0.20, 0.20], seed=1234567)

    print("Training Dataset Count: ", str(trainingData.count()))
    print("Test Dataset Count: ", str(testData.count()))
    print("Validation Dataset Count: ", str(validationData.count()))
    print("Total Sanity: ", str(trainingData.count() + testData.count() + validationData.count()))

    logistic_regression(trainingData=trainingData, testData=testData)
    # decision_tree_classifier(trainingData=trainingData,testData=testData)
    # random_forest_classifier(trainingData=trainingData,testData=testData)
    # gradient_boosted_tree_classifier(trainingData=trainingData, testData=testData)
    # bayes_classifier(trainingData=trainingData,testData=testData)
