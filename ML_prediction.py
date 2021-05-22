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
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from hdf5_getters import *
import pyspark
from pyspark.sql import SparkSession


def coefficient_plot(lrModel):
    beta = np.sort(lrModel.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()


def roc_plot(lrModel):
    trainingSummary = lrModel.summary
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'], roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


def precision_plot(lrModel):
    trainingSummary = lrModel.summary
    pr = trainingSummary.pr.toPandas()
    plt.plot(pr['recall'], pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def metric_plotting(lrModel):
    coefficient_plot(lrModel)
    roc_plot(lrModel)
    precision_plot(lrModel)


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

def model_evaluator(predict_train,predict_valid):

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")
    evaluator3 = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                               metricName="areaUnderPR")
    evaluator2 = MulticlassClassificationEvaluator(labelCol='label', metricName='f1')
    # evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label')

    # predict_test.select("*").show(5)
    print("areaUnderROC {}".format(evaluator.evaluate(predict_train)))
    # print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))
    print("areaUnderROC valid {}".format(evaluator.evaluate(predict_valid)))


    print("f1 {}".format(evaluator2.evaluate(predict_train)))
    # print("The area under ROC for test set is {}".format(evaluator2.evaluate(predict_test)))
    print("f1 valid {}".format(evaluator2.evaluate(predict_valid)))

    print("areaUnderPR {}".format(evaluator3.evaluate(predict_train)))
    # print("The area under ROC for test set is {}".format(evaluator3.evaluate(predict_test)))
    print("areaUnderPR valid {}".format(evaluator3.evaluate(predict_valid)))


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

    metric_plotting(lrModel)

    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    predictions = lrModel.transform(testData)
    predictions.show()

    evaluate_metrics(predictions)

    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary
    #
    # # Obtain the objective per iteration
    # objectiveHistory = trainingSummary.objectiveHistory
    # print("objectiveHistory:")
    # for objective in objectiveHistory:
    #     print(objective)
    #
    # evaluate_metrics(predictions)

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)

    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))


def weighted_logistic_regression(trainingData, testData, validationData):
    # ROC: 0.68
    lr = LogisticRegression(featuresCol='scaled_features', labelCol='label', weightCol='classWeights', maxIter=100)
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
    # lr = LogisticRegression(featuresCol='scaled_features', labelCol='label',weightCol='classWeights', maxIter=100, regParam=0.01,
    #                         elasticNetParam=0.2)

    # Train model using training Data
    model = lr.fit(trainingData)

    # metric_plotting(model)

    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))



    # Make predictions on test data using the transform method
    # LogisticRegression.transform() will only use the features column
    predict_train = model.transform(trainingData)
    predict_test = model.transform(testData)
    predict_valid = model.transform(validationData)

    evaluate_metrics(predict_test)

    # View the predictions
    predict_test.select('*').show(10)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',metricName="areaUnderROC")
    evaluator3 = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',metricName="areaUnderPR")
    evaluator2 = MulticlassClassificationEvaluator(labelCol='label',metricName='f1')
    # evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label')

    predict_test.select("*").show(5)
    print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
    print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))
    print("The area under ROC for valid set is {}".format(evaluator.evaluate(predict_valid)))

    print("The area under ROC for train set is {}".format(evaluator2.evaluate(predict_train)))
    print("The area under ROC for test set is {}".format(evaluator2.evaluate(predict_test)))
    print("The area under ROC for valid set is {}".format(evaluator2.evaluate(predict_valid)))

    print("The area under ROC for train set is {}".format(evaluator3.evaluate(predict_train)))
    print("The area under ROC for test set is {}".format(evaluator3.evaluate(predict_test)))
    print("The area under ROC for valid set is {}".format(evaluator3.evaluate(predict_valid)))

    # Create ParamGrid for Cross Validation
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(lr.aggregationDepth, [2, 5, 10]) \
    #     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    #     .addGrid(lr.fitIntercept, [False, True]) \
    #     .addGrid(lr.maxIter, [10, 100, 1000]) \
    #     .addGrid(lr.regParam, [0.01, 0.5, 2.0]) \
    #     .build()
    #
    # # Create 5-fold CrossValidator
    # cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    # # Run cross validations
    # cvModel = cv.fit(trainingData)
    #
    # predict_train = cvModel.transform(trainingData)
    # predict_test = cvModel.transform(testData)
    #
    # # ROC: 0.694. Took for ever with these parameters. Execute with different parameters.
    # # This seems to be the best we are going to get with this model
    # print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
    # print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))


# Simple Decision tree performs poorly because it is too weak given the range of different features
def decision_tree_classifier(trainingData, testData, validationData):
    from pyspark.ml.classification import DecisionTreeClassifier

    # ROC 0.69
    # dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=3)

    # ROC 0.46
    # dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=10)

    # ROC 0.68
    dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=3,impurity='entropy')

    model = dt.fit(trainingData)


    predict_valid = model.transform(validationData)
    predict_train = model.transform(trainingData)
    # predict_valid.show(10)

    evaluate_metrics(predict_valid)

    predict_valid.select('*').show(10)

    model_evaluator(predict_train,predict_valid)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [1, 2, 6, 10]) \
        .addGrid(dt.maxBins, [20, 40, 80]) \
        .build()
    #
    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

    # Run cross validations
    cvModel = cv.fit(trainingData)

    print("numNodes = ", cvModel.bestModel.numNodes)
    print("depth = ", cvModel.bestModel.depth)

    # Use test set to measure the accuracy of the model on new data
    predict_valid = cvModel.transform(validationData)
    # cvModel uses the best model found from the Cross Validation
    # Evaluate best model

    print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_valid)))




def random_forest_classifier(trainingData, testData,validationData):
    from pyspark.ml.classification import RandomForestClassifier

    # ROC: 0.73
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label')

    # ROC: 0.75
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', numTrees=50, maxDepth=30, maxBins=32)

    # ROC: 0.75
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', weightCol='classWeights', numTrees=50,
    #                             maxDepth=30, maxBins=32)

    # ROC: 0.75
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', numTrees=50, impurity='entropy',
    #                             maxDepth=30, maxBins=32)

    # ROC: 0.70
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', numTrees=50, impurity='entropy',
    #                             maxDepth=30, maxBins=2)

    # ROC: 0.76
    # rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', numTrees=100,
    #                             maxDepth=30, maxBins=100)

    rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', numTrees=25, maxDepth=5, maxBins=32)

    rfModel = rf.fit(trainingData)
    predict_valid = rfModel.transform(validationData)
    predict_train = rfModel.transform(trainingData)
    predict_valid.show(5)

    # evaluate_metrics(predict_valid)

    model_evaluator(predict_train, predict_valid)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")
    # evaluator = BinaryClassificationEvaluator()
    # print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_valid)))

    # from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    #
    #
    #
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(rf.maxDepth, [2, 4, 6]) \
    #     .addGrid(rf.maxBins, [20, 60]) \
    #     .addGrid(rf.numTrees, [5, 20]) \
    #     .build()
    #
    # # Create 5-fold CrossValidator
    # cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    #
    # # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
    # cvModel = cv.fit(trainingData)
    #
    # predict_valid = cvModel.transform(validationData)
    #
    # print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_valid)))
    #
    # finalPredictions = cvModel.bestModel.transform(testData)
    #
    # print("Final prediction is {}".format(evaluator.evaluate(finalPredictions)))




def gradient_boosted_tree_classifier(trainingData, testData, validationData):
    # ROC: 0.70
    from pyspark.ml.classification import GBTClassifier
    gbt = GBTClassifier(featuresCol='scaled_features', labelCol='label', maxIter=10)
    gbtModel = gbt.fit(trainingData)

    predict_valid = gbtModel.transform(validationData)
    predict_train = gbtModel.transform(trainingData)
    predict_valid.show(5)

    # evaluate_metrics(predict_valid)

    model_evaluator(predict_train, predict_valid)

    # predictions = gbtModel.transform(testData)
    # predictions.show(10)
    #
    # evaluator = BinaryClassificationEvaluator()
    # print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")



    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [2, 4, 6])
                 .addGrid(gbt.maxBins, [20, 60])
                 .addGrid(gbt.maxIter, [10, 20])
                 .build())
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
    cvModel = cv.fit(trainingData)
    predictions = cvModel.transform(validationData)
    evaluator.evaluate(predictions)

    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# The worst of all. Don't search any further
def bayes_classifier(trainingData, testData, validationData):
    from pyspark.ml.classification import NaiveBayes

    dt = NaiveBayes(featuresCol='scaled_features', labelCol='label', smoothing=0.00001)

    # ROC 0.43
    dtModel = dt.fit(trainingData)
    predictions = dtModel.transform(validationData)
    predictions.show(10)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")
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

    # weighted_logistic_regression(trainingData=trainingData, testData=testData, validationData=validationData)
    # logistic_regression(trainingData=trainingData, testData=testData)
    # decision_tree_classifier(trainingData=trainingData,testData=testData, validationData=validationData)
    random_forest_classifier(trainingData=trainingData,testData=testData, validationData=validationData)
    # gradient_boosted_tree_classifier(trainingData=trainingData, testData=testData, validationData=validationData)
    # bayes_classifier(trainingData=trainingData,testData=testData, validationData=validationData)
