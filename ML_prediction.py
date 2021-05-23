import os
import sys
import glob
import threading
import json
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
    NaiveBayes, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

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

    print("\n\nTotal count ", count_total)
    print("Correct prediction ", correct)
    print("Wrongly Identified: ", wrong)
    print("True Positive: ", truep)
    print("False Negative: ", falseN)
    print("False Positive: ", falseP)
    print("ratioWrong: ", ratioWrong)
    print("ratioCorrect: ", ratioCorrect)


def model_evaluator(evaluator, evaluator_name, data, data_type):
    print("\n", evaluator_name, " for", data_type, ": ", evaluator.evaluate(data))


def weighted_logistic_regression(training_data, test_data, validation_data):
    # ROC: 0.69
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
    model = lr.fit(training_data)

    metric_plotting(model)

    print("\nCoefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    # Make predictions on test data using the transform method
    # LogisticRegression.transform() will only use the features column
    # predict_train = model.transform(training_data)
    # predict_test = model.transform(test_data)
    predict_valid = model.transform(validation_data)

    evaluate_metrics(predict_valid)

    # View the predictions
    predict_valid.select('*').show(10)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    # After experimenting and reading more about different metrics, areaUnderROC metric seems to be proper for our purposes
    # evaluator3 = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
    #                                            metricName="areaUnderPR")
    # evaluator2 = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_valid, data_type="valid_data")
    # model_evaluator(evaluator=evaluator2, evaluator_name="accuracy", data=predict_valid, data_type="valid_data")
    # model_evaluator(evaluator=evaluator3, evaluator_name="areaUnderPR", data=predict_valid, data_type="valid_data")

    # Create ParamGrid for Cross Validation
    # This grid takes a while. Choose another one for the next implementation
    # ROC: 0.694. Took for ever with these parameters. Execute with different parameters.
    # This seems to be the best we are going to get with this model
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(lr.aggregationDepth, [2, 5, 10]) \
    #     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    #     .addGrid(lr.fitIntercept, [False, True]) \
    #     .addGrid(lr.maxIter, [10, 100, 1000]) \
    #     .addGrid(lr.regParam, [0.01, 0.5, 2.0]) \
    #     .build()

    print("\n\nParameter Grid and cross validation")
    # ROC: 0.694
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.aggregationDepth, [2, 5, 10]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(lr.fitIntercept, [False, True]) \
        .addGrid(lr.maxIter, [20, 50, 100]) \
        .addGrid(lr.regParam, [0.01, 0.2, 1.0]) \
        .build()

    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    # Run cross validations
    cvModel = cv.fit(training_data)

    # predict_train = cvModel.transform(training_data)
    predict_cross_valid = cvModel.transform(validation_data)
    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_cross_valid,
                    data_type="valid_data")


# Simple Decision tree performs poorly because it is too weak given the range of different features
def decision_tree_classifier(training_data, test_data, validation_data):
    # ROC 0.69
    # dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=3)

    # ROC 0.46
    # dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=10)

    # ROC 0.68
    dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol='label', maxDepth=3, impurity='entropy')

    model = dt.fit(training_data)

    predict_valid = model.transform(validation_data)
    # predict_train = model.transform(training_data)
    # predict_valid.show(10)

    evaluate_metrics(predict_valid)

    predict_valid.select('*').show(10)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_valid, data_type="valid_data")

    print("\n\nParameter Grid and cross validation")

    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [1, 2, 6, 10]) \
        .addGrid(dt.maxBins, [20, 40, 80]) \
        .build()

    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

    # Run cross validations
    cvModel = cv.fit(training_data)

    print("numNodes = ", cvModel.bestModel.numNodes)
    print("depth = ", cvModel.bestModel.depth)

    # Use test set to measure the accuracy of the model on new data
    predict_cross_valid = cvModel.transform(validation_data)
    # cvModel uses the best model found from the Cross Validation
    # Evaluate best model

    # ROC 0.706, Slightly better than Logistic Regresion
    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_cross_valid,
                    data_type="valid_data")


def random_forest_classifier(training_data, test_data, validation_data):
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

    rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='label', weightCol='classWeights', numTrees=25,
                                maxDepth=5, maxBins=32)

    rfModel = rf.fit(training_data)

    # print(rfModel.featureImportances)

    # Plot roc curve
    roc_plot(rfModel)

    predict_valid = rfModel.transform(validation_data)
    # predict_train = rfModel.transform(training_data)
    predict_valid.show(5)

    evaluate_metrics(predict_valid)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_valid,
                    data_type="valid_data")

    # print("\n\nParameter Grid and cross validation")
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
    # cvModel = cv.fit(training_data)
    #
    # predict_cross_valid = cvModel.transform(validation_data)
    #
    # model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_cross_valid,
    #                 data_type="valid_data")
    #
    # predict_final = cvModel.bestModel.transform(test_data)
    #
    # model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_final,
    #                 data_type="test_data")


def gradient_boosted_tree_classifier(training_data, test_data, validation_data):
    # ROC: 0.71
    gbt = GBTClassifier(featuresCol='scaled_features', labelCol='label', maxIter=10)
    gbtModel = gbt.fit(training_data)

    predict_valid = gbtModel.transform(validation_data)
    # predict_train = gbtModel.transform(training_data)
    predict_valid.show(5)

    evaluate_metrics(predict_valid)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_valid,
                    data_type="valid_data")

    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [2, 4, 6])
                 .addGrid(gbt.maxBins, [20, 60])
                 .addGrid(gbt.maxIter, [10, 20])
                 .build())

    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    # Run cross validations.
    cvModel = cv.fit(training_data)
    predict_cross_valid = cvModel.transform(validation_data)

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_cross_valid,
                    data_type="valid_data")

    predict_final = cvModel.bestModel.transform(test_data)

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_final,
                    data_type="test_data")


# The worst of all. Don't search any further
def bayes_classifier(training_data, test_data, validation_data):
    dt = NaiveBayes(featuresCol='scaled_features', labelCol='label', smoothing=0.00001)

    # ROC 0.43
    dtModel = dt.fit(training_data)
    predict_valid = dtModel.transform(validation_data)
    predict_valid.show(10)

    evaluate_metrics(predict_valid)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_valid,
                    data_type="valid_data")


def linear_support_vector_machines(training_data, test_data, validation_data):
    # ROC 0.67
    # lsvc = LinearSVC(featuresCol='scaled_features', labelCol='label',maxIter=10, regParam=0.1)

    # ROC 0.69
    lsvc = LinearSVC(featuresCol='scaled_features', labelCol='label', maxIter=100, regParam=0.01)
    # Fit the model
    lsvcModel = lsvc.fit(training_data)

    predict_valid = lsvcModel.transform(validation_data)
    predict_valid.show(10)

    evaluate_metrics(predict_valid)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label',
                                              metricName="areaUnderROC")

    model_evaluator(evaluator=evaluator, evaluator_name="areaUnderROC", data=predict_valid,
                    data_type="valid_data")


'''
    The baseline model for the implementation. Always predict the same value. ROC value expected to be 0.5(and is)
'''


def baseline_model(df_data):
    bs_model = df_data.withColumn("base_prediction", lit(float(1)))
    bs_model.select('*').show(20)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="base_prediction", labelCol='label',
                                              metricName="areaUnderROC")

    print('Baseline model Area Under ROC', evaluator.evaluate(bs_model))


# Just a word count sanity test to make sure that pyspark works as expected
if __name__ == "__main__":
    # create Spark context with necessary configuration
    sc = SparkSession.builder.appName('PySpark ML').master('local[*]').getOrCreate()

    sparkContext = sc.sparkContext
    sparkContext.setLogLevel("OFF")

    # Persist as we will use this multiple times
    df_parquet = sc.read.parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetAfterProcess").persist()

    # Sanity check print schema and some of the values
    df_parquet.printSchema()
    df_parquet.show(5, True, True)

    print("Sanity check counter ", df_parquet.count())
    print("Column Count ", len(df_parquet.columns))

    print("Count 1s :", df_parquet.filter(col('label') == 1).count())
    print("Count 0s", df_parquet.filter(col('label') == 0).count())
    print("Sanity check sum 1s and 0s",
          df_parquet.filter(col('label') == 1).count() + df_parquet.filter(col('label') == 0).count())

    # Baseline model implementation
    baseline_model(df_data=df_parquet)

    # https://www.quora.com/What-are-the-best-ways-to-predict-data-once-you-have-your-input-splitted-into-train-cross_validation-and-test-sets
    training_data, validation_data, test_data = df_parquet.randomSplit(weights=[0.60, 0.20, 0.20], seed=1234)

    print("\n\nDataset Counting")
    print("Training Dataset Count: ", str(training_data.count()))
    print("Test Dataset Count: ", str(test_data.count()))
    print("Validation Dataset Count: ", str(validation_data.count()))
    print("Total Sanity: ", str(training_data.count() + test_data.count() + validation_data.count()))

    # weighted_logistic_regression(training_data=training_data, test_data=test_data, validation_data=validation_data)
    # decision_tree_classifier(training_data=training_data,test_data=test_data, validation_data=validation_data)
    random_forest_classifier(training_data=training_data, test_data=test_data, validation_data=validation_data)
    # gradient_boosted_tree_classifier(training_data=training_data, test_data=test_data, validation_data=validation_data)
    # bayes_classifier(training_data=training_data,test_data=test_data, validation_data=validation_data)
    # linear_support_vector_machines(training_data=training_data, test_data=test_data, validation_data=validation_data)
