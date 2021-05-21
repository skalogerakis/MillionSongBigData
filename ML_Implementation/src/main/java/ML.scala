import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.dsl.expressions.{DslExpression, StringToAttributeConversionHelper}
import org.apache.spark.{SparkConf, SparkContext}
//import spark.implicits._
import org.apache.spark.sql.functions.col

object ML {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)


    //As the link below states, spark-shell creates by default Spark Context so use that
    //    https://sparkbyexamples.com/spark/sparksession-vs-sparkcontext/?fbclid=IwAR15guKOla8APJa3paaFCNbmfkRRhVp_Il_tOo9F005XpECpj2m1R-uGXkU
//    val spark = new SparkContext(new SparkConf().setAppName("SimpleApp").setMaster("local[*]"))

    val spark = SparkSession
      .builder()
      .appName("ML prediction")
      .master("local[*]")
      .getOrCreate()

    val parquetFile = spark.read.parquet("/home/skalogerakis/Projects/MillionSongBigData/parquetTimeBig")

    parquetFile.printSchema()
    parquetFile.show(2,true)

    println("Sanity check counter "+parquetFile.count())
    parquetFile.describe().show()
    println("Column length "+parquetFile.columns.length)

    println("Counter1 ", +parquetFile.filter(x => x!=0).count())
    println("Counter2 ", +parquetFile.filter(x => x!=1).count())
//    println("Column length "+parquetFile.filter($""))

    val columns = Array("artist_familiarity", "end_of_fade_in", "start_of_fade_out", "tempo", "time_signature_confidence",
      "artist_playmeid","artist_7digitalid", "release_7digitalid", "track_7digitalid", "key", "loudness", "mode",
      "mode_confidence", "time_signature", "label")


    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

    val df2 = assembler.transform(parquetFile)

    df2.show(3,true)

    val splitSeed = 1234567
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), splitSeed)


    val lr = new LogisticRegression().setMaxIter(2).setRegParam(0.01).setElasticNetParam(0.01)
    val model = lr.fit(trainingData)

    val predictions = model.transform(testData)
    predictions.show()


    val trainingSummary = model.summary
    val objectiveHistory = trainingSummary.objectiveHistory
//    objectiveHistory.foreach(loss => println(loss))

    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]


    val roc = binarySummary.roc
    roc.show()
    println("Area Under ROC: " + binarySummary.areaUnderROC)


    import org.apache.spark.sql.functions._

    // Calculate the performance metrics
//    val lp = predictions.select("label", "prediction")
//    val counttotal = predictions.count()
//    val correct = lp.filter($"label" === $"prediction").count()
//    val wrong = lp.filter(not($"label" === $"prediction")).count()
//    val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
//    val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()
//    val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
//    val ratioWrong = wrong.toDouble / counttotal.toDouble
//    val ratioCorrect = correct.toDouble / counttotal.toDouble
//
//    println("Total Count: " + counttotal)
//    println("Correctly Predicted: " + correct)
//    println("Wrongly Identified: " + wrong)
//    println("True Positive: " + truep)
//    println("False Negative: " + falseN)
//    println("False Positive: " + falseP)
//    println("ratioWrong: " + ratioWrong)
//    println("ratioCorrect: " + ratioCorrect)
//
//
//    val fMeasure = binarySummary.fMeasureByThreshold
//    val fm = fMeasure.col("F-Measure")
//    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
//    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
//    model.setThreshold(bestThreshold)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)


  }


}
