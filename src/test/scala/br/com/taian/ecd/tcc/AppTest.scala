package br.com.taian.ecd.tcc

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.scalatest.BeforeAndAfter
import org.scalatest.wordspec.AnyWordSpec
class AppTest extends AnyWordSpec with BeforeAndAfter with SparkSessionTrait {

  import spark.implicits._

  val datasetsPath: String           = "datasets"
  val modelsPath: String             = s"$datasetsPath/model"
  val trainingData: DataFrame        = spark.read
    .parquet(s"$datasetsPath/training")
    .cache()
  val testData: DataFrame            =
    spark.read.parquet(s"$datasetsPath/test").cache()
  val columnsFeatures: Array[String] = spark.read
    .json(s"$datasetsPath/features")
    .as[String]
    .collect()
  val times: DataFrame               = spark.read.json(s"$datasetsPath/times")
  val columnsMetrics: Seq[String]    = Seq(
    "modelo",
    "tempo de treino",
    "numTrains",
    "tempo por modelo",
    "accuracy",
    "weightedPrecision",
    "weightedRecall",
    "f1"
  )
  val evaluator                      = new MulticlassClassificationEvaluator()
  val predictionCols: Seq[String]    =
    Seq("label", "prediction", "rawPrediction", "probability", "features")

  "Evaluate models" should {

    "create tables" in {
      Seq
        .empty[(String, Int, Double, Double, Double, Double, Double, Double)]
        .toDF(columnsMetrics: _*)
        .write
        .mode(SaveMode.Overwrite)
        .saveAsTable("metrics")

      Seq
        .empty[(String, Double, String)]
        .toDF("feature", "coefficient", "model")
        .write
        .mode(SaveMode.Overwrite)
        .saveAsTable("coefficients")
    }

    "test Logistic Regression" in {
      val savePath    = s"$modelsPath/spark-model-lr"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "regressão logística",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    = bestPipeline.stages.last
        .asInstanceOf[LogisticRegressionModel]
      println(s"Melhor modelo Logistic Regression:\n$bestModel")
      Seq(
        (
          bestModel.getRegParam,
          bestModel.getElasticNetParam,
          bestModel.getMaxIter
        )
      ).toDF("regParam", "elasticNetParam", "maxIter").show()

      val listOfCoefficients = bestModel.coefficients.toArray
      val intercept          = bestModel.intercept

      val columnsCoefficientsDF =
        (columnsFeatures ++ Seq("intercept") zip
          listOfCoefficients ++ Seq(intercept)).toList
          .toDF("feature", "coefficient")
          .withColumn("model", lit("logistic"))
      columnsCoefficientsDF
        .orderBy("coefficient")
        .show(listOfCoefficients.length, truncate = false)
      columnsCoefficientsDF.write
        .mode(SaveMode.Append)
        .saveAsTable("coefficients")
    }

    "test Random Forest" in {
      val savePath    = s"$modelsPath/spark-model-rf"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "Floresta Aleatória",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    = bestPipeline.stages.last
        .asInstanceOf[RandomForestClassificationModel]
      println(s"Melhor modelo Random Forest:\n$bestModel")
      Seq((bestModel.getNumTrees, bestModel.getMaxDepth))
        .toDF("numTrees", "maxDepth")
        .show()

    }

    "test gbt" in {
      val savePath    = s"$modelsPath/spark-model-gbt"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "gbt",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    = bestPipeline.stages.last
        .asInstanceOf[GBTClassificationModel]
      println(s"Melhor modelo GBT:\n$bestModel")
      Seq((bestModel.getMaxIter, bestModel.getMaxDepth))
        .toDF("maxIter", "maxDepth")
        .show()

    }

    "test MLP" in {
      val savePath    = s"$modelsPath/spark-model-mlp"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "mlp",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    = bestPipeline.stages.last
        .asInstanceOf[MultilayerPerceptronClassificationModel]
      println(s"Melhor modelo MLP:\n$bestModel")
      Seq(bestModel.getLayers).toDF("layers").show()

    }

    "test lsvc" in {
      val savePath    = s"$modelsPath/spark-model-lsvc"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "lsvc",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    = bestPipeline.stages.last.asInstanceOf[LinearSVCModel]
      println(s"Melhor modelo lsvc:\n$bestModel")
      Seq((bestModel.getMaxIter, bestModel.getRegParam))
        .toDF("maxIter", "regparam")
        .show()

      val listOfCoefficients = bestModel.coefficients.toArray
      val intercept          = bestModel.intercept

      val columnsCoefficientsDF =
        (columnsFeatures ++ Seq("intercept") zip
          listOfCoefficients ++ Seq(intercept)).toList
          .toDF("feature", "coefficient")
          .withColumn("model", lit("lsvc"))
      columnsCoefficientsDF
        .orderBy($"coefficient".desc)
        .show(listOfCoefficients.length, truncate = false)
      columnsCoefficientsDF.write
        .mode(SaveMode.Append)
        .saveAsTable("coefficients")

    }

    "test nb" in {
      val savePath    = s"$modelsPath/spark-model-nb"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "nb",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    =
        bestPipeline.stages.last.asInstanceOf[NaiveBayesModel]
      println(bestModel.theta.toDense)
      println(s"Melhor modelo nb:\n$bestModel")
    }

    "test fm" in {
      val savePath    = s"$modelsPath/spark-model-fm"
      val model       = CrossValidatorModel.load(savePath)
      val predictions = model.transform(testData)
      val metrics     = evaluator.getMetrics(predictions)
      val tempoTreino =
        times.where(s"model = '$savePath'").select("time").as[Long].head()
      val numTreino   = times
        .where(s"model = '$savePath'")
        .select("numTrains")
        .as[Long]
        .head()

      val confusionMatrix = metrics.confusionMatrix
      val tp              = confusionMatrix(0, 0)
      val tn              = confusionMatrix(1, 1)
      val fp              = confusionMatrix(1, 0)
      val fn              = confusionMatrix(0, 1)
      val accuracy        = (tp + tn) / (tp + fp + fn + tn)
      val precision       = tp / (tp + fp)
      val recall          = tp / (tp + fn)
      val f1              = (2 * precision * recall) / (precision + recall)

      val metricsDF = Seq(
        (
          "fm",
          tempoTreino,
          numTreino,
          1.0 * tempoTreino / numTreino,
          accuracy,
          precision,
          recall,
          f1
        )
      ).toDF(columnsMetrics: _*)
      metricsDF.show()
      metricsDF.write.mode(SaveMode.Append).saveAsTable("metrics")

      val bestPipeline = model.bestModel.asInstanceOf[PipelineModel]
      val bestModel    = bestPipeline.stages.last
        .asInstanceOf[FMClassificationModel]
      println(s"Melhor modelo fm:\n$bestModel")
      Seq((bestModel.getMaxIter, bestModel.getRegParam, bestModel.getStepSize))
        .toDF("maxIter", "regParam", "stepSize")
        .show()

      println(
        s"Factors:\n${bestModel.factors}\nLinear:\n${bestModel.linear}\nIntercept:\n${bestModel.intercept}"
      )
    }

    "view all metrics" in {
      spark.table("metrics").orderBy($"f1".desc).show()
      val coefficientsDF = spark
        .table("coefficients")
        .groupBy("feature")
        .pivot("model")
        .sum("coefficient")
        .orderBy($"logistic".desc)
      val interceptDf    = coefficientsDF
        .where(col("feature").equalTo("intercept"))
        .withColumn(
          "probability (based on logistic coefficients)",
          pow(scala.math.E, col("logistic"))
            / (pow(scala.math.E, col("logistic")) + 1)
            * 100
        )
      interceptDf.show()
      coefficientsDF
        .where(col("feature").notEqual("intercept"))
        .withColumn(
          "probability (based on logistic coefficients)",
          round(pow(scala.math.E, col("logistic")) * 100, 2)
        )
        .withColumn(
          "probability total",
          round(
            col("probability (based on logistic coefficients)") * interceptDf
              .select("probability (based on logistic coefficients)")
              .as[Double]
              .head() / 100,
            2
          )
        )
        .show()
    }
  }
}
