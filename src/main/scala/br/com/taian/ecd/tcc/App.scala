package br.com.taian.ecd.tcc

import br.com.taian.ecd.tcc.session.SparkSessionTrait
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SaveMode.Overwrite
import org.apache.spark.sql.functions.{col, when}

object App extends SparkSessionTrait {
  // Ajuste o valor do tamanho dos dados de treino.
  def main(args: Array[String]): Unit = {
    generate(0.8)
  }

  def generate(trainingSplit: Double): Unit = {

    import spark.implicits._

    val datasetsPath = "work/datasets"
    //  val splitSeed    = 19900111L
    val selectCols   = Seq(
      "PUERPERA",
      "CARDIOPATI",
      "HEMATOLOGI",
      "SIND_DOWN",
      "HEPATICA",
      "ASMA",
      "DIABETES",
      "NEUROLOGIC",
      "PNEUMOPATI",
      "IMUNODEPRE",
      "RENAL",
      "OBESIDADE",
      "NU_IDADE_N",
      "CS_SEXO",
      // "VACINA_COV",
      // "VACINA",
      "EVOLUCAO"
    )
    val featureList  = selectCols.filter(_ != "EVOLUCAO")

    val rawDF = spark.read
      .option("header", "true")
      .option("sep", ";")
      .csv(s"$datasetsPath/raw")

    // Filtrando apenas casos de covid
    val filteredDf = rawDF
      .where("CLASSI_FIN == 5")
      .where("VACINA_COV <> '12/02/2021'")
      .where("EVOLUCAO = '1' or EVOLUCAO = '2'")
      .where("TP_IDADE = 3")
      .select(selectCols.map(col): _*)

    // Convertendo string para int
    val intDf = featureList
      .foldLeft(filteredDf) { case (fdf, column) =>
        if (column == "NU_IDADE_N")
          fdf.withColumn(column, col(column).cast("Integer"))
        else if (column == "CS_SEXO")
          fdf.withColumn(
            column,
            when(col(column) === "M", 1)
              .when(col(column) === "F", 0)
              .otherwise(null)
          )
        else fdf.withColumn(column, when(col(column) === "1", 1).otherwise(0))
      }
      .where(col("CS_SEXO").isNotNull)
      .withColumn("label", ($"EVOLUCAO".as[Int] - 1).cast("Integer"))

    // /** Já foi verificado ter menos óbitos que curados. Isso garante que vamos
    //   * ter praticamente a mesma quantidade de curados e óbitos para evitar
    //   * vieses no modelo gerado.
    //   */
    //  val dfObitos                 = intDf.where(col("EVOLUCAO").equalTo("2"))
    //  val dfCurados                = intDf.where(col("EVOLUCAO").equalTo("1"))
    //  val qtdObitos                = dfObitos.count().toDouble
    //  val qtdCurados               = dfCurados.count().toDouble
    //  val taxaCuradosTreino        = qtdObitos / qtdCurados
    //  val Array(parcialCurados, _) = dfCurados.randomSplit(
    //   Array(taxaCuradosTreino, 1 - taxaCuradosTreino)
    //   // seed = splitSeed
    // )
    //  val df                       = dfObitos.unionAll(parcialCurados)

    // valores de camadas inciais e finais para o MLP
    val featuresNumber = featureList.length
    val classesNumber  = intDf.select("label").distinct().count().toInt

    val assembler   = new VectorAssembler()
      .setInputCols(featureList.toArray)
      .setOutputCol("features")
    val assembledDF = assembler.transform(intDf)

    // Modelos a testar
    val lr   = new LogisticRegression().setFamily("binomial")
    val rf   = new RandomForestClassifier()
    val gbt  = new GBTClassifier()
    val mlp  = new MultilayerPerceptronClassifier()
    val lsvc = new LinearSVC()
    val nb   = new NaiveBayes()
    val fm   = new FMClassifier()

    val pipelineLr   = getPipeline(lr)
    val pipelineRf   = getPipeline(rf)
    val pipelineGbt  = getPipeline(gbt)
    val pipelineMlp  = getPipeline(mlp)
    val pipelineLsvc = getPipeline(lsvc)
    val pipelineNb   = getPipeline(nb)
    val pipelineFm   = getPipeline(fm)

    val paramGridLr   = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.02, 0.1))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
      .addGrid(lr.maxIter, Array(1, 2, 3))
      .build()
    val paramGridRf   = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10, 20, 30))
      .addGrid(rf.maxDepth, Array(5, 10, 15, 25, 30))
      .build()
    val paramGridGbt  = new ParamGridBuilder()
      .addGrid(gbt.maxIter, Array(10, 20, 30, 40, 50))
      .addGrid(gbt.maxDepth, Array(2, 6, 10, 15, 20))
      .build()
    val paramGridMlp  = new ParamGridBuilder()
      .addGrid(
        mlp.layers,
        Array[Array[Int]](
          Array[Int](featuresNumber, 5, classesNumber),
          Array[Int](featuresNumber, 10, classesNumber),
          Array[Int](featuresNumber, 20, classesNumber),
          Array[Int](featuresNumber, 5, 5, classesNumber),
          Array[Int](featuresNumber, 10, 10, classesNumber),
          Array[Int](featuresNumber, 20, 20, classesNumber),
          Array[Int](featuresNumber, 5, 5, 5, classesNumber),
          Array[Int](featuresNumber, 10, 10, 10, classesNumber),
          Array[Int](featuresNumber, 20, 20, 20, classesNumber)
        )
      )
      .addGrid(mlp.solver, Array("l-bfgs", "gd"))
      .build()
    val paramGridLsvc = new ParamGridBuilder()
      .addGrid(lsvc.maxIter, Array(1, 2, 3))
      .addGrid(lsvc.regParam, Array(0.01, 0.02, 0.1))
      .build()
    val paramGridNb   = new ParamGridBuilder().build()
    val paramGridFm   = new ParamGridBuilder()
      .addGrid(fm.maxIter, Array(1, 5, 10, 15, 20))
      .addGrid(fm.regParam, Array(0.01, 0.02, 0.1))
      .addGrid(fm.stepSize, Array(0.001, 0.005, 0.01, 0.05, 0.1))
      .build()

    val cvLr   = getCV(pipelineLr, paramGridLr)
    val cvRf   = getCV(pipelineRf, paramGridRf)
    val cvGbt  = getCV(pipelineGbt, paramGridGbt)
    val cvMlp  = getCV(pipelineMlp, paramGridMlp)
    val cvLsvc = getCV(pipelineLsvc, paramGridLsvc)
    val cvNb   = getCV(pipelineNb, paramGridNb)
    val cvFm   = getCV(pipelineFm, paramGridFm)

    // Separar os dados em treino e teste
    val Array(trainingData, testData) = assembledDF
      .randomSplit(
        Array(trainingSplit, 1 - trainingSplit)
        // seed = splitSeed
      )
      .map(_.repartition().cache())

    featureList.toDS().write.mode(Overwrite).json(s"$datasetsPath/features")
    trainingData.write.mode(Overwrite).parquet(s"$datasetsPath/training")
    testData.write.mode(Overwrite).parquet(s"$datasetsPath/test")

    lazy val f1 =
      model(cvLr, trainingData, s"$datasetsPath/model/spark-model-lr")
    lazy val f2 =
      model(cvRf, trainingData, s"$datasetsPath/model/spark-model-rf")
    lazy val f3 =
      model(cvGbt, trainingData, s"$datasetsPath/model/spark-model-gbt")
    lazy val f4 =
      model(cvMlp, trainingData, s"$datasetsPath/model/spark-model-mlp")
    lazy val f5 =
      model(cvLsvc, trainingData, s"$datasetsPath/model/spark-model-lsvc")
    lazy val f6 =
      model(cvNb, trainingData, s"$datasetsPath/model/spark-model-nb")
    lazy val f7 =
      model(cvFm, trainingData, s"$datasetsPath/model/spark-model-fm")

    Seq(
      f1,
      f2,
      f3,
      f4,
      f5,
      f6,
      f7
    )
      .toDF("model", "time", "numTrains")
      .write
      .mode(Overwrite)
      .json(s"$datasetsPath/times")

  }

  private def getCV(
      pipeline: Pipeline,
      paramGrid: Array[ParamMap]
  ): CrossValidator = new CrossValidator()
    .setEstimator(pipeline)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(new MulticlassClassificationEvaluator())
    .setNumFolds(5)
    .setParallelism(12)

  private def getPipeline(classifier: PipelineStage) = new Pipeline()
    .setStages(Array(classifier))

  private def model(
      crossval: CrossValidator,
      trainingData: DataFrame,
      savePath: String
  ): (String, Long, Int) = {
    val startTime = LocalDateTime.now()
    println(s"Start time of $savePath: $startTime")
    val cvModel   = crossval.fit(trainingData)
    val endTime   = LocalDateTime.now()
    cvModel.write.overwrite().save(savePath)
    println(s"End time of $savePath: $endTime")
    val duration  = ChronoUnit.MILLIS.between(startTime, endTime)
    println(s"Total time of $savePath: $duration s")
    (
      savePath,
      duration,
      crossval.getEstimatorParamMaps.length * crossval.getNumFolds
    )
  }

}
