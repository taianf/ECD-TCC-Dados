import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col
from pyspark.sql.functions import when
from pyspark.sql.types import StringType

# Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "14g") \
    .config("spark.executor.memory", "14g") \
    .config("spark.hadoop.parquet.enable.summary-metadata", "false") \
    .config("spark.jars", "/opt/sparkRapidsPlugin/rapids-4-spark_2.12-22.06.0.jar") \
    .config("spark.locality.wait", "0s") \
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
    .config("spark.rapids.memory.pinnedPool.size", "2G") \
    .config("spark.rapids.sql.concurrentGpuTasks", "1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
    .config("spark.sql.files.maxPartitionBytes", "512m") \
    .config("spark.sql.parquet.mergeSchema", "false") \
    .appName("ECD_TCC") \
    .getOrCreate()

# Filtrando apenas casos de covid
df_all = spark.read.parquet("datasets/parquet").where("CLASSI_FIN == 5")

# Colunas do dataset
print(f"Colunas: {df_all.columns}")

# Selecionando apenas as colunas relevantes
select_cols = ["NOSOCOMIAL", "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO", "DIARREIA", "VOMITO",
               "PUERPERA", "FATOR_RISC", "CARDIOPATI", "HEMATOLOGI", "SIND_DOWN", "HEPATICA", "ASMA", "DIABETES",
               "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE", "VACINA_COV", "VACINA", "EVOLUCAO"]

# Preenchendo valores nulos como ignorados e normalizando os preenchimentos de fatores de risco
df = df_all \
    .select(select_cols) \
    .withColumn("FATOR_RISC",
                when(col("FATOR_RISC") == "S", lit("1"))
                .when(col("FATOR_RISC") == "N", lit("2"))
                .otherwise(col("FATOR_RISC"))
                ) \
    .where("VACINA_COV <> '12/02/2021'") \
    .fillna(value="9") \
    .cache()

print("Dataset com os dados relevantes")
df.show()

# Valores de camadas inciais e finais para o MLP
features_n = len(select_cols) - 1
classes_n = df.select("EVOLUCAO").distinct().count() + 1

# Preparando modelos de validação cruzada

# Separando as colunas em label/feature

feature_list = []
string_feature_list = []
string_feature_list_out = []
for name in df.columns:
    if name == "EVOLUCAO":
        string_feature_list.append(name)
        string_feature_list_out.append("label")
    else:
        if df.schema[name].dataType == StringType():
            string_feature_list.append(name)
            string_feature_list_out.append(name + "_vec")
            feature_list.append(name + "_vec")
        else:
            feature_list.append(name)

# Transoformar as colunas em variáveis categóricas
indexer = StringIndexer(inputCols=string_feature_list,
                        outputCols=string_feature_list_out,
                        handleInvalid="keep")
assembler = VectorAssembler(inputCols=feature_list,
                            outputCol="features",
                            handleInvalid="keep")

# Modelos a testar: Floresta aleatória e regressão logística
rf = RandomForestClassifier(maxBins=5000)
mlp = MultilayerPerceptronClassifier()

pipeline_rf = Pipeline(stages=[indexer, assembler, rf])
pipeline_mlp = Pipeline(stages=[indexer, assembler, mlp])

paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start=10, stop=50, num=3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start=5, stop=25, num=3)]) \
    .build()

paramGrid_mlp = ParamGridBuilder() \
    .addGrid(mlp.layers, [[features_n, 3, 3, classes_n],
                          [features_n, 4, 4, classes_n],
                          [features_n, 3, 3, 3, classes_n],
                          [features_n, 4, 4, 4, classes_n], ]) \
    .build()

crossval_rf = CrossValidator(estimator=pipeline_rf,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=MulticlassClassificationEvaluator())
crossval_mlp = CrossValidator(estimator=pipeline_mlp,
                              estimatorParamMaps=paramGrid_mlp,
                              evaluator=MulticlassClassificationEvaluator())

# Separar os dados em treino e teste
(trainingData, testData) = df.randomSplit([0.8, 0.2])

# Treinando o modelo
print("Treinando o modelo")
cvModel = crossval_rf.fit(trainingData)

# Salvando o modelo para testar depois sem precisar retreinar
print("Salvando o modelo para testar depois sem precisar retreinar")
cvModel.write.overwrite().save("model/spark-random-forest-model")

# Avaliando o modelo
predictions = cvModel.transform(testData)
evaluator = MulticlassClassificationEvaluator()
rmse = evaluator.evaluate(predictions)

predictions.select("label", "features", "rawPrediction", "probability", "prediction").show()
predictions.select("prediction").distinct().show()

result = predictions.toPandas()

plt.plot(result.label, result.prediction, 'bo')
plt.xlabel('Sobrevivencia')
plt.ylabel('Prediction')
plt.suptitle("Model Performance RMSE: %f" % rmse)
plt.show()

# Selecionando o melhor modelo
bestPipeline = cvModel.bestModel
bestModel = bestPipeline.stages[2]

importances = bestModel.featureImportances

x_values = list(range(len(importances)))

plt.bar(x_values, importances, orientation='vertical')
plt.xticks(x_values, feature_list, rotation=40)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

print('numTrees - ', bestModel.getNumTrees)
print('maxDepth - ', bestModel.getOrDefault('maxDepth'))

# cvModel.write.overwrite().save("model/spark-random-forest-model")
