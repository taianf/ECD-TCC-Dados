package br.com.taian.ecd.tcc

import org.apache.spark.sql.SparkSession

import java.io.File

trait SparkSessionTrait {
  val warehouseLocation: String = new File("spark-warehouse").getAbsolutePath
  val spark: SparkSession       = SparkSession.builder
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.executor.memory", "12g")
    .config("spark.sql.shuffle.partitions", "24")
    .config("spark.default.parallelism", "24")
    .config("spark.scheduler.mode", "FAIR")
    // .config("spark.executor.resource.gpu.amount", "1")
    // .config("spark.hadoop.parquet.summary.metadata.level", "none")
    // .config("spark.locality.wait", "0s")
    // .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
    // .config("spark.rapids.memory.pinnedPool.size", "2G")
    // .config("spark.rapids.sql.concurrentGpuTasks", "1")
    // .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    // .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    // .config("spark.sql.files.maxPartitionBytes", "512m")
    // .config("spark.sql.parquet.mergeSchema", "false")
    .config("spark.sql.warehouse.dir", warehouseLocation)
    .enableHiveSupport()
    .appName("ECD_TCC")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

}
