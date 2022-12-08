ThisBuild / version               := "0.1.0"
ThisBuild / scalaVersion          := "2.12.16"
ThisBuild / assemblyMergeStrategy :=
  (_ => MergeStrategy.first)

val SparkVersion = "3.3.0"
lazy val root    = (project in file(".")).settings(
  name                 := "ECD-TCC-Dados",
  assembly / test      := {},
  assembly / mainClass := Option("br.com.taian.ecd.tcc.TCC"),
  Compile / mainClass  := Option("br.com.taian.ecd.tcc.TCC")
)

libraryDependencies ++=
  Seq(
    "ai.rapids"         % "cudf"            % "22.10.0",
    "com.nvidia"       %% "rapids-4-spark"  % "22.10.0",
    "org.apache.spark" %% "spark-core"      % SparkVersion,
    "org.apache.spark" %% "spark-mllib"     % SparkVersion,
    "org.apache.spark" %% "spark-hive"      % SparkVersion,
    "org.apache.spark" %% "spark-sql"       % SparkVersion,
    "org.scalatest"    %% "scalatest"       % "3.2.14" % "test",
    "ml.dmlc"          %% "xgboost4j"       % "1.7.1",
    "ml.dmlc"          %% "xgboost4j-spark" % "1.7.1"
  )

Test / scalacOptions ++= Seq("-Yrangepos")
assembly / test      := {}
assembly / mainClass := Option("br.com.taian.ecd.tcc.TCC")
Compile / mainClass  := Option("br.com.taian.ecd.tcc.TCC")
