name := "lsh-anomaly"

version := "0.1"

organization := "es.udc"

scalaVersion := "2.11.11"

val sparkVersion = "2.4.0"

val vegasVersion = "0.3.11"


resolvers ++= Seq(
  "apache-snapshots" at "http://repository.apache.org/snapshots/"
)

unmanagedJars in Compile += file("lib/spark-knine-0.2.jar")

libraryDependencies += "org.vegas-viz" %% "vegas" % {vegasVersion}

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}
