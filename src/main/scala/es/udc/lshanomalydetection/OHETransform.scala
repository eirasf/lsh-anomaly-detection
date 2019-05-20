package es.udc.lshanomalydetection

import java.io.File
import java.io.PrintWriter

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector => BDV}

import es.udc.graph.sparkContextSingleton
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors

object OHETransform
{
  val DEFAULT_NUM_PARTITIONS:Int=512
  
  def OneHotEncode(d:Array[Double], possibleValues:Array[Array[Int]]):Array[Double]=
  {
    val binarizedValues=Array.fill[Double](possibleValues.map(_.size).sum)(0)
    var i=0
    var current=0
    while (i<d.length)
    {
      val codingValue=possibleValues(i).indexOf(d(i).toInt)
      binarizedValues(current+codingValue)=1
      current=current+(possibleValues(i).size)
      i=i+1
    }
    return binarizedValues
  }
  
  def OHERDD(rdd:RDD[LabeledPoint], firstNumerical:Int):RDD[LabeledPoint]=
  {
    val possibleValues=rdd.flatMap({ case point => point.features.toArray.slice(0, firstNumerical)
                                                    .zipWithIndex
                                                    .map({case (c,i) => (i,List(c.toInt))})})
                          .reduceByKey({case (a,b) => (a++b).toSet.toList})
                          .map({case (i,l) => (i,l.sorted.toArray)})
                          .collect()
                          .sortBy(_._1)
                          .map(_._2)
    val bPossibleValues=rdd.sparkContext.broadcast(possibleValues)
    possibleValues.foreach({case l => println(l.mkString(","))})                      
    return rdd.map({case point =>
                      val pV=bPossibleValues.value
                      val categorical=point.features.toArray.slice(0, firstNumerical)
                      val numerical=point.features.toArray.slice(firstNumerical,point.features.size)
                      val oheCategorical=OneHotEncode(categorical,pV)
                      LabeledPoint(point.label, Vectors.dense(oheCategorical++numerical).toSparse)
                      })
  }
  
  def main(args: Array[String])
  {
    //Set up Spark Context
    val sc=sparkContextSingleton.getInstance()
    println(s"Default parallelism: ${sc.defaultParallelism}")
    
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    
    
    val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/Users/eirasf/Downloads/Datasets/german_statlog.libsvm", DEFAULT_NUM_PARTITIONS)
    MLUtils.saveAsLibSVMFile(OHERDD(dataRDD, 13).coalesce(1), "/Users/eirasf/Downloads/Datasets/german_statlog-ohe.libsvm")

    //Stop the Spark Context
    sc.stop()
  }
}