package es.udc.lshanomalydetection

import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import Array._
import scala.util.Random
import scala.util.control.Breaks._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
//import es.udc.graph.utils.GraphUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.{Level, Logger}

import sys.process._
import org.apache.spark.sql.SparkSession
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import es.udc.graph.EuclideanLSHasher
import es.udc.graph.KNiNe
import es.udc.graph.sparkContextSingleton
import es.udc.graph.LSHKNNGraphBuilder
import es.udc.graph.EuclideanLSHasherForAnomaly

object LSHAnomalyDetector
{
  val DEFAULT_NUM_PARTITIONS:Double=512
  val DEFAULT_THRESHOLD:Int=1
  
  def showUsageAndExit()=
  {
    println("""teste Usage: LSHAnomalyDetector dataset [options]
    Dataset must be a libsvm file
Options:
    -r    Starting radius (default: """+LSHKNNGraphBuilder.DEFAULT_RADIUS_START+""")
    -p    Number of partitions for the data RDDs (default: """+DEFAULT_NUM_PARTITIONS+""")
    -f    threshold value in % (default: """+DEFAULT_THRESHOLD+""")

Advanced LSH options:
    -n    Number of hashes per item (default: auto)
    -l    Hash length (default: auto)""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("radius_start" -> LSHKNNGraphBuilder.DEFAULT_RADIUS_START,
                                                    "num_partitions" -> KNiNe.DEFAULT_NUM_PARTITIONS,
                                                    "threshold" -> DEFAULT_THRESHOLD.toDouble)
    if (p.length<=0)
      showUsageAndExit()
    
    m("dataset")=p(0)
    
    var i=1
    while (i < p.length)
    {
      if ((i>=p.length-1) || (p(i).charAt(0)!='-'))
      {
        println("Unknown option: "+p(i))
        showUsageAndExit()
      }
      val readOptionName=p(i).substring(1)
      val option=readOptionName match
        {
          case "r"   => "radius_start"
          case "n"   => "num_tables"
          case "l"   => "key_length"
          case "p"   => "num_partitions"
          case "f"   => "threshold"
          case somethingElse => readOptionName
        }
      if (!m.keySet.exists(_==option) && option==readOptionName)
      {
        println("Unknown option:"+readOptionName)
        showUsageAndExit()
      }
      m(option)=p(i+1).toDouble
      
      i=i+2
    }
    return m.toMap
  }
  def main(args: Array[String])
  {
    //println("JM-> args: "+args(0))
    if (args.length <= 0)
    {
      showUsageAndExit()
      return
    }
    
    val options=parseParams(args)
    
    val datasetFile=options("dataset").asInstanceOf[String]
    
    val threshold = options("threshold").asInstanceOf[Double].toInt
    val threshold_per = threshold/100f
    
    val numPartitions=options("num_partitions").asInstanceOf[Double].toInt
    val paramRadius = options("radius_start").asInstanceOf[Double]
    val keyLength=7//5
    val numTables=2//5000
    
    //Set up Spark Context
    val sc=sparkContextSingleton.getInstance()
    println(s"Default parallelism: ${sc.defaultParallelism}")
    
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    
    //Load data from file
    val dataRDD: RDD[(Long,LabeledPoint)] = MLUtils.loadLibSVMFile(sc, datasetFile)
                                              .zipWithIndex()
                                              .map(_.swap)
                                              .partitionBy(new HashPartitioner(numPartitions))

    
    val ANOMALY_VALUE = 1
    //TODO - Retain only 60% of anomalies - WHY?
    val trainingDataRDD = dataRDD.filter({case (id, point) => point.label!=ANOMALY_VALUE || (math.random<0.6)  })
    trainingDataRDD.cache()
    
    val numAnomalies=trainingDataRDD.map({case (id, point) => if (point.label==ANOMALY_VALUE) 1 else 0}).sum.toInt
    println(s"Number of elements: ${trainingDataRDD.count()} ($numAnomalies anomalies)")
    println("Tuning hasher...")
    //Get a new hasher
    //Autoconfig
    val (hasher,nComps,suggestedRadius)=EuclideanLSHasherForAnomaly.getHasherForDataset(dataRDD, 5000) //Make constant size buckets
    //Quick Parameters
    //val hasher = new EuclideanLSHasher(dataRDD.first()._2.features.size, keyLength, numTables)
    
    println(s"Arguments:\n\tDataset:$datasetFile\n\tKL:${hasher.keyLength}\n\tR0:$suggestedRadius\n\tNT:${hasher.numTables}\n\tThreshold:$threshold_per")
    
    println("Training...")
    val newHasher = new EuclideanLSHasher(dataRDD.first()._2.features.size, hasher.keyLength, 100*hasher.numTables)
    val hashNeighborsRDD = EuclideanLSHasherForAnomaly.getHashNeighbors(trainingDataRDD, newHasher, suggestedRadius) // ((a,b,c), (b,d), (c), (a,h,f,e) (a))
    if(hashNeighborsRDD!=null)
    {
      val numNeighborsPerPointRDD = hashNeighborsRDD.flatMap({case l => l.map({case x => (x, l.size-1)})})
                                                    .reduceByKey(_ + _)
                                                    //.sortBy(_._2) //No need to sort the entire RDD here
                                                    //.partitionBy(new HashPartitioner(numPartitions))
      
      val maxNeighbors = numNeighborsPerPointRDD.map({case (id,rec) => rec }).max()
      //DEBUG
      val minNeighbors = numNeighborsPerPointRDD.map({case (id,rec) => rec }).min()

      
      //Retrieve the N=numAnomalies elements with fewer neighbors. They will constitute the predicted anomalies. The largest number of neighbors is the threshold.
      val maxNeighborsForAnomaly = numNeighborsPerPointRDD.map(_._2).takeOrdered(numAnomalies).last
      val predictionsAndEstimatorsRDD = numNeighborsPerPointRDD.map({case (id,rec) => (id,(rec<=maxNeighborsForAnomaly,(1-(rec.toDouble/maxNeighbors))))})
      val nAnomaliesDetected=predictionsAndEstimatorsRDD.filter(_._2._1).count()
      println(s"Results:\n\t# of anomalies detected: $nAnomaliesDetected (${nAnomaliesDetected*100.0/numNeighborsPerPointRDD.count()}%)")
      println(s"DEBUG -- maxNeighbors:$maxNeighbors maxNeighborsForAnomaly:$maxNeighborsForAnomaly minNeighbors:$minNeighbors")

      val labelsRDD = trainingDataRDD.map({ case (id, point) => (id, point.label==ANOMALY_VALUE) })
      val checkRDD = predictionsAndEstimatorsRDD.join(labelsRDD)
                                            .map(
                                                {
                                                  case (id, ((pred,estimator), label)) => 
                                                   //(if (pred) 1.toDouble else 0.toDouble, estimator, if (label) 1.toDouble else 0.toDouble)
                                                    (pred,estimator,label)
                                                })
      
      //estimationsRDD.take(100).foreach(println)
      //val totalRows = numNeighborsPerPointRDD.count().toInt
      //println("DEBUG -- threshold_per "+totalRows*threshold_per)
      println("DEBUG -- labelsRDD count: "+labelsRDD.count())
      println("DEBUG -- predictionsRDD count "+predictionsAndEstimatorsRDD.count())
      println("DEBUG -- Estimators:")
      checkRDD.sample(false, 0.01, 2165149).sortBy(_._2).collect().foreach(println)
      val confMat =  checkRDD.map(
                                  {
                                    case (pred,estimator,label) => 
                                          var vp=if(label && pred) 1 else 0
                                          var vn=if(!label && !pred) 1 else 0
                                          var fp=if(!label && pred) 1 else 0
                                          var fn=if(label && !pred) 1 else 0 
                                          (vp,vn,fp,fn)
                                  })
                              .reduce(
                                  {
                                    case ((tp1,tn1,fp1,fn1), (tp2,tn2,fp2,fn2)) =>
                                          (tp1+tp2, tn1+tn2, fp1+fp2, fn1+fn2)
                                  })
      val tp =confMat._1.toFloat
      val tn =confMat._2.toFloat
      val fp =confMat._3.toFloat
      val fn =confMat._4.toFloat
      
      val accuracy = (tp+tn)/(tp+tn+fp+fn)
      val precision = (tp)/(tp+fp)
      val recall = (tp)/(tp+fn)
      val f1score = 2*((precision*recall)/(precision+recall))
      val metricsForEstimation = new BinaryClassificationMetrics(checkRDD.map({case (pred,estimator,label) => (estimator, if (label) 1.toDouble else 0.toDouble)}))
      val metricsForPrediction = new BinaryClassificationMetrics(checkRDD.map({case (pred,estimator,label) => (if (pred) 1.toDouble else 0.toDouble, if (label) 1.toDouble else 0.toDouble)}))
      // AUROC
      val auROCForEstimation = metricsForEstimation.areaUnderROC
      val auROCForPrediction = metricsForPrediction.areaUnderROC
      
      //print("pointsPerHash")
      //numNeighborsPerPointRDD.take(100).foreach(println)
      
      println("\tconfMatTuple: "+confMat)
      println("\taccuracy: "+accuracy)
      println("\tprecision: "+precision)
      println("\trecall: "+recall)
      println("\tf1score: "+f1score)
      println("\tArea under ROC (estimation)= " + auROCForEstimation)
      println("\tArea under ROC (prediction)= " + auROCForPrediction)
    }
    else
      println("No data")
      
    //Stop the Spark Context
    sc.stop()
  }
}