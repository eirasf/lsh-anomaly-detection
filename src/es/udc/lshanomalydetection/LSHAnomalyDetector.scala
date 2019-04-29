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
    
                 
    //println("Using "+method+" to compute "+numNeighbors+"NN graph for dataset "+justFileName)
    //println("R0:"+radius0+(if (numTables!=null)" num_tables:"+numTables else "")+(if (keyLength!=null)" keyLength:"+keyLength else "")+(if (maxComparisons!=null)" maxComparisons:"+maxComparisons else ""))
    
    //Set up Spark Context
    val sc=sparkContextSingleton.getInstance()
    println(s"Default parallelism: ${sc.defaultParallelism}")
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    
    //Load data from file
    val dataRDD: RDD[(Long,LabeledPoint)] = MLUtils.loadLibSVMFile(sc, datasetFile).zipWithIndex().map(_.swap)
                                              .partitionBy(new HashPartitioner(numPartitions))

    
    var iAnomalyVal = 1
    var data_count = dataRDD.count()
    val data = dataRDD.filter({case (point, isAnomaly) => isAnomaly.label!=iAnomalyVal || (math.random<0.6)  })
    
    println("dataRDD: "+dataRDD.first())
    data.cache()
    var anomaliesRDD=data.filter({case (point, isAnomaly) => isAnomaly.label==iAnomalyVal})
    println("anomaliesRDD Count: "+anomaliesRDD.count( ) )
    
    println("data Count: "+data.count() )
    
    
                                          
//    val (hasher,nComps,suggestedRadius)=EuclideanLSHasher.getHasherForDataset(data, 200) //Make constant size buckets

    ///////////////////////// USE Quick Parameterss
////    val nComps = 300
    val suggestedRadius = options("radius_start").asInstanceOf[Double]
    val keyLength=5
    val numTables=25000
    val hasher = new EuclideanLSHasher(dataRDD.first()._2.features.size, keyLength, numTables)
    ///////////////////////////////////////////
    
    //val primElem = data.first()
    //val dataAll = data.take(5)
    //println("data first: "+data.first())
    //println("primElem._2: "+primElem._2)
   // println("primElem._1: "+primElem._1)
//    println(hasher.getHashes(primElem._2.features, primElem._1, suggestedRadius))
    //hasher.getHashes(primElem._2.features, primElem._1, suggestedRadius)
    //println("Size: "+hasherPoints.size)
     //println(" hasher.dim: "+hasher.dim+" hasherkLength: " + hasher.keyLength +" hasher.ntables: "+hasher.numTables +", ncomps: "+nComps+ ",sugestedRadius: "+suggestedRadius) 
    
    val hashNeighbors = EuclideanLSHasherForAnomaly.getHashNeighbors(data, hasher, suggestedRadius) // ((a,b,c), (b,d), (c), (a,h,f,e) (a))
    //println("hashNeighbors: "+hashNeighbors.collect)
    
    if(hashNeighbors!=null)
    {
      var pointsPerHash = hashNeighbors.flatMap({case l => l.map({case x => (x, l.size-1)})}).reduceByKey(_ + _).sortBy(_._2)
      //.partitionBy(new HashPartitioner(numPartitions))
      
      var maxNeighbors = pointsPerHash.map({case (id,rec) => (rec,id) }).max()._1.toFloat
      var totalRows = pointsPerHash.count().toInt
      //var numElementsAnomalies = math.ceil(totalRows*threshold_per).toInt // Use the threshold parameter
      var numElementsAnomalies = anomaliesRDD.count().toInt
      
      //var numElemNormal = (totalRows-numElementsAnomalies).toInt
      //var rddAnomalies = pointsPerHash.map({case (x,y) => (x,1.0)})
      //var rddAnomalies = sc.parallelize(pointsPerHash.take(numElementsAnomalies),numPartitions)//map({case (x,y) => (x,1.0)})
      //var rddNormal = pointsPerHash.take(numElemNormal).map({case (x,y) => (x,0.0)})
      //var rddFinal = pointsPerHash.leftOuterJoin(rddAnomalies).map({case (id,(rec,anomaly)) => (id,anomaly.isDefined) })
      
      val mapAnomalies = pointsPerHash.take(numElementsAnomalies).toMap
      val bMapAnomalies = sc.broadcast(mapAnomalies)
      val rddFinal = pointsPerHash.map({case (id,rec) => (id,bMapAnomalies.value.contains(id)) })
      println("anomalias detetadas: "+rddFinal.map({case (id, isanomaly) => if(isanomaly) 1 else 0 }).sum())
      //var rddFinal = pointsPerHash.leftOuterJoin(rddAnomalies).map({case (id,(rec,anomaly)) => (id,anomaly.isDefined) })
      

      var pointsPerHashPerc = pointsPerHash.map({case (id,rec) => (id, (1-(rec/maxNeighbors)).toDouble) })

      var label = data.map({ case (h, n) => (h, n.label==iAnomalyVal) })
      var rddPredLabel2 = rddFinal.join(label).map({case (id, (pred, label)) => 
                                               (if(pred) 1.toDouble else 0.toDouble, if(label) 1.toDouble else 0.toDouble) })
      var rddPredLabel = pointsPerHashPerc.join(label).map({case (id, (pred, label)) => (pred, if(label) 1.toDouble else 0.toDouble) })
      pointsPerHashPerc.take(100).foreach(println)                                         
      println("threshold_per "+totalRows*threshold_per)
      println("label count "+label.count())
      println("rddFinal count "+rddFinal.count())
      var confMat =  rddFinal.join(label).map({case (id,(pred,label)) => 
                                              var vp=if(label && pred) 1 else 0
                                              var vn=if(!label && !pred) 1 else 0
                                              var fp=if(!label && pred) 1 else 0
                                              var fn=if(label && !pred) 1 else 0 
                                              (vp,vn,fp,fn) })
                                              .reduce({case ((tp1,tn1,fp1,fn1), (tp2,tn2,fp2,fn2)) => (tp1+tp2, tn1+tn2, fp1+fp2, fn1+fn2)   })
      var tp =confMat._1.toFloat
      var tn =confMat._2.toFloat
      var fp =confMat._3.toFloat
      var fn =confMat._4.toFloat
      
      var accuracy = (tp+tn)/(tp+tn+fp+fn)
      var precision = (tp)/(tp+fp)
      var recall = (tp)/(tp+fn)
      var f1score = 2*((precision*recall)/(precision+recall))
      var metrics = new BinaryClassificationMetrics(rddPredLabel)// Donde DATA_RDD es de (Estimator:Double, Double)
      var metrics2 = new BinaryClassificationMetrics(rddPredLabel2)// Donde DATA_RDD es de (Estimator:Double, Double)
      // AUROC
      var auROC = metrics.areaUnderROC
      var auROC2 = metrics2.areaUnderROC
      
      print("pointsPerHash")
      pointsPerHash.take(100).foreach(println)
//      println("label")
//      label.take(500).foreach(println)
      
      println("confMatTuple: "+confMat)
      println("accuracy: "+accuracy)
      println("precision: "+precision)
      println("recall: "+recall)
      println("f1score: "+f1score)
      println("Area under ROC (training)= " + auROC)
      println("AUC 2= " + auROC2)
      //print("Num Anomalias: "+label.filter({ case (n1, x) => x == true }).count())
      

//      println("rddFinal")
//      rddFinal.take(500).foreach(println)
//      println("pointsPerHash")
//      pointsPerHash.take(500).foreach(println)
      //print("bMapAnomalies: "+rddAnomalies.id)
      
      

      

      
    }
    else
    {
      println("No data")
    }
    //Stop the Spark Context
    sc.stop()
  }
}