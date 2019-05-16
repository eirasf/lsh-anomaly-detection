package es.udc.lshanomalydetection

import java.io.File
import java.io.PrintWriter

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.HashPartitioner
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.{ Vectors => MLVectors }
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
//import es.udc.graph.utils.GraphUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import es.udc.graph.EuclideanLSHasher
import es.udc.graph.EuclideanLSHasherForAnomaly
import es.udc.graph.Hash
import es.udc.graph.Hasher
import es.udc.graph.KNiNe
import es.udc.graph.LSHKNNGraphBuilder
import es.udc.graph.sparkContextSingleton
import vegas.AggOps
import vegas.Bar
import vegas.Bin
import vegas.DefaultValueTransformer
import vegas.Quantitative
import vegas.Vegas
import org.apache.spark.mllib.feature.StandardScaler

trait LSHReachabilityAnomalyDetectorParams extends Params
{
  final val minBucketSize=new Param[Int](this, "desiredSize", "Minimum size of the buckets generated. Only used if no keyLength, numTables and radius are provided.")
  final val numTablesMultiplier=new Param[Int](this, "numTablesMultiplier", "Multiplying factor for the autogenerated numTables")
  final val anomalyValue=new Param[Double](this, "anomalyValue", "Label used to identify an anomaly in the training dataset")
  final val numPartitions=new Param[Int](this, "numPartitions", "Number of partitions to use")
  final val keyLength=new Param[Option[Int]](this, "keyLength", "Length of the hashes generated. Only used if numTables and radius are provided.")
  final val numTables=new Param[Option[Int]](this, "numTables", "Number of hashes per key. Only used if keyLength and radius are provided.")
  final val radius=new Param[Option[Double]](this, "radius", "Similarity radius for the euclidean hashing procedure. Only used if keyLength and numTables are provided.")
  //DEBUG
  final val histogramFilePath= new Param[Option[String]](this, "histogramFilePath", "DEBUG - Path to save the histograms")
}

class LSHReachabilityAnomalyDetectorModel(private val hashCounts:scala.collection.Map[Hash,Double], private val hasher:Hasher, private val threshold:Int, private val radius:Double)
  extends PredictionModel[Vector, LSHReachabilityAnomalyDetectorModel]
  with Serializable
{
  def predict(features:Vector):Double=
  {
    val hashes=hasher.getHashes(Vectors.dense(features.toArray), -1, radius)
    if (hashes.map({case (h,id) => hashCounts.getOrElse(h, 0.0)}).sum<=threshold)
      return 1.0
    return 0.0
  }
  
  def getEstimator(features:Vector):Double=
  {
    val hashes=hasher.getHashes(Vectors.dense(features.toArray), -1, radius)
    return -hashes.map({case (h,id) => hashCounts.getOrElse(h, 0.0)}).sum
  }
  val uid: String = Identifiable.randomUID("LSHReachabilityAnomalyDetectorModel")
  override def copy(extra:ParamMap): LSHReachabilityAnomalyDetectorModel = defaultCopy(extra)
}

class LSHReachabilityAnomalyDetector(override val uid: String)
  extends LSHReachabilityAnomalyDetectorParams
  with Logging
{
  def this() = this(Identifiable.randomUID("LSHReachabilityAnomalyDetector"))
  override def copy(extra:ParamMap): LSHReachabilityAnomalyDetector = defaultCopy(extra)
  
  def setMinBucketSize(v:Int):this.type=set(minBucketSize, v)
  setDefault(minBucketSize, 50)
  def setAnomalyValue(v:Double):this.type=set(anomalyValue, v)
  setDefault(anomalyValue, 1.0)
  def setNumPartitions(v:Int):this.type=set(numPartitions, v)
  setDefault(numPartitions, 512)
  def setHistogramFilePath(v:Option[String]):this.type=set(histogramFilePath, v)
  setDefault(histogramFilePath, None)
  def setNumTablesMultiplier(v:Int):this.type=set(numTablesMultiplier, v)
  setDefault(numTablesMultiplier, 1)
  def setManualParams(kl:Int, nt:Int, r:Double):this.type=
  {
    set(keyLength, Some(kl))
    set(numTables, Some(nt))
    set(radius, Some(r))
  }
  setDefault(keyLength, None)
  setDefault(numTables, None)
  setDefault(radius, None)
  
  def fit(rddData: RDD[LabeledPoint]): LSHReachabilityAnomalyDetectorModel=
  {
    
    val dataRDD=rddData.zipWithIndex()
                        .map(_.swap)
                        .partitionBy(new HashPartitioner($(numPartitions)))
    
    val desiredSize=$(minBucketSize)
    val ANOMALY_VALUE = $(anomalyValue)
    
    //Make sure the number of anomalies is around 1%
    val dataNumElems=dataRDD.count()
    val dataNumAnomalies=dataRDD.filter({case (id, point) => point.label==ANOMALY_VALUE}).count()
    
    val (trainingDataRDD,numElems,numAnomalies)=
      if (dataNumAnomalies.toDouble/dataNumElems<=0.02)
        (dataRDD,dataNumElems,dataNumAnomalies)
      else
      {
        val tr=dataRDD.filter({case (id, point) => point.label!=ANOMALY_VALUE || (math.random<0.01*dataNumElems.toDouble/dataNumAnomalies)})
        (tr,tr.count(),tr.filter({case (id, point) => point.label==ANOMALY_VALUE}).count())
      }
    
    logDebug(s"Tuning hasher with desiredSize=$desiredSize...")
    //Get a new hasher
    val (hasher,nComps,suggestedRadius)=
      if ($(keyLength).isEmpty || $(numTables).isEmpty || $(radius).isEmpty)
        EuclideanLSHasherForAnomaly.getHasherForDataset(trainingDataRDD, desiredSize)//Autoconfig
      else
        (new EuclideanLSHasher(dataRDD.first()._2.features.size, $(keyLength).get, $(numTables).get),desiredSize,$(radius).get) 
    
    val message=s"Training params:\n\tKL:${hasher.keyLength}\n\tR0:$suggestedRadius\n\tNT:${hasher.numTables}"
    logDebug(message)
    println(message)
    
    val newHasher = new EuclideanLSHasher(trainingDataRDD.first()._2.features.size, hasher.keyLength, $(numTablesMultiplier)*hasher.numTables)
    
    //val hashNeighborsRDD = EuclideanLSHasherForAnomaly.getHashNeighbors(trainingDataRDD, newHasher, suggestedRadius) // ((a,b,c), (b,d), (c), (a,h,f,e) (a))
    val hashedDataRDDPrevious=EuclideanLSHasherForAnomaly.hashData(trainingDataRDD, newHasher, suggestedRadius)
    println(s"Generated ${hashedDataRDDPrevious.count()} hashes for ${trainingDataRDD.count()} elements.")
    val hashedDataRDD=hashedDataRDDPrevious.groupByKey()
    val reachabilityRDD=hashedDataRDD.flatMap({case (hash,l) => l.map({case p => (p, (l.size, 1, List(hash)))})})
                                     .reduceByKey({case ((r1,c1,hl1),(r2,c2,hl2)) => (r1+r2,c1+c2,hl1++hl2)})
                                     .flatMap({case (p,(r,c,hl)) => hl.map({case h => (h,r.toDouble/c)})})
                                     .reduceByKey(_+_)
    //hashedDataRDD.cache()
    //val hashNeighborsRDD=hashedDataRDD.map(_._2)
    
    /*
    if ($(histogramFilePath).isDefined)
    {
      //DEBUG PLOTTING
        val histogramPath=$(histogramFilePath).get
        val plotGeneral=Vegas("A simple bar chart with embedded data.").
          withData(hashNeighborsRDD.map({case l => Map("a" -> l.size)}).collect().toSeq).
          encodeX("a", Quantitative, bin=Bin(maxbins=20.0)).
          encodeY(field="*", Quantitative, aggregate=AggOps.Count).
          mark(Bar)
          
        val normalID=trainingDataRDD.filter(_._2.label!=ANOMALY_VALUE).first()._1
        val plotNormal=Vegas("A simple bar chart with embedded data.").
          withData(hashNeighborsRDD.filter(_.toSet.contains(normalID)).map({case l => Map("a" -> l.size)}).collect().toSeq).
          encodeX("a", Quantitative, bin=Bin(maxbins=20.0)).
          encodeY(field="*", Quantitative, aggregate=AggOps.Count).
          mark(Bar)
          
        val anomalyID=trainingDataRDD.filter(_._2.label==ANOMALY_VALUE).first()._1
        val plotAnomaly=Vegas("A simple bar chart with embedded data.").
          withData(hashNeighborsRDD.filter(_.toSet.contains(anomalyID)).map({case l => Map("a" -> l.size)}).collect().toSeq).
          encodeX("a", Quantitative, bin=Bin(maxbins=20.0)).
          encodeY(field="*", Quantitative, aggregate=AggOps.Count).
          mark(Bar)
        
        val pw = new PrintWriter(new File(histogramPath))
        pw.write(plotGeneral.html.headerHTML(""))
        pw.write(plotGeneral.html.plotHTML("general"))
        pw.write(plotNormal.html.plotHTML("normal"))
        pw.write(plotAnomaly.html.plotHTML("anomaly"))
        pw.write(plotGeneral.html.footerHTML)
        pw.close
      //END OF DEBUG PLOTTING
    }
    */
    /*val numNeighborsPerPointRDD = hashNeighborsRDD.flatMap({case l => l.map({case x =>(x, l.size-1)})})
                                                  .reduceByKey(_ + _)
                                                  //.sortBy(_._2) //No need to sort the entire RDD here
                                                  //.partitionBy(new HashPartitioner(numPartitions))
    
    //Retrieve the N=numAnomalies elements with fewer neighbors. They will constitute the predicted anomalies. The largest number of neighbors is the threshold.
    val maxNeighborsForAnomaly = numNeighborsPerPointRDD.map(_._2).takeOrdered(numAnomalies.toInt).last
    */
    new LSHReachabilityAnomalyDetectorModel(reachabilityRDD.collectAsMap(),
                                newHasher,
                                0,
                                suggestedRadius)
  }
}

object LSHReachabilityAnomalyDetector
{
  val DEFAULT_NUM_PARTITIONS:Double=512
  val ANOMALY_VALUE=1.0
  
  def showUsageAndExit()=
  {
    println("""Usage: LSHReachabilityAnomalyDetector dataset [options]
    Dataset must be a libsvm file
Options:
    -p    Number of partitions for the data RDDs (default: """+DEFAULT_NUM_PARTITIONS+""")

Advanced LSH options:
    -r    Starting radius (default: """+LSHKNNGraphBuilder.DEFAULT_RADIUS_START+""")
    -n    Number of hashes per item (default: auto)
    -l    Hash length (default: auto)""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("num_partitions" -> DEFAULT_NUM_PARTITIONS)
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
  
  def evaluateModel(model:LSHReachabilityAnomalyDetectorModel, testDataRDD:RDD[LabeledPoint]):Double=
  {
    val bModel=testDataRDD.sparkContext.broadcast(model)
    val checkRDD = testDataRDD.map(
                                    {
                                      case lp =>
                                        val m=bModel.value
                                        val f=MLVectors.dense(lp.features.toArray)
                                        (m.predict(f)>0,m.getEstimator(f),lp.label==ANOMALY_VALUE)
                                    })
                                    
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
      
      return auROCForEstimation
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
    
    val numPartitions=options("num_partitions").asInstanceOf[Double].toInt
    val paramRadius:Option[Double]=if (options.contains("radius_start")) Some(options("radius_start").asInstanceOf[Double]) else None
    val keyLength:Option[Int]=if (options.contains("key_length")) Some(options("key_length").asInstanceOf[Double].toInt) else None
    val numTables:Option[Int]=if (options.contains("num_tables")) Some(options("num_tables").asInstanceOf[Double].toInt) else None
    //Set up Spark Context
    val sc=sparkContextSingleton.getInstance()
    println(s"Default parallelism: ${sc.defaultParallelism}")
    
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    
    val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, datasetFile)
      
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRDD.map(x => x.features))
    val standardDataRDD=dataRDD.map({case p => new LabeledPoint(p.label,scaler.transform(p.features))})
    
    val splits=standardDataRDD.randomSplit(Array(0.8,0.2), System.nanoTime())
    
    val trainDataRDD=splits(0)
    val testDataRDD=splits(1)
    
    
    val model=
      if (paramRadius.isDefined && keyLength.isDefined && numTables.isDefined)
      {
        new LSHReachabilityAnomalyDetector()
                        .setNumPartitions(options("num_partitions").asInstanceOf[Double].toInt)
                        //MANUAL
                        .setManualParams(keyLength.get, numTables.get, paramRadius.get)
                        //.setHistogramFilePath(Some(s"/home/eirasf/Escritorio/test-5-5.html"))
                        .fit(trainDataRDD)
      }
      else
      {
        new LSHReachabilityAnomalyDetector()
                        .setNumPartitions(options("num_partitions").asInstanceOf[Double].toInt)
                        //AUTO TUNING
                        .setMinBucketSize(20)
                        .setNumTablesMultiplier(5)
                        //.setHistogramFilePath(Some(s"/home/eirasf/Escritorio/test-5-5.html"))
                        .fit(trainDataRDD)
      }
              
    val totalAUC=evaluateModel(model,testDataRDD)
    
    //Stop the Spark Context
    sc.stop()
  }
}