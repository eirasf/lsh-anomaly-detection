package es.udc.lshanomalydetection

import java.io.File
import java.io.PrintWriter

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

//JM
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
//
import math._

import es.udc.graph.sparkContextSingleton
import es.udc.graph.EuclideanLSHasher
import org.apache.spark.HashPartitioner
import es.udc.graph.EuclideanLSHasherForAnomaly
import scala.annotation.meta.field
//import vegas.DSL.Vegas
import scala.annotation.meta.field
//import vegas.spec.Spec.Bin
import scala.annotation.meta.field
import scala.annotation.meta.field
import scala.annotation.meta.field
import scala.annotation.meta.field
import scala.annotation.meta.field
import scala.annotation.meta.field
import scala.annotation.meta.field

import vegas.AggOps
import vegas.Bar
import vegas.Bin
import vegas.DefaultValueTransformer
import vegas.Quantitative
import vegas.Vegas
import vegas.Line
import scala.annotation.meta.field
import es.udc.graph.LookupProvider
import es.udc.graph.DistanceProvider
import es.udc.graph.LookupProvider
import es.udc.graph.BroadcastLookupProvider
import es.udc.graph.EuclideanDistanceProvider

object Wplot
{
  val DEFAULT_NUM_PARTITIONS:Int=512
  val DEFAULT_THRESHOLD:Int=1
  val ANOMALY_VALUE=1.0
  protected def log2(n: Double): Double =
  {
    Math.log10(n) / Math.log10(2)
  }
  
  
  def ComputeDistance(points: Array[Long], lookup: LookupProvider, distance: DistanceProvider):Double ={
      val pointReference = lookup.lookup(points(0))
      val pointsDist = points.map({case id => 
                          val point = lookup.lookup(id)
                          distance.getDistance(pointReference, point)
                          })
     val avDistance = pointsDist.sum/pointsDist.size
     avDistance
  }
  
  def main(args: Array[String])
  {

    //Set up Spark Context
    val sc=sparkContextSingleton.getInstance()
    println(s"Default parallelism: ${sc.defaultParallelism}")
    
    val spark2 = SparkSession.builder//.appName("KNiNe")
                                    //.master("local[1]")
                                    //.config("spark.driver.maxResultSize", "2048MB")
                                    .getOrCreate()
    import spark2.implicits._
    
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    
    //val DATASETS_ROOT="file:///mnt/NTFS/owncloud/Datasets/datasets-anomalias/"
    val DATASETS_ROOT="file:///Users/jorgemeira/OneDrive - Instituto Superior de Engenharia do Porto/Doutoramento/datasets/datasets-anomalias/"
    val NUM_FOLDS=5
    
    //val pw = new PrintWriter(new File("/home/eirasf/Escritorio/reachability/grid-summary-fast-full.txt"))
    //Array("abalone1-8","abalone9-11","abalone11-29", "arritmia-fix-ohe","german_statlog-ohe", "covtype2vs4",
    // "kddcup10-http-normal-vs-all","kddcup10-normal-vs-all","kddcup10-smtp-normal-vs-all", "2_banana_clusters", "2_cirucular_clusters",
    // "2_point_clouds_with_variance", "3_anisotropic_clusters", "3_point_clouds", "scikit_1_cluster", "scikit_1_cluster_with_variance", "scikit_2_bananas_shape", "scikit_2_clusters" ))
    for (datasetName <- Array("abalone1-8","arritmia-fix","german_statlog","one_hot_covtype2vs4_sample","kddcup10-smtp-normal-vs-all"))
    {
      val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, DATASETS_ROOT+datasetName+".libsvm")
      
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRDD.map(x => x.features))
      val standardDataRDD=dataRDD.map({case p => new LabeledPoint(p.label,scaler.transform(p.features))})
      
      
          
      val trainingData=standardDataRDD.zipWithIndex()
                        .map(_.swap)
                        .partitionBy(new HashPartitioner(DEFAULT_NUM_PARTITIONS))

      val totalDataElem = trainingData.count()
      val numTables = 10
      val lowRangeExp = 0
      val highRangeExp = 2
      val numSteps = 100 //100
      //val initialKLength: Int = Math.ceil(log2(trainingData.count() / trainingData.first._2.features.size)).toInt + 1
      //val minKeyLength: Int = Math.ceil(initialKLength/2).toInt+1
     // val maxKeyLength: Int = Math.ceil(initialKLength*2).toInt+1
      var avBucketSizeVEachKeyLSeq: Seq[(Double,Double)] = Seq()
      var avBucketSizeVsWSeq: Seq[(Double,Double)] = Seq()
      var avBucketDistanceVsKeyL: Seq[(Double,Double)] = Seq()
      var avBucketDistanceVsRadious: Seq[(Double,Double)] = Seq()
      val histogramPath= "/Users/jorgemeira/OneDrive - Instituto Superior de Engenharia do Porto/Doutoramento/Projetos/LSH/histograms/"
      val lookup = new BroadcastLookupProvider(trainingData)
      val distance = new EuclideanDistanceProvider()
//      for (mf <- Array(10,50,100,500))
       for (w <-Array(2.0))
       {
         for (keyL <- Array(4))
         {
            for (radious <- Array(0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0))
              //for (bs <- Array(42)) // experimentar aplicar raiz quadrada de total numero de instancias ou ln ou 1%??
              {
                  //val w = math.pow(10, lowRangeExp+step*(highRangeExp-lowRangeExp)/100.0)
                  println(s">>>> $datasetName - KeyLength:$radious - W:$w ")
                  try
                  {
                    
                    val newHasher = new EuclideanLSHasher(trainingData.first()._2.features.size, keyL, numTables, w)
                    val hashedDataRDD = EuclideanLSHasherForAnomaly.hashData(trainingData, newHasher, radious)
                    val bucketDistanceRDD = hashedDataRDD.groupByKey().filter({case (h, ids) => ids.size>1 }).map({case (h, ids) => ComputeDistance(ids.toArray, lookup, distance) })
                    val avBucketDistnce = bucketDistanceRDD.sum/bucketDistanceRDD.count().toDouble
                    //val bucketSizeRDD = hashedDataRDD.map({case (h, id) => (h,1) }).reduceByKey(_+_) //buckets Size
                    //val avBucketSize = numTables/bucketSizeRDD.count().toDouble
                   // avBucketSizeVEachKeyLSeq ++= Seq((w,avBucketSize))
                    //avBucketSizeVsWSeq ++= Seq((w,avBucketSize))
                    //avBucketDistanceVsKeyL ++= Seq((keyL.toDouble,avBucketDistnce))
                    avBucketDistanceVsRadious ++= Seq((radious.toDouble,avBucketDistnce))
                    
                    
                    
                    println("KeyLength: "+keyL)
                    println("w: "+w)
                    println("avBucketDistnce: "+avBucketDistnce)
                    //println("dist: "+ComputeDistance(distArray))
                    //bucketDistanceRDD.take(10).foreach(println)
                    
                    
                    //println("Bucket Size RDD: ")
                    //bucketSizeRDD.take(10).foreach(println)
                    //dataRDD.first(): (0.0,(10,[0,1,2,3,4,5,6,7,8,9],[1.0,0.0,0.0,0.455,0.365,0.095,0.514,0.2245,0.101,0.15]))
  //                  if(step%10==0)
  //                  {
                        
                        val plotGeneral=Vegas("A simple bar chart with embedded data.",width=800, height=600).
                          withData(bucketDistanceRDD.map({case l => Map("distance" -> l)}).collect().toSeq).
                          encodeX("distance", Quantitative, bin=Bin(maxbins=100.0)).
                          encodeY(field="*", Quantitative, aggregate=AggOps.Count).
                          mark(Bar)
                         
                          
                          
                        val pw = new PrintWriter(new File(histogramPath+"Plot_"+datasetName+"KeyL_"+keyL+"w_"+w+"radious_"+radious+".html"))
                          pw.write(plotGeneral.html.headerHTML(""))
                          pw.write(plotGeneral.html.plotHTML("general"))
                          pw.write(plotGeneral.html.footerHTML)
                          pw.close
      
  //                  }
                    
                  }
                  catch
                  {
                    case e : Exception =>
                      println("ERROR")
                  }
                  
      //          pw.write(s"$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS} - Time:${totalTime/NUM_FOLDS}\n")
      //          pw.flush()
              }
           }
            
            val plotKeyL_AvgBucketVsW=Vegas("A simple bar chart with embedded data.",width=800, height=600).
            withData(avBucketDistanceVsRadious.map({case l => Map("Radious" -> l._1, "distance" -> l._2)}).toSeq).
            encodeX("Radious", Quantitative).
            encodeY("distance", Quantitative).
            mark(Line)
            
            val pw = new PrintWriter(new File(histogramPath+s"Plot_"+datasetName+"_radious_vs_avgBucketDistance.html"))
            pw.write(plotKeyL_AvgBucketVsW.html.headerHTML(""))
            pw.write(plotKeyL_AvgBucketVsW.html.plotHTML("general"))
            pw.write(plotKeyL_AvgBucketVsW.html.footerHTML)
            pw.close
            
            avBucketDistanceVsRadious = Seq()
            
            println(s"--------------------------------W: --\n$w ------END----")
          
         }
      
        
      
        println(s"----------------------------------\n$datasetName ------END----")
    }
    //pw.close

    //Stop the Spark Context
    sc.stop()
  }
}