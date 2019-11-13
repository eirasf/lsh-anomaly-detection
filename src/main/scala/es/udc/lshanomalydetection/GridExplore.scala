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

import es.udc.graph.sparkContextSingleton
import es.udc.graph.EuclideanDistanceProvider
import es.udc.graph.BroadcastLookupProvider

object GridExplore
{
  val DEFAULT_NUM_PARTITIONS:Double=512
  val DEFAULT_THRESHOLD:Int=1
  val ANOMALY_VALUE=1.0
  
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
    for (datasetName <- Array("one_hot_kdd10_sample"))
    {
      val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, DATASETS_ROOT+datasetName+".libsvm")
      
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRDD.map(x => x.features))
      val standardDataRDD=dataRDD.map({case p => new LabeledPoint(p.label,scaler.transform(p.features))})
      
      val folds=MLUtils.kFold(dataRDD, NUM_FOLDS, System.nanoTime().toInt)
      
//      for (mf <- Array(10,50,100,500))
//        for (bs <- Array(5,10,100,1000))
      for (mf <- Array(1))
        for (bs <- Array(42)) // experimentar aplicar raiz quadrada de total numero de instancias ou ln ou 1%??
        {
          var i=0
          var totalAvBucketSize=0.0
          var totalAvBucketDistance = 0.0
          var totalAvBucketCount = 0.0
          var totalAUC=0.0
          var totalTime:Long=0
          for (f <- folds)
          {
            i=i+1
            val trainDataRDD=f._1
            val testDataRDD=f._2
            //trainDataRDD.take(100).foreach(println)
            
//         convert DataFrame columns
            //val convertedVecDF_train = MLUtils.convertVectorColumnsToML(trainDataRDD.toDF("label","features"))
           // convertedVecDF_train.coalesce(1).write.format("libsvm").save("trainDataRDD_"+i)
           // val convertedVecDF_test = MLUtils.convertVectorColumnsToML(testDataRDD.toDF("label","features"))
           // convertedVecDF_test.coalesce(1).write.format("libsvm").save("testDataRDD_"+i)
            
            
            println(s">>>> $datasetName - Fold #$i - MinBucketSize:$bs - Multiplying factor:$mf")
            try
            {
              
              val timeStart=System.currentTimeMillis()
              val model=new LSHReachabilityAnomalyDetector()
                            //.setMinBucketSize(bs)
                            //.setNumTablesMultiplier(mf)
                           .setManualParams(4, 50, 380)
                            //.setHistogramFilePath(Some(s"/home/eirasf/Escritorio/reachability/$datasetName-$bs-$mf.html"))
                            .fit(trainDataRDD)
                            
              totalTime+=System.currentTimeMillis()-timeStart
              totalAUC+=LSHReachabilityAnomalyDetector.evaluateModel(model,testDataRDD)
              
              totalAvBucketSize += model.avBucketSize              
              totalAvBucketDistance+= model.avBucketDistance
              totalAvBucketCount+= model.bucketCount
              
            }
            catch
            {
              case e : Exception =>
                println("ERROR")
            }
          }
          println(s"----------------------------------\n$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS} - Time:${totalTime/NUM_FOLDS} - \n\n\n\n")
          println(s"----------------------------------\n Avg. BucketSize=${totalAvBucketSize/NUM_FOLDS} - Avg. BucketDistance:${totalAvBucketDistance/NUM_FOLDS} - Avg. BucketCount:${totalAvBucketCount/NUM_FOLDS} - \n\n\n\n")
//          pw.write(s"$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS} - Time:${totalTime/NUM_FOLDS}\n")
//          pw.flush()
        }
    }
    //pw.close

    //Stop the Spark Context
    sc.stop()
  }
}