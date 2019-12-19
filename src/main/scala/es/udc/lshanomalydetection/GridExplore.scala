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
    
    val DATASETS_ROOT="file:///mnt/NTFS/owncloud/Datasets/datasets-anomalias/jorge/"
    //val DATASETS_ROOT="file:///Users/jorgemeira/OneDrive - Instituto Superior de Engenharia do Porto/Doutoramento/datasets/datasets-anomalias/"
    val NUM_FOLDS=5
    
    //val pw = new PrintWriter(new File("/home/eirasf/Escritorio/reachability/grid-summary-fast-full.txt"))
    //Array("abalone1-8","abalone9-11","abalone11-29", "arritmia-fix-ohe","german_statlog-ohe", "covtype2vs4", "one_hot_covtype2vs4_sample",
    // "kddcup10-http-normal-vs-all","kddcup10-normal-vs-all_sample","kddcup10-smtp-normal-vs-all", "one_hot_ids_sample", "2_banana_clusters", "2_cirucular_clusters",
    // "2_point_clouds_with_variance", "3_anisotropic_clusters", "3_point_clouds", "scikit_1_cluster", "scikit_1_cluster_with_variance", "scikit_2_bananas_shape", "scikit_2_clusters" ))
    //for (datasetName <- Array("abalone1-8","abalone9-11","abalone11-29", "german_statlog", "arritmia-fix", "one_hot_covtype2vs4_sample",
    for (datasetName <- Array(//"abalone1-8"
                  //"abalone1-8","abalone9-11","abalone11-29", "arritmia-fix-ohe","german_statlog-ohe", "covtype2vs4", "one_hot_covtype2vs4_sample",
                  //"covtype2vs4", 
                  "one_hot_covtype2vs4_sample",
                 "kddcup10-http-normal-vs-all","kddcup10-normal-vs-all_sample","kddcup10-smtp-normal-vs-all", "one_hot_ids_sample"))
    {
      val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, DATASETS_ROOT+datasetName+".libsvm")
      
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRDD.map(x => x.features))
      val standardDataRDD=dataRDD.map({case p => new LabeledPoint(p.label,scaler.transform(p.features))})
      
      val folds=MLUtils.kFold(standardDataRDD, NUM_FOLDS, System.nanoTime().toInt)
      
//      for (mf <- Array(10,50,100,500))
//        for (bs <- Array(5,10,100,1000))
//      for (keyLength <- Array(1,2,4,8,16,32,64,128))
//      for (nt <- Array(5,10,50,100))//,500,1000))
        for (w <- Array(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)) // experimentar aplicar raiz quadrada de total numero de instancias ou ln ou 1%??
//        for (w <- Array(16))
        {
          val keyLength=4
          val nt=50
          var i=0
          var totalAvBucketSize=0.0
          var totalAbsoluteAvBucketSize=0.0
          var totalAvBucketDistance = 0.0
          var totalBucketCount = 0.0
          var totalFilteredBucketCount = 0.0
          var totalAUCs=List(0.0,0.0,0.0,0.0)
          var totalTime:Long=0
          var totalstdBucketSize = 0.0
          
          //DEBUG
          //val f=folds(0)
          var aucPerFoldStrings=List("","","","")
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
            
            
            println(s">>>> $datasetName - Fold #$i - W:$w - KL:$keyLength - NT:$nt")
            /*try
            {*/
              
              val timeStart=System.currentTimeMillis()
              val model=new LSHReachabilityAnomalyDetector()
                            //.setMinBucketSize(bs)
                            //.setNumTablesMultiplier(mf)
                           .setManualParams(keyLength, nt, 1.0, w)
                            //.setHistogramFilePath(Some(s"/home/eirasf/Escritorio/reachability/$datasetName-$bs-$mf.html"))
                            //.setKeyLength(Some(keyLength))
                            //.setNumTables(Some(nt))
                            .fit(trainDataRDD)
                            
              totalTime+=System.currentTimeMillis()-timeStart
              val foldAUCs=LSHReachabilityAnomalyDetector.evaluateModel(model,testDataRDD)
              totalAUCs=totalAUCs.zip(foldAUCs).map({case (x,y) => x+y})
              
              aucPerFoldStrings=aucPerFoldStrings.zip(foldAUCs.map(_.toString)).map({case (x,y) => x+";"+y})

              totalAvBucketSize += model.avBucketSize              
              totalAvBucketDistance+= model.avBucketDistance
              totalBucketCount+= model.bucketCount
              totalFilteredBucketCount+= model.filteredBucketCount
              totalAbsoluteAvBucketSize+=model.absoluteAvBucketSize
              totalstdBucketSize+=model.stdBucketSize
              
            /*}
            catch
            {
              case e : Exception =>
                println("ERROR")
            }*/
          }//TODO DEBUG
          val avgAUCs=totalAUCs.map({case x => x/NUM_FOLDS})
          val stdAUCs=aucPerFoldStrings.zip(avgAUCs).map({case (s,avg) => math.sqrt(s.substring(1).split(";").map({case x => math.pow(x.toDouble-avg,2)}).sum / NUM_FOLDS)})
          println(s"FOLD>>>>$datasetName;$keyLength;$nt;$w${aucPerFoldStrings(0)}")
          println(s"FOLD>>>>$datasetName;$keyLength;$nt;$w${aucPerFoldStrings(1)}")
          println(s"FOLD>>>>$datasetName;$keyLength;$nt;$w${aucPerFoldStrings(2)}")
          println(s"FOLD>>>>$datasetName;$keyLength;$nt;$w${aucPerFoldStrings(3)}")
          println(s"TOTAL>>>>$datasetName;$keyLength;$nt;$w;${totalAUCs.map({case x => (x/NUM_FOLDS).toString}).mkString(";")};${stdAUCs.map(_.toString).mkString(";")};${totalTime/NUM_FOLDS};${totalAbsoluteAvBucketSize/NUM_FOLDS};${totalAvBucketSize/NUM_FOLDS};${totalstdBucketSize/NUM_FOLDS};${totalBucketCount/NUM_FOLDS};${totalFilteredBucketCount/NUM_FOLDS}")
          println(s"----------------------------------\n$datasetName - W:$w - KeyLength: $keyLength - NT: $nt - Time:${totalTime/NUM_FOLDS} - \n\n\n\n")
          avgAUCs.foreach(println)
          println(s"----------------------------------\n Avg. AbsBucketSize=${totalAbsoluteAvBucketSize/NUM_FOLDS} - Avg. BucketSize=${totalAvBucketSize/NUM_FOLDS} - std. BucketSize :${totalstdBucketSize/NUM_FOLDS} - Avg. BucketDistance:${totalAvBucketDistance/NUM_FOLDS} - Avg. BucketCount:${totalBucketCount/NUM_FOLDS} - Avg. FilteredBucketCount:${totalFilteredBucketCount/NUM_FOLDS} - \n\n\n\n")
//          pw.write(s"$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS} - Time:${totalTime/NUM_FOLDS}\n")
//          pw.flush()
        }
    }
    //pw.close

    //Stop the Spark Context
    sc.stop()
  }
}
