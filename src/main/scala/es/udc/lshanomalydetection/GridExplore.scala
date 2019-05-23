package es.udc.lshanomalydetection

import java.io.File
import java.io.PrintWriter

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import es.udc.graph.sparkContextSingleton

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
    
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    
    val DATASETS_ROOT="file:///mnt/NTFS/owncloud/Datasets/datasets-anomalias/"
    val NUM_FOLDS=5
    
    val pw = new PrintWriter(new File("/home/eirasf/Escritorio/reachability/grid-summary-fast-full.txt"))
    
    for (datasetName <- Array("abalone1-8","abalone9-11","abalone11-29", "arritmia-fix-ohe","german_statlog-ohe","kddcup10-http-normal-vs-all","kddcup10-normal-vs-all","kddcup10-smtp-normal-vs-all"))
    {
      val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, DATASETS_ROOT+datasetName+".libsvm")
      
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRDD.map(x => x.features))
      val standardDataRDD=dataRDD.map({case p => new LabeledPoint(p.label,scaler.transform(p.features))})
      
      val folds=MLUtils.kFold(dataRDD, NUM_FOLDS, System.nanoTime().toInt)
      
      for (mf <- Array(10,50,100,500))
        for (bs <- Array(5,10,100,1000))
        {
          var i=0
          var totalAUC=0.0
          var totalTime:Long=0
          for (f <- folds)
          {
            i=i+1
            val trainDataRDD=f._1
            val testDataRDD=f._2
            
            println(s">>>> $datasetName - Fold #$i - MinBucketSize:$bs - Multiplying factor:$mf")
            try
            {
              val timeStart=System.currentTimeMillis()
              val model=new LSHReachabilityAnomalyDetector()
                            .setMinBucketSize(bs)
                            .setNumTablesMultiplier(mf)
                            .setHistogramFilePath(Some(s"/home/eirasf/Escritorio/reachability/$datasetName-$bs-$mf.html"))
                            .fit(trainDataRDD)
                            
              totalTime+=System.currentTimeMillis()-timeStart
              totalAUC+=LSHReachabilityAnomalyDetector.evaluateModel(model,testDataRDD)
            }
            catch
            {
              case e : Exception =>
                println("ERROR")
            }
          }
          println(s"----------------------------------\n$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS} - Time:${totalTime/NUM_FOLDS}\n\n\n\n")
          pw.write(s"$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS} - Time:${totalTime/NUM_FOLDS}\n")
          pw.flush()
        }
    }
    pw.close

    //Stop the Spark Context
    sc.stop()
  }
}