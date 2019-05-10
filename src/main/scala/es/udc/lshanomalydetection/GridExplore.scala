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
    
    val pw = new PrintWriter(new File("/home/eirasf/Escritorio/summary.txt"))
    
    //for (datasetName <- Array("abalone1-8","abalone9-11","abalone11-29","arritmia","german_statlog"))
    for (datasetName <- Array("arritmia","german_statlog"))
    {
      val dataRDD: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, DATASETS_ROOT+datasetName+".libsvm")
      
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRDD.map(x => x.features))
      val standardDataRDD=dataRDD.map({case p => new LabeledPoint(p.label,scaler.transform(p.features))})
      
      val folds=MLUtils.kFold(dataRDD, NUM_FOLDS, System.nanoTime().toInt)
      
      for (mf <- Array(5,10,100))
        for (bs <- Array(5,10,100))
        {
          var i=0
          var totalAUC=0.0
          for (f <- folds)
          {
            i=i+1
            val trainDataRDD=f._1
            val testDataRDD=f._2
            
            println(s">>>> $datasetName - Fold #$i - MinBucketSize:$bs - Multiplying factor:$mf")
            try
            {
              val model=new LSHAnomalyDetector()
                            .setMinBucketSize(bs)
                            .setNumTablesMultiplier(mf)
                            .setHistogramFilePath(Some(s"/home/eirasf/Escritorio/$datasetName-$bs-$mf.html"))
                            .fit(trainDataRDD)
                            
              
              totalAUC+=LSHAnomalyDetector.evaluateModel(model,testDataRDD)
            }catch
            {
              case e : Exception =>
                println("ERROR")
            }
          }
          println(s"----------------------------------\n$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS}\n\n\n\n")
          pw.write(s"$datasetName - MinBucketSize:$bs - Multiplying factor:$mf - Avg. AUROC=${totalAUC/NUM_FOLDS}\n")
          pw.flush()
        }
    }
    pw.close

    //Stop the Spark Context
    sc.stop()
  }
}