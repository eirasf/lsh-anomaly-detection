package es.udc.graph

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

object EuclideanLSHasherForAnomaly
{
  protected final def hashData(data: RDD[(Long, LabeledPoint)], hasher: EuclideanLSHasher, radius: Double): RDD[(Hash, Long)] =
  {
    data.flatMap({ case (index, point) => hasher.getHashes(point.features, index, radius) });
  }
  
  /*def getBucketCount(data:RDD[(Long,LabeledPoint)], hasher:EuclideanLSHasher, radius:Double):(Int,Int)=
  {
    val currentHashes = hashData(data, hasher, radius)
    //bucketCountBySize is a list of (bucket_size, count) tuples that indicates how many buckets of a given size there are. Count must be >1.
    val bucketCountBySize = currentHashes.aggregateByKey(0)({ case (n, index) => n + 1 }, { case (n1, n2) => n1 + n2 })
                                         .map({ case (h, n) => (n, 1) })
                                         .reduceByKey(_ + _)
                                         .filter({ case (n1, x) => n1 != 1 })


    
    val numBuckets = if (bucketCountBySize.isEmpty()) 0 else bucketCountBySize.reduce({ case ((n1, x), (n2, y)) => (n1 + n2, x + y) })._2
    val largestBucketSize = if (bucketCountBySize.isEmpty()) 0 else bucketCountBySize.map(_._1).max()
    return (numBuckets, largestBucketSize)
  }*/
  def getHashNeighbors(data:RDD[(Long,LabeledPoint)], hasher:EuclideanLSHasher, radius:Double):RDD[(Iterable[Long])]=
  {
    val currentHashes = hashData(data, hasher, radius)
    //bucketCountBySize is a list of (bucket_size, count) tuples that indicates how many buckets of a given size there are. Count must be >1.
//    val bucketCountBySize = currentHashes.aggregateByKey(0)({ case (n, index) => n + 1 }, { case (n1, n2) => n1 + n2 })
//                                         .map({ case (h, n) => (n, 1) })
//                                         .reduceByKey(_ + _)
//                                         .filter({ case (n1, x) => n1 != 1 })

    val hashNeighbors = currentHashes.groupByKey.map({ case (h, n) => (n) })

    //hashNeighbors.take(100).foreach(println)
 
//    bucketCountBySize.sortBy(_._1)
//                      .repartition(1)
//                      .foreach({ case x => println(x._2 + " buckets with " + x._1 + " elements") })

    return hashNeighbors
  }
}