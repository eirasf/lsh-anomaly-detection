package es.udc.graph

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.internal.Logging

trait LoggingTEMP
{
  def logInfo(s: => String):Unit=println(s)
  def logDebug(s: => String):Unit=println(s)
  def logWarning(s: => String):Unit=println(s)
}

object EuclideanLSHasherForAnomaly extends AutotunedHasher with LoggingTEMP
{
  override val MIN_TOLERANCE=0.2
  override val MAX_TOLERANCE=2.0
  protected def log2(n: Double): Double =
  {
    Math.log10(n) / Math.log10(2)
  }
  
  def hashData(data: RDD[(Long, LabeledPoint)], hasher: EuclideanLSHasher, radius: Double): RDD[(Hash, Long)] =
  {
    data.flatMap({ case (index, point) => hasher.getHashes(point.features, index, radius) });
  }
  
  def getHashNeighbors(data:RDD[(Long,LabeledPoint)], hasher:EuclideanLSHasher, radius:Double):RDD[Iterable[Long]]=
  {
    val currentHashes = hashData(data, hasher, radius)

    val hashNeighbors = currentHashes.groupByKey.map({ case (h, n) => n })

    return hashNeighbors
  }
  
  private def computeBestKeyLength(data: RDD[(Long,LabeledPoint)], dimension:Int, desiredMinComparisons:Int): (EuclideanLSHasher,Double) =
  {
    val FRACTION=1.0//0.01
    val INITIAL_RADIUS=0.1
    val initialData = data//data.sample(false, FRACTION, 56804023).map(_.swap)
    
    val numElems=data.count()
    var initialKLength: Int = Math.ceil(log2(numElems / dimension)).toInt + 1
    if (initialKLength<2) initialKLength=2
    logDebug(s"DEBUG: numElems=$numElems dimension=$dimension initialKLength=$initialKLength")
    val minKLength=if (initialKLength>2) (initialKLength / 2).toInt else 1 
    val maxKLength=if (initialKLength>15) (initialKLength * 1.5).toInt else 22
    val hNTables: Int = Math.floor(Math.pow(log2(dimension), 2)).toInt
    val desiredMinComparisonsAdjusted=desiredMinComparisons
    
    val currentData=initialData
    //val currentData=initialData.sample(false, 0.2, 34652912) //20% of the data usually does the job.
    
    logInfo(s"Starting hyperparameter adjusting with:\n\tL:$initialKLength\n\tN:$hNTables\n\tR:$INITIAL_RADIUS\n\tC:$desiredMinComparisonsAdjusted")
    
    var (leftLimit,rightLimit)=(minKLength,maxKLength)
    var radius = INITIAL_RADIUS
    var isRadiusAdjusted = false

    while(true)
    {
      val currentLength=Math.floor((leftLimit+rightLimit)/2.0).toInt
      val tmpHasher = new EuclideanLSHasher(dimension, currentLength, hNTables)
      val hashNeighborsRDD = getHashNeighbors(data, tmpHasher, radius)
      val numNeighborsPerPointRDD = hashNeighborsRDD.flatMap({case l => l.map({case x => (x, l.size-1)})})
                                                    .reduceByKey(_ + _)
      
      val minNumNeighbors=numNeighborsPerPointRDD.map({case (id,rec) => rec }).min()///FRACTION
      
      if ((minNumNeighbors>=desiredMinComparisonsAdjusted*MIN_TOLERANCE) && (minNumNeighbors<=desiredMinComparisonsAdjusted*MAX_TOLERANCE))
      {
        logInfo(s"Found suitable hyperparameters:\n\tL:${tmpHasher.keyLength}\n\tN:${tmpHasher.numTables}\n\tR:$radius")
        return (tmpHasher,radius)
      }
      else
      {
        if (minNumNeighbors<desiredMinComparisonsAdjusted*MIN_TOLERANCE) //Buckets are too small
        {
          //if ((numBuckets==0) || (rightLimit-1 == currentLength)) //If we ended up with no buckets with more than one element or the size is less than the desired minimum
          if (rightLimit-1 == currentLength) //If we ended up with no buckets with more than one element or the size is less than the desired minimum
          {
            if (isRadiusAdjusted)
            {
              logWarning(s"WARNING! - Had to go with hyperparameters:\n\tL:${tmpHasher.keyLength}\n\tN:${tmpHasher.numTables}\n\tR:$radius")
              return (tmpHasher,radius)
            }
            //We start over with a larger the radius
            radius=getSuitableRadius(currentData, new EuclideanLSHasher(dimension, initialKLength, hNTables), radius, None, desiredMinComparisonsAdjusted)
            isRadiusAdjusted=true
            leftLimit=minKLength
            rightLimit=maxKLength
          }
          else
            rightLimit=currentLength
        }
        else //Buckets are too large
        {
          if (leftLimit == currentLength)
          {
            if (isRadiusAdjusted)
            {
              logWarning(s"WARNING! - Had to go with hyperparameters:\n\tL:${tmpHasher.keyLength}\n\tN:${tmpHasher.numTables}\n\tR:$radius")
              return (tmpHasher,radius)
            }
            //We start over with a smaller the radius
            radius=getSuitableRadius(currentData, tmpHasher, 0.000000000001, Some(radius), desiredMinComparisonsAdjusted)
            isRadiusAdjusted=true
            leftLimit=minKLength
            rightLimit=maxKLength
          }
          else
            leftLimit=currentLength
        }
        if (rightLimit<=leftLimit)
        {
          logWarning(s"WARNING! - Had to go with hyperparameters:\n\tL:${tmpHasher.keyLength}\n\tN:${tmpHasher.numTables}\n\tR:$radius")
          return (tmpHasher,radius)
        }
      }
      
      logDebug(s"keyLength update to ${tmpHasher.keyLength} [$leftLimit - $rightLimit] with radius $radius because minNumNeighbors was $minNumNeighbors and wanted [${desiredMinComparisonsAdjusted*MIN_TOLERANCE} - ${desiredMinComparisonsAdjusted*MAX_TOLERANCE}]")
    }
    return (new EuclideanLSHasher(dimension, 1, hNTables), radius)//Dummy
  }
  
  def getSuitableRadius(data:RDD[(Long,LabeledPoint)], hasher:EuclideanLSHasher, minValue:Double, maxValue:Option[Double], desiredMinCount:Int):Double=
  {
    val MIN_TOLERANCE_RADIUS=0.1
    val MAX_TOLERANCE_RADIUS=10.0
    var leftLimit=minValue
    var rightLimit=if (maxValue.isDefined)
                     maxValue.get
                   else
                   {
                     //Find a radius that is too large
                     var done=false
                     var currentValue=leftLimit*2
                     while (!done)
                     {
                       val hashNeighborsRDD = getHashNeighbors(data, hasher, currentValue)
                       val numNeighborsPerPointRDD = hashNeighborsRDD.flatMap({case l => l.map({case x => (x, l.size-1)})})
                                                                      .reduceByKey(_ + _)
                        
                       val minNumNeighbors=numNeighborsPerPointRDD.map({case (id,rec) => rec }).min()///FRACTION
                       done=minNumNeighbors>desiredMinCount*2
                       logDebug(s"Radius range updated to [$leftLimit - $currentValue] got a minNumNeighbors of $minNumNeighbors")
                       if (!done)
                         currentValue*=2
                       if ((minNumNeighbors>MIN_TOLERANCE_RADIUS*desiredMinCount) && (minNumNeighbors<MAX_TOLERANCE_RADIUS*desiredMinCount))
                       {
                         logInfo(s"Found suitable radius at $currentValue")
                         return currentValue
                       }
                     }
                     currentValue
                   }
    while(true)
    {
      val radius=(leftLimit+rightLimit)/2
      val hashNeighborsRDD = getHashNeighbors(data, hasher, radius)
      val numNeighborsPerPointRDD = hashNeighborsRDD.flatMap({case l => l.map({case x => (x, l.size-1)})})
                                                    .reduceByKey(_ + _)
      
      val minNumNeighbors=numNeighborsPerPointRDD.map({case (id,rec) => rec }).min()///FRACTION
      logDebug(s"Radius update to $radius [$leftLimit - $rightLimit] got a minNumNeighbors of $minNumNeighbors")
      if ((minNumNeighbors>=MIN_TOLERANCE_RADIUS*desiredMinCount) && (minNumNeighbors<=MAX_TOLERANCE_RADIUS*desiredMinCount))
      {
        logInfo(s"Found suitable radius at $radius")
        return radius
      }
      //if ((numBuckets==0) || (largestBucketSize<MIN_TOLERANCE*desiredCount))
      if (minNumNeighbors<MIN_TOLERANCE_RADIUS*desiredMinCount)
        leftLimit=radius
      else
        if (minNumNeighbors>MIN_TOLERANCE_RADIUS*desiredMinCount)
        {
          rightLimit=radius
          /*
          //DEBUG
          
            val currentHashes = hashData(data, hasher, radius)
            var lookup:BroadcastLookupProvider=new BroadcastLookupProvider(data)
            //bucketCountBySize is a list of (bucket_size, count) tuples that indicates how many buckets of a given size there are. Count must be >1.
            val bucketCountBySize = currentHashes.groupByKey().filter({case (h,indexList) => indexList.size>1}).flatMap({case (h,indexList) => indexList.map(lookup.lookup(_))}).take(100).foreach(println)
            System.exit(0) 
          */
        }
      if (rightLimit-leftLimit<0.000000001)
      {
        logWarning(s"WARNING! - Had to select radius = $radius")
        return radius
      }
    }
    return 1.0//Dummy
  }
  
  override def getHasherForDataset(data: RDD[(Long,LabeledPoint)], dimension:Int, desiredComparisons:Int):(EuclideanLSHasher,Int,Double)=
  {
    //val factorLevel=Math.pow(10,-minusLogOperations)/0.001
    //val predictedNTables: Int = Math.floor(Math.pow(log2(dimension), 2)).toInt
    //var mComparisons: Int = Math.abs(Math.ceil(hasher.numTables * Math.sqrt(log2(data.count()/(dimension*0.1))))).toInt
    //var mComparisons: Int = Math.abs(Math.ceil(predictedNTables * Math.sqrt(log2(data.count()/(dimension*0.1)*factorLevel)))).toInt
    //println(s"CMAX set to $mComparisons do approximately ${Math.pow(10,-minusLogOperations)} of the calculations wrt brute force.")
    //val (hasher,radius) = computeBestKeyLength(data, dimension, (desiredComparisons/1.5).toInt)
    val (hasher,radius) = computeBestKeyLength(data, dimension, desiredComparisons)

    logDebug("R0:" + radius + " num_tables:" + hasher.numTables + " keyLength:" + hasher.keyLength + " desiredComparisons:" + desiredComparisons)
    //System.exit(0) //DEBUG
    return (hasher, desiredComparisons, radius)
  }
  
  override def getHashes(point:Vector, index:Long, radius:Double):List[(Hash, Long)]=
  {
    return List() //Workaround
  }
}