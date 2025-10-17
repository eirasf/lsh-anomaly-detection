# LSH Anomaly Detection
Implementation of a fast anomaly detection algorithm for Big Data using Locality-sensitive Hashing

Compilation requires compiling [KNiNe](https://github.com/eirasf/KNiNe/) and placing the jar in the lib folder. After that:

    cd <PATH_TO_LSH_ANOMALY_DETECTION>

    sbt clean assembly

Execution:

    spark-submit [--master "local[NUM_THREADS]"] --class es.udc.lshanomalydetection.LSHAnomalyDetector <PATH_TO_JAR_FILE> <INPUT_DATASET> <OUTPUT_GRAPH> [options]

## Execution

```
Usage: LSHAnomalyDetector dataset [options]
    Dataset must be a libsvm file
Options:
    -p    Number of partitions for the data RDDs (default: 512)

Advanced LSH options:
    -r    Starting radius (default: 0.1)
    -n    Number of hashes per item (default: auto)
    -l    Hash length (default: auto)
```

## Reference
```
@article{meira2022fast,
  title={Fast anomaly detection with locality-sensitive hashing and hyperparameter autotuning},
  author={Meira, Jorge and Eiras-Franco, Carlos and Bol{\'o}n-Canedo, Ver{\'o}nica and Marreiros, Goreti and Alonso-Betanzos, Amparo},
  journal={Information Sciences},
  volume={607},
  pages={1245--1264},
  year={2022},
  publisher={Elsevier}
}
```
