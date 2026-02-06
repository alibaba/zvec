
# Benchmarking scripts

This directory contains benchmarking scripts and reproducing steps.

## COHERE experiments

### Getting COHERE Data

Please download the COHERE 10M dataset to cohere_large_10m as follows:

```bash
... ...           
neighbors.parquet    
shuffle_train-00-of-10.parquet     
shuffle_train-01-of-10.parquet          
shuffle_train-02-of-10.parquet  
shuffle_train-03-of-10.parquet 
shuffle_train-04-of-10.parquet  
shuffle_train-05-of-10.parquet 
shuffle_train-06-of-10.parquet
shuffle_train-07-of-10.parquet
shuffle_train-08-of-10.parquet
shuffle_train-09-of-10.parquet
scalar_labels.parquet     
test.parquet      
```

For convenience, we prepared a docker image with cohere bench datasets: registry.cn-hongkong.cr.aliyuncs.com/zvec/cohere-bench-data. 

You can run a container as follows:

```bash
docker run -it --net=host -d -e DEBUG_MODE=true  --user root --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/zvec/:/home/zvec/  -w /home/zvec --name=cohere_bench zvec-registry.cn-hongkong.cr.aliyuncs.com/zvec/cohere-bench-data:0.0.1 bash

docker exec -it cohere_bench bash
```

The datasets locate at /tmp/cohere/

### Preparing Environment 
Clone code and init:
```bash
$ git clone git@github.com:alibaba/zvec.git
$ cd zvec
$ git submodule update --init
```

Build source code:
```
$ cd /home/zvec/workspace/zvec
$ mkdir build
$ cd build  
$ cmake -DENABLE_SKYLAKE=ON -DCMAKE_BUILD_TYPE=Release ..
```

### Converting Dataset 
Export vector data using python script:
```bash
$ mkdir 10m.output
$ python3 convert_cohere_parquet.py
```

Convert vector data to binary formatted file.
```bash
/home/zvec/workspace/zvec/bin/txt2vecs -input=cohere_train_vector_10m.txt --output=cohere_train_vector_10m.zvec.vecs --dimension=768
```

We've also prepared preprocessed binary formatted files, which can be found in the container below:

```bash
root@iZj6caifjouj5yu8xgsiysZ:/home/zvec# ls -al /tmp/cohere/*zvec 
/tmp/cohere/cohere_large_10m_zvec:
total 30204572
drwxr-xr-x 2 root root        4096 Feb  5 13:12 .
drwxr-xr-x 6 root root        4096 Feb  6 03:38 ..
-rw-r--r-- 1 root root     8664837 Feb  5 13:06 cohere_test_vector_10m.1000.new.txt
-rw-r--r-- 1 root root 30920004295 Feb  5 13:04 cohere_train_vector_10m.new.zvec.vecs
-rw-r--r-- 1 root root      792835 Feb  5 13:05 neighbors.txt

/tmp/cohere/cohere_medium_1m_zvec:
total 3028688
drwxr-xr-x 2 root root       4096 Feb  5 13:14 .
drwxr-xr-x 6 root root       4096 Feb  6 03:38 ..
-rw-r--r-- 1 root root    8661108 Feb  5 13:07 cohere_test_vector_1m.1000.new.txt
-rw-r--r-- 1 root root 3092004295 Feb  5 13:08 cohere_train_vector_1m.new.zvec.vecs
-rw-r--r-- 1 root root     692969 Feb  5 13:08 neighbors.txt
```

### Preparing Bench Config 
Prepare Build Config

```yaml
BuilderCommon:
    BuilderClass: HnswStreamer
    BuildFile: /tmp/cohere/cohere_large_10m_zvec/cohere_train_vector_10m.zvec.vecs
    NeedTrain: true 
    TrainFile: /tmp/cohere/cohere_large_10m_zvec/cohere_train_vector_10m.zvec.vecs
    DumpPath:  /home/zvec/bench/config/cohere_train_vector_10m.dump.index
    IndexPath: /home/zvec/bench/config/cohere_train_vector_10m.index

    ConverterName: CosineInt8Converter
    MetricName: Cosine

    ThreadCount: 16

BuilderParams: 
    proxima.general.builder.thread_count: !!int 16
    proxima.hnsw.builder.thread_count: !!int 16
```

Prepare Search Config

```yaml
SearcherCommon:
    SearcherClass: HnswStreamer
    IndexPath: /home/zvec/bench/config/cohere_train_vector_10m.index
    TopK: 1,10,50,100 
    QueryFile: /tmp/cohere/cohere_large_10m_zvec/cohere_test_vector_1000.new.txt
    QueryType: float 
    QueryFirstSep: ";" 
    QuerySecondSep: " "
    GroundTruthFile: /tmp/cohere/cohere_large_10m_zvec/neighbors.txt
    RecallThreadCount: 1
    BenchThreadCount: 16 
    BenchIterCount: 1000000000 
    CompareById: true

SearcherParams: 
    proxima.hnsw.streamer.ef: !!int 250
```

### Building Index 
Conduct Build 
```bash
$ /home/zvec/workspace/zvec/build/bin/local_build_original ./build.yaml 
```

### Performing Bench
Conduct Recall 
```bash
$ /home/zvec/workspace/zvec/build/bin/recall_original ./search.yaml
```

Conduct Bench 
```bash
$ /home/zvec/workspace/zvec/build/bin/bench_original ./search.yaml
```


