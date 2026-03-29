
# Benchmarking scripts

This directory contains benchmarking scripts and reproducing steps.

## COHERE experiments

### Getting COHERE Data

Please download the COHERE 10M dataset to cohere_large_10m as follows:

```bash
neighbors_head_1p.parquet   
neighbors_tail_1pgit.parquet   
neighbors_labels_label_20p.parquet  
neighbors_labels_label_50p.parquet  
neighbors_labels_label_80p.parquet  
neighbors_labels_label_95p.parquet             
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

### Preparing Environment 
Clone code and init:
```bash
$ git clone git@github.com:alibaba/zvec.git
$ cd zvec
$ git submodule update --init
```

Make build docker image:
```bash
docker build -t  zvec/build-image:latest -f ./.github/workflows/docker/Dockerfile.ubuntu18.10-glibc228 .
```

Start bulld container:
```bash
docker run -it --net=host -d -e DEBUG_MODE=true  --user root --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/zvec/:/home/zvec/  -w /home/zvec --name=build_zvec zvec/build-image:latest bash
```

Turn-off complation option:
```
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)
=>
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
```

Build source code:
```
$ docker exec -it build_zvec bash
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

### Preparing Bench Config 
Prepare Build Config

```yaml
BuilderCommon:
    BuilderClass: HnswStreamer
    BuildFile: /home/zvec/bench/data/10m/cohere_train_vector_10m.zvec.vecs
    NeedTrain: true 
    TrainFile: /home/zvec/bench/data/10m/cohere_train_vector_10m.zvec.vecs
    DumpPath:  /home/zvec/bench/config/cohere_train_vector_10m.index
    IndexPath: /home/zvec/bench/config/cohere_train_vector_10m.2.index

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
    IndexPath: /home/zvec/bench/config/cohere_train_vector_10m.2.index
    TopK: 1,10,50,100 
    QueryFile: /home/zvec/bench/data/10m/cohere_test_vector_1000.new.txt
    QueryType: float 
    QueryFirstSep: ";" 
    QuerySecondSep: " "
    GroundTruthFile: /home/zvec/bench/data/10m/neighbors.txt
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


