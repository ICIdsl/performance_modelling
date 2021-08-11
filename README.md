# perf4sight

This repository is the wrapper for the open-source implementation of the tool **perf4sight** proposed in the paper 
*perf4sight: A toolflow to model CNN training performance on Edge GPUs* published at the Workshop on "Embeeded and Real-World Computer Vision in Autonomous Driving" at ICCV 2021.
If you reference this work in a publication, please cite it through the following citation:
```
Add citation
```

## Installation 
`git submodule update --init --recursive` 

This will pull the submodules perf4sight and perf4sight/pruners which are required to use the tool. 

## Usage
This repo is just a wrapper around the perf4sight repo which implements the actual tool.
- **main.py** implements the basic interface that reads the configurations from the **config.ini** file and calls the relevant functions within perf4sight.
- **models/imagenet** contains various PyTorch model description files 
- **models/pruned** contains the pruned versions of the models that were found in **models/imagenet** 
- **profiling_csvs/** contains csvs with profiled latency and memory consumption for various networks on the Tx2 
- **fingerprinting/** stores the various performance models that are developed by the tool

