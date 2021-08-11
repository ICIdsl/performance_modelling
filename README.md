# perf4sight usage example 

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

### perf4sight/create_network_dataset.py
This prunes a all the networks specified in the **create_network_dataset** section of config.ini and stores them in *[model_path]/imagenet*.
*[model_desc_dir]* is the location on your machine where the model description files are stored.
Refer to the README in perf4sight/pruners on how to prune a custom model.
Decorators need to be placed on the model description file in order to do so.

## perf4sight/profiling
This performs the profiling on the target device and is controlled by the parameters in the **profile** section of the *config.ini*.
The *create_config* part of the section can be run on any machine and creates a config file in a format understood by the profilers present in *perf4sight/profiling/tx2/*
and *perf4sight/profiling/2080ti*.
The config files are created in *perf4sight/profiling/profiling_configs*.
The *[device_model_loc]* parameter is a path to the model description files that are found in *[local_model_loc]* but on the target device.

Following this step, the model description files need to be copied over to the location in *[device_model_loc]* on the target device, and the generated test config
should be copied over to the location specified in *[test_config]* on the target device.

The *run_profiling* section has to be run on the target device and the resulting data will be stored in *[log_dir]* on the target device.
The *[till_first_fail]* option causes the profiler to stop profiling after the first network configuration that can't be run on that device without running out of memory.
