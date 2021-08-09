import os
import sys
import copy
import json
import time
import random
import subprocess

import torch
import torch.cuda
import torch.nn as nn

import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import configparser as cp

import src.app as appSrc

import src.adapt.performance_prediction.dapr as daprSrc
import src.adapt.performance_prediction.param_parser as ppSrc
import src.adapt.performance_prediction.model_creator as mcSrc
import src.adapt.performance_prediction.profiler as profilerSrc
import src.adapt.performance_prediction.training as trainingSrc
import src.adapt.performance_prediction.inference as inferenceSrc
import src.adapt.performance_prediction.model_search as modelSearchSrc
import src.adapt.performance_prediction.perf_model_eval as modelEvalSrc
import src.adapt.performance_prediction.input_preprocessor as preprocSrc
import src.adapt.performance_prediction.checkpointing as checkpointingSrc
import src.adapt.performance_prediction.fingerprinter_no_backfill as fingerprinter

import src.adapt.performance_prediction.performance_model.utils as utils
import src.adapt.performance_prediction.performance_model.model as modelSrc

import src.adapt.performance_prediction.pruners.base as pruningSrc
from src.adapt.performance_prediction.pruners.vgg import VGGPruning
from src.adapt.performance_prediction.pruners.alexnet import AlexNetPruning
from src.adapt.performance_prediction.pruners.mnasnet import MnasNetPruning 
from src.adapt.performance_prediction.pruners.googlenet import GoogLeNetPruning 
from src.adapt.performance_prediction.pruners.squeezenet import SqueezeNetPruning 
from src.adapt.performance_prediction.pruners.resnet import ResNet20PruningDependency as ResNetPruning
from src.adapt.performance_prediction.pruners.mobilenetv2 import MobileNetV2PruningDependency as MobileNetV2Pruning 

class Application(appSrc.Application):
    def main(self):
    #{{{
        self.perfModel = modelSrc.PerformanceModel(self.params)
        
        if self.params.writeTests is not None:
            if self.params.writeTests['test_type'] == 'from_model':
                model = utils.read_model(self.params.writeTests['model_file'], self.params.writeTests['net_name'])
                self.perfModel.create_test_from_model(model)
            elif self.params.writeTests['test_type'] == 'single_change':
                self.perfModel.create_single_change_tests()
            elif self.params.writeTests['test_type'] == 'random':
                self.perfModel.create_random_tests()
            else:
                raise ValueError("test_type not defined")
        
        elif self.params.collectData is not None:
            self.dataCollector = profilerSrc.DataCollector(self.params)
            self.dataCollector.collect(int(self.params.gpu_id))

        elif self.params.fingerprint is not None:
            fingerprinter.fingerprint_device(self.params)

        elif self.params.dapr is not None:
            assert self.params.modelSearch is not None and self.params.pruner is not None,\
                    "Config file must have models-search and pruner sections defined"
            if eval(self.params.dapr['finetune_only']):
                daprSrc.finetune(self)
            elif eval(self.params.dapr['retrain_only']):
                if type(eval(self.params.pruner['pruning_perc'])) is list: 
                    levels = eval(self.params.pruner['pruning_perc'])
                    for pp in levels: 
                        self.params.pruner['pruning_perc'] = pp
                        daprSrc.retrain(self)
                else:
                    daprSrc.retrain(self)
            else:
                daprSrc.perform_search(self, self.params)

        elif self.params.modelSearch is not None:
            self.setup_dataset()
            self.setup_model()
            fittingModels = modelSearchSrc.find_fitting_model(self, self.params, verbose=True)

        elif self.params.pruner is not None:
            self.setup_dataset()
            self.setup_model()

            print("Pre-pruning Inference Test")
            self.run_inference()

            if type(eval(self.params.pruner['pruning_perc'])) is list:
                origModel = copy.deepcopy(self.model)
                for pp in eval(self.params.pruner['pruning_perc']):
                    repeats = 1 if self.params.pruner['mode'] != 'random_weighted'\
                                else int(self.params.pruner['repeats'])
                    for i in range(repeats): 
                        self.model = origModel
                        self.params.pruner['pruning_perc'] = pp
                        self.setup_pruners()
                        pn = i if self.params.pruner['mode'] == 'random_weighted' else None
                        channelsPruned, self.model, self.optimiser =\
                                self.pruner.prune_model(self.model, pruneNum=pn)
            else:
                self.setup_pruners()
                channelsPruned, self.model, self.optimiser = self.pruner.prune_model(self.model)
                print("Post-pruning Inference Test")
                self.run_inference()

        elif self.params.modelEval is not None:
            modelEvalSrc.evaluate(self)
        
        elif self.params.decTreeModel is not None:
            self.perfModel.fit_decision_tree_regression()
        
        elif self.params.linearModel is not None:
            self.perfModel.fit_data()
        
        elif self.params.evaluateModel is not None:
            self.perfModel.evaluate_model()
        
        elif self.params.visualise is not None:
            self.perfModel.visualise_trends()

        else:
            self.setup_dataset()
            self.setup_model()
            self.run_training()
    #}}}

    def plot_channels_pruned(self, channelsPruned): 
    #{{{
        toPlot = pd.DataFrame()
        row = pd.Series()
        for k,v in channelsPruned.items(): 
            row['Layer'] = k.replace('.', '_')
            row['Channels Pruned'] = len(v)
            toPlot = toPlot.append(row, ignore_index=True)
        toPlot.plot.bar(x='Layer', y='Channels Pruned')        
    #}}}

    def setup_pruners(self):
    #{{{
        if 'alexnet' in self.params.arch:
            self.pruner = AlexNetPruning(self.params, self.model)
            self.netName = 'AlexNet'
            self.trainableLayers = ['classifier']
        elif 'resnet' in self.params.arch:
            self.pruner = ResNetPruning(self.params, self.model)
            self.netName = 'ResNet{}'.format(self.params.depth)
            self.trainableLayers = ['fc']
        elif 'mobilenet' in self.params.arch:
            self.pruner = MobileNetV2Pruning(self.params, self.model)
            self.netName = 'MobileNetv2'
            self.trainableLayers = ['linear']
        elif 'squeezenet' in self.params.arch:
            self.pruner = SqueezeNetPruning(self.params, self.model)
            self.netName = 'SqueezeNet'
            self.trainableLayers = ['module.conv2']
        elif 'vgg' in self.params.arch:
            self.pruner = VGGPruning(self.params, self.model)
            self.netName = 'VGG'
            self.trainableLayers = ['classifier']
        elif 'mnasnet' in self.params.arch:
            self.pruner = MnasNetPruning(self.params, self.model)
            self.netName = 'MnasNet'
            self.trainableLayers = ['classifier']
        elif 'googlenet' in self.params.arch:
            self.pruner = GoogLeNetPruning(self.params, self.model)
            self.netName = 'GoogLeNet'
            self.trainableLayers = ['classifier']
        else:
            raise ValueError("Pruning not implemented for architecture ({})".format(self.params.arch))
    #}}}
    
    def setup_param_checkpoint(self, configFile):
    #{{{
        config = cp.ConfigParser() 
        config.read(configFile)
        self.params = ppSrc.Params(config)
        self.checkpointer = checkpointingSrc.Checkpointer(self.params, configFile)
        self.setup_params()
        assert len(self.params.gpu_list) == 1, "This is an embedded project, not considering multiple GPUs"
    #}}}
    
    def setup_others(self):
    #{{{
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.trainer = trainingSrc.Trainer()
        self.inferer = inferenceSrc.Inferer()
    #}}}
    
    def run_inference(self):
    #{{{
        # perform inference only
        print('==> Performing Inference')
        return self.inferer.test_network(self.params, self.test_loader, self.model, self.criterion, self.optimiser)
    #}}}
