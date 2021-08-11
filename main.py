import os
import sys
import copy
import json
import time
import random
import argparse
import subprocess
import configparser as cp

import torch
import torch.cuda
import torch.nn as nn

import math
import numpy as np
import pandas as pd

import perf4sight

def parse_command_line_args() : 
#{{{
    parser = argparse.ArgumentParser(description='PyTorch Pruning')
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    args = parser.parse_args()
    return args
#}}}

def parse_config_file(configFile):
#{{{
    class Params():
        def __init__(self, configFile):
            # pruner is used here as a label as the pruners/ submodule in perf4sight requires
            # params.pruner as an object
            self.pruner = dict(configFile['create_network_dataset']) if 'create_network_dataset' in\
                    configFile.sections() else None
            self.profile = dict(configFile['profile']) if 'profile' in configFile.sections() else None
            self.fingerprint = dict(configFile['fingerprint']) if 'fingerprint' in configFile.sections()\
                    else None

    return Params(configFile)
#}}}

def main():
#{{{
    args = parse_command_line_args()
    
    config = cp.ConfigParser()
    config.read(args.config_file)
    params = parse_config_file(config)

    if params.pruner is not None:
        perf4sight.create_dataset(params)

    if params.profile is not None:
        perf4sight.profile_network(params)

    if params.fingerprint is not None:
        perf4sight.fingerprint_device(params)
#}}}

if __name__ == '__main__':
    main()
