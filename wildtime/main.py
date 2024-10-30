import argparse
import os
import random

import numpy as np
import torch
from torch import cuda

from baseline_trainer import trainer_init
from wildtime.methods.agem.agem import AGEM
from wildtime.methods.coral.coral import DeepCORAL
from wildtime.methods.erm.erm import ERM
from wildtime.methods.ewc.ewc import EWC
from wildtime.methods.ft.ft import FT
from wildtime.methods.groupdro.groupdro import GroupDRO
from wildtime.methods.irm.irm import IRM
from wildtime.methods.si.si import SI
from wildtime.methods.simclr.simclr import SimCLR
#from wildtime.methods.swa.swa import SWA
#from wildtime.methods.swav.swav import SwaV

device = 'cuda' if cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run WildTime training")
    
    # Define the arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--method', type=str, required=True, choices=['groupdro', 'coral', 'irm', 'ft', 'erm', 'ewc', 'agem', 'si', 'simclr', 'swav', 'swa'], help='Method to use')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode')
    parser.add_argument('--mini_batch_size', type=int, default=32, help='Mini batch size')
    parser.add_argument('--train_update_iter', type=int, default=6000, help='Number of training update iterations')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--split_time', type=int, default=1970, help='Split time for dataset')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--load_model', action='store_true', help='Load model from checkpoint')

    # Parse arguments
    configs = parser.parse_args()

    # Set random seeds
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.cuda.manual_seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    # Check if directories exist
    if not os.path.isdir(configs.data_dir):
        raise ValueError(f'Data directory {configs.data_dir} does not exist!')
    if configs.load_model and not os.path.isdir(configs.log_dir):
        raise ValueError(f'Model checkpoint directory {configs.log_dir} does not exist!')
    if not os.path.isdir(configs.results_dir):
        raise ValueError(f'Results directory {configs.results_dir} does not exist!')

    if configs.method in ['groupdro', 'irm']:
        configs.reduction = 'none'

    dataset, criterion, network, optimizer, scheduler = trainer_init(configs)

    # Initialize the trainer based on the selected method
    if   configs.method == 'groupdro': trainer = GroupDRO(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'coral': trainer = DeepCORAL(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'irm': trainer = IRM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'ft': trainer = FT(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'erm': trainer = ERM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'ewc': trainer = EWC(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'agem': trainer = AGEM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'si': trainer = SI(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'simclr': trainer = SimCLR(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'swav': trainer = SwaV(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'swa': trainer = SWA(configs, dataset, network, criterion, optimizer, scheduler)
    else: raise ValueError("Invalid method specified")

    trainer.run()
