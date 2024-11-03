import argparse
import os
import random

import numpy as np
import torch
from torch import cuda

from baseline_trainer import trainer_init
from config import config  # Import default configuration
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
    # 1. Create default configuration
    configs = argparse.Namespace(**config)
    
    # 2. Define command line arguments
    parser = argparse.ArgumentParser(description="Run WildTime training with config overrides")
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--method', type=str, choices=['groupdro', 'coral', 'irm', 'ft', 'erm', 'ewc', 'agem', 'si', 'simclr', 'swav', 'swa'], help='Method to use')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode')
    parser.add_argument('--mini_batch_size', type=int, help='Mini batch size')
    parser.add_argument('--train_update_iter', type=int, help='Number of training update iterations')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--split_time', type=int, help='Split time for dataset')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--log_dir', type=str, help='Directory for saving logs and models')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--results_dir', type=str, help='Results directory')
    parser.add_argument('--load_model', action='store_true', help='Load model from checkpoint')

    '''
    # 3. Override default configuration with command line arguments
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:  # Only override non-None command line arguments
            setattr(configs, key, value)
    '''
    print(configs)
    # 4. Set random seed
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.cuda.manual_seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    # 5. Check if directories exist
    if not os.path.isdir(configs.data_dir):
        raise ValueError(f'Data directory {configs.data_dir} does not exist!')
    if configs.load_model and not os.path.isdir(configs.log_dir):
        raise ValueError(f'Model checkpoint directory {configs.log_dir} does not exist!')
    if not os.path.isdir(configs.results_dir):
        raise ValueError(f'Results directory {configs.results_dir} does not exist!')

    if configs.method in ['groupdro', 'irm']:
        configs.reduction = 'none'

    # Initialize dataset, model, and optimizer
    dataset, criterion, network, optimizer, scheduler = trainer_init(configs)

    # Select trainer based on method
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

    # Run the trainer
    trainer.run()
