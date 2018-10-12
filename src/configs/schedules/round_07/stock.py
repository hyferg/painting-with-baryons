import torch
import os
import numpy as np
from src.features.bahamas import BahamasLoaderPaired
from src.features.transforms import transform_fcs, XTF, FCS

# stock adam optimizer options

adam_opts = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False
}

# configuration from stanford tutorial

loss_params = {
    'n_critic': 5,
    'grad_lambda': 10,
    'l1_lambda': 100
}


paper_opts = adam_opts
paper_opts['betas'] = (0.0, 0.9)
paper_opts['lr'] = 5e-5


epoch_end = 100
ntest = 64

sets = [
    os.getenv('D32Z00'),
    os.getenv('D32Z05'),
    os.getenv('D32Z10'),
    os.getenv('D32Z20'),
    os.getenv('D32V2Z00'),
    os.getenv('P32Z00'),
    os.getenv('P32Z05'),
    os.getenv('P32Z10'),
    os.getenv('P32Z20'),
    os.getenv('P32V2Z00')
]
grouping = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]
]

transforms = []
for i, val in enumerate(sets):
    transforms.append(FCS(k=4, inverse=False, totorch=True, scale=1.75))

def Schedule(name):
    schedule = {
        'type': 'translator',
        'subtype': 'wgp',
        'warm_start': True,
        'loss': 'l1_plus',
        'loss_params': loss_params,
        'g_optimizer': 'adam',
        'd_optimizer': 'adam',
        'g_optim_opts': paper_opts,
        'd_optim_opts': paper_opts,
        'g_decay': torch.optim.lr_scheduler.StepLR,
        'd_decay': torch.optim.lr_scheduler.StepLR,
        'lrdecay_opts': {
            'step_size': 75
        },
        'sample_interval': 100,
        'batch_size': 4,
        'epochs': epoch_end,
        'save_model_interval': None,
        'save_img_interval': None,
        'save_dir': os.getenv('SDIR') + name,
        'save_summary': {
            'epochs': np.arange(0, (epoch_end+1), 2).tolist(),
            'box_size': (100,100),
            'n': 4,
            'grid_size': (2,2)
        }
    }
    return schedule


def TrainLoader(schedule):
    return BahamasLoaderPaired(sets=sets,
                                grouping=grouping,
                                batch_size=schedule['batch_size'],
                                ntest=ntest,
                                transform=transforms,
                                train_set=True)


def TestLoader(schedule):
    return BahamasLoaderPaired(sets=sets,
                               grouping=grouping,
                               batch_size=ntest,
                               ntest=ntest,
                               transform=transforms,
                               train_set=False)

