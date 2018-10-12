import torch
import os
import numpy as np
from src.features.bahamas import BahamasLoaderPaired
from src.features.transforms import transform_fcs, XTF, FCS
folder = os.path.basename(os.path.dirname(__file__))
subfolder = os.path.splitext(os.path.basename(__file__))[0]
name = '/' + folder + '/' + subfolder + '/'

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

from src.configs.resnet.b1leaky5 import g_structure
from src.configs.patchgan.nobn_nosig_bfalse import d_structure

epoch_end = 130
ntest = 36

sets = [os.getenv('D32'), os.getenv('G32')]
grouping = [
    [0],
    [1]
]

transforms = []
for i, val in enumerate(sets):
    transforms.append(XTF(k=6, mean=None, inverse=False, totorch=True))
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
        'step_size': 60
    },
    'sample_interval': 100,
    'batch_size': 16,
    'epochs': epoch_end,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': os.getenv('SDIR') + name,
    'save_summary': {
        'epochs': np.arange(0, (epoch_end+1), 5).tolist(),
        'box_size': (100,100),
        'n': 4,
        'grid_size': (2,2)
    }
}


train_loader = BahamasLoaderPaired(sets=sets,
                                   grouping=grouping,
                                   batch_size=schedule['batch_size'],
                                   ntest=ntest,
                                   transform=transforms,
                                   train_set=True)


test_loader = BahamasLoaderPaired(sets=sets,
                                  batch_size=ntest,
                                  ntest=ntest,
                                  transform=transforms,
                                  train_set=False)

