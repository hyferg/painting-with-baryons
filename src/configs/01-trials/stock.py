import os
import numpy as np
from src.features.bahamas import BahamasLoaderPaired
from src.features.transforms import transform_fcs, XTF
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
    'd_period': 2,
    'lambda': 1
}


paper_opts = adam_opts
paper_opts['betas'] = (0.5, 0.999)
paper_opts['lr'] = 0.0002

from src.configs.resnet.stock import g_structure
from src.configs.patchgan.stock import d_structure

epoch_end = 90
ntest=30

schedule = {
    'type': 'translator',
    'loss': 'l1_plus',
    'loss_params': loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': paper_opts,
    'd_optim_opts': paper_opts,
    'sample_interval': 205,
    'batch_size': 2,
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

sets = [os.getenv('D32'), os.getenv('G32')]
grouping = [
    [0],
    [1]
]

transforms = []
for i, val in enumerate(sets):
    transforms.append(XTF(k=6, mean=None, inverse=False, totorch=True))

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

