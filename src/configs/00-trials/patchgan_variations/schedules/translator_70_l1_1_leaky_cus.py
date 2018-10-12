import numpy as np
import os
from src.features.bahamas import BahamasLoaderPaired
from src.features.transforms import transform_fcs, FCS

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
paper_opts['lr'] = 0.00004

schedule = {
    'type': 'translator',
    'loss': 'l1_plus',
    'loss_params': loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': paper_opts,
    'd_optim_opts': paper_opts,
    'sample_interval': 205,
    'batch_size': 8,
    'epochs': 500,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': os.getenv('SDIR') + '/patchgan_variations/patch_70_l1_1_leaky_cus/',
    'save_summary': {
        'epochs': np.arange(0, 501, 5).tolist(),
        'box_size': (100,100),
        'transform': FCS(k=4, inverse=True),
        'n': 4,
        'grid_size': (2,2)
    }
}



train_loader = BahamasLoaderPaired([os.getenv('D32'), os.getenv('G32')],
                                   batch_size=schedule['batch_size'],
                                   ntest=10,
                                   transform=transform_fcs,
                                   train_set=True)


test_loader = BahamasLoaderPaired([os.getenv('D32'), os.getenv('G32')],
                                  batch_size=10,
                                  ntest=10,
                                  transform=transform_fcs,
                                  train_set=False)
