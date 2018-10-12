import os
import numpy as np
from src.features.bahamas import BahamasLoaderPaired
from src.features.transforms import transform_fcs, FCS
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
        'transform': FCS(k=4, inverse=True),
        'n': 4,
        'grid_size': (2,2)
    }
}



train_loader = BahamasLoaderPaired([os.getenv('D32'), os.getenv('G32')],
                                   batch_size=schedule['batch_size'],
                                   ntest=30,
                                   transform=transform_fcs,
                                   train_set=True)


test_loader = BahamasLoaderPaired([os.getenv('D32'), os.getenv('G32')],
                                  batch_size=10,
                                  ntest=30,
                                  transform=transform_fcs,
                                  train_set=False)

