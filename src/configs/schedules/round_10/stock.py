import torch
import os
import numpy as np
from baryon_painter.utils.data_transforms import \
    create_range_compress_transforms, chain_transformations, \
    atleast_3d, squeeze


# stock adam optimizer options

adam_opts = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False
}

loss_params = {
    'n_critic': 5,
    'grad_lambda': 10,
    'l1_lambda': (1e4)/0.05
}


paper_opts = adam_opts
paper_opts['betas'] = (0.5, 0.999)
paper_opts['lr'] = 0.0002

range_compress_transform, range_compress_inv_transform = \
 create_range_compress_transforms(k_values={"dm": 2, "gas": 2, "pressure": 4})


transform = chain_transformations([range_compress_transform,
                                   atleast_3d])

inv_transform = chain_transformations([squeeze,
                                       range_compress_inv_transform])


def Schedule(name, transform=transform, inv_transform=inv_transform,
             loss_params=loss_params, paper_opts=paper_opts,
             epoch_end=100, n_test=64):
    schedule = {
        'type': 'translator',
        'transform': transform,
        'inv_transform': inv_transform,
        'subtype': 'wgp',
        'warm_start': True,
        'loss': 'l1_plus',
        'loss_params': loss_params,
        'g_optimizer': 'adam',
        'd_optimizer': 'adam',
        'g_optim_opts': paper_opts,
        'd_optim_opts': paper_opts,
        'g_decay': torch.optim.lr_scheduler.ExponentialLR,
        'd_decay': torch.optim.lr_scheduler.ExponentialLR,
        'lrdecay_opts': {
            'gamma': 0.98
        },
        'sample_interval': 50,
        'batch_size': 4,
        'n_test': n_test,
        'epochs': epoch_end,
        'save_dir': os.getenv('SDIR') + name,
        'debug_plot': False,
        'mem_debug': True,
        'save_summary': {
            'iters': np.arange(0, 10000, 50).tolist(), #np.arange(0, 1000, 10).tolist()
            'box_size': (100, 100),
            'n': 4,
            'grid_size': (2, 2)
        }
    }
    return schedule
