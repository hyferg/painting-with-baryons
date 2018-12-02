import torch
import os
from baryon_painter.utils.data_transforms import \
    create_range_compress_transforms, chain_transformations, \
    atleast_3d, squeeze

range_compress_transform, range_compress_inv_transform = \
 create_range_compress_transforms(k_values={"dm": 1.5, "pressure": 4},
                                  modes={'dm': 'x/(1+x)',
                                         'pressure': 'log'})


transform = chain_transformations([range_compress_transform,
                                   atleast_3d])

inv_transform = chain_transformations([squeeze,
                                       range_compress_inv_transform])

init_params = {
    'g': 'kaiming',
    'd': 'kaiming',
}

optim_opts = {
    'lr': 0.0002,
    'betas': (0.5, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False
}

optimizer_params = {
    'grad_lambda': 10,
    'perceptual_lambda': 1,
    'g': {
        'type': 'adam',
        'opts': optim_opts,
        'decay': {
            'obj': torch.optim.lr_scheduler.ExponentialLR,
            'opts': {
                'gamma': 0.98
            }
        }
    },
    'd': {
        'type': 'adam',
        'opts': optim_opts,
        'decay': {
            'obj': torch.optim.lr_scheduler.ExponentialLR,
            'opts': {
                'gamma': 0.98
            }
        }
    }
}


schedule = {
    'type': 'spectral',
    'iterator_type': 'troster',
    'transform': transform,
    'inv_transform': inv_transform,
    'optimizer_params': optimizer_params,
    'redshifts': [0.0, 0.5, 1.0],
    'batch_size': 4,
    'n_test': 64,
    'epochs':  10000,
    'save_dir': os.getenv('SDIR'),
}
