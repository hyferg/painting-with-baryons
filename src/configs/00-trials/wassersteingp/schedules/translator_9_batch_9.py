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

wgangp_loss_params = {
    'n_critic': 5,
    'grad_lambda': 10,
    'l1_lambda': 100
}

wgangp_paper_opts = adam_opts
wgangp_paper_opts['lr'] = 1e-4
wgangp_paper_opts['betas'] = (0.0, 0.9)

schedule = {
    'type': 'translator',
    'subtype': 'wgp',
    'loss': 'l1_plus',
    'loss_params': wgangp_loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': wgangp_paper_opts,
    'd_optim_opts': wgangp_paper_opts,
    'sample_interval': 205,
    'batch_size': 9,
    'epochs': 51,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': os.getenv('SDIR') + '/wgp_9_batch_9/',
    'save_summary': {
        'epochs': [1, 3, 5, 10, 30, 50],
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
