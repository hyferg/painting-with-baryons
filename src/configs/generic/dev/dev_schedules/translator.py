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
    'd_period': 1,
    'lambda': 100
}

paper_opts = adam_opts
paper_opts['betas'] = (0.5, 0.999)
paper_opts['lr'] = 0.0002

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
    'epochs': 100,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': None,
}
