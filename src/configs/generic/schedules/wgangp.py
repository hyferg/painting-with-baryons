LATENT = 100

# stock adam optimizer options

adam_opts = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False
}

# configuration from WGANGP paper for 64x64 DCGAN trainer

wgangp_loss_params = {
    'n_critic': 5,
    'lambda': 10
}

wgangp_paper_opts = adam_opts
wgangp_paper_opts['lr'] = 1e-4
wgangp_paper_opts['betas'] = (0., 0.9)

schedule = {
    'type': 'dc_gan',
    'loss': 'wgangp',
    'loss_params': wgangp_loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': wgangp_paper_opts,
    'd_optim_opts': wgangp_paper_opts,
    'sample_interval': None,
    'batch_size': None,
    'epochs': None,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': None,
    'latent_dim': LATENT
}
