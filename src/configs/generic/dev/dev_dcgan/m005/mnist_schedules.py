LATENT = 100

# DEFAULT OPTIMIZER OPTIONS +++

rms_opts={
    'lr': 0.01,
    'alpha': 0.99,
    'eps': 1e-08,
    'weight_decay': 0,
    'momentum': 0,
    'centered': False
}


adam_opts = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

wgangp_loss_params = {
    'n_critic': 5,
    'lambda': 10
}

wgangp_paper_opts = adam_opts
wgangp_paper_opts['lr'] = 1e-4
wgangp_paper_opts['betas'] = (0., 0.9)

wgangp_schedule = {
    'type': 'dc_gan',
    'loss': 'wgangp',
    'loss_params': wgangp_loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': wgangp_paper_opts,
    'd_optim_opts': wgangp_paper_opts,
    'sample_interval': 33,
    'batch_size': 64,
    'epochs': 1300,
    'save_model_interval': None,
    'save_img_interval': 500,
    'save_dir': None,
    'latent_dim': LATENT
}



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

wgan_loss_params = {
        'n_critic': 5,
        'clamp': 0.01
    }
wgan_paper_opts = rms_opts
wgan_paper_opts['lr'] = 5e-5

wgan_schedule = {
    'type': 'dc_gan',
    'loss': 'wass',
    'loss_params': wgan_loss_params,
    'g_optimizer': 'rms',
    'd_optimizer': 'rms',
    'g_optim_opts': wgan_paper_opts,
    'd_optim_opts': wgan_paper_opts,
    'sample_interval': 33,
    'batch_size': 64,
    'epochs': 25,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': None,
    'latent_dim': LATENT
}
