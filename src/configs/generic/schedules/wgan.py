LATENT = 100

# stock rms optimizer options

rms_opts={
    'lr': 0.01,
    'alpha': 0.99,
    'eps': 1e-08,
    'weight_decay': 0,
    'momentum': 0,
    'centered': False
}

# configuration for a DCGAN trainer

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
    'sample_interval': None,
    'batch_size': None,
    'epochs': None,
    'save_model_interval': None,
    'save_img_interval': None,
    'save_dir': None,
    'latent_dim': LATENT
}
