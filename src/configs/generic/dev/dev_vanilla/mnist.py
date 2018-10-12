import torch.nn as nn

latent = 100
image_dimensions = (1, 28, 28)
flat_image = 1*28*28

g_structure = {
    'type': 'vanilla_generator',
    'latent_dim': latent,
    'image_dimensions': image_dimensions,
    'linear_stack': {
        'input': latent,
        'output': flat_image,
        'layers': [
            {
                'out_features': 512,
                'activation': nn.ReLU(inplace=True),
                'pre_batch_norm': True,
                'post_batch_norm': False

            },
            {
                'out_features': 512,
                'activation': nn.ReLU(inplace=True),
                'pre_batch_norm': True,
                'post_batch_norm': False
            },
            {
                'out_features': 512,
                'activation': nn.ReLU(inplace=True),
                'pre_batch_norm': True,
                'post_batch_norm': False
            },
            {
                'out_features': flat_image,
                'activation': nn.Tanh(),
                'pre_batch_norm': True,
                'post_batch_norm': False
            }
        ]
    }
}

d_structure = {
    'type': 'vanilla_discriminator',
    'linear_stack': {
        'input': flat_image,
        'output': 1,
        'layers': [
            {
                'out_features': 512,
                'activation': nn.ReLU(inplace=True),
                'pre_batch_norm': True,
                'post_batch_norm': False
            },
            {
                'out_features': 512,
                'activation': nn.ReLU(inplace=True),
                'pre_batch_norm': True,
                'post_batch_norm': False
            },
            {
                'out_features': 512,
                'activation': nn.ReLU(inplace=True),
                'pre_batch_norm': True,
                'post_batch_norm': False
            },
            {
                'out_features': 1,
                'activation': nn.Sigmoid(),
                'pre_batch_norm': True,
                'post_batch_norm': False
            }
        ]
    }
}

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

wgangp_paper_opts = adam_opts
wgangp_paper_opts['lr'] = 0.0001
wgangp_paper_opts['betas'] = (0.5, 0.9)

wgangp_loss_params = {
    'n_critic': 5,
    'lambda': 10
}

wgangp_schedule = {
    'type': 'vanilla_gan',
    'loss': 'wgangp',
    'loss_params': wgangp_loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': wgangp_paper_opts,
    'd_optim_opts': wgangp_paper_opts,
    'iters': 200000,
    'img_shape': 28,
    'img_channels': 1,
    'sample_interval': 100,
    'batch_size': 64,
    'latent_dim': 100
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


wgan_loss_params = {
        'n_critic': 5,
        'clamp': 0.01
    }

wgan_paper_opts = rms_opts
wgan_paper_opts['lr'] = 0.00005

wgan_schedule = {
    'type': 'vanilla_gan',
    'loss': 'wass',
    'loss_params': wgan_loss_params,
    'g_optimizer': 'rms',
    'd_optimizer': 'rms',
    'g_optim_opts': wgan_paper_opts,
    'd_optim_opts': wgan_paper_opts,
    'iters': 200000,
    'img_shape': 28,
    'img_channels': 1,
    'sample_interval': 100,
    'batch_size': 64,
    'latent_dim': 100
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_loss_params = {
    'n_critic': 2
}

n_schedule = {
    'type': 'vanilla_gan',
    'loss': 'normal',
    'loss_params': n_loss_params,
    'g_optimizer': 'adam',
    'd_optimizer': 'adam',
    'g_optim_opts': adam_opts,
    'd_optim_opts': adam_opts,
    'iters': 200000,
    'img_shape': 28,
    'img_channels': 1,
    'sample_interval': 100,
    'batch_size': 64,
    'latent_dim': 100
}
