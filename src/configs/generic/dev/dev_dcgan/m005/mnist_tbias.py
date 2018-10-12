import torch.nn as nn

LATENT = 100
G_CONV_INPUT = (1024, 4, 4)
G_CONV_INPUT_FLAT = 1024 * 4 * 4
D_CONV_OUTPUT = G_CONV_INPUT
D_CONV_OUTPUT_FLAT = G_CONV_INPUT_FLAT
IMAGE_DIMENSIONS = (1, 64, 64)
g_bias = True
d_bias = True

g_structure = {
    'type': 'dcgan_generator',
    'latent_dim': LATENT,
    'conv_stack': {
        'input': LATENT,
        'output': IMAGE_DIMENSIONS,
        'filters': [
            {
                'type': 'transpose',
                'out_channels': 1024,
                'kernel_size': 4,
                'stride': 1,
                'padding': 0,
                'bias': g_bias,
                'pre_batch_norm': nn.BatchNorm2d(1024),
                'activation': nn.ReLU(True),
                'post_batch_norm': False
            },
            {
                'type': 'transpose',
                'out_channels': 512,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': g_bias,
                'pre_batch_norm': nn.BatchNorm2d(512),
                'activation': nn.ReLU(True),
                'post_batch_norm': False
            },
            {
                'type': 'transpose',
                'out_channels': 256,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': g_bias,
                'pre_batch_norm': nn.BatchNorm2d(256),
                'activation': nn.ReLU(True),
                'post_batch_norm': False
            },
            {
                'type': 'transpose',
                'out_channels': 128,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': g_bias,
                'pre_batch_norm': nn.BatchNorm2d(128),
                'activation': nn.ReLU(True),
                'post_batch_norm': False
            },
            {
                'type': 'transpose',
                'out_channels': 1,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': g_bias,
                'pre_batch_norm': False,
                'activation': nn.Tanh(),
                'post_batch_norm': False
            },
        ]

    }

}

d_structure = {
    'type': 'dcgan_discriminator',
    'conv_stack': {
        'input': IMAGE_DIMENSIONS,
        'output': 1,
        'filters': [
            {
                'type': 'normal',
                'out_channels': 128,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': d_bias,
                'pre_batch_norm': False,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': 256,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': d_bias,
                'pre_batch_norm': None,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': 512,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': d_bias,
                'pre_batch_norm': None,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': 1024,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': d_bias,
                'pre_batch_norm': None,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': 1,
                'kernel_size': 4,
                'stride': 1,
                'padding': 0,
                'bias': d_bias,
                'pre_batch_norm': False,
                'activation': None,
                'post_batch_norm': False
            },
        ]
    },
}
