import torch.nn as nn

IMAGE_DIMENSIONS = (3, 256, 256)

d_structure = {
    'type': 'patchgan_discriminator',
    'conv_stack': {
        'input': IMAGE_DIMENSIONS,
        'output': 1,
        'filters': [
            {
                'type': 'normal',
                'spectral': True,
                'out_channels': 64,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': True,
                'pre_batch_norm': False,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'spectral': True,
                'out_channels': 128,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': False,
                'pre_batch_norm': False,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'spectral': True,
                'out_channels': 256,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1,
                'bias': False,
                'pre_batch_norm': False,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'spectral': True,
                'out_channels': 512,
                'kernel_size': 4,
                'stride': 1,
                'padding': 1,
                'bias': False,
                'pre_batch_norm': False,
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'spectral': True,
                'init_type': 'xavier',
                'init_gain': 0.75,
                'out_channels': 1,
                'kernel_size': 4,
                'stride': 1,
                'padding': 1,
                'bias': True,
                'pre_batch_norm': False,
                'activation': nn.Sigmoid(),
                'post_batch_norm': False
            },
        ]
    },
}
