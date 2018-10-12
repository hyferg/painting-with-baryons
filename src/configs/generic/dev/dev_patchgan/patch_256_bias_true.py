import torch.nn as nn

IMAGE_DIMENSIONS = (6, 256, 256)
d_bias = True

d_structure = {
    'type': 'patchgan_discriminator',
    'conv_stack': {
        'input': IMAGE_DIMENSIONS,
        'output': 1,
        'filters': [
            {
                'type': 'normal',
                'out_channels': 64,
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
                'out_channels': 128,
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
                'stride': 1,
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
                'padding': 1,
                'bias': d_bias,
                'pre_batch_norm': False,
                'activation': nn.Sigmoid(),
                'post_batch_norm': False
            },
        ]
    },
}
