import torch.nn as nn

RES_INPUT = (128, 64, 64)
RES_OUTPUT = RES_INPUT

IMAGE_IN =  (2, 512, 512)
IMAGE_OUT = (1, 512, 512)
t_bias = True

g_structure = {
    'type': 'resnet_translator',
    'encode_stack': {
        'input': IMAGE_IN,
        'output': RES_INPUT,
        'filters': [
            {
                'type': 'normal',
                'out_channels': 32,
                'kernel_size': 9,
                'stride': 1,
                'padding': 4,
                'bias': False,
                'pre_batch_norm': nn.BatchNorm2d(32),
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': 64,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'bias': t_bias,
                'pre_batch_norm': nn.BatchNorm2d(64),
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': 128,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'bias': t_bias,
                'pre_batch_norm': nn.BatchNorm2d(128),
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
        ]

    },
    'res_blocks': {
        'n_blocks': 9,
        'input_shape': RES_INPUT,
        'activation': nn.LeakyReLU,
        'acti_params': {'negative_slope': 0.2, 'inplace': True}
    },
    'decode_stack': {
        'input': RES_OUTPUT,
        'output': IMAGE_OUT,
        'filters': [
            {
                'type': 'transpose',
                'out_channels': 64,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'output_padding': 1,
                'bias': t_bias,
                'pre_batch_norm': nn.BatchNorm2d(64),
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'transpose',
                'out_channels': 32,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'output_padding': 1,
                'bias': t_bias,
                'pre_batch_norm': nn.BatchNorm2d(32),
                'activation': nn.LeakyReLU(0.2, True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'init_type': 'xavier',
                'init_gain': 0.75,
                'out_channels': IMAGE_OUT[0],
                'kernel_size': 9,
                'stride': 1,
                'padding': 4,
                'bias': True,
                'pre_batch_norm': None,
                'activation': nn.Tanh(),
                'post_batch_norm': False
            },
        ]

    }

}
