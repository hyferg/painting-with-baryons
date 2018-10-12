import torch.nn as nn

RES_INPUT = (128, 64, 64)
RES_OUTPUT = RES_INPUT

IMAGE_DIMENSIONS = (1, 256, 256)
t_bias = False

t_structure = {
    'type': 'resnet_translator',
    'encode_stack': {
        'input': IMAGE_DIMENSIONS,
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
                'activation': nn.ReLU(True),
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
                'activation': nn.ReLU(True),
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
                'activation': nn.ReLU(True),
                'post_batch_norm': False
            },
        ]

    },
    'res_blocks': {
        'n_blocks': 9,
        'input_size': RES_INPUT
    },
    'decode_stack': {
        'input': RES_OUTPUT,
        'output': IMAGE_DIMENSIONS,
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
                'activation': nn.ReLU(True),
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
                'activation': nn.ReLU(True),
                'post_batch_norm': False
            },
            {
                'type': 'normal',
                'out_channels': IMAGE_DIMENSIONS[0],
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
