import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,
                 input_shape,
                 kernel_size=(3,3),
                 activation=nn.ReLU,
                 norm=nn.BatchNorm2d,
                 bias=False,
                 clip = False,
                 clip_val=0):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = norm
        self.clip = clip
        self.clip_val = clip_val

        self.channels = input_shape[0]
        self.stride = (1,1)
        pad = (kernel_size[0] -1 )//2
        self.padding = (pad, pad)
        self.stack = nn.Sequential()
        self.bias=bias
        self._build_stack()

    def _build_stack(self):

        self.stack.add_module('Conv-1',
                              nn.Conv2d(self.channels, self.channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        bias=self.bias))

        self.stack.add_module('Norm-1',
                              self.norm(self.channels))

        self.stack.add_module('Acti-1',
                              self.activation(self.channels))

        self.stack.add_module('Conv-2',
                              nn.Conv2d(self.channels, self.channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        bias=self.bias))

        self.stack.add_module('Norm-2',
                              self.norm(self.channels))

    def forward(self, x):
        out = self.stack(x)
        if self.clip:
            x = torch.clamp(x, -1, self.clip_val)
            return x + out
        return x + out


class ResBlocks(nn.Module):
    def __init__(self,
                 input_shape,
                 n_blocks=1,
                 kernel_size=(3,3),
                 activation=nn.ReLU,
                 norm=nn.BatchNorm2d,
                 clip=False,
                 clip_val=0):

        super().__init__()

        self.input_shape = input_shape
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.clip = clip
        self.clip_val = clip_val
        self.network = nn.Sequential()
        self._build_network()

    def _build_network(self):
        for i in range(self.n_blocks):

            self.network.add_module(
                'res-{}'.format(i+1),
                ResBlock(self.input_shape, self.kernel_size, clip=self.clip,
                         clip_val=self.clip_val))

    def forward(self, x):
        return self.network(x)

