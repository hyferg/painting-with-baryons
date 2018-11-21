import torch
import torch.nn as nn
from collections import OrderedDict
from src.models.layers.residual import ResBlocks

class Network(nn.Module):
    @staticmethod
    def factory(network_structure):
        print('Creating...')
        _dcgan_types = [
            'dcgan_generator', 'dcgan_discriminator', 'patchgan_discriminator'
        ]
        _vanilla_types = ['vanilla_generator', 'vanilla_discriminator']
        _translator_types = ['resnet_translator']
        if (network_structure['type'] in _dcgan_types or
            network_structure['type'] in _translator_types):
            network = CNN.factory(network_structure)
        elif (network_structure['type'] in _vanilla_types):
            if network_structure['type'] == 'vanilla_generator':
                print(' -> Vanilla Generator')
                return VanillaGenerator(network_structure)
            elif network_structure['type'] == 'vanilla_discriminator':
                print(' -> Vanilla Discriminator')
                return VanillaDiscriminator(network_structure)
        else: raise Exception('({}) is not a known network structure'.format(
                network_structure['type']))

        return network


    def __repr__(self):
        layers = ''
        for param in self.parameters():
            layers += (str(param.data.device) + ' [req grad: %s]' % param.requires_grad + str(param.size()) + '\n')
        to_append = (layers)
        super_modules = super().__repr__()
        return('\n===\n\n{}\n\n{}\n==='.format(super_modules, to_append))


    def _flat_dim(self, tensor_size):
        val = 1
        for dim in tensor_size:
            val *= dim
        return val


    def save_self(self, PATH):
        torch.save(self.state_dict(), PATH)


    def load_self(self, PATH, force_cpu=False):
        if force_cpu:
            self.load_state_dict(
                torch.load(
                    PATH, map_location=lambda storage, location: storage
                ),
            )
        else:
            self.load_state_dict(torch.load(PATH))


    def _build_linear_stack(self, layer_stack):
        _layer_stack = OrderedDict()
        for i, layer in enumerate(layer_stack\
                                  ['layers']):
            _layer = []
            _out_features=(layer['out_features'] if layer['out_features'] else
                           layer_stack['output']
            )
            _layer.append(
                nn.Linear(
                    in_features=(layer_stack\
                                 ['input'] if i == 0 else
                                 layer_stack\
                                 ['layers'][i-1]['out_features']),
                    out_features=(_out_features),
                    bias=layer['bias']
                )
            )

            if layer['pre_batch_norm']:
                _layer.append(nn.BatchNorm1d(_out_features))

            if layer['activation']:
                _layer.append(layer['activation'])

            if layer['post_batch_norm']:
                _layer.append(nn.BatchNorm1d(_out_features))

            _layer_stack['layer_%d' % (i+1)] = nn.Sequential(*_layer)

        return nn.Sequential(_layer_stack)


class CNN(Network):
    @staticmethod
    def factory(network_structure):
        if network_structure['type'] == 'dcgan_generator':
            print(' -> DCGAN Generator')
            return DcganGenerator(network_structure)

        elif network_structure['type'] == 'dcgan_discriminator':
            print(' -> DCGAN Discriminator')
            return DcganDiscriminator(network_structure)

        elif network_structure['type'] == 'resnet_translator':
            print(' -> ResNet Translator')
            return ResNet(network_structure)

        elif network_structure['type'] == 'patchgan_discriminator':
            print(' -> PatchGAN Discriminator')
            return PatchGanDiscriminator(network_structure)


        else: raise Exception()


    def _build_conv_stack(self, conv_stack):
        _conv_stack = OrderedDict()
        for i, filter in enumerate(conv_stack\
                                   ['filters']):
            _filter = []
            if filter['type']:
                _opts = {
                    'in_channels': (conv_stack\
                                    ['input'][0] if i == 0 else
                                    conv_stack\
                                    ['filters'][i-1]['out_channels']
                    ),
                    'out_channels': filter['out_channels'],
                    'kernel_size': filter['kernel_size'],
                    'stride': filter['stride'],
                    'padding': filter['padding'],
                    'bias': filter['bias']
                }

            try: _opts['output_padding'] = filter['output_padding']
            except: pass

            if filter['type'] == 'normal':
                _filter.append(
                    nn.Conv2d(**_opts)
                )

            if filter['type'] == 'transpose':
                _filter.append(
                    nn.ConvTranspose2d(**_opts)
                )

            if filter['pre_batch_norm']:
                _filter.append(filter['pre_batch_norm'])

            if filter['activation']:
                _filter.append(filter['activation'])

            if filter['post_batch_norm']:
                _filter.append(filter['post_batch_norm'])

            _conv_stack['filter_%d' % (i+1)] = nn.Sequential(*_filter)

        return nn.Sequential(_conv_stack)


class PatchGanDiscriminator(CNN):
    def __init__(self, network_structure):
        super().__init__()
        self.network_structure = network_structure
        self._build_network()


    def __repr__(self):
        to_prepend = ''
        super_modules = super().__repr__()
        return(super_modules)


    def _build_network(self):
        self.conv1 = self._build_conv_stack(self.network_structure['conv_stack'])


    def forward(self, z):
        batch_size = z.shape[0]
        z = self.conv1(z)
        return z.view((batch_size, -1)).mean(dim=1)


class ResNet(CNN):
    def __init__(self, network_structure):
        super().__init__()
        self.network_structure = network_structure
        self.schedule = None
        self.losses = {}
        self.niter = 0
        self.epoch = 0
        self._build_network()


    def __repr__(self):
        to_prepend = ''
        super_modules = super().__repr__()
        return(super_modules)


    def _build_network(self):
        self.encode = self._build_conv_stack(
            self.network_structure['encode_stack'])

        self.residual = ResBlocks(
            **self.network_structure['res_blocks']
        )

        self.decode = self._build_conv_stack(
            self.network_structure['decode_stack'])


    def forward(self, x):
        x = self.encode(x)
        x = self.residual(x)
        x = self.decode(x)
        return x


#TODO class MetaResNet(ResNet):

class DcganGenerator(CNN):
    '''
    INPUT dims: ((N,) + z)
    OUTPUT dims: ((N,) (img_channels, img_height, img_width))
    '''
    def __init__(self, network_structure):
        super().__init__()
        self.network_structure = network_structure
        self.schedule = None
        self.losses = {}
        self.niter = 0
        self.epoch = 0
        self._build_network()


    def __repr__(self):
        to_prepend = ''
        super_modules = super().__repr__()
        return(super_modules)


    def _build_network(self):
        self.conv1 = self._build_conv_stack(self.network_structure['conv_stack'])


    def forward(self, z):
        z = self.conv1(z)
        return z


class DcganDiscriminator(CNN):
    def __init__(self, network_structure):
        super().__init__()
        self.network_structure = network_structure
        self._build_network()


    def __repr__(self):
        to_prepend = ''
        super_modules = super().__repr__()
        return(super_modules)


    def _build_network(self):
        self.conv1 = self._build_conv_stack(self.network_structure['conv_stack'])


    def forward(self, x):
        x = self.conv1(x)
        return x.view(-1, 1).squeeze(1)


class VanillaGenerator(Network):
    '''
    INPUT dims: ((N,) + z)
    OUTPUT dims: ((N,) + (img_channels, img_height, img_width))
    '''
    def __init__(self, network_structure):
        super().__init__()
        self.network_structure = network_structure
        self._build_network()


    def __repr__(self):
        to_prepend = ''
        super_modules = super().__repr__()
        return(super_modules)


    def _build_network(self):
        self.fc1 = self._build_linear_stack(self.network_structure['linear_stack'])


    def forward(self, z):
        img = self.fc1(z)
        img = img.view(-1,
                       self.network_structure['image_dimensions'][0],
                       self.network_structure['image_dimensions'][1],
                       self.network_structure['image_dimensions'][2]
        )
        return img


class VanillaDiscriminator(Network):
    '''
    INPUT dims: ((N,) + (img_channels, img_height, img_width))
    OUTPUT dims: ((N,) + 1)
    '''
    def __init__(self, network_structure):
        super().__init__()
        self.network_structure = network_structure
        self._build_network()


    def __repr__(self):
        to_prepend = ''
        super_modules = super().__repr__()
        return(super_modules)


    def _build_network(self):
        self.fc1 = self._build_linear_stack(self.network_structure['linear_stack'])


    def forward(self, x):
        x = x.view(-1, self.network_structure['linear_stack']['input'])
        probs = self.fc1(x)
        return probs
