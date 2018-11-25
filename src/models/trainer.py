import os
import torch
import objgraph
import logging
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import scipy.signal as signal
from src.visualization.show import BahamasShow, MnistShow, MultiTargets, \
    LossShow
from src.visualization.new_compare import PowerCompare, MetaCompare
from src.tools.weights import FreezeModel, UnFreezeModel
from src.tools.memory import get_gpu_memory_map
from src.models.wgp_d_iter import iter_discriminator
from src.models.wgp_g_iter import iter_generator


class Trainer():
    @staticmethod
    def factory(schedule, Generator=None, Discriminator=None, dataloader=None, device=None, testloader=None, **kwargs):
        if device == None:
            raise Exception('Please set a device')
        _gan_types = ['vanilla_gan', 'dc_gan', 'translator']
        if schedule['type'] in _gan_types:
            if schedule['type'] == 'vanilla_gan':
                assert Generator.network_structure['type'] == 'vanilla_generator'
                assert Discriminator.network_structure['type'] == 'vanilla_discriminator'
                return GAN_Trainer.factory(Generator=Generator,
                                           Discriminator=Discriminator,
                                           schedule=schedule,
                                           dataloader=dataloader,
                                           device=device)
            elif schedule['type'] == 'dc_gan':
                assert Generator.network_structure['type'] == 'dcgan_generator'
                assert Discriminator.network_structure['type'] == 'dcgan_discriminator'
                return GAN_Trainer.factory(Generator=Generator,
                                           Discriminator=Discriminator,
                                           schedule=schedule,
                                           dataloader=dataloader,
                                           device=device)

            elif schedule['type'] == 'translator':
                assert Generator.network_structure['type'] == 'resnet_translator'
                assert Discriminator.network_structure['type'] in ['patchgan_discriminator', 'dcgan_discriminator']
                return GAN_Trainer.factory(Generator=Generator,
                                           Discriminator=Discriminator,
                                           schedule=schedule,
                                           dataloader=dataloader,
                                           device=device,
                                           testloader=testloader,
                                           **kwargs)
        else: raise Exception('{} is not a known trainer type'.format(schedule['type']))

    def print_param(self, param, name=None):
        to_print = (str(param.data.device) + ' [req grad: %s]' % param.requires_grad + str(param.size()))
        if name:
            to_print = (name + ': ' + to_print)
        print(to_print)

    def print_model(self, model):
        print(model.__class__.__name__)
        for param in model.parameters():
            self.print_param(param)
        print('')

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def text_if(self, gen_iterations, epoch, d_real, d_fake, adv_loss, percep_loss):
        if (gen_iterations % self.schedule['sample_interval'] == 0 |
            self.schedule['debug_plot']):

            print('epoch: {}/{}'.format(epoch, self.schedule['epochs']))
            print('d-real: {}\nd-fake: {}'.format(d_real, d_fake))
            print('g-adv: {} \ng-percep: {} [{}]'.format(adv_loss, percep_loss, percep_loss/self.schedule['loss_params']['l1_lambda']))
            for param_group in self.g_optimizer.param_groups:
                print('g lr {}'.format(param_group['lr']))
            for param_group in self.d_optimizer.param_groups:
                print('d lr {}'.format(param_group['lr']))


class GAN_Trainer(Trainer):
    @staticmethod
    def factory(schedule, Generator, Discriminator, dataloader, device, testloader, **kwargs):
        print('Creating...')
        if schedule['loss'] == 'normal':
            print(' -> Normal Trainer')
            return Normal(schedule, Generator, Discriminator, dataloader, device)

        elif schedule['loss'] == ['wgangp', 'wass']:
            return Wass.factory(schedule, Generator, Discriminator, dataloader, device, **kwargs)

        elif schedule['loss'] == 'l1_plus':
            print(' -> Translator Trainer')
            return Translator(schedule, Generator, Discriminator, dataloader, device, testloader, **kwargs)

        else: raise Exception('{} is unknown loss type'.format(schedule['loss']))


    def __init__(self, schedule, Generator, Discriminator, dataloader, device, testloader, dataset):
        self.device = device
        self.schedule = schedule

        self.Generator = Generator
        self.Generator.to(self.device)

        self.Discriminator = Discriminator
        self.Discriminator.to(self.device)

        self.schedule = schedule
        self.dataloader = dataloader

        self.testloader = testloader
        self.dataset = dataset

        self.pre_train()

    def pre_train(self):

        if self.schedule['save_dir']:
            os.makedirs(self.schedule['save_dir'], exist_ok=True)

        if self.schedule['save_summary']['iters']:

            self.save_path_individual = (self.schedule['save_dir'] + '/spec_power/')
            self.save_path_meta = (self.schedule['save_dir'] + '/meta_power/')
            self.save_path_loss = (self.schedule['save_dir'] + '/stat/')
            self.img_path = (self.schedule['save_dir'] + '/imgs/')

            os.makedirs(self.save_path_individual, exist_ok=True)
            os.makedirs(self.save_path_meta, exist_ok=True)
            os.makedirs(self.save_path_loss, exist_ok=True)
            os.makedirs(self.img_path, exist_ok=True)

        if self.schedule['g_optimizer'] == 'adam':
            print(' -> G optimizer: Adam')
            self.g_optimizer = self.make_adam(self.Generator, self.schedule['g_optim_opts'])

        elif self.schedule['g_optimizer'] == 'rms':
            print(' -> G optimizer: RMS Prop')
            self.g_optimizer = self.make_rms(self.Generator, self.schedule['g_optim_opts'])

        if self.schedule['d_optimizer'] == 'adam':
            print(' -> D optimizer: Adam')
            self.d_optimizer = self.make_adam(self.Discriminator, self.schedule['d_optim_opts'])

        elif self.schedule['d_optimizer'] == 'rms':
            print(' -> D optimizer: RMS Prop')
            self.d_optimizer = self.make_rms(self.Discriminator, self.schedule['d_optim_opts'])

        self.g_decay = self.schedule['g_decay']
        self.d_decay = self.schedule['d_decay']
        _opts = self.schedule['lrdecay_opts']
        self.g_decay = self.g_decay(self.g_optimizer, **_opts)
        self.d_decay = self.d_decay(self.d_optimizer, **_opts)
        print('decaying lr...')


        self.Generator.schedule = self.schedule
        self.Discriminator.schedule = self.schedule

        print('Running on {}'.format(self.device))

    def make_rms(self, model, opts):
        return optim.RMSprop(model.parameters(), **opts)

    def make_adam(self, model, opts):
        return optim.Adam(model.parameters(), **opts)

    def loss_if(self, gen_iterations, d_loss, g_loss, save=False, med=True, nmed=101):
        if (gen_iterations % self.schedule['sample_interval'] == 0 |
           self.schedule['debug_plot']):
            LossShow(d_loss, g_loss, save=save, med=med, nmed=nmed)

    def clear_if(self, gen_iterations):
        if (gen_iterations % self.schedule['sample_interval'] == 0 |
           self.schedule['debug_plot']):
            display.clear_output(wait=True)


class Translator(GAN_Trainer):
    def __init__(self, schedule, Generator, Discriminator, dataloader,
                 device, testloader, **kwargs):
        self.running_g_loss = []
        self.running_d_loss = []
        self._epoch = 0
        self._gen_iterations = 0
        self.set_criterion()
        self.g_decay = None
        self.d_decay = None
        super().__init__(schedule, Generator, Discriminator, dataloader,
                         device, testloader, **kwargs)

    def summary_save(self, n_iter, idxs=None):
        with torch.no_grad():
            self.Generator.eval()
            self.Generator.save_self(self.schedule['save_dir'] + '/parts/g_{}_iter.cp'.format(n_iter))
            self.Discriminator.save_self(self.schedule['save_dir'] + '/d_{}_iter.cp'.format(n_iter))
            data = next(self.test_iter)
            idxs = data[1]
            imgs0, imgs1 = data[0][0], data[0][1]
            imgs0      = imgs0.to(self.device)
            imgs1      = imgs1.to(self.device)

            transforms = [[], []]
            for idx in idxs:
                _transform, _inv_transform = self.dataset.get_inverse_transforms(idx)
                transforms[0].append(_transform)
                transforms[1].append(_inv_transform)

            PowerCompare(
                generator=self.Generator,
                inputs=imgs0,
                targets=imgs1,
                box_size=self.schedule['save_summary']['box_size'],
                n=self.schedule['save_summary']['n'],
                grid_size=self.schedule['save_summary']['grid_size'],
                transform=transforms,
                save=True,
                save_path=(self.save_path_individual + '/s_{}_n_iter.png'
                        .format(n_iter))
            )

            MetaCompare(
                generator=self.Generator,
                inputs=imgs0,
                targets=imgs1,
                box_size=self.schedule['save_summary']['box_size'],
                transform=transforms,
                save=True,
                save_path=(self.save_path_meta + 'm_{}_n_iter.png'.format(n_iter)),
                debug=True
            )
            if any(self.running_d_loss):
                LossShow(
                    self.running_d_loss,
                    self.running_g_loss,
                    save=True,
                    nmed=21,
                    save_path=(self.save_path_loss + 'l_{}_n_iter.png'.format(n_iter))
                )
            self.Generator.train()

    def summary_if_iter(self, n_iter, _n_iter):
        if self.schedule['debug_plot'] is True:
                self.summary_save(n_iter)

        elif self.schedule['save_summary']:
            if (n_iter in self.schedule['save_summary']['iters'] and
                    _n_iter+1 in self.schedule['save_summary']['iters']):
                self.summary_save(n_iter)

        self._gen_iterations = n_iter

    def plot_if(self, gen_iterations, img_set, n=6):
        if (gen_iterations % self.schedule['sample_interval'] == 0 |
            self.schedule['debug_plot']
        ):
            multi = MultiTargets(
                [img_set[0],
                    img_set[1],
                    img_set[2]], n=n)
            BahamasShow(multi, figsize=(16, 16))

    def set_criterion(self, use_l1=True, use_bce=True):
        if use_l1:
            self.percep_loss = nn.L1Loss()
        if use_bce:
            self.adv_loss = nn.BCELoss()

    def train_iter(self):
        subtype = None
        try:
            subtype = self.schedule['subtype']
        except KeyError:
            pass

        if subtype == 'normal':
            self.normal_iter()
        elif subtype == 'wgp':
            self.wgp_iter()
        else:
            self.normal_iter()

    def decay_if_iter(self, i):
        decay_cond = self.schedule['decay_iter']
        if (i % decay_cond == 0) and i is not 0:
            print('decaying')
            self.g_decay.step()
            self.d_decay.step()

    def check_networks(self, networks):
        meta_status = []
        all_meta = False
        bands = None
        for network in networks:
            if hasattr(network, 'type'):
                if network.type == 'meta-network':
                    meta_status.append(True)
                else:
                    meta_status.append(False)
            else:
                meta_status.append(False)

            if any(meta_status) is True:
                assert all(meta_status) is True
                assert all(x.bands == networks[0].bands for x in networks)
                all_meta = True
                bands = networks[0].bands

        return all_meta, bands

    def band_select(self, imgs, idx):
        return imgs[:, idx, :, :].unsqueeze(1)

    def break_if(self, n_iter):
        if 'iter_break' in self.schedule:
            if n_iter >= self.schedule['iter_break']:
                return True
        else:
            return False

    def wgp_iter(self):
        self.Generator.train()
        self.gen_iterations = 0
        self.epoch = 0
        all_meta, bands = self.check_networks(
            [self.Discriminator, self.Generator]
        )
        self.test_iter = iter(self.testloader)
        self.summary_save(0)
        for epoch in range(self.schedule['epochs']):
            if self.break_if(self.gen_iterations):
                break
            self.epoch += 1
            i = 0
            data_iter = iter(self.dataloader)
            while i < len(self.dataloader):

                print(self.gen_iterations)
                UnFreezeModel(self.Discriminator)

                self.decay_if_iter(self.gen_iterations)

                try:
                    self.schedule['warm_start']
                    if self.schedule['warm_start'] is True:
                        if self.gen_iterations < 25 or \
                           self.gen_iterations % 500 == 0:
                            Diters = 25
                        else:
                            Diters = self.schedule['loss_params']['n_critic']
                    else:
                        Diters = self.schedule['loss_params']['n_critic']
                except KeyError:
                    Diters = self.schedule['loss_params']['n_critic']

                j = 0
                _running_d_loss = []
                _d_loss_real = []
                _d_loss_fake = []

                # train D
                while j < Diters and i < len(self.dataloader):
                    j += 1

                    data = data_iter.next()
                    imgs0, imgs1 = data[0][0], data[0][1]
                    i += 1

                    imgs0 = imgs0.to(self.device)
                    imgs1 = imgs1.to(self.device)

                    self.d_optimizer.zero_grad()

                    if all_meta:
                        for i in range(bands):
                            input_feed = self.band_select(imgs0, i)
                            target_feed = self.band_select(imgs1, i)
                            d_loss_real, d_loss_fake, = iter_discriminator(
                                self.Generator.networks[i],
                                self.Discriminator.networks[i],
                                input_feed, target_feed,
                                self.schedule['loss_params']['grad_lambda'],
                                self.device)
                    else:
                        d_loss_real, d_loss_fake, = iter_discriminator(
                            self.Generator,
                            self.Discriminator,
                            imgs0, imgs1,
                            self.schedule['loss_params']['grad_lambda'],
                            self.device)

                    _d_loss_real.append(d_loss_real)
                    _d_loss_fake.append(d_loss_fake)
                    _running_d_loss.append(-(d_loss_real + d_loss_fake))

                    self.d_optimizer.step()

                # save info
                # train G

                FreezeModel(self.Discriminator)

                self.g_optimizer.zero_grad()

                data = data_iter.next()
                imgs0, imgs1 = data[0][0], data[0][1]
                i += 1

                imgs0 = imgs0.to(self.device)
                imgs1 = imgs1.to(self.device)

                if all_meta:
                    for i in range(bands):
                        input_feed = self.band_select(imgs0, i)
                        target_feed = self.band_select(imgs1, i)
                        g_loss, l1_loss = iter_generator(
                            self.Generator.networks[i],
                            self.Discriminator.networks[i],
                            self.percep_loss,
                            input_feed,
                            target_feed,
                            self.schedule['loss_params']['l1_lambda'])
                else:
                    g_loss, l1_loss = iter_generator(
                        self.Generator, self.Discriminator,
                        self.percep_loss, imgs0, imgs1,
                        self.schedule['loss_params']['l1_lambda'])

                self.g_optimizer.step()

                # Collect stats
                self.gen_iterations += 1
                _d_loss_real = np.mean(_d_loss_real)
                _d_loss_fake = np.mean(_d_loss_fake)
                self.running_d_loss.append(np.mean(_running_d_loss))
                self.running_g_loss.append(
                    g_loss + l1_loss
                )

                self.clear_if(self.gen_iterations)

                self.text_if(self.gen_iterations,
                             self.epoch,
                             d_real=_d_loss_real,
                             d_fake=_d_loss_fake,
                             adv_loss=g_loss,
                             percep_loss=l1_loss)

                self.loss_if(self.gen_iterations,
                             self.running_d_loss,
                             self.running_g_loss,
                             med=True)

                '''
                self.plot_if(self.gen_iterations, [
                    imgs0,
                    imgs1,
                    imgs1_fake
                ], n=2)
                '''

                self.summary_if_iter(self.gen_iterations, self._gen_iterations)

                torch.cuda.empty_cache()
                if self.break_if(self.gen_iterations):
                    break
