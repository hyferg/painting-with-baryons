import os
import torch
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
            print('g-adv: {} \ng-percep: {}'.format(adv_loss, percep_loss))
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

        if self.schedule['save_summary']['epochs']:

            self.save_path_individual = (self.schedule['save_dir'] + '/spec_power/')
            self.save_path_meta = (self.schedule['save_dir'] + '/meta_power/')
            self.save_path_loss = (self.schedule['save_dir'] + '/stat/')

            os.makedirs(self.save_path_individual, exist_ok=True)
            os.makedirs(self.save_path_meta, exist_ok=True)
            os.makedirs(self.save_path_loss, exist_ok=True)

        if self.schedule['save_img_interval']:
            os.makedirs(self.schedule['save_dir'] + '/imgs/', exist_ok=True)

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

    def save_if(self, gen_iterations):
        if self.schedule['save_model_interval']:
            if gen_iterations % self.schedule['save_model_interval'] == 0:
                self.Generator.save_self(self.schedule['save_dir'] + '/G_{}_giter.cpk'.format(gen_iterations))
                self.Discriminator.save_self(self.schedule['save_dir'] + '/D_{}_giter.cpk'.format(gen_iterations))

        if (self.schedule['save_img_interval'] |
            self.schedule['debug_plot']):
            if (gen_iterations % self.schedule['save_img_interval'] == 0 |
                self.schedule['debug_plot']):
                BahamasShow(
                    self.fake_imgs,
                    save=True,
                    save_path=(self.schedule['save_dir'] + '/imgs/{}_giter.png'.format(gen_iterations)))

                self.plot_wgp(
                    save=True,
                    save_path=(self.schedule['save_dir'] + 'stat_{}_giter.png'.format(gen_iterations)))


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
        self.set_criterion()
        self.g_decay = None
        self.d_decay = None
        super().__init__(schedule, Generator, Discriminator, dataloader,
                         device, testloader, **kwargs)

    def summary_save(self, epoch, idxs=None):
        FreezeModel(self.Generator)

        data = iter(self.testloader).next()
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
            save_path=(self.save_path_individual + '/s_{}_epoch.png'
                       .format(epoch))
        )

        MetaCompare(
            generator=self.Generator,
            inputs=imgs0,
            targets=imgs1,
            box_size=self.schedule['save_summary']['box_size'],
            transform=transforms,
            save=True,
            save_path=(self.save_path_meta + 'm_{}_epoch.png'.format(epoch))
        )
        LossShow(
            self.running_d_loss,
            self.running_g_loss,
            save=True,
            save_path=(self.save_path_loss + 'l_{}_epoch.png'.format(epoch))
        )
        UnFreezeModel(self.Generator)

    def summary_if(self, epoch):
        if self.schedule['debug_plot'] is True:
                self.summary_save(epoch)

        elif self.schedule['save_summary']:
            if (epoch in self.schedule['save_summary']['epochs'] and
                    self._epoch+1 in self.schedule['save_summary']['epochs']):
                self.summary_save(epoch)

        self._epoch = epoch

    def plot_if(self, gen_iterations, img_set, n=6):
        if (gen_iterations % self.schedule['sample_interval'] == 0 |
            self.schedule['debug_plot']
        ):
            if self.dataloader.name == 'bahamas_paired':
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

    def normal_iter(self):

        gen_iterations = 0

        for epoch in range(self.schedule['epochs']):
            data_iter = iter(self.dataloader)
            i = 0
            while i < len(self.dataloader):

                imgs0, imgs1 = iter(self.dataloader).next()
                i += 1

                imgs0      = imgs0.to(self.device)
                imgs1      = imgs1.to(self.device)
                imgs1_fake = self.Generator(imgs0)
                real_cat = torch.cat([imgs0, imgs1], dim=1)
                fake_cat = torch.cat([imgs0, imgs1_fake], dim=1)

                UnFreezeModel(self.Discriminator)

                # train D
                if gen_iterations % self.schedule['loss_params']['d_period'] == 0:

                    self.d_optimizer.zero_grad()

                    real_probs = self.Discriminator(real_cat)
                    fake_probs = self.Discriminator(fake_cat.detach())

                    real_loss = self.adv_loss(real_probs, torch.ones_like(real_probs, device=self.device))
                    fake_loss = self.adv_loss(fake_probs, torch.zeros_like(fake_probs, device=self.device))

                    d_loss = 0.5*(real_loss + fake_loss)

                    d_loss.backward()

                    self.running_d_loss.append(d_loss.data.cpu().numpy())

                    self.d_optimizer.step()


                # train G
                FreezeModel(self.Discriminator)

                self.g_optimizer.zero_grad()

                adv_probs = self.Discriminator(fake_cat)
                adv_loss = self.adv_loss(adv_probs, torch.ones_like(adv_probs, device=self.device))

                l1_loss = self.percep_loss(imgs1_fake, imgs1)
                l1_loss = l1_loss * self.schedule['loss_params']['lambda']

                g_loss = adv_loss + l1_loss

                g_loss.backward()


                self.running_g_loss.append(g_loss.data.cpu().numpy())

                self.g_optimizer.step()

                gen_iterations += 1


                self.clear_if(gen_iterations)

                self.text_if(gen_iterations,
                             epoch,
                             d_real=real_loss,
                             d_fake=fake_loss,
                             adv_loss=adv_loss.data.cpu().numpy(),
                             percep_loss=l1_loss.data.cpu().numpy())

                self.loss_if(gen_iterations,
                             self.running_d_loss,
                             self.running_g_loss,
                             med=True)

                self.plot_if(gen_iterations, [
                    imgs0,
                    imgs1,
                    imgs1_fake
                ], n=2)

                self.save_if(gen_iterations)

                self.summary_if(epoch)

    def wgp_iter(self):
        gen_iterations = 0
        self.epoch = 0
        for epoch in range(self.schedule['epochs']):

            self.g_decay.step()
            self.d_decay.step()

            self.epoch += 1
            i = 0
            while i < len(self.dataloader):

                UnFreezeModel(self.Discriminator)

                try:
                    self.schedule['warm_start']
                    if self.schedule['warm_start'] is True:
                        if gen_iterations < 25 or gen_iterations % 500 == 0:
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

                    data = iter(self.dataloader).next()
                    imgs0, imgs1 = data[0][0], data[0][1]
                    i += 1

                    imgs0      = imgs0.to(self.device)
                    imgs1      = imgs1.to(self.device)
                    imgs1_fake = self.Generator(imgs0)
                    real_cat = torch.cat([imgs0, imgs1], dim=1)
                    fake_cat = torch.cat([imgs0, imgs1_fake], dim=1).detach()

                    batch_size = imgs0.size()[0]
                    self.d_optimizer.zero_grad()

                    # Real Loss Backward
                    real_probs = self.Discriminator(real_cat)
                    d_loss_real = -real_probs.mean()
                    _d_loss_real.append(d_loss_real.data.cpu().numpy())
                    d_loss_real.backward()

                    # Fake Loss Backward
                    fake_probs = self.Discriminator(fake_cat)
                    d_loss_fake = fake_probs.mean()
                    _d_loss_fake.append(d_loss_fake.data.cpu().numpy())
                    d_loss_fake.backward()

                    # Gradient Penalty Backwards
                    epsilon = torch.rand(
                        (batch_size, 1, 1, 1),
                        device=self.device)

                    x_hat = epsilon*real_cat.data + (1-epsilon)*fake_cat.data
                    x_hat.requires_grad_(True)

                    d_interpolates = self.Discriminator(x_hat)

                    placeholder = torch.ones(
                        d_interpolates.size(),
                        device=self.device)

                    gradients = torch.autograd.grad(outputs=d_interpolates,
                                                    inputs=x_hat,
                                                    grad_outputs=placeholder,
                                                    create_graph=True,
                                                    retain_graph=True,
                                                    only_inputs=True)[0]

                    gp = (
                        ((gradients.norm(p=2, dim=1) - 1.) ** 2).mean() *
                        self.schedule['loss_params']['grad_lambda']
                    )
                    gp.backward()

                    self.d_loss = (d_loss_real + d_loss_fake).detach()
                    _running_d_loss.append(-self.d_loss.cpu().numpy())

                    self.d_optimizer.step()

                    if self.schedule['debug_plot']:
                        print('d train done')
                        break



                # save info
                self.running_d_loss.append(np.mean(_running_d_loss))
                _d_loss_real = np.mean(_d_loss_real)
                _d_loss_fake = np.mean(_d_loss_fake)
                # train G

                FreezeModel(self.Discriminator)

                self.g_optimizer.zero_grad()

                data = iter(self.dataloader).next()
                imgs0, imgs1 = data[0][0], data[0][1]
                i += 1

                imgs0      = imgs0.to(self.device)
                imgs1      = imgs1.to(self.device)
                imgs1_fake = self.Generator(imgs0)
                fake_cat = torch.cat([imgs0, imgs1_fake], dim=1)

                self.g_loss = -self.Discriminator(fake_cat).mean()

                l1_loss = self.percep_loss(imgs1_fake, imgs1)
                l1_loss = l1_loss * self.schedule['loss_params']['l1_lambda']

                combined_loss = self.g_loss + l1_loss
                combined_loss.backward()

                self.g_optimizer.step()

                gen_iterations += 1

                self.running_g_loss.append(
                    self.g_loss.data.cpu().numpy() + l1_loss.data.cpu().numpy()
                )

                if self.schedule['debug_plot']:
                    print('g train done')

                self.clear_if(gen_iterations)

                self.text_if(gen_iterations,
                             self.epoch,
                             d_real=_d_loss_real,
                             d_fake=_d_loss_fake,
                             adv_loss=self.g_loss.data.cpu().numpy(),
                             percep_loss=l1_loss.data.cpu().numpy())

                self.loss_if(gen_iterations,
                             self.running_d_loss,
                             self.running_g_loss,
                             med=True)

                '''
                self.plot_if(gen_iterations, [
                    imgs0,
                    imgs1,
                    imgs1_fake
                ], n=2)
                '''

                self.save_if(gen_iterations)

                self.summary_if(self.epoch)


class Wass(GAN_Trainer):
    @staticmethod
    def factory(schedule, Generator, Discriminator, dataloader, device):
        if schedule['loss'] == 'wgangp':
            print(' -> Wasserstein Improved Trainer')
            return WassersteinGP(schedule, Generator, Discriminator, dataloader, device)
        if schedule['loss'] == 'wass':
            print(' -> Wasserstein Trainer')
            return Wasserstein(schedule, Generator, Discriminator, dataloader, device)


    def plot_w(self, save=False, save_path=None):
        fig = plt.figure(figsize=(16,16))

        ax1 = plt.subplot(2,2,1)
        ax1.set_title('D loss: {}'.format(self.running_d_loss[-1]))
        ax1plt = signal.medfilt(self.running_d_loss, 101)
        ax1.scatter(np.arange(len(ax1plt)), ax1plt, color='blue')

        ax2 = plt.subplot(2,2,2)
        ax2.set_title('G Loss: {}'.format(self.running_g_loss[-1]))
        ax2plt = signal.medfilt(self.running_g_loss, 101)
        ax2.scatter(np.arange(len(ax2plt)),ax2plt, color='red')

        ax3 = plt.subplot(2,2,3)
        ax3.set_title('D loss last 2k')
        ax3plt = ax1plt[-2000:]
        ax3.scatter(np.arange(len(ax3plt)), ax3plt, color='blue')

        ax4 = plt.subplot(2,2,4)
        ax4.set_title('G loss last 2k')
        ax4plt = ax2plt[-2000:]
        ax4.scatter(np.arange(len(ax4plt)), ax4plt, color='red')


        if not save==True:
            display.display(plt.gcf())
        if save==True:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()


class WassersteinGP(Wass):
    def __init__(self, schedule, Generator, Discriminator, dataloader, device):
        self.running_wd = []
        self.running_g_loss = []
        self.running_d_loss = []
        super().__init__(schedule, Generator, Discriminator, dataloader, device)

    def train(self):
        gen_iterations = 0
        for epoch in range(self.schedule['epochs']):
            data_iter = iter(self.dataloader)
            i = 0
            while i < len(self.dataloader):

                self.unfreeze_model(self.Discriminator)


                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    Diters = 25
                else:
                    Diters = self.schedule['loss_params']['n_critic']
                Diters = self.schedule['loss_params']['n_critic']

                j = 0
                _wd = []
                _running_d_loss = []

                # train D
                while j < Diters and i < len(self.dataloader):

                    j += 1

                    if self.dataloader.name == 'bahamas':
                        imgs_real = data_iter.next()
                    elif self.dataloader.name == 'mnist':
                        imgs_real, _ = data_iter.next()
                    i += 1

                    imgs_real = imgs_real.to(device=self.device)
                    batch_size = imgs_real.size()[0]
                    self.d_optimizer.zero_grad()

                    z = torch.randn(
                        (batch_size, self.schedule['latent_dim'], 1, 1),
                        device=self.device)

                    # Real Loss Backward
                    real_probs = self.Discriminator(imgs_real)
                    d_loss_real = -real_probs.mean()
                    d_loss_real.backward()


                    # Fake Loss Backward
                    fake_imgs = self.Generator(z).detach()
                    fake_probs = self.Discriminator(fake_imgs)
                    d_loss_fake = fake_probs.mean()
                    d_loss_fake.backward()


                    # Gradient Penalty Backwards
                    epsilon = torch.rand(
                        (batch_size, 1, 1, 1),
                        device=self.device)

                    x_hat = epsilon*imgs_real.data + (1-epsilon)*fake_imgs.data
                    x_hat.requires_grad_(True)

                    d_interpolates = self.Discriminator(x_hat)

                    placeholder = torch.ones(
                        d_interpolates.size(),
                        device=self.device)

                    gradients = torch.autograd.grad(outputs=d_interpolates,
                                                    inputs=x_hat,
                                                    grad_outputs=placeholder,
                                                    create_graph=True,
                                                    retain_graph=True,
                                                    only_inputs=True)[0]


                    gp = ((gradients.norm(p=2, dim=1) - 1.) ** 2).mean() * self.schedule['loss_params']['lambda']
                    gp.backward()


                    self.d_loss = (d_loss_real + d_loss_fake).detach()
                    _wd.append(-self.d_loss.cpu().numpy())
                    _running_d_loss.append(-self.d_loss.cpu().numpy())

                    self.d_optimizer.step()

                # save info

                # train G

                self.freeze_model(self.Discriminator)

                self.g_optimizer.zero_grad()

                z = torch.randn(
                    (self.schedule['batch_size'], self.schedule['latent_dim'], 1, 1),
                    device=self.device)

                self.fake_imgs = self.Generator(z)
                self.g_loss = -self.Discriminator(self.fake_imgs).mean()
                self.g_loss.backward()

                self.g_optimizer.step()

                gen_iterations += 1

                self.running_wd.append(np.mean(_wd))
                self.running_d_loss.append(np.mean(_running_d_loss))
                self.running_g_loss.append(self.g_loss.data.cpu().numpy())

                if gen_iterations % self.schedule['sample_interval'] == 0:
                    display.clear_output(wait=True)
                    print('epoch: {}/{}'.format(epoch, self.schedule['epochs']))
                    if self.dataloader.name == 'bahamas':
                        BahamasShow(self.fake_imgs)
                    elif self.dataloader.name == 'mnist':
                        MnistShow(self.fake_imgs)

                    self.plot_wgp()


                self.save_if(gen_iterations)


class Wasserstein(Wass):
    def __init__(self, schedule, Generator, Discriminator, dataloader, device):
        self.running_wd = []
        self.running_g_loss = []
        self.running_d_loss = []
        super().__init__(schedule, Generator, Discriminator, dataloader, device)

    def train(self):
        gen_iterations = 0
        for epoch in range(self.schedule['epochs']):
            data_iter = iter(self.dataloader)
            i = 0
            while i < len(self.dataloader):

                self.unfreeze_model(self.Discriminator)

                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    Diters = 100
                else:
                    Diters = self.schedule['loss_params']['n_critic']
                j = 0
                _wd = []
                _running_d_loss = []

                # train D
                while j < Diters and i < len(self.dataloader):

                    j += 1

                    for param in self.Discriminator.parameters():
                        param.data.clamp_(-self.schedule['loss_params']['clamp'],
                                          self.schedule['loss_params']['clamp'])

                    if self.dataloader.name == 'bahamas':
                        imgs_real = data_iter.next()
                    elif self.dataloader.name == 'mnist':
                        imgs_real, _ = data_iter.next()
                    i += 1

                    imgs_real = imgs_real.to(device=self.device)
                    batch_size = imgs_real.size()[0]
                    self.d_optimizer.zero_grad()

                    z = torch.randn(
                        (batch_size, self.schedule['latent_dim'], 1, 1),
                        device=self.device)

                    real_probs = self.Discriminator(imgs_real)
                    d_loss_real = real_probs.mean()
                    d_loss_real.backward()


                    self.fake_imgs = self.Generator(z).detach()
                    fake_probs = self.Discriminator(self.fake_imgs)
                    d_loss_fake = -fake_probs.mean()
                    d_loss_fake.backward()

                    self.d_loss = (d_loss_real + d_loss_fake).detach()
                    _wd.append(-self.d_loss.cpu().numpy())
                    _running_d_loss.append(-self.d_loss.cpu().numpy())

                    self.d_optimizer.step()

                # save info

                # train G

                self.freeze_model(self.Discriminator)

                self.g_optimizer.zero_grad()

                z = torch.randn(
                    (self.schedule['batch_size'], self.schedule['latent_dim'], 1, 1),
                    device=self.device)

                fake_imgs = self.Generator(z)
                self.g_loss = self.Discriminator(fake_imgs).mean()
                self.g_loss.backward()

                self.g_optimizer.step()

                gen_iterations += 1

                self.running_wd.append(np.mean(_wd))
                self.running_d_loss.append(np.mean(_running_d_loss))
                self.running_g_loss.append(self.g_loss.data.cpu().numpy())

                if gen_iterations % self.schedule['sample_interval'] == 0:
                    display.clear_output(wait=True)
                    if self.dataloader.name == 'bahamas':
                        BahamasShow(self.fake_imgs)
                    elif self.dataloader.name == 'mnist':
                        MnistShow(self.fake_imgs)

                    self.plot_wgp()


                self.save_if(gen_iterations)


class Normal(GAN_Trainer):
    '''
    This method trains input G, D with standard D_{loss} and
    'heuristic' non-saturating G_{loss} as explicitly defined in
    section 3.2 of...

    NIPS 2016 Tutorial:
    Generative Adversarial Networks
    Ian Goodfellow
    '''
    def __init__(self, schedule, Generator, Discriminator, device):
        super().__init__(schedule, Generator, Discriminator, dataloader, device)

    def train_iter(self):
        # Train D
        for i in range(0, self.schedule['loss_params']['n_critic']):
            self.imgs, _ = iter(self.dataloader).next()
            self.imgs = self.imgs.to(self.device)
            self.batch_size = self.imgs.size()[0]
            self.real_truth = torch.ones((self.batch_size, 1), device=self.device)
            self.fake_truth = torch.zeros((self.batch_size, 1), device=self.device)

            z = torch.randn((self.batch_size, self.schedule['latent_dim']),
                            dtype=torch.float, device=self.device)

            if niter == 0:
                self.print_param(z, 'z')
                self.print_param(self.imgs, 'imgs')
                self.print_param(self.real_truth, 'real_truth')
                self.print_param(self.fake_truth, 'fake_truth')
                self._imshow(make_grid(self.imgs[:5], nrow=5))

            self.d_optimizer.zero_grad()
            self.unfreeze_model(self.Discriminator)
            self.freeze_model(self.Generator)
            self.fake_imgs = self.Generator(z).detach()
            real_probs = self.Discriminator(self.imgs)
            fake_probs = self.Discriminator(self.fake_imgs)
            d_loss_real = torch.log(real_probs).mean()
            d_loss_fake = torch.log(
                torch.add(self.real_truth, -fake_probs)).mean()
            self.d_loss = -0.5*(d_loss_real + d_loss_fake)
            self.d_loss.backward()
            self.d_optimizer.step()

        # train G
        self.g_optimizer.zero_grad()
        self.unfreeze_model(self.Generator)
        self.freeze_model(self.Discriminator)
        z = torch.randn(
            (self.batch_size, self.schedule['latent_dim']),
            dtype=torch.float, device=self.device)
        fake_imgs = self.Generator(z)
        self.g_loss = -0.5*torch.log(self.Discriminator(fake_imgs)).mean()
        self.g_loss.backward()
        self.g_optimizer.step()
