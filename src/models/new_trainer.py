import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from src.models.spectral_iter import spectral_d_iter, spectral_g_iter
from IPython.display import clear_output
from src.tools.data import parse_data

class NewTrainer:
    def __init__(self, schedule, generator, discriminator,
                 train_loader, test_loader, device, **kwargs):
        self.schedule = schedule
        self.generator = generator
        self.discriminator = discriminator
        self.perceptual_loss = None
        if self.schedule['optimizer_params']['perceptual_loss_opts'] is not None:
            self.perceptual_loss = self.make_perceptual(
                **self.schedule['optimizer_params']['perceptual_loss_opts']
            )
        self.g_optim, self.g_decay = self.make_optim(
            self.generator,
            schedule['optimizer_params']['g'])
        self.d_optim, self.d_decay = self.make_optim(
            self.discriminator,
            schedule['optimizer_params']['d'])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.g_loss_adv = []
        self.g_loss_percep = []
        self.d_loss = []

    @staticmethod
    def factory(strategy):
        if strategy['schedule']['type'] == 'spectral':
            return Spectral(strategy)

    def dump_data(self):
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

    def save_parts(self, epoch, n_iter):
        with torch.no_grad():
            self.generator.eval()
            self.generator.save_self(
                self.schedule['save_dir'] + f'/parts/g_{epoch}_{n_iter}_iter.cp'
            )

    def get_test_batch(self, inputs, targets, generator):
        with torch.no_grad():
            generator.eval()
            fake_targets = torch.zeros_like(targets)
            for i, img in enumerate(inputs):
                img = img.view(1, *img.shape)
                fake_img = generator(img)
                fake_targets[i] = fake_img
            inputs = inputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            fake_targets = fake_targets.detach().cpu().numpy()
            generator.train()

        return inputs, targets, fake_targets

    def validate(self, generator, test_iter, iterator_type, device):
        inputs, targets = parse_data(test_iter, device,
                                     iterator_type)

        inputs, targets, fake_targets = self.get_test_batch(
            inputs, targets, generator)

        self.validate_show(inputs, targets, fake_targets)

    def validate_show(self, inputs, targets, fake_targets, n=2, fig_width=8):
        if inputs.shape[1] == 2:
            inputs = inputs[:,0,:,:]
        images = [inputs, targets, fake_targets]
        fig, axs = plt.subplots(n, 3)
        for i in range(n):
            for j in range(3):
                ax = axs[i][j]
                img = ax.imshow(images[j][i].squeeze(), vmin=-1, vmax=1)
                ax.set_axis_off()
                fig.colorbar(img, ax=ax)
        fig.set_size_inches(fig_width, fig_width/3*n)

        fig, axs = plt.subplots(1,1)
        axs.set_title(f'Pixel Distribution from {len(inputs)} Samples')
        axs.hist(targets.flatten(), alpha=0.5, color='blue', bins=50,
                 label='Targets')
        axs.hist(fake_targets.flatten(), alpha=0.5, color='red', bins=50,
                 label='Fake Targets')
        axs.set_xlim([-2, 2])
        axs.set_yscale('log', nonposy='clip')
        fig.set_size_inches(fig_width, fig_width//2)


        if any(self.g_loss_adv):
            fig = self.loss_show(self.g_loss_adv, self.g_loss_percep, self.d_loss)
            fig.set_size_inches(fig_width, fig_width)
            fig.tight_layout()

        plt.show()

    def loss_show(self, g_loss_adv, g_loss_percep, d_loss, median_window=21):
        fig, axs = plt.subplots(2, 2)

        g_loss = np.add(g_loss_adv, g_loss_percep)
        axs[0][0].set_title(f'G Loss Adv [{g_loss_adv[-1]}]')
        axs[0][0].scatter(np.arange(len(g_loss)),
                          signal.medfilt(g_loss_adv, median_window))

        axs[0][1].set_title(f'G Loss Percep [{g_loss_percep[-1]}]')
        axs[0][1].scatter(np.arange(len(g_loss)),
                          signal.medfilt(g_loss_percep, median_window))

        axs[1][0].set_title(f'G Loss [{g_loss[-1]}]')
        axs[1][0].scatter(np.arange(len(g_loss)),
                          signal.medfilt(g_loss, median_window))


        axs[1][1].set_title(f'D Loss [{d_loss[-1]}]')
        axs[1][1].scatter(np.arange(len(d_loss)),
                          signal.medfilt(d_loss, median_window))

        return fig

    def make_optim(self, model, params):
        if params['type'] is 'adam':
            optimizer = optim.Adam(model.parameters(), **params['opts'])
            decay = params['decay']['obj'](optimizer, **params['decay']['opts'])
            return optimizer, decay

    def make_perceptual(self, type, percep_lambda):
        if type is 'l1':
            return lambda x, x_fake: percep_lambda * 0.5*torch.abs(x - x_fake).mean()
        else:
            raise NotImplementedError

    def validate_save():
        raise NotImplementedError

    def train():
        raise NotImplementedError


class Spectral(NewTrainer):
    def __init__(self, strategy):
        super(Spectral, self).__init__(**strategy)

    def pseudo_epoch(self, iteration, pseudo_epoch_iterations):
        if (iteration % pseudo_epoch_iterations is 0) and iteration is not 0:
            return True
        else:
            return False

    def train(self):
        generator = self.generator
        discriminator = self.discriminator
        g_optim = self.g_optim
        d_optim = self.d_optim
        g_decay = self.g_decay
        d_decay = self.d_decay
        iterator_type = self.schedule['iterator_type']
        device = self.device
        epoch = 0
        epochs = self.schedule['epochs']
        pseudo_epoch_iters = self.schedule['pseudo_epoch_iters']
        train_loader = self.train_loader
        train_loader_len = len(self.train_loader)
        test_loader = self.train_loader
        perceptual_loss = self.perceptual_loss

        data_iters = 0

        for epoch in range(epochs):
            i = 0
            while True:
                if data_iters >= train_loader_len or data_iters is 0:
                    train_iter = iter(train_loader)
                    test_iter = iter(test_loader)
                    self.validate(self.generator, test_iter, iterator_type, device)
                    self.save_parts(epoch, i)
                    data_iters = 0

                epoch_break = self.pseudo_epoch(i, pseudo_epoch_iters)
                if epoch_break:
                    g_decay.step()
                    d_decay.step()
                    break

                inputs, targets = parse_data(train_iter, device,
                                                  iterator_type)

                d_loss = spectral_d_iter(inputs, targets,
                                         generator, discriminator, d_optim)

                inputs, targets = parse_data(train_iter, device,
                                                  iterator_type)

                g_loss_adv, g_loss_percep = spectral_g_iter(
                    inputs, targets,
                    generator, discriminator, g_optim,
                    perceptual_loss)

                i += 1
                data_iters += 2

                self.g_loss_adv.append(g_loss_adv)
                self.g_loss_percep.append(g_loss_percep)
                self.d_loss.append(d_loss)
                if i%5 is 0:
                    print(f'e{epoch}i{i}')
                if i%101 is 0 or (i in [1, 2, 3, 4, 5, 10, 20 , 50, 75, 125] and epoch is 0):
                    os.system('clear')
                    clear_output()

                    g_lr = g_optim.param_groups[0]['lr']
                    d_lr = d_optim.param_groups[0]['lr']
                    print(f'g_lr {g_lr}')
                    print(f'd_lr {d_lr}')

                    self.validate(self.generator, test_iter, iterator_type, device)
                if i%95 is 0:
                    self.save_parts(epoch, i)




