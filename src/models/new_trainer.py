import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal as signal
from src.models.spectral_iter import spectral_d_iter, spectral_g_iter


class NewTrainer:
    def __init__(self, schedule, generator, discriminator,
                 train_loader, test_loader, device, **kwargs):
        self.schedule = schedule
        self.generator = generator
        self.discriminator = discriminator
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
        self.g_loss = []
        self.d_loss = []

    @staticmethod
    def factory(strategy):
        if strategy['schedule']['type'] == 'spectral':
            return Spectral(strategy)

    def get_test_batch(self, inputs, targets, generator):
        with torch.no_grad():
            generator.eval()
            fake_targets = torch.zeros_like(inputs)
            for i, img in enumerate(inputs):
                fake_img = generator(img.view(1, *img.shape))
                fake_targets[i] = fake_img
            inputs = inputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            fake_targets = fake_targets.detach().cpu().numpy()
            generator.train()

        return inputs, targets, fake_targets

    def parse_data(self, iterator, device, iterator_type='troster'):
        if iterator_type is 'troster':
            data = iterator.next()
            inputs = data[0][0].to(device)
            targets = data[0][1].to(device)

        else:
            raise NotImplementedError

        return inputs, targets

    def validate(self, generator, test_iter, iterator_type, device):
        inputs, targets = self.parse_data(test_iter, device,
                                          iterator_type)
        print(inputs.shape)

        inputs, targets, fake_targets = self.get_test_batch(
            inputs, targets, generator)

        [print(x.shape) for x in [inputs, targets, fake_targets]]
        self.validate_show(inputs, targets, fake_targets)

    def validate_show(self, inputs, targets, fake_targets, n=2, fig_width=4):
        images = [inputs, targets, fake_targets]
        fig, axs = plt.subplots(n, 3)
        for i in range(n):
            for j in range(3):
                ax = axs[i][j]
                img = ax.imshow(images[j][i].squeeze(), vmin=-1, vmax=1)
                ax.set_axis_off()
                fig.colorbar(img, ax=ax)
        fig.set_size_inches(fig_width, fig_width*n)


        self.loss_show(self.g_loss, self.d_loss)

        raise NotImplementedError

    def loss_show(self, g_loss, d_loss, median_window=21):
        fig, axs = plt.subplots(2, 1)

        axs[0].set_title('G Loss')
        axs[0].scatter(np.arange(len(g_loss)),
                       signal.medfilt(g_loss, median_window))

        axs[1].set_title('G Loss')
        axs[1].scatter(np.arange(len(d_loss)),
                       signal.medfilt(d_loss, median_window))

    def make_optim(self, model, params):
        if params['type'] is 'adam':
            optimizer = optim.Adam(model.parameters(), **params['opts'])
            decay = params['decay']['obj'](optimizer, **params['decay']['opts'])
            return optimizer, decay

    def validate_save():
        raise NotImplementedError

    def train():
        raise NotImplementedError


class Spectral(NewTrainer):
    def __init__(self, strategy):
        super(Spectral, self).__init__(**strategy)

    def train(self):
        generator = self.generator
        discriminator = self.discriminator
        g_optim = self.g_optim
        d_optim = self.d_optim
        iterator_type = self.schedule['iterator_type']
        device = self.device
        epoch = 0
        epochs = self.schedule['epochs']
        train_loader = self.train_loader
        test_loader = self.train_loader

        for epoch in range(epochs):
            # TODO decay iter
            train_iter = iter(train_loader)
            test_iter = iter(test_loader)
            for i in range(len(train_iter)):
                inputs, targets = self.parse_data(train_iter, device,
                                                  iterator_type)

                d_loss = spectral_d_iter(inputs, targets,
                                         generator, discriminator, d_optim)

                inputs, targets = self.parse_data(train_iter, device,
                                                  iterator_type)

                g_loss = spectral_g_iter(inputs, targets,
                                         generator, discriminator, g_optim)

                raise
                self.g_loss.append(g_loss)
                self.d_loss.append(d_loss)
                self.validate(self.generator, test_iter, iterator_type, device)

                raise
            raise

