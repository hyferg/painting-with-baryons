import torch
import matplotlib.pyplot as plt
from spectral_iter import spectral_d_iter, spectral_g_iter


class NewTrainer:
    def __init__(self, schedule, generator, discriminator, device):
        self.schedule = schedule
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

    @staticmethod
    def factory(strategy):
        if strategy['schedule']['type'] == 'spectral':
            return Spectral(**strategy)

    def get_test_batch(inputs, targets, generator):
        with torch.no_grad():
            generator.eval()
            fake_targets = torch.zeros_like(inputs)
            for i, img in enumerate(inputs):
                fake_img = generator(img.view(1, *img.shape))
                fake_targets[i] = fake_img
            inputs = inputs.cpu().numpy()
            targets = inputs.cpu().numpy()
            fake_targets = inputs.cpu().numpy()
            generator.train()

        return inputs, targets, fake_targets

    def parse_data(self, iterator, device, iterator_type='troster'):
        if iterator_type is 'troster':
            data = iterator.next()
            data = data.to(device)
            inputs = data[0][0]
            targets = data[0][1]

        else:
            raise NotImplementedError

        return inputs, targets

    def validate_show(inputs, targets, fake_targets):
        images = [inputs, targets, fake_targets]
        num_images = len(inputs)
        fig, axs = plt.subplots(num_images, 3)
        for i in range(num_images):
            for j in range(3):
                axs[i][j].imshow(images[j][i].squeeze())

        raise NotImplementedError

    def validate_save():
        raise NotImplementedError

    def train():
        raise NotImplementedError


class Spectral():
    def __init__(self, schedule, generator, discriminator, device):
        super(Spectral, self).__init__(
            schedule, generator, discriminator, device
        )

    def validate(self, generator, test_iter, iterator_type, device):
        inputs, targets = self.parse_data(test_iter, device,
                                          iterator_type)

        inputs, targets, fake_targets = self.get_test_batch(inputs, generator)

        self.validate_show(inputs, targets, fake_targets)

    def train(self):
        iterator_type = self.schedule['iterator_type']
        device = self.device
        epoch = 0
        epochs = self.schedule['epochs']
        train_loader = self.train_loader
        test_loader = self.train_loader

        for epoch in range(epochs):
            train_iter = iter(train_loader)
            test_iter = iter(test_loader)
            for i in range(len(train_iter)):
                inputs, targets = self.parse_data(train_iter, device,
                                                  iterator_type)

                spectral_d_iter(inputs, targets)

                inputs, targets = self.parse_data(train_iter, device,
                                                  iterator_type)

                spectral_g_iter(inputs, targets)

                self.validate(self.generator, test_iter, iterator_type, device)

