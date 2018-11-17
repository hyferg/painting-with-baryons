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


class Wass(GAN_Trainer):
    @staticmethod
    def factory(schedule, Generator, Discriminator, dataloader, device):
        if schedule['loss'] == 'wgangp':
            print(' -> Wasserstein Improved Trainer')
            return WassersteinGP(schedule, Generator, Discriminator, dataloader, device)
        if schedule['loss'] == 'wass':
            print(' -> Wasserstein Trainer')
            return Wasserstein(schedule, Generator, Discriminator, dataloader, device)


    def plot_wgp(self, save=False, save_path=None):
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
