import torch


def spectral_d_iter(inputs, targets, generator, discriminator, D_optim):
    with torch.no_grad():
        generator.eval()
        fake_targets = generator(inputs)
        generator.train()

    D_optim.zero_grad()

    D_in_real = torch.cat([inputs, targets], dim=1)
    D_in_fake = torch.cat([inputs, fake_targets], dim=1)

    D_out_real = discriminator(D_in_real)
    D_out_fake = discriminator(D_in_fake)

    D_loss_real = torch.log(D_out_real).mean()
    D_loss_fake = torch.log(1 - D_out_fake).mean()

    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()

    D_optim.step()


def spectral_g_iter(inputs, targets, generator, discriminator,
                    G_optim, perceptual_loss=None):

    G_optim.zero_grad()

    fake_targets = generator(inputs)

    D_in_fake = torch.cat([inputs, fake_targets], dim=1)

    D_out_fake = discriminator(D_in_fake)

    G_loss_fake = torch.log(D_out_fake).mean()

    if perceptual_loss:
        G_loss_fake += perceptual_loss(targets, fake_targets)

    G_loss_fake.backward()

    G_optim.step()
