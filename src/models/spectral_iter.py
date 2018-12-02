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

    D_loss = -0.5*(D_loss_real + D_loss_fake)

    D_loss.backward()

    D_optim.step()

    return float(D_loss.detach().cpu().numpy())


def spectral_g_iter(inputs, targets, generator, discriminator,
                    G_optim, perceptual_loss=None):
    g_loss_percep = None
    G_optim.zero_grad()

    fake_targets = generator(inputs)

    D_in_fake = torch.cat([inputs, fake_targets], dim=1)

    D_out_fake = discriminator(D_in_fake)

    G_loss_fake = -0.5*torch.log(D_out_fake).mean()
    g_loss_adv = float(G_loss_fake.detach().cpu().numpy())

    if perceptual_loss is not None:
        p_loss = perceptual_loss(targets, fake_targets)
        g_loss_percep = float(p_loss.detach().cpu().numpy())
        G_loss_fake += p_loss

    G_loss_fake.backward()

    G_optim.step()

    return g_loss_adv, g_loss_percep
