import torch

def iter_discriminator(Generator, Discriminator, imgs0, imgs1,
                       grad_lambda, device):
    """

    Returns:
       TODO
    """
    imgs1_fake = Generator(imgs0)
    real_cat = torch.cat([imgs0, imgs1], dim=1)
    fake_cat = torch.cat([imgs0, imgs1_fake], dim=1).detach()

    batch_size = imgs0.size()[0]

    # Real Loss Backward
    real_probs = Discriminator(real_cat)
    d_loss_real = -real_probs.mean()
    _d_loss_real = (d_loss_real.data.cpu().numpy())
    d_loss_real.backward()

    # Fake Loss Backward
    fake_probs = Discriminator(fake_cat)
    d_loss_fake = fake_probs.mean()
    _d_loss_fake = (d_loss_fake.data.cpu().numpy())
    d_loss_fake.backward()

    # Gradient Penalty Backwards
    epsilon = torch.rand(
        (batch_size, 1, 1, 1),
        device=device)

    x_hat = epsilon*real_cat.data + (1-epsilon)*fake_cat.data
    x_hat.requires_grad_(True)

    d_interpolates = Discriminator(x_hat)

    placeholder = torch.ones(
        d_interpolates.size(),
        device=device)

    gradients = torch.autograd.grad(outputs=d_interpolates,
                                    inputs=x_hat,
                                    grad_outputs=placeholder,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]

    gp = (
        ((gradients.norm(p=2, dim=1) - 1.) ** 2).mean() *
        grad_lambda
    )
    gp.backward()

    return _d_loss_real, _d_loss_fake
