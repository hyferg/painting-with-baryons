import torch


def iter_generator(Generator, Discriminator, percep_loss,
                   imgs0, imgs1, l1_lambda):
    """
    Returns:
       TODO
    """
    imgs1_fake = Generator(imgs0)
    fake_cat = torch.cat([imgs0, imgs1_fake], dim=1)

    g_loss = -Discriminator(fake_cat).mean()

    l1_loss = percep_loss(imgs1_fake, imgs1)
    l1_loss = l1_loss * l1_lambda

    combined_loss = g_loss + l1_loss
    combined_loss.backward()

    return g_loss.detach().cpu().numpy(), l1_loss.detach().cpu().numpy()
