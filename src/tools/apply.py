def batch_apply(Generator, imgs, batch_size):
    """
    Running inference on large tensors clogs up memory
    this is batch strategy to keep memeory free
    Args:
       Generator  -- nn.Module
       imgs       -- torch.Tensor dim (n, depth, x, y)
       batch_size -- how many images to run inference on at a time
    Returns:
       TODO
    """
    fake_imgs = imgs.clone().zero_()

    num_imgs = len(imgs[:, 0])

    for idx in range(num_imgs//batch_size):
        select = slice(idx*batch_size, (idx+1)*batch_size)
        fake_imgs[select] = Generator(imgs[select])

    if num_imgs % batch_size:
        residual = slice(num_imgs - num_imgs % batch_size,
                         num_imgs)
        fake_imgs[residual] = Generator(imgs[residual])

    return fake_imgs
