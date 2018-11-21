from src.features.transforms import fast_cosmic_sim_func
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from IPython import display
import torch
import scipy.signal as signal

def LossShow(d_loss, g_loss, save=False, med=True, nmed=101, save_path=None):
    fig = plt.figure(figsize=(16,16))

    ax1 = plt.subplot(2,2,1)
    ax1.set_title('D loss: {}'.format(d_loss[-1]))
    if med: ax1plt = signal.medfilt(d_loss, nmed)
    else: ax1plt = d_loss
    ax1.set_yscale('log')
    ax1.plot(np.arange(len(ax1plt)), ax1plt, color='blue', marker='o', linewidth=0)

    ax2 = plt.subplot(2,2,2)
    ax2.set_title('G Loss: {}'.format(g_loss[-1]))
    if med: ax2plt = signal.medfilt(g_loss, nmed)
    else: ax2plt = g_loss
    ax2.set_yscale('log')
    ax2.plot(np.arange(len(ax2plt)),ax2plt, color='red', marker='o', linewidth=0)

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


def MnistShow(imgs, nrow=6):
    fig = plt.figure(figsize=(16,16))
    imgs = imgs.detach()
    imgs = imgs[:12]
    if not imgs.device == 'cpu':
        imgs = imgs.cpu()
    imgs = imgs / 2 + 0.5
    grid = make_grid(imgs, nrow=nrow)
    grid = np.transpose(grid, (1,2,0))
    plt.subplot(1,1,1)
    imgplot = plt.imshow(grid)
    display.display(plt.gcf())
    plt.close()


def BahamasShow(imgs, nimg=5, nrow=5, save=False, save_path=None, figsize=(16,16)):
    fig = plt.figure(figsize=figsize)

    imgs = imgs.detach()
    imgs = imgs[:nimg]
    if not imgs.device == 'cpu':
        imgs = imgs.cpu()
    grid = make_grid(imgs, nrow=nrow, normalize=False, pad_value=1)
    img = np.transpose(grid, (1,2,0))
    img = img.squeeze()
    lum_img = img[:,:,0]
    plt.subplot(1,1,1)
    imgplot = plt.imshow(lum_img, vmin=-1, vmax=1)
    imgplot.set_cmap('plasma')
    plt.axis('off')
    if not save == True:
        display.display(plt.gcf())
    if save==True:
        plt.axis('off')
        display.display(plt.gcf())
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def TargetMerge(img_input, target_true, target_fake):
    img_input = torch.nn.functional.pad(img_input, (4,4,2,2), value=0)
    target_true = torch.nn.functional.pad(target_true, (4,4,2,2), value=0)
    target_fake = torch.nn.functional.pad(target_fake, (4,4,2,2), value=0)
    img = torch.cat([img_input, target_true, target_fake], dim=0)
    return img.view((1, img.shape[0], img.shape[1]))


def MultiTargets(img_set, n=6):
    merge = torch.Tensor()
    for i, imgs in enumerate(img_set):
        if not imgs.device == 'cpu':
            img_set[i] = imgs.cpu()
    for i in range(0, n):
        _merge = TargetMerge(
            img_set[0][i][0],
            img_set[1][i][0],
            img_set[2][i][0],
        )
        merge = torch.cat((merge, _merge), dim=0)
    return merge.view((merge.shape[0], 1, merge.shape[1], merge.shape[2]))


def PixelDist(imgs, fromtorch=True, xlim=True, field=None):
    if fromtorch:
        imgs = imgs.cpu().numpy()
    n_samples = imgs.shape[0]
    imgs = imgs.flatten()
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(111)
    ax.set_title('Pixel Dist from {}, {} samples'.format(field, n_samples))
    if xlim:
        ax.set_xlim([-1, 1])
    ax.hist(imgs, bins=50)
    ax.set_yscale('log', nonposy='clip')
    fig.add_subplot(ax)


def PixelFrame(img, xlim=False):
    img = img.flatten()
    plt.title('Pixel Distribution')
    plt.yscale('log')
    plt.hist(img, bins=50)


def FracFrame(lines, base):
    plt.title('Auto Fractional')
    yfracs = []
    for line in lines:
        yfracs.append(
            line[1]/base - 1
        )
    x = line[0]
    plt.xscale('log')
    for i, yfrac in enumerate(yfracs):
        plt.plot(x, yfrac, label='root_{}'.format(i))
    plt.ylim([-1, 1])
    plt.legend()
