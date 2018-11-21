import numpy as np
import torch
from torchvision import transforms


class ChainTransformations:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x, field, z, stats):
        for t in self.transformations:
            x = t(x, field, z, stats)
        return x



def create_fcs(k_values, scale, shift):
    def transform(x, field, z=None, stats=None):
        k = k_values[field]
        return (scale*x)/(x+k) + shift

    def inv_transform(x, field, z=None, stats=None):
        k = k_values[field]
        return k*(shift - x)/(x-shift-scale)

    return transform, inv_transform



class XTF(object):
    def __init__(self, mean, k, inverse=False, totorch=False):
        self.k = k
        self.mean = mean
        self.inverse = inverse
        self.totorch = totorch

    def func(self, img):
        field_transform = lambda d: np.piecewise(d, [d > 0,], [lambda x: np.tanh(np.log(x)/self.k), lambda x: -1])
        return field_transform(img)

    def inv_func(self, img):
        inv_field_transforms = lambda d: np.piecewise(d, [d > -1,], [lambda x: np.exp(np.arctanh(x)*self.k), lambda x: 0])
        return inv_field_transforms(img)

    def __call__(self, img):
        if self.totorch==False:
            if self.inverse == False:
                return self.func(img)
            elif self.inverse ==True:
                return self.inv_func(img)

        elif self.totorch==True:
            if self.inverse == False:
                toret = transforms.Compose(
                    [lambda x: self.func(x),
                     torch.from_numpy]
                )
                img = toret(img)
                return img
            elif self.inverse == True:
                toret = transforms.Compose(
                    [lambda x: self.inv_func(x),
                     torch.from_numpy]
                )
                img = (img)
                return img


class FCS(object):
    def __init__(self, k, inverse=False, mean=None, totorch=False, scale=2, shift=-1):
        self.k = k
        self.inverse = inverse
        self.totorch= totorch
        self.scale = scale
        self.shift = shift

    def func(self, img):
        return ((self.scale*img)/(img+self.k) + self.shift)

    def inv_func(self, img):
        return (self.k*(self.shift -img)/(img-self.shift-self.scale))

    def __call__(self, img):
        if self.totorch==False:
            if self.inverse == False:
                return self.func(img)
            elif self.inverse ==True:
                return self.inv_func(img)

        elif self.totorch==True:
            if self.inverse == False:
                toret = transforms.Compose(
                    [lambda x: self.func(x),
                     torch.from_numpy]
                )
                img = toret(img)
                return img
            elif self.inverse == True:
                toret = transforms.Compose(
                    [lambda x: self.inv_func(x),
                     torch.from_numpy]
                )
                img = (img)
                return img


def fast_cosmic_sim_func(img, k=4, inverse=False):
    if inverse == True:
        return ((-img*k - k)/(img-1))
    elif inverse == False:
        return ((2*img)/(img+k) - 1)


transform_fcs = transforms.Compose(
    [lambda x: fast_cosmic_sim_func(x),
     torch.from_numpy]
)

transform_fcs_quart = transforms.Compose(
    [lambda x: fast_cosmic_sim_func(x),
     torch.from_numpy,
     torch.nn.AvgPool2d(4)]
)

transform_fcs_half = transforms.Compose(
    [lambda x: fast_cosmic_sim_func(x),
     torch.from_numpy,
     torch.nn.AvgPool2d(2)]
)
