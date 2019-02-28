import numpy as np
import torch
from torchvision import transforms

def create_range_compress_transforms(k_values, modes="log", scale=1, delta=0):
    def interpolate_z(stats, z):
        """Interpolate statisitcs dict to redshift z."""
        z_list = list(stats.keys())
        idx = np.searchsorted(z_list, z, side="right")
        if idx >= len(z_list):
            return stats[z_list[-1]]
        elif idx <= 0:
            return stats[z_list[0]]
        w = (z - z_list[idx-1])/(z_list[idx]-z_list[idx-1])
        stats_names = stats[z_list[0]].keys()
        intp_stats = {s : w*stats[z_list[idx]][s] + (1-w)*stats[z_list[idx-1]][s] for s in stats_names}

        return intp_stats

    def transform(x, field, z, stats):
        k = k_values[field]
        mode = modes[field]
        mean = np.sqrt(interpolate_z(stats[field], z)["mean"])
        std = np.sqrt(interpolate_z(stats[field], z)["var"])
        if mode.lower() == "log":
            return np.where(x > 0, np.tanh(np.log(x/std)/k)*scale - delta, -1)
        elif mode.lower() == "x/(1+x)":
            return np.where(x+mean*k>0, np.tanh(x/(x+mean*k)), -1)
        else:
             raise ValueError(f"Mode '{mode}' not supported.")               
    
    def inv_transform(x, field, z, stats):
        k = k_values[field]
        mode = modes[field]
        mean = np.sqrt(interpolate_z(stats[field], z)["mean"])
        std = np.sqrt(interpolate_z(stats[field], z)["var"])
        if mode.lower() == "log":
            return np.where(x > -1, np.exp(np.arctanh((x+delta)/scale)*k)*std, 0)
        elif mode.lower() == "x/(1+x)":
            return np.where(x > -1, mean*k/(1/np.arctanh(x)-1), -mean*k)
        else:
             raise ValueError(f"Mode '{mode}' not supported.")   
    
    return transform, inv_transform 



class ChainTransformations:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x, field, z, stats):
        for t in self.transformations:
            x = t(x, field, z, stats)
        return x



def create_fcs_clipped(k_values, scale, shift):
    def transform(x, field, z, stats):
        k = k_values[field]
        return (scale*x)/(x+k) + shift

    def inv_transform(x, field, z, stats):
        k = k_values[field]
        return k*(shift - x)/(x-shift-scale)

    return transform, inv_transform

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

def z_transform(shift):
    def func(z):
        return z - shift
    return func
