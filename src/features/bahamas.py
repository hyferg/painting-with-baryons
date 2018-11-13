from torch.utils.data import Dataset
import numpy as np
import pprint
import torch
import collections
from collections import OrderedDict
import time
import copy
import gc
import sys

class BahamasDatasetPaired(Dataset):
    def __init__(self,
                 sets=None,
                 grouping=None,
                 transform=None,
                 train_set=True,
                 ntest=None,
                 seed = 1010):


        self.train_set = train_set
        self.ntest = ntest

        pp = pprint.PrettyPrinter(depth=5)
        self.transform = transform
        self.info = {}
        self.slices = OrderedDict()
        for i, root in enumerate(sets):
            data = np.load(root)
            self.info['root_{}'.format(i)] = {}
            self.info['root_{}'.format(i)]['resolution']      = data['resolution']
            self.info['root_{}'.format(i)]['n_side']          = data['n_side']
            self.info['root_{}'.format(i)]['n_depth']         = data['n_depth']

            if self.transform:
                self.transform[i].mean = data['mean']

            self.info['root_{}'.format(i)]['shape_original']  = data['slices'].shape
            info = self.info['root_{}'.format(i)]
            self.slices['root_{}'.format(i)] = np.reshape(
                data['slices'],
                (-1, 1, info['shape_original'][4], info['shape_original'][5])
            )
            info['shape_final'] = self.slices['root_{}'.format(i)].shape
            info['n_samples'] = info['shape_final'][0]
            del data

        self.inv_transform = []
        if self.transform:
            for trans in self.transform:
                _trans = copy.copy(trans)
                _trans.inverse = True
                _trans.totorch= False
                self.inv_transform.append(_trans)


        if self.transform is not None:
            for i, key in enumerate(self.slices):
                print('transforming {} with mean {}'.format(key, self.transform[i].mean))
                self.slices[key] = self.transform[i](self.slices[key])

        self.test = OrderedDict()
        self.train = OrderedDict()

        self.merged_slices = OrderedDict()

        for i, group in enumerate(grouping):
            nsets = len(group)
            data_shape = self.info['root_' + str(group[0])]['shape_final']
            self.merged_slices['m_{}'.format(i)] = np.zeros(shape=(data_shape[0]*nsets,)+(data_shape[1:]), dtype='float32')

            for j, root_idx in enumerate(group):
                self.merged_slices['m_{}'.format(i)][j*data_shape[0]:(j+1)*data_shape[0]] = self.slices['root_'+str(root_idx)]
                del self.slices['root_'+str(root_idx)]

            self.info['m_{} shape'.format(i)] = self.merged_slices['m_{}'.format(i)].shape

        del self.slices

        np.random.seed(seed=seed)
        testIdx = np.random.randint(0,self.info['m_0 shape'][0], size=(self.ntest,))

        for i, key in enumerate(self.merged_slices):
            if not self.train_set:
                self.merged_slices[key] = np.take(self.merged_slices[key], testIdx, axis=0)
                self.info['m_{} test_shape'.format(i)] = self.merged_slices[key].shape

            elif self.train_set:
                self.merged_slices[key] = np.delete(self.merged_slices[key], testIdx, axis=0)
                self.info['m_{} train_shape'.format(i)] = self.merged_slices[key].shape


        # end sample test set

        print('\n=== data ===\n')
        pp.pprint(self.info)


    def __len__(self):
        if self.train_set == True:
            return int(self.info['m_0 train_shape'][0])
        elif self.train_set == False:
            return int(self.info['m_0 test_shape'][0])

    def __getitem__(self, idx):
        imgs = []
        for i, key in enumerate(self.merged_slices):
            imgs.append(self.merged_slices[key][idx])
        return imgs


def BahamasLoaderPaired(sets,
                        batch_size,
                        transform,
                        ntest=10,
                        train_set=True,
                        grouping=[
                            [0],
                            [1]
                        ]
):

    sim_data = BahamasDatasetPaired(
        sets,
        grouping=grouping,
        transform=transform,
        ntest=ntest,
        train_set=train_set)

    if train_set:
        loader = torch.utils.data.DataLoader(
            sim_data, batch_size=batch_size, shuffle=True)
    elif not train_set:
        loader = torch.utils.data.DataLoader(
            sim_data, batch_size=ntest, shuffle=False)

    loader.name = 'bahamas_paired'
    loader.inv_transform = sim_data.inv_transform
    return loader


class BahamasDataset(Dataset):
    def __init__(self, root=None, transform=None):
        self.transform = transform
        data = np.load(root)
        self.info = {}
        self.info['resolution'] = data['resolution']
        self.info['n_side'] = data['n_side']
        self.info['n_depth'] = data['n_depth']
        self.info['mean'] = data['mean']
        self.info['shape_original'] = data['slices'].shape

        self.slices = data['slices']
        self.slices = np.reshape(self.slices, (-1, 1, self.info['shape_original'][4], self.info['shape_original'][5]))

        self.info['shape_final'] = self.slices.shape
        self.info['n_samples'] = self.info['shape_final'][0]

        pp = pprint.PrettyPrinter(depth=5)
        print('\n=== data ===\n')
        pp.pprint(self.info)


    def __len__(self):
        return self.info['n_samples']

    def __getitem__(self, idx):
        img = self.slices[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img


def BahamasLoader(PATH, batch_size, transform):
    sim_data = BahamasDataset(PATH, transform=transform)
    loader = torch.utils.data.DataLoader(sim_data, batch_size=batch_size, shuffle=True)
    loader.name = 'bahamas'
    return loader
