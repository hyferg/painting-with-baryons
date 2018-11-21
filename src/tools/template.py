import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.tools.boilerplate import files_info
from baryon_painter.painter import Painter
from src.models.network import Network
from baryon_painter.utils.datasets import BAHAMASDataset
import src.visualization.show as show
from torch.utils.data import DataLoader


class GAN_Painter(Painter):
    def __init__(self, parts_folder,
                 structure_file='/g_struc.pickle',
                 checkpoint_file='/g_weights.cp',
                 transform_file='/transform.pickle',
                 inv_transform_file='/inv_transform.pickle',
                 device='cuda:0',
                 img_dim=(1, 512, 512),
                 input_field='dm',
                 label_fields=['pressure'],
    ):

        with open(parts_folder + structure_file, 'rb') as handle:
            g_struc = pickle.load(handle)

        self.compute_device = device
        self.input_field = input_field
        self.label_fields = label_fields
        self.img_dim = img_dim
        self.transform = None
        self.inv_transform = None
        self.test_data = None

        self.generator = Network.factory(g_struc)
        self.load_state_from_file(parts_folder + checkpoint_file)

        if transform_file:
            with open(parts_folder + transform_file, 'rb') as handle:
                self.transform = pickle.load(handle)

        if inv_transform_file:
            with open(parts_folder + inv_transform_file, 'rb') as handle:
                self.inv_transform = pickle.load(handle)

    def load_state_from_file(self, filename):
        self.generator.load_self(filename)
        self.generator.to(self.compute_device)

    def paint(self, input, z=0.0, stats=None, inverse_transform=True):
        with torch.no_grad():
            self.generator.eval()
            if self.transform is not None:
                y = self.transform(
                    input, field=self.input_field, z=z, stats=stats)
            else:
                y = input
            y = y.reshape(1, *y.shape)
            y = torch.tensor(y, device=self.compute_device)
            prediction = self.generator(y).cpu().numpy()

        if inverse_transform and self.inv_transform is not None:
            return self.inv_transform(
                prediction, field=self.label_fields[0], z=z, stats=stats)
        else:
            return prediction

    def load_test_data(self, data_path, redshifts=[0.0, 0.5, 1.0]):
        _, test_files_info = files_info(data_path)
        label_fields = ["pressure"]

        test_dataset = BAHAMASDataset(test_files_info,
                                      root_path=data_path,
                                      redshifts=redshifts,
                                      label_fields=label_fields)

        self.test_iter = iter(test_dataset)

    def get_batch(self, batch_size):
        inputs = np.zeros((batch_size, *self.img_dim))
        outputs = np.zeros((batch_size, *self.img_dim))
        painted = np.zeros((batch_size, *self.img_dim))
        for i in range(batch_size):
            img, idx = next(self.test_iter)
            inputs[i] = img[0]
            outputs[i] = img[1]
            painted[i] = self.paint(img[0])

        return [x.squeeze() for x in [inputs, outputs, painted]]

    def validate_batch(self, batch_size):
        inputs, outputs, painted = self.get_batch(4)

        data = [inputs, outputs, painted]
        fields = ['dm', 'pressure', 'pressure']
        names = ['dm', 'pressure', 'painted pressure']
        for i, imgs in enumerate(data):
            transformed_imgs = self.transform(imgs,
                                              field=fields[i],
                                              z=None, stats=None)

            flipped_imgs = self.inv_transform(transformed_imgs,
                                              field=fields[i],
                                              z=None, stats=None)

            assert(np.allclose(imgs, flipped_imgs))


        for i, imgs in enumerate([inputs, outputs, painted]):

            show.PixelDist(imgs, field='normal ' + names[i], fromtorch=False, xlim=False)
            _imgs = self.transform(imgs, field=fields[i], z=None, stats=None)
            show.PixelDist(_imgs, field='transformed ' + names[i], fromtorch=False)
