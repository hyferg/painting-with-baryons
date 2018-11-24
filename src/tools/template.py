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
import src.tools.stats as stat


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

        self.test_dataset = BAHAMASDataset(test_files_info,
                                           root_path=data_path,
                                           redshifts=redshifts,
                                           label_fields=label_fields)

        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)

        self.test_iter = iter(self.test_loader)

    def get_batch(self, batch_size):
        inputs = np.zeros((batch_size, *self.img_dim))
        outputs = np.zeros((batch_size, *self.img_dim))
        painted = np.zeros((batch_size, *self.img_dim))
        idxs = []
        for i in range(batch_size):
            img, idx = next(self.test_iter)
            z = self.test_dataset.sample_idx_to_redshift(idx)
            inputs[i] = img[0].numpy()
            outputs[i] = img[1].numpy()
            painted[i] = self.paint(img[0].numpy(), z=z, stats=self.test_dataset.stats)
            idxs.append(idx.numpy())

        return [x.squeeze() for x in [inputs, outputs, painted]] + [idxs]

    def validate_transforms(self, data, fields, stats, idxs):
        for i, imgs in enumerate(data):
            for j, img in enumerate(imgs):
                z = self.test_dataset.sample_idx_to_redshift(idxs[j])[0]
                transformed_img = self.transform(img,
                                                 field=fields[i],
                                                 z=z, stats=stats)

                flipped_img = self.inv_transform(transformed_img,
                                                 field=fields[i],
                                                 z=z, stats=stats)

                assert(np.allclose(img, flipped_img))

    def compare_cc(self, true, fake, box_size, batch_size, n_k_bin):
        fig, axs = plt.subplots(3,3)
        fig.set_size_inches(20, 10)
        y_frac_batch = np.zeros((batch_size, n_k_bin))
        for i in range(batch_size):
            x, y  = stat.cross([true[0][i], true[1][i]], box_size=box_size, n_k_bin=n_k_bin)
            _x, _y  = stat.cross([fake[0][i], fake[1][i]], box_size=box_size, n_k_bin=n_k_bin)
            frac_diff = _y/y-1
            y_frac_batch[i,:] = frac_diff

            axs[0][0].plot(x, frac_diff, alpha=0.2)

            axs[0][1].plot(x, frac_diff, alpha=0.2)
            axs[0][1].set_ylim([-1, 1])

            axs[0][2].plot(x, frac_diff, alpha=0.2)
            axs[0][2].set_ylim([-0.2, 0.2])

        y_frac = np.average(y_frac_batch, axis=0)
        y_frac_std = y_frac_batch.std(axis=0)

        axs[1][0].plot(x, y_frac)

        axs[1][1].plot(x, y_frac)
        axs[1][1].set_ylim([-1, 1])

        axs[1][2].plot(x, y_frac)
        axs[1][2].set_ylim([-0.2, 0.2])

        axs[2][0].plot(x, y_frac_std)

        axs[2][1].plot(x, y_frac_std)
        axs[2][1].set_ylim([0, 10])

        axs[2][2].plot(x, y_frac_std)
        axs[2][2].set_ylim([0, 1])

        for i, row in enumerate(axs):
            for ax in row:
                ax.set_xscale('log')
                if i == 0:
                    ax.set_title('fractional differences')
                if i == 1:
                    ax.set_title('average fractional difference')
                    ax.axhline(y=0, linewidth=1, color='black')
                if i == 2:
                    ax.set_title('standard deviation')

        return fig

    def validate_batch(self, batch_size, box_size, n_k_bin=20, transform_closure=False):
        inputs, outputs, painted, idxs = self.get_batch(batch_size)

        data = [inputs, outputs, painted]
        fields = ['dm', 'pressure', 'pressure']
        names = ['dm', 'pressure', 'painted pressure']

        if transform_closure:
            self.validate_transforms(data,
                                     fields,
                                     self.test_dataset.stats,
                                     idxs,
            )

        cc_params = {'box_size': box_size, 'batch_size': batch_size,
                     'n_k_bin': n_k_bin}
        fig = self.compare_cc([inputs, outputs], [inputs, painted], **cc_params)
        fig.suptitle(f'Dark Matter x Pressure Cross Correlation [{batch_size} samples]')

        fig = self.compare_cc([outputs, outputs], [painted, painted], **cc_params)
        fig.suptitle('Auto Pressure Cross Correlation [{batch_size} samples]')

        plt.show()
