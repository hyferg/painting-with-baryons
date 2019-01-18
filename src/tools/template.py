import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import gridspec
from src.tools.boilerplate import files_info
from baryon_painter.painter import Painter
from src.models.network import Network
from baryon_painter.utils.datasets import BAHAMASDataset
import src.visualization.show as show
from torch.utils.data import DataLoader
import src.tools.stats as stat
from src.tools.data import parse_data
from baryon_painter.models.utils import merge_aux_label


class GAN_Painter(Painter):
    def __init__(self, parts_folder,
                 structure_file='/g_struc.pickle',
                 checkpoint_file='/g_weights.cp',
                 transform_file='/transform.pickle',
                 inv_transform_file='/inv_transform.pickle',
                 device='cuda:0',
                 img_dim=(1, 512, 512),
                 input_field='dm' ,
                 label_fields=['pressure'],
    ):


        self.compute_device = device
        self.input_field = input_field
        self.label_fields = label_fields
        self.img_dim = img_dim
        self.transform = None
        self.inv_transform = None
        self.test_data = None
        self.inputs = None
        self.outputs = None
        self.painted = None
        self.t_outputs = None
        self.t_painted = None
        self.idxs = None

        with open(parts_folder + structure_file, 'rb') as handle:
            g_struc = torch.load(handle, map_location=torch.device(self.compute_device))

        self.generator = Network.factory(g_struc)
        self.load_state_from_file(parts_folder + checkpoint_file, self.compute_device)

        if transform_file:
            with open(parts_folder + transform_file, 'rb') as handle:
                self.transform = pickle.load(handle)

        if inv_transform_file:
            with open(parts_folder + inv_transform_file, 'rb') as handle:
                self.inv_transform = pickle.load(handle)

    def load_state_from_file(self, filename, device):
        self.generator.load_self(filename, device)
        self.generator.to(self.compute_device)

    def paint(self, input, z=0.0, stats=None, inverse_transform=True, red=None):
        with torch.no_grad():
            self.generator.eval()

            y = torch.tensor(y, device=self.compute_device)
            prediction = self.generator(y).cpu().numpy()

        if inverse_transform and self.inv_transform is not None:
            return self.inv_transform(
                prediction, field=self.label_fields[0], z=z, stats=stats)
        else:
            return prediction

    def load_test_data(self, data_path, redshifts=[0.0, 0.5, 1.0], test=True):
        train_file_info, test_files_info = files_info(data_path)
        label_fields = ["pressure"]

        (print('test') if test else print('train'))
        self.test_dataset = BAHAMASDataset(
            (test_files_info if test else train_file_info),
            root_path=data_path,
            redshifts=redshifts,
            label_fields=label_fields)

        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)

        self.test_iter = iter(self.test_loader)

    def get_batch(self, batch_size, inverse_transform=True, full=False, type=None):
        inputs = np.zeros((batch_size, *self.img_dim))
        outputs = np.zeros((batch_size, *self.img_dim))
        painted = np.zeros((batch_size, *self.img_dim))
        if full:
            self.t_outputs = np.zeros((batch_size, *self.img_dim))
            self.t_painted = np.zeros((batch_size, *self.img_dim))
        idxs = []
        for i in range(batch_size):
            img, idx, red = parse_data(self.test_iter, self.compute_device, type)

            z = self.test_dataset.sample_idx_to_redshift(idx)

            if inverse_transform is False:
                inputs[i] = self.transform(img[0].numpy(),
                                           self.input_field, z=z,
                                           stats=self.test_dataset.stats)
                outputs[i] = self.transform(img[1].numpy(),
                                            self.label_fields[0], z=z,
                                            stats=self.test_dataset.stats)


            else:
                inputs[i] = img[0].cpu().numpy()
                outputs[i] = img[1].cpu().numpy()

            img = [x.cpu().numpy() for x in img]

            #TODO
            if self.transform is not None:
                y = self.transform(
                    img[0], field=self.input_field, z=z, stats=stats)
            y = y.reshape(1, *y.shape)

            if type == 'troster-redshift-validate':
                y = merge_aux_label(y, red)

            painted[i] = self.paint(y, z=z, stats=self.test_dataset.stats,
                                    inverse_transform=inverse_transform, red=red)
            if full:
                self.t_painted[i] = self.paint(y, z=z, stats=self.test_dataset.stats,
                                          inverse_transform=False)
                self.t_outputs[i] = self.transform(img[1],
                                            self.label_fields[0], z=z,
                                            stats=self.test_dataset.stats)

            idxs.append(int(idx.numpy()))

        return [x.squeeze() for x in [inputs, outputs, painted]] + [np.array(idxs, dtype='uint32')]

    def get_self_batch(self, batch_size, **kwargs):
        self.inputs, self.outputs, self.painted, self.idxs = self.get_batch(batch_size, **kwargs)

    def pressure_grid(self, n):
        self.get_self_batch(n*n, inverse_transform=False)
        fig, axs = plt.subplots(n,n)
        fig.set_size_inches(30, 10)
        idx = 0
        for i in range(n):
            for j in range(n):
                axs[i][j].imshow(self.outputs[idx], vmin=-1, vmax=1)
                axs[i][j].set_title(f'index : {self.idxs[idx]}')
                axs[i][j].set_axis_off()
                idx+=1

    def plot_self(self, idx=0):
        to_plot = [self.inputs, self.outputs, self.painted]
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(30, 10)
        for i, img in enumerate(to_plot):
            im = axs[i].imshow(img[idx], vmin=-1, vmax=1)
            axs[i].set_axis_off()
            axs[i].set_title(f'index : {self.idxs[idx]}')
            fig.colorbar(im, ax=axs[i])

        fig, axs = plt.subplots(1,1)
        fig.set_size_inches(20, 10)
        axs.hist(to_plot[1].flatten())
        axs.hist(to_plot[2].flatten())
        axs.set_yscale('log', nonposy='clip')

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

    def compare_cc(self, true, fake, box_size, batch_size, n_k_bin, idxs):
        fig, axs = plt.subplots(3,3)
        fig.set_size_inches(20, 10)
        y_frac_batch = np.zeros((batch_size, n_k_bin))
        for i in range(batch_size):
            z = str(self.test_dataset.sample_idx_to_redshift(idxs[i]))
            x, y  = stat.cross([true[0][i], true[1][i]], box_size=box_size, n_k_bin=n_k_bin)
            _x, _y  = stat.cross([fake[0][i], fake[1][i]], box_size=box_size, n_k_bin=n_k_bin)
            frac_diff = _y/y-1
            y_frac_batch[i,:] = frac_diff

            colors = {'0.0': 'blue',
                      '0.5': 'orange',
                      '1.0': 'red'}

            axs[0][0].plot(x, frac_diff, colors[z], alpha=0.2)

            axs[0][1].plot(x, frac_diff, colors[z], alpha=0.2)
            axs[0][1].set_ylim([-1, 1])

            axs[0][2].plot(x, frac_diff, colors[z], alpha=0.2)
            axs[0][2].set_ylim([-0.2, 0.2])

        y_frac_max_vals = np.amax(np.absolute(y_frac_batch), axis=1)
        y_frac_max_vals_sorted = np.flip(np.sort(y_frac_max_vals))
        y_frac_max_vals_sorted_idxs = np.flip(idxs[y_frac_max_vals.argsort()])

        print('worst max differences')
        for i in range(5):
            diff = np.trunc(y_frac_max_vals_sorted[i])
            idx = y_frac_max_vals_sorted_idxs[i]
            z = self.test_dataset.sample_idx_to_redshift(idx)
            print(f'z:{z}, diff: {diff}, idx:{idx}')

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

        return fig, y_frac_max_vals_sorted_idxs

    def show_idxs(self, painted, idxs, bad_idxs):
        fig, axs = plt.subplots(len(bad_idxs), 2)
        fig.set_size_inches(20, 20)
        for k, _ in enumerate(bad_idxs):
            img = self.test_dataset.get_input_sample(bad_idxs[k])
            bad_idx = np.argwhere(idxs == bad_idxs[k])
            axs[k][0].imshow(
                img,
                vmin=-1, vmax=1)
            axs[k][1].imshow(
                painted[bad_idx][0][0],
                vmin=-1, vmax=1)
            axs[k][0].set_title(f'idx:{bad_idxs[k]} real')
            axs[k][1].set_title(f'fake')

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
                     'n_k_bin': n_k_bin, 'idxs':idxs}
        fig, bad_idxs = self.compare_cc([inputs, outputs], [inputs, painted], **cc_params)
        fig.suptitle(f'Dark Matter x Pressure Cross Correlation [{batch_size} samples]')

        #self.show_idxs(painted, idxs, bad_idxs[0:4])

        fig, _ = self.compare_cc([outputs, outputs], [painted, painted], **cc_params)
        fig.suptitle('Auto Pressure Cross Correlation [{batch_size} samples]')



        plt.show()

    def report_cc(self, true, fake, box_size, batch_size, n_k_bin, idxs, xlim, colors):
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        axes = []
        ax = plt.subplot(gs[0])
        axes.append(ax)
        ax.set_title(f'Fractional Difference of\nFake over Real DM x Pressure Cross Correlations\nfrom {batch_size} samples')
        ax.axhline(color='black', lw=1.5)


        z_idxs = {
            '0.0': [],
            '0.5': [],
            '1.0': [],
        }

        y_frac_batch = np.zeros((batch_size, n_k_bin))
        for i in range(batch_size):
            z = str(self.test_dataset.sample_idx_to_redshift(idxs[i]))
            x, y  = stat.cross([true[0][i], true[1][i]], box_size=box_size, n_k_bin=n_k_bin)
            _x, _y  = stat.cross([fake[0][i], fake[1][i]], box_size=box_size, n_k_bin=n_k_bin)
            frac_diff = _y/y-1
            y_frac_batch[i,:] = frac_diff

            ax.plot(x, frac_diff, colors[z], alpha=0.2)
            ax.set_ylim([-2, 4])

            z_idxs[z].append([x, y, _y, frac_diff])

        y_frac = np.average(y_frac_batch, axis=0)
        y_frac_std = y_frac_batch.std(axis=0)
        ax.plot(x, y_frac, label='Mean',
                color='white',
                lw=2.5, path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])

        handles, labels = ax.get_legend_handles_labels()
        z0_patch = mpatches.Patch(color=colors['0.0'], label='Redshift 0.0')
        z05_patch = mpatches.Patch(color=colors['0.5'], label='Redshift 0.5')
        z10_patch = mpatches.Patch(color=colors['1.0'], label='Redshift 1.0')
        ax.legend(handles=[z0_patch, z05_patch, z10_patch, handles[0]])

        ax = plt.subplot(gs[1])
        axes.append(ax)
        ax.set_title('Standard Deviation on the Mean (cropped)')
        ax.set_ylim([0, 2])
        ax.plot(x, y_frac_std)

        ax = plt.subplot(gs[2])
        axes.append(ax)
        ax.set_title('Standard Deviation on the Mean')
        ax.plot(x, y_frac_std)
        ax.set_xlabel('$k \;[h \;Mpc^{-1}]$')

        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlim(xlim)

        y_frac_max_vals = np.amax(np.absolute(y_frac_batch), axis=1)
        y_frac_max_vals_sorted = np.flip(np.sort(y_frac_max_vals))
        y_frac_max_vals_sorted_idxs = np.flip(idxs[y_frac_max_vals.argsort()])

        print('worst max differences')
        for i in range(5):
            diff = np.trunc(y_frac_max_vals_sorted[i])
            idx = y_frac_max_vals_sorted_idxs[i]
            z = self.test_dataset.sample_idx_to_redshift(idx)
            print(f'z:{z}, diff: {diff}, idx:{idx}')

        return fig, y_frac_max_vals_sorted_idxs, z_idxs

    def report_plot(self, batch_size, box_size, n_k_bin=20, name=None):
        xlim = [6.15*10**-2, 10]
        colors = {'0.0': 'blue',
                  '0.5': 'orange',
                  '1.0': 'red'}

        cc_params = {'box_size': box_size,
                     'batch_size': batch_size,
                     'n_k_bin': n_k_bin,
                     'idxs': self.idxs,
                     'xlim': xlim,
                     'colors': colors}
        plt.rcParams.update({'font.family': 'serif',
                             'pgf.texsystem': 'pdflatex'})

        # mean analysis
        fig, bad_idxs, z_idxs = self.report_cc(
            [self.inputs, self.outputs], [self.inputs, self.painted], **cc_params)
        fig.set_size_inches(6, 8)
        fig.tight_layout()
        fig.savefig(name+f'_meta.pdf', format='pdf', bbox_inches='tight')

        for z in ['0.0', '0.5', '1.0']:
            fig = self.z_plot(z_idxs[z], color=colors[z], xlim=xlim, z=z)
            fig.set_size_inches(4, 6)
            fig.tight_layout()
            fig.savefig(name+f'_z{z}.pdf', format='pdf', bbox_inches='tight')

        # pixel dist
        fig, axs = plt.subplots(1,1)
        fig.set_size_inches(8, 4)
        axs.set_title('Transformed Pressure Pixel Distribution')
        axs.hist(self.t_outputs.flatten(), alpha=0.5, color='blue', bins=50,
                 label='True Transformed Pressure')
        axs.hist(self.t_painted.flatten(), alpha=0.5, color='red', bins=50,
                 label='Fake Transformed Pressure')
        axs.set_xlim([-1, 1])
        axs.legend()
        axs.set_yscale('log', nonposy='clip')
        fig.tight_layout()
        fig.savefig(name+f'_pixel.pdf', format='pdf', bbox_inches='tight')

        plt.show()

    def z_plot(self, data, color, xlim, z):
        '''
        data of form [[x, y, _y, frac_diff],...]
        '''

        bins = len(data[0][0])
        x = np.array(data[0][0])
        y = np.zeros((len(data), bins))
        _y = np.zeros((len(data), bins))
        for i, line in enumerate(data):
            y[i,:] = line[1]
            _y[i,:] = line[2]

        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)
        _y_mean = _y.mean(axis=0)
        _y_std = _y.std(axis=0)

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax = plt.subplot(gs[0])
        ax.set_title(f'Real vs Fake Cross Correlations\nfor z={z} from {len(data)} samples')
        ax.set_ylabel('$P(k)$')
        #ax.set_ylim(bottom=10**-5, top=None)

        for line in data:
            ax.plot(line[0], line[1], color='grey', alpha=0.2)
            ax.plot(line[0], line[2], color=color, alpha=0.2)

       #ax.plot(x, y_mean, color='green')
       #ax.plot(x, _y_mean, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlim)

        real_patch = mpatches.Patch(color='grey', label= 'Real DM x Real Pressure')
        fake_patch = mpatches.Patch(color=color, label= 'Real DM x Fake Pressure')
        ax.legend(handles=[real_patch, fake_patch])

        ax = plt.subplot(gs[1])
        ax.set_title(f'Cross Fractional Fake/Real\nStandard Deviations for z={z}')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xscale('log')
        ax.axhline(color='black', lw=2.5)
        ax.set_xlabel('$k \;[h \;Mpc^{-1}]$')

        ax.plot(x, _y_std/y_std-1, color='red')


        return fig



