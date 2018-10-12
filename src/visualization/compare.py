import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display
from src.tools.stats import CrossCompare, AutoCompare
from src.tools.weights import FreezeModel, UnFreezeModel

def parse_imgs(inputs, targets, fake_targets, transform, debug):
    if debug:
        for i, val in enumerate(inputs):
            _val = targets[i].cpu().numpy()
            _fake_targets = fake_targets.cpu().numpy()
            print(
                'power compare real: {} | {}'.format(
                    np.ndarray.min(_val),
                    np.ndarray.max(_val)
                )
            )
            print(
                'power compare fake: {} | {}\n'.format(
                    np.ndarray.min(_fake_targets[i]),
                    np.ndarray.max(_fake_targets[i])
                )
            )

    flat_targets = targets.cpu().numpy().flatten()
    flat_fake_targets = fake_targets.cpu().numpy().flatten()

    if transform:
        if debug:
            print('transform')
        inputs = transform[0](inputs.cpu().numpy())
        targets = transform[1](targets.cpu().numpy())
        fake_targets = transform[1](fake_targets.cpu().numpy())


    if debug:
        for i, val in enumerate(inputs):
            _val = targets[i].cpu().numpy()
            _fake_targets = fake_targets.cpu().numpy()
            print(
                'power compare real: {} | {}'.format(
                    np.ndarray.min(_val),
                    np.ndarray.max(_val)
                )
            )
            print(
                'power compare fake: {} | {}\n'.format(
                    np.ndarray.min(_fake_targets[i]),
                    np.ndarray.max(_fake_targets[i])
                )
            )

    return inputs, targets, fake_targets, flat_targets, flat_fake_targets


def GetSet(generator, inputs, targets, transform, debug):
    '''
    returns:
        inputs, targets, fake_targets, flat_targets, flat_fake_targets
    '''
    fake_targets = generator(inputs)
    return parse_imgs(inputs, targets, fake_targets, transform=transform, debug=debug)


def meta_compare(inputs, targets, fake_targets, flat_targets, flat_fake_targets,
                 box_size, transform, save, save_path, debug):
    try:
        inputs.device
        targets.device
        fake_targets.device
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        fake_targets = fake_targets.cpu().numpy()
    except:
        pass


    auto = {}
    cross = {}

    for i, val in enumerate(inputs):
        power_a = AutoCompare(
            {'name': 'real', 'data': targets[i]},
            {'name': 'fake', 'data': fake_targets[i]},
            box_size=box_size
        )

        power_c = CrossCompare(
            {'name': 'real', 'data': [inputs[i], targets[i]]},
            {'name': 'fake', 'data': [inputs[i], fake_targets[i]]},
            box_size=box_size
        )

        if i == 0:
            for key in power_a:
                auto[key] = {}
                auto[key]['pcl'] = np.zeros(
                    shape=(
                        len(inputs),
                        len(power_a[key][1])
                    )
                )
                auto[key]['ell'] = np.zeros(
                    shape=(
                        len(inputs),
                        len(power_a[key][0]
                        )
                    )
                )
            for key in power_c:
                cross[key] = {}
                cross[key]['pcl'] = np.zeros(
                    shape=(
                        len(inputs),
                        len(power_c[key][1])
                    )
                )
                cross[key]['ell'] = np.zeros(
                    shape=(
                        len(inputs),
                        len(power_c[key][0])
                    )
                )


        for key in power_a:
            auto[key]['pcl'][i] = power_a[key][1]
            auto[key]['ell'][i] = power_a[key][0]

        for key in power_c:
            cross[key]['pcl'][i] = power_c[key][1]
            cross[key]['ell'][i] = power_c[key][0]

    n_samples = None
    for key in auto:
        if n_samples == None:
            n_samples = len(auto[key]['pcl'])
        auto[key]['pcl'] = np.mean(auto[key]['pcl'], axis=0)
        auto[key]['ell'] = np.mean(auto[key]['ell'], axis=0)

    for key in power_c:
        cross[key]['pcl'] = np.mean(cross[key]['pcl'], axis=0)
        cross[key]['ell'] = np.mean(cross[key]['ell'], axis=0)


    colors = ['blue', 'red']
    fig = plt.figure(figsize=(10, 20))
    for j in range(7):
        if j != 6:
            ax = plt.subplot(7, 2, j+1)
        else:
            ax = plt.subplot(7, 1, 4)
        if j == 0:
            ax.set_title('Auto from {} samples'.format(n_samples))
            ax.set_xscale('log')
            ax.set_yscale('log')
            for k, key in enumerate(auto):
                x = auto[key]['ell']
                y = auto[key]['pcl']
                ax.plot(x, y, label=key, color=colors[k])
            ax.legend()
        elif j == 1:
            ax.set_title('Cross from {} samples'.format(n_samples))
            ax.set_xscale('log')
            ax.set_yscale('log')
            for k, key in enumerate(cross):
                x = cross[key]['ell']
                y = cross[key]['pcl']
                ax.plot(x, y, label=key, color=colors[k])
            ax.legend()
        elif j == 2:
            ax.set_title('Auto Fractional fake/real from {}'.format(n_samples))
            yfrac = auto['fake']['pcl']/auto['real']['pcl'] - 1
            x = auto[key]['ell']
            ax.set_xscale('log')
            ax.plot(x, yfrac, color='purple')
            ax.set_ylim([-0.2, 0.2])
            ax.axhline(y=0.1, linewidth=1, color='black')
            ax.axhline(y=-0.1, linewidth=1, color='black')
        elif j == 3:
            ax.set_title('Cross Fractional fake/real from {}'.format(n_samples))
            yfrac = cross['fake']['pcl']/cross['real']['pcl'] - 1
            x = cross[key]['ell']
            ax.set_xscale('log')
            ax.plot(x, yfrac, color='purple')
            ax.set_ylim([-0.2, 0.2])
            ax.axhline(y=0.1, linewidth=1, color='black')
            ax.axhline(y=-0.1, linewidth=1, color='black')
        elif j == 4:
            ax.set_title('Auto Fractional fake/real from {}'.format(n_samples))
            yfrac = auto['fake']['pcl']/auto['real']['pcl'] - 1
            x = auto[key]['ell']
            ax.set_xscale('log')
            ax.plot(x, yfrac, label='f/r', color='purple')
            ax.set_ylim([-1, 1])
            ax.axhline(y=0, linewidth=1, color='black')
            ax.legend()
        elif j == 5:
            ax.set_title('Cross Fractional fake/real from {}'.format(n_samples))
            yfrac = cross['fake']['pcl']/cross['real']['pcl'] - 1
            x = cross[key]['ell']
            ax.set_xscale('log')
            ax.plot(x, yfrac, label='f/r', color='purple')
            ax.set_ylim([-1, 1])
            ax.axhline(y=0, linewidth=1, color='black')
            ax.legend()
        elif j ==6:
            ax.set_title('Pixel Distribution fake/real from {}'.format(n_samples))
            alpha = 0.5
            colors = ['red', 'blue']
            names = ['fake', 'real']
            flat_data = [flat_fake_targets, flat_targets]
            for i, color in enumerate(colors):
                ax.hist(flat_data[i], alpha=alpha, color=color, bins=50, label=names[i], log=True)
            ax.set_xlim([-1, 1])
            ax.legend()
        fig.add_subplot(ax)

    fig.tight_layout()
    if save == False:
        display.display(plt.gcf())
    elif save == True:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def MetaCompareLW(inputs, targets, fake_targets, box_size,
                  transform=None,
                  save=False,
                  save_path=None,
                  debug=False):

    inputs, targets, fake_targets, flat_targets, flat_fake_targets = parse_imgs(
        inputs, targets, fake_targets, transform, debug
    )

    meta_compare(inputs, targets, fake_targets, flat_targets, flat_fake_targets,
                 box_size, transform, save, save_path, debug)


def MetaCompare(generator, inputs, targets, box_size,
                transform=None,
                save=False,
                save_path=None,
                debug=False):

    inputs, targets, fake_targets, flat_targets, flat_fake_targets = GetSet(generator, inputs, targets,
                                                                            transform=transform, debug=debug)

    meta_compare(inputs, targets, fake_targets, flat_targets, flat_fake_targets,
                 box_size, transform, save, save_path, debug)


def PowerCompareLW(inputs, targets, fake_targets, box_size, n=4, grid_size=(2,2),
                   transform=None,
                   save=False,
                   save_path=None,
                   debug=False):

    inputs, targets, fake_targets, flat_targets, flat_fake_targets = parse_imgs(
        inputs, targets, fake_targets, transform, debug
    )

    power_compare(inputs, targets, fake_targets, box_size, n, grid_size, transform, debug, save, save_path)


def power_compare(inputs, targets, fake_targets, box_size, n, grid_size, transform, debug, save, save_path):

    autos = []
    crosses = []

    for i, val in enumerate(inputs):
        autos.append(
            AutoCompare(
                {'name': 'real', 'data': targets[i]},
                {'name': 'fake', 'data': fake_targets[i]},
                box_size=box_size
            )
        )
        crosses.append(
            CrossCompare(
                {'name': 'real', 'data': [inputs[i], targets[i]]},
                {'name': 'fake', 'data': [inputs[i], fake_targets[i]]},
                box_size=box_size
            )
        )


    fig = plt.figure(figsize=(15, 15))
    outer = gridspec.GridSpec(n//2, n//2, wspace=0.2, hspace=0.2)

    for i in range(n):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[i], wspace=0.1, hspace=0.3)

        colors = ['blue', 'red']
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            if j == 0:
                ax.set_title('Auto')
                for k, key in enumerate(autos[i]):
                    x, y = autos[i][key]
                    ax.plot(x, y, label=key, color=colors[k])
            elif j == 1:
                ax.set_title('Cross')
                for k, key in enumerate(crosses[i]):
                    x, y = crosses[i][key]
                    ax.plot(x, y, label=key, color=colors[k])
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            fig.add_subplot(ax)

    if not save == True:
        display.display(plt.gcf())
    elif save == True:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def PowerCompare(generator, inputs, targets, box_size, n=4, grid_size=(2,2),
                 transform=None, debug=False,
                 save=False,
                 save_path=None):

    inputs = inputs[:n]

    inputs, targets, fake_targets, _, _ = GetSet(
        generator, inputs, targets, transform=transform, debug=debug)

    power_compare(inputs, targets, fake_targets, box_size, n, grid_size,
                  transform, debug, save, save_path)
