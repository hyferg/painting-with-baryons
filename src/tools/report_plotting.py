def report_cc(true, fake, box_size, batch_size, n_k_bin, idxs, xlim, colors):
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

