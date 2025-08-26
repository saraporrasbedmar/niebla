import numpy as np

from astropy.table import Table

def metallicity(z_sun=0.02, ax=None, color='k', marker='o',
                markersize=30, alpha=0.8, zorder=100, label=None):
    data_array = np.array(
        [[0.55, -0.16, 0.11, 0.14],
         [1.64, -0.4, 0.09, 0.09],
         [2.26, -0.56, 0.11, 0.1],
         [2.70, -0.41, 0.30, 0.21],
         [3.21, -0.79, 0.12, 0.11],
         [3.95, -0.82, 0.17, 0.14],
         [4.54, -1.03, 0.20, 0.12]
         ], dtype=float)

    data_array[:, 2] = (
            z_sun * (-10 ** (data_array[:, 1] - data_array[:, 2])
                     + 10 ** data_array[:, 1]))
    data_array[:, 3] = (
            z_sun * (10 ** (data_array[:, 3] + data_array[:, 1])
                     - 10 ** data_array[:, 1]))
    data_array[:, 1] = 10 ** data_array[:, 1] * z_sun

    data_array = Table(data=data_array,
                       names=('z', 'metall',
                              'metall_err_low', 'metall_err_up'))

    if ax is not None:
        if label is None:
            label = r'$Z_\odot = $%s' % z_sun

        ax.errorbar(x=data_array['z'], y=data_array['metall'],
                    yerr=[data_array['metall_err_low'],
                          data_array['metall_err_up']],
                    linestyle='', color=color,
                    ms=0, zorder=zorder-1, alpha=alpha
                    )
        ax.scatter(x=data_array['z'], y=data_array['metall'],
                   lw=0, color=color,
                   s=markersize,
                   marker=marker, zorder=zorder, alpha=alpha,
                   label=label
                   )
    return data_array
