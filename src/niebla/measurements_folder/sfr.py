import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from astropy.table import Table, vstack


def sfr(plot=True, ax=None,
        markers=['*', '<', '>', 'H', '^', 'd'], colors=None,
        markersize=12, zorder=100, alpha=0.8,
        title_legend='Measurements', loc_legend=1,
        fontsize_legend=16,
        title_fontsize_legend=18, framealpha_legend=0.8
        ):
    data_path = os.path.join(
        os.path.split(__file__)[0], '../data/measurements/sfr/')

    md_data = np.loadtxt(
        data_path + 'MadauDickinson_uvdata.txt', usecols=range(1, 7))
    z_lo = md_data[:, 0]
    z_up = md_data[:, 1]
    z_uv_dat = 0.5 * (z_up + z_lo)
    z_uv_lo = np.absolute(z_uv_dat - z_lo)
    z_uv_up = np.absolute(z_uv_dat - z_up)
    sfr_uv = 10 ** md_data[:, 3]
    sfr_uv_lo = sfr_uv * np.log(10) * md_data[:, 5]
    sfr_uv_up = sfr_uv * np.log(10) * md_data[:, 4]

    sfr1_table = Table(
        data=(z_uv_dat, z_uv_lo, z_uv_up, sfr_uv, sfr_uv_lo, sfr_uv_up),
        names=('z_value', 'zerr_low', 'zerr_up',
               'sfr_value', 'sfrerr_low', 'sfrerr_up'))
    sfr1_table.add_column(
        col=r'Madau $&$ Dickinson UV data', name='Reference')

    md_data = np.loadtxt(
        data_path + 'MadauDickinson_irdata.txt', usecols=range(1, 6))

    z_lo = md_data[:, 0]
    z_up = md_data[:, 1]
    z_ir_dat = 0.5 * (z_up + z_lo)
    z_ir_lo = np.absolute(z_ir_dat - z_lo)
    z_ir_up = np.absolute(z_ir_dat - z_up)
    sfr_ir = 10 ** md_data[:, 2]
    sfr_ir_lo = sfr_ir * np.log(10) * md_data[:, 3]
    sfr_ir_up = sfr_ir * np.log(10) * md_data[:, 4]

    sfr2_table = Table(
        data=(z_ir_dat, z_ir_lo, z_ir_up, sfr_ir, sfr_ir_lo, sfr_ir_up),
        names=('z_value', 'zerr_low', 'zerr_up',
               'sfr_value', 'sfrerr_low', 'sfrerr_up'))
    sfr2_table.add_column(
        col=r'Madau $&$ Dickinson IR data', name='Reference')

    dr_data = np.loadtxt(data_path + 'Driver_SFR.txt')
    z_lo = dr_data[:, 1]
    z_up = dr_data[:, 2]
    z_dr = 0.5 * (z_up + z_lo)
    z_dr = 10 ** (0.5 * (np.log10(z_up) + np.log10(z_lo)))
    z_dr_lo = np.absolute(z_dr - z_lo)
    z_dr_up = np.absolute(z_dr - z_up)
    sfr_dr = 10 ** dr_data[:, 3] / 0.63
    sfr_dr_lo = sfr_dr * np.log(10) * np.sum(dr_data[:, 5:], axis=1)
    sfr_dr_up = sfr_dr_lo

    sfr3_table = Table(
        data=(z_dr, z_dr_lo, z_dr_up,
              sfr_dr, sfr_dr_lo, sfr_dr_up),
        names=('z_value', 'zerr_low', 'zerr_up',
               'sfr_value', 'sfrerr_low', 'sfrerr_up'))
    sfr3_table.add_column(
        col=r'Driver et al. 2018', name='Reference')

    bo_data = np.loadtxt(data_path + 'Bourne2017.txt')
    z_bo_lo = bo_data[:, 2]
    z_bo_up = bo_data[:, 3]
    z_bo = bo_data[:, 0]
    sfr_bo = bo_data[:, 1] / 0.63
    sfr_bo_lo = bo_data[:, 4]
    sfr_bo_up = bo_data[:, 5]

    sfr4_table = Table(
        data=(z_bo, z_bo_lo, z_bo_up, sfr_bo, sfr_bo_lo, sfr_bo_up),
        names=('z_value', 'zerr_low', 'zerr_up',
               'sfr_value', 'sfrerr_low', 'sfrerr_up'))
    sfr4_table.add_column(
        col='Bourne et al. 2017', name='Reference')

    z_bouw = np.array([3.8, 4.9, 5.9])
    sfr_bouw = 10 ** (-np.array([1.0, 1.26, 1.55])) * (1.15 / 1.4)
    dsfr_bouw = np.array([0.06, 0.06, 0.06])
    dsfr_bouw = sfr_bouw * np.log(10) * dsfr_bouw

    sfr5_table = Table(
        data=(z_bouw, np.zeros(len(z_bouw)), np.zeros(len(z_bouw)),
              sfr_bouw, dsfr_bouw, dsfr_bouw),
        names=('z_value', 'zerr_low', 'zerr_up',
               'sfr_value', 'sfrerr_low', 'sfrerr_up'))
    sfr5_table.add_column(
        col='Bouwens et al. 2015', name='Reference')

    dict_srd = vstack([
        sfr1_table, sfr2_table, sfr3_table, sfr4_table, sfr5_table])

    if plot:
        refs = np.unique(dict_srd['Reference'])

        handles = []
        labels = []

        for nref, ref in enumerate(refs):
            dict_values = dict_srd[dict_srd['Reference'] == ref]

            if colors is None:
                color_i = next(ax._get_lines.prop_cycler)['color']
            else:
                color_i = colors[nref % len(colors)]

            ax.errorbar(
                x=dict_values['z_value'],
                xerr=(dict_values['zerr_low'], dict_values['zerr_up']),
                y=dict_values['sfr_value'],
                yerr=(dict_values['sfrerr_low'], dict_values['sfrerr_up']),
                linestyle='',
                marker=markers[nref % len(markers)],
                ms=markersize,
                zorder=zorder, c=color_i, alpha=alpha)

            handles.append(Line2D([], [], ls='',
                                  marker=markers[nref % len(markers)],
                                  markersize=markersize, color=color_i
                                  ))
            labels.append(ref)

        legend1 = plt.legend(
            handles, labels,
            title=title_legend, loc=loc_legend,
            fontsize=fontsize_legend,
            title_fontsize=title_fontsize_legend,
            framealpha=framealpha_legend)
        ax.add_artist(legend1)

    return dict_srd
