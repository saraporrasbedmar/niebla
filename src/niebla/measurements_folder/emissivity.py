import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from astropy.table import Table, vstack


def emissivity(z_min=None, z_max=None, lambda_min=0,
               lambda_max=5,
               take_only_refs=None, plot=False, ax=None,
               markers=['*', '<', '>', 'H', '^', 'd',
                        'h', 'o', 'p', 's', 'v'],
               colors=None, zorder=100, markersize=12,
               alpha=0.8,
               show_legend=False,
               title_legend='Measurements', loc_legend=1,
               fontsize_legend=16,
               title_fontsize_legend=18, framealpha_legend=0.8
               ):
    data_path = os.path.join(
        os.path.split(__file__)[0], '../data/measurements/')

    data = np.loadtxt(data_path + 'emissivities_finke22.txt',
                      dtype={'names': ('z', 'lambda', 'eje',
                                       'eje_p', 'eje_n', 'reference'),
                             'formats': (
                             float, float, float, float, float, 'S1')}
                      )
    data = Table(data,
                 names=('z', 'lambda', 'eje',
                        'eje_p', 'eje_n', 'reference'))

    if z_min is not None:
        data = data[data['z'] >= z_min]
    if z_max is not None:
        data = data[data['z'] <= z_max]

    data = data[(data['lambda'] >= lambda_min)
                * (data['lambda'] <= lambda_max)]

    if take_only_refs is not None:
        data_new = Table(
            names=('z', 'lambda', 'eje',
                   'eje_p', 'eje_n', 'reference'))

        for take1ref in take_only_refs:
            data_new = vstack([
                data_new, data[data['reference'].astype('str') == take1ref]])
        data = data_new.copy()

    if plot:
        list_references = np.unique(data['reference'])

        if colors is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

        if show_legend:
            handles = []
            labels = []

        for nref, ref in enumerate(list_references):

            color_i = colors[nref % len(colors)]

            data_individual = data[data['reference'] == ref]

            plt.errorbar(x=data_individual['z'],
                         y=data_individual['eje'],
                         yerr=(data_individual['eje_n'],
                               data_individual['eje_p']),
                         linestyle='',
                         marker=markers[nref % len(markers)],
                         ms=markersize,
                         zorder=zorder,
                         c=color_i, alpha=alpha)

            if show_legend:
                handles.append(
                    Line2D([], [], ls='',
                           marker=markers[nref % len(markers)],
                           markersize=markersize, color=color_i
                           ))
                labels.append(ref)

        if show_legend:
            legendlegend = {
                'a': 'Driver et al.(2012)',
                'b': 'Andrews et al. (2017)',
                'c': 'Tresse et al. (2007)',
                'd': 'Arnouts et al. (2007)',
                'e': 'Beare et al. (2019)',
                'f': 'Budavari et al. (2005)',
                'g': 'Cucciati et al. (2012)',
                'h': 'Dahlen et al. (2007)',
                'i': 'Marchesini et al. (2012)',
                'j': 'Schiminovich et al. (2015)',
                'k': 'Stefanon & Marchesini (2013)',
                'l': 'Bouwens et al. (2015)',
                'm': 'Finkelstein et al. (2015)',
                'n': 'McLeod et al. (2016)',
                'o': 'Dai et al. (2009)',
                'p': 'Babbedge et al. (2006)',
                'q': 'Takeuchi et al. (2006)',
                'r': 'Saldana-Lopez et al. (2021)',
                's': 'Yoshida et al. (2006)'}
            labelsend = [legendlegend[i] for i in labels]
            legend1 = plt.legend(
                handles, labelsend,
                title=title_legend, loc=loc_legend,
                fontsize=fontsize_legend,
                title_fontsize=title_fontsize_legend,
                framealpha=framealpha_legend)
            ax.add_artist(legend1)

    return data
