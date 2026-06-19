import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import astropy.units as u
from astropy.table import QTable, vstack

from scipy.special import ndtr, erfinv


def _import_spectrum_data(
        obs_not_taken=None, import_one_type=None,
        lambda_min=0., lambda_max=1e20):

    nu = []
    nuI_nu = []
    dnuI_nu = [[], []]
    lim = []
    ref = []
    length = []

    parent_dir = (Path(__file__).resolve().parent.parent
                  / "data" / "measurements" / "ebl_intensity")

    with open(parent_dir / 'all_freq/CB_complete.txt', 'r') as filein:
        d = filein.readlines()
        while len(d) != 0:
            aaa = ' '.join(d[0].split())
            length.append(int(d[1].split()[0]))

            for i in range(2, length[-1] + 2):
                ref.append(aaa)
                nu.append(float(d[i].split()[0]))
                nuI_nu.append(float(d[i].split()[1]))
                dnuI_nu[0].append(float(d[i].split()[2]))
                dnuI_nu[1].append(float(d[i].split()[3]))
                aa = int(float(d[i].split()[4]))
                if aa == 0:
                    lim.append('IGL')
                elif aa == 1:
                    lim.append('UL')
                elif aa == 2:
                    lim.append('CXB_measurement')
                else:
                    raise NameError('Format of file not correct.')
            d = np.delete(d, range(0, length[-1] + 2))

    nu = np.array(nu)
    nuI_nu = np.array(nuI_nu)
    dnuI_nu[0] = np.array(dnuI_nu[0])
    dnuI_nu[1] = np.array(dnuI_nu[1])
    lim = np.array(lim)

    data = QTable(data=(nu, nuI_nu, dnuI_nu[0], dnuI_nu[1], lim, ref),
                 units=(u.Hz, u.W / (u.m ** 2 * u.sr),
                        u.W / (u.m ** 2 * u.sr), u.W / (u.m ** 2 * u.sr),
                        None, None),
                 names=('lambda', 'nuInu',
                        'nuInu_errn', 'nuInu_errp',
                        'type', 'reference'))

    data['lambda'] = data['lambda'].to(u.um, equivalencies=u.spectral())
    data['nuInu'] = data['nuInu'].to(u.nW / (u.m ** 2 * u.sr))
    data['nuInu_errn'] = data['nuInu_errn'].to(u.nW / (u.m ** 2 * u.sr))
    data['nuInu_errp'] = data['nuInu_errp'].to(u.nW / (u.m ** 2 * u.sr))

    data = data[((data['lambda'].value >= lambda_min)
                 & (data['lambda'].value <= lambda_max))]

    if obs_not_taken is not None:
        mask = np.isin(data['reference'].astype(str), obs_not_taken)
        data = data[~mask]

    if import_one_type is not None:
        mask = data['type'] == import_one_type
        data = data[mask]

    return data


def _dictionary_datatype(
        obs_type,
        lambda_min=0., lambda_max=1e15,
        obs_not_taken=None):

    parent_dir = (Path(__file__).resolve().parent.parent/"data"/
                  "measurements"/"ebl_intensity"/'optical_data_2023')

    if obs_not_taken is None:
        obs_not_taken = []

    list_dirs = os.listdir(parent_dir)
    list_dirs.sort()

    t = QTable(
        names=('lambda', 'nuInu', 'nuInu_errn',
               'nuInu_errp', 'type', 'reference'),
        units=(u.um, u.nW / u.m ** 2 / u.sr, u.nW / u.m ** 2 / u.sr,
               u.nW / u.m ** 2 / u.sr, None, None),
        dtype=(np.float64, np.float64, np.float64,
               np.float64, str, str))

    for directory in list_dirs:

        list_files = os.listdir(parent_dir / directory)
        list_files.sort()

        for ni, name in enumerate(list_files):

            data = QTable.read(
                parent_dir / directory / name, format='ascii.ecsv')

            if data.meta['label'] in obs_not_taken:
                continue

            data.add_column(data.meta['observable_type'], name='type')
            data.add_column(data.meta['label'], name='reference')

            x_data = data.colnames[0]
            if x_data != 'lambda':
                data.rename_column(x_data, 'lambda')
                x_data = data.colnames[0]

            if data.meta['observable_type'] != obs_type:
                continue

            # Change of units to our standard
            if data['nuInu'].unit.is_equivalent(u.Jy / u.sr):
                data['nuInu'] = (data['nuInu'].to(
                            u.W / u.m ** 2 / u.Hz / u.sr)
                        * data[x_data].to(
                    u.Hz, equivalencies=u.spectral()))
                data['nuInu_errn'] = (
                        data['nuInu_errn'].to(
                            u.W / u.m ** 2 / u.Hz / u.sr)
                        * data[x_data].to(
                    u.Hz, equivalencies=u.spectral()))
                data['nuInu_errp'] = (
                        data['nuInu_errp'].to(
                            u.W / u.m ** 2 / u.Hz / u.sr)
                        * data[x_data].to(
                    u.Hz, equivalencies=u.spectral()))

            data[x_data] = data[x_data].to(
                u.um, equivalencies=u.spectral())
            data['nuInu'] = data['nuInu'].to(
                u.nW / u.m ** 2 / u.sr)
            data['nuInu_errn'] = data['nuInu_errn'].to(
                u.nW / u.m ** 2 / u.sr)
            data['nuInu_errp'] = data['nuInu_errp'].to(
                u.nW / u.m ** 2 / u.sr)

            lambda_accepted = (
                    (data[x_data].to(u.um).value >= lambda_min)
                    & (data[x_data].to(u.um).value <= lambda_max))

            t = vstack([t, data[lambda_accepted]],
                       metadata_conflicts='silent')

    return t


def ebl(lambda_min_total=0., lambda_max_total=1e4,
        plot=True, axis=None, obs_not_taken=None,
        markers=None,
        colors_UL=None, colors_IGL=None,
        zorder_UL=10, alpha_UL=0.8, markersize_UL=10,
        colors_nh=None,
        markersize_nh=28, markeredgewidth_nh=2,
        marker_errorbar_nh='.', mfc_errorbar_nh='k',
        markersize_errorbar_nh=8,
        zorder_IGL=10, alpha_IGL=0.8, markersize_IGL=10,
        show_legend=True,
        ncol_legend=2,
        title_legend='Measurements',
        fontsize_legend=11.5,
        title_fontsize_legend=20,
        framealpha_legend=1,
        loc_legend=6,
        bbox_to_anchor_legend=(1.005, 0.5),
        order_legend=None
        ):

    if plot is True and axis is None:
        raise AttributeError(
            f"sfr data points plotting: "
            f"plot boolean set to True but axis is None")

    if markers is None:
        markers = ['>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

    if order_legend is None:
        order_legend = ['UL', 'NH', 'CXB_measurement', 'IGL']

    # Datapoints we use in the study
    upper_lims_cob = _dictionary_datatype(obs_type='UL')

    upperlims_cub = _import_spectrum_data(
        obs_not_taken=[
            '$\\mathrm{COBE \\ (Arendt \\ & \\ Dwek \\ 2003)}$',
            '$\\mathrm{Voyager \\ I/II \\ (Edelstein \\ et \\ al. \\ 2000)}$',
            '$\\mathrm{COBE \\ (Sano \\ et \\ al. \\ 2015)}$',
            '$\\mathrm{COBE \\ (Sano \\ et \\ al. \\ 2016)}$',
            '$\\mathrm{HST \\ (Bernstein \\ 2007)}$',
            '$\\mathrm{HST \\ (Kawara \\ et \\ al. \\ 2017)}$',
            '$\\mathrm{UVX \\ (Martin \\ et \\ al. \\ 1991)}$',
            '$\\mathrm{UVX \\ (Murthy \\ et \\ al. \\ 1989)}$',
            '$\\mathrm{UVX \\ (Murthy \\ et \\ al. \\ 1990)}$',
            '$\\mathrm{CIBER \\ (Matsuura \\ et \\ al. \\ 2017)}$',
            '$\\mathrm{AKARI \\ (Tsumura \\ et \\ al. \\ 2013)}$',
            '$\\mathrm{IRTS \\ (Matsumoto \\ et \\ al. \\ 2015)}$',
            '$\\mathrm{Pioneer \\ 10/11 \\ (Matsuoka \\ et \\ al. \\ 2011)}$',
            '$\\mathrm{HST \\ (Brown \\ et \\ al. \\ 2000)}$'],
        lambda_max=5.,
        import_one_type='UL')

    upper_lims_cxb = _import_spectrum_data(
        obs_not_taken=[
            '$\\mathrm{ASCA \\ (Miyaji \\ et \\ al. \\ 1998)}$',
            '$\\mathrm{Apollo \\ Soyuz \\ (Stern \\ & \\ Bowyer \\ 1979)}$',
            '$\\mathrm{BeppoSAX \\ (Frontera \\ et \\ al. \\ 2007)}$',
            '$\\mathrm{Compton \\ (Strong \\ et \\ al. \\ 2003)}$',
            '$\\mathrm{Compton \\ (Weidenspointner \\ 2000)}$',
            '$\\mathrm{DUVE \\ (Korpela \\ et \\ al. \\ 1998)}$',
            '$\\mathrm{EUVE \\ (Jelinsky \\ et \\ al. \\ 1995)}$',
            '$\\mathrm{EUVE \\ (Lieu \\ et \\ al. \\ 1993)}$',
            '$\\mathrm{Fermi \\ (Ackermann \\ et \\ al. \\ 2015)}$',
            '$\\mathrm{HEAO \\ (Gruber \\ et \\ al. \\ 1999)}$',
            r'$\mathrm{INTEGRAL \ (Turler\ et\ al. \ 2010)}$',
            '$\\mathrm{ROSAT \\ (Miyaji \\ et \\ al. \\ 1998)}$',
            '$\\mathrm{SMM \\ (Watanabe \\ et \\ al. \\ 2000)}$',
            '$\\mathrm{Swift \\ (Moretti \\ et \\ al. \\ 2009)}$',
            '$\\mathrm{Voyager \\ I/II \\ (Edelstein \\ et \\ al. \\ 2000)}$',
            '$\\mathrm{XMM}$-$\\mathrm{Newton \\ (De\\,Luca \\ & \\ Molendi '
            '\\ 2004)}$'],
        lambda_max=0.01, import_one_type='CXB_measurement')

    lowerlimits_cub = _import_spectrum_data(
        obs_not_taken=[
            '$\\mathrm{FOCA \\ (Milliard \\ et \\ al. \\ 1992)}$',
            '$\\mathrm{GALEX \\ (Xu \\ et \\ al. \\ 2005)}$'],
        lambda_max=5.,
        import_one_type='IGL')

    lowerlimits_cob = _dictionary_datatype('IGL')


    measurements_table = vstack(
        [upper_lims_cob, upperlims_cub, upper_lims_cxb,
         lowerlimits_cub, lowerlimits_cob], metadata_conflicts='silent')

    measurements_table['type'][measurements_table['reference']
                               == r'NH/LORRI (Symons+ ‘23)'] = 'NH'
    measurements_table['type'][measurements_table['reference']
                               == r'NH/LORRI (Postman+ ‘24)'] = 'NH'
    measurements_table['reference'][
        measurements_table['reference'] == r'COBE/DIRBE (Arendt \& Dwek ‘03)'] = \
        r'COBE/DIRBE (Arendt & Dwek ‘03)'

    # Fix the 1sigma value of the direct detections that are reported to be
    # upper limits, and therefore we want to plot them as downwards arrows
    dict_sigmas = {
        '$\\mathrm{Dark \\ Cloud \\ (Mattila \\ et \\ al. \\ 2012)}$': 2,
        '$\\mathrm{Apollo \\ Soyuz \\ (Stern \\ & \\ Bowyer \\ 1979)}$': 1,
        '$\\mathrm{EUVE \\ (Lieu \\ et \\ al. \\ 1993)}$': 3,
        '$\\mathrm{Voyager \\ I/II \\ (Murthy \\ et \\ al. \\ 1999)}$': 1,
        '$\\mathrm{DUVE \\ (Korpela \\ et \\ al. \\ 1998)}$': 2,
        '$\\mathrm{EUVE \\ (Jelinsky \\ et \\ al. \\ 1995)}$': 3
    }
    measurements_table.add_column(
        (measurements_table['nuInu_errn']
         + measurements_table['nuInu_errp']) / 2., name='1 sigma')

    # Some papers cite their 2 or 3 sigma constraints of the CB, not always
    # 1 sigma values. THerefore, here we calculate the 1 sigma value from
    # the given values, which are listed in dict_sigmas
    def sigma_from_UL(ul_value, sigma):
        return ul_value / erfinv(ndtr(sigma))

    for obs in dict_sigmas:
        individual = ((measurements_table['nuInu_errp'] == 0)
                      & (measurements_table['reference'] == obs))

        measurements_table['1 sigma'][individual] = sigma_from_UL(
            measurements_table['nuInu'][individual], dict_sigmas[obs])

        measurements_table['nuInu'][individual] = 0.

        measurements_table['type'][individual] = 'UL_arrow'

    measurements_table['nuInu_errn'][
        measurements_table['reference']
        == '$\\mathrm{Voyager \\ I/II \\ (Murthy \\ et \\ al. \\ 1999)}$'] \
        = measurements_table['nuInu'][
        measurements_table['reference']
        == '$\\mathrm{Voyager \\ I/II \\ (Murthy \\ et \\ al. \\ 1999)}$']

    # We order the data
    for ref_i in measurements_table['reference']:
        refi = ref_i
        if "+ '" in refi:
            refi = refi.replace("+ '", ' ')
            number = refi[-3:-1]
            if float(number) > 25:
                refi = refi.replace(number, 'et al. 19' + number)
            else:
                refi = refi.replace(number, 'et al. 20' + number)
        if "+ ‘" in refi:
            refi = refi.replace("+ ‘", ' ')
            number = refi[-3:-1]
            if float(number) > 25:
                refi = refi.replace(number, 'et al. 19' + number)
            else:
                refi = refi.replace(number, 'et al. 20' + number)
        if "+ ’" in refi:
            refi = refi.replace("+ ’", ' ')
            number = refi[-3:-1]
            if float(number) > 25:
                refi = refi.replace(number, 'et al. 19' + number)
            else:
                refi = refi.replace(number, 'et al. 20' + number)

        refi = refi.replace("$\\mathrm{", '')
        refi = refi.replace("}$", '')
        refi = refi.replace(" \\", '')
        measurements_table['reference'][
        measurements_table['reference'] == ref_i] = refi

    measurements_table = measurements_table[
        ((measurements_table['lambda'].value >= lambda_min_total)
         * (measurements_table['lambda'].value <= lambda_max_total))]


    if obs_not_taken is not None:
        mask = np.isin(
            measurements_table['reference'].astype(str), obs_not_taken)
        measurements_table = measurements_table[~mask]

    t = QTable(
        names=('lambda', 'nuInu', 'nuInu_errn',
               'nuInu_errp', 'type', 'reference', '1 sigma'),
        units=(u.um, u.nW / u.m ** 2 / u.sr, u.nW / u.m ** 2 / u.sr,
               u.nW / u.m ** 2 / u.sr, None, None, u.nW / u.m ** 2 / u.sr),
        dtype=(np.float64, np.float64, np.float64,
               np.float64, str, str, np.float64))

    for orderi in order_legend:
        mask = [orderi in a for a in measurements_table['type']]
        aa = np.argsort(measurements_table[mask]['reference'])
        t = vstack((t, measurements_table[mask][aa]))

    del measurements_table

    names_all = []
    [names_all.append(i) for i in t['reference'] if i not in names_all]

    i = 0
    i_nh = 0

    handles = []
    labels = []

    if plot:
        if colors_UL is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_UL = prop_cycle.by_key()['color']

        if colors_nh is None:
            colors_nh = ['lime', '#00A2FF']

        if colors_IGL is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_IGL = prop_cycle.by_key()['color']

        for ni, name in enumerate(names_all):
            data_total = t[t['reference'] == name]
            type_i = np.unique(data_total['type'])

            color_i = colors_UL[ni % len(colors_UL)]
            color_ni = colors_nh[ni % len(colors_nh)]
            color_igl = colors_IGL[ni % len(colors_IGL)]

            for datatype in type_i:
                data = data_total[data_total['type'] == datatype]

                if datatype == 'UL':
                    axis.errorbar(
                        x=data['lambda'], y=data['nuInu'],
                        yerr=[data['nuInu_errn'], data['nuInu_errp']],
                        linestyle='', color=color_i, ms=markersize_UL,
                        marker=markers[i % len(markers)],
                        alpha=alpha_UL, zorder=zorder_UL,
                        mfc='white'
                    )

                    if name in labels:
                        continue
                    else:
                        handles.append(Line2D(
                            [], [], ls='',
                            marker=markers[i % len(markers)],
                            markersize=markersize_UL, color=color_i,
                            alpha=alpha_UL, mfc='white'
                            ))
                        labels.append(name)

                elif datatype == 'CXB_measurement':
                    axis.errorbar(
                        x=data['lambda'], y=data['nuInu'],
                        yerr=[data['nuInu_errn'], data['nuInu_errp']],
                        linestyle='', color=color_i, ms=markersize_UL,
                        marker=markers[i % len(markers)],
                        alpha=alpha_UL, zorder=zorder_UL
                    )
                    handles.append(Line2D(
                        [], [], ls='',
                        marker=markers[i % len(markers)],
                        markersize=markersize_UL, color=color_i
                    ))
                    labels.append(name)

                elif datatype == 'NH':

                    axis.errorbar(
                        x=data['lambda'], y=data['nuInu'],
                        linestyle='', color='w',
                        marker='*',
                        markerfacecolor='w',
                        markersize=markersize_nh,
                        markeredgewidth=markeredgewidth_nh,
                        zorder=zorder_UL, alpha=alpha_UL
                    )
                    axis.errorbar(
                        x=data['lambda'], y=data['nuInu'],
                        linestyle='', color=color_ni,
                        marker='*',
                        markerfacecolor='none',
                        markersize=markersize_nh,
                        markeredgewidth=markeredgewidth_nh,
                        zorder=zorder_UL, alpha=alpha_UL
                    )
                    axis.errorbar(
                        x=data['lambda'], y=data['nuInu'],
                        yerr=[data['nuInu_errn'], data['nuInu_errp']],
                        linestyle='',
                        color=mfc_errorbar_nh,
                        mfc=mfc_errorbar_nh,
                        marker=marker_errorbar_nh,
                        markersize=markersize_errorbar_nh,
                        zorder=zorder_UL, alpha=alpha_UL
                    )
                    handles.append((
                        plt.Line2D([], [], linestyle='',
                                   color=color_ni,
                                   markerfacecolor='w',
                                   marker='*',
                                   markersize=markersize_nh*0.7),
                        plt.Line2D([], [], linestyle='',
                                   color=mfc_errorbar_nh,
                                   markerfacecolor=mfc_errorbar_nh,
                                   marker=marker_errorbar_nh,
                                   markersize=markersize_errorbar_nh*0.7)
                      ))
                    labels.append(name)
                    i_nh += 1

                elif datatype == 'UL_arrow':
                    if name in labels:
                        continue
                    else:
                        handles.append(Line2D(
                            [], [], ls='',
                            marker=markers[i % len(markers)],
                            markersize=markersize_UL, color=color_i,
                            alpha=alpha_UL, mfc='white',
                        ))
                        labels.append(name)

                    axis.errorbar(
                        x=data['lambda'],
                        y=data['nuInu_errn'],
                        yerr=data['nuInu_errn'] * 0.4,
                        linestyle='', color=color_i, ms=markersize_UL,
                        marker=markers[i % len(markers)],
                        zorder=zorder_UL, alpha=alpha_UL,
                        mfc='white', uplims=True
                    )

                elif datatype == 'IGL':

                    axis.errorbar(
                        x=data['lambda'], y=data['nuInu'],
                        yerr=[data['nuInu_errn'], data['nuInu_errp']],
                        linestyle='', color=color_igl, ms=markersize_IGL,
                        marker=markers[i % len(markers)],
                        zorder=zorder_IGL, alpha=alpha_IGL,
                    )
                    handles.append(Line2D(
                        [], [], ls='',
                        marker=markers[i % len(markers)],
                        markersize=markersize_IGL, color=color_igl,
                        alpha=alpha_IGL
                    ))
                    labels.append(name)
                i += 1

        if show_legend:
            legend_measurements = axis.legend(
                handles, labels,
                handler_map={tuple: HandlerTuple(ndivide=1)},
                ncol=ncol_legend,
                title=title_legend, loc=loc_legend,
                fontsize=fontsize_legend,
                title_fontsize=title_fontsize_legend,
                framealpha=framealpha_legend,
                bbox_to_anchor=bbox_to_anchor_legend)

            return t, legend_measurements

    return t
