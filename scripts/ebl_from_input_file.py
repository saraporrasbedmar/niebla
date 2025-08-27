# IMPORTS --------------------------------------------#
import os
import yaml
# import psutil
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline, RegularGridInterpolator

from astropy.constants import c


from niebla import ebl_model, measurements
# from src.niebla import ebl_model
# import src.niebla.measurements as measurements

# from ebltable.ebl_from_model import EBL

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=7)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=5)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

# input_file_dir = ('outputs/outputs_dust_final_new/')
input_file_dir = 'notebooks/'

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")


# def memory_usage_psutil():
#     # return the memory usage in MB
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info()[0] / float(10 ** 6)
#     return mem

# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            # print(pa)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'g', 'k',
          'purple', 'cyan', 'brown']

linstyles_ssp = ['solid', '--', 'dotted', '-.', (0, (3, 5, 1, 5, 1, 5))]

markers = ['.', 'x', '+', '*', '^', '>', '<']

# We initialize the class with the input file
config_data = read_config_file(input_file_dir + 'input_example.yml')
ebl_class = ebl_model.EBL_model.input_yaml_data_into_class(
    config_data, log_prints=True)

# FIGURE: METALLICITIES FOR DIFFERENT MODELS ---------------------------
fig_met, ax_met = plt.subplots(figsize=(8, 8))
plt.yscale('log')
measurements.metallicity(ax=ax_met)
measurements.metallicity(ax=ax_met, z_sun=0.014, color='b')

plt.xlabel('redshift z')
plt.ylabel('Z')

# FIGURE: COB FOR DIFFERENT MODELS -------------------------------------
fig_cob, ax_cob = plt.subplots(figsize=(10, 8))
# This command is necessary because we create the legend in a
# subroutine, so pyplot does not recognize it straight away
plt.tight_layout()

waves_ebl = np.logspace(-1, 3, num=205)
freq_array_ebl = c.value / (waves_ebl * 1e-6)

measurements.ebl(
    lambda_min_total=0.08, fig=fig_cob, ax=ax_cob,
    colors_UL=['grey'], plot_UL=True, plot_IGL=True,
    show_legend=True)


# Axion component calculation

wv_alp, int_alp = ebl_class.ebl_axion_calculation(
    wavelength=waves_ebl, zz_array=0.,
    axion_mass=float(config_data['axion_params']['axion_mass']),
    axion_gayy=float(config_data['axion_params']['axion_gamma'])
)
plt.loglog(wv_alp, int_alp, linestyle=models[3], color='k',
           label=r'$m_a=$%s eV  $g_{a \gamma} = %s $ GeV$^{-1}$'
                 % (config_data['axion_params']['axion_mass'],
                    config_data['axion_params']['axion_gamma']))

wv_alp, int_alp = ebl_class.ebl_axion_calculation(
    wavelength=waves_ebl, zz_array=0.,
    axion_mass=10., axion_gayy=1e-11)
plt.loglog(wv_alp, int_alp, linestyle=models[3], color='r',
           label=r'$m_a=10$eV  $g_{a \gamma} = 1e-11$ GeV$^{-1}$')

legend22 = plt.legend(loc=3,# bbox_to_anchor=(0.5, 0.1),
                      title=r'Cosmic ALP decay',
                      fontsize=14, title_fontsize=16, framealpha=0.95)

# Intrahalo component calculation
# ebl_class.ebl_intrahalo_calculation(float(
#                                       config_data['ihl_params']['A_ihl']),
#                                     float(
#                                     config_data['ihl_params']['alpha']))
# plt.plot(waves_ebl, 10 ** ebl_class.ebl_ihl_spline(
# freq_array_ebl, 0., grid=False),
# linestyle=models[2], color='k')
plt.figure()

emiss_data = measurements.emissivity(
    z_min=None, z_max=None,
    lambda_min=0., lambda_max=3e3,
    take_only_refs=None, plot=False)

plt.scatter(x=emiss_data['lambda'], y=emiss_data['z'],
            c=np.log10(emiss_data['eje']),
            cmap='viridis')

plt.xscale('log')
plt.yscale('log')

fig_emiss_lambda, (ax_emiss_lambda0, ax_emiss_lambda1) = plt.subplots(
    2, 1, figsize=(10, 14))
plt.subplot(211)
for nz, zz in enumerate(np.unique(emiss_data['z'])):
    ax_emiss_lambda0.scatter(x=emiss_data['lambda'][emiss_data['z'] == zz],
                y=emiss_data['eje'][emiss_data['z'] == zz],
            color=plt.cm.CMRmap(nz / float(len(np.unique(emiss_data['z'])))),
                )

ax_emiss_lambda0.set_xscale('log')
ax_emiss_lambda0.set_yscale('log')

plt.ylabel(r'$\nu \varepsilon_{\nu}$ (W / Mpc**3)')

plt.subplot(212, sharex=ax_emiss_lambda0)

ax_emiss_lambda1.set_xscale('log')
# ax_emiss_lambda1.set_yscale('log')

plt.axhline(1, c='grey', zorder=0)
plt.axhline(-1, c='grey', zorder=0)

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(
    r'$\frac{\mathrm{model}(\lambda_\mathrm{i})- \nu \varepsilon_{\nu,'
    r'\mathrm{i}}}'
    r'{\sigma_\mathrm{i}}$')
# plt.show()

fig_emiss_z, axes_emiss_z = plt.subplots(4, 3, figsize=(12, 16))

z_array = np.linspace(1e-9, 10., num=100)

for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                               0.44, 0.55, 0.79,
                               1.22, 2.2, 3.6,
                               4.5, 5.8, 8.0]):
    ax = plt.subplot(4, 3, n_lambda + 1)
    measurements.emissivity(
        z_min=None, z_max=None,
        lambda_min=ll - 0.05, lambda_max=ll + 0.05,
        plot=True, ax=ax)

    # if n_lambda != 8:
    plt.annotate(r'%r$\,\mu m$' % ll, xy=(5, 1e35), fontsize=28)

    plt.xlim(min(z_array), max(z_array))
    plt.ylim(1e33, 3e35)

    plt.yscale('log')

handles_emiss, labels_emiss = [], []

plt.subplot(4, 3, 11)
plt.xlabel(r'redshift z', fontsize=34)

plt.subplot(4, 3, 4)
plt.ylabel(r'$_{\nu} \varepsilon_{_{\nu} \,\,(\mathrm{W\, / \, Mpc}^3)}$',
           fontsize=40)

# plt.subplot(3, 3, 9)
# plt.annotate(r'3.6$\,\mu m$', xy=(6, 1e34), fontsize=28)

ax = [plt.subplot(4, 3, i) for i in [2, 3, 5, 6, 8, 9, 11, 12]]
for a in ax:
    a.set_yticklabels([])

ax = [plt.subplot(4, 3, i + 1) for i in range(9)]
for a in ax:
    a.set_xticklabels([])

ax = [plt.subplot(4, 3, i + 1) for i in range(9, 12)]
for a in ax:
    a.set_xticks([0, 2, 4, 6, 8])

ax = [plt.subplot(4, 3, i) for i in range(1, 10)]
for a in ax:
    a.set_xticks([0, 2, 4, 6, 8, 10])

a = plt.subplot(4, 3, 12)
a.set_xticks([0, 2, 4, 6, 8, 10])

# SFRs FOR SSP MODELS ------------------------------
fig_sfr, ax_sfr = plt.subplots(figsize=(12, 8))

z_data = np.linspace(float(config_data['redshift_array']['z_min']),
                     float(config_data['redshift_array']['z_max']),
                     num=500)

sfr_data = measurements.sfr(plot=True, ax=ax_sfr)


plt.yscale('log')
plt.ylabel('sfr(z)')
plt.xlabel('z')

# SSP SYNTHETIC SPECTRA

fig_ssp, ax_ssp = plt.subplots(figsize=(10, 8))
plt.xscale('log')
plt.title('More transparency, less metallicity')
# plt.yscale('log')
# plt.ylim(0., 30.)

xx_amstrongs = np.logspace(2, 6, 2000)

ax_ssp.set_xlabel('Wavelength [A]')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{\AA}^{-1}$ M$_{\odot}^{-1}$])')

previous_ssp = []
labels_ssp1 = []
handles_ssp1 = []
labels_ssp2 = []
handles_ssp2 = []

fig_dustabs, ax_dustabs = plt.subplots()
wv_dustabs = np.logspace(-2, 1, num=5000)
zz_dustabs = np.array([0, 2, 4, 6])
color_dustabs = []

# print('%.3f' %(memory_usage_psutil()))

# SSPs component calculation (all models listed in the input file)
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    ebl_class.write_ebl_to_ascii(output_path=input_file_dir, name=key)

    print(ebl_class.ebl_ssp_spline(0.608, 0.),
          21.98 - ebl_class.ebl_ssp_spline(0.608, 0.))
    # print('%.3f' % (memory_usage_psutil()))

    ax_cob.plot(waves_ebl, ebl_class.ebl_ssp_spline(waves_ebl, 0.),
                linestyle='-', color=colors[nkey % len(colors)],
                lw=2,
                # markersize=16, marker=markers[nkey]
                )

    plt.figure(fig_emiss_lambda)
    for nz, zz in enumerate(np.unique(emiss_data['z'])):
        plt.subplot(211)
        ax_emiss_lambda0.plot(waves_ebl, ebl_class.emiss_ssp_spline(
                     waves_ebl, zz) * freq_array_ebl * 1e-7,
                    color=plt.cm.CMRmap(
                        nz / float(len(np.unique(emiss_data['z'])))),
                    zorder=0, alpha=0.75, ls=linstyles_ssp[nkey])
        freq_arr_emiss = (3e8*1e6/emiss_data['lambda'][emiss_data['z'] == zz])

        plt.subplot(212)
        plt.plot(
            emiss_data['lambda'][emiss_data['z'] == zz],
            (ebl_class.emiss_ssp_spline(
                emiss_data['lambda'][emiss_data['z'] == zz], zz)
               * freq_arr_emiss * 1e-7
               - emiss_data['eje'][emiss_data['z'] == zz])
              / ((emiss_data['eje_n'][emiss_data['z'] == zz]
                  + emiss_data['eje_p'][emiss_data['z'] == zz]) / 2.),
                    color=plt.cm.CMRmap(
                        nz / float(len(np.unique(emiss_data['z'])))),
            ls='', marker=markers[nkey]
                    )

    plt.figure(fig_emiss_z)
    for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                                   0.44, 0.55, 0.79,
                                   1.22, 2.2, 3.6,
                                   4.5, 5.8, 8.0]):
        plt.subplot(4, 3, n_lambda + 1)

        plt.plot(z_array,
                 (c.value / (ll * 1e-6))
                 * ebl_class.emiss_ssp_spline(
                     ll * np.ones(len(z_array)), z_array)
                 * 1e-7,
                 linestyle='-',  # marker=markers[nkey],
                 color=colors[nkey % len(colors)], lw=2)

    labels_emiss.append(
        config_data['ssp_models'][key]['name'])
    handles_emiss.append(plt.Line2D([], [], linewidth=2,
                                    linestyle='-',
                                    color=colors[nkey % len(colors)]))

    ax_met.plot(
        z_array,
        ebl_model.metall_model(
            zz_array=z_array,
            metall_model=config_data['ssp_models'][key]['metall_formula'],
            metall_params=config_data['ssp_models'][key]['metall_params']
        ),
        label=config_data['ssp_models'][key]['name'],
        color=colors[nkey % len(colors)])

    ax_sfr.plot(
        z_data, ebl_model.sfr_model(
            zz_array=z_data,
            sfr_model=config_data['ssp_models'][key]['sfr_formula'],
            sfr_params=config_data['ssp_models'][key]['sfr_params']),
        label=config_data['ssp_models'][key]['name'],
        color=colors[nkey % len(colors)])

    color_ssp = ['b', 'orange', 'k', 'r', 'green', 'grey', 'limegreen',
                 'purple', 'brown']

    if ('path_ssp' not in config_data['ssp_models'][key]['ssp']
            or
            config_data['ssp_models'][key]['ssp']['path_ssp']
            not in previous_ssp):
        # Check so the examples are only plotted once
        if (config_data['ssp_models'][key]['ssp']['ssp_type']
                in previous_ssp):
            continue
        try:
            previous_ssp.append(
                config_data['ssp_models'][key]['ssp']['path_ssp'])
            labels_ssp2.append(
                config_data['ssp_models'][key]['ssp']['path_ssp'])
        except KeyError:
            previous_ssp.append(
                config_data['ssp_models'][key]['ssp']['ssp_type'])
            labels_ssp2.append(
                config_data['ssp_models'][key]['ssp']['ssp_type'])

        handles_ssp2.append(
            plt.Line2D([], [], linewidth=2,
                       linestyle=linstyles_ssp[len(previous_ssp) - 1],
                       color='k'))

        list_met = np.sort(ebl_class._ssp_metall)

        for n_met, met in enumerate(list_met):
            for i, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
                ax_ssp.plot(
                    xx_amstrongs,
                    ebl_class.ssp_lumin_spline(
                        wv_array=xx_amstrongs, age_array=age,
                        metall_array=np.log10(met)),
                    linestyle=linstyles_ssp[len(previous_ssp) - 1],
                    color=color_ssp[i],
                    alpha=float(n_met) / len(list_met) * 1.1
                )
                if n_met == 0 and len(previous_ssp) == 1:
                    labels_ssp1.append(age)
                    handles_ssp1.append(
                        plt.Line2D([], [], linewidth=2, linestyle='-',
                                   color=color_ssp[i]))

    plt.figure(fig_dustabs)
    if nkey == 0:
        for ni, ii in enumerate(zz_dustabs):
            if ni == 0:
                color_dustabs.append(
                    plt.cm.CMRmap(ni / float(len(zz_dustabs))))

                plt.plot(wv_dustabs,
                         ebl_model.dust_abs_fraction(
                             wv_array=wv_dustabs,
                             models=config_data['ssp_models'][key][
                                 'dust_abs_models'],
                             z_array=zz_dustabs[ni],
                             dust_params=config_data['ssp_models'][key][
                                 'dust_abs_params'],
                             verbose=False),
                         ls=linstyles_ssp[nkey], c=color_dustabs[ni],
                         alpha=1, label=key)

            else:
                color_dustabs.append(
                    plt.cm.CMRmap(ni / float(len(zz_dustabs))))
                plt.plot(wv_dustabs,
                         ebl_model.dust_abs_fraction(
                             wv_array=wv_dustabs,
                             models=config_data['ssp_models'][key][
                                 'dust_abs_models'],
                             z_array=zz_dustabs[ni],
                             dust_params=config_data['ssp_models'][key][
                                 'dust_abs_params'],
                             verbose=False),
                         ls=linstyles_ssp[nkey], c=color_dustabs[ni],
                         alpha=1)

# print('%.3f' %(memory_usage_psutil()))
plt.figure(fig_cob)

# We introduce the Finke22 and CUBA splines
# ebl = {}
# for m in EBL.get_models():
#     ebl[m] = EBL.readmodel(m)
# nuInu = {}
# for m, e in ebl.items():
#     nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
# spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
# spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)
# ax_cob.plot(waves_ebl, spline_finke(waves_ebl),
#             c='orange', label='Finke22 model A')
# ax_cob.plot(waves_ebl, spline_cuba(waves_ebl),
#             c='fuchsia', label='CUBA')


plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength (µm)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

legend33 = plt.legend(
    [plt.Line2D([], [], linewidth=2, linestyle='-', color=colors[i])
     for i in range(len(config_data['ssp_models']))],
    [config_data['ssp_models'][key]['name']
     for key in config_data['ssp_models']],
    title=r'SSP models',
    # bbox_to_anchor=(0.99, 0.01),
    loc=4, ncol=1, fontsize=14, framealpha=0.9
).set_zorder(200)

# ax_ssp.add_artist(legend11)
ax_cob.add_artist(legend22)


plt.xlim([.1, 1000])
plt.ylim(1e-2, 100)

ax_sfr.legend(loc=3)
ax_met.legend()
print(previous_ssp)

plt.figure(fig_ssp)
legend11 = plt.legend(handles_ssp1, labels_ssp1,
                      loc=4,
                      title=r'Age (log$_{10}$(years))')
legend22 = plt.legend(handles_ssp2, labels_ssp2,
                      loc=1,
                      )
ax_ssp.add_artist(legend11)
ax_ssp.add_artist(legend22)

plt.ylim(bottom=0)

plt.figure(fig_dustabs)

plt.ylabel('Escape fraction of photons')
plt.xlabel(r'Wavelength ($\mu$m)')

# aa = plt.legend()
bb = plt.legend([plt.Line2D([], [], linewidth=2,
                            linestyle='-', color=color_dustabs[i])
                 for i in range(len(zz_dustabs))],
                zz_dustabs, title='Redshift')
# ax_dustabs.add_artist(aa)
ax_dustabs.add_artist(bb)

plt.xscale('log')

plt.ylim(0., 1.1)
plt.xlim(0.05, 10)

fig_cob.savefig(input_file_dir + '/ebl_bare' + '.png',
                bbox_inches='tight')
fig_cob.savefig(input_file_dir + '/ebl_bare' + '.pdf',
                bbox_inches='tight')

fig_sfr.savefig(input_file_dir + '/sfr_bare' + '.png',
                bbox_inches='tight')
fig_sfr.savefig(input_file_dir + '/sfr_bare' + '.pdf',
                bbox_inches='tight')

fig_met.savefig(input_file_dir + '/Zev_bare' + '.png',
                bbox_inches='tight')
fig_met.savefig(input_file_dir + '/Zev_bare' + '.pdf',
                bbox_inches='tight')

fig_ssp.savefig(input_file_dir + '/ssp_bare' + '.png',
                bbox_inches='tight')
fig_ssp.savefig(input_file_dir + '/ssp_bare' + '.pdf',
                bbox_inches='tight')

fig_emiss_z.subplots_adjust(wspace=0, hspace=0)
fig_emiss_z.savefig(
    input_file_dir + '/emiss_redshift_bare' + '.png',
    bbox_inches='tight')
fig_emiss_z.savefig(
    input_file_dir + '/emiss_redshift_bare' + '.pdf',
    bbox_inches='tight')

fig_dustabs.savefig(
    input_file_dir + '/dustabs' + '.png',
    bbox_inches='tight', dpi=500)
fig_dustabs.savefig(
    input_file_dir + '/dustabs' + '.pdf',
    bbox_inches='tight')

plt.show()
