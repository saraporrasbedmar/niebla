import os
import numpy as np

from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator

from astropy.io import fits
from astropy import units as u
import astropy.constants as c

from src.niebla.safe_log10 import log10_safe

data_path = os.path.join(os.path.split(__file__)[0], 'data/')
print("data_path", data_path)

def _dust_reem_chary2001(wave_array, l_tir_array, metall_array,
            params, verbose=True):
    """
    Provide the spline of dust reemission, assuming either BOSA
    or Chary2001 templates are being used. For other templates,
    please upload them manually.
    :param yaml_data: dictionary
    Tells which templates to use.
    :return:
    """
    f_tir = 10 ** float(params['f_tir'])
    chary = fits.open(
        data_path
        + 'ssp_synthetic_spectra/chary2001/chary_elbaz.fits')

    ir_wv = chary[1].data.field('LAMBDA')[0]
    ir_freq = c.c.value / ir_wv * 1e6

    ir_lum = (
            log10_safe(chary[1].data.field('NULNUINLSUN')[0])
            - np.log10(f_tir))

    aaa = np.zeros((np.shape(ir_lum)[0], np.shape(ir_lum)[1] + 2))
    aaa[:, 1:-1] = ir_lum
    aaa[:, 0] = ir_lum[:, 0] - 10.
    aaa[:, -1] = ir_lum[:, -1] + 10.
    ir_lum = aaa

    # Cap the dust reemisison to the wavelength where there is
    # proper reemission, not the whole possible spectrum
    ir_lum[ir_wv < params['wv_reem_min'], :] = -43

    l_tir = np.log(10) * simpson(
        10 ** ir_lum[::-1], x=np.log10(ir_freq)[::-1], axis=0)

    sort_order = np.argsort(l_tir)
    l_tir = l_tir[sort_order]
    l_tir *= c.L_sun.to(u.erg / u.s).value

    ir_lum -= np.log10(ir_freq[:, np.newaxis])
    ir_lum += np.log10(c.L_sun.to(u.erg / u.s).value)
    ir_lum[ir_lum < -43] = -43
    ir_lum = ir_lum[:, sort_order]

    ir_lum_expanded = np.zeros(
        (np.shape(ir_lum)[0], np.shape(ir_lum)[1], 2))
    ir_lum_expanded[:, :, 0] = ir_lum
    ir_lum_expanded[:, :, 1] = ir_lum

    dust_reem_spline = RegularGridInterpolator(
            points=(np.log10(ir_wv),
                    np.log10(l_tir),
                    [-43, 1.]),
            values=ir_lum_expanded,
            method='linear',
            bounds_error=False, fill_value=-43
        )

    return dust_reem_spline((wave_array, l_tir_array, metall_array))


def _dust_reem_bosa(wave_array, l_tir_array, metall_array,
                    params=None, verbose=True):

    data = fits.open(
        data_path + 'ssp_synthetic_spectra/bosa/Z.fits')
    aaa = data[1].data

    yyy = np.column_stack((
        aaa['nuLnu[Z=6.99103]'],
        aaa['nuLnu[Z=6.99103]'], aaa['nuLnu[Z=7.99103]'],
        aaa['nuLnu[Z=8.29205999]'], aaa['nuLnu[Z=8.69]'],
        aaa['nuLnu[Z=9.08794001]'], aaa['nuLnu[Z=9.08794001]']))
    yyy = (yyy * (c.L_sun.to(u.erg / u.s)).value
           * (aaa['wavelength'] * 1e-9 / c.c.value)[:, np.newaxis])

    yyy = log10_safe(yyy)

    yyy_whole = np.zeros((np.shape(yyy)[0], 2, np.shape(yyy)[1]))
    yyy_whole[:, 0, :] = yyy - np.log10(1e5)
    yyy_whole[:, 1, :] = yyy + np.log10(1e8)

    yyy_whole[np.isnan(yyy_whole)] = -43.
    yyy_whole[np.invert(np.isfinite(yyy_whole))] = -43.

    l_tir = np.array([1e-5, 1e8]) * c.L_sun.to(u.erg / u.s).value

    dust_reem_spline = RegularGridInterpolator(
        points=(np.log10(aaa['wavelength'] * 1e-3),
                np.log10(l_tir),
                np.log10([1e-43, 0.0004, 0.004,
                          0.008, 0.02, 0.05, 1.])),
        values=yyy_whole,
        method='linear',
        bounds_error=False, fill_value=-43.
    )
    return dust_reem_spline((wave_array, l_tir_array, metall_array))


def _dust_reem_custom(wave_array, l_tir_array, metall_array,
                      dust_reem_model,
                      params=None, verbose=True):

    if callable(dust_reem_model):
        if params is None:
            return dust_reem_model(
                wave_array, l_tir_array, metall_array)
        else:
            return dust_reem_model(
                wave_array, l_tir_array, metall_array, params)

    elif type(dust_reem_model) == str:

        ssp_metall = np.sort(np.array(
                os.listdir(dust_reem_model), dtype=float))

        # Open one of the files and check for wavelengths and Ltir
        data_generic = np.loadtxt(
            dust_reem_model + str(ssp_metall[0]))

        # Get unique wavelengths and Ltir
        ltir_total = data_generic[0, 1:]
        wave_total = data_generic[1:, 0]

        # Load all the luminosity data for different metallicities
        ssp_log_emis = np.zeros(
            (wave_total.shape[0],
             ltir_total.shape[0] + 2,
             len(ssp_metall) + 2))

        for n_met, met in enumerate(ssp_metall):
            data = np.loadtxt(
                dust_reem_model + str(met))
            ssp_log_emis[:, 1:-1, n_met + 1] = data[1:, 1:]

            ssp_log_emis[:, 0, n_met + 1] = (
                    ssp_log_emis[:, 1, n_met + 1] - 1e5)
            ssp_log_emis[:, -1, n_met + 1] = (
                    ssp_log_emis[:, -2, n_met + 1] + 1e5)

        # Extend the stellar spectra to very low metallicities
        ssp_metall = np.insert(ssp_metall, 0, 1e-43)
        ssp_log_emis[:, :, 0] = ssp_log_emis[:, :, 1]

        # Extend the stellar spectra to very high metallicities
        ssp_metall = np.append(ssp_metall, 1.)
        ssp_log_emis[:, :, -1] = ssp_log_emis[:, :, -2]


        # Calculate the logarithms of all the values
        ssp_log_metall = log10_safe(ssp_metall)
        ssp_log_ltir = log10_safe(ltir_total)
        ssp_log_wave = log10_safe(wave_total)
        ssp_log_emis = log10_safe(ssp_log_emis)

        ssp_log_ltir = np.insert(ssp_log_ltir, 0, ssp_log_ltir[0] - 5.)
        ssp_log_ltir = np.append(ssp_log_ltir, ssp_log_ltir[-1] + 5.)

        dust_reem_spline = RegularGridInterpolator(
            points=(ssp_log_wave, ssp_log_ltir, ssp_log_metall),
            values=ssp_log_emis,
            method='linear',
            bounds_error=False, fill_value=-43.
        )
        return dust_reem_spline((wave_array, l_tir_array, metall_array))

    else:
        NameError(
            'Unrecognized name/type of dust reemission.'
            '\nListed libraries: '
            'dust_reem_chary2001, dust_reem_bosa, dust_reem_custom .')


# ---------------------------------------------------------------------
model_list = {
    'dust_reem_chary2001': _dust_reem_chary2001,
    'dust_reem_bosa': _dust_reem_bosa
}


def dust_reem(wave_array, l_tir_array, metall_array,
              dust_reem_model, dust_reem_params=None,
              verbose=True):
    """
    Units:

    zz_array: nD array
        Redshift values to input in the formula.
    dust_reem_model: string or callable
        Formula (analytical or numerical) of the metallicity.
    params: 1D array or list
        Optional parameters that can enter the metallicity formula.

    :return: nD array
        Metallicity values with the same shape of the zz_array.
    """
    if dust_reem_model in model_list.keys():
        return model_list[dust_reem_model](
            wave_array, l_tir_array, metall_array,
            params=dust_reem_params, verbose=verbose)
    else:
        return _dust_reem_custom(
            wave_array, l_tir_array, metall_array,
            dust_reem_model=dust_reem_model,
            params=dust_reem_params, verbose=verbose)