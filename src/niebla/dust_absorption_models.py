import numpy as np
import logging
from src.niebla.safe_evaluation_strings import safe_formula

logger = logging.getLogger(__name__)


def dust_abs_fraction(wv_array, z_array=0.,
                      models=None,
                      dust_params=None,
                      verbose=False):
    """
    Function to calculate the fraction of photons which will escape
    absorption by dust. Dependencies both with wavelength and redshift.

    Parameters:
    :param wv_array: float or array  [microns]
        Wavelength values at which to calculate the dust absorption.

    :param z_array: float or array
        Redshift values at which to calculate the dust absorption.

    :param models: list of strings or None
        Models of dust fraction as a function of wavelength
        and redshift.

        -> If 2 strings are given, the first model is applied for
            wavelength and the second for redshift dependence.
            The models will usually assume that these two dependencies
            are multiplied.
            If either of them does not correspond to any listed models,
            no dust absorption will be calculated.
            - Wavelength accepted values:
                kneiske2002, razzaque2009, dust_att_finke
            - Redshift accepted values:
                fermi2018

        -> If 1 string is given, a combined wavelength and redshift
            model will be applied.
            - Accepted values: comb_model_1, finke2022

        -> If the number of strings is neither 1 nor 2, no dust
            absorption model will be applied.

    :param dust_params: dictionary
        Desired parameters for any of the listed functions of dust
        absorption. Be careful when adding new functions,
        not to overlap two different parameters under the same name.

    :param verbose: boolean
        Choose whether to print all the outputs of the calculations.


    Outputs:
    :return: 2D array with shape (wv_array x z_array)
        Result of the calculation, as a fraction between [0, 1].
    """

    wv_array = np.asarray(wv_array, dtype=float)
    z_array = np.asarray(z_array, dtype=float)
    zz, ww = np.meshgrid(z_array, wv_array)

    if callable(models):

        if dust_params is None:
            dust_att = models(ww, zz)
        else:
            dust_att = models(ww, zz, dust_params)

        return np.clip(np.asarray(dust_att, dtype=float), 0., 1.)

    if models is None:
        if verbose:
            logger.info(
                '   -> Dust absorption: no model chosen, list of '
                        'models: %s', models)
        return np.ones(
            (wv_array.shape[0], z_array.shape[0]), dtype=float)

    if isinstance(models, str):
        models = [models]
    else:
        models = models

    try:
        n_models = len(models)
    except Exception as e:
        raise TypeError(
            'The input models must be None, a string, a list of strings,'
            ' or a callable.') from e

    # The absorption models are defined in one definition
    if n_models == 1:
        models = models[0]

        if models == 'comb_model_1':
            dust_att = comb_model_1(wv_array, z_array, dust_params,
                                    verbose=verbose)

        elif models == 'finke2022':
            dust_att = finke2022(wv_array, z_array, dust_params,
                                 verbose=verbose)

        else:
            if not isinstance(models, str):
                raise TypeError(
                    'When models has one element, '
                    'it must be a string (model name or expression).'
                    ' In this case, models: %s', models)

            try:
                return safe_formula(
                    models, xx=(ww, zz), params=dust_params)

            except Exception as e:
                raise ValueError(
                    '    -> No dust absorption dependency with either'
                    ' wavelength or redshift. '
                    'If the string is a expression to evaluate,'
                    ' there is something wrong in it, check it.' + '\n'
                    f'Error evaluating string formula ' + models
                    + f' with wavelength_array {wv_array}'
                    + f' with z_array {z_array}'
                      f'and dust_params {dust_params}: {e}') from e


    # The absorption models for wavelength and redshift are
    # separately defined
    elif n_models == 2:

        dust_att = np.zeros(
            [np.shape(wv_array)[0], np.shape(z_array)[0]])

        # Wavelength dependency
        if models[0] == 'kneiske2002':
            dust_att += kneiske2002(ww, dust_params, verbose=verbose)

        elif models[0] == 'razzaque2009':
            dust_att += razzaque2009(ww, dust_params, verbose=verbose)

        elif models[0] == 'dust_att_finke':
            dust_att += dust_att_finke(ww, dust_params, verbose=verbose)

        else:
            if verbose:
                logger.info(
                    '   -> No dust absorption dependency with wavelength.')

        # Redshift dependency
        if models[1] == 'fermi2018':
            dust_att += fermi2018(zz, dust_params, verbose=verbose)

        else:
            if verbose:
                logger.info('   -> No dust absorption dependency with redshift.')

    else:
        raise ValueError(
            "models must have length 1 or 2 (when not callable),"
            " or be None/string.")

    dust_att = 10 ** dust_att
    dust_att[np.isnan(dust_att)] = -43.
    dust_att[~(np.isfinite(dust_att))] = -43.
    return dust_att


def kneiske2002(wv_array, dust_params, verbose=False):
    """
    Dust attenuation as a function of wavelength following
    Kneiske02 or 0202104

    :param wv_array: float or array  [microns]
        Wavelength values to compute dust absorption.
    :param Ebv_Kn02: float
        E(B-V) or color index
    :param R_Kn02: float
        Random index
    """
    try:
        Ebv_Kn02 = dust_params['params_kneiske2002'][0]
        R_Kn02 = dust_params['params_kneiske2002'][1]
    except KeyError or TypeError:
        Ebv_Kn02 = 0.15
        R_Kn02 = 3.2
        if verbose:
            logger.info(
                '   -> Default parameter for Ebv, R chosen: %s, %s',
                Ebv_Kn02, R_Kn02)

    return (np.minimum(-.4 * Ebv_Kn02 * .68 * R_Kn02
                       * (1. / wv_array - .35), 0.))


def razzaque2009(wv_array, dust_params, verbose=False):
    """
    Dust attenuation as a function of wavelength following
    Razzaque09 or 0807.4294

    :param wv_array: float or array  [microns]
        Wavelength values to compute dust absorption.
    """
    try:
        lambda_cuts_rz09 = dust_params['lambda_cuts_rz09']
    except KeyError or TypeError:
        lambda_cuts_rz09 = [0.165, 0.220, 0.422]
        if verbose:
            logger.info('   -> Default parameters for '
                  'lambda_cuts_rz09 chosen:  %s', lambda_cuts_rz09)

    try:
        initial_value_rz09 = dust_params['initial_value_rz09']
    except KeyError or TypeError:
        initial_value_rz09 = [0.688, 0.151, 1.0, 0.728]
        if verbose:
            logger.info('   -> Default parameters for '
                  'initial_value_rz09 chosen:  %s', initial_value_rz09)

    try:
        multipl_factor_rz09 = dust_params['multipl_factor_rz09']
    except KeyError or TypeError:
        multipl_factor_rz09 = [0.556, -0.136, 1.148, 0.422]
        if verbose:
            logger.info('   -> Default parameters for '
                  'multipl_factor_rz09 chosen:  %s', multipl_factor_rz09)

    yy = np.zeros(np.shape(wv_array))
    yy += ((initial_value_rz09[0]
            + multipl_factor_rz09[0] * np.log10(wv_array))
           * (wv_array < lambda_cuts_rz09[0]))
    yy += ((initial_value_rz09[1]
            + multipl_factor_rz09[1] * np.log10(wv_array))
           * (wv_array < lambda_cuts_rz09[1])
           * (wv_array > lambda_cuts_rz09[0]))
    yy += ((initial_value_rz09[2]
            + multipl_factor_rz09[2] * np.log10(wv_array))
           * (wv_array < lambda_cuts_rz09[2])
           * (wv_array > lambda_cuts_rz09[1]))
    yy += ((initial_value_rz09[3]
            + multipl_factor_rz09[3] * np.log10(wv_array))
           * (wv_array > lambda_cuts_rz09[2]))
    yy[yy < 1e-43] = 1e-43
    return np.log10(yy)


def fermi2018(z_array, params_dust=None, verbose=False):
    """
    Dust attenuation as a function of redshift following Abdollahi18
    or 1812.01031, in the supplementary material.
    (supplement in https://pubmed.ncbi.nlm.nih.gov/30498122/).
    Result in log10(dust_att).

    :param z_array: float or array
        Redshift values to compute dust absorption.
    :param params_fermi18: array or None
        Parameters following the fitting of the paper.
        Order: [m_d, n_d, p_d, q_d]
    """
    try:
        params_fermi18 = params_dust['params_fermi18']
    except KeyError or TypeError:
        params_fermi18 = [1.49, 0.64, 3.4, 3.54]
        if verbose:
            logger.info(
                '   -> Default parameters for params_fermi18 chosen: %s',
                  params_fermi18)

    return (-0.4 * params_fermi18[0]
            * (1. + z_array) ** params_fermi18[1]
            / (1. + ((1. + z_array) / params_fermi18[2]) ** params_fermi18[3]))


def dust_att_finke(wv_array, params_dust=None, verbose=False):
    """

    :param wv_array:
    :param lambda_steps_fn22:
    :param fesc_steps_fn22:
    :return:
    """
    try:
        lambda_steps_fn22 = params_dust['lambda_steps_fn22']
    except KeyError or TypeError:
        lambda_steps_fn22 = [0.15, 0.167, 0.218, 0.422, 2.]
        if verbose:
            logger.info('   -> Default parameters for '
                  'lambda_steps_fn22 chosen:  %s', lambda_steps_fn22)

    try:
        fesc_steps_fn22 = params_dust['fesc_steps_fn22']
    except KeyError or TypeError:
        fesc_steps_fn22 = np.array([1.88, 2.18, 2.93, 3.93, 8.57]) * 0.1
        if verbose:
            logger.info('   -> Default parameters for '
                  'fesc_steps_fn22 chosen:  %s', fesc_steps_fn22)

    yy = np.zeros(np.shape(wv_array))
    yy += ((fesc_steps_fn22[1]
            + (fesc_steps_fn22[1] - fesc_steps_fn22[0])
            / (np.log10(lambda_steps_fn22[1] / lambda_steps_fn22[0]))
            * (np.log10(wv_array) - np.log10(lambda_steps_fn22[1])))
           * (wv_array <= lambda_steps_fn22[1]))
    yy += ((fesc_steps_fn22[2]
            + (fesc_steps_fn22[2] - fesc_steps_fn22[1])
            / (np.log10(lambda_steps_fn22[2] / lambda_steps_fn22[1]))
            * (np.log10(wv_array) - np.log10(lambda_steps_fn22[2])))
           * (wv_array <= lambda_steps_fn22[2])
           * (wv_array > lambda_steps_fn22[1]))
    yy += ((fesc_steps_fn22[3]
            + (fesc_steps_fn22[3] - fesc_steps_fn22[2])
            / (np.log10(lambda_steps_fn22[3] / lambda_steps_fn22[2]))
            * (np.log10(wv_array) - np.log10(lambda_steps_fn22[3])))
           * (wv_array <= lambda_steps_fn22[3])
           * (wv_array > lambda_steps_fn22[2]))
    yy += ((fesc_steps_fn22[4]
            + (fesc_steps_fn22[4] - fesc_steps_fn22[3])
            / (np.log10(lambda_steps_fn22[4] / lambda_steps_fn22[3]))
            * (np.log10(wv_array) - np.log10(lambda_steps_fn22[4])))
           * (wv_array > lambda_steps_fn22[3]))

    yy[yy < 1e-43] = 1e-43
    return np.log10(yy)


def comb_model_1(wv_array, z_array, dust_params, verbose=False):
    """
    Dust attenuation as a function of wavelength and redshift
    following Finke22 or 2210.01157

    :param wv_array:  float or array  [microns]
        Wavelength values to compute dust absorption.
    :param z_array:  float or array
        Redshift values to compute dust absorption.
    :return:
    """
    wv_array = np.asarray(wv_array, dtype=float)
    z_array = np.asarray(z_array, dtype=float)
    zz, ww = np.meshgrid(z_array, wv_array)

    yy = fermi2018(zz, dust_params, verbose=verbose)
    yy += (razzaque2009(ww, dust_params, verbose=verbose)
           - razzaque2009(0.15, dust_params, verbose=False))
    return np.minimum(yy, 0)


def finke2022(wv_array, z_array, dust_params, verbose=False):
    """
    Dust attenuation as a function of wavelength and redshift
    following Finke22 or 2210.01157.
    Following the second definition, formula 13.

    :param wv_array:  float or array  [microns]
        Wavelength values to compute dust absorption.
    :param z_array:  float or array
        Redshift values to compute dust absorption.
    :return:
    """
    wv_array = np.asarray(wv_array, dtype=float)
    z_array = np.asarray(z_array, dtype=float)
    zz, ww = np.meshgrid(z_array, wv_array)

    yy = fermi2018(zz, dust_params, verbose=verbose)
    yy += (dust_att_finke(ww, dust_params, verbose=verbose)
           - dust_att_finke(0.15, dust_params, verbose=False))

    return np.minimum(yy, 0)
