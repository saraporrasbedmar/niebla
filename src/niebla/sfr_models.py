import numpy as np
import logging
from src.niebla.safe_evaluation_strings import safe_formula

logger = logging.getLogger(__name__)


def _sfr_madau14(zz_array, sfr_params=None, verbose=False):
    """
    Stellar Formation Rate Density (SFRD) is the density of stars
    that are born as a function of redshift.
    Eq. 15 from
    https://www.annualreviews.org/content/journals/10.1146/annurev-astro-081811-125615#f9

    zz_array: nD array
        Redshift values to input in the SFRD formula.
    sfr_params: 1D array or list
        Optional parameters that can enter the SFRD formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

    :return: nD array
        SFRD values with the same shape of the zz_array.
        Units: M_Sun / yr / Mpc^3
     """
    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)
    if sfr_params is None:
        sfr_params = [0.015, 2.7, 2.9, 5.6]
        if verbose:
            logger.info('   -> SFR: default parameters chosen: %s', sfr_params)
    return (sfr_params[0] * (1 + zz_array) ** sfr_params[1]
            / (1 + ((1 + zz_array) / sfr_params[2]) ** sfr_params[3]))


def _sfr_finke22a(zz_array, sfr_params=None, verbose=False):
    """
    Stellar Formation Rate Density (SFRD) is the density of stars
    that are born as a function of redshift.
    Eq. 15 from https://iopscience.iop.org/article/10.3847/1538-4357/ac9843

    zz_array: nD array
        Redshift values to input in the SFRD formula.
    sfr_params: 1D array or list
        Optional parameters that can enter the SFRD formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

    :return: nD array
        SFRD values with the same shape of the zz_array.
        Units: M_Sun / yr / Mpc^3
     """
    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)
    if sfr_params is None:
        sfr_params = [-2.04, 2.81, 1.25, -1.25, -1.84, -4.40, 1., 2., 3., 4.]
        if verbose:
            logger.info('   -> SFR: default parameters chosen: %s', sfr_params)
    return (10 ** sfr_params[0] * (
            ((1 + zz_array) ** sfr_params[1] * (zz_array < sfr_params[-4]))
            + ((1 + sfr_params[-4]) ** (sfr_params[1] - sfr_params[2]) * (1 + zz_array) **
               sfr_params[2] * (zz_array >= sfr_params[-4]) * (zz_array < sfr_params[-3]))
            + ((1 + sfr_params[-4]) ** (sfr_params[1] - sfr_params[2]) * (
            1 + sfr_params[-3]) ** (sfr_params[2] - sfr_params[3]) * (1 + zz_array) **
               sfr_params[3] * (zz_array >= sfr_params[-3]) * (zz_array < sfr_params[-2]))
            + ((1 + sfr_params[-4]) ** (sfr_params[1] - sfr_params[2]) * (
            1 + sfr_params[-3]) ** (sfr_params[2] - sfr_params[3]) * (
                       1 + sfr_params[-2]) ** (sfr_params[3] - sfr_params[4]) * (
                       1 + zz_array) ** sfr_params[4] * (
                       zz_array >= sfr_params[-2]) * (zz_array < sfr_params[-1]))
            + ((1 + sfr_params[-4]) ** (sfr_params[1] - sfr_params[2]) * (
            1 + sfr_params[-3]) ** (sfr_params[2] - sfr_params[3]) * (
                       1 + sfr_params[-2]) ** (sfr_params[3] - sfr_params[4]) * (
                       1 + sfr_params[-1]) ** (sfr_params[4] - sfr_params[5]) * (
                       1 + zz_array) ** sfr_params[5] * (
                       zz_array >= sfr_params[-1]))))


def _sfr_cuba(zz_array, sfr_params=None, verbose=False):
    """
    Stellar Formation Rate Density (SFRD) is the density of stars
    that are born as a function of redshift.
    Eq. 53 from https://iopscience.iop.org/article/10.1088/0004-637X/746/2/125

    zz_array: nD array
        Redshift values to input in the SFRD formula.
    sfr_params: 1D array or list
        Optional parameters that can enter the SFRD formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

    :return: nD array
        SFRD values with the same shape of the zz_array.
        Units: M_Sun / yr / Mpc^3
     """

    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)
    if sfr_params is None:
        sfr_params = [6.9e-3, 0.14, 2.2, 1.5, 2.7, 4.1]
        if verbose:
            logger.info('   -> SFR: default parameters chosen: %s', sfr_params)
    return ((sfr_params[0] + sfr_params[1] * (zz_array / sfr_params[2]) ** sfr_params[3])
            / (1. + (zz_array / sfr_params[4]) ** sfr_params[5]))


def _sfr_constant(zz_array, sfr_params=None, verbose=False):
    """
    Stellar Formation Rate Density (SFRD) is the density of stars
    that are born as a function of redshift.
    Constant SFRD.

    zz_array: nD array
        Redshift values to input in the formula.
    sfr_params: 1D array or list
        Optional parameters that can enter the sfr formula.
    verbose: boolean
       Print the values of the input parameters if the default are used.

    :return: nD array
        SFRD values with the same shape of the zz_array.
        Units: M_Sun / yr / Mpc^3
    """
    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)
    if isinstance(sfr_params, list) or isinstance(sfr_params, np.ndarray):
        if len(sfr_params) != 1:
            raise ValueError(
                "SFRD only accepts exactly "
                "one parameter in sfr_params: %s", sfr_params)
        sfr_params = float(sfr_params[0])
    if sfr_params is None:
        sfr_params = 1.
        if verbose:
            logger.info('   -> SFR: default parameters chosen: %s', sfr_params)
    return np.ones_like(zz_array) * sfr_params


# ---------------------------------------------------------------------

model_list = {
    'sfr_madau14': _sfr_madau14,
    'sfr_finke22a': _sfr_finke22a,
    'sfr_cuba': _sfr_cuba,
    'sfr_constant': _sfr_constant
}


def sfr_model(zz_array, sfr_model, sfr_params=None, verbose=False):
    """
    Stellar Formation Rate Density (SFRD) is the density of stars
    that are born as a function of redshift.

     zz_array: nD array
         Redshift values to input in the formula.
     sfr_model: string or callable
         Formula (analytical or numerical) of the SFRD.
     params: 1D array or list
         Optional parameters that can enter the sfr formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

     :return: nD array
         SFRD values with the same shape of the zz_array.
         Units: M_Sun / yr / Mpc^3
     """

    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)

    if sfr_model in model_list.keys():
        return model_list[sfr_model](
            zz_array=zz_array, sfr_params=sfr_params, verbose=verbose)

    if isinstance(sfr_model, str):
        try:
            return safe_formula(sfr_model, xx=zz_array, params=sfr_params)

        except Exception as e:
            raise ValueError(
                f'Error evaluating string formula ' + sfr_model
                + f' with z_array {zz_array}'
                  f'and sfr_params {sfr_params}: {e}') from e

    elif callable(sfr_model):
        if sfr_params is not None:
            if isinstance(sfr_params, dict):
                return sfr_model(zz_array, **sfr_params)
            return sfr_model(zz_array, sfr_params)
        else:
            return sfr_model(zz_array)
    raise ValueError(f"Unrecognized sfr_model type: {type(sfr_model)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
