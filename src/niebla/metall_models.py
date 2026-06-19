import numpy as np
import logging
from src.niebla.safe_evaluation_strings import safe_formula

logger = logging.getLogger(__name__)


def _metall_tanikawa22(zz_array, metall_params=None, verbose=False):
    """
    Mean metallicity evolution of the Universe as a function of redshift.
    Eq. 4 from
    https://iopscience.iop.org/article/10.3847/1538-4357/ac4247/pdf

    zz_array: nD array
        Redshift values to input in the SFRD formula.
    metall_params: 1D array or list
        Optional parameters that can enter the metallicity formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

    :return: nD array
        Mean metallicity values with the same shape of the zz_array.
        Units: 1
    """
    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)
    if metall_params is None:
        metall_params = [0.153, 0.074, 1.34, 0.02]
        if verbose:
            logger.info(
                '   -> Metallicity: default parameters chosen: %s',
                metall_params)
    return 10 ** (metall_params[0]
                  - metall_params[1] * zz_array ** metall_params[2]
                  ) * metall_params[3]


def _metall_constant(zz_array, metall_params=None, verbose=False):
    """
    Mean metallicity evolution of the Universe as a function of redshift.
    Constant value.

    zz_array: nD array
        Redshift values to input in the SFRD formula.
    metall_params: 1D array or list
        Optional parameters that can enter the metallicity formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

    :return: nD array
        Mean metallicity values with the same shape of the zz_array.
        Units: 1
    """
    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)

    if isinstance(metall_params, list) or isinstance(metall_params, np.ndarray):
        if len(metall_params) != 1:
            raise ValueError(
                "Constant metallicity only accepts exactly "
                "one parameter in metall_params: %s", metall_params)
        metall_params = float(metall_params[0])
    if metall_params is None:
        metall_params = [0.02]
        if verbose:
            logger.info(
                '   -> Metallicity: default parameters chosen: %s',
                metall_params)
    return np.ones_like(zz_array) * metall_params


# ---------------------------------------------------------------------
model_list = {
    'metall_tanikawa22': _metall_tanikawa22,
    'metall_constant': _metall_constant
}


def metall_model(zz_array, metall_model, metall_params=None, verbose=False):
    """
    Z(z) is the mean metallicity of the Universe as a function of
    redshift.

    Units: unitless [1]

    zz_array: nD array
        Redshift values to input in the formula.
    metall_model: string or callable
        Formula (analytical or numerical) of the metallicity.
    metall_params: 1D array or list
        Optional parameters that can enter the metallicity formula.
    verbose: boolean
        Print the values of the input parameters if the default are used.

    :return: nD array
        Mean metallicity values with the same shape of the zz_array.
        Units: 1
    """
    if isinstance(zz_array, list):
        zz_array = np.array(zz_array)

    if metall_model in model_list.keys():
        return model_list[metall_model](
            zz_array=zz_array, metall_params=metall_params, verbose=verbose)

    if isinstance(metall_model, str):
        try:
            return safe_formula(metall_model, xx=zz_array, params=metall_params)

        except Exception as e:
            raise ValueError(
                f'Error evaluating string formula ' + metall_model
                + f' with z_array {zz_array}'
                  f'and sfr_params {metall_params}: {e}') from e

    elif callable(metall_model):
        if metall_params is not None:
            if isinstance(metall_params, dict):
                return metall_model(zz_array, **metall_params)
            return metall_model(zz_array, metall_params)
        else:
            return metall_model(zz_array)
    raise ValueError(f"Unrecognized sfr_model type: {type(metall_model)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
