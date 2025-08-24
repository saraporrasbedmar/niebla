import numpy as np


def _sfr_madau14(zz_array, params=None, verbose=True):
    """
    Eq. 15](https://www.annualreviews.org/content/journals/10.1146/annurev-astro-081811-125615#f9
    :param zz_array:
    :param params:
    :param verbose:
    :return:
    """
    if params is None:
        params = [0.015, 2.7, 2.9, 5.6]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return (params[0] * (1 + zz_array) ** params[1]
            / (1 + ((1 + zz_array) / params[2]) ** params[3]))


def _sfr_finke22a(zz_array, params=None, verbose=True):
    """

    :param zz_array:
    :param params:
    :param verbose:
    :return:
    """
    if params is None:
        params = [-2.04, 2.81, 1.25, -1.25, -1.84, -4.40, 1., 2., 3., 4.]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return (10 ** params[0] * (
            ((1 + zz_array) ** params[1] * (zz_array < params[-4]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (1 + zz_array) **
               params[2] * (zz_array >= params[-4]) * (zz_array < params[-3]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (
            1 + params[-3]) ** (params[2] - params[3]) * (1 + zz_array) **
               params[3] * (zz_array >= params[-3]) * (zz_array < params[-2]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (
            1 + params[-3]) ** (params[2] - params[3]) * (
                       1 + params[-2]) ** (params[3] - params[4]) * (
                       1 + zz_array) ** params[4] * (
                       zz_array >= params[-2]) * (zz_array < params[-1]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (
            1 + params[-3]) ** (params[2] - params[3]) * (
                       1 + params[-2]) ** (params[3] - params[4]) * (
                       1 + params[-1]) ** (params[4] - params[5]) * (
                       1 + zz_array) ** params[5] * (
                       zz_array >= params[-1]))))


def _sfr_cuba(zz_array, params=None, verbose=True):
    """
    Eq. 53 from https://iopscience.iop.org/article/10.1088/0004-637X/746/2/125
    :param zz_array:
    :param params:
    :param verbose:
    :return:
    """
    if params is None:
        params = [6.9e-3, 0.14, 2.2, 1.5, 2.7, 4.1]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return ((params[0] + params[1] * (zz_array / params[2]) ** params[3])
            / (1. + (zz_array / params[4]) ** params[5]))


def _sfr_constant(zz_array, params=None, verbose=True):
    if params is None:
        params = [1.]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return np.ones_like(zz_array) * params


def _sfr_custom(zz_array, sfr_model, params=None, verbose=True):

    if type(sfr_model) == str:
        try:
            return eval(sfr_model, {'zz': zz_array, 'ci': params})
        except NameError:
            raise NameError(
                'Unrecognized type of metallicity dependency.' + '\n'
                + 'Implemented models: ' + '\n'
                + str([*model_list]) + '\n'
                + 'If the string is a expression to evaluate,' + '\n'
                + 'there is something wrong in it, check it.' + '\n'
                + 'Independent variable must be called \'zz\' ' + '\n'
                + 'and parameters be an array or list called \'ci\'.' + '\n'
                + 'Inputs given:' + '\n'
                + '\'metall_model\': ' + str(sfr_model) + '\n'
                + '\'ci\': ' + str(params)
            )

    elif callable(sfr_model):
        if params is None:
            return sfr_model(zz_array)
        else:
            return sfr_model(zz_array, params)

    else:
        raise ValueError(
            'Unrecognized type of metallicity dependency.' + '\n'
            + 'Implemented models: ' + '\n'
            + str([*model_list]) + '\n'
            + 'If the string is a expression to evaluate,' + '\n'
            + 'there is something wrong in it, check it.' + '\n'
            + 'Independent variable must be called \'zz\' ' + '\n'
            + 'and parameters be an array or list called \'ci\'.' + '\n'
            + 'Inputs given:' + '\n'
            + '\'metall_model\': ' + str(sfr_model) + '\n'
            + '\'ci\': ' + str(params)
        )


# ---------------------------------------------------------------------

model_list = {
    'sfr_madau14': _sfr_madau14,
    'sfr_finke22a': _sfr_finke22a,
    'sfr_cuba': _sfr_cuba,
    'sfr_constant': _sfr_constant
}


def sfr_model(zz_array, sfr_model, sfr_params=None, verbose=True):
    """
    Stellar Formation Rate (sfr) is the density of stars that are born
    as a function of redshift.
    Units expected: TODO

    zz_array: nD array
        Redshift values to input in the formula.
    sfr_model: string or callable
        Formula (analytical or numerical) of the sfr.
    params: 1D array or list
        Optional parameters that can enter the sfr formula.

    :return: nD array
        sfr values with the same shape of the zz_array.
    """
    if sfr_model in model_list.keys():
        return model_list[sfr_model](
            zz_array=zz_array, params=sfr_params, verbose=verbose)
    else:
        return _sfr_custom(
            zz_array=zz_array, sfr_model=sfr_model,
            params=sfr_params, verbose=verbose)
