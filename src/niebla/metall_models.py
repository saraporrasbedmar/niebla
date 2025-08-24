import numpy as np


def _metall_tanikawa22(zz_array, params=None, verbose=True):
    if params is None:
        params = [0.153, 0.074, 1.34, 0.02]
        if verbose:
            print('   -> Metallicity: default parameters chosen: ',
                  params)
    return 10 ** (params[0] - params[1] * zz_array ** params[2]
                  ) * params[3]


def _metall_constant(zz_array, params=None, verbose=True):
    if params is None:
        params = [0.02]
        if verbose:
            print('   -> Metallicity: default parameters chosen: ',
                  params)
    return np.ones_like(zz_array) * params


def _metall_custom(zz_array, metall_model, params=None, verbose=True):

    if type(metall_model) == str:
        try:
            return eval(metall_model, {'zz': zz_array, 'ci': params})
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
                + '\'metall_model\': ' + str(metall_model) + '\n'
                + '\'ci\': ' + str(params)
            )

    elif callable(metall_model):
        if params is None:
            return metall_model(zz_array)
        else:
            return metall_model(zz_array, params)

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
            + '\'metall_model\': ' + str(metall_model) + '\n'
            + '\'ci\': ' + str(params)
        )


# ---------------------------------------------------------------------
model_list = {
    'metall_tanikawa22': _metall_tanikawa22,
    'metall_constant': _metall_constant
}


def metall_model(zz_array, metall_model, metall_params=None,
                 verbose=True):
    """
    Z(z) is the mean metallicity of the Universe as a function of
    redshift.

    Units: unitless [1]

    zz_array: nD array
        Redshift values to input in the formula.
    metall_model: string or callable
        Formula (analytical or numerical) of the metallicity.
    params: 1D array or list
        Optional parameters that can enter the metallicity formula.

    :return: nD array
        Metallicity values with the same shape of the zz_array.
    """
    if metall_model in model_list.keys():
        return model_list[metall_model](
            zz_array=zz_array, params=metall_params, verbose=verbose)
    else:
        return _metall_custom(
            zz_array=zz_array, metall_model=metall_model,
            params=metall_params, verbose=verbose)
