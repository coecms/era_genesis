#!/usr/bin/env python

# These are helper files to analyse the data. They do not contain any references
# to either the netcdf nor the namelists.

import numpy as np
import datetime


def find_fractions(array, val):
    """( numpy array, str ) -> list of int

    >>> find_fractions(np.linspace(-3, 3, 7), 0.5)
    array([ 0. ,  0. ,  0. ,  0.5,  0.5,  0. ,  0. ])
    >>> find_fractions(np.linspace(-3, 3, 7), -2.7)
    array([ 0.7,  0.3,  0. ,  0. ,  0. ,  0. ,  0. ])
    >>> find_fractions(np.linspace(-3, 3, 7), 2.)
    array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.])
    """

    return_array = np.zeros_like(array)

    idx = np.abs(array - val).argmin()
    if array[idx] == val:
        return_array[idx] = 1.0
    else:
        if array[idx] < val:
            left = idx
            right = idx+1
        else:
            right = idx
            left = idx-1
        left_frac = (array[right] - val)/(array[right]-array[left])
        right_frac = 1.0 - left_frac
        return_array[left] = left_frac
        return_array[right] = right_frac

    return return_array



if __name__ == '__main__':
    import doctest
    doctest.testmod()
