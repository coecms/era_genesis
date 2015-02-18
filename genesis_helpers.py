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
    >>> find_fractions(np.arange(4, dtype=float), 0.5)
    array([ 0.5,  0.5,  0. ,  0. ])
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

def get_interpolation_array( lon_array, lon, lat_array, lat ):
    """(array, num, array, num) -> 2d array of float

    >>> lon_array = np.arange(4)
    >>> lat_array = np.arange(3)
    >>> get_interpolation_array( lon_array, 1.5, lat_array, 0.5)
    array([[ 0.  ,  0.  ,  0.  ],
           [ 0.25,  0.25,  0.  ],
           [ 0.25,  0.25,  0.  ],
           [ 0.  ,  0.  ,  0.  ]])
    >>> get_interpolation_array( lon_array, 1.2, lat_array, 1.0)
    array([[ 0. ,  0. ,  0. ],
           [ 0. ,  0.8,  0. ],
           [ 0. ,  0.2,  0. ],
           [ 0. ,  0. ,  0. ]])
    """


    return_array = np.zeros((len(lon_array), len(lat_array)), dtype=float)

    return_array[:,:] = find_fractions( 1.0 * lon_array, lon)[:, np.newaxis]
    return_array[:,:] *= find_fractions( 1.0 * lat_array, lat)[np.newaxis, :]

    return return_array


if __name__ == '__main__':
    import doctest
    doctest.testmod()
