#!/usr/bin/env python

# These are helper files to analyse the data. They do not contain any references
# to either the netcdf nor the namelists.

import numpy as np
import datetime

def find_nearest_indices(array, val):
    """( numpy array, val) -> list of int

    Returns a list of the indices of the array values nearest to value val.
    If val is in array, it returns a list with this id and the next, otherwise
    it returns a list with two indices, the one preceeding and following.

    Prerequisite: array[0] <= val < array[-1] or array[-1] <= val < array[0]

    >>> find_nearest_indices( np.arange(4), 0.7 )
    [1, 0]
    >>> find_nearest_indices( np.arange(4), 2 )
    [2, 3]
    >>> find_nearest_indices( np.arange(-3, 4), 1.2 )
    [4, 5]
    """

    ascending = ( array[0] < array[-1] )
    if ascending:
        assert( array[0] <= val )
        assert( val < array[-1])
    else:
        assert( array[0] > val )
        assert( val >= array[-1])

    idx = np.abs(array-val).argmin()

    if ascending == (array[idx] > val):
        return [idx, idx-1]
    else:
        return [idx, idx+1]

def find_fractions( target, val1, val2 ):
    """(scalar, scalar, scalar) -> list of float

    Interpolates values val1 and val2 and returns a list of two floats
    so that target = val1 * find_fraction[0] + val2 * find_fraction[1]

    >>> find_fractions( 1.5, 1., 2. )
    [0.5, 0.5]
    >>> find_fractions( 1, 1, 2 )
    [1.0, 0.0]
    >>> find_fractions( -3, -1, -3 )
    [0.0, 1.0]
    >>> find_fractions( 0.25, 0., 1. )
    [0.75, 0.25]
    >>> find_fractions( 3, 0, 4 )
    [0.25, 0.75]
    """

    frac = float(target - val2) / float(val1 - val2)
    return [frac, 1.-frac]

def find_fractions_array(array, val, idxs = None):
    """( numpy array, str ) -> numpy array

    Returns an array of fractions that can be used for a linear interpolation.
    If idxs is a list or tuple with length 2, then its values are used as the
    indices of array over which to interpolate.

    Otherwise, idxs is calculated with find_nearest_indices.

    >>> find_fractions_array(np.linspace(-3, 3, 7), 0.5)
    array([ 0. ,  0. ,  0. ,  0.5,  0.5,  0. ,  0. ])
    >>> find_fractions_array(np.linspace(-3, 3, 7), -2.7)
    array([ 0.7,  0.3,  0. ,  0. ,  0. ,  0. ,  0. ])
    >>> find_fractions_array(np.linspace(-3, 3, 7), 2.)
    array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.])
    >>> find_fractions_array(np.arange(4, dtype=float), 0.5)
    array([ 0.5,  0.5,  0. ,  0. ])
    >>> find_fractions_array(np.arange(4, dtype=float), 2., [1, 3])
    array([ 0. ,  0.5,  0. ,  0.5])
    """

    return_array = np.zeros_like(array)

    if type(idxs) in [list, tuple] and len(idxs) == 2:
        left = idxs[0]
        right = idxs[1]
    else:
        left, right = find_nearest_indices( array, val )

    left_frac, right_frac = find_fractions(val, array[left], array[right])
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

    return_array[:,:] = find_fractions_array( 1.0 * lon_array, lon)[:, np.newaxis]
    return_array[:,:] *= find_fractions_array( 1.0 * lat_array, lat)[np.newaxis, :]

    return return_array

def create_dates_list(args):
    """(Namelist) -> list of datetime

    >>> class a(object):
    ...     start_date = datetime.datetime(2010, 1, 1)
    ...     end_date = datetime.datetime(2010, 1, 2, 6)
    ...     intervall = datetime.timedelta(hours=6)
    >>> args = a()
    >>> for d in create_dates_list(args):
    ...     print( d )
    2010-01-01 00:00:00
    2010-01-01 06:00:00
    2010-01-01 12:00:00
    2010-01-01 18:00:00
    2010-01-02 00:00:00
    2010-01-02 06:00:00
    """
    date = args.start_date
    return_list = [date]
    while date < args.end_date:
        date += args.intervall
        return_list.append(date)
    return return_list

def convert_to_datetime( date, hour ):
    """( str, str ) or (str, int) or (int, int) -> datetime.datetime

    returns a datetime object with the date set to what date represents, and the time
    to hour.

    >>> convert_to_datetime( '20000101', '0' )
    datetime.datetime(2000, 1, 1, 0, 0)
    >>> convert_to_datetime( '20100331', '6' )
    datetime.datetime(2010, 3, 31, 6, 0)
    >>> convert_to_datetime( '19671129', 18 )
    datetime.datetime(1967, 11, 29, 18, 0)
    >>> convert_to_datetime( 19760121, 12 )
    datetime.datetime(1976, 1, 21, 12, 0)
    """

    if type(date) == int:
        date = str( date )
    if type(hour) == str:
        hour = int(hour)

    day = datetime.datetime.strptime(date, '%Y%m%d')

    return day.replace(hour=hour)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
