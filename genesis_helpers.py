#!/usr/bin/env python

# These are helper files to analyse the data. They do not contain any references
# to either the netcdf nor the namelists.

import numpy as np
import datetime


class Genesis_Config(object):
    """Stores various configuration options. Gets its data from both
    arguments and the base namelist.
    """

    intervall = datetime.timedelta(hours=6)

    def __read_date_from_args(self, s):
        """(str) -> datetime

        reads the string s and tries to interpret it as datetime.

        >>> __read_date_from_args('20150101')
        datetime.datetime(2015, 1, 1, 0, 0)
        >>> __read_date_from_args('2015013104')
        datetime.datetime(2015, 1, 31, 4, 0)
        >>> __read_date_from_args('201512011231')
        datetime.datetime(2015, 12, 1, 12, 31)
        """

        if len(s) == 8:
            return_date = datetime.datetime.strptime(s, '%Y%m%d')
        elif len(s) == 10:
            return_date = datetime.datetime.strptime(s, '%Y%m%d%H')
        elif len(s) == 12:
            return_date = datetime.datetime.strptime(s, '%Y%m%d%H%M')
        else:
            raise ValueError("Can't read datetime: {}".format(s))

        return return_date

    def __read_date_from_base(self, date, hour):
        """(int, int) -> datetime

        Inputs:
            date: int in the form year * 10000 + month * 100 _ day
            hour: int

        Output:
            datetime
        """
        if date < 10000000 or date >= 100000000:
            raise ValueError(
                "Can't read date from base: date = {}".format(date))

        if hour < 0 or hour >= 24:
            raise ValueError(
                "Cant't read hour from base: {}".format(hour))

        year = int(date/10000)
        monthday = date % 10000
        month = int(monthday / 100)
        day = monthday % 100

        return datetime.datetime(
            year=year, month=month, day=day, hour=hour
        )

    def __calc_num(self):
        """Calculates the number of timesteps between and including
        self.start_date and self.end_date
        """

        self.num = int((self.end_date - self.start_date).total_seconds() /
                       self.intervall.total_seconds()) + 1

    def set_start_date(self, args_start, sdate, shour):
        """(str, int, int) -> datetime

        Inputs:
            args_start: None or string consisting of YYYYmmdd[HH[MM]]
                        typically from arguments
            sdate:      int: year*10000 + month*100 + day
            shour:      int: hours

        Output:
            if args_start is set, then ignores the other two, and sets
            start_date to what the arguments describe. Otherwise it uses
            base.
        """

        if args_start:
            self.start_date = self.__read_date_from_args(args_start)
        else:
            self.start_date = self.__read_date_from_base(sdate, shour)

    def set_end_date(self, args_end, args_num, edate, ehour):
        """(str, int, int) -> datetime

        Inputs:
            args_end: None or string consisting of YYYYmmdd[HH[MM]]
                      typically from arguments
            args_num: int: how many
            edate:    int: year*10000 + month*100 + day
            ehour:    int: hours

        Output:
            if args_start is set, then ignores the other two, and sets
            start_date to what the arguments describe. Otherwise it uses
            base.
        """

        if args_end:
            self.end_date = self.__read_date_from_args(args_end)
        elif args_num:
            self.end_date = self.start_date + (args_num-1) * self.intervall
        else:
            self.end_date = self.__read_date_from_base(edate, ehour)

    def __init__(self, args, base):
        """(Namespace, Namelist)"""
        if not args.lat:
            raise ValueError("Latitude not set!")
        if not -90. <= args.lat <= 90.:
            raise ValueError("Latitude not in range: {}".format(args.lat))
        self.lat = args.lat

        if not args.lon:
            raise ValueError("Longitude not set!")
        if not -180. <= args.lon <= 360.:
            raise ValueError("Longitude not in range: {}".format(args.lon))
        self.lon = args.lon % 360.0

        self.set_start_date(args.start_date, base['time']['sdate'],
                            base['time']['shour'])
        self.set_end_date(args.end_date, args.num,
                          base['time']['edate'], base['time']['ehour'])
        self.__calc_num()

        self.output = args.output
        self.relhum = args.relhum
        self.debug = args.debug

        self.z_terrain_asl = base['base']['z_terrain_asl']
        self.nzum = base['base']['nzum']

        vertlevs = base['vertlevs']
        self.z_top_of_model = vertlevs['z_top_of_model']
        self.first_constant_r_rho_level = vertlevs['first_constant_r_rho_level']
        self.eta_theta = vertlevs['eta_theta']
        self.eta_rho = vertlevs['eta_rho']

        usrfields1 = base['usrfields_1']
        self.ui = usrfields1['ui']
        self.vi = usrfields1['vi']
        self.wi = usrfields1['wi']
        self.theta = usrfields1['theta']
        self.qi = usrfields1['qi']
        self.p_in = usrfields1['p_in']

        usrfields2 = base['usrfields_2']
        self.l_windrlx = usrfields2['l_windrlx']
        self.tau_rlx = usrfields2['tau_rlx']
        self.l_vertadv = usrfields2['l_vertadv']
        self.tstar_forcing = usrfields2['tstar_forcing']
        self.flux_e = usrfields2['flux_e']
        self.flux_h = usrfields2['flux_h']
        self.u_inc = usrfields2['u_inc']
        self.v_inc = usrfields2['v_inc']
        self.w_inc = usrfields2['w_inc']
        self.t_inc = usrfields2['t_inc']
        self.q_star = usrfields2['q_star']
        self.ichgf = usrfields2['ichgf']

        if args.template:
            self.template = args.template
        else:
            self.template = base['usrfields_3']['namelist_template']


def find_nearest_indices(array, val):
    """(numpy array, val) -> list of int

    Returns a list of the indices of the array values nearest to value val.
    If val is in array, it returns a list with this id and the next, otherwise
    it returns a list with two indices, the one preceeding and following.

    Prerequisite: array[0] <= val < array[-1] or array[-1] <= val < array[0]

    >>> find_nearest_indices(np.arange(4), 0.7)
    [1, 0]
    >>> find_nearest_indices(np.arange(4), 2)
    [2, 3]
    >>> find_nearest_indices(np.arange(-3, 4), 1.2)
    [4, 5]
    >>> find_nearest_indices(np.arange(4, -3, -1), 1.2)
    [2, 3]
    """

    ascending = (array[0] < array[-1])
    if ascending:
        assert(array[0] <= val)
        assert(val < array[-1])
    else:
        assert(array[0] > val)
        assert(val >= array[-1])

    idx = np.abs(array-val).argmin()

    if ascending == (array[idx] > val):
        return [idx-1, idx]
    else:
        return [idx, idx+1]


def find_fractions(target, val1, val2):
    """(scalar, scalar, scalar) -> list of float

    Interpolates values val1 and val2 and returns a list of two floats
    so that target = val1 * find_fraction[0] + val2 * find_fraction[1]

    >>> find_fractions(1.5, 1., 2.)
    [0.5, 0.5]
    >>> find_fractions(1, 1, 2)
    [1.0, 0.0]
    >>> find_fractions(-3, -1, -3)
    [0.0, 1.0]
    >>> find_fractions(0.25, 0., 1.)
    [0.75, 0.25]
    >>> find_fractions(3, 0, 4)
    [0.25, 0.75]
    """

    frac = float(target - val2) / float(val1 - val2)
    return [frac, 1.-frac]


def interpolate(target_x, x_array, y_array):
    """(numeric, array, array) -> numeric

    >>> interpolate(1.5, np.arange(4), np.arange(4)*2)
    3.0
    >>> interpolate(7.2, np.arange(10), np.arange(2, 12))
    9.2
    >>> interpolate([1.5, 7.2], np.arange(10), 2. * np.arange(10))
    [3.0, 14.4]
    >>> interpolate(np.array([1.5, 7.2]), np.arange(10), np.arange(10, 20))
    array([ 11.5,  17.2])
    """

    if type(target_x) == list:
        return [interpolate(x, x_array, y_array) for x in target_x]
    if type(target_x) == np.ndarray:
        return np.array(interpolate(target_x.tolist(), x_array, y_array))

    idx1, idx2 = find_nearest_indices(x_array, target_x)
    fct1, fct2 = find_fractions(target_x, x_array[idx1], x_array[idx2])

    return fct1 * y_array[idx1] + fct2 * y_array[idx2]


def radian(angle):
    """(number) -> float

    returns the radian of the degree angle angle

    >>> radian(0.)
    0.0
    >>> radian(180.)
    3.141592653589793
    """

    return np.pi * angle / 180.0


def surface_distance_y(lat1, lat2):
    """(number, number) -> float

    Returns the distance between two latitudes.

    >>> surface_distance_y(0., 1.)
    111198.76636891312
    """

    r_earth = 6371220
    return (np.pi * r_earth) / 180.0 * abs(lat2 - lat1)


def surface_distance_x(lon1, lon2, lat):
    """(number, number, number) -> float

    Returns the distance between two longitudes at a certain latitude.

    >>> surface_distance_x(0., 1., 0.)
    111198.76636891312
    >>> surface_distance_x(0., 1., 80.)
    19309.463138772513
    """

    return surface_distance_y(lon1, lon2) * np.cos(radian(lat))


def find_fractions_array(array, val, idxs=None):
    """(numpy array, str) -> numpy array

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

    ascending = (array[0] < array[-1])

    if val == array[0]:
        return_array[0] = 1.0
        return return_array
    if val == array[-1]:
        return_array[-1] = 1.0
        return return_array
    if ascending == (val > array[-1]):
        return_array[-1] = 1.0
        return return_array
    if ascending == (val < array[0]):
        return_array[0] = 1.0
        return return_array

    if type(idxs) in [list, tuple] and len(idxs) == 2:
        left = idxs[0]
        right = idxs[1]
    else:
        left, right = find_nearest_indices(array, val)

    left_frac, right_frac = find_fractions(val, array[left], array[right])
    return_array[left] = left_frac
    return_array[right] = right_frac

    return return_array


def create_dates_list(conf):
    """(Namelist) -> list of datetime

    >>> class a(object):
    ...     start_date = datetime.datetime(2010, 1, 1)
    ...     end_date = datetime.datetime(2010, 1, 2, 6)
    ...     intervall = datetime.timedelta(hours=6)
    >>> conf = a()
    >>> for d in create_dates_list(conf):
    ...     print(d)
    2010-01-01 00:00:00
    2010-01-01 06:00:00
    2010-01-01 12:00:00
    2010-01-01 18:00:00
    2010-01-02 00:00:00
    2010-01-02 06:00:00
    """
    date = conf.start_date
    return_list = [date]
    while date < conf.end_date:
        date += conf.intervall
        return_list.append(date)
    return return_list


def convert_to_datetime(date, hour):
    """(str, str) or (str, int) or (int, int) -> datetime.datetime

    returns a datetime object with the date set to what date represents,
    and the time to hour.

    >>> convert_to_datetime('20000101', '0')
    datetime.datetime(2000, 1, 1, 0, 0)
    >>> convert_to_datetime('20100331', '6')
    datetime.datetime(2010, 3, 31, 6, 0)
    >>> convert_to_datetime('19671129', 18)
    datetime.datetime(1967, 11, 29, 18, 0)
    >>> convert_to_datetime(19760121, 12)
    datetime.datetime(1976, 1, 21, 12, 0)
    """

    if type(date) == int:
        date = str(date)
    if type(hour) == str:
        hour = int(hour)

    day = datetime.datetime.strptime(date, '%Y%m%d')

    return day.replace(hour=hour)


def calc_ht_conversion(ht_in, ht_out):
    """(array, array) -> matrix

    Returns a matrix that interpolates from the ht_in height array into
    the ht_out height array.

    >>> h1 = np.array(range(5))*25.
    >>> h2 = np.array(range(6))*20.
    >>> calc_ht_conversion(h1, h2)
    matrix([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.75,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.25,  0.5 ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.5 ,  0.25,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.75,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ]])
    """

    len_in = len(ht_in)
    len_out = len(ht_out)

    conv_matrix = np.matrix(np.zeros((len_out, len_in), dtype=np.float))

    for i in range(len_in):
        conv_matrix[:, i] = \
            find_fractions_array(ht_out, ht_in[i])[:, np.newaxis]

    return conv_matrix


def get_eta_theta(conf):
    """(Namelist) -> np.array

    retrieves the eta_theta array from base namelist, and multiplies it with
    z_top_of_model.
    """

    return_array = np.array(conf.eta_theta)
    return_array *= conf.z_top_of_model
    return_array += conf.z_terrain_asl
    return return_array


def get_eta_rho(conf):
    """(Namelist) -> np.array

    retrieves the eta_rho array from base namelist, and multiplies it with
    z_top_of_model.
    """

    return_array = np.array(conf.eta_rho)
    return_array *= conf.z_top_of_model
    return_array += conf.z_terrain_asl
    return return_array


def convert_height(array_in, calc_matrix):
    """(np.array, np.matrix) -> np.array

    """

    shape_in = array_in.shape

    shape_out = (shape_in[0], calc_matrix.shape[0])

    array_out = np.empty(shape_out, dtype=np.float)

    for i in range(shape_in[0]):
        new_vert_array = calc_matrix * np.matrix(array_in[i, :]).T
        array_out[i, :] = np.array(new_vert_array.flat)[:]

    return array_out


if __name__ == '__main__':
    import doctest
    doctest.testmod()
