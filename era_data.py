#!/usr/bin/env python

import numpy as np
import netCDF4 as nc
import datetime


class EraException(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class EraDimException(EraException):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class era_dataset(object):
    """Container for data from the ERA-Interim Dataset.
    """

    allvars = ['U', 'V', 'T', 'Z', 'Q', 'P', 'SST']
    vars2d = ['P', 'SST']

    ntime = 0
    nht = 0
    nlat = 0
    nlon = 0

    time_array = None
    ht_array = None
    lat_array = None
    lon_array = None

    time_units = ''
    ht_units = ''
    lat_units = ''
    lon_units = ''

    data = None
    units = ''

    filename_list = []
    reference_date = datetime.datetime(2000, 1, 1, 0, 0)

    def __init__(self, var):
        """initialises the metadata"""
        self.var = var

    def __convert_date(self, date):
        """(int) -> datetime
        Input:
            date: 'YYYY-MM-DD[ HH[:MM]]' or datetime
        Output:
            datetime

        if date_string is a datetime, return that.
        >>> ed.__convert_date('2000-01-01')
        datetime.datetime(2000, 1, 1, 0, 0)
        >>> ed.__convert_date('2010-03-31 12')
        datetime.datetime(2010, 3, 31, 12, 0)
        >>> ed.__convert_date('1976-01-21 18:34')
        datetime.datetime(1976, 1, 21, 18, 34)
        >>> ed.__convert_date(datetime.datetime(2000, 2, 29, 21, 3))
        datetime.datetime(2000, 2, 29, 21, 3)
        test doctest
        """

        if type(date) == datetime.datetime:
            return date

        if not type(date) == str:
            raise ValueError("{} is not a string, but a {}".format(
                date, type(date)))

        if len(date) == 10:
            this_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        elif len(date) == 13:
            this_date = datetime.datetime.strptime(date, '%Y-%m-%d %H')
        elif len(date) == 15:
            this_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M')
        else:
            raise ValueError('Cannot convert {} into datetime'.format(date))

        return this_date

    def __get_var_name(self, var=None):
        """(str) -> str

        returns the proper var name for the given var in
        ['U', 'V', 'T', 'Z', 'Q', 'P', 'SST']

        >>> ed.__get_var_name('U')
        'U_GDS0_ISBL'
        >>> ed.__get_var_name('V')
        'V_GDS0_ISBL'
        >>> ed.__get_var_name('T')
        'T_GDS0_ISBL'
        >>> ed.__get_var_name('Z')
        'Z_GDS0_ISBL'
        >>> ed.__get_var_name('Q')
        'Q_GDS0_ISBL'
        >>> ed.__get_var_name('P')
        'MSL_GDS0_SFC'
        """

        if not var:
            var = self.var

        assert(var in self.allvars)

        if var == 'P':
            return 'MSL_GDS0_SFC'
        elif var == 'SST':
            return 'SSTK_GDS0_SFC'
        else:
            return '{:1}_GDS0_ISBL'.format(var)

    def __get_lat_name(self, var=None):
        """ (str) -> str

        Returns the name of the latitude array for the variable var.

        >>> ed.__get_lat_name('U')
        'g0_lat_2'
        >>> ed.__get_lat_name('V')
        'g0_lat_2'
        >>> ed.__get_lat_name('T')
        'g0_lat_2'
        >>> ed.__get_lat_name('U')
        'g0_lat_2'
        >>> ed.__get_lat_name('Q')
        'g0_lat_2'
        >>> ed.__get_lat_name('P')
        'g0_lat_1'
        """
        if not var:
            var = self.var

        assert(var in self.allvars)
        if var in self.vars2d:
            return 'g0_lat_1'
        return 'g0_lat_2'

    def __get_lon_name(self, var=None):
        """ (str) -> str

        Returns the name of the longitude array for the variable var.

        >>> ed.__get_lon_name('U')
        'g0_lon_3'
        >>> ed.__get_lon_name('V')
        'g0_lon_3'
        >>> ed.__get_lon_name('T')
        'g0_lon_3'
        >>> ed.__get_lon_name('U')
        'g0_lon_3'
        >>> ed.__get_lon_name('Q')
        'g0_lon_3'
        >>> ed.__get_lon_name('P')
        'g0_lon_2'
        """
        if not var:
            var = self.var

        assert(var in self.allvars)
        if var in self.vars2d:
            return 'g0_lon_2'
        return 'g0_lon_3'

    def __get_time_name(self, var=None):
        """ (str) -> str

        Returns the name of the time array for the variable var.

        >>> ed.__get_time_name('U')
        'initial_time0_hours'
        >>> ed.__get_time_name('V')
        'initial_time0_hours'
        >>> ed.__get_time_name('T')
        'initial_time0_hours'
        >>> ed.__get_time_name('U')
        'initial_time0_hours'
        >>> ed.__get_time_name('Q')
        'initial_time0_hours'
        >>> ed.__get_time_name('P')
        'initial_time0_hours'
        """
        if not var:
            var = self.var

        assert(var in self.allvars)
        return 'initial_time0_hours'

    def __get_ht_name(self, var=None):
        """ (str) -> str

        Returns the name of the height dimension for variable var.
        Returns empty string for 'P' because 'P' is a 2D field.

        >>> ed.__get_ht_name('U')
        'lv_ISBL1'
        >>> ed.__get_ht_name('V')
        'lv_ISBL1'
        >>> ed.__get_ht_name('T')
        'lv_ISBL1'
        >>> ed.__get_ht_name('Z')
        'lv_ISBL1'
        >>> ed.__get_ht_name('Q')
        'lv_ISBL1'
        >>> ed.__get_ht_name('P')
        ''
        """
        if not var:
            var = self.var

        assert (var in self.allvars)
        if var in self.vars2d:
            return ''
        return 'lv_ISBL1'

    def __find_nearest_indices(self, array, val):
        """(numpy array, val) -> list of int

        Returns a list of the indices of the array values nearest to value val.
        If val is in array, it returns a list with this id and the next,
        otherwise it returns a list with two indices, the one preceeding and
        following.

        Prerequisite: array[0] <= val < array[-1] or array[-1] <= val < array[0]

        >>> ed.__find_nearest_indices(np.arange(4), 0.7)
        [1, 0]
        >>> ed.__find_nearest_indices(np.arange(4), 2)
        [2, 3]
        >>> ed.__find_nearest_indices(np.arange(-3, 4), 1.2)
        [4, 5]
        >>> ed.__find_nearest_indices(np.arange(4, -3, -1), 1.2)
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

    def get_file_name(self, date=None, var=None):
        """(str or datetime) -> str

        Returns the filename where the data for the variable var can be read
        for the date.

        Input:
            date: either str in the form of 'YYYY-mm-dd[ HH[:MM]]',
                  or a datetime, or None for the reference_date
            var: 'U', 'V', 'T', 'Z', 'Q', 'P', or None to default to the
                variable for this data
        Output:
            A string containing the file name.
        """

        file_template = '/g/data1/ua8/erai/netcdf/oper_an_{level:}/fullres/' + \
            'sub-daily/{year:4}/{var:}_6hrs_{level:}_{year:4}_{month:02}.nc'

        if var:
            this_var = var
        else:
            this_var = self.var

        assert(this_var in self.allvars)

        if not date:
            this_date = self.reference_date
        else:
            this_date = self.__convert_date(date)

        vals = {
            'year': this_date.year,
            'month': this_date.month,
            'level': 'pl',
            'var': this_var
        }
        if this_var == 'P':
            vals['var'] = 'MSL'
        elif this_var == 'SST':
            vals['var'] = 'SSTK'
        if this_var in self.vars2d:
            vals['level'] = 'sfc'

        return file_template.format(**vals)

    def set_time_len(self, ntime):
        """(int) -> Null
        Set the time dimension length
        """

        assert(ntime >= 1)
        self.ntime = ntime

    def set_ht_len(self, nht):
        """(int) -> Null
        Set the height dimension length
        """

        assert(nht >= 1)
        self.nht = nht

    def set_lat_len(self, nlat):
        """(int) -> Null
        Set the latitude dimension length.
        """

        assert(nlat >= 1)
        self.nlat = nlat

    def set_lon_len(self, nlon):
        """(int) -> Null
        Set the longitude dimension length.
        """

        assert(nlon >= 1)
        self.nlon = nlon

    def set_dim_lengths(self, ntime, nht, nlat, nlon):
        """(int, int, int, int) -> Null
        Set all dimension lengths in the order: ntime, nht, nlat, nlon
        """

        self.set_time_len(ntime)
        self.set_ht_len(nht)
        self.set_lat_len(nlat)
        self.set_lon_len(nlon)

    def set_time_array(self, time_array, units=None):
        """(np.array) -> None
        Sets the time array to time_array.
        If time dimension length isn't set, it sets it.
        Throws EraIncompatibleDimension exception if time array has the wrong
        length.
        """

        if self.ntime == 0:
            self.set_time_len(len(time_array))
        if len(time_array) != self.ntime:
            raise EraDimException(("Time dimension length wrong: " +
                                   "ntime = {}, len(time_array = {}"
                                   ).format(self.ntime, len(time_array)))
        self.time_array = time_array
        if units:
            self.time_units = units

    def set_ht_array(self, ht_array, units=None):
        """(np.array) -> None
        Sets the ht array to ht_array.
        If ht dimension length isn't set, it sets it.
        Throws EraIncompatibleDimension exception if ht array has the wrong
        length.
        """

        if self.nht == 0:
            self.set_ht_len(len(ht_array))
        if len(ht_array) != self.nht:
            raise EraDimException(("Height dimension length wrong: " +
                                   "nht = {}, len(ht_array = {}"
                                   ).format(self.nht, len(ht_array)))
        self.ht_array = ht_array
        if units:
            self.ht_units = units

    def set_lat_array(self, lat_array, units=None):
        """(np.array) -> None
        Sets the lat array to lat_array.
        If lat dimension length isn't set, it sets it.
        Throws EraIncompatibleDimension exception if lat array has the wrong
        length.
        """

        if self.nlat == 0:
            self.set_lat_len(len(lat_array))
        if len(lat_array) != self.nlat:
            raise EraDimException(("Lat dimension length wrong: " +
                                   "nlat = {}, len(lat_array = {}"
                                   ).format(self.nlat, len(lat_array)))
        self.lat_array = lat_array
        if units:
            self.lat_units = units

    def set_lon_array(self, lon_array, units=None):
        """(np.array) -> None
        Sets the lon array to lon_array.
        If lon dimension length isn not set, it sets it.
        Throws EraIncompatibleDimension exception if lon array has the wrong
        length.
        """

        if self.nlon == 0:
            self.set_lon_len(len(lon_array))
        if len(lon_array) != self.nlon:
            raise EraDimException(("Lon dimension length wrong: " +
                                   "nlon = {}, len(lon_array = {}"
                                   ).format(self.nlon, len(lon_array)))
        self.lon_array = lon_array
        if units:
            self.lon_units = units

    def __read_dim_array(self, dim_name, date=None):
        """Reads the whole dimension array
        """

        ncid = nc.Dataset(self.get_file_name(date=date), 'r')
        return_array = ncid.variables[dim_name][:]
        units = ncid.variables[dim_name].units
        ncid.close()

        return return_array, units

    def read_ht_array(self):
        """Reads the height array into this object
        """

        if self.var in self.vars2d:
            self.set_ht_array(np.zeros(1))
        else:
            self.set_ht_array(
                *(self.__read_dim_array(self.__get_ht_name()))
            )
        self.ht_idxs = list(range(len(self.ht_array)))

    def read_lat_array(self):
        """Reads the height array into this object
        """

        self.set_lat_array(
            *(self.__read_dim_array(self.__get_lat_name()))
        )
        self.lat_idxs = list(range(len(self.lat_array)))

    def read_lon_array(self):
        """Reads the heiglon array into this object
        """

        self.set_lon_array(
            *(self.__read_dim_array(self.__get_lon_name()))
        )
        self.lon_idxs = list(range(len(self.lon_array)))

    def read_time_array(self, date=None):
        """Reads the heigtime array into this object
        """

        self.set_time_array(
            *(self.__read_dim_array(self.__get_time_name(), date=date))
        )
        self.time_idxs = list(range(len(self.time_array)))

    def select_lats_near(self, lat):
        """
        Selects the two latitudes in the grid that are closest to lat.
        Sets lat_array to these two, and sets lat_idxs to the indices.
        """

        if self.data:
            raise EraException('Cannot change dimensions once data is read')

        self.lat_idxs = self.__find_nearest_indices(self.lat_array, lat)
        self.lat_array = self.lat_array[self.lat_idxs]
        self.nlat = 2

    def select_lons_near(self, lon):
        """
        Selects the two lonitudes in the grid that are closest to lon.
        Sets lon_array to these two, and sets lon_idxs to the indices.
        """

        if self.data:
            raise EraException('Cannot change dimensions once data is read')

        self.lon_idxs = self.__find_nearest_indices(self.lon_array, lon)
        self.lon_array = self.lon_array[self.lon_idxs]
        self.nlon = 2

    def select_time_array(self, start, end):
        """
        Creates a list of files, and a list of lists of corresponding
        indices and values.
        """

        def next_month(date):
            if date.month == 12:
                return datetime.datetime(date.year+1, 1, 1)
            else:
                return datetime.datetime(date.year, date.month+1, 1)

        start_date = self.__convert_date(start)
        end_date = self.__convert_date(end)
        files = []
        times_idxs = []
        times_vals = []

        date = start_date
        while (date <= end_date):
            file_name = self.get_file_name(date=date)
            files.append(file_name)
            ncid = nc.Dataset(file_name, 'r')
            time_var = ncid.variables[self.__get_time_name()]
            units = time_var.units

            # Convert the time data into a numpy array.
            times = time_var[:]

            # Covert start and end date into the same format as times
            s_date = nc.date2num(start_date, units=time_var.units)
            e_date = nc.date2num(end_date, units=time_var.units)

            # Find the closest location to both start and end date
            start_idx = np.abs(times - s_date).argmin()
            end_idx = np.abs(times - e_date).argmin()

            # Calculate the indices and values for this file
            t_idxs = list(range(start_idx, end_idx+1))
            t_vals = times[t_idxs].tolist()

            # Append both the indices and values to the total list.
            times_idxs.append(t_idxs)
            times_vals += t_vals

            date = next_month(date)
            ncid.close()

        self.filename_list = files
        self.time_idxs = times_idxs
        self.set_time_array(np.array(times_vals), units)

    def read_data(self):
        """Reads the data from the ERA NetCDF files"""

        data_array = np.empty((0, self.nht, self.nlat, self.nlon))
        varname = self.__get_var_name()
        for f, t in zip(self.filename_list, self.time_idxs):
            ncid = nc.Dataset(f, 'r')
            units = ncid.variables[varname].units
            fill_value = (ncid.variables[varname]._FillValue *
                          ncid.variables[varname].scale_factor +
                          ncid.variables[varname].add_offset)
            if self.var in self.vars2d:
                data = ncid.variables[varname][t, self.lat_idxs, self.lon_idxs]
                data.shape = (data.shape[0], 1, data.shape[1], data.shape[2])
            else:
                data = ncid.variables[varname][t, self.ht_idxs, self.lat_idxs,
                                               self.lon_idxs]
            data_array = np.concatenate((data_array, data), axis=0)
            ncid.close()
        self.data = data_array
        self.units = units
        self.fill_value = fill_value

    def ensure_Pa(self):
        """Ensure that Pressure units is Pascal and not hPa
        """
        if self.ht_units == 'hPa':
            self.ht_array *= 100.0
            self.ht_units = 'Pa'
        if self.units == 'hPa':
            self.data *= 100.0
            self.units = 'Pa'

    def ensure_ascending(self):

        if self.var != 'P':
            if self.ht_array[0] < self.ht_array[-1]:
                self.ht_array = self.ht_array[::-1]
                self.data[:, :, :, :] = self.data[:, ::-1, :, :]

    def convert_geop_to_m(self):
        """Converts the geopotential to meters by dividing
        everything by g."""

        from genesis_globals import grav

        assert(self.var == 'Z')
        if self.units == 'm**2 s**-2':
            self.data *= (1.0 / grav)
            self.units = 'm'

    def __interp(self, x, xp, fp, left=None, right=None):
        """Wrapper for np.interp for when xp is descending."""

        if np.all(fp == self.fill_value):
            return self.fill_value
        my_xp = xp[fp != self.fill_value]
        my_fp = fp[fp != self.fill_value]


        if my_xp[0] > my_xp[-1]:
            return np.interp(x, my_xp[::-1], my_fp[::-1], left, right)
        else:
            return np.interp(x, my_xp, my_fp, left, right)

    def interp_lon(self, lon):
        """Returns a copy of this dataset, except that the longitude
        dimension is reduced to len=1 and set to the value of lon.
        Data is lineraly interpolated.
        """

        return_set = era_dataset(self.var)
        return_set.set_time_array(self.time_array, self.time_units)
        return_set.set_ht_array(self.ht_array, self.ht_units)
        return_set.set_lat_array(self.lat_array, self.lat_units)
        return_set.set_lon_array(np.array([lon]), self.lon_units)
        return_set.reference_date = self.reference_date
        return_set.units = self.units
        return_set.fill_value = self.fill_value

        return_set.data = np.empty((self.ntime, self.nht, self.nlat, 1))

        for t in range(self.ntime):
            for h in range(self.nht):
                for lat in range(self.nlat):
                    return_set.data[t, h, lat, 0] = self.__interp(
                        lon,
                        self.lon_array,
                        self.data[t, h, lat, :].flatten()
                    )
        return return_set

    def interp_lat(self, lat):
        """Returns a copy of this dataset, except that the longitude
        dimension is reduced to len=1 and set to the value of lon.
        Data is lineraly interpolated.
        """

        return_set = era_dataset(self.var)
        return_set.set_time_array(self.time_array, self.time_units)
        return_set.set_ht_array(self.ht_array, self.ht_units)
        return_set.set_lat_array(np.array([lat]), self.lat_units)
        return_set.set_lon_array(self.lon_array, self.lon_units)
        return_set.reference_date = self.reference_date
        return_set.units = self.units
        return_set.fill_value = self.fill_value

        return_set.data = np.empty((self.ntime, self.nht, 1, self.nlon))

        for t in range(self.ntime):
            for h in range(self.nht):
                for lon in range(self.nlon):
                    return_set.data[t, h, 0, lon] = self.__interp(
                        lat,
                        self.lat_array,
                        self.data[t, h, :, lon].flatten()
                    )
        return return_set


if __name__ == '__main__':

    u = era_dataset('SST')
    u.read_ht_array()

    u.read_lat_array()
    u.select_lats_near(37.2)

    u.read_lon_array()
    u.select_lons_near(147.2)

    u.select_time_array(start='2000-01-31 12', end='2000-02-01 06')
    u.read_data()

    print(u.data[0, 0, :, :])
    print(u.units)

    u_xi = u.interp_lon(147.2)
    print(u_xi.data.shape)
    print(u_xi.data[0, 0, :, :])

    u_si = u_xi.interp_lat(37.2)
    print(u_si.data[0, 0, :, :])
