#!/usr/bin/env python


import argparse
import os
import numpy as np
import datetime
import genesis_netcdf_helpers as nch
# import genesis_helpers as h

try:
    import f90nml
except:
    print("""
    Failed to import f90nml module.
    module use ~access/modules
    module load python pythonlib/f90nml pythonlib/netCDF4
    """)

# try:
#    import netCDF4 as cdf
# except:
#    print( """
#    Failed to import netCDF4 module
#    module use ~access/modules
#    module load python pythonlib/f90nml pythonlib/netCDF4
#    """ )


class genesis_logger(object):
    active = False

    def __init__(self, active):
        self.active = active

    def write(self, obj):
        if self.active:
            print(obj)

    def write_by_item(self, obj, indent=0, key_length=12, header=None):
        if self.active:
            if header:
                print(header)
            if type(obj) == dict:
                for key, val in obj.iteritems():
                    print(' '*indent + '{:key_length}: {}'.format(key, val))
            elif type(obj) == argparse.Namespace:
                for key in obj.__dict__:
                    val = getattr(obj, key)
                    print(' '*indent + '{:key_length}: {}'.format(key, val))
            else:
                for val in obj:
                    print(' '*indent + '{}'.format(val))


def read_netcdf_data(var, idxs, args):
    """(str, dict, Namelist) -> np.ndarray

    Returns all data from the ERA-Interim with variable named var in
    ['U', 'V', 'T', 'Z', 'Q', 'P'] for all the relevant dates.

    """

    assert(var in ['U', 'V', 'T', 'Z', 'Q', 'P'])

    logger = genesis_logger(args.debug)

    files = nch.file_list(var, args)

    logger.write('files to open for variable {}'.format(var))
    for f in files:
        logger.write(f)

    first_file = True

    idxs_to_read = [1, 1, 1] if var == 'P' else [1, 1, 1, 1]

    for f, time_idxs in zip(files, idxs['time']['idxs_list']):
        ncid, opened_here = nch.genesis_open_netCDF(f)
        shape, dims = nch.get_shape(ncid, var)
        if first_file:
            first_dims = dims
            lat_axis = dims.index(nch.get_lat_name(var))
            lon_axis = dims.index(nch.get_lon_name(var))
            time_axis = dims.index(nch.get_time_name(var))
            if not var == 'P':
                ht_axis = dims.index(nch.get_ht_name(var))
        else:
            if not dims == first_dims:
                raise IndexError("NetCDF files are not consistent")

        idxs_to_read[lat_axis] = idxs['lat']['idxs']
        idxs_to_read[lon_axis] = idxs['lon']['idxs']
        idxs_to_read[time_axis] = time_idxs
        if not var == 'P':
            idxs_to_read[ht_axis] = list(range(shape[ht_axis]))

        try:
            this_data = nch.read_array(ncid, nch.get_varname(var), idxs_to_read)
        except Exception as e:
            print(idxs_to_read)
            raise e
        if first_file:
            data_array = this_data
        else:
            data_array = np.concatenate((data_array, this_data), axis=time_axis)
        nch.genesis_close_netCDF(ncid, opened_here)
        first_file = False
    if var == 'P':
        p_shape = list(data_array.shape)
        p_shape.insert(idxs['dims'].index('ht'), 1)
        data_array = data_array.reshape(p_shape)
    logger.write('shape of read data for variable {}: {}'.format(
                 var, data_array.shape))
    return data_array


def read_all_data(args, idxs):

    variables = {}
    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        variables[var] = read_netcdf_data(var, idxs, args)
    return variables


def clean_all_vars(args, all_vars, idxs, units):
    """(Namelist, dict of arrays, dict, dict) -> dict of arrays

    Performs several transformations on the datasets:

        1) ensures that the height dimension is in Pascals
        2) ensures that the surface pressure is in Pascals
        3) ensures that the height dimension is ascending

    """

    if units['P'] == 'hPa':
        all_vars['P'] = 100.0 * all_vars['P']
        units['P'] = 'Pa'

    if units['ht'] == 'hPa':
        idxs['ht']['vals'] = 100.0 * idxs['ht']['vals']
        units['ht'] = 'Pa'

    # Pressure values, so higher value is lower level.
    if idxs['ht']['vals'][0] < idxs['ht']['vals'][-1]:
        if idxs['dims'][1] != 'ht':
            raise IndexError("At the moment, only know how to invert the" +
                             "second axis, but that isn't height")
        idxs['ht']['vals'][:] = idxs['ht']['vals'][::-1]
        idxs['ht']['idxs'][:] = idxs['ht']['idxs'][::-1]
        for var in ['U', 'V', 'T', 'Z', 'Q']:
            all_vars[var][:, :, :, :] = all_vars[var][:, ::-1, :, :]

    return all_vars, idxs, units


def spacially_interpolate(args, read_vars, idxs):
    """( Namelist, dict, dict ) -> dict

    Creates a new namelist with spacially interpolated data of args.
    Adds new fields for temperature and humidity gradients.
    """

    import genesis_helpers as h

    return_dict = {}

    lon_frac = np.array(h.find_fractions(args.lon, idxs['lon']['vals'][0],
                                         idxs['lon']['vals'][1]))
    lat_frac = np.array(h.find_fractions(args.lat, idxs['lat']['vals'][0],
                                         idxs['lat']['vals'][1]))

    fracts = lon_frac[np.newaxis, :] * lat_frac[:, np.newaxis]

    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        if not (idxs['dims'][3] == 'lon' and idxs['dims'][2] == 'lat'):
            raise ValueError("At the moment, can only reduce variables" +
                             "with dimension ( time, ht, lat, lon )")
        return_dict[var] = np.add.reduce(read_vars[var] * fracts[..., :, :],
                                         (2, 3))

    dy = h.surface_distance_y(*idxs['lat']['vals'])
    dx = h.surface_distance_x(*idxs['lon']['vals'],
                              lat=idxs['lat']['vals'][1])

    return_dict['dx'] = dx
    return_dict['dy'] = dy

    return return_dict


def cleanup_args(args, base):
    """(Namelist, dict of dict) -> Namelist

    Converts start- and end time to datetime.datetime, calculates endtime if
    num is given, and end_time, if num is given. Can read start and end time
    from base config file, but options in args are gived priority.
    """

    def read_date_from_args(s):
        """(str) -> datetime

        reads the string s and tries to interpret it as datetime.

        >>> read_date_from_args('20150101')
        datetime.datetime(2015, 1, 1, 0, 0)
        >>> read_date_from_args('2015013104')
        datetime.datetime(2015, 1, 31, 4, 0)
        >>> read_date_from_args('201512011231')
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

    stop = False
    # See whether there are any parameters in the base configuration file
    # that we might use.
    base_knows_sdate = 'time' in base.keys() and 'sdate' in base['time'].keys()
    base_knows_edate = 'time' in base.keys() and 'edate' in base['time'].keys()
    if not args.lon:
        print("Need longitude")
        stop = True
    if not args.lat:
        print("Need latitude")
        stop = True
    if not args.start_date:
        if base_knows_sdate:
            if args.debug:
                print(" No Start Date given in arguments, " +
                      "reading from base config file ")
            args.start_date = '{:08}{:02}'.format(base['time']['sdate'],
                                                  base['time']['shour'])
        else:
            print("Need start date")
            stop = True
    if not (args.end_date or args.num):
        if base_knows_edate:
            if args.debug:
                print(" No End Date given in arguments, " +
                      "reading from base config file ")
            args.end_date = '{:08}{:02}'.format(base['time']['edate'],
                                                base['time']['ehour'])
        else:
            print ("Need end date or number")
            stop = True

    if stop:
        print(" Please see help file for details (-h)")
        exit()

    if ':' in args.intervall:
        intervall_hours, intervall_minutes = args.intervall.split(':')
    else:
        intervall_hours = int(args.intervall)
        intervall_minutes = '00'
    args.intervall = datetime.timedelta(hours=int(intervall_hours),
                                        minutes=int(intervall_minutes))

    args.start_date = read_date_from_args(args.start_date)

    if args.end_date:
        args.end_date = read_date_from_args(args.end_date)
        args.num = 0
        while args.start_date + args.num*args.intervall <= args.end_date:
            args.num += 1
    else:
        if args.num:
            args.end_date = (args.num - 1) * args.intervall + args.start_date
        else:
            raise ValueError("Need either end date or num")
    if args.lat < -90. or args.lat > 90.:
        raise ValueError("Latitude out of bounds: {}".format(args.lat))
    if args.lon < 0. or args.lon > 360.:
        raise ValueError("Longitude out of bounds: {}".format(args.lon))

    return args


def parse_arguments():
    """(None) -> args

    returns the results of parse_args()
    """

    parser = argparse.ArgumentParser(description='Cleans up the template file')
    parser.add_argument('-X', '--lon', help='longitude', type=float)
    parser.add_argument('-Y', '--lat', help='latitude', type=float)
    parser.add_argument('-S', '--start-date', help='start date: YYYYMMDD[HHMM]')
    parser.add_argument('-E', '--end-date', help='end date: YYYYMMDD[HHMM]')
    parser.add_argument('-N', '--num', help='number of times', type=int)
    parser.add_argument('--lon-range',
                        help='longitude range -- not implemented', type=float,
                        default=3.0)
    parser.add_argument('--lat-range',
                        help='latitude range -- not implemented', type=float,
                        default=3.0)
    parser.add_argument('-b', '--base', metavar='FILE', default='base.inp',
                        help='Namelist Template')
    parser.add_argument('-t', '--template', metavar='FILE',
                        default='template.scm', help='Namelist Template')
    parser.add_argument('-o', '--output', metavar='FILE', default='t_out.scm',
                        help='Output Namelist')
    parser.add_argument('-r', '--relhum', default=False, action='store_true',
                        help='Convert Relative to Specific Humidity')
    parser.add_argument('-M', '--hPa', default=False, action='store_true',
                        help='Convert surface pressure from hPa to Pa')
    parser.add_argument('-O', '--offset', metavar='FILE',
                        help='User Offset File')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='Debug')
    parser.add_argument('-T', '--test', help='run doctest on this module',
                        default=False, action='store_true')

    args = parser.parse_args()

    setattr(args, 'intervall', '06:00')

    return args


def main():
    args = parse_arguments()

    # Run unit Tests if required
    if args.test:
        import doctest
        doctest.testmod()
        exit()

    # Read the base configuration file
    if os.path.isfile(args.base):
        base = f90nml.read(args.base)
    else:
        print("Base configuration file not found: {}".format(args.base))
        exit(1)

    # Convert the start_date and end_date to datetime format, plus read them in
    # in case there isn't anything given.
    args = cleanup_args(args, base)

    logger = genesis_logger(args.debug)

    logger.write_by_item(args, indent=2, header='command-line parameters:')
    logger.write(" Read from base configuration file.")
    for k in base.keys():
        logger.write_by_item(base[k], key_length=12, indent=4, header=k)

    idxs = nch.get_indices(args)
    logger.write_by_item(idxs, indent=2, key_length=12, header='indices:')

    units = nch.get_all_units(args)
    logger.write_by_item(units, indent=2, key_length=12, header='units:')

    allvars = read_all_data(args, idxs)
    allvars, idxs, units = clean_all_vars(args, allvars, idxs, units)

    spacially_interpolated = spacially_interpolate(args, allvars, idxs)


if __name__ == '__main__':
    main()
