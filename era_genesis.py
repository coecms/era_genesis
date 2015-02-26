#!/usr/bin/env python


import argparse
import os
import numpy as np
import datetime
import genesis_netcdf_helpers as nch
import genesis_helpers as h

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
                    ki = '{:'+str(key_length)+'}'
                    print((' '*indent + ki + ': {}').format(key, val))
            elif type(obj) == argparse.Namespace:
                for key in obj.__dict__:
                    val = getattr(obj, key)
                    ki = '{:'+str(key_length)+'}'
                    print((' '*indent + ki + ': {}').format(key, val))
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


def vertically_interpolate(data_in, eta_theta, eta_rho, orig_levs):
    """(dict, array, array) -> dict

    """

    return_dict = {
        'eta_theta': eta_theta,
        'eta_rho': eta_rho
    }

    theta_converter = h.calc_ht_conversion(orig_levs, eta_theta)
    rho_converter = h.calc_ht_conversion(orig_levs, eta_rho)

    for v in ['T', 'Q']:
        return_dict[v] = h.convert_height(data_in[v], theta_converter)

    for v in ['U', 'V']:
        return_dict[v] = h.convert_height(data_in[v], rho_converter)

    return return_dict


def replace_namelist(template, out_data, base, args):
    """(namelist, dict, namelist, Namespace) -> namelist

    Replaces all relevant data in template, then returns this new
    namelist.
    """

    from copy import deepcopy

    return_namelist = deepcopy(template)

    l_windrlx = base['usrfields_2']['l_windrlx']

    return_namelist['inobsfor']['l_windrlx'] = l_windrlx
    return_namelist['cntlscm']['nfor'] = args.num

    print( "l_windrlx: {}".format(l_windrlx))

    if l_windrlx:
        if base['usrfields_2']['tau_rlx']:
            return_namelist['inobsfor']['tau_rlx'] = args.intervall.seconds
        if base['usrfields_2']['u_inc']:
            return_namelist['inobsfor']['u_inc'] = '' #out_data['U'].tolist()
        if base['usrfields_2']['v_inc']:
            return_namelist['inobsfor']['v_inc'] = 'not implemented yet'
        if base['usrfields_2']['w_inc']:
            return_namelist['inobsfor']['w_inc'] = 'not implemented yet'
        if base['usrfields_2']['t_inc']:
            return_namelist['inobsfor']['t_inc'] = 'not implemented yet'
        if base['usrfields_2']['qstar']:
            return_namelist['inobsfor']['q_star'] = 'not implemented yet'
    else:
        if base['usrfields_2']['u_inc']:
            return_namelist['inobsfor']['u_inc'] = out_data['U'].flatten().tolist()
        if base['usrfields_2']['v_inc']:
            return_namelist['inobsfor']['v_inc'] = 'not implemented yet'
        if base['usrfields_2']['w_inc']:
            return_namelist['inobsfor']['w_inc'] = 'not implemented yet'
        if base['usrfields_2']['t_inc']:
            return_namelist['inobsfor']['t_inc'] = 'not implemented yet'
        if base['usrfields_2']['q_star']:
            return_namelist['inobsfor']['q_star'] = 'not implemented yet'

    if base['usrfields_1']['ui']:
        return_namelist['inprof']['ui'] = 'not implemented yet'
    if base['usrfields_1']['vi']:
        return_namelist['inprof']['vi'] = 'not implemented yet'
    if base['usrfields_1']['wi']:
        return_namelist['inprof']['wi'] = 'not implemented yet'
    if base['usrfields_1']['theta']:
        return_namelist['inprof']['theta'] = 'not implemented yet'
    if base['usrfields_1']['qi']:
        return_namelist['inprof']['qi'] = 'not implemented yet'
    if base['usrfields_1']['p_in']:
        return_namelist['inprof']['p_in'] = 'not implemented yet'

    return_namelist['indata']['lat'] = args.lat
    return_namelist['indata']['long'] = args.lon
    return_namelist['indata']['year_init'] = args.start_date.year
    return_namelist['indata']['month_init'] = args.start_date.month
    return_namelist['indata']['day_init'] = args.start_date.day
    return_namelist['indata']['hour_init'] = args.start_date.hour

    delta = args.end_date - args.start_date
    return_namelist['rundata']['nminin'] = int(delta.total_seconds() / 60)

    return_namelist['inobsfor']['tstar_forcing'] = [288.0] * args.num

    return return_namelist


def convert_base_time_to_args_time(base, datename, hourname):
    """(namelist, str, str) -> str

    Converts the data from the base namelist in time->datename and
    time->hourname into an 12 character time representation according
    to the arguments:

    >>> base={'time':{'sdate':20100101, 'shour':0, 'edate':20103101, \
                      'ehour':18}}
    >>> convert_base_time_to_args_time(base, 'sdate', 'shour')
    '2010010100'
    >>> convert_base_time_to_args_time(base, 'edate', 'ehour')
    '2010310118'
    """

    return '{:08}{:02}'.format(base['time'][datename],
                               base['time'][hourname])


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


def get_start_date(args, base):
    """(Namespace, namelist) -> datetime

    Returns the start time. If args.start_date is set, then it does nothing.
    Otherwise, it reads the start_date from the base namelist.

    >>> base={'time':{'sdate':20100101, 'shour':0, 'edate':20100131, \
                      'ehour':18}}
    >>> args=argparse.Namespace()
    >>> args.start_date = None
    >>> get_start_date(args, base) # args not set
    datetime.datetime(2010, 1, 1, 0, 0)
    >>> args.start_date = '20000323' ; get_start_date(args, base)
    datetime.datetime(2000, 3, 23, 0, 0)
    >>> args.start_date = '2001012112' ; get_start_date(args, base)
    datetime.datetime(2001, 1, 21, 12, 0)
    >>> args.start_date = '196711290630' ; get_start_date(args, base)
    datetime.datetime(1967, 11, 29, 6, 30)
    """

    if args.start_date:
        date_string = args.start_date
    else:
        date_string = convert_base_time_to_args_time(base, 'sdate', 'shour')

    return read_date_from_args(date_string)


def get_end_date(args, base):
    """(Namespace, namelist) -> datetime

    Returns the start time. If args.start_date is set, then it does nothing.
    Otherwise, it reads the start_date from the base namelist.

    >>> base={'time':{'sdate':20000101, 'shour':0, 'edate':20000131, \
                      'ehour':18}}
    >>> args=argparse.Namespace()
    >>> args.start_date = datetime.datetime(2000, 1, 1, 0, 0)
    >>> args.intervall = datetime.timedelta(hours=6)
    >>> args.end_date = None ; args.num = None ; get_end_date(args, base)
    datetime.datetime(2000, 1, 31, 18, 0)
    >>> args.num=24 ; get_end_date(args, base)
    datetime.datetime(2000, 1, 6, 18, 0)
    >>> args.end_date = '20000323' ; get_end_date(args, base)
    datetime.datetime(2000, 3, 23, 0, 0)
    >>> args.end_date = '2001012112' ; get_end_date(args, base)
    datetime.datetime(2001, 1, 21, 12, 0)
    >>> args.end_date = '196711290630' ; get_end_date(args, base)
    datetime.datetime(1967, 11, 29, 6, 30)
    """

    if args.end_date:
        date_string = args.end_date
    elif args.start_date and args.num:
        return args.start_date + (args.num-1) * args.intervall
    else:
        date_string = convert_base_time_to_args_time(base, 'edate', 'ehour')

    return read_date_from_args(date_string)


def cleanup_args(args, base):
    """(Namelist, dict of dict) -> Namelist

    Converts start- and end time to datetime.datetime, calculates endtime if
    num is given, and end_time, if num is given. Can read start and end time
    from base config file, but options in args are gived priority.
    """

    stop = False
    logger = genesis_logger(args.debug)
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
    if not (args.start_date or base_knows_sdate):
        logger.write(" No Start Date given")
        stop = True
    if not (args.end_date or args.num or base_knows_edate):
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

    args.start_date = get_start_date(args, base)
    args.end_date = get_end_date(args, base)

    args.num = 1
    while args.start_date + args.num*args.intervall <= args.end_date:
        args.num += 1

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

    pressure_levs = idxs['ht']['vals']
    eta_theta = h.get_eta_theta(base)
    eta_rho = h.get_eta_rho(base)

    out_data = vertically_interpolate(spacially_interpolated, eta_theta,
                                      eta_rho, pressure_levs)
    template = f90nml.read(args.template)

    out_namelist = replace_namelist(template, out_data, base, args)

    out_namelist.write(args.output, force=True)


if __name__ == '__main__':
    main()
