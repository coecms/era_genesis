#!/usr/bin/env python


import argparse
import os
import numpy as np
import datetime

try:
    import f90nml
except:
    print( """
    Failed to import f90nml module.
    module use ~access/modules
    module load python pythonlib/f90nml pythonlib/netCDF4
    """ )

try:
    import netCDF4 as cdf
except:
    print( """
    Failed to import netCDF4 module
    module use ~access/modules
    module load python pythonlib/f90nml pythonlib/netCDF4
    """ )


def read_netcdf_data(var, args):
    """(str, Namelist) -> np.ndarray

    Returns all data from the ERA-Interim with variable named var in ['U', 'V', 'T', 'Z', 'Q', 'P']
    for all the relevant dates.

    """

    import genesis_netcdf_helpers as nch
    import genesis_helpers as h

    assert( var in ['U', 'V', 'T', 'Z', 'Q', 'P'] )

    files = nch.file_list(var, args)

    if args.debug:
        print( 'files to open for variable {}'.format(var) )
        for f in files:
            print(f)

    ncid, opened_here = nch.genesis_open_netCDF(files[0])

    lat_array = nch.read_array(ncid, nch.get_lat_name(var))
    lat_idxs = h.find_nearest_indices( lat_array, args.lat )
    if args.debug:
        print( "Lat array grid points: {},  indices: {}".format(lat_array[lat_idxs], lat_idxs))

    lon_array = nch.read_array(ncid, nch.get_lon_name(var))
    lon_idxs = h.find_nearest_indices( lon_array, args.lon )
    if args.debug:
        print( "Lon array grid points: {},  indices: {}".format(lon_array[lon_idxs], lon_idxs))

    nch.genesis_close_netCDF( ncid, opened_here )

    first_file = True
    for f in files:
        ncid, opened_here = nch.genesis_open_netCDF( f )
        shape, dims = nch.get_shape( ncid, nch.get_varname(var) )
        for i in range(len(dims)):
            if dims[i] == nch.get_lat_name(var):
                shape[i] = lat_idxs
                if first_file:
                    lat_axis = i
                else:
                    if lat_axis != i:
                        raise IndexError( "NetCDF files are not consistent" )
            elif dims[i] == nch.get_lon_name(var):
                shape[i] = lon_idxs
                if first_file:
                    lon_axis = i
                else:
                    if lon_axis != i:
                        raise IndexError( "NetCDF files are not consistent" )
            elif dims[i] == nch.get_time_name(var):
                times_var = ncid.variables[nch.get_time_name(var)]
                times = cdf.num2date( times_var[:], units=times_var.units )
                try:
                    start_idx = np.where( times == args.start_date )[0][0]
                except IndexError:
                    start_idx = 0
                try:
                    end_idx = np.where( times == args.end_date )[0][0]
                except IndexError:
                    end_idx = len(times)
                shape[i] = list(range(start_idx, end_idx+1))
                if first_file:
                    time_axis = i
                else:
                    if time_axis != i:
                        raise IndexError( "NetCDF files are not consistent" )
            else:
                if dims[i] == nch.get_ht_name( var ):
                    if first_file:
                        ht_axis = i
                    else:
                        if ht_axis != i:
                            raise IndexError( "NetCDF files are not consistent" )
                shape[i] = list(range(shape[i]))
        try:
            this_data = nch.read_array( ncid, nch.get_varname(var), shape )
        except Exception as e :
            print( shape )
            raise e
        if first_file:
            data_array = this_data
        else:
            data_array = np.concatenate( (data_array, this_data), axis=time_axis )
        nch.genesis_close_netCDF( ncid, opened_here )
        first_file = False
    if args.debug:
        print( 'shape of read data for variable {}: {}'.format(var, data_array.shape ))
    return data_array

def read_all_data(args):

    variables = {}
    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        variables[var] = read_netcdf_data( var, args )
    return variables






def cleanup_args(args, base):
    """(Namelist, dict of dict) -> Namelist

    Converts start- and end time to datetime.datetime, calculates endtime if num is given,
    and end_time, if num is given. Can read start and end time from base config file, but
    options in args are gived priority.

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
        print( "Need longitude" )
        stop = True
    if not args.lat:
        print( "Need latitude" )
        stop = True
    if not args.start_date:
        if base_knows_sdate:
            if args.debug: print(" No Start Date given in arguments, reading from base config file ")
            args.start_date = '{:08}{:02}'.format(base['time']['sdate'], base['time']['shour'])
        else:
            print( "Need start date" )
            stop = True
    if not (args.end_date or args.num):
        if base_knows_edate:
            if args.debug: print(" No End Date given in arguments, reading from base config file ")
            args.end_date = '{:08}{:02}'.format(base['time']['edate'], base['time']['ehour'])
        else:
            print ( "Need end date or number" )
            stop = True

    if stop:
        print( " Please see help file for details (-h) ")
        exit()

    if ':' in args.intervall:
        intervall_hours, intervall_minutes = args.intervall.split(':')
    else:
        intervall_hours = int(args.intervall)
        intervall_minutes = '00'
    args.intervall = datetime.timedelta(hours=int(intervall_hours), minutes=int(intervall_minutes))

    args.start_date = read_date_from_args( args.start_date )

    if args.end_date:
        args.end_date = read_date_from_args( args.end_date )
        args.num = 0
        while args.start_date + args.num*args.intervall <= args.end_date:
            args.num += 1
    else:
        if args.num:
            args.end_date = (args.num - 1) * args.intervall + args.start_date
        else:
            raise ValueError( "Need either end date or num")
    if args.lat < -90. or args.lat > 90.:
        raise ValueError( "Latitude out of bounds: {}".format(args.lat))
    if args.lon < 0. or args.lon > 360.:
        raise ValueError( "Longitude out of bounds: {}".format(args.lon))


    return args

def parse_arguments():
    """(None) -> args

    returns the results of parse_args()
    """

    import genesis_helpers as helpers


    parser = argparse.ArgumentParser(description='Cleans up the template file')
    parser.add_argument('-X', '--lon', help='longitude', type=float) #, required=True)
    parser.add_argument('-Y', '--lat', help='latitude', type=float) #, required=True)
    parser.add_argument('-S', '--start-date', help='start date: YYYYMMDD[HHMM]') #,
#                      required=True)
    parser.add_argument('-E', '--end-date', help='end date: YYYYMMDD[HHMM]')
    parser.add_argument('-N', '--num', help='number of times', type=int)
#    parser.add_argument('-I', '--intervall', help='intervall: HH[:MM]',
#                      default='06:00')
    parser.add_argument('--lon-range', help='longitude range -- not implemented', type=float,
                        default=3.0)
    parser.add_argument('--lat-range', help='latitude range -- not implemented', type=float,
                        default=3.0)
    parser.add_argument('-b', '--base', metavar='FILE', default='base.inp',
                      help='Namelist Template')
    parser.add_argument('-t', '--template', metavar='FILE', default='template.scm',
                      help='Namelist Template')
    parser.add_argument('-o', '--output', metavar='FILE', default='t_out.scm',
                      help='Output Namelist')
    parser.add_argument('-r', '--relhum', default=False, action='store_true',
                      help='Convert Relative to Specific Humidity')
    parser.add_argument('-M', '--hPa', default=False, action='store_true',
                      help='Convert surface pressure from hPa to Pa')
#    parser.add_argument('-D', '--date', default='dates.dat', metavar='FILE',
#                      help='User Date file')
    parser.add_argument('-O', '--offset', metavar='FILE',
                      help='User Offset File')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                      help='Debug')
    parser.add_argument('-T', '--test', help='run doctest on this module', default=False, action='store_true')

    args = parser.parse_args()

    setattr( args, 'intervall', '06:00')

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
        print( "Base configuration file not found: {}".format(args.base) )
        exit(1)

    # Convert the start_date and end_date to datetime format, plus read them in
    # in case there isn't anything given.
    args = cleanup_args(args, base)

    if args.debug:
        print( " Read from command line arguments, and cleaned up ")
        for k in args.__dict__:
            print( "{:12}: {}".format(k, args.__dict__[k]) )

        print( "\n Read from base configuration file." )
        for k in base.keys():
            print( '{}'.format(k) )
            for kk in base[k].keys():
              print( '     {:12}: {}'.format(kk, base[k][kk]))

    read_all_data(args)

if __name__ == '__main__':
    main()
