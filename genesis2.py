#!/usr/bin/env python


import argparse
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


def get_filename(var, date):
    """( str, datetime.datetime ) -> str

    Creates a string pointing to the file that contains the required
    variable var for the date date.

    >>> d = datetime.datetime( year=2000, month=10, day=5 )
    >>> get_filename( 'U', d )
    '/g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2000/U_6hrs_pl_2000_10.nc'
    >>> get_filename( 'V', d )
    '/g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2000/V_6hrs_pl_2000_10.nc'
    >>> get_filename( 'T', d )
    '/g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2000/T_6hrs_pl_2000_10.nc'
    >>> get_filename( 'Z', d )
    '/g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2000/Z_6hrs_pl_2000_10.nc'
    >>> get_filename( 'Q', d )
    '/g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2000/Q_6hrs_pl_2000_10.nc'
    >>> get_filename( 'P', d )
    '/g/data1/ua8/erai/netcdf/oper_an_sfc/fullres/sub-daily/2000/MSL_6hrs_sfc_2000_10.nc'
    >>> d = datetime.datetime( year=2010, month=1, day=1 )
    >>> get_filename( 'T', d )
    '/g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2010/T_6hrs_pl_2010_01.nc'

    """

    file_template = '/g/data1/ua8/erai/netcdf/oper_an_{level:}/fullres/sub-daily/{year:4}/{var:}_6hrs_{level:}_{year:4}_{month:02}.nc'
    assert( var in ['U', 'V', 'T', 'Z', 'Q', 'P'] )

    vals = {
        'year'  : date.year,
        'month' : date.month,
        'level' : 'pl',             # Pressure levels
        'var'   : var
    }
    if var == 'P':
        vals['level'] = 'sfc'       # Surface
        vals['var'] = 'MSL'

    return file_template.format(**vals)

def get_varname(var):
    """(str) -> str

    returns the proper var name for the given var in ['U', 'V', 'T', 'Z', 'Q', 'P']

    >>> get_varname('U')
    'U_GDS0_ISBL'
    >>> get_varname('V')
    'V_GDS0_ISBL'
    >>> get_varname('T')
    'T_GDS0_ISBL'
    >>> get_varname('Z')
    'Z_GDS0_ISBL'
    >>> get_varname('Q')
    'Q_GDS0_ISBL'
    >>> get_varname('P')
    'MSL_GDS0_SFC'
    """

    assert( var in ['U', 'V', 'T', 'Z', 'Q', 'P'] )

    if var == 'P':
        return 'MSL_GDS0_SFC'
    else:
        return '{:1}_GDS0_ISBL'.format(var)

def parse_arguments():
    """(None) -> args

    returns the results of parse_args()
    """

    def cleanup_args(args):
        """(Namelist) -> Namelist

        Converts start- and end time to datetime.datetime, calculates endtime if num is given,
        and end_time, if num is given.

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

        return args

    parser = argparse.ArgumentParser(description='Cleans up the template file')
    parser.add_argument('-X', '--lon', help='longitude', type=float) #, required=True)
    parser.add_argument('-Y', '--lat', help='latitude', type=float) #, required=True)
    parser.add_argument('-S', '--start-date', help='start date: YYYYMMDD[HHMM]') #,
#                      required=True)
    parser.add_argument('-E', '--end-date', help='end date: YYYYMMDD[HHMM]')
    parser.add_argument('-N', '--num', help='number of times', type=int)
#    parser.add_argument('-I', '--intervall', help='intervall: HH[:MM]',
#                      default='06:00')
    parser.add_argument('-t', '--template', metavar='FILE', default='template.scm',
                      help='Namelist Template')
    parser.add_argument('-o', '--output', metavar='FILE', default='t_out.scm',
                      help='Output Namelist')
    parser.add_argument('-r', '--relhum', default=False, action='store_true',
                      help='Convert Relative to Specific Humidity')
    parser.add_argument('-M', '--hPa', default=False, action='store_true',
                      help='Convert surface pressure from hPa to Pa')
    parser.add_argument('-D', '--date', default='dates.dat', metavar='FILE',
                      help='User Date file')
    parser.add_argument('-O', '--offset', metavar='FILE',
                      help='User Offset File')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                      help='Debug')
    parser.add_argument('-T', '--test', help='run doctest on this module', default=False, action='store_true')

    args = parser.parse_args()

    if args.test:
        import doctest
        doctest.testmod()
        exit()

    setattr( args, 'intervall', '06:00')

    if (not (args.lon and args.lat and args.start_date and (args.end_date or args.num))):
        if not args.lon: print( "Need longitude" )
        if not args.lat: print( "Need latitude" )
        if not args.start_date: print( "Need start date" )
        if not (args.end_date or args.num): print ( "Need end date or number" )
        parser.print_help()
        exit()

    args = cleanup_args(args)
    return args

def main():
    args = parse_arguments()

    if args.debug:
        for k in args.__dict__:
            print( "{:10}: {}".format(k, args.__dict__[k]) )




if __name__ == '__main__':
    main()
