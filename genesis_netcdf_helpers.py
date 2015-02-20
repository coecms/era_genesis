#!/usr/bin/env python

import datetime
import numpy as np
import netCDF4 as cdf

class dummyargs(object):
    start_date = datetime.datetime(2000, 1, 1, 0, 0)
    end_date = datetime.datetime(2001, 2, 1, 0, 0)
    def __init__(self):
        self.start_date = datetime.datetime(2000, 1, 1, 0, 0)
        self.end_date = datetime.datetime(2001, 2, 1, 0, 0)


def next_month(month):
    """(datetime) -> datetime

    Returns the 1. of the next month.

    >>> d = datetime.datetime(year=2000, month=10, day=30)
    >>> d = next_month(d)
    >>> d # month after october 2000
    datetime.datetime(2000, 11, 1, 0, 0)
    >>> d = next_month(d)
    >>> d # month after Nov 2000
    datetime.datetime(2000, 12, 1, 0, 0)
    >>> d = next_month(d)
    >>> d # month after Dec 2000
    datetime.datetime(2001, 1, 1, 0, 0)
    """
    if month.month == 12: return datetime.datetime(month.year+1, 1, 1)
    return datetime.datetime(month.year, month.month+1, 1)
def file_list(var, args):
    """(str, Namelist) -> list of str

    Returns a list of file names that need to be opened to gather variable var
    for all dates between args.start_date and args.end_date

    >>> a = dummyargs()
    >>> a.start_date = datetime.datetime(2011, 1, 1, 3, 0)
    >>> a.end_date = datetime.datetime(2012, 2, 1, 0, 0)
    >>> files = file_list('U', a)
    >>> for f in files: print( f )
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_01.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_02.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_03.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_04.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_05.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_06.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_07.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_08.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_09.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_10.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_11.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2011/U_6hrs_pl_2011_12.nc
    /g/data1/ua8/erai/netcdf/oper_an_pl/fullres/sub-daily/2012/U_6hrs_pl_2012_01.nc

    """
    files = []
    this_month = datetime.datetime(year=args.start_date.year, month=args.start_date.month, day=1)
    while this_month < args.end_date:
        files.append(get_filename(var,this_month))
        this_month = next_month(this_month)
    return files

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

def get_lat_name(var):
    """ (str) -> str

    Returns the name of the latitude array for the variable var.

    >>> get_lat_name('U')
    'g0_lat_2'
    >>> get_lat_name('V')
    'g0_lat_2'
    >>> get_lat_name('T')
    'g0_lat_2'
    >>> get_lat_name('U')
    'g0_lat_2'
    >>> get_lat_name('Q')
    'g0_lat_2'
    >>> get_lat_name('P')
    'g0_lat_1'
    """
    assert( var in ['U', 'V', 'T', 'Z', 'Q', 'P'] )
    if var == 'P': return 'g0_lat_1'
    return 'g0_lat_2'

def get_lon_name(var):
    """ (str) -> str

    Returns the name of the longitude array for the variable var.

    >>> get_lon_name('U')
    'g0_lon_3'
    >>> get_lon_name('V')
    'g0_lon_3'
    >>> get_lon_name('T')
    'g0_lon_3'
    >>> get_lon_name('U')
    'g0_lon_3'
    >>> get_lon_name('Q')
    'g0_lon_3'
    >>> get_lon_name('P')
    'g0_lon_2'
    """
    assert( var in ['U', 'V', 'T', 'Z', 'Q', 'P'] )
    if var == 'P': return 'g0_lon_2'
    return 'g0_lon_3'

def get_time_name(var):
    """ (str) -> str

    Returns the name of the time array for the variable var.

    >>> get_time_name('U')
    'initial_time0_hours'
    >>> get_time_name('V')
    'initial_time0_hours'
    >>> get_time_name('T')
    'initial_time0_hours'
    >>> get_time_name('U')
    'initial_time0_hours'
    >>> get_time_name('Q')
    'initial_time0_hours'
    >>> get_time_name('P')
    'initial_time0_hours'
    """
    assert( var in ['U', 'V', 'T', 'Z', 'Q', 'P'] )
    return 'initial_time0_hours'

def get_ht_name(var):
    """ (str) -> str

    Returns the name of the height dimension for variable var.
    Returns empty string for 'P' because 'P' is a 2D field.

    >>> get_ht_name('U')
    'lv_ISBL1'
    >>> get_ht_name('V')
    'lv_ISBL1'
    >>> get_ht_name('T')
    'lv_ISBL1'
    >>> get_ht_name('Z')
    'lv_ISBL1'
    >>> get_ht_name('Q')
    'lv_ISBL1'
    >>> get_ht_name('P')
    ''
    """
    assert (var in ['U', 'V', 'T', 'Z', 'Q', 'P'])
    if var == 'P': return ''
    return 'lv_ISBL1'

def genesis_open_netCDF( file_handle ):
    if type(file_handle) == str:
        return cdf.Dataset( file_handle, 'r' ), True
    elif type(file_handle) == cdf.Dataset:
        return file_handle, False
    raise ValueError( "File Handle must be string or Dataset" )

def genesis_close_netCDF( file_handle, opened ):
    if opened:
        file_handle.close()

def get_dimension_lengths(file_name):
    """(str) -> dict of str->int

    Returns a dictionary with dimension names as keys and the dimension sizes as values

    """

    return_dict = {}

    dataset = cdf.Dataset( file_name, 'r' )
    dimensions = dataset.dimensions
    for dim in dimensions:
        return_dict[ dim ] = len(dimensions[dim])
    dataset.close()

    return return_dict


def get_shape(file_handle, var_name):
    """ (str, str) -> tuple of (list of int, list of str)

    Returns the shape of the variable var_name in the file file_name, and a list of dimension names.
    file_handle can be a string pointing to a netCDF source, or a netcdf Dataset. If it is a Dataset,
    it will remain open.

    >>> file_name = get_filename('U', datetime.datetime(2010, 1, 1))
    >>> var_name = get_varname('U')
    >>> get_shape(file_name, var_name)
    ([124, 37, 241, 480], [u'initial_time0_hours', u'lv_ISBL1', u'g0_lat_2', u'g0_lon_3'])
    >>> ncid = cdf.Dataset(file_name, 'r')
    >>> get_shape(ncid, var_name)
    ([124, 37, 241, 480], [u'initial_time0_hours', u'lv_ISBL1', u'g0_lat_2', u'g0_lon_3'])
    >>> ncid.close()
    """

    ncid, opened_here = genesis_open_netCDF( file_handle )
    variable = ncid.variables[ var_name ]
    shape = []
    names = []
    for d in variable.dimensions:
        names.append( d )
        shape.append(len(ncid.dimensions[d]))
    genesis_close_netCDF( ncid, opened_here )

    return shape, names

def read_array( file_handle, var_name, shape = None):
    """ (str or Dataset, str) -> ndarray

    Returns all data from the netCDF file file_handle with the variable name var_name.
    file_handle can be a string containing the file name, or it can be a netCDF dataset.
    If it contains the file name, it will be opened, read, and closed, otherwise it will remain open.

    >>> file_name = get_filename('U', datetime.datetime(2000, 1, 1))
    >>> var_name = get_ht_name('U')
    >>> read_array( file_name, var_name )
    array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
            125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
            550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
            925,  950,  975, 1000], dtype=int32)
    >>> read_array( file_name, var_name, [[5, 6, 7, 8]] )
    array([10, 20, 30, 50], dtype=int32)
    """

    ncid, opened_here = genensis_open_netCDF( file_handle )

    if shape:
        return_array = ncid.variables[var_name][shape]
    else:
        return_array = ncid.variables[var_name][...]

    genesis_close_netCDF( ncid, opened_here )

    return return_array






if __name__ == '__main__':
    import doctest
    doctest.testmod()

