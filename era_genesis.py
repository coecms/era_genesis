#!/usr/bin/env python


import argparse
import os
import numpy as np
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


def read_netcdf_data(var, idxs, conf):
    """(str, dict, Namelist) -> np.ndarray

    Returns all data from the ERA-Interim with variable named var in
    ['U', 'V', 'T', 'Z', 'Q', 'P'] for all the relevant dates.

    """

    assert(var in ['U', 'V', 'T', 'Z', 'Q', 'P'])

    logger = genesis_logger(conf.debug)

    files = nch.file_list(var, conf)

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


def read_all_data(conf, idxs):

    variables = {}
    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        variables[var] = read_netcdf_data(var, idxs, conf)
    return variables


def clean_all_vars(conf, all_vars, idxs, units):
    """(Namelist, dict of arrays, dict, dict) -> dict of arrays

    Performs several transformations on the datasets:

        1) ensures that the height dimension is in Pascals
        2) ensures that the surface pressure is in Pascals
        3) ensures that the height dimension is ascending
        4) ensures that the z-variable is in m

    """

    from genesis_globals import grav

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

    if units['Z'] != 'm':
        all_vars['Z'] = all_vars['Z'] / grav
        units['Z'] = 'm'

    return all_vars, idxs, units


def calc_pt_in(t_in, levs_in):
    """Calculates the potential temperatures on the ERA_Interim levels

    """

    from genesis_globals import rcp

    pt_in = np.empty((0, t_in.shape[1]))

    for t in t_in:
        pt = t[:] * (1e5/levs_in[:])**rcp
        pt_in = np.concatenate((pt_in, pt[np.newaxis, :]), axis=0)
    return pt_in


def calc_geostrophic_winds(z_in, dx, lat):
    """Calculates the geostrophic winds

    Inputs:
        z_in = np.ndarray((rec, height, lat/lon)), geopotential
        dx = float, distance between grid in m
        lat = float, latitude

    Output:
        g = np.ndarray((rec, height))
    """

    from genesis_globals import omg

    f = 2 * omg * np.sin(lat * np.pi / 180.)

    g = np.empty((0, z_in.shape[1]))
    for z in z_in:
        dphidx = (z[:, 1] - z[:, 0]) / dx if dx > 0. else np.zeros_like(z[:, 0])
        g = np.concatenate((g, (1/f) * dphidx[np.newaxis, :]), axis=0)

    return g


def calc_p_in(z_in, msl_array, eta_rho, levs_in):
    """(array, array, array, array) -> array

    z_in: Z in the input files, but in m, shape (nrecs, nlvls_in)
    msl_array: What's read as P in the input files, shape (nrecs)
    eta_rho: From base.inp, eta_rho * z_top_of_model + z_terrain_asl
    levs_in: height values from netCDF files, shape (nlvls_in)

    Calculates pressure profile
    """

    from genesis_globals import grav, rho, maxz

    zzr = np.concatenate((eta_rho, np.array([maxz])))
    p_in = np.zeros((0, len(zzr)))

    for z, msl in zip(z_in, msl_array):

        p_line = np.zeros((1, len(zzr)))

        xp = np.concatenate(
            (np.array([0, 0.999*z[0]]), z, np.array([maxz]))
        )
        yp = np.concatenate((
            np.array([msl, msl-rho*grav*0.999*z[0]]),
            levs_in, np.array([100.])
        ))

        p_line[0, :] = np.interp(zzr, xp, yp)

        p_in = np.concatenate((p_in, p_line))

    return p_in


def vert_interp(var_in, z_in, z_out):
    """Makes a vertical interpolation of the variable var_in

    Input:
        var_in: variable to be interpolated, dimensions (nrec, nlvls)
        z_out: eta_rho levels of the um
        z_in: Z levels of the ERA-Interim

    """

    var_out = np.empty((0, len(z_out)))

    for v, z in zip(var_in, z_in):
        v_um = np.interp(z_out, z, v)
        var_out = np.concatenate((var_out, v_um[np.newaxis, :]), axis=0)

    return var_out


def spatially_interpolate(conf, read_vars, idxs):
    """( Namelist, dict, dict ) -> dict

    Creates a new namelist with spacially interpolated data of conf.
    Adds new fields for temperature and humidity gradients.
    """

    import genesis_helpers as h

    return_dict = {}

    lon_frac = np.array(h.find_fractions(conf.lon, idxs['lon']['vals'][0],
                                         idxs['lon']['vals'][1]))
    lat_frac = np.array(h.find_fractions(conf.lat, idxs['lat']['vals'][0],
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
                              lat=conf.lat)

    return_dict['dx'] = dx
    return_dict['dy'] = dy

    return return_dict


def replace_namelist(template, out_data, conf):
    """(namelist, dict, namelist, Namespace) -> namelist

    Replaces all relevant data in template, then returns this new
    namelist.
    """

    from copy import deepcopy
    from genesis_globals import omg

    f = 2 * omg * np.sin(conf.lat * np.pi / 180.)
    fact = 24 / (conf.intervall.total_seconds() / 3600.0)

    return_namelist = deepcopy(template)

    # Instead of writing lots of return_namelist['something']['varname']
    # I can make use of the shallow copying.
    # So every time I change for example inobsfor, return_namelist['inobsfor']
    # changes as well.
    inobsfor = return_namelist['inobsfor']
    inprof = return_namelist['inprof']
    indata = return_namelist['indata']

    inobsfor['l_windrlx'] = conf.l_windrlx
    return_namelist['cntlscm']['nfor'] = conf.num

    if conf.l_windrlx:
        if conf.tau_rlx:
            inobsfor['tau_rlx'] = conf.intervall.seconds
        if conf.u_inc:
            inobsfor['u_inc'] = out_data['u'].flatten(order='F').tolist()
        if conf.v_inc:
            inobsfor['v_inc'] = out_data['v'].flatten(order='F').tolist()
        if conf.w_inc:
            inobsfor['w_inc'] = 'not implemented yet'
        if conf.t_inc:
            inobsfor['t_inc'] = (
                (out_data['t'][1:, :] - out_data['t'][:-1, :]) * fact
            ).flatten(order='F').tolist()
        if conf.q_star:
            inobsfor['q_star'] = (
                (out_data['qi'][1:, :] - out_data['qi'][:-1, :]) * fact
            ).flatten(order='F').tolist()
    else:
        if conf.u_inc:
            inobsfor['u_inc'] = ((
                out_data['u'][1:, :] - (1.-f) * out_data['u'][:-1, :]
            ) * fact).flatten(order='F').tolist()
        if conf.v_inc:
            inobsfor['v_inc'] = ((
                out_data['v'][1:, :] - (1.+f) * out_data['v'][:-1, :]
            ) * fact).flatten(order='F').tolist()
        if conf.w_inc:
            inobsfor['w_inc'] = 'not implemented yet'
        if conf.t_inc:
            inobsfor['t_inc'] = 'not implemented yet'
        if conf.q_star:
            inobsfor['q_star'] = 'not implemented yet'

    if conf.ui:
        inprof['ui'] = out_data['u'][0, :].flatten(order='F').tolist()
    if conf.vi:
        inprof['vi'] = out_data['v'][0, :].flatten(order='F').tolist()
    if conf.wi:
        inprof['wi'] = 'not implemented yet'
    if conf.theta:
        inprof['theta'] = out_data['theta'][0, :].flatten(order='F').tolist()
    if conf.qi:
        inprof['qi'] = out_data['qi'][0, :].flatten(order='F').tolist()
    if conf.p_in:
        inprof['p_in'] = out_data['p_in'][0, :].flatten(order='F').tolist()

    indata['lat'] = conf.lat
    indata['long'] = conf.lon
    indata['year_init'] = conf.start_date.year
    indata['month_init'] = conf.start_date.month
    indata['day_init'] = conf.start_date.day
    indata['hour_init'] = conf.start_date.hour

    delta = conf.end_date - conf.start_date
    return_namelist['rundata']['nminin'] = int((delta.total_seconds()+1) / 60)

    inobsfor['tstar_forcing'] = [288.0] * conf.num

    return return_namelist


def write_charney(out_vars, levs, file_name='charney.csv'):
    """writes the data like genesis' charney.scm, except in csv format
    """

    format_data = '{p_um:9.0f},{adum:9.0f},{t_um:9.2f},{pt_um:9.2f},' + \
        '{q_um:12.5e},{u_um:8.2f},{v_um:8.2f}\n'
    format_header = '{p_um:>9},{adum:>9},{t_um:>9},{pt_um:>9},{q_um:>12},' + \
        '{u_um:>8},{v_um:>8}\n'
    with open(file_name, 'w') as charney:
        w_dict = {
            'p_um': "p_in",
            'adum': "pres.lvl",
            't_um': "temp",
            'pt_um': "theta",
            'q_um': "sp. hum.",
            'u_um': "Wind U",
            'v_um': "Wind V"
        }
        charney.write(format_header.format(**w_dict))
        for i in range(out_vars['theta'].shape[1]):
            rho_lev = i < out_vars['u'].shape[1]
            w_dict = {
                'p_um': out_vars['p_in'][-1, i],
                'adum': levs[i] if i < len(levs) else 0.0,
                't_um': out_vars['t'][-1, i],
                'pt_um': out_vars['theta'][-1, i],
                'q_um': out_vars['qi'][-1, i],
                'u_um': out_vars['u'][-1, i] if rho_lev else 0.0,
                'v_um': out_vars['v'][-1, i] if rho_lev else 0.0
            }
            charney.write(format_data.format(**w_dict))
    charney.close()


def write_genesis(allvars, levs, file_name='genesis.csv'):

    format_header = '{:>9},{:>9},{:>8},{:>8},{:>12},{:>8},{:>8},{:>8},{:>8}\n'
    format_data = '{z:9.0f},{levs:9.0f},{t:8.2f},{pt:8.2f},{q:12.5e},' + \
        '{u:8.2f},{v:8.2f},{ug:8.2f},{vg:8.2f}\n'

    with open(file_name, 'w') as genesis:
        genesis.write(format_header.format(
            'Z', 'levs', 't', 'pt', 'q', 'u', 'v', 'ug', 'vg'
        ))
        w_dict = {
            'z': 0.0,
            'levs': allvars['P'][0, 0],
            't': allvars['T'][0, 0],
            'pt': allvars['pt'][0, 0],
            'q': allvars['Q'][0, 0],
            'u': allvars['U'][0, 0],
            'v': allvars['V'][0, 0],
            'ug': allvars['ug'][0, 0],
            'vg': allvars['vg'][0, 0]
        }
        genesis.write(format_data.format(**w_dict))
        for i in range(allvars['U'].shape[1]):
            w_dict = {
                'z': allvars['Z'][0, i],
                'levs': levs[i],
                't': allvars['T'][0, i],
                'pt': allvars['pt'][0, i],
                'q': allvars['Q'][0, i],
                'u': allvars['U'][0, i],
                'v': allvars['V'][0, i],
                'ug': allvars['ug'][0, i],
                'vg': allvars['vg'][0, i]
            }
            genesis.write(format_data.format(**w_dict))
    genesis.close()


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
    parser.add_argument('-b', '--base', metavar='FILE', default='base.inp',
                        help='Namelist Template')
    parser.add_argument('-t', '--template', metavar='FILE',
                        default='template.scm', help='Namelist Template')
    parser.add_argument('-o', '--output', metavar='FILE', default='t_out.scm',
                        help='Output Namelist')
    parser.add_argument('-r', '--relhum', default=False, action='store_true',
                        help='Convert Relative to Specific Humidity')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='Debug')
    parser.add_argument('-T', '--test', help='run doctest on this module',
                        default=False, action='store_true')

    args = parser.parse_args()

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
    conf = h.Genesis_Config(args, base)

    logger = genesis_logger(conf.debug)

    logger.write_by_item(conf, indent=2, header='command-line parameters:')
    logger.write(" Read from base configuration file.")
    for k in base.keys():
        logger.write_by_item(base[k], key_length=12, indent=4, header=k)

    idxs = nch.get_indices(conf)
    logger.write_by_item(idxs, indent=2, key_length=12, header='indices:')

    units = nch.get_all_units(conf)
    logger.write_by_item(units, indent=2, key_length=12, header='units:')

    allvars = read_all_data(conf, idxs)
    allvars, idxs, units = clean_all_vars(conf, allvars, idxs, units)

    allvars_si = spatially_interpolate(conf, allvars, idxs)

    pressure_levs = idxs['ht']['vals']
    allvars_si['pt'] = calc_pt_in(allvars_si['T'], pressure_levs)
    allvars_si['ug'] = -calc_geostrophic_winds(
        allvars['Z'][:, :, :, 1], allvars_si['dy'], conf.lon
        )
    allvars_si['vg'] = calc_geostrophic_winds(
        allvars['Z'][:, :, 0, :], allvars_si['dx'], conf.lat
        )

    eta_theta = h.get_eta_theta(conf)
    eta_rho = h.get_eta_rho(conf)
    z_in = allvars_si['Z']
    logger.write('era_pl: ' + str(pressure_levs))
    logger.write('eta_theta: ' + str(eta_theta))
    logger.write('eta_rho: ' + str(eta_rho))
    logger.write('z_in: ' +
                 str(z_in[0, :].flatten().tolist()))

    out_data = {}

    levs = idxs['ht']['vals']

    out_data['p_in'] = calc_p_in(allvars_si['Z'], allvars_si['P'].flatten(),
                                 eta_rho, levs)
    out_data['qi'] = vert_interp(allvars_si['Q'], z_in, eta_theta)
    out_data['theta'] = vert_interp(allvars_si['pt'], z_in, eta_theta)
    out_data['u'] = vert_interp(allvars_si['U'], z_in, eta_rho)
    out_data['v'] = vert_interp(allvars_si['V'], z_in, eta_rho)
    out_data['t'] = vert_interp(allvars_si['T'], z_in, eta_theta)

    write_genesis(allvars_si, levs)
    write_charney(out_data, levs)

    template = f90nml.read(conf.template)

    out_namelist = replace_namelist(template, out_data, conf)

    out_namelist.write(conf.output, force=True)


if __name__ == '__main__':
    main()
