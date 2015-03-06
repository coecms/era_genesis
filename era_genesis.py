#!/usr/bin/env python


import argparse
import os
import numpy as np
from era_data import era_dataset
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


def calc_pt(t_si, ht):
    from genesis_globals import rcp

    pt_si = np.empty((0, t_si.shape[1]))
    for t in t_si:
        pt = t[:] * (1e5/ht[:])**rcp
        pt_si = np.concatenate((pt_si, pt[np.newaxis, :]))
    return pt_si


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


def calc_p_um(ht, z_si, z_rho, msl):
    """
        / MSL - rho*grav*z_rho                      for z_rho < z_si[0]
p_um = {  interp_ht(HT, z_si, z_rho                 for z[0] <= z_rho <= z[-1]
        \ HT[-1] * (maxz - z_rho)/(maxz - z[-1])    for z[-1] < z_rho
    """

    from genesis_globals import rho, grav, maxz

    def interp(x, xp, fp):
        if xp[0] > xp[-1]:
            return np.interp(x, xp[::-1], fp[::-1])
        else:
            return np.interp(x, xp, fp)

    p_um = np.empty((z_si.shape[0], len(z_rho)+1))

    for rec in range(z_si.shape[0]):
        for i in range(len(z_rho)):
            if (z_rho[i] < z_si[rec, 0]):
                p_um[rec, i] = msl[rec] - rho*grav*z_rho[i]
            elif (z_rho[i] > z_si[rec, -1]):
                p_um[rec, i] = ht[-1] * (maxz - z_rho[i])/(maxz - z_si[rec, -1])
            else:
                p_um[rec, i] = interp(z_rho[i], z_si[rec, :], ht[:])

    p_um[:, -1] = 100.0

    return p_um


def interp_ht(dat, z_old, z_new):

    return_dat = np.empty((0, len(z_new)))

    for d, z in zip(dat, z_old):
        if z[0] > z[-1]:
            n = np.interp(z_new, z[::-1], d[::-1])
        else:
            n = np.interp(z_new, z, d)
        return_dat = np.concatenate((return_dat, n[np.newaxis, :]))

    return return_dat


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

    def radian(angle):
        return angle * np.pi / 180.

    return surface_distance_y(lon1, lon2) * np.cos(radian(lat))


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
            inobsfor['t_inc'] = (
                out_data['gradt'].flatten(order='F').tolist()
            )
        if conf.q_star:
            inobsfor['q_star'] = (
                out_data['gradq'].flatten(order='F').tolist()
            )

    if conf.ui:
        inprof['ui'] = out_data['u'][0, :].flatten(order='F').tolist()
    if conf.vi:
        inprof['vi'] = out_data['v'][0, :].flatten(order='F').tolist()
    if conf.wi:
        inprof['wi'] = 'not implemented yet'
    if conf.theta:
        inprof['theta'] = out_data['pt'][0, :].flatten(order='F').tolist()
    if conf.qi:
        inprof['qi'] = out_data['q'][0, :].flatten(order='F').tolist()
    if conf.p_in:
        inprof['p_in'] = out_data['p'][0, :-1].flatten(order='F').tolist()

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
        for i in range(out_vars['pt'].shape[1]):
            rho_lev = i < out_vars['u'].shape[1]
            w_dict = {
                'p_um': out_vars['p'][-1, i],
                'adum': levs[i] if i < len(levs) else 0.0,
                't_um': out_vars['t'][-1, i],
                'pt_um': out_vars['pt'][-1, i],
                'q_um': out_vars['q'][-1, i],
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
            'levs': allvars['msl'][0],
            't': allvars['t'][0, 0],
            'pt': allvars['pt'][0, 0],
            'q': allvars['q'][0, 0],
            'u': allvars['u'][0, 0],
            'v': allvars['v'][0, 0],
            'ug': allvars['ug'][0, 0],
            'vg': allvars['vg'][0, 0]
        }
        genesis.write(format_data.format(**w_dict))
        for i in range(allvars['u'].shape[1]):
            w_dict = {
                'z': allvars['z'][0, i],
                'levs': levs[i],
                't': allvars['t'][0, i],
                'pt': allvars['pt'][0, i],
                'q': allvars['q'][0, i],
                'u': allvars['u'][0, i],
                'v': allvars['v'][0, i],
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

    logger = h.logger(conf.debug)

    # Read the data from the ERA-Interim NetCDF files
    data_in = {}
    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        data_in[var] = era_dataset(var)
        data_in[var].read_ht_array()

        data_in[var].read_lat_array()
        data_in[var].select_lats_near(conf.lat)

        data_in[var].read_lon_array()
        data_in[var].select_lons_near(conf.lon)

        data_in[var].select_time_array(
            start=conf.start_date, end=conf.end_date
        )

        data_in[var].read_data()

        # Make sure that the pressure variables (ht, MSL) are in units of Pa, not hPa
        data_in[var].ensure_Pa()

        # Make sure that the vertical dimension is in ascending order (decending
        # values of pressure)
        data_in[var].ensure_ascending()

        logger.log('var {} read. Shape is {}'.format(var, data_in[var].data.shape))

    logger.log('latitudes: {}'.format(data_in['Z'].lat_array))
    logger.log('longitudes: {}'.format(data_in['Z'].lon_array))

    # Convert the Geopotential to height in metres
    data_in['Z'].convert_geop_to_m()
    logger.log('Z converted to metres')

    # Calculate the surface distances dx and dy
    dx = surface_distance_x(*data_in['Z'].lon_array.tolist(), lat=conf.lat)
    dy = surface_distance_y(*data_in['Z'].lat_array)
    logger.log('dx = {}'.format(dx))
    logger.log('dy = {}'.format(dy))

    # Interpolate the data along its longitude
    data_xi = {}
    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        data_xi[var] = data_in[var].interp_lon(conf.lon)
        logger.log('{} interpolated all -> xi, shape = {}'.format(
            var, data_xi[var].data.shape))

    # To calculate its gradients, we need the latitude-only interpolated data of
    # the temperature and the specific humidity.
    data_yi = {}
    for var in ['T', 'Q']:
        data_yi[var] = data_in[var].interp_lat(conf.lat)
        logger.log('{} interpolated all -> yi, shape = {}'.format(
            var, data_yi[var].data.shape))

    # Interpolate the data along the latitude as well so that we now have the
    # values for the one location we are looking at.
    data_si = {}
    for var in ['U', 'V', 'T', 'Z', 'Q', 'P']:
        data_si[var] = data_xi[var].interp_lat(conf.lat)
        logger.log('{} interpolated xi -> si, shape = {}'.format(
            var, data_si[var].data.shape))

    t_xi = data_xi['T'].data
    t_yi = data_yi['T'].data
    q_xi = data_xi['Q'].data
    q_yi = data_yi['Q'].data

    u_si = data_si['U'].data[:, :, 0, 0]
    v_si = data_si['V'].data[:, :, 0, 0]
    t_si = data_si['T'].data[:, :, 0, 0]
    z_si = data_si['Z'].data[:, :, 0, 0]
    q_si = data_si['Q'].data[:, :, 0, 0]
    msl_si = data_si['P'].data[:, 0, 0, 0]

    # 4 Height arrays:
    #  ht is the height dimension of the netCDF vars, and is in pressure
    #  z_theta and z_rho come from the base.inp namelist and are in metres
    #  z_si is actually the data from one file, and tells how high each pressure
    #       level is in metres for each timestep
    ht = data_in['Z'].ht_array
    z_theta = np.array(conf.eta_theta) * conf.z_top_of_model + conf.z_terrain_asl
    z_rho = np.array(conf.eta_rho) * conf.z_top_of_model + conf.z_terrain_asl
    logger.log('ht: {}'.format(ht))
    logger.log('z_theta: {}'.format(z_theta))
    logger.log('z_rho: {}'.format(z_rho))

    # Potential temperature is calculated from the actual temperature and the
    # pressure level.
    pt_si = calc_pt(t_si, ht)
    logger.log('calculated pt_si: {}'.format(pt_si[0, :]))

    # This is a bit dodgy: I think that the original genesis code calculated the
    # temperature and humidity gradients wrongly. For example, gradtx, which is
    # eventually multiplied with u, I expect should contain the temperature
    # gradient in x-direction, so east-west. And it is devided by dx, the
    # distance between east and west. But the difference is taken between the
    # north and the south values.
    # So set the next value to False to use what I think is the right
    # calculation, or to true to behave in the same way as the original genesis
    # code.
    use_genesis_calculation_for_gradients = True

    if use_genesis_calculation_for_gradients:
        gradtx_si = (t_xi[:, :, 0, 0] - t_xi[:, :, 1, 0]) / dx
        gradty_si = (t_yi[:, :, 0, 1] - t_yi[:, :, 0, 0]) / dy
        gradqx_si = (q_xi[:, :, 0, 0] - q_xi[:, :, 1, 0]) / dx
        gradqy_si = (q_yi[:, :, 0, 1] - q_yi[:, :, 0, 0]) / dy
    else:
        gradtx_si = (t_yi[:, :, 0, 1] - t_yi[:, :, 0, 0]) / dx
        gradty_si = (t_xi[:, :, 0, 0] - t_xi[:, :, 1, 0]) / dy
        gradqx_si = (q_yi[:, :, 0, 1] - q_yi[:, :, 0, 0]) / dx
        gradqy_si = (q_xi[:, :, 0, 0] - q_xi[:, :, 1, 0]) / dy

    logger.log('calculated gradtx_si: {}'.format(gradtx_si[0, :]))
    logger.log('calculated gradty_si: {}'.format(gradty_si[0, :]))
    logger.log('calculated gradqx_si: {}'.format(gradqx_si[0, :]))
    logger.log('calculated gradqy_si: {}'.format(gradqy_si[0, :]))

    gradt_si = -(u_si * gradtx_si + v_si * gradty_si)
    gradq_si = -(u_si * gradqx_si + v_si * gradqy_si)

    # Change units for gradients from 'per second' to 'per day'
    gradt_si *= 24 * 60 * 60
    gradq_si *= 24 * 60 * 60

    logger.log('calculated gradt_si: {}'.format(gradt_si[0, :]))
    logger.log('calculated gradq_si: {}'.format(gradq_si[0, :]))

    ug = -calc_geostrophic_winds(
        data_in['Z'].data[:, :, :, 1], dy, conf.lon
        )
    vg = calc_geostrophic_winds(
        data_in['Z'].data[:, :, 0, :], dx, conf.lat
        )
    allvars_si = {
        'u': u_si,
        'v': v_si,
        't': t_si,
        'q': q_si,
        'z': z_si,
        'pt': pt_si,
        'msl': msl_si,
        'ug': ug,
        'vg': vg
    }
    write_genesis(allvars_si, ht)
    logger.log('written genesis.csv, compare to genesis.scm of original genesis program')

    # Make a vertical interpolation to get the levels that the UM will finally
    # need.
    u_um = interp_ht(u_si, z_si, z_rho)
    v_um = interp_ht(v_si, z_si, z_rho)
    q_um = interp_ht(q_si, z_si, z_theta)
    pt_um = interp_ht(pt_si, z_si, z_theta)
    t_um = interp_ht(t_si, z_si, z_theta)
    p_um = calc_p_um(ht, z_si, z_rho, msl_si)
    gradt_um = interp_ht(gradt_si, z_si, z_theta)
    gradq_um = interp_ht(gradq_si, z_si, z_theta)

    logger.log('u_um calculated, shape={}'.format(u_um.shape))
    logger.log('v_um calculated, shape={}'.format(v_um.shape))
    logger.log('t_um calculated, shape={}'.format(t_um.shape))
    logger.log('pt_um calculated, shape={}'.format(pt_um.shape))
    logger.log('q_um calculated, shape={}'.format(q_um.shape))
    logger.log('p_um calculated, shape={}'.format(p_um.shape))
    logger.log('gradt_um calculated, shape={}'.format(gradt_um.shape))
    logger.log('gradq_um calculated, shape={}'.format(gradq_um.shape))

    out_data = {
        'u': u_um,
        'v': v_um,
        't': t_um,
        'pt': pt_um,
        'q': q_um,
        'p': p_um,
        'gradt': gradt_um,
        'gradq': gradq_um
    }

    write_charney(out_data, ht)
    logger.log('written charney.csv, compare to charney.scm of original genesis program')

    # Read the template for the namelist.
    template = f90nml.read(conf.template)

    # Update the values that we want to change.
    out_namelist = replace_namelist(template, out_data, conf)

    # Write the output.
    out_namelist.write(conf.output, force=True)


if __name__ == '__main__':
    main()
