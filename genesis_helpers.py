#!/usr/bin/env python

# These are helper files to analyse the data. They do not contain any references
# to either the netcdf nor the namelists.

import numpy as np
import datetime


class logger(object):

    active = False

    def __init__(self, active):
        self.active = active

    def set_active(self):
        self.active = True

    def set_deactive(self):
        self.active = False

    def log(self, value):
        if self.active:
            print(value)


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
        assert(args.surface_type.lower() in ['auto', 'land', 'sea', 'coast'],
               "Surface type must be 'auto', 'land', 'sea', or 'coast'")
        self.surface = args.surface_type.lower()
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
