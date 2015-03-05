#!/usr/bin/env python

import f90nml
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--old-genesis', metavar='FILE', default='namelist.scm')
    parser.add_argument('-n', '--new-genesis', metavar='FILE', default='delme.scm')

    args = parser.parse_args()
    return args


def calc_diff(var1, var2):

    l = min(len(var1), len(var2))

    v1 = np.array(var1[:l])
    v2 = np.array(var2[:l])

    diff = np.zeros_like(v1)

    if np.any(v1+v2 == 0.0):
        diff = 2 * (v1 - v2) / np.mean(v1 + v2)
    else:
        diff = 2 * (v1 - v2) / (v1 + v2)


    return np.mean(abs(diff)), np.max(diff), np.min(diff)



def main():

    args = get_args()

    old = f90nml.read(args.old_genesis)
    new = f90nml.read(args.new_genesis)

    print('{:>7}| {:>8} {:>8} {:>8}'.format('var', 'mean', 'max', 'min'))
    for var in ['ui', 'vi', 'theta', 'qi', 'p_in']:
        print('{:>7}: {:8.2%} {:8.2%} {:8.2%}'.format(var, *calc_diff(old['inprof'][var], new['inprof'][var])))
    for var in ['u_inc', 'v_inc']:
        print('{:>7}: {:8.2%} {:8.2%} {:8.2%}'.format(var, *calc_diff(old['inobsfor'][var], new['inobsfor'][var])))



if __name__ == '__main__':
    main()