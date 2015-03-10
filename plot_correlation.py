#!/usr/bin/env python

import f90nml
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--old-genesis', metavar='FILE', default='namelist-genesis.scm')
    parser.add_argument('-n', '--new-genesis', metavar='FILE', default='namelist.scm')
    parser.add_argument('-v', '--var', metavar='VAR', default='q_star')
    parser.add_argument('-g', '--group', metavar='VAR', default='inobsfor')

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    old = f90nml.read(args.old_genesis)
    new = f90nml.read(args.new_genesis)

    old_var = old[args.group][args.var]
    new_var = new[args.group][args.var]

    l = min(len(old_var), len(new_var))

    old_var = old_var[:l]
    new_var = new_var[:l]

    plt.scatter(old_var, new_var)
    plt.show()

if __name__ == '__main__':
    main()
