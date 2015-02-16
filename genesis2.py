#!/usr/bin/env python


import argparse
import numpy as np
try:
  import f90nml
except:
  print( """
  Failed to import f90nml module.
  module use ~access/modules
  module load pythonlib/f90nml
  """ )
try:
  import netCDF4 as cdf
except:
  print( """
  Failed to import netCDF4 module
  module use ~access/modules
  module load pythonlib/netCDF4
  """


def parse_arguments():
  """(None) -> args

  returns the results of parse_args()
  """

  parser = argparse.ArgumentParser(description='Cleans up the template file')
  parser.add_argument('-t', '--template', metavar='FILE', default='template.scm', 
                      help='Namelist Template')
  parser.add_argument('-o', '--output', metavar='FILE', default='t_out.scm',
                      help='Output Namelist')
  parser.add_argument('-r', '--relhum', default=False, action='store_true',
                      help='Convert Relative to Specific Humidity')
  parser.add_argument('-M', '--hPa', default=False, action='store_true',
                      help='Convert surface pressure from hPa to Pa')
  parser.add_argument('-D', '--date', default='dates.dat', metavar='FILE',
                      action='store_true', help='User Date file')
  parser.add_argument('-O', '--offset', metavar='FILE',
                      action='store_true', help='User Offset File')
  parser.add_argument('-d', '--debug', default=False, action='store_true',
                      help='Debug')


  return parser.parse_args()


def main():

  args = parse_arguments()


if __name__ == '__main__':
  main()
