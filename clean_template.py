#!/usr/bin/env python


import argparse
try:
  import f90nml
except:
  print( """
  Failed to import f90nml module.
  module use ~access/modules
  module load pythonlib/f90nml
  """ )

def main():

  parser = argparse.ArgumentParser(description='Cleans up the template file')
  parser.add_argument('-i', '--input', metavar='FILE', default='template.scm', 
                      help='input file')
  parser.add_argument('-o', '--output', metavar='FILE', default='t_out.scm',
                      help='output file')

  args = parser.parse_args()

  nml=f90nml.read(args.input)
  nml.write(args.output, force = True)

if __name__ == '__main__':
  main()
