'''
   Explore all the python functions in the user-specified directory
       and decorate the appropriate functions with @profile
'''

import os

TARGET_FUNCS = ['calc_local_params', 'get_global_suff_stats', 'update_global_params', \
                'update_obs_params_EM', 'update_obs_params_VB', 'log_pdf', 'dist_mahalanobis', \
                'getPosteriorDistr', 'rho_update', 'get_covar']

path = '../expfam/'

list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename[-3:] == '.py': 
            list_of_files[filename] = os.sep.join([dirpath, filename])

for key in list_of_files.keys():
  origfname = list_of_files[key]
  proffname = list_of_files[key]+'PROFILE'
  proffile = open( proffname, 'w')
  with open( origfname, 'r') as f:
    for line in f.readlines():
      sline = line.strip()
      if sline.startswith( 'def' ):
        funcname = sline.split( ' ' )[1]
        for targetname in TARGET_FUNCS:
          if funcname.startswith( targetname ):
            nspaces = len(line) -len( line.lstrip())
            proffile.write( ' '*nspaces +'@profile\n' )
            break
      proffile.write( line )
  proffile.close()
  # NOW, REPLACE .py files with their .pyPROF counterparts
  os.rename( proffname, origfname)

