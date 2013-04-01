'''
   Explore all the python functions in the user-specified directory
       and remove decoration @profile from appropriate functions
'''

import os

path = '../expfam/'

list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename[-3:] == '.py': 
            list_of_files[filename] = os.sep.join([dirpath, filename])

for key in list_of_files.keys():
  origfname = list_of_files[key]
  proffname = list_of_files[key]+'CLEAN'
  proffile = open( proffname, 'w')
  with open( origfname, 'r') as f:
    for line in f.readlines():
      if line.count( '@profile' ) == 0:
        proffile.write( line )
  proffile.close()
  os.rename( proffname, origfname)

