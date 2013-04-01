'''
   Remove blank function records from the overall report
'''

import os

origfname  = '../profiles/pyprofile.txt'

with open( origfname,'r') as f:
  Records = list()
  line = f.readline()
  CurRecord = ''
  doStart = False
  while line:    
    if line.startswith('File:'):
      if doStart:
        Records.append( CurRecord )
      else:
        Preamble = CurRecord
      doStart = True
      CurRecord = ''
    CurRecord += line
    line = f.readline()
  # Add final record
  Records.append( CurRecord )

with open( origfname, 'w') as f:
  f.write( Preamble)
  for Record in Records:
    lines = Record.split('\n')
    totalTime = lines[2]
    if totalTime.count( '0 s' ) > 0:
      continue
    for line in lines:
      f.write( line+'\n' )
  
