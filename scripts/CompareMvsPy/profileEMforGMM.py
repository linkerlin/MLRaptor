import subprocess

# First, copy the standard python script to a new one that's properly decorated.
proffileName ='runEMforGMMPROFILED.py'
proffile = open( proffileName, 'w')
funcsToProfile = ['main', 'Estep', 'Mstep', 'loggausspdf', 'distMahal']

for line in open( 'runEMforGMM.py', 'r'):
  for funcName in funcsToProfile:
    if line.startswith( 'def %s' % (funcName) ):
      proffile.write('@profile\n')
  proffile.write( line)

proffile.close()

argString = 'BerkSegTrain25.mat 25 3 out2.mat 8675309'
CMD = 'python /Users/mhughes/git/MLRaptor/util/kernprof.py --line-by-line %s %s' % (proffileName, argString)
#status = subprocess.call( CMD.split(' ') )

txtfilename = 'pyprofile.txt'
CMD = 'python -m line_profiler %s' % (proffileName+'.lprof')
print CMD
status = subprocess.call( CMD.split(' '), stdout=open(txtfilename,'w') )


'''EM for Mixture of 25 Gaussians | seed=8675309
    1/3 after 45 sec | 2.53173615e+07
    2/3 after 89 sec | 2.95042530e+07
    3/3 after 134 sec | 3.36832938e+07
became
EM for Mixture of 25 Gaussians | seed=8675309
    1/3 after 33 sec | 2.53173615e+07
    2/3 after 65 sec | 2.95042530e+07
    3/3 after 97 sec | 3.36832938e+07
'''
