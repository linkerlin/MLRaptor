import scipy.optimize
import numpy as np
import time


def v2beta( v ):
  v = np.asarray( v.flatten().copy() )
  v = np.hstack( [v, 1] )
  c1mv = np.cumprod( 1 - v )
  c1mv = np.hstack( [1, c1mv] )
  beta = v * c1mv[:-1]
  return beta

def np2flatstr( X ):
  bigstr = ' '.join( ['%.4f'% (x) for x in X] )
  if X.size < 10:
    return bigstr
  else:
    biglist = bigstr.split(' ')
    mystr = '\n'
    nRows = np.ceil(X.size / 10.0)
    mRange = np.arange( np.minimum(2,nRows) )
    if nRows-1 not in mRange:
      mRange = np.hstack( [mRange, nRows-1] )
    for m in mRange:
      myinds = m*10 + np.arange( 0, 10 )
      mystr += ' '.join( [biglist[ int(ii) ] for ii in myinds if ii < X.size] )
      mystr += '\n'
      if m==0: 
        remstr = ' '*len( mystr )
    return mystr+remstr


def get_short_msg( statusMsg ):
  if statusMsg.count( 'REL_REDUCTION' ) > 0:
    return 'diff(f) <= factr*epsmch'
  if statusMsg.count( 'ABNORMAL' ) >0:
    return 'abnormal termination.'
  return statusMsg

WARN_MSG = {0:'!!!', 1:'   '}
class FracVecOptimizer(object):
  def __init__(self, ProbDict, doBounds=True, LB=1e-6 ):
    self.objfunc = lambda x: ProbDict['objfunc']( x, *ProbDict['args'])
    self.objgrad = lambda x: ProbDict['objgrad']( x, *ProbDict['args'])
    self.xTRUE = ProbDict['vTrue']
    self.fTRUE = self.objfunc( self.xTRUE )
    self.K = self.xTRUE.size

    self.LB = LB
    self.UB = 1-LB
    if doBounds:
      self.Bounds = [(self.LB,self.UB) for k in xrange( self.K )]
    else:
      self.Bounds = None

    assert self.K == len(self.Bounds)

  def projectBounds( self, x):
    x = np.minimum(x,self.UB)
    x = np.maximum(x,self.LB)
    return x

  def run_optim( self, xinit, doWarnInit=True ):
    xinit = self.projectBounds( xinit )
    finit = self.objfunc(xinit)
    x,f,d = scipy.optimize.fmin_l_bfgs_b( self.objfunc, x0=xinit, fprime=self.objgrad, bounds=self.Bounds)
    statusMsg = get_short_msg( d['task'] )
    x,f,goodFlag, statusMsg = self.check_bounds( x, f, finit, statusMsg, doWarnInit)
    return x, f, goodFlag, statusMsg


  def check_bounds(self, x, f, finit, statusMsg, doWarnInit):
    try:
      assert np.all( x >= self.LB )
      assert np.all( x <= self.UB )
    except AssertionError:
      statusMsg += ' !OutOfBounds %.2f %.2f' % ( x.min(), x.max() )
      x = self.projectBounds(x)
      f = self.objfunc( x )

    if doWarnInit:
      if f >= finit or np.allclose(f, finit):
        statusMsg += ' !Init=Final'
    
    goodFlag = 1
    if f > self.fTRUE: goodFlag = 0
    return x, f, goodFlag, statusMsg

  def run_optim_from_true( self, doWarnInit=False):
    xinit = self.xTRUE
    return self.run_optim( xinit, doWarnInit )

  def run_optim_from_near_true(self, sig):
    xinit = sig*np.random.randn(self.K) + self.xTRUE
    return self.run_optim( xinit )

  def run_optim_from_rand(self):
    xinit = np.random.rand( self.K )
    return self.run_optim( xinit )

  def run_many_trials( self, Ntrial, initname):
    ngood = 0
    print '%s | %.4e%s | %s' % ( np2flatstr(v2beta(self.xTRUE)), self.fTRUE, WARN_MSG[1], 'Ground Truth' )
    starttime = time.time()
    for trial in xrange( Ntrial ):
      if initname == 'truth':
        x,f,sFlag, sMsg =self.run_optim_from_truth( )
      elif initname == 'near_truth':
        x,f,sFlag, sMsg =self.run_optim_from_near_truth()
      else:
        x,f,sFlag, sMsg =self.run_optim_from_rand()
      rep=0
      while rep < 20 and sum( x == self.LB ) > 0:
        x[x==self.LB] = np.random.rand( sum( x == self.LB ) )
        x,f,sFlag, sMsg =self.run_optim( x )
        rep+=1
      sMsg = '$'*rep + sMsg
      print '%s | %.4e%s | %s' % ( np2flatstr(v2beta(x)), f, WARN_MSG[sFlag], sMsg )
      ngood += sFlag
    elapsedTime = time.time()-starttime
    print 'Total: %d/%d successful. | %.2f sec/run' % (ngood, Ntrial, elapsedTime/Ntrial)
   

