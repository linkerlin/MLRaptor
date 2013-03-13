import numpy as np

def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
class GMMPrior(object):
 
  #meanP = dict()
  #covarP = dict()
 
  def __init__(self, degFree, invW, muPrec, muMean=0.0):
   #self.meanP['prec'] = muPrec
   #self.meanP['mean'] = 0
   #self.covarP['degFree'] = degFree
   #self.covarP['W'] = W
   self.beta = muPrec
   self.invW = invW
   self.dF   = degFree
   self.D = invW.shape[0]
   muMean =np.asarray( muMean)
   if muMean.size == self.D:
     self.m    = muMean
   elif muMean.size ==1:
     self.m    = np.tile( muMean, (self.D) )

  def __str__(self): 
    return '%s %s %s %s' % (np2flatstr( self.beta ),  np2flatstr(self.m), np2flatstr(self.dF), np2flatstr(self.invW)  )
    #print 'Prior on mu: Normal with \n   mean %s\n covar %s' % (self.meanP['mean'], self.meanP['prec'] )
    #print 'Prior on prec. matrix: Wishart with\n %d deg freedom\n mean = %s' % (self.covarP['degFree'], self.covarP['W'] )
    
  def getMean( self ):
    mu = self.m
    cov = self.invW / ( self.dF - self.D - 1 )
    return mu,cov  
    
  def getMAP( self ):
    assert self.dF > self.D+1
    muMAP = self.m
    covMAP = self.invW / (self.dF + self.D + 1 )
    return muMAP, covMAP
    
  def getPosteriorParams( self, N, mean, covar ):
    beta = self.beta+N
    m = ( self.beta*self.m + N*mean ) / beta
    mdiff = mean - self.m 
    invW  = self.invW + N*covar  \
            + (self.beta*N)/(self.beta+N)*np.outer(mdiff,mdiff)

    return GMMPrior( self.dF+N, invW,  beta, m ) 
