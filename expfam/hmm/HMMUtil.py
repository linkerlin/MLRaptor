'''
  HMMUtil.py
  Provides standard algorithms for inference in HMMs,
     like the forward-backward algorithm.
     
  Intentionally separated from rest of HMM code, so that we can swap in 
     any fast routine for this calculation with ease.
'''
#from ..util.MLUtil import logsoftev2softev
import numpy as np

def FwdBwdAlg( PiInit, PiMat, logSoftEv ):
  '''
     Returns
     -------
       resp : TxK matrix, where
               resp[t,k] = marginal probability that step t assigned to state K
                           p( z_t = k | x_1, x_2, ... x_T )
       respPair : TxKxK matrix,
               respPair[t,j,k] = marg. probability that step t -> j, t+1 -> k
                           p( z_t-1 =j, z_t = k | x_1, x_2, ... x_T )
  '''
  SoftEv, lognormC = logsoftev2softev( logSoftEv )
  
  fmsg, scaleC = FwdAlg( PiInit, PiMat, SoftEv )
  bmsg = BwdAlg( PiInit, PiMat, SoftEv, scaleC )
  resp = fmsg * bmsg
  
  T,K = resp.shape
  respPair = np.zeros( (T,K,K) )
  for t in xrange( 1, T ):
    vecprev = fmsg[t-1]
    veccur  = bmsg[t] * SoftEv[t]
    respPair[t] = PiMat * np.outer(vecprev,veccur) / scaleC[t]
    assert np.allclose( respPair[t].sum(), 1.0 )
  margLogPr = np.log( scaleC ).sum() + lognormC.sum()
  return resp, respPair, margLogPr  
    
def FwdAlg( PiInit, PiMat, SoftEv ):
  '''
     Returns
     -------
        fmsg : T x K matrix
                  fmsg[t,k] = p( z_t = k | x_1, x_2, ... x_t )
        logp : scalar log probability of entire sequence
                  logp = p( x_1, x_2, ... x_T | trans params Pi, emit params Phi )
  '''
  T = SoftEv.shape[0]
  K = PiInit.size
  PiMat = PiMat.T.copy()
  scaleC = np.zeros( T )
  fmsg = np.empty( (T,K) )
  for t in xrange( 0, T ):
    if t == 0:
      fmsg[t] = PiInit * SoftEv[0]
    else:
      fmsg[t] = np.dot(PiMat,fmsg[t-1]) * SoftEv[t]
    scaleC[t] = np.sum( fmsg[t] )
    fmsg[t] /= scaleC[t]
  return fmsg, scaleC
  
def BwdAlg( PiInit, PiMat, SoftEv, scaleC ):
  '''
     Returns
     -------
        bmsg : T x K matrix
                  bmsg[t,k] = p( z_t = k , x_t+1, x_t+2, ... x_T )
      pwmsg  : T x K x K
                pwmsg[t][j,k] = p( z_t=j, z_t+1=k|x_t+1,...)
  '''
  T = SoftEv.shape[0]
  K = PiInit.size
  
  bmsg = np.ones( (T,K) )
  pwmsg = np.zeros( (T,K) )
  for t in xrange( T-2, -1, -1 ):
    bmsg[t] = np.dot(PiMat, bmsg[t+1] * SoftEv[t+1] )
    bmsg[t] /= scaleC[t+1]
  return bmsg
