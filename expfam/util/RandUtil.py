
def discrete_single_draw_fast( self, ps):
  ''' Given vector of K weights "ps",
         draw a single integer assignment in {1,2, ...K}
      such that Prob( choice=k) = ps[k]
  '''
  totals = np.cumsum(ps)
  norm = totals[-1]
  throw = random.random()*norm
  return np.searchsorted(totals, throw)
