class QM( object ):
  def __init__( self, allocModel, obsModel):
    self.aModel = allocModel
    self.obsModel   = obsModel
    
    self.qObs = [None for k in xrange( self.aModel.K)]
    
  def get_local_params( self, Data, LP=None):
    LP = self.aModel.get_local_params( Data, LP)
    
  def update_global_params( self, Data, LP):
    SS = dict()
    SS = self.aModel.get_global_suff_stats(Data, LP, SS)
    SS = self.obsModel.get_global_suff_stats(Data, LP, SS)
