#!/bin/bash
# Usage:
#    s2gDenoiseBerkSeg modelName infName [taskID]


#./runDenoiseBerkSeg.sh $*

qsub -t 1-4 -l long runDenoiseBerkSeg.sh $*

echo " "
exit

##############################################################################

