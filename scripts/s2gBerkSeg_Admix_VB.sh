#!/bin/bash

COMMAND="../LearnExpFam.py BerkSegPatchData AdmixModel Gaussian VB --jobname mikefast --initname randsample --printEvery 1 --saveEvery 10 --K 25"
COMMAND=$COMMAND" --nIter 100"
echo $COMMAND

#qsub -t 1-2 -l test $COMMAND
qsub -t 1-10 -l long $COMMAND

echo " "
exit

##############################################################################
Your job-array 2856309.1-10:1 ("LearnExpFam.py") has been submitted
