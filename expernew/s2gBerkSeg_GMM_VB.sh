#!/bin/bash

COMMAND="../LearnExpFam.py BerkSegPatchData QMixModel Gaussian VB --jobname mikedemo --initname random --printEvery 1 --saveEvery 5 --K 25"
COMMAND=$COMMAND" --nIter 100"
echo $COMMAND

#qsub -t 1-2 -l test $COMMAND
qsub -t 1-10 -l long $COMMAND

echo " "
exit

##############################################################################
Your job-array 2706673.1-10:1 ("LearnExpFam.py") has been submitted

