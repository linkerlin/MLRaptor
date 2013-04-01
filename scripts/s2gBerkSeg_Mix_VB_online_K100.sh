#!/bin/bash

COMMAND="../LearnExpFam.py BerkSegPatchData MixModel Gaussian VB --jobname K100fast --initname randsample --printEvery 5 --saveEvery 10 --K 100 --alpha0 0.5"
COMMAND=$COMMAND" --doonline --batch_size 9401 --nBatch 50 --nRep 20"
echo $COMMAND

#qsub -t 1-2 -l test $COMMAND
qsub -t 1-10 -l long $COMMAND

echo " "
exit

##############################################################################
K100
Your job-array 2857555.1-10:1 ("LearnExpFam.py") has been submitted


