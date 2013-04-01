#!/bin/bash

COMMAND="../LearnExpFam.py BerkSegPatchData MixModel Gaussian VB --jobname K200fast --initname randsample --printEvery 5 --saveEvery 5 --K 200 --alpha0 0.2"
COMMAND=$COMMAND" --doonline --batch_size 9401 --nBatch 50 --nRep 20"
echo $COMMAND

#qsub -t 1-2 -l test $COMMAND
qsub -t 1-3 -l long $COMMAND

echo " "
exit

##############################################################################
K200
Your job-array 2857556.1-3:1 ("LearnExpFam.py") has been submitted


