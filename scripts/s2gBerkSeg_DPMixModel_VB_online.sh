#!/bin/bash

COMMAND="../LearnExpFam.py BerkSegPatchData DPMixModel Gaussian VB --jobname mikefast15 --initname randsample --printEvery 5 --saveEvery 10 --K 25 --alpha0 15.0"
COMMAND=$COMMAND" --doonline --batch_size 9401 --nBatch 50 --nRep 20"
echo $COMMAND

#qsub -t 1-2 -l test $COMMAND
qsub -t 1-5 -l long $COMMAND

echo " "
exit

##############################################################################
mikefast has alpha0 = 5
Your job-array 2857542.1-10:1 ("LearnExpFam.py") has been submitted

mikefast15  has alpha0=15
Your job-array 2857543.1-5:1 ("LearnExpFam.py") has been submitted

