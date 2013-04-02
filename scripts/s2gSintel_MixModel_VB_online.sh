#!/bin/bash

# NOTES: batchsize 17544 = 4 total frames

COMMAND="../LearnExpFam.py SintelPatchData MixModel Gaussian VB --jobname K25fast --initname randsample --printEvery 5 --saveEvery 10 --K 25 --alpha0 0.5"
COMMAND=$COMMAND" --doonline --batch_size 17544 --nBatch 50 --nRep 20"
echo $COMMAND

qsub -t 1-4 -l long $COMMAND

echo " "
exit

##############################################################################
Your job-array 2857631.1-4:1 ("LearnExpFam.py") has been submitted
