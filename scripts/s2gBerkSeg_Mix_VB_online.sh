#!/bin/bash

COMMAND="../LearnExpFam.py BerkSegPatchData MixModel Gaussian VB --jobname mikefast --initname randsample --printEvery 5 --saveEvery 10 --K 25"
COMMAND=$COMMAND" --doonline --batch_size 9400 --nBatch 50 --nRep 20"
echo $COMMAND

#qsub -t 1-2 -l test $COMMAND
qsub -t 1-10 -l long $COMMAND

echo " "
exit

##############################################################################


