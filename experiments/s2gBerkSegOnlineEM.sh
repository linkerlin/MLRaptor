#!/bin/bash

COMMAND="./RunGMMExperiment.py BerkSegPatchData oEM --jobname berk50 --batch_size=18000 --nBatch=50 --nRep=100 --initname kmeans --saveEvery=25 --printEvery=10 --K=50"

echo $COMMAND

qsub -t 1-10 -l long $COMMAND

echo " "
exit



