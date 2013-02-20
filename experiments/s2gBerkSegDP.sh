#!/bin/bash

COMMAND="./RunGMMExperiment.py BerkSegPatchData DPVB --jobname berk50 --nIter 100 --initname kmeans --K 50 --saveEvery=5 --printEvery=1"

echo $COMMAND

qsub -t 1-10 -l long $COMMAND

echo " "
exit



