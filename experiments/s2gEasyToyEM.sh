#!/bin/bash

COMMAND="./RunGMMExperiment.py EasyToyGMMData EM --jobname mikedemo --nIter 100 --initname kmeans --printEvery 1 --K 10"

echo $COMMAND

qsub -t 1-2 -l test $COMMAND

echo " "
exit



