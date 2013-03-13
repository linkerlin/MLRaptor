#!/bin/bash

COMMAND="../LearnExpFam.py EasyToyGMMData MixModel Gaussian EM --jobname demoeasy --nIter 100 --initname random --printEvery 1 --K 10"

echo $COMMAND

qsub -t 1-2 -l test $COMMAND

echo " "
exit
