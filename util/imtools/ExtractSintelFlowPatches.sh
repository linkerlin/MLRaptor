#! /bin/bash
# Usage:
#    ExtractSintelFlowPatches.sh alley_4

DS=/data/liv/visiondatasets/sintel/
mkdir $DS/patches/$1
python FlowPatchExtractor.py $DS/training/flow/$1/ $DS/patches/$1/
