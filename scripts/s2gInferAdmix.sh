export PYTHONPATH=/home/mhughes/git/MLRaptor/

echo "Warning: FileIO is stupid with python on the grid. So dont expect to watch things as they happen"

#qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/oVB/mikefast/2/ /data/liv/mhughes/img/BSDS300/patches/test/
#qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/oVB/mikefast/3/ /data/liv/mhughes/img/BSDS300/patches/test/
#qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/oVB/mikefast/4/ /data/liv/mhughes/img/BSDS300/patches/test/

qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/VB/mikefast/1/ /data/liv/mhughes/img/BSDS300/patches/test/
qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/VB/mikefast/2/ /data/liv/mhughes/img/BSDS300/patches/test/
qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/VB/mikefast/3/ /data/liv/mhughes/img/BSDS300/patches/test/
qsub -l short ./InferAdmixWeightsForHeldout.py ../results/BerkSeg/AdmixModel/VB/mikefast/4/ /data/liv/mhughes/img/BSDS300/patches/test/

exit
#################################################33
Admix oVB
Your job 2857535 ("InferAdmixWeightsForHeldout.py") has been submitted
Your job 2857536 ("InferAdmixWeightsForHeldout.py") has been submitted
Your job 2857537 ("InferAdmixWeightsForHeldout.py") has been submitted

Admix VB
Your job 2857538 ("InferAdmixWeightsForHeldout.py") has been submitted
Your job 2857539 ("InferAdmixWeightsForHeldout.py") has been submitted
Your job 2857540 ("InferAdmixWeightsForHeldout.py") has been submitted
Your job 2857541 ("InferAdmixWeightsForHeldout.py") has been submitted

Looks like everything works.  All VB and oVB runs have been run on the test data.
