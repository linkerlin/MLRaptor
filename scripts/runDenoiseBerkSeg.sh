#!/bin/bash
# Usage:
#    runDenoiseBerkSeg modelName infName
#$ -S /bin/sh 
#$ -cwd 
# ------ attach job number
#$ -j n
# put stdout and stderr files in the right place for your system.
#   NOTE that $TASK_ID is the correct var here
#          but not in rest of script (where SGE_TASK_ID is correct)
#$ -o ../logs/$JOB_ID.$TASK_ID.out
#$ -e ../logs/$JOB_ID.$TASK_ID.err

# ======================================================= SET JOB/TASK ID
if [[ ! $SGE_TASK_ID && ${SGE_TASK_ID-_} ]]
then
  # if not deployed on the grid, JOB_ID and SGE_TASK_ID are undefined
  #    so we manually set both to make sure this works
  JOB_ID=1
  SGE_TASK_ID=1
fi

EXPECTED_ARGS=3
if [ $# -lt $EXPECTED_ARGS ]  # use -lt instead of <
then
  taskID=$SGE_TASK_ID
else
  taskID=$3
fi

echo "JOB ID: " $JOB_ID
echo "SGE TASK ID: "$SGE_TASK_ID
echo "STORED TASK ID: "$taskID

MLABCMD="cd Denoise; DenoiseHeldoutData( 30, 'BerkSeg', '$1', '$2', 'mikefast', $taskID);"

/local/projects/matlab/R2011b/bin/matlab -nodesktop -nosplash -r "$MLABCMD; exit;"


