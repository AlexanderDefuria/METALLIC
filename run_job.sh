#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-pbranco
#SBATCH --mail-user=adefu020@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=40
#SBATCH --array=1-10

HOME_DIR=/home/adefu020/projects/def-pbranco/adefu020/METALLIC
module load StdEnv/2023 python/3.10
module load scipy-stack

python3 -m venv $SLURM_TMPDIR/venv 
source $SLURM_TMPDIR/venv/bin/activate
pip3 -q install --no-index --upgrade pip
pip3 -q install --no-index -r $HOME_DIR/requirements.txt

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
    SLURM_ARRAY_TASK_COUNT=1
    SLURM_ARRAY_TASK_ID=1
    cp -r $HOME_DIR/data $SLURM_TMPDIR/
    echo $SLURM_TMPDIR
else
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -r $HOME_DIR/data  $SLURM_TMPDIR
    echo $SLURM_TMPDIR
fi
echo $SLURM_ARRAY_TASK_ID

#if [ -f $HOME_DIR/metafeatures_*.csv ]; 
#then
#    python3 $HOME_DIR/merge.py
#fi

echo "STARTING"
python3 $HOME_DIR/src/create_metafeatures.py --quick --debug --cpu=$SLURM_CPUS_PER_TASK --slurmid=$SLURM_ARRAY_TASK_ID --slurmcount=$SLURM_ARRAY_TASK_COUNT --datadir=$SLURM_TMPDIR/data 



echo "END JOB SCRIPT"
