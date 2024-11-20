#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=def-pbranco
#SBATCH --mail-user=adefu020@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=10
#SBATCH --array=1-2

HOME_DIR=/home/adefu020/projects/def-pbranco/adefu020/METALLIC
module load StdEnv/2023 python/3.10

python3 -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip3 -q install --no-index --upgrade pip
pip3 -q install --no-index -r $HOME_DIR/requirements.txt
# cp $HOME_D

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

python3 test.py


