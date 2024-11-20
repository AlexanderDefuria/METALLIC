#!/bin/bash
#SBATCH --time=3-00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-user=j.gaudreault@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0,54,69
CPUS=32

module load StdEnv/2023 python/3.11
module load scipy-stack
module load boost
module load rust

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install /home/jgaud/environment/SQLAlchemy-1.4.1-cp311-cp311-linux_x86_64.whl --no-index
pip install --no-index -r /home/jgaud/environment/requirements.txt
pip install /home/jgaud/environment/clusopt_core-1.0.0-cp311-cp311-linux_x86_64.whl --no-index

mkdir $SLURM_TMPDIR/$SLURM_ARRAY_TASK_ID
aim init --repo $SLURM_TMPDIR/$SLURM_ARRAY_TASK_ID
python run_experiments.py $CPUS $SLURM_ARRAY_TASK_ID $SLURM_TMPDIR/$SLURM_ARRAY_TASK_ID

ID_JOB="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}"
zip -r $SLURM_TMPDIR/$ID_JOB.zip $SLURM_TMPDIR/$SLURM_ARRAY_TASK_ID
cp $SLURM_TMPDIR/$ID_JOB.zip /home/jgaud/scratch/