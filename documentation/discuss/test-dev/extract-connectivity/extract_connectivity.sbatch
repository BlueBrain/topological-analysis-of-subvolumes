#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --mem=0
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --job-name=test-extract-connectivity
#SBATCH --output=extract-connectivity.out
#SBATCH --error=extract-connectivity.err
source /gpfs/bbp.cscs.ch/home/sood/work/workspaces/venvs/load-py39.sh
python ./extract_connectivity.py "$@"
