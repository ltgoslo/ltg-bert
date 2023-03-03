#!/bin/bash

#SBATCH --job-name=NCC_CLEAN
#SBATCH --output=preprocess_extrinsic.out
#SBATCH --account=project_465000157
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=small

echo "" > preprocess_extrinsic.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1
# module load rocm/5.0.2

export PYTHONUSERBASE='/projappl/project_465000157/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

VOCAB_FILE=$1

# GLUE
GLUE_PATH="../data/extrinsic/glue_wiki"
mkdir -p "${GLUE_PATH}"
python3 tokenize_glue.py --output_path "${GLUE_PATH}"  --tokenizer_path "${VOCAB_FILE}" || exit 1
