#!/bin/bash

#SBATCH --job-name=PRE-INPUT
#SBATCH --output=preprocess_input.out
#SBATCH --account=project_465000157
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=small

# Load modules
module --quiet purge
module --quiet --force purge
module load LUMI/22.06
module load cray-python/3.9.4.2
module load rocm/5.1.4

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
#export PS1=\$

# activate the virtual environment
source /project/project_465000157/pytorch_1.12.1/bin/activate

SOURCE_FILE=$1
VOCAB_FILE=$2
OUTPUT_FILE=$3

python3 tokenize_input.py --input_path "${SOURCE_FILE}" --vocab_path "${VOCAB_FILE}" --output_path "${OUTPUT_FILE}" || exit 1

# Successful exit
exit 0
