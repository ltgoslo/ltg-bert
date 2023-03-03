#!/bin/bash

#SBATCH --job-name=PRE-SHARD
#SBATCH --output=preprocess_shard.out
#SBATCH --account=project_465000157
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=eap

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

echo "" > preprocess_shard.out

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

SOURCE_FOLDER=$1
TARGET_FOLDER=$2
N_TRAIN_SHARDS=$3
N_VALID_SHARDS=$4

rm -rf "${TARGET_FOLDER}"
mkdir -p "${TARGET_FOLDER}"
python3 shard.py --input_path "${SOURCE_FOLDER}" --output_path "${TARGET_FOLDER}" --n_train_shards "${N_TRAIN_SHARDS}" --n_valid_shards "${N_VALID_SHARDS}" || exit 1

exit 0
