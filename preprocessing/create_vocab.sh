#!/bin/bash

#SBATCH --job-name=PRE-VOCAB
#SBATCH --output=preprocess_vocab.out
#SBATCH --account=project_465000157
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=eap

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

echo "" > preprocess_vocab.out

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
TOKENIZER=$3

python3 create_vocab.py --input_path "${SOURCE_FILE}" --vocab_path "${VOCAB_FILE}" --tokenizer_type "${TOKENIZER}" --no-lowercase || exit 1

# Successful exit
exit 0
