#!/bin/bash

#SBATCH --job-name=BNC_TST
#SBATCH --account=project_465000157
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=pilot
#SBATCH --output=report/%j.out
#SBATCH --signal=B:TERM


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1
module load rocm/5.0.2

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

export PYTHONUSERBASE='/projappl/project_465000157/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

export WANDB_MODE=offline

#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export LOGLEVEL=INFO

trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

srun python3 train.py --batch_size 256 --max_steps 31250 "$@" &
# srun python3 train.py --batch_size 256 "$@" &
# torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_NTASKS_PER_NODE --max_restarts=3 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --master_addr ${MASTER_ADDR} --master_port $MASTER_PORT train.py --batch_size 64 "$@" &

PID="$!"
wait "${PID}"

# torchrun --standalone --nnodes=1 --nproc_per_node=${N_GPUS} train.py "$@" &
