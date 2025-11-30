#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --gpus-per-node="1"
#SBATCH --mem="64G"
#SBATCH --time="01:00:00"
#SBATCH --account="llonpp"
#SBATCH --job-name="finger_o5_interpret"
#SBATCH --chdir="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/"
#SBATCH --error="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/logs/%x.e%A_%a"
#SBATCH --output="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/logs/%x.o%A_%a"
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be"

source /vsc-hard-mounts/leuven-data/355/vsc35565/miniconda/etc/profile.d/conda.sh
conda activate decode

echo "==================== SLURM/Conda/Python ===================="
echo "Running on node: $(hostname)"
echo "Assigned GPUs (SLURM_JOB_GPUS): $SLURM_JOB_GPUS"
echo "Verification Python Path: $(which python)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================= nvidia-smi ========================="
nvidia-smi

export PYTHONHASHSEED=42
python regression_o5_nn_interpretModel.py
