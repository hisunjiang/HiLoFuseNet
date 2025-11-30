#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --gpus-per-node="1"
#SBATCH --mem="64G"
#SBATCH --time="02:00:00"
#SBATCH --account="llonpp"
#SBATCH --job-name="finger_o1"
#SBATCH --chdir="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/"
#SBATCH --error="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/logs/%x.e%A_%a"
#SBATCH --output="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/logs/%x.o%A_%a"
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be"
#SBATCH --array=0-59

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

python - <<PYCODE
import itertools
import subprocess
import os
import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA is available: {cuda_available}")

datasets = ['BCIIV', 'Stanford'] # 'BCIIV', 'Stanford'
decoders = ['LSTM', 'CNN_LSTM', 'HiLoFuseNet'] # 'LSTM', 'CNN_LSTM', 'HiLoFuseNet'
lossFuncs = ['mse'] # 'SCloss', 'SCloss', 
seeds = [42,43,44,45, 46,47,48, 49,50,51]

combinations = list(itertools.product(datasets, decoders, lossFuncs, seeds))

task_id = int("${SLURM_ARRAY_TASK_ID}")
dataset, decoder, lossFunc, seed = combinations[task_id]

print(f"Running combination: dataset={dataset}, decoder={decoder}, lossFunc={lossFunc}, seed={seed}")

subprocess.run([
    "python", "regression_o1_nn.py",
    "--dataset", dataset,
    "--decoder", decoder,
    "--lossFunc", lossFunc,
    "--seed", str(seed)
])
PYCODE
