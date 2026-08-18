#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=5:00:00
#SBATCH --account=llonpp
#SBATCH --job-name="finger_o5_featureSwap"
#SBATCH --chdir="/vsc-hard-mounts/leuven-user/355/vsc35565/HiLoFuseNet/"
#SBATCH --error="/vsc-hard-mounts/leuven-user/355/vsc35565/HiLoFuseNet/logs/%x.e%A_%a"
#SBATCH --output="/vsc-hard-mounts/leuven-user/355/vsc35565/HiLoFuseNet/logs/%x.o%A_%a"
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be"
#SBATCH --array=0-3

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
import time
import numpy as np

datasets = ['BCIIV', 'Stanford'] # 'BCIIV', 'Stanford'
decoders = ['MLP', 'LSTM'] # 'MLP', 'LSTM', 'EEGNet', 'DeepFingerNet', 'CNN_LSTM', 'WaT', 'WaTEi','HiLoFuseNet'
lossFuncs = ['mse'] # 'mse', 
seeds = [42] #[42,43,44,45,46]
win_sizes = [2.8] # np.arange(0.4, 3, 0.2) -> np.arange(0.4, 2, 0.2) , np.arange(2, 3, 0.2)

combinations = []
for ds_name in datasets:
    for dec in decoders:
        for ls in lossFuncs:
            for sd in seeds:
                for ws in win_sizes:
                    combinations.append((ds_name, dec, ls, sd, ws))

# get current task ID
task_id = int("${SLURM_ARRAY_TASK_ID}")
if task_id >= len(combinations):
    print(f"Task ID {task_id} out of range (max {len(combinations)-1})")
    exit(1)

dataset, decoder, lossFunc, seed, win_size = combinations[task_id]

print(f"--- Task {task_id} ---")
print(f"Dataset: {dataset} | Decoder: {decoder} | LossFunc: {lossFunc} | Seed: {seed} | Win: {win_size}")

# run jobs
start_time  = time.time()
subprocess.run([
    "python", "regression_o5_nn_featureSwap.py",
    "--dataset", dataset,
    "--decoder", decoder,
    "--lossFunc", lossFunc,
    "--seed", str(seed),
    "--win_size", str(win_size)
])

end_time = time.time()
duration = (end_time - start_time) / 60
print(f"==================================================")
print(f"Task {task_id} completed in: {duration:.2f} minutes")
print(f"==================================================")
PYCODE
