#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --gpus-per-node="1"
#SBATCH --mem="64G"
#SBATCH --time="00:40:00"
#SBATCH --account="xxxx" 						# <- change here
#SBATCH --job-name="finger_o5_hyperparameter"
#SBATCH --chdir="/project/" 					# <- change here
#SBATCH --error="/project/logs/%x.e%A_%a" 		# <- change here
#SBATCH --output="/project/logs/%x.o%A_%a" 		# <- change here
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be" 	# <- change here
#SBATCH --array=0-79							# <- change here

source /xxxx/miniconda/etc/profile.d/conda.sh 	# <- change here
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
CNN_Ds = [4,8,12,16,20,24,28,32,36,40]
LSTM_hiddensizes = [64,128,192,256]

combinations = []
for ds_name in datasets:
    for cnnd in CNN_Ds:
        for hid in LSTM_hiddensizes:
            combinations.append((ds_name, cnnd, hid))

# get current task ID
task_id = int("${SLURM_ARRAY_TASK_ID}")
if task_id >= len(combinations):
    print(f"Task ID {task_id} out of range (max {len(combinations)-1})")
    exit(1)

dataset, d, hid = combinations[task_id]

print(f"--- Task {task_id} ---")
print(f"Dataset: {dataset} | CNN_D: {d} | LSTM_hiddensize: {hid}")

# run jobs
start_time  = time.time()
subprocess.run([
    "python", "regression_o5_nn_hyperparameter.py",
    "--dataset", dataset,
    "--CNN_D", str(d),
    "--LSTM_hiddensize", str(hid)
])

end_time = time.time()
duration = (end_time - start_time) / 60
print(f"==================================================")
print(f"Task {task_id} completed in: {duration:.2f} minutes")
print(f"==================================================")
PYCODE
