#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --gpus-per-node="1"
#SBATCH --mem="64G"
#SBATCH --time="02:00:00"
#SBATCH --account="xxxx" 						# <- change here
#SBATCH --job-name="finger_o5"
#SBATCH --chdir="/project/" 					# <- change here
#SBATCH --error="/project/logs/%x.e%A_%a" 		# <- change here
#SBATCH --output="/project/logs/%x.o%A_%a" 		# <- change here
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be" 	# <- change here
#SBATCH --array=0-59

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

datasets = ['BCIIV', 'Stanford'] # 'BCIIV', 'Stanford'
decoders = ['LSTM', 'CNN_LSTM', 'HiLoFuseNet'] # 'LSTM', 'CNN_LSTM', 'HiLoFuseNet'
lossFuncs = ['mse'] # 'SCloss', 'SCloss', 
seeds = [42,43,44,45, 46,47,48, 49,50,51]

combinations = list(itertools.product(datasets, decoders, lossFuncs, seeds))

task_id = int("${SLURM_ARRAY_TASK_ID}")
dataset, decoder, lossFunc, seed = combinations[task_id]

print(f"Running combination: dataset={dataset}, decoder={decoder}, lossFunc={lossFunc}, seed={seed}")

subprocess.run([
    "python", "regression_o5_nn.py",
    "--dataset", dataset,
    "--decoder", decoder,
    "--lossFunc", lossFunc,
    "--seed", str(seed)
])
PYCODE
