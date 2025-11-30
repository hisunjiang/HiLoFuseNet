#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --gpus-per-node="1"
#SBATCH --mem="64G"
#SBATCH --time="02:00:00"
#SBATCH --account="xxxx" 						# <- change here
#SBATCH --job-name="finger_o5_hyperparameter"
#SBATCH --chdir="/project/" 					# <- change here
#SBATCH --error="/project/logs/%x.e%A_%a" 		# <- change here
#SBATCH --output="/project/logs/%x.o%A_%a" 		# <- change here
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be" 	# <- change here
#SBATCH --array=0-79

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
decoders = ['HiLoFuseNet']
CNN_Ds = [4,8,12,16,20,24,28,32,36,40]
LSTM_hiddensizes = [64,128,192,256]

combinations = list(itertools.product(datasets, decoders, CNN_Ds, LSTM_hiddensizes))

task_id = int("${SLURM_ARRAY_TASK_ID}")
dataset, decoder, CNN_D, LSTM_hiddensize = combinations[task_id]

subprocess.run([
    "python", "regression_o5_nn_hyperparameter.py",
    "--dataset", dataset,
    "--decoder", decoder,
    "--CNN_D", str(CNN_D),
    "--LSTM_hiddensize", str(LSTM_hiddensize)
])
PYCODE
