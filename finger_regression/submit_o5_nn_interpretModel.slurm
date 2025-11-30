#!/bin/bash -l
#SBATCH --cluster wice
#SBATCH --partition gpu_a100
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --gpus-per-node="1"
#SBATCH --mem="64G"
#SBATCH --time="01:00:00"
#SBATCH --account="xxxx" 						# <- change here
#SBATCH --job-name="finger_o5_interpret"
#SBATCH --chdir="/project/" 					# <- change here
#SBATCH --error="/project/logs/%x.e%A_%a" 		# <- change here
#SBATCH --output="/project/logs/%x.o%A_%a" 		# <- change here
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be" 	# <- change here

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
python regression_o5_nn_interpretModel.py
