#!/bin/bash -l
#SBATCH --cluster genius
#SBATCH --partition batch
#SBATCH --ntasks="1"
#SBATCH --mem="64G"
#SBATCH --time="24:00:00"
#SBATCH --account="llonpp"
#SBATCH --job-name="finger_o5_ml"
#SBATCH --chdir="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/"
#SBATCH --error="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/logs/%x.e%A"
#SBATCH --output="/vsc-hard-mounts/leuven-user/355/vsc35565/TNNLS2025/logs/%x.o%A"
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be"
#SBATCH --array=0-5

# export PATH="/vsc-hard-mounts/leuven-data/355/vsc35565/miniconda/bin:$PATH"
# source activate pytorch

source /vsc-hard-mounts/leuven-data/355/vsc35565/miniconda/etc/profile.d/conda.sh
conda activate decode

export PYTHONHASHSEED=42

python - <<PYCODE
import itertools
import subprocess

datasets = ['BCIIV', 'Stanford'] # 'BCIIV', 'Stanford'
decoders = ['PLS', 'HOPLS', 'BTTR'] # 'PLS', 'HOPLS', 'BTTR'

combinations = list(itertools.product(datasets, decoders))

task_id = int("${SLURM_ARRAY_TASK_ID}")
dataset, decoder = combinations[task_id]

print(f"Running combination: dataset={dataset}, decoder={decoder}")

subprocess.run([
    "python", "regression_o5_ml.py",
    "--dataset", dataset,
    "--decoder", decoder,
])
PYCODE
