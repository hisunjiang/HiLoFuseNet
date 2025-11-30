#!/bin/bash -l
#SBATCH --cluster genius
#SBATCH --partition batch
#SBATCH --ntasks="1"
#SBATCH --mem="64G"
#SBATCH --time="24:00:00"
#SBATCH --account="xxxx" 						# <- change here
#SBATCH --job-name="finger_o5_ml"
#SBATCH --chdir="/project/" 					# <- change here
#SBATCH --error="/project/logs/%x.e%A_%a" 		# <- change here
#SBATCH --output="/project/logs/%x.o%A_%a" 		# <- change here
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="qiang.sun@kuleuven.be" 	# <- change here
#SBATCH --array=0-5

source /xxxx/miniconda/etc/profile.d/conda.sh 	# <- change here
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
