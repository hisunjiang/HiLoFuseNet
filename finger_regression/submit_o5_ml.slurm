#!/bin/bash -l
#SBATCH --cluster wice
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
#SBATCH --array=0-155							# <- change here

source /xxxx/miniconda/etc/profile.d/conda.sh 	# <- change here
conda activate decode

export PYTHONHASHSEED=42

python - <<PYCODE
import itertools
import subprocess
import time
import numpy as np

datasets_meta = {
    "BCIIV": ['sub1', 'sub2', 'sub3'], 
    "Stanford": ['bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt']
}
decoders = ['RF'] # 'PLS',  'NPLS','HOPLS', 'eBTTR', 'RF'
win_sizes = [round(f, 1) for f in np.arange(0.4, 3, 0.2)] #

combinations = []
for ds_name, subjects in datasets_meta.items():
    for sub in subjects:
        for dec in decoders:
            for ws in win_sizes:
                combinations.append((ds_name, sub, dec, ws))

# get current task ID
task_id = int("${SLURM_ARRAY_TASK_ID}")
if task_id >= len(combinations):
    print(f"Task ID {task_id} out of range (max {len(combinations)-1})")
    exit(1)

dataset, subject, decoder, win_size = combinations[task_id]

print(f"--- Task {task_id} ---")
print(f"Dataset: {dataset} | Subject: {subject} | Decoder: {decoder} | Win: {win_size}")

# run jobs
start_time  = time.time()
subprocess.run([
    "python", "regression_o5_ml.py",
    "--dataset", dataset,
    "--subject", subject,
    "--decoder", decoder,
    "--win_size", str(win_size)
])

end_time = time.time()
duration = (end_time - start_time) / 60
print(f"==================================================")
print(f"Task {task_id} completed in: {duration:.2f} minutes")
print(f"==================================================")

PYCODE
