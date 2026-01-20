#!/bin/bash
#SBATCH --job-name=dp_${1}_cr${2}_dp${3}_b${4}
#SBATCH --output=logs/dp_%j.out
#SBATCH --error=logs/dp_%j.err
#SBATCH --constraint=h100

# --- Resources: 1 H100 GPU, ~2500 minutes (~41h40) ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00

#SBATCH --account zwf@h100

set -euo pipefail

DATASET="$1"
CR="$2"
DP="$3"
BS="$4"

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "dataset         = ${DATASET}"
echo "compression_rate= ${CR}"
echo "dp_scale        = ${DP}"
echo "batch_size      = ${BS}"


module purge # nettoyer les modules herites par defaut
module load arch/h100 # selectionner les modules compiles pour les H100
module load pytorch-gpu/py3/2.4.0
set -x

# Optional: conda/venv activation
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

mkdir -p logs

srun python launch_experiments_jean_zay.py \
    --dataset "${DATASET}" \
    --epochs 40 \
    --sigma 1.0 \
    --max-per-sample-grad_norm 1.0 \
    --compression-rate "${CR}" \
    --dp-scale "${DP}" \
    --batch-size "${BS}" \
    --device cuda:0
