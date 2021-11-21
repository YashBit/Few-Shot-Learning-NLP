#!/bin/bash
#SBATCH --job-name=fine-RSST2
#SBATCH --open-mode=append
#SBATCH --output=/home/yb1025/Research/ML_2/robustness/Few-Shot-Learning-NLP/baseline_experimentation/roberta-base/fine-tuning/GLUE/SST-2/results/train/train.out
#SBATCH --error=/home/yb1025/Research/ML_2/robustness/Few-Shot-Learning-NLP/baseline_experimentation/roberta-base/fine-tuning/GLUE/SST-2/results/train/train.err
#SBATCH --export=ALL
#SBATCH --time=50:0:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -c 2

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity \
    exec --nv \
    --overlay /home/yb1025/pyenv/overlay-7.5GB-300K.ext3:ro \
    --overlay /scratch/work/public/singularity/openmpi4.1.1-ubuntu20.04.sqf:ro \
    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif  \
    /bin/bash -c "
source /ext3/env.sh
source /ext3/openmpi.sh
source activate allegro_gym
python run_hyperparameter_search.py 
"
