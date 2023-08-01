#!/bin/bash
#SBATCH --array=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                        # memory per node
#SBATCH --time=00-12:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_score_xray
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate
python $HOME/scratch/RIM_xray_spectra/scripts/train_score.py\
    --data_path=$HOME/scratch/RIM_xray_spectra/data/RIM_clusters/\
    --checkpoints_directory=$HOME/scratch/RIM_xray_spectra/models/score_xray_clusters/\
    --hyperparameters=$HOME/scratch/RIM_xray_spectra/scripts/score_xray.json\
    --epochs=10000\
    --ema_decay=0.999
    --batch_size=32\
    --learning_rate=1e-3\
    --checkpoints=5\
    --models_to_keep=3\
    --max_time=10
