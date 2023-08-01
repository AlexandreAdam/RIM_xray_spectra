#!/bin/bash
#SBATCH --array=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                        # memory per node
#SBATCH --time=00-23:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_xray_prior
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate
python $HOME/scratch/RIM_xray_spectra/scripts/train_rim.py \
    --data_path=$HOME/scratch/RIM_xray_spectra/data/\
    --prior_model=$HOME/scratch/RIM_xray_spectra/models/score_xray_clusters_20230731_2\
    --prior_temperature=0.1\
    --logdir=$HOME/scratch/RIM_xray_spectra/models/\
    --logname_prefix=rim_xray_clusters_heavy_with_prior\
    --snr_max=100\
    --snr_min=5\
    --noise_distribution=log_uniform\
    --hyperparameters=$HOME/scratch/RIM_xray_spectra/scripts/rim_xray_heavy_hparams.json\
    --epochs=20000\
    --clip=1\
    --ema_decay=0.99\
    --batch_size=32\
    --learning_rate=1e-4\
    --checkpoints=1\
    --models_to_keep=3\
    --max_time=22\
    --lr_schedule_step_size=1000\
    --lr_schedule_gamma=0.7
