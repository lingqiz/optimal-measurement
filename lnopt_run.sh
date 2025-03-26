#!/bin/bash
#SBATCH --job-name=lnopt
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --constraint=a100
#SBATCH -c 48
#SBATCH -t 7-00:00:00

cd ~/denoiser_recon/
module load python/3.10 cuda cudnn nccl
source recon/bin/activate

# set parameter values
# (add --avg $AVG to use average of multiple samples)

MTD=Denoiser
LST=MSE
DCR=0.90
BSZ=64
NEP=16
LNR=0.0005
AVG=true

python3 train_olm.py --recon_method $MTD --loss_type $LST --decay_rate $DCR \
                     --n_sample $NSP --data_path $DST --model_path $MDP \
                     --lr $LNR --batch_size $BSZ --n_epoch $NEP --avg $AVG
