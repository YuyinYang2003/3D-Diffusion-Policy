#!/bin/sh

#SBATCH -N 1 -n 6 --gres=gpu:2 -p smartbot -A smartbot
# srun -N 1 -n 12 --gres=gpu:4 -p smartbot bash scripts/srun_ddp_train.sh
# srun -N 1 -n 24 --gres=gpu:4 -p smartbot bash ./srun_ddp_train.sh
# sbatch -N 1 -n 24 --gres=gpu:4 -p smartbot -A smartbot /ailab/user/yangyuyin/3D-Diffusion-Policy/3D-Diffusion-Policy/srun_ddp_train.sh

module load anaconda/2024.02
module load cuda/12.1
module load gcc/9.3.0
module load tensorboard/2.14
source activate aloha

export PYTHONUNBUFFERED=1

# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh
DEBUG=False
ngpus=4

# if [ $DEBUG = True ]; then
#     wandb_mode=offline
#     # wandb_mode=online
#     echo -e "Debug mode"
# else
#     wandb_mode=online
#     echo -e "Train mode"
# fi

cd /ailab/user/yangyuyin/3D-Diffusion-Policy/3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
torchrun --nnodes 1 --nproc_per_node $ngpus --master_port 12354 ddp_train.py \
    task='push_box' \
    name='train_dualdp3_1gpus' \
    addition_info="ddp_1013_mlp_continuous" \
    wandb_name="dp3k2_push_box" \
    horizon=8 \
    n_obs_steps=6 \
    n_action_steps=3 \
    policy.diffusion_step_embed_dim=256 \
    policy.encoder_output_dim=128 \
    policy.use_pc_color=false \
    policy.use_lang=true \
    task.shape_meta.obs.point_cloud.shape=[1024,3] \
    task.dataset.pcd_fps=1024 \
    dataloader.batch_size=32 \
    val_dataloader.batch_size=16 \

                                