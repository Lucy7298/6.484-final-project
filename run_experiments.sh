#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH -a 3,4,6,17,30
#SBATCH --time=12:00:00
#SBATCH --output=/nobackup/users/yunxingl/models/sac_exp/slurm_logs/debug-%a.log

e_cost=("" "--use_electricity_cost --use_torque_cost")
d_cost=("--torque_cost -0.1" "--use_cost_diff") #use default parameters if you don't use cost diff
s_cost=("" "--use_strain_cost")
e_surprise=("" "--use_electricity_surprise")
s_surprise=("" "--use_strain_surprise")

ct=0

for e_flag in "${e_cost[@]}";
do 
    for d_flag in "${d_cost[@]}";
    do 
        for s_flag in "${s_cost[@]}";
        do 
            for e_s_flag in "${e_surprise[@]}"; 
            do 
                for s_s_flag in "${s_surprise[@]}"; 
                do
                if [[ "$ct" = "$SLURM_ARRAY_TASK_ID" ]]
                then
                echo "starting experiment"
                echo $SLURM_ARRAY_TASK_ID $ct
                echo $e_flag 
                echo $d_flag
                echo $s_flag
                echo $e_s_flag
                echo $s_s_flag
                output_dir=/nobackup/users/yunxingl/models/sac_exp/$SLURM_ARRAY_TASK_ID
                python3 train_models.py \
                        --output_dir $output_dir \
                        --use_progress_reward \
                        --train_type SAC \
                        $e_flag \
                        --use_limits_cost \
                        $d_flag \
                        $s_flag \
                        $e_s_flag \
                        $s_s_flag 
                exit
                fi 
                ct=$((ct + 1))
                done 
            done 
        done
    done
done