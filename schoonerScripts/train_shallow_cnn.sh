#!/usr/bin/bash
#
#SBATCH --job-name=train_cnn
#SBATCH --ntasks=1
#SBATCH -o train_unet_%J.out
#SBATCH -e train_unet_%J.err
#SBATCH --mail-user=katherine.avery@ou.edu
#SBATCH --mail-type=ALL
#SBATCH -p idea
#SBATCH -t 47:00:00
#SBATCH --array=0-3
#SBATCH --mem 32G

# cd to directory where job was submitted from
cd $SLURM_SUBMIT_DIR

RADARS_PRODUCTS=(0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3)
TIMES=(0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3)
DUAL_POLS=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1)
# get the day information from the array
RADARS_PRODUCT=${RADARS_PRODUCTS[$SLURM_ARRAY_TASK_ID]}
TIME=${TIMES[$SLURM_ARRAY_TASK_ID]}
DUAL_POL=${DUAL_POLS[$SLURM_ARRAY_TASK_ID]}

echo $SLURM_ARRAY_TASK_ID

python /condo/swatwork/keavery/masters_thesis/gitRepos/BirdRoostLocation/\
BirdRoostLocation/BuildModels/ShallowCNN/train.py \
--radar_product=0 \
--log_path=model/Reflectivity/ \
--eval_increment=5 \
--num_iterations=4000 \
--checkpoint_frequency=100 \
--learning_rate=.0001 \
--model=0 \
--model_type="unet" \
--high_memory_mode="" \
--num_temporal_data=1 \
--dual_pol="true" \
--coord_conv="" \
--problem="localization"
