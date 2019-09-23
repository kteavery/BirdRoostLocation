#!/usr/bin/bash
#
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH -o eval_%J.out
#SBATCH -e eval_%J.err
#SBATCH --mail-user=katherine.avery@ou.edu
#SBATCH --mail-type=ALL
#SBATCH -p swat_plus
#SBATCH -t 47:00:00
#SBATCH --array=0-3
#SBATCH --mem 120G

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
BirdRoostLocation/BuildModels/ShallowCNN/eval.py \
--radar_product=1 \
--log_path=model/Velocity/ \
--coord_conv="" \
--problem="localization"