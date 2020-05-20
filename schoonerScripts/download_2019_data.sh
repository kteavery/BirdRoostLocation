#!/usr/bin/bash

#SBATCH --job-name=download_data
#SBATCH --ntasks=1
#SBATCH -o download_data_%J.out
#SBATCHi -e download_data_%J.err
#SBATCH --mail-user=katherine.avery@ou.edu
#SBATCH --mail-type=ALL
#SBATCH -p swat_plus
#SBATCH -t 6:00:00
#SBATCH --array=0-80

# cd to directory where job was submitted from
cd $SLURM_SUBMIT_DIR

python /condo/swatwork/keavery/masters_thesis/gitRepos/BirdRoostLocation/BirdRoostLocation/PrepareData/DownloadData.py
