#!/usr/bin/bash

#SBATCH --job-name=make_images
#SBATCH --ntasks=1
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --mail-user=katherine.avery@ou.edu
#SBATCH --mail-type=ALL
#SBATCH -p swat_plus
#SBATCH -t 6:00:00
#SBATCH -D /condo/swatwork/keavery/masters_thesis/downloaded_filtered_data
#SBATCH --array=0-80

# cd to directory where job was submitted from
cd $SLURM_SUBMIT_DIR

RADARS=(KLCH KFCX KCRP KTBW KILN KSHV KENX KVWX KFWS KJGX KPBZ KLOT KHTX KDIX KJKL KLZK KSJT KLVX KJAX KHPX KFFC KOKX KLSX KDVN KIND KEAX KDGX KMOB KLWX KDTX KBOX KDMX KBMX KMXX KEVX KGWX KCAE KEOX KLIX KGSP KBRO KDOX KOAX KINX KVAX KBGM KBUF KMKX KDLH KSRX KNQA KRAX KPOE KILX KGRR KOHX KAMX KPAH KTLX KMPX KICT KCLX KRLX KLTX KGRB KIWX KHGX KMVX KEWX KGRK KABR KFSD KTLH KTWX KMHX KAKQ KSGF KCLE KTYX KDYX KMRX)
# get the day information from the array
RADAR=${RADARS[$SLURM_ARRAY_TASK_ID]}

echo $SLURM_ARRAY_TASK_ID

python /home/cchilson/gitRepositories/BirdRoostDetection/BirdRoostDetection\
/PrepareData/CreateImagesFromData.py \
$RADAR \
ml_labels.csv \
/condo/swatwork/keavery/masters_thesis












