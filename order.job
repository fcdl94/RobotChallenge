#!/bin/bash
#SBATCH --job-name=RGBDTC
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

# clean the module environment that we may have inherited from the calling session
ml purge

# load the relevant modules
ml PyTorch
ml torchvision

#run the scripts
t=1

while [ ${t} -le $6 ]
do
python main.py -t $1 -n $2 --name $1_$2_$3-$4_$5-${t} --rgb $3 --depth $4 --order $5 ${@:7}
(( t++ ))
done

# example: ./run.sh SC piggyback 3 -> run SC on piggyback three times in all settings
# example: ./run.sh PE resnet 1 -> run PE on resnet one time in all settings
