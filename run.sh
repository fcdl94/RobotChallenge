#!/bin/bash
x=1

while [ ${x} -le $2 ]
do
python main.py --t $1 --name $1_$3-$4_${x} --set rod --net resnet --rgb $3 --depth $4
(( x++ ))
done

# example: ./run.sh SC 3 1 0 -> run SC three times with only RGB
# example: ./run.sh PE 1 0 1 -> run PE one time with only Dpt