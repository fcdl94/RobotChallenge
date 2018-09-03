#!/bin/bash
x=1

while [ ${x} -le $5 ]
do
python main.py --t $1 --name $1_$3-$4_${x} --set rod -n $2 --rgb $3 --depth $4
(( x++ ))
done

# example: ./run.sh SC piggyback 1 0 3 -> run SC on piggyback three times with only RGB
# example: ./run.sh PE resnet 0 1 1 -> run PE on resnet one time with only Dpt