#!/usr/bin/env bash

NAMES=(
    'ape' 'benchviseblue' 'bowl' 'can' 'cat' 'cup' 'driller' 'duck' 'glue' 'holepuncher' 'iron' 'lamp' 'phone' 'cam' 'eggbox'
)

for ad in ${NAMES[@]};do
    if [ ! -d "$ad" ]; then
        wget "http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/${ad}.zip"
        unzip "${ad}.zip"
        cd "$ad"
        rm distance.txt mesh.ply object.xyz OLDmesh.ply transform.dat
        mv data/* ./
        rm -r data
        cd ..
    fi
done
exit 0