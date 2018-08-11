#!/usr/bin/env bash

folder=$1
folders=$(find . -name "$1*")
i=0

mkdir $folder
for f in $folders
do
    convert $f/*.ppm $folder/image${i}_%d.png
    rm -r $f
    let i=i+1
done
