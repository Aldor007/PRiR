#!/bin/bash
hashes=( fbade9e36a3f36d3d676c1b808451dd7 25ed1bcb423b0b7200f485fc5ff71c8e f3abb86bd34cf4d52698f14c0da1dc60 02c425157ecd32f259548b33402ff6d3 95ebc3c7b3b9f1d2c40fec14415d3cb8 )
i=0
for h in ${hashes[*]}; do
    echo -n "$i;">>results.csv
    ./md5_gpu $h >> results.csv
    i=$(($i+1))
done
