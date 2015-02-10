#!/bin/bash
echo 'Running compiled file 15 times for lab1'
echo 'Passed args  size/ncount: '$1

echo '----------------------'
echo "size: $1">>results.csv 
for i in {1..15..1}
do
       
        echo "iteration:"$i
        ./pi_omp $i $1 >> results.csv
    done
echo 'Done !'

