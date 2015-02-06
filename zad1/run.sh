#!/bin/bash
echo 'Running compiled file 15 times for lab1'
echo 'Passed args  size/ncount: '$1

echo '----------------------'
echo "size: $1">>results2.csv 
for i in {1..15..1}
do
        echo "----------------" >> results.txt 2>&1
        echo "iteration:"$i
        echo "watkow; " $i >> results2.csv 2>&1

        echo 'sum; mnozenie' >> results2.csv
        ./macierz_omp $i $1 >> results2.csv
        echo "./macierz_omp2 $i $1 >> results2.csv 2>&1"
    done
echo 'Done !'

