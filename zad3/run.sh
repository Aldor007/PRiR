#!/bin/bash
echo 'Running compiled file 15 times for lab1'
echo 'Passed args  size/ncount: '$1

for i in {1..1024..10}
do
        echo "iteration:"$i

        echo -n "$i;" >> results2.csv
        ./gauss_gpu $i /home/shared/prir/video/helicopter.avi  test.avi  >> results2.csv
    done
echo 'Done !'

